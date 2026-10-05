//! forge-hbvm — High-Bandwidth Virtual Memory.
//!
//! One contiguous, 64-byte-aligned region of f32 words plays the role of the
//! device memory. Tensors are addressed by handles (`Buf`), never by pointers,
//! so the compiler can plan placement and the runtime can report exact usage.
//! Allocation is first-fit over an address-ordered free list with immediate
//! coalescing; every offset is aligned to 16 words (one cache line).
//!
//! Contract: docs/DESIGN.md §4.

use std::fmt;

/// Words per alignment unit (64 bytes).
pub const ALIGN: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Buf(pub u32);

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HbvmStats {
    /// Capacity in f32 words.
    pub capacity: usize,
    /// Words currently allocated (aligned sizes).
    pub used: usize,
    /// High-water mark of `used`.
    pub peak: usize,
    /// Live buffers.
    pub live: usize,
    pub allocs: u64,
    pub frees: u64,
    /// 1 - largest_free_block / total_free (0 = one contiguous hole).
    pub fragmentation: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HbvmError {
    OutOfMemory {
        requested: usize,
        largest_free: usize,
        free: usize,
    },
    InvalidHandle(Buf),
}

impl fmt::Display for HbvmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HbvmError::OutOfMemory { requested, largest_free, free } => write!(
                f,
                "HBVM out of memory: requested {requested} words, largest hole {largest_free}, free {free}"
            ),
            HbvmError::InvalidHandle(b) => write!(f, "invalid HBVM handle {}", b.0),
        }
    }
}

impl std::error::Error for HbvmError {}

#[derive(Clone, Copy, Debug)]
struct Slot {
    offset: usize,
    len: usize,
    reserved: usize,
}

pub struct Hbvm {
    /// Backing store; `base` is the first 64-byte-aligned index.
    store: Vec<f32>,
    base: usize,
    capacity: usize,
    /// Address-ordered holes: (offset, len), both multiples of ALIGN.
    free: Vec<(usize, usize)>,
    slots: Vec<Option<Slot>>,
    recycled: Vec<u32>,
    stats: HbvmStats,
}

fn round_up(n: usize) -> usize {
    n.max(1).div_ceil(ALIGN) * ALIGN
}

impl Hbvm {
    /// Reserve `capacity` f32 words (rounded up to a cache line).
    pub fn new(capacity: usize) -> Hbvm {
        let capacity = round_up(capacity);
        let store = vec![0.0f32; capacity + ALIGN];
        let misalign = (store.as_ptr() as usize / 4) % ALIGN;
        let base = (ALIGN - misalign) % ALIGN;
        Hbvm {
            store,
            base,
            capacity,
            free: vec![(0, capacity)],
            slots: Vec::new(),
            recycled: Vec::new(),
            stats: HbvmStats {
                capacity,
                ..HbvmStats::default()
            },
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Allocate `len` words, zero-initialised.
    pub fn alloc(&mut self, len: usize) -> Result<Buf, HbvmError> {
        let reserved = round_up(len);
        let Some(i) = self.free.iter().position(|&(_, l)| l >= reserved) else {
            return Err(HbvmError::OutOfMemory {
                requested: reserved,
                largest_free: self.largest_free(),
                free: self.capacity - self.stats.used,
            });
        };
        let (offset, hole) = self.free[i];
        if hole == reserved {
            self.free.remove(i);
        } else {
            self.free[i] = (offset + reserved, hole - reserved);
        }
        let start = self.base + offset;
        self.store[start..start + reserved].fill(0.0);
        let slot = Slot {
            offset,
            len,
            reserved,
        };
        let id = match self.recycled.pop() {
            Some(id) => {
                self.slots[id as usize] = Some(slot);
                id
            }
            None => {
                self.slots.push(Some(slot));
                (self.slots.len() - 1) as u32
            }
        };
        self.stats.used += reserved;
        self.stats.peak = self.stats.peak.max(self.stats.used);
        self.stats.live += 1;
        self.stats.allocs += 1;
        Ok(Buf(id))
    }

    pub fn free(&mut self, buf: Buf) -> Result<(), HbvmError> {
        let slot = self
            .slots
            .get_mut(buf.0 as usize)
            .and_then(Option::take)
            .ok_or(HbvmError::InvalidHandle(buf))?;
        self.recycled.push(buf.0);
        // Insert the hole in address order and coalesce with both neighbours.
        let pos = self.free.partition_point(|&(o, _)| o < slot.offset);
        self.free.insert(pos, (slot.offset, slot.reserved));
        if pos + 1 < self.free.len() && self.free[pos].0 + self.free[pos].1 == self.free[pos + 1].0
        {
            self.free[pos].1 += self.free[pos + 1].1;
            self.free.remove(pos + 1);
        }
        if pos > 0 && self.free[pos - 1].0 + self.free[pos - 1].1 == self.free[pos].0 {
            self.free[pos - 1].1 += self.free[pos].1;
            self.free.remove(pos);
        }
        self.stats.used -= slot.reserved;
        self.stats.live -= 1;
        self.stats.frees += 1;
        Ok(())
    }

    fn slot(&self, buf: Buf) -> Slot {
        self.slots
            .get(buf.0 as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("invalid HBVM handle {}", buf.0))
    }

    pub fn len(&self, buf: Buf) -> usize {
        self.slot(buf).len
    }

    pub fn is_live(&self, buf: Buf) -> bool {
        matches!(self.slots.get(buf.0 as usize), Some(Some(_)))
    }

    pub fn slice(&self, buf: Buf) -> &[f32] {
        let s = self.slot(buf);
        let start = self.base + s.offset;
        &self.store[start..start + s.len]
    }

    pub fn slice_mut(&mut self, buf: Buf) -> &mut [f32] {
        let s = self.slot(buf);
        let start = self.base + s.offset;
        &mut self.store[start..start + s.len]
    }

    /// Raw base pointer and length of a buffer. The pointer stays valid until the
    /// buffer is freed; callers must uphold Rust aliasing rules themselves.
    pub fn raw_parts(&mut self, buf: Buf) -> (*mut f32, usize) {
        let s = self.slot(buf);
        let base = self.base;
        (
            unsafe { self.store.as_mut_ptr().add(base + s.offset) },
            s.len,
        )
    }

    /// Simultaneous access to several distinct buffers: `outs` mutable, `ins` shared.
    ///
    /// Panics if any handle repeats (the interpreter verifies aliasing first).
    pub fn views<const O: usize, const I: usize>(
        &mut self,
        outs: [Buf; O],
        ins: [Buf; I],
    ) -> ([&mut [f32]; O], [&[f32]; I]) {
        let mut all: Vec<Buf> = outs.iter().chain(ins.iter()).copied().collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), O + I, "HBVM views: aliased operands");
        let base = self.store.as_mut_ptr();
        let ptr = |s: Slot| unsafe { base.add(self.base + s.offset) };
        // SAFETY: every slot is a disjoint, in-bounds range of `store` (the
        // allocator never hands out overlapping ranges) and the handles were
        // checked to be pairwise distinct above, so the views do not alias.
        let o = outs.map(|b| {
            let s = self.slot(b);
            unsafe { std::slice::from_raw_parts_mut(ptr(s), s.len) }
        });
        let i = ins.map(|b| {
            let s = self.slot(b);
            unsafe { std::slice::from_raw_parts(ptr(s) as *const f32, s.len) }
        });
        (o, i)
    }

    fn largest_free(&self) -> usize {
        self.free.iter().map(|&(_, l)| l).max().unwrap_or(0)
    }

    pub fn stats(&self) -> HbvmStats {
        let total_free = self.capacity - self.stats.used;
        let fragmentation = if total_free == 0 {
            0.0
        } else {
            1.0 - self.largest_free() as f64 / total_free as f64
        };
        HbvmStats {
            fragmentation,
            ..self.stats
        }
    }

    /// Byte address of a buffer relative to the region (for placement reports).
    pub fn offset_bytes(&self, buf: Buf) -> usize {
        self.slot(buf).offset * 4
    }

    pub fn is_aligned(&self, buf: Buf) -> bool {
        (self.slice(buf).as_ptr() as usize).is_multiple_of(64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_are_aligned_zeroed_and_disjoint() {
        let mut m = Hbvm::new(10_000);
        let a = m.alloc(5).unwrap();
        let b = m.alloc(33).unwrap();
        assert!(m.is_aligned(a) && m.is_aligned(b));
        m.slice_mut(a).fill(1.0);
        assert!(m.slice(b).iter().all(|&x| x == 0.0));
        assert_eq!(m.len(b), 33);
        assert_eq!(m.stats().used, 16 + 48);
        let ([x], [y]) = m.views([a], [b]);
        x[0] = y[0] + 2.0;
        assert_eq!(m.slice(a)[0], 2.0);
    }

    #[test]
    fn free_coalesces_and_reuses() {
        let mut m = Hbvm::new(64);
        let bufs: Vec<Buf> = (0..4).map(|_| m.alloc(16).unwrap()).collect();
        assert!(m.alloc(1).is_err());
        m.free(bufs[1]).unwrap();
        m.free(bufs[2]).unwrap();
        let s = m.stats();
        assert_eq!(s.fragmentation, 0.0, "adjacent holes must coalesce");
        let big = m.alloc(32).unwrap();
        assert_eq!(m.offset_bytes(big), 16 * 4);
        m.free(bufs[0]).unwrap();
        m.free(bufs[3]).unwrap();
        let s = m.stats();
        assert!(s.fragmentation > 0.0 && s.fragmentation < 1.0);
        m.free(big).unwrap();
        assert_eq!(m.stats().used, 0);
        assert_eq!(m.stats().fragmentation, 0.0);
        assert_eq!(m.stats().peak, 64);
    }

    #[test]
    fn rejects_double_free_and_reports_oom() {
        let mut m = Hbvm::new(32);
        let a = m.alloc(10).unwrap();
        m.free(a).unwrap();
        assert_eq!(m.free(a), Err(HbvmError::InvalidHandle(a)));
        match m.alloc(100) {
            Err(HbvmError::OutOfMemory { requested, .. }) => assert_eq!(requested, 112),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "aliased")]
    fn views_refuse_aliasing() {
        let mut m = Hbvm::new(64);
        let a = m.alloc(4).unwrap();
        let _ = m.views([a], [a]);
    }

    #[test]
    fn random_workload_keeps_invariants() {
        let mut m = Hbvm::new(1 << 16);
        let mut live: Vec<(Buf, f32)> = Vec::new();
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for i in 0..5000 {
            if live.is_empty() || next() % 3 != 0 {
                if let Ok(b) = m.alloc((next() % 900) as usize + 1) {
                    let tag = i as f32;
                    m.slice_mut(b).fill(tag);
                    live.push((b, tag));
                }
            } else {
                let (b, tag) = live.swap_remove((next() as usize) % live.len());
                assert!(
                    m.slice(b).iter().all(|&x| x == tag),
                    "buffer contents clobbered"
                );
                m.free(b).unwrap();
            }
        }
        for (b, tag) in live.drain(..) {
            assert!(m.slice(b).iter().all(|&x| x == tag));
            m.free(b).unwrap();
        }
        assert_eq!(m.stats().used, 0);
    }
}
