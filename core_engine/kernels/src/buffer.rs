//! Cache-line aligned scratch buffers and the per-thread buffer cache used for
//! packed GEMM operands.
//!
//! Every worker thread keeps one grow-only buffer per slot (packed A blocks,
//! packed B slabs), so steady-state GEMM calls do not allocate. Rayon's work
//! stealing can re-enter a thread while it still holds a buffer (a thread
//! blocked in a join may run an unrelated task that also needs the slot); in
//! that case the borrow fails and a temporary buffer is allocated instead.

use std::alloc::{self, Layout};
use std::cell::RefCell;
use std::ptr::NonNull;
use std::thread::LocalKey;

/// Alignment of every scratch buffer: one cache line, and the natural
/// alignment of a 512-bit vector or an AMX tile row.
pub(crate) const ALIGN: usize = 64;

/// Growth granularity, so that slowly growing requests do not reallocate on
/// every call.
const GRANULE: usize = 64 * 1024;

/// A grow-only, 64-byte aligned, uninitialised byte buffer.
///
/// The contents are never read before the packing routines write them, so the
/// buffer is handed out as a raw pointer and never as a reference to possibly
/// uninitialised memory.
pub(crate) struct AlignedBuf {
    ptr: NonNull<u8>,
    cap: usize,
}

impl AlignedBuf {
    pub(crate) const fn new() -> Self {
        AlignedBuf {
            ptr: NonNull::dangling(),
            cap: 0,
        }
    }

    /// Returns a pointer to at least `bytes` bytes, 64-byte aligned. Previous
    /// contents are not preserved when the buffer grows.
    pub(crate) fn reserve(&mut self, bytes: usize) -> *mut u8 {
        if bytes > self.cap {
            let cap = bytes
                .checked_next_multiple_of(GRANULE)
                .expect("scratch buffer size overflows usize");
            let layout = Layout::from_size_align(cap, ALIGN).expect("scratch buffer layout");
            // Free the old block first so peak memory does not double.
            self.release();
            // SAFETY: `cap` is non-zero (bytes > self.cap >= 0) and the layout is valid.
            let raw = unsafe { alloc::alloc(layout) };
            self.ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
            self.cap = cap;
        }
        self.ptr.as_ptr()
    }

    fn release(&mut self) {
        if self.cap != 0 {
            let layout = Layout::from_size_align(self.cap, ALIGN).expect("scratch buffer layout");
            // SAFETY: `ptr` was allocated with exactly this layout.
            unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
            self.ptr = NonNull::dangling();
            self.cap = 0;
        }
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.cap
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        self.release();
    }
}

thread_local! {
    /// Packed A blocks (one per thread, used inside a single sequential task).
    pub(crate) static PACK_A: RefCell<AlignedBuf> = const { RefCell::new(AlignedBuf::new()) };
    /// Packed B slabs (owned by the thread that drives a GEMM).
    pub(crate) static PACK_B: RefCell<AlignedBuf> = const { RefCell::new(AlignedBuf::new()) };
}

/// Runs `f` with a 64-byte aligned scratch area of at least `bytes` bytes taken
/// from the calling thread's cache slot `key`, or from a temporary allocation
/// when the slot is already in use on this thread (re-entrancy through work
/// stealing) or thread-local storage is being torn down.
pub(crate) fn with_buf<R>(
    key: &'static LocalKey<RefCell<AlignedBuf>>,
    bytes: usize,
    f: impl FnOnce(*mut u8) -> R,
) -> R {
    let mut f = Some(f);
    let cached = key.try_with(|cell| match cell.try_borrow_mut() {
        Ok(mut buf) => {
            let ptr = buf.reserve(bytes);
            let f = f.take().expect("closure runs once");
            Some(f(ptr))
        }
        Err(_) => None,
    });
    match cached {
        Ok(Some(r)) => r,
        _ => {
            let mut tmp = AlignedBuf::new();
            let ptr = tmp.reserve(bytes);
            let f = f.take().expect("closure runs once");
            f(ptr)
        }
    }
}

/// A raw pointer that may be shared between the tasks of one GEMM call. The
/// driver guarantees that concurrent tasks write disjoint regions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SendPtr<T>(pub(crate) *mut T);

// SAFETY: see the type documentation; the pointer is only dereferenced under
// the driver's disjointness discipline.
unsafe impl<T> Send for SendPtr<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// The wrapped pointer. Closures must go through this method (not the
    /// field) so that they capture the `Send`/`Sync` wrapper as a whole.
    #[inline(always)]
    pub(crate) fn get(self) -> *mut T {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_is_aligned_and_grows() {
        let mut b = AlignedBuf::new();
        assert_eq!(b.capacity(), 0);
        let p = b.reserve(10);
        assert_eq!(p as usize % ALIGN, 0);
        assert!(b.capacity() >= 10);
        // SAFETY: the buffer holds at least 10 bytes.
        unsafe { p.write_bytes(0xAB, 10) };
        let cap = b.capacity();
        let p2 = b.reserve(cap);
        assert_eq!(p, p2, "no reallocation within capacity");
        let p3 = b.reserve(cap + 1);
        assert_eq!(p3 as usize % ALIGN, 0);
        assert!(b.capacity() > cap);
    }

    #[test]
    fn nested_use_of_a_slot_falls_back_to_a_temporary() {
        with_buf(&PACK_A, 1000, |outer| {
            with_buf(&PACK_A, 1000, |inner| {
                assert_ne!(outer, inner, "re-entrant use must not alias");
                assert_eq!(inner as usize % ALIGN, 0);
            });
        });
        // The slot is usable again afterwards.
        with_buf(&PACK_A, 10, |p| assert_eq!(p as usize % ALIGN, 0));
    }
}
