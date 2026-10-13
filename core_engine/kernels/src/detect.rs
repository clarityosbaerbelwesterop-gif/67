//! Runtime detection of the GEMM variants this process can execute.

use std::sync::OnceLock;

use crate::GemmVariant;

static AVAILABLE: OnceLock<Vec<GemmVariant>> = OnceLock::new();

/// Variants usable on this machine in ascending order of preference
/// (`Scalar` first). Detected once per process.
pub(crate) fn available() -> &'static [GemmVariant] {
    AVAILABLE.get_or_init(detect)
}

fn detect() -> Vec<GemmVariant> {
    #[allow(unused_mut)]
    let mut v = vec![GemmVariant::Scalar];
    #[cfg(target_arch = "x86_64")]
    {
        if crate::x86::avx2::supported() {
            v.push(GemmVariant::Avx2Fma);
        }
        if crate::x86::avx512::supported() {
            v.push(GemmVariant::Avx512F32);
        }
        if amx_opt_in() && crate::x86::amx::supported() {
            v.push(GemmVariant::AmxBf16);
        }
    }
    v
}

/// AMX is opt-in (`FORGE_AMX=1`): on a Sapphire-Rapids-class box the AmxBf16
/// kernel passed its self-test but failed the reference tests (wrong products up
/// to sign flips), so the default is the verified AVX-512 path.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn amx_opt_in() -> bool {
    amx_opt_in_from(std::env::var("FORGE_AMX").ok().as_deref())
}

fn amx_opt_in_from(v: Option<&str>) -> bool {
    matches!(v, Some("1") | Some("true") | Some("on"))
}

/// Preference rank: Scalar < Avx2Fma < Avx512F32 < AmxBf16.
fn rank(v: GemmVariant) -> usize {
    match v {
        GemmVariant::Scalar => 0,
        GemmVariant::Avx2Fma => 1,
        GemmVariant::Avx512F32 => 2,
        GemmVariant::AmxBf16 => 3,
    }
}

/// The variant that actually runs when `v` is requested: `v` itself when it is
/// available, otherwise the best available variant ranked below it.
pub(crate) fn resolve_in(avail: &[GemmVariant], v: GemmVariant) -> GemmVariant {
    avail
        .iter()
        .copied()
        .filter(|&x| rank(x) <= rank(v))
        .max_by_key(|&x| rank(x))
        .unwrap_or(GemmVariant::Scalar)
}

pub(crate) fn resolve(v: GemmVariant) -> GemmVariant {
    resolve_in(available(), v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use GemmVariant::*;

    #[test]
    fn resolve_falls_back_downwards() {
        let all = [Scalar, Avx2Fma, Avx512F32, AmxBf16];
        for v in all {
            assert_eq!(resolve_in(&all, v), v);
        }
        let no_amx = [Scalar, Avx2Fma, Avx512F32];
        assert_eq!(resolve_in(&no_amx, AmxBf16), Avx512F32);
        let avx2_only = [Scalar, Avx2Fma];
        assert_eq!(resolve_in(&avx2_only, AmxBf16), Avx2Fma);
        assert_eq!(resolve_in(&avx2_only, Avx512F32), Avx2Fma);
        assert_eq!(resolve_in(&avx2_only, Scalar), Scalar);
        assert_eq!(resolve_in(&[Scalar], AmxBf16), Scalar);
        // A gap in the list (AMX without AVX-512 is impossible, but resolve must
        // still never pick a variant above the request).
        assert_eq!(resolve_in(&[Scalar, AmxBf16], Avx512F32), Scalar);
    }

    #[test]
    fn amx_is_opt_in() {
        assert!(!amx_opt_in_from(None));
        assert!(!amx_opt_in_from(Some("0")));
        assert!(!amx_opt_in_from(Some("")));
        assert!(amx_opt_in_from(Some("1")));
        if std::env::var("FORGE_AMX").is_err() {
            assert!(!available().contains(&AmxBf16));
        }
    }

    #[test]
    fn available_is_sorted_and_starts_with_scalar() {
        let a = available();
        assert_eq!(a[0], Scalar);
        assert!(a.windows(2).all(|w| rank(w[0]) < rank(w[1])));
    }
}
