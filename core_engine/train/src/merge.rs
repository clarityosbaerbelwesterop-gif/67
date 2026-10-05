//! Model fusion on a shared base: task arithmetic, linear and TIES
//! (Yadav et al., 2023: trim → elect sign → disjoint mean). Deterministic, CPU.

use crate::ckpt::Tensor;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// θ = θ0 + λ·mean(τ_i)
    Linear,
    /// θ = θ0 + λ·TIES(τ_i, density)
    Ties { density: f32 },
}

/// Keep the top `density` fraction of |τ| (ties at the threshold are kept).
fn trim(t: &mut [f32], density: f32) {
    if density >= 1.0 || t.is_empty() {
        return;
    }
    let keep = ((t.len() as f32 * density).ceil() as usize).clamp(1, t.len());
    let mut mags: Vec<f32> = t.iter().map(|x| x.abs()).collect();
    let idx = t.len() - keep;
    let (_, thr, _) = mags.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    let thr = *thr;
    t.iter_mut()
        .filter(|x| x.abs() < thr)
        .for_each(|x| *x = 0.0);
}

pub fn merge_tensor(base: &[f32], children: &[&[f32]], method: Method, lambda: f32) -> Vec<f32> {
    let n = base.len();
    assert!(
        children.iter().all(|c| c.len() == n),
        "children must share the base shape"
    );
    let mut taus: Vec<Vec<f32>> = children
        .iter()
        .map(|c| c.iter().zip(base).map(|(x, b)| x - b).collect())
        .collect();
    let k = taus.len() as f32;
    let mut out = base.to_vec();
    match method {
        Method::Linear => {
            for i in 0..n {
                out[i] += lambda * taus.iter().map(|t| t[i]).sum::<f32>() / k;
            }
        }
        Method::Ties { density } => {
            for t in &mut taus {
                trim(t, density);
            }
            for i in 0..n {
                let elected = taus.iter().map(|t| t[i]).sum::<f32>().signum();
                if elected == 0.0 {
                    continue;
                }
                let (mut sum, mut cnt) = (0.0f32, 0u32);
                for t in &taus {
                    if t[i] != 0.0 && t[i].signum() == elected {
                        sum += t[i];
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    out[i] += lambda * sum / cnt as f32;
                }
            }
        }
    }
    out
}

/// Merge whole checkpoints; every tensor name and shape must match the base.
pub fn merge(
    base: &BTreeMap<String, Tensor>,
    children: &[BTreeMap<String, Tensor>],
    method: Method,
    lambda: f32,
) -> Result<BTreeMap<String, Tensor>, String> {
    let mut out = BTreeMap::new();
    for (name, b) in base {
        let mut cs: Vec<&[f32]> = Vec::new();
        for (i, c) in children.iter().enumerate() {
            let t = c
                .get(name)
                .ok_or_else(|| format!("child {i} lacks {name}"))?;
            if t.shape != b.shape {
                return Err(format!(
                    "child {i}: {name} shape {:?} != base {:?}",
                    t.shape, b.shape
                ));
            }
            cs.push(&t.data);
        }
        out.insert(
            name.clone(),
            Tensor {
                shape: b.shape.clone(),
                data: merge_tensor(&b.data, &cs, method, lambda),
            },
        );
    }
    for (i, c) in children.iter().enumerate() {
        if let Some(extra) = c.keys().find(|k| !base.contains_key(*k)) {
            return Err(format!("child {i} has tensor {extra} that the base lacks"));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_trims_elects_and_averages_agreeing_signs() {
        let base = [0.0f32; 4];
        let a = [1.0f32, -0.1, 2.0, 0.0];
        let b = [3.0f32, 0.3, -0.5, 0.0];
        let m = merge_tensor(&base, &[&a, &b], Method::Ties { density: 0.5 }, 1.0);
        // a keeps {1, 2}, b keeps {3, -0.5}: idx0 both +, mean 2; idx2 sign(2-0.5)=+ → only a: 2.
        assert_eq!(m, vec![2.0, 0.0, 2.0, 0.0]);
        let lin = merge_tensor(&base, &[&a, &b], Method::Linear, 1.0);
        for (x, y) in lin.iter().zip([2.0, 0.1, 0.75, 0.0]) {
            assert!((x - y).abs() < 1e-6, "{lin:?}");
        }
    }

    #[test]
    fn merging_identical_children_returns_the_child() {
        let base = [1.0f32, 2.0, 3.0];
        let c = [1.5f32, 1.0, 3.0];
        for method in [Method::Linear, Method::Ties { density: 1.0 }] {
            assert_eq!(merge_tensor(&base, &[&c, &c], method, 1.0), c.to_vec());
        }
    }
}
