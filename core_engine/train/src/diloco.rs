//! DiLoCo — distributed low-communication training (Douillard et al., 2023,
//! arXiv 2311.08105). Every worker runs `inner_steps` local AdamW steps from
//! the shared parameters θ; then only the pseudo-gradient Δ = θ − θ_local is
//! averaged across workers and applied with an outer Nesterov-momentum SGD
//! step. Communication drops by the factor `inner_steps` compared with
//! synchronous data parallelism, which is what makes training across weak,
//! loosely connected machines feasible. It adds no FLOPs.

use crate::model;
use crate::trainer::Trainer;

/// Averaging transport between workers (forge-comm ring, or in-process for tests).
pub trait Collective: Send {
    fn rank(&self) -> usize;
    fn world(&self) -> usize;
    /// Replace `buf` by the element-wise mean over all workers.
    fn all_reduce_mean(&mut self, buf: &mut [f32]) -> Result<(), String>;
}

pub struct DiLoCo {
    pub inner_steps: usize,
    pub outer_lr: f32,
    pub momentum: f32,
    names: Vec<String>,
    global: Vec<f32>,
    velocity: Vec<f32>,
    pub rounds: usize,
    pub bytes_communicated: u64,
}

impl DiLoCo {
    /// Snapshot the trainer's current parameters as the shared θ.
    pub fn new(tr: &Trainer, inner_steps: usize, outer_lr: f32, momentum: f32) -> DiLoCo {
        let names: Vec<String> = model::params(&tr.cfg.model)
            .into_iter()
            .map(|p| p.name)
            .collect();
        let global: Vec<f32> = names
            .iter()
            .flat_map(|n| tr.param(n).expect("bound").iter().copied())
            .collect();
        let velocity = vec![0.0; global.len()];
        DiLoCo {
            inner_steps,
            outer_lr,
            momentum,
            names,
            global,
            velocity,
            rounds: 0,
            bytes_communicated: 0,
        }
    }

    pub fn global(&self) -> &[f32] {
        &self.global
    }

    /// Start every worker from the same θ (rank 0's), via one averaging round.
    pub fn synchronise(
        &mut self,
        tr: &mut Trainer,
        coll: &mut dyn Collective,
    ) -> Result<(), String> {
        if coll.rank() != 0 {
            self.global.iter_mut().for_each(|x| *x = 0.0);
        }
        coll.all_reduce_mean(&mut self.global)?;
        let w = coll.world() as f32;
        self.global.iter_mut().for_each(|x| *x *= w);
        self.write_back(tr);
        Ok(())
    }

    fn write_back(&self, tr: &mut Trainer) {
        let mut off = 0;
        for n in &self.names {
            let dst = tr.machine.persistent_mut(n).expect("bound");
            dst.copy_from_slice(&self.global[off..off + dst.len()]);
            off += dst.len();
        }
    }

    /// One outer round: `inner_steps` local steps, average Δ, Nesterov outer step.
    /// Returns the mean local training loss of the round.
    pub fn round(&mut self, tr: &mut Trainer, coll: &mut dyn Collective) -> Result<f32, String> {
        let mut loss = 0.0;
        for _ in 0..self.inner_steps {
            loss += tr.train_step()?;
        }
        let mut delta = Vec::with_capacity(self.global.len());
        let mut off = 0;
        for n in &self.names {
            let local = tr.param(n).expect("bound");
            delta.extend(
                local
                    .iter()
                    .zip(&self.global[off..off + local.len()])
                    .map(|(l, g)| g - l),
            );
            off += local.len();
        }
        coll.all_reduce_mean(&mut delta)?;
        self.bytes_communicated += 4 * delta.len() as u64;
        // torch.optim.SGD(nesterov=True): v = μv + Δ; θ -= lr·(Δ + μv)
        let (mu, lr) = (self.momentum, self.outer_lr);
        for ((v, g), d) in self.velocity.iter_mut().zip(self.global.iter_mut()).zip(&delta) {
            *v = mu * *v + d;
            *g -= lr * (d + mu * *v);
        }
        self.write_back(tr);
        self.rounds += 1;
        Ok(loss / self.inner_steps as f32)
    }
}

/// In-process collective for tests and single-machine multi-worker runs.
pub mod local {
    use super::Collective;
    use std::sync::{Arc, Barrier, Mutex};

    pub struct Shared {
        sum: Mutex<Vec<f32>>,
        barrier: Barrier,
        world: usize,
    }

    pub struct Local {
        rank: usize,
        shared: Arc<Shared>,
    }

    pub fn group(world: usize) -> Vec<Local> {
        let shared = Arc::new(Shared {
            sum: Mutex::new(Vec::new()),
            barrier: Barrier::new(world),
            world,
        });
        (0..world)
            .map(|rank| Local {
                rank,
                shared: shared.clone(),
            })
            .collect()
    }

    impl Collective for Local {
        fn rank(&self) -> usize {
            self.rank
        }
        fn world(&self) -> usize {
            self.shared.world
        }
        fn all_reduce_mean(&mut self, buf: &mut [f32]) -> Result<(), String> {
            let s = &self.shared;
            if self.rank == 0 {
                *s.sum.lock().unwrap() = vec![0.0; buf.len()];
            }
            s.barrier.wait();
            {
                let mut sum = s.sum.lock().unwrap();
                if sum.len() != buf.len() {
                    return Err("all_reduce_mean: length mismatch across workers".into());
                }
                sum.iter_mut().zip(buf.iter()).for_each(|(a, b)| *a += b);
            }
            s.barrier.wait();
            let inv = 1.0 / s.world as f32;
            buf.iter_mut()
                .zip(s.sum.lock().unwrap().iter())
                .for_each(|(b, a)| *b = a * inv);
            s.barrier.wait();
            Ok(())
        }
    }
}
