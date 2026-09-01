//! Dev-only optimizer evaluation counters, behind the off-by-default
//! `counters` feature.
//!
//! Four quantities, all observation-only — nothing here is read by any numeric
//! path, exactly like the `Note::PirlsExhausted` bookkeeping this mirrors:
//!
//! 1. the per-stage evaluation split of the two-stage GLMM search;
//! 2. evaluations recorded after the last strict improvement of a stage's
//!    incumbent (the trust-radius shrink phase);
//! 3. PIRLS iterations per outer evaluation, as a histogram;
//! 4. AGQ evaluations and the node evaluations they cost.
//!
//! With the feature OFF the struct is zero-sized and every recording method is
//! an empty `#[inline(always)]` body, so call sites need no `cfg` and the
//! shipped build is byte-identical to one with no counters in it at all.
//! Plain `Copy` data, fixed-size arrays only: the struct may never allocate.

#[cfg(feature = "counters")]
use crate::glmm::PIRLS_MAX_ITERS;

/// Which of the two BOBYQA searches an evaluation belongs to. The LMM routes
/// and the single-stage GLMM path record everything as `Two`, matching
/// `GlmmFit::n_eval`'s "0 + stage 2 on the single-stage path" convention.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    /// The θ-only first search of the two-stage GLMM fit.
    One = 0,
    /// The full/final search: the second stage of the two-stage fit, or the
    /// only search on the LMM and single-stage GLMM routes.
    Two = 1,
}

/// One histogram bucket per possible PIRLS iteration count, `0..=PIRLS_MAX_ITERS`.
#[cfg(feature = "counters")]
pub const PIRLS_HIST_LEN: usize = PIRLS_MAX_ITERS + 1;

/// Observation-only optimizer counters for one fit: per-stage eval counts,
/// shrink-phase markers, and the PIRLS-iteration histogram.
#[cfg(feature = "counters")]
#[derive(Clone, Copy, Debug)]
pub struct EvalCounters {
    /// Objective evaluations per stage, indexed by `Stage as usize`.
    pub stage_evals: [u32; 2],
    /// 1-based index of the evaluation that last improved the stage's
    /// incumbent; 0 if the stage never improved on its first value.
    pub stage_last_improve: [u32; 2],
    /// Best objective seen per stage. Private: it exists to decide
    /// `stage_last_improve`, and exposing it would invite a caller to read a
    /// value the fit reports properly elsewhere.
    stage_best: [f64; 2],
    /// `pirls_hist[i]` = number of outer evaluations whose PIRLS solve ran `i`
    /// iterations. Bucket `PIRLS_MAX_ITERS` also absorbs a cap-out.
    pub pirls_hist: [u32; PIRLS_HIST_LEN],
    /// Iterations of the PIRLS solve currently running, committed to the
    /// histogram by `commit_pirls_iters` once the outer evaluation ends.
    pending_pirls_iters: u32,
    /// AGQ deviance evaluations (`nagq > 1` only).
    pub agq_evals: u32,
    /// Sum over AGQ evaluations of (clusters x nodes-per-cluster) — the
    /// evaluations-x-nodes product the AGQ counter asks for.
    pub agq_node_evals: u64,
}

#[cfg(feature = "counters")]
impl EvalCounters {
    pub(crate) fn new() -> Self {
        EvalCounters {
            stage_evals: [0; 2],
            stage_last_improve: [0; 2],
            stage_best: [f64::INFINITY; 2],
            pirls_hist: [0; PIRLS_HIST_LEN],
            pending_pirls_iters: 0,
            agq_evals: 0,
            agq_node_evals: 0,
        }
    }

    /// Per-fit reset, called where the workspace's other observation-only
    /// counters reset, so a reused workspace never carries a prior draw's counts.
    pub(crate) fn reset(&mut self) {
        *self = Self::new();
    }

    /// One objective evaluation with its returned value. Strict improvement
    /// only, mirroring the incumbent-gated snapshots in `fit_glmm`.
    pub(crate) fn record_eval(&mut self, stage: Stage, obj: f64) {
        let s = stage as usize;
        self.stage_evals[s] += 1;
        if obj < self.stage_best[s] {
            self.stage_best[s] = obj;
            self.stage_last_improve[s] = self.stage_evals[s];
        }
    }

    /// Called once per PIRLS iteration; the last write of a solve is the one
    /// that counts, so an early return out of the iteration loop still leaves
    /// the right value pending.
    pub(crate) fn set_pirls_iters(&mut self, iters: usize) {
        self.pending_pirls_iters = iters as u32;
    }

    pub(crate) fn commit_pirls_iters(&mut self) {
        let bucket = (self.pending_pirls_iters as usize).min(PIRLS_MAX_ITERS);
        self.pirls_hist[bucket] += 1;
        self.pending_pirls_iters = 0;
    }

    pub(crate) fn record_agq_eval(&mut self, nodes: u64) {
        self.agq_evals += 1;
        self.agq_node_evals += nodes;
    }

    /// Evaluations recorded after the stage's last incumbent improvement.
    pub fn evals_after_last_improve(&self, stage: Stage) -> u32 {
        let s = stage as usize;
        self.stage_evals[s] - self.stage_last_improve[s]
    }
}

/// Feature-off twin: same names and signatures as the real one. Zero-cost
/// contract: see the module header.
#[cfg(not(feature = "counters"))]
#[derive(Clone, Copy, Debug)]
pub struct EvalCounters;

#[cfg(not(feature = "counters"))]
impl EvalCounters {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        EvalCounters
    }
    #[inline(always)]
    pub(crate) fn reset(&mut self) {}
    #[inline(always)]
    pub(crate) fn record_eval(&mut self, _stage: Stage, _obj: f64) {}
    #[inline(always)]
    pub(crate) fn set_pirls_iters(&mut self, _iters: usize) {}
    #[inline(always)]
    pub(crate) fn commit_pirls_iters(&mut self) {}
    #[inline(always)]
    pub(crate) fn record_agq_eval(&mut self, _nodes: u64) {}
}
