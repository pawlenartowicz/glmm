//! GLMM dual-arithmetic scratch and the `N` dispatch.
//!
//! Sizes and owns the θ-dependent PIRLS/AGQ buffers at a non-`f64` scalar
//! (`Dual<N>` for a gradient, `HyperDual<N, H>` for a gradient + Hessian), so
//! `laplace_gradient`/`laplace_hessian` can differentiate the joint
//! Laplace deviance without touching the `f64` fit path's own buffers. `N` is
//! a compile-time lane count; the runtime model dimension `m = n_theta + p`
//! picks the smallest instantiated `N` that covers it via [`GlmmDualScratch`],
//! an enum over the ten instantiated variants (`D4`/`D5`/`D6`/`D8`/`D12` and
//! `H4`/`H5`/`H6`/`H8`/`H12`) — or, above the top rung, the top rung with the
//! gradient run in `⌈m / N⌉` passes.
//!
//! ## When a derivative request falls back
//!
//! `laplace_gradient` and `laplace_hessian` return `DerivStatus::Unsupported`
//! — a routing answer, not an error — in exactly three cases. Each names the
//! caller's own fallback:
//!
//! - **(a) [`supports_shape`] is false** — the packed-row layout
//!   (`GlmmLayout::Packed`), the one kernel with no dual twin.
//!   `GlmmLayout::for_design` sends two families of design there: every design
//!   `classify_design` answers `Solver::Sparse` for (past the dense envelope,
//!   slopes on an extra grouping, a crossed tail past the dense-tail byte
//!   bound), and an in-envelope extras design whose core block is too wide for
//!   the structured route (`!structured_extras_eligible()`). Caller fallback:
//!   on the SE path `se::joint_hessian_cov` asks `assembled::joint_hessian`
//!   first, and that engine DOES take the packed-row layout, so the stencil
//!   (`se::packed_fd_hessian_cov`) is reached only where the assembled pass
//!   declines as well; the optimizer runs BOBYQA on the objective.
//!   `supports_shape` also refuses a STRUCTURED design whose crossed tail is
//!   wider than [`DUAL_TAIL_MAX`]: such a design (a GLMM with more than 500
//!   crossed levels whose dense tail fits) searches with BOBYQA, and since the
//!   assembled engine reads `supports_shape` off the packed-row layout, its SE
//!   takes the FD Hessian of the blocked and structured layouts. Every other
//!   blocked and structured shape IS differentiated here — the structured ones
//!   through `structured_laplace_deviance` over the same
//!   `pirls_solve_blocked_extras` the `f64` fit path runs. Checked first in
//!   both entry points' routing gate, before `m` is even computed.
//! - **(b) `m = n_theta + p > MAX_DUAL_N` (12)** — for the **Hessian only**
//!   (`NLanes::pick`). The gradient is not refused there: it runs in
//!   `⌈m / 12⌉` passes of at most 12 seeded coordinates on the top rung, each
//!   pass writing its own slice of `grad`. This module's second-order pass
//!   cannot do the same — a second-derivative block spanning two chunks needs
//!   both coordinates' first-order lanes live in the same pass, which is `2N`
//!   lanes, the cap again. The caller's fallback here is not the FD stencil:
//!   `se::joint_hessian_cov` asks `assembled::joint_hessian` first, and that
//!   pass reads second-order information off FIRST-order lanes, so it chunks
//!   like a gradient and covers every `m`. This clause is a statement about
//!   the hyper-dual pass, not about whether an exact Hessian exists.
//! - **(c) the packed-row layout** (`deviance.rs`'s `GlmmLayout::Packed` arm)
//!   — is **(a) itself, not a separate guard.** `supports_shape` refuses
//!   `Packed` on the layout alone, and `deviance.rs` takes that arm on exactly
//!   the designs `GlmmLayout::for_design` maps to `Packed` — the
//!   `Solver::Sparse` ones and the oversized-core extras ones alike. Named here
//!   so the fallback table stays complete against `deviance.rs`'s own
//!   three-way routing.
//!
//! **Memory is never a fallback reason.** Only compute shape, (a)–(c), routes
//! to `Unsupported`; there is no byte budget on the blocked and structured
//! layouts. Within that, the scratch follows three rules:
//!
//! - **A buffer a shape never reads is not allocated.** The structured row
//!   buffer (`StructuredScratch::m_core_buf`) exists only on the structured
//!   layout, the observed factors only on a non-canonical link, the crossed
//!   values once per grouping rather than once per row, and the assembled
//!   Hessian keeps no per-row pass values: its row walk forms and drops them
//!   (`assembled::assemble`). The coupling-shaped twins of the structured
//!   layout hold the design's own coupling pattern ([`dual_coupling_pairs`])
//!   once it is known, not the `rows · MAX_EXTRA_GROUPINGS` bound.
//! - **On the blocked layout the assembled Hessian runs one block of rows at
//!   a time** (`assembled::run_assembled_hessian_blocked`, above
//!   `assembled::SE_ROW_BLOCK` rows): every per-row and per-cluster dual
//!   buffer is block-sized ([`GlmmDualScratch::for_shape_row_blocked`]), so
//!   the scratch does not grow with `n`, and the pass is faster for it
//!   (cache residency and fewer first-touch page faults, the same work).
//! - **On the structured layout it streams its row values instead**
//!   (above the same block size): the crossed Schur couples every cluster, so
//!   the cluster- and pair-level buffers stay whole, but the dual PIRLS
//!   (`pirls::structured_stream_deviance`) and the assembly form each row
//!   block's values and drop them ([`GlmmDualScratch::for_shape_streamed`]),
//!   with a bit-identical Hessian at the same speed.
//! - **The assembled Hessian's lane width comes from a measured table**
//!   ([`NLanes::pick_assembled`]). At `m` 13–18 the narrower rung is faster
//!   on every host and also holds less scratch. At `m` 9–10 it is faster on
//!   x86 but up to 12% slower on an Apple M4, taken for 54% less scratch.
//! - **No other speed is traded for memory.** There is no slower memory-lean
//!   route beyond that table row; one would apply only above a size
//!   threshold, and none is implemented.
//!
//! Measured 2026-10-07 on a binomial fit with four random effects per group
//! (`m = 14`, pawel, locked P-core): at 300k rows the buffer and lane rules
//! take the SE scratch from 906 MB to 324 MB, its first-touch page faults
//! from 227k to 75k, and the SE step from 3.34 s to 2.15 s; the row
//! blocks then take the scratch to 19 MB, the faults to about ten and the SE
//! step to 0.70 of that. At 3M rows they take the fit's peak from 10.7 GB to
//! 4.4 GB with identical results, and with the row blocks to 1.33 GB, within
//! 1% of the same fit with `WaldSe::Rx` (1.31 GB), with Hessian SEs that
//! move by at most 3e-13 relative. On a binomial fit with four crossed
//! groupings (`m = 6`) at 3M rows the streamed pass takes the SE scratch from
//! 2.0 GB to 0.9 GB and the fit's peak from 3.2 GB to 2.1 GB (1.25 GB with
//! `WaldSe::Rx`), with identical SEs; what is left is pair-level.

use super::agq::{agq_deviance, agq_deviance_vec, agq_eligible, agq_len, ClusterRowIndex};
use super::deviance::{blocked_laplace_deviance, structured_laplace_deviance};
use super::pirls::{obs_len, BetaStep, DualStep, TailKernel};
use super::workspace::{
    FitData, GlmmLayout, GlmmWorkspace, PirlsScratch, StructuredPattern, StructuredScratch,
};
use crate::dual::{Dual, HyperDual};
use crate::lmm::LmmGroupings;
use crate::scalar::Scalar;
use faer::{Mat, MatRef};

/// Outcome of a derivative request. `Unsupported` is a routing answer, not an
/// error: the caller falls back to the finite-difference Hessian (`se.rs`) or
/// to BOBYQA on the objective.
pub(crate) enum DerivStatus {
    /// Objective value at the seeded parameters; the gradient (and Hessian, if
    /// requested) has been written to the caller's buffers.
    Ok(f64),
    /// PIRLS did not converge, or the objective is non-finite. On the AGQ route
    /// the non-finite objective IS the failure signal — `agq_deviance` returns a
    /// bare `+∞` and carries no `conv` flag. Buffers are untouched.
    NotConverged,
    /// Not a shape `supports_shape` accepts, or `m` exceeds the largest
    /// instantiated `N`.
    Unsupported,
}

/// The `(cluster, crossed column)` pairs of the design's full coupling
/// pattern, when the workspace's pattern is known to be that full pattern:
/// `pattern.coup_mask == Some(0)`, the CSR built for this fit's data with no
/// crossed grouping pinned, which is the pattern every dual kernel call
/// builds and reads (`deviance::structured_laplace_deviance` keys its CSR
/// cache on mask 0 at a dual `T`). `None` otherwise (no CSR built yet this
/// fit, or one built at a pinned θ), and the dual twins then take the
/// [`coupling_len`](super::workspace::coupling_len) bound.
///
/// The bound is `q_core · min(s·e, rows·MAX_EXTRA_GROUPINGS)`; the pattern
/// holds one pair per distinct crossed level a cluster's rows touch, about
/// 2.2 per row on a four-way crossed binomial at 3M rows against the bound's
/// 6, which on `Dual<8>` is about 0.8 GB per coupling-sized buffer.
pub(crate) fn dual_coupling_pairs(
    pattern: &super::workspace::StructuredPattern,
    s: usize,
) -> Option<usize> {
    (pattern.coup_mask == Some(0)).then(|| pattern.coup_ptr[s] as usize)
}

/// Compact coupling length of the dual twins: `q_core` values per pair of
/// [`dual_coupling_pairs`] when known and the shape has crossed extras, the
/// [`coupling_len`](super::workspace::coupling_len) bound otherwise.
pub(crate) fn dual_coupling_len(
    rows: usize,
    s: usize,
    q_core: usize,
    e: usize,
    coup_pairs: Option<usize>,
) -> usize {
    match coup_pairs {
        Some(pairs) if e > 0 && s > 0 => (q_core * pairs).max(1),
        _ => super::workspace::coupling_len(rows, s, q_core, e),
    }
}

/// The θ-dependent buffers `blocked_laplace_deviance` and the AGQ kernels write,
/// at a non-`f64` scalar: [`PirlsScratch`] and [`StructuredScratch`]
/// instantiated at `T`, sized by the same two constructors the `f64` workspace
/// uses (`PirlsScratch::for_shape`, `StructuredScratch::for_shape`).
pub(crate) struct GlmmDualBufs<T: Scalar> {
    pub(super) params: Vec<T>, // m (seeded per call)
    pub(super) beta: Vec<T>,   // p
    pub(super) pirls: PirlsScratch<T>,
    /// Sized via `StructuredScratch::for_shape` even on the no-extras blocked
    /// path, where the row buffers (`m_core_buf`, `cross_theta`), the coupling
    /// and the Schur block sit at their `.max(1)` minimum and only
    /// `core_blocks` keeps its `s·q_p²`: the zero-alloc gate is about repeat
    /// calls, and a lazy first-extras-call allocation would break it on the
    /// very shape it is meant to cover.
    pub(super) structured: StructuredScratch<T>,
    // Per-solve controls handed to every dual kernel call; see `pirls::DualStep`.
    // Carries the blocked twin's scratch (`obs_blocks`, `s · q_p²`) and the
    // structured twin's (`obs_core_blocks` `(q_core² · s).max(1)`,
    // `obs_coupling` `coupling_len`, `obs_schur_blk` `(e²).max(1)`,
    // `obs_rhs` `k`) — same sizes `for_shape` allocates.
    pub(super) dual: DualStep<T>,
    /// The assembled gradient at `T`, one entry per `γ` coordinate: the value
    /// part is the gradient, the lanes of a seeded pass are that pass's
    /// columns of the joint Hessian.
    pub(super) grad_t: Vec<T>, // m
    /// Row passes, adjoint solve and observed factor of the assembled
    /// derivative, at `T`. Its own struct because every length in it is a
    /// function of the same shape terms and the assembly reads them as one
    /// group.
    pub(super) asm: super::assembled::AssemblyBufs<T>,
    /// The row-blocked assembled Hessian's block scratch
    /// (`assembled::run_assembled_hessian_blocked`), present only on a scratch
    /// built by [`GlmmDualScratch::for_shape_row_blocked`]. That scratch sizes
    /// every per-row and per-cluster buffer above (and `asm`) at its minimum:
    /// the blocked pass keeps them in here, block-sized, and reads only the
    /// `k`-length mode `pirls.u`, `params`, `beta` and `grad_t` of the twin.
    pub(super) row_blocks: Option<Box<super::assembled::RowBlockBufs<T>>>,
    /// The streamed structured pass's block buffers (`pirls::StreamRows`),
    /// present only on a scratch built by
    /// [`GlmmDualScratch::for_shape_streamed`]. That scratch sizes the dual
    /// PIRLS row buffers (`pirls.eta`, `prob`, `w`, `eta_fixed`, `mu`,
    /// `m_buf`) and `structured.m_core_buf` at their minimum: the streamed
    /// solve (`pirls::structured_stream_deviance`) and the assembly form every
    /// per-row value block by block in here. The cluster- and pair-level
    /// buffers (`core_blocks`, `coupling`, `schur_blk`, `asm.tail_h`) stay
    /// whole, since the crossed Schur couples every cluster.
    pub(super) stream_rows: Option<Box<super::pirls::StreamRows<T>>>,
}

/// `f64` mode-transfer buffers of a dual scratch, sized once in `for_shape`
/// alongside its `GlmmDualBufs<T>` and kept as a SEPARATE field of
/// [`GlmmDualScratch`] rather than folded into `GlmmDualBufs<T>` itself:
/// `laplace_gradient`/`laplace_hessian` need to read `u_mode` while `bufs` is
/// borrowed `&mut` for the dual kernel call below, and destructuring
/// `GlmmDualScratch` gives `bufs` and `mode` as disjoint bindings the borrow
/// checker accepts — folding `u_mode` into `GlmmDualBufs` would make it alias
/// the same struct `bufs` already borrows whole.
///
/// Sized once, so `laplace_gradient`/`laplace_hessian` allocate no `saved_u` or
/// `u_mode` `Vec` per call (pinned by
/// `dual_gradient_repeat_calls_allocate_nothing`, `tests.rs`).
pub(crate) struct GlmmModeBufs {
    /// `f64` snapshot of `ws.pirls.u[..k.max(1)]` taken right before the mode solve
    /// mutates `ws.pirls.u` in place, and copied back after so the workspace's fit
    /// state comes back as found.
    pub(super) saved_u: Vec<f64>,
    /// The converged PIRLS mode `ws.pirls.u[..k]`, copied out of `ws.pirls.u` once per
    /// call before the dual kernel(s) below read it as `run_gradient`'s /
    /// `run_hessian`'s `u_mode` argument.
    pub(super) u_mode: Vec<f64>,
}

impl GlmmModeBufs {
    fn for_shape(k: usize) -> GlmmModeBufs {
        GlmmModeBufs {
            saved_u: vec![0.0; k.max(1)],
            u_mode: vec![0.0; k],
        }
    }
}

/// The θ-dependent buffers, at a non-`f64` scalar, at the lane count they
/// were built for, with the `ClusterRowIndex` and the `GlmmModeBufs` every
/// lane count carries alike.
///
/// `pub(crate)` rather than module-private: `GlmmWorkspace::dual_scratch`
/// (`workspace.rs`) names this type in its field, so it — and, transitively,
/// `GlmmDualBufs` in [`DualLanes`]'s variants — must be nameable from a
/// sibling module (`private_interfaces` requires a field's type be at least
/// as visible as the field itself).
pub(crate) struct GlmmDualScratch {
    pub(super) lanes: DualLanes,
    /// The index the AGQ cluster-outer arm needs, built once when the scratch
    /// is (re)allocated. The AGQ dual entry hands it to
    /// `agq_deviance`/`agq_deviance_vec` as `Some(idx)` EXPLICITLY, and the
    /// f64 mode solve reads the same index so it runs the identical
    /// cluster-outer arm before the dual kernel ever sees the mode — never
    /// `ws.cluster_rows`, which `fit_glmm` fills only under the `parallel`
    /// feature or for serial vector AGQ without extra groupings, so a serial
    /// scalar-AGQ fit would silently drop to the node-outer arm.
    pub(super) idx: ClusterRowIndex,
    /// The mode-transfer snapshot/restore buffers.
    pub(super) mode: GlmmModeBufs,
}

/// The typed buffer set of a [`GlmmDualScratch`], one variant per
/// instantiated lane count.
pub(crate) enum DualLanes {
    D4(GlmmDualBufs<Dual<4>>),
    D5(GlmmDualBufs<Dual<5>>),
    D6(GlmmDualBufs<Dual<6>>),
    D8(GlmmDualBufs<Dual<8>>),
    D12(GlmmDualBufs<Dual<12>>),
    H4(GlmmDualBufs<HyperDual<4, 10>>),
    H5(GlmmDualBufs<HyperDual<5, 15>>),
    H6(GlmmDualBufs<HyperDual<6, 21>>),
    H8(GlmmDualBufs<HyperDual<8, 36>>),
    H12(GlmmDualBufs<HyperDual<12, 78>>),
}

impl GlmmDualScratch {
    /// Which `NLanes` member this scratch was built for — half of the
    /// reuse-policy check every entry point runs before calling `for_shape`
    /// (the other half is [`Self::matches_shape`]): a request at the same
    /// `(order, N)` and shape reuses the stored scratch, a different one
    /// reallocates once (the zero-alloc-on-repeat gate is about
    /// repeat calls at the SAME shape; a shape change is not a repeat call).
    pub(super) fn lanes(&self) -> NLanes {
        match self.lanes {
            DualLanes::D4(..) => NLanes::D4,
            DualLanes::D5(..) => NLanes::D5,
            DualLanes::D6(..) => NLanes::D6,
            DualLanes::D8(..) => NLanes::D8,
            DualLanes::D12(..) => NLanes::D12,
            DualLanes::H4(..) => NLanes::H4,
            DualLanes::H5(..) => NLanes::H5,
            DualLanes::H6(..) => NLanes::H6,
            DualLanes::H8(..) => NLanes::H8,
            DualLanes::H12(..) => NLanes::H12,
        }
    }

    /// The `DualStep::exact` flag the last dual kernel call left: true iff
    /// that call's lanes were the answer on their own, so the refinement loop
    /// in `run_gradient` / `run_hessian` did not have to run. The one thing
    /// that separates a cell where one kernel call suffices from one where it
    /// does not, and therefore the observable a test pins that split on.
    #[cfg(test)]
    pub(crate) fn exit_exact(&self) -> bool {
        match &self.lanes {
            DualLanes::D4(b) => b.dual.exact,
            DualLanes::D5(b) => b.dual.exact,
            DualLanes::D6(b) => b.dual.exact,
            DualLanes::D8(b) => b.dual.exact,
            DualLanes::D12(b) => b.dual.exact,
            DualLanes::H4(b) => b.dual.exact,
            DualLanes::H5(b) => b.dual.exact,
            DualLanes::H6(b) => b.dual.exact,
            DualLanes::H8(b) => b.dual.exact,
            DualLanes::H12(b) => b.dual.exact,
        }
    }

    /// `max|u − u_prev|` over the value parts of the buffers the last dual
    /// kernel call left — how far that call's final PIRLS step moved.
    ///
    /// The Laplace objective is built entirely at `u`, so this distance
    /// measures how converged that call left the mode, not a mismatch between
    /// the objective's terms. A pass entered with a step floor
    /// (`DualStep::min_iters`) exits at a different distance from one entered
    /// without, which is the one thing the two Hessian passes do not share.
    #[cfg(test)]
    pub(crate) fn exit_mode_step(&self) -> f64 {
        fn step<T: Scalar>(b: &GlmmDualBufs<T>) -> f64 {
            b.pirls
                .u
                .iter()
                .zip(&b.pirls.u_prev)
                .map(|(a, c)| (a.value() - c.value()).abs())
                .fold(0.0, f64::max)
        }
        match &self.lanes {
            DualLanes::D4(b) => step(b),
            DualLanes::D5(b) => step(b),
            DualLanes::D6(b) => step(b),
            DualLanes::D8(b) => step(b),
            DualLanes::D12(b) => step(b),
            DualLanes::H4(b) => step(b),
            DualLanes::H5(b) => step(b),
            DualLanes::H6(b) => step(b),
            DualLanes::H8(b) => step(b),
            DualLanes::H12(b) => step(b),
        }
    }
}

/// Largest instantiated FIRST-derivative lane count. `NLanes::pick` rounds `m`
/// up to this; above it a first-order request runs in `⌈m / MAX_DUAL_N⌉`
/// passes of at most `MAX_DUAL_N` seeded coordinates on this same top rung.
/// The gradient chunks that way here, and so does the joint Hessian in
/// `assembled.rs`, which reads second-order information off FIRST-order lanes.
/// This module's hyper-dual Hessian cannot — a cross-chunk second-derivative
/// block needs both coordinates' first-order lanes live in one pass — so
/// `NLanes::pick(m, true)` and `laplace_hessian` return
/// `DerivStatus::Unsupported` above the cap and the caller takes the assembled
/// pass. Mirrors the `GlmmDualScratch` variants —
/// change together, along with `MAX_DUAL_H` below, which hardcodes the same
/// lane set. The assembled Hessian does not take the covering rung:
/// [`NLanes::pick_assembled`] reads its rung off a measured table, because a
/// pass costs little and padded lanes are the work (two `Dual<8>` passes beat
/// two `Dual<12>` at `m` 13 to 16).
pub(crate) const MAX_DUAL_N: usize = 12;

/// Packed length of the largest instantiated `HyperDual<N, H>`'s second-
/// derivative block, sized to hold any of them — `run_hessian`'s settle-check
/// and prev-call scratch are sized once at this bound rather than per-`N`. Its
/// own constant rather than `MAX_DUAL_N·(MAX_DUAL_N+1)/2`: the two ladders are
/// allowed to differ, because the gradient can chunk and the Hessian cannot, so
/// a gradient-only rung above the largest `HyperDual` would otherwise grow this
/// buffer for nothing. Equal to `12·13/2` — change together with the
/// `HyperDual` variants of `GlmmDualScratch`.
const MAX_DUAL_H: usize = 78;

// While the top `Dual` and `HyperDual` rungs are the same `N`, the two
// constants must agree; a later gradient-only rung is what breaks the equality,
// and that change removes this assertion deliberately rather than by accident.
const _: () = assert!(MAX_DUAL_H == MAX_DUAL_N * (MAX_DUAL_N + 1) / 2);

/// Cap on the dual re-entries the FALLBACK refinement loop may take before
/// the returned derivatives stop moving. A dual call normally takes an
/// exact-Hessian step (`pirls::DualStep`: canonical `A`, or the
/// observed-information `A_obs` on a non-canonical link), so the IFT lanes are
/// reached in one step and the loop is skipped. It is entered on
/// `DualStep::exact == false`, a non-PD observed factor — one block on the
/// blocked path, the whole crossed-tail Schur on the structured-extras path. There the lanes contract by
/// `‖I − A⁻¹h_uu‖` per step; a Fisher-only fallback needed 5–7
/// calls on the FD gates' draws and 9–10 on `sim_gamma` at its converged fit
/// (each call two steps), so 12 keeps two calls of headroom above the worst
/// measured. Hitting the cap is `DerivStatus::NotConverged`, not a silently
/// truncated gradient.
///
/// **Counted differently at its two use sites — read the comment at each
/// before changing either:** `run_gradient`'s `max_calls` treats this as the
/// TOTAL kernel-call cap for the loop. `run_hessian`'s `max_reads` treats it
/// as the cap on the loop's READ calls only, counted after the first call
/// that sits outside the loop — so the Hessian's true total is
/// `1 + MAX_DUAL_REFINEMENTS` kernel calls, one more than the gradient's own
/// use of this same constant. Both give up to `MAX_DUAL_REFINEMENTS - 1`
/// refinement compares (the first read call has no previous call to compare
/// against).
pub(crate) const MAX_DUAL_REFINEMENTS: usize = 12;

/// Settle band of the dual refinement loops: a re-entry's derivatives count as
/// settled when every entry moved by less than `DUAL_SETTLE_REL·(1 + |d|)`
/// since the previous call. The shape is the kernel's own mixed-deviance
/// convergence check (`PirlsControl::stop`'s `tol * (1.0 + mixed.abs())`):
/// relative, with an absolute floor so a near-zero lane still settles. The
/// value is not tuned; it guards against reading lanes the loop is still
/// moving. Read by `run_gradient`, `run_hessian`, and `assembled.rs`'s
/// `run_assembled_hessian` and `blocked_chunks`.
pub(super) const DUAL_SETTLE_REL: f64 = 1e-10;

/// Largest crossed tail width `e` the dual kernel factors densely. Above it
/// the derivative entry points return `Unsupported` and the caller falls
/// back (BOBYQA on the objective, the FD Hessian in `se.rs`); there is no
/// hand adjoint on faer's sparse tail.
///
/// Measured 2026-09-02 on a clock-locked machine: the sweep found no
/// crossover in `(0, 500]` — dual Hessian 17–27× one objective (rising to a
/// plateau near 25× from `e = 192`), gradient 3–6×, against the
/// `2m² = 32` FD equivalent; VerbAgg/grouseticks 161×/97× vs 162×/98×. It
/// therefore sits at the end of the measured range. `classify_design` keeps a
/// GLMM on the structured layout past it while the dense tail fits, so the
/// `Structured` clause of [`supports_shape`] fires: such a design searches
/// with BOBYQA and takes the FD Hessian.
///
/// NOT the same boundary as `sparse::TAIL_SPARSE_MIN` — that one chooses
/// between two f64 factorizations of the LMM tail and has its own
/// measurement.
pub(crate) const DUAL_TAIL_MAX: usize = 500;

/// Shapes the dual kernel can differentiate: the blocked layout, or the
/// structured layout with a crossed tail the dense generic factor can carry.
/// Nested-only designs have `k_crossed() == 0` and are always in. The packed-row
/// layout is out — there is no dual twin of its kernel.
///
/// One of the TWO owners of the layout question, and the one that answers
/// "is there a dual twin of this layout's PIRLS kernel": `laplace_gradient`
/// and `laplace_hessian` route on this alone, so they can never drift apart.
/// The other owner is `assembled::assembly_routes`, which answers "does the
/// assembled engine run here" and takes the packed-row layout this one
/// refuses; the exact-Hessian SE branch (`se::joint_hessian_cov`) reads both.
/// Widening this one is a hand-adjoint change (the regime above the
/// boundary), not a local edit at a call site.
pub(crate) fn supports_shape(layout: GlmmLayout, g: &LmmGroupings) -> bool {
    match layout {
        GlmmLayout::Blocked => true,
        GlmmLayout::Structured => g.k_crossed() <= DUAL_TAIL_MAX,
        GlmmLayout::Packed => false,
    }
}

/// Which instantiated `(order, N)` pair a derivative request resolves to.
/// [`GlmmDualScratch::for_shape`] picks the smallest member whose `N` covers
/// `m`, for the requested order — `Dual` for a gradient-only request,
/// `HyperDual` when a Hessian is also wanted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NLanes {
    D4,
    D5,
    D6,
    D8,
    D12,
    H4,
    H5,
    H6,
    H8,
    H12,
}

impl NLanes {
    /// Smallest instantiated lane count at or above `m`, for the given order.
    /// `hessian == true` selects `HyperDual` (second derivatives wanted);
    /// `false` selects `Dual` (gradient only). A gradient request is always
    /// `Some` — the top rung covers every `m` by chunking; a Hessian request
    /// is `None` iff `m > MAX_DUAL_N`, the same guard `laplace_hessian` runs
    /// first, exposed here so a caller that only needs the routing decision
    /// (not the allocation) can make it.
    pub(crate) fn pick(m: usize, hessian: bool) -> Option<NLanes> {
        // Only the second-order request is capped. A first-order one above the
        // top rung runs in `⌈m / N⌉` passes of at most `N` seeded coordinates,
        // so the top rung covers every `m`; a second-order pass cannot chunk,
        // because a cross-chunk second-derivative block needs both
        // coordinates' first-order lanes live in the same pass, which is `2N`
        // lanes — the cap again.
        if hessian && m > MAX_DUAL_N {
            return None;
        }
        Some(match (hessian, m) {
            (false, 0..=4) => NLanes::D4,
            (false, 5) => NLanes::D5,
            (false, 6) => NLanes::D6,
            (false, 7..=8) => NLanes::D8,
            (false, _) => NLanes::D12,
            (true, 0..=4) => NLanes::H4,
            (true, 5) => NLanes::H5,
            (true, 6) => NLanes::H6,
            (true, 7..=8) => NLanes::H8,
            (true, _) => NLanes::H12,
        })
    }

    /// The first-order rung the assembled Hessian (`assembled.rs`) runs its
    /// chunked passes on, chosen for speed from a measured table rather than
    /// as the smallest rung covering `m`. A pass costs a fixed part (the dual
    /// PIRLS settle and the assembly's own row walk) plus a part per lane, so
    /// `⌈m/N⌉` passes of `N` lanes trade the two: two `Dual<8>` passes beat two
    /// `Dual<12>` at `m` 13 to 16 because the `Dual<12>` pair carries 8 padded
    /// lanes, and three `Dual<6>` beat two `Dual<12>` at 17 and 18. Where the
    /// covering rung is as fast, it stays (fewer passes, same numbers).
    ///
    /// Measured 2026-10-07 (pawel, locked P-core, and an Apple M4; SE step on
    /// the binomial blocked `q1`, `q3`, `q4` and structured `cross4` series at
    /// 30k and 300k rows, geometric mean of the ratio to `pick(m, false)`):
    /// `m` 13–16 `D8×2` 0.78–0.80, 17–18 `D6×3` 0.92, on both hosts, with
    /// bit-identical Hessians and 31–46% less dual scratch. At 9–10 `D5×2`
    /// replaces one `Dual<12>` pass: 0.93–0.96 on the blocked x86 series and
    /// 1.006 on `cross4`, but 1.00–1.12 on the M4, for 54% less dual scratch
    /// (a slowdown on the M4 alone is traded for that memory). Every other `m` up to 24 keeps the covering rung, and one table
    /// serves every target.
    ///
    /// A pinned table, a function of `m` alone: no timing input and no
    /// per-target branch, so a fit picks the same rung (and moves by the same
    /// round-off) on every host, WASM included. Above 24 (unmeasured) it is
    /// `pick(m, false)`'s top rung. The hyper-dual Hessian and the gradient
    /// keep [`NLanes::pick`].
    pub(crate) fn pick_assembled(m: usize) -> NLanes {
        match m {
            9..=10 => NLanes::D5,
            13..=16 => NLanes::D8,
            17..=18 => NLanes::D6,
            _ => NLanes::pick(m, false).expect("the gradient ladder covers every m"),
        }
    }
}

/// True iff this buffer set was sized for exactly this shape — every
/// shape-determining length `for_shape` chose is re-derived and compared
/// (checking `eta` covers `prob`/`w`/`eta_fixed`/`mu`, allocated together at
/// the same `rows`; `u` pins `k` and with it `u_prev`, `a_rhs`, and the
/// `GlmmModeBufs`; `core_blocks` pins `q_core`, `schur_blk` pins `e` — the
/// pair together pins `coupling` — and `m_core_buf` pins `extras`). The
/// `DualStep` twins mirror the same pinning: `obs_core_blocks` re-checks
/// `q_core`, `obs_schur_blk` re-checks `e`, and `obs_rhs` needs no
/// separate check because `u`'s `k` already covers it.
/// Both twin checks also pin `observed`: an empty twin matches only an
/// `observed == false` request, so a scratch sized without the twins is never
/// handed to a call that reads them.
/// The assembly scratch adds one length nothing else pins — `G_γ`'s `m·k`
/// product — plus `grad_t`'s `m`; every other buffer in `AssemblyBufs` is a
/// function of `m`, `k`, `rows`, `s`, `q_core` or `e`, each already pinned
/// above. `assembly` says which of the two cases the caller's variant is:
/// the `Dual` rungs carry the sized set, the `HyperDual` rungs an empty one.
/// The packed sub-struct adds `rows·width`, pinned here through `m_vals`.
/// Lengths only: the `ClusterRowIndex` built from `cluster_ids` is not
/// covered — same-shape data with different cluster assignment is still the
/// caller's responsibility.
#[allow(clippy::too_many_arguments)]
fn bufs_match_shape<T: Scalar>(
    b: &GlmmDualBufs<T>,
    m: usize,
    p: usize,
    k: usize,
    rows: usize,
    s: usize,
    q_p: usize,
    q_core: usize,
    e: usize,
    // `!extra_offsets.is_empty()`: whether the structured row buffers
    // (`m_core_buf`, `cross_theta`) carry storage; see `StructuredScratch::for_shape`.
    extras: bool,
    nagq: u8,
    // Whether this fit takes the observed-information step (`!is_canonical`),
    // which is what decides whether the `DualStep` twins carry storage.
    observed: bool,
    // Packed-row width on `GlmmLayout::Packed`, 0 on every other layout — the
    // same discriminator `for_shape` sizes from.
    packed_width: usize,
    // True on the `Dual` rungs, where the assembly runs and its scratch is
    // sized; false on the `HyperDual` rungs, which carry none.
    assembly: bool,
    // Whether the scratch must be the row-blocked one
    // (`GlmmDualScratch::for_shape_row_blocked`); a scratch of the other kind
    // never matches, so the two routes rebuild rather than share it.
    row_blocked: bool,
    // The design's coupling pairs ([`dual_coupling_pairs`]); the
    // coupling-shaped twins must hold at least what they give.
    coup_pairs: Option<usize>,
    // The streamed structured pass's block size
    // (`GlmmDualScratch::for_shape_streamed`), `None` on every other scratch;
    // a scratch of the other kind, or of another block size, never matches.
    stream_block: Option<usize>,
) -> bool {
    let d = DenseTwinShape::of(
        packed_width,
        k,
        rows,
        s,
        q_p,
        q_core,
        e,
        extras,
        observed,
        row_blocked,
    );
    let coup_len = dual_coupling_len(d.rows, d.s, d.q_core, d.e, coup_pairs);
    let asm_coup = if packed_width > 0 {
        1
    } else {
        dual_coupling_len(rows, s, q_core, e, coup_pairs)
    };
    // The streamed pass keeps no `n`-length dual row buffers (see
    // `GlmmDualBufs::stream_rows`).
    let prow = if stream_block.is_some() { 0 } else { d.rows };
    b.row_blocks.is_some() == row_blocked
        && b.stream_rows.as_ref().map(|r| r.block) == stream_block
        && b.params.len() == m
        // At least, not equal: a twin sized at the bound serves a later call
        // that knows the (smaller) pattern, and the kernels index only the
        // pattern's `q_core · coup_ptr[s]` prefix.
        && b.structured.coupling.len() >= coup_len
        && b.dual.obs_coupling.len() >= obs_len(d.observed, coup_len)
        && b.beta.len() == p
        && b.pirls.lam.len() == d.q_p * d.q_p
        && b.pirls.m_buf.len() == prow * d.q_p
        && b.pirls.eta.len() == prow
        && b.pirls.u.len() == d.k.max(1)
        && b.pirls.a_blocks.len() == (d.s * d.q_p * d.q_p).max(1)
        && b.pirls.agq_scratch.len() == agq_len(d.s, d.q_p, nagq)
        && b.structured.core_blocks.len() == (d.q_core * d.q_core * d.s).max(1)
        && b.structured.schur_blk.len() == (d.e * d.e).max(1)
        // Required, not defensive: a nested-only structured shape (`e = 0`,
        // `q_core = q_p + np`) and an extras-free shape agree on every other
        // length here at equal `q_core`, and `build_packed_m` writes
        // `m_core_buf[i·q_core + …]` on every row.
        && b.structured.m_core_buf.len()
            == (if d.extras && stream_block.is_none() {
                d.rows * d.q_core
            } else {
                0
            })
            .max(1)
        // `obs_core_blocks` pins `q_core` (and with it `obs_coupling`),
        // `obs_schur_blk` pins `e`; `u`'s `k`, already checked above, pins
        // `obs_rhs` — no separate check needed.
        && b.dual.obs_core_blocks.len()
            == obs_len(d.observed, (d.q_core * d.q_core * d.s).max(1))
        && b.dual.obs_schur_blk.len() == obs_len(d.observed, (d.e * d.e).max(1))
        && b.grad_t.len() == m
        && if assembly {
            b.asm.g_gamma.len() == (m * if row_blocked { 0 } else { k }).max(1)
                && (row_blocked || b.asm.tail_h.len() >= asm_coup)
                && (row_blocked || b.asm.obs_coup.len() >= obs_len(observed, asm_coup))
                && b.asm.packed_len() == rows * packed_width
                // `PackedAsmBufs::for_shape` sizes `obs` to `k²` on a
                // non-canonical packed shape and to nothing otherwise.
                // `packed_width > 0` mirrors that function, which allocates
                // nothing when `width == 0` — change together.
                && b.asm.packed_obs_len()
                    == obs_len(observed, if packed_width > 0 { k * k } else { 0 })
        } else {
            b.asm.g_gamma.is_empty()
        }
}

/// Shape terms the blocked and structured dual twins (`PirlsScratch<T>`,
/// `StructuredScratch<T>` and the `DualStep` observed twins) are sized from:
/// the caller's own on those two layouts, the MINIMUM on the packed one.
///
/// The packed-row route runs no dual PIRLS kernel — the assembled engine
/// rebuilds `M`, η, μ, `W` and `A` at `T` in its own buffers
/// (`assembled::PackedAsmBufs`) — so none of those twins is ever read there,
/// and on a wide crossed shape they are megabytes of `Dual` elements. Sizing
/// them down is what keeps the packed engine's footprint the one its own
/// memory guard accounts for.
///
/// `for_shape` and [`bufs_match_shape`] both go through here, so the sized
/// lengths and the checked lengths cannot drift apart.
struct DenseTwinShape {
    k: usize,
    rows: usize,
    s: usize,
    q_p: usize,
    q_core: usize,
    e: usize,
    extras: bool,
    observed: bool,
}

impl DenseTwinShape {
    #[allow(clippy::too_many_arguments)]
    fn of(
        packed_width: usize,
        k: usize,
        rows: usize,
        s: usize,
        q_p: usize,
        q_core: usize,
        e: usize,
        extras: bool,
        observed: bool,
        // Built for the row-blocked assembled Hessian: the per-row and
        // per-cluster twins live in its block scratch, so they sit at their
        // minimum here; `k` (the shared mode `u`) stays.
        row_blocked: bool,
    ) -> DenseTwinShape {
        if packed_width > 0 {
            // `q_p`/`q_core` stay 1: `PirlsScratch::for_shape` divides no
            // length by them but `lam` is `q_p²`, and 1 is the smallest width
            // that keeps every `.max(1)` minimum honest.
            DenseTwinShape {
                k: 0,
                rows: 0,
                s: 0,
                q_p: 1,
                q_core: 1,
                e: 0,
                extras: false,
                observed: false,
            }
        } else {
            DenseTwinShape {
                k,
                rows: if row_blocked { 0 } else { rows },
                s: if row_blocked { 0 } else { s },
                q_p,
                q_core,
                e,
                extras,
                observed,
            }
        }
    }
}

impl GlmmDualScratch {
    /// Allocate the dual scratch for one model shape at one lane count.
    /// `observed` says whether this fit takes the observed-information step;
    /// the `DualStep` twins are sized through [`obs_len`] off it, so a
    /// canonical link allocates none of them. Row
    /// buffers (`eta`, `prob`, `w`, `eta_fixed`, `m_buf`) are `rows`-length on
    /// every route — `rows` is `n`, the global row count: the row passes and
    /// the AGQ kernels both index by global row (see the module doc on
    /// `GlmmDualBufs`'s row buffers). Infallible: there is no size ceiling
    /// here — memory is not a routing reason (the module doc's memory rules
    /// say what is sized and what is not). Only a Hessian request is
    /// refused above `MAX_DUAL_N`, and that refusal runs before this is
    /// called; a gradient request reaches this at any `m` and chunks on the
    /// top rung instead.
    ///
    /// Builds its own `ClusterRowIndex` from `cluster_ids` unconditionally
    /// (`ClusterRowIndex::build`), rather than reusing `ws.cluster_rows` when
    /// `fit_glmm` has already built it for this fit — that
    /// reuse would need `ClusterRowIndex: Clone`. `for_shape` only runs on the first
    /// derivative request for a shape or on a shape change (the reuse
    /// policy skips it on repeat calls), so this is a bounded, not a
    /// per-call, allocation — the zero-alloc-on-repeat gate is
    /// about repeat calls at the same shape, not about `for_shape` itself.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_shape(
        n: NLanes,
        m: usize,
        p: usize,
        k: usize,
        rows: usize,
        s: usize,
        q_p: usize,
        q_core: usize,
        e: usize,
        extras: bool,
        nagq: u8,
        observed: bool,
        // Packed-row width on `GlmmLayout::Packed`, 0 on every other layout.
        packed_width: usize,
        cluster_ids: &[u32],
        // The design's coupling pairs ([`dual_coupling_pairs`]) when known;
        // `None` sizes the coupling-shaped twins at the `coupling_len` bound.
        coup_pairs: Option<usize>,
    ) -> GlmmDualScratch {
        Self::build(
            n,
            m,
            p,
            k,
            rows,
            s,
            q_p,
            q_core,
            e,
            extras,
            nagq,
            observed,
            packed_width,
            cluster_ids,
            None,
            None,
            coup_pairs,
        )
    }

    /// The dual scratch of the row-blocked assembled Hessian on the blocked
    /// layout (no extra groupings): the twin keeps only its `k`-length mode
    /// state, `params`, `beta` and `grad_t`, and every per-row and
    /// per-cluster buffer lives block-sized in its `row_blocks`
    /// (`assembled::RowBlockBufs`), planned at `block_rows` rows per block.
    /// At 3M rows of a binomial fit with four random effects per group that
    /// is about 150 MB of SE scratch instead of 3.2 GB. A `Dual` rung only:
    /// the assembled Hessian runs on nothing else.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_shape_row_blocked(
        n: NLanes,
        m: usize,
        p: usize,
        k: usize,
        rows: usize,
        s: usize,
        q_p: usize,
        nagq: u8,
        observed: bool,
        cluster_ids: &[u32],
        block_rows: usize,
    ) -> GlmmDualScratch {
        Self::build(
            n,
            m,
            p,
            k,
            rows,
            s,
            q_p,
            q_p,
            0,
            false,
            nagq,
            observed,
            0,
            cluster_ids,
            Some(block_rows),
            None,
            None,
        )
    }

    /// The dual scratch of the streamed assembled Hessian on the structured
    /// layout: the same as [`Self::for_shape`] except that the dual PIRLS row
    /// buffers and `structured.m_core_buf` sit at their minimum and the
    /// per-row values live block-sized in `stream_rows`
    /// (`pirls::StreamRows`, `block_rows` rows). The crossed Schur couples
    /// every cluster, so unlike [`Self::for_shape_row_blocked`] the cluster-
    /// and pair-level buffers stay whole; only the per-row ones go. A `Dual`
    /// rung only.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_shape_streamed(
        n: NLanes,
        m: usize,
        p: usize,
        k: usize,
        rows: usize,
        s: usize,
        q_p: usize,
        q_core: usize,
        e: usize,
        nagq: u8,
        observed: bool,
        cluster_ids: &[u32],
        coup_pairs: Option<usize>,
        block_rows: usize,
    ) -> GlmmDualScratch {
        Self::build(
            n,
            m,
            p,
            k,
            rows,
            s,
            q_p,
            q_core,
            e,
            true,
            nagq,
            observed,
            0,
            cluster_ids,
            None,
            Some(block_rows),
            coup_pairs,
        )
    }

    /// [`Self::for_shape`]'s body, with the row-blocked flag
    /// (`Some(block_rows)`) that [`Self::for_shape_row_blocked`] sets.
    #[allow(clippy::too_many_arguments)]
    fn build(
        n: NLanes,
        m: usize,
        p: usize,
        k: usize,
        rows: usize,
        s: usize,
        q_p: usize,
        q_core: usize,
        e: usize,
        extras: bool,
        nagq: u8,
        observed: bool,
        packed_width: usize,
        cluster_ids: &[u32],
        row_block: Option<usize>,
        stream_block: Option<usize>,
        coup_pairs: Option<usize>,
    ) -> GlmmDualScratch {
        let row_blocked = row_block.is_some();
        // The streamed pass forms its per-row values block by block
        // (`GlmmDualBufs::stream_rows`), so the dual PIRLS row buffers and the
        // structured core rows sit at their minimum.
        let streamed = stream_block.is_some();
        let d = DenseTwinShape::of(
            packed_width,
            k,
            rows,
            s,
            q_p,
            q_core,
            e,
            extras,
            observed,
            row_blocked,
        );
        // The row-blocked pass assembles in its block scratch, so the shared
        // assembly buffers sit at their minimum.
        let (k_asm, rows_asm, s_asm) = if row_blocked { (0, 0, 0) } else { (k, rows, s) };
        let coup_len = dual_coupling_len(d.rows, d.s, d.q_core, d.e, coup_pairs);
        let idx = ClusterRowIndex::build(cluster_ids, s);
        macro_rules! build {
            ($T:ty, $variant:ident, $asm:expr, $rb:expr, $sr:expr) => {
                GlmmDualScratch {
                    lanes: DualLanes::$variant(GlmmDualBufs::<$T> {
                        params: vec![<$T as Scalar>::ZERO; m],
                        beta: vec![<$T as Scalar>::ZERO; p],
                        pirls: PirlsScratch::<$T>::for_shape(
                            if streamed { 0 } else { d.rows },
                            d.k,
                            d.q_p,
                            d.s,
                            nagq,
                        ),
                        structured: StructuredScratch::<$T>::for_shape_coup(
                            d.rows,
                            d.s,
                            d.q_core,
                            d.e,
                            d.extras && !streamed,
                            coup_len,
                        ),
                        dual: DualStep {
                            observed: false,
                            obs_blocks: vec![
                                <$T as Scalar>::ZERO;
                                obs_len(d.observed, d.s * d.q_p * d.q_p)
                            ],
                            obs_core_blocks: vec![
                                <$T as Scalar>::ZERO;
                                obs_len(
                                    d.observed,
                                    (d.q_core * d.q_core * d.s).max(1)
                                )
                            ],
                            obs_coupling: vec![<$T as Scalar>::ZERO; obs_len(d.observed, coup_len)],
                            obs_schur_blk: vec![
                                <$T as Scalar>::ZERO;
                                obs_len(d.observed, (d.e * d.e).max(1))
                            ],
                            obs_rhs: vec![<$T as Scalar>::ZERO; obs_len(d.observed, d.k)],
                            min_iters: 0,
                            exact: false,
                        },
                        grad_t: vec![<$T as Scalar>::ZERO; m],
                        asm: $asm,
                        row_blocks: $rb,
                        stream_rows: $sr,
                    }),
                    idx,
                    mode: GlmmModeBufs::for_shape(k),
                }
            };
        }
        // The assembled Hessian runs on the `Dual` rungs only, so the
        // `HyperDual` rungs carry no assembly scratch at all — see
        // `AssemblyBufs::empty`.
        macro_rules! sized_asm {
            ($T:ty) => {
                super::assembled::AssemblyBufs::<$T>::for_shape_coup(
                    m,
                    k_asm,
                    rows_asm,
                    s_asm,
                    q_core,
                    e,
                    observed,
                    packed_width,
                    (!row_blocked).then(|| dual_coupling_len(rows, s, q_core, e, coup_pairs)),
                )
            };
        }
        macro_rules! row_blocks {
            ($T:ty) => {
                row_block.map(|b| {
                    Box::new(super::assembled::RowBlockBufs::<$T>::for_shape(
                        &idx, s, b, m, p, q_p, nagq, observed,
                    ))
                })
            };
        }
        macro_rules! stream_rows {
            ($T:ty) => {
                stream_block.map(|b| Box::new(super::pirls::StreamRows::<$T>::new(b, q_core, q_p)))
            };
        }
        match n {
            NLanes::D4 => build!(
                Dual<4>,
                D4,
                sized_asm!(Dual<4>),
                row_blocks!(Dual<4>),
                stream_rows!(Dual<4>)
            ),
            NLanes::D5 => build!(
                Dual<5>,
                D5,
                sized_asm!(Dual<5>),
                row_blocks!(Dual<5>),
                stream_rows!(Dual<5>)
            ),
            NLanes::D6 => build!(
                Dual<6>,
                D6,
                sized_asm!(Dual<6>),
                row_blocks!(Dual<6>),
                stream_rows!(Dual<6>)
            ),
            NLanes::D8 => build!(
                Dual<8>,
                D8,
                sized_asm!(Dual<8>),
                row_blocks!(Dual<8>),
                stream_rows!(Dual<8>)
            ),
            NLanes::D12 => build!(
                Dual<12>,
                D12,
                sized_asm!(Dual<12>),
                row_blocks!(Dual<12>),
                stream_rows!(Dual<12>)
            ),
            NLanes::H4 => build!(
                HyperDual<4, 10>,
                H4,
                super::assembled::AssemblyBufs::empty(),
                None,
                None
            ),
            NLanes::H5 => build!(
                HyperDual<5, 15>,
                H5,
                super::assembled::AssemblyBufs::empty(),
                None,
                None
            ),
            NLanes::H6 => build!(
                HyperDual<6, 21>,
                H6,
                super::assembled::AssemblyBufs::empty(),
                None,
                None
            ),
            NLanes::H8 => build!(
                HyperDual<8, 36>,
                H8,
                super::assembled::AssemblyBufs::empty(),
                None,
                None
            ),
            NLanes::H12 => build!(
                HyperDual<12, 78>,
                H12,
                super::assembled::AssemblyBufs::empty(),
                None,
                None
            ),
        }
    }

    /// Shape half of the reuse-policy check (see [`Self::lanes`]) —
    /// [`bufs_match_shape`] against the stored buffers, whatever the variant.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn matches_shape(
        &self,
        m: usize,
        p: usize,
        k: usize,
        rows: usize,
        s: usize,
        q_p: usize,
        q_core: usize,
        e: usize,
        extras: bool,
        nagq: u8,
        observed: bool,
        packed_width: usize,
        coup_pairs: Option<usize>,
    ) -> bool {
        macro_rules! check {
            ($b:expr, $asm:expr) => {
                bufs_match_shape(
                    $b,
                    m,
                    p,
                    k,
                    rows,
                    s,
                    q_p,
                    q_core,
                    e,
                    extras,
                    nagq,
                    observed,
                    packed_width,
                    $asm,
                    false,
                    coup_pairs,
                    None,
                )
            };
        }
        match &self.lanes {
            DualLanes::D4(b) => check!(b, true),
            DualLanes::D5(b) => check!(b, true),
            DualLanes::D6(b) => check!(b, true),
            DualLanes::D8(b) => check!(b, true),
            DualLanes::D12(b) => check!(b, true),
            DualLanes::H4(b) => check!(b, false),
            DualLanes::H5(b) => check!(b, false),
            DualLanes::H6(b) => check!(b, false),
            DualLanes::H8(b) => check!(b, false),
            DualLanes::H12(b) => check!(b, false),
        }
    }

    /// [`Self::matches_shape`] for a scratch built by
    /// [`Self::for_shape_streamed`] (a `Dual` rung, structured layout).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn matches_shape_streamed(
        &self,
        m: usize,
        p: usize,
        k: usize,
        rows: usize,
        s: usize,
        q_p: usize,
        q_core: usize,
        e: usize,
        nagq: u8,
        observed: bool,
        coup_pairs: Option<usize>,
        block_rows: usize,
    ) -> bool {
        macro_rules! check {
            ($b:expr) => {
                bufs_match_shape(
                    $b,
                    m,
                    p,
                    k,
                    rows,
                    s,
                    q_p,
                    q_core,
                    e,
                    true,
                    nagq,
                    observed,
                    0,
                    true,
                    false,
                    coup_pairs,
                    Some(block_rows),
                )
            };
        }
        match &self.lanes {
            DualLanes::D4(b) => check!(b),
            DualLanes::D5(b) => check!(b),
            DualLanes::D6(b) => check!(b),
            DualLanes::D8(b) => check!(b),
            DualLanes::D12(b) => check!(b),
            DualLanes::H4(..)
            | DualLanes::H5(..)
            | DualLanes::H6(..)
            | DualLanes::H8(..)
            | DualLanes::H12(..) => false,
        }
    }

    /// [`Self::matches_shape`] for a scratch built by
    /// [`Self::for_shape_row_blocked`] (a `Dual` rung, blocked layout).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn matches_shape_row_blocked(
        &self,
        m: usize,
        p: usize,
        k: usize,
        rows: usize,
        s: usize,
        q_p: usize,
        nagq: u8,
        observed: bool,
    ) -> bool {
        macro_rules! check {
            ($b:expr) => {
                bufs_match_shape(
                    $b, m, p, k, rows, s, q_p, q_p, 0, false, nagq, observed, 0, true, true, None,
                    None,
                )
            };
        }
        match &self.lanes {
            DualLanes::D4(b) => check!(b),
            DualLanes::D5(b) => check!(b),
            DualLanes::D6(b) => check!(b),
            DualLanes::D8(b) => check!(b),
            DualLanes::D12(b) => check!(b),
            DualLanes::H4(..)
            | DualLanes::H5(..)
            | DualLanes::H6(..)
            | DualLanes::H8(..)
            | DualLanes::H12(..) => false,
        }
    }
}

/// Per-coordinate unit-derivative seeding and derivative-lane access for the
/// dual scalar types. Not part of `Scalar` itself — the `f64` kernel has no
/// concept of a derivative lane, and only the derivative entry points below
/// need this, so it lives here rather than widening the kernel's own trait.
/// Both `Dual<N>` and `HyperDual<N, H>` implement it, below.
pub(super) trait Seed: TailKernel {
    /// Instantiated first-derivative lane count of this type — `N`. The chunk
    /// width `run_gradient` seeds per pass.
    const LANES: usize;
    /// `v` with first-derivative lane `lane` set to 1 and every other lane
    /// zero. Coordinates are `[θ_0..θ_{n_theta-1} | β_0..β_{p-1}]`; a pass
    /// seeding coordinates `base..base + width` gives coordinate
    /// `base + lane` the lane `lane`.
    fn unit(v: f64, lane: usize) -> Self;
    /// First-derivative lanes: `d[j]` is `∂value/∂` the coordinate this pass
    /// seeded into lane `j`.
    fn dslice(&self) -> &[f64];
}

impl<const N: usize> Seed for Dual<N> {
    const LANES: usize = N;
    fn unit(v: f64, lane: usize) -> Self {
        let mut d = [0.0f64; N];
        d[lane] = 1.0;
        Dual { v, d }
    }
    fn dslice(&self) -> &[f64] {
        &self.d
    }
}

impl<const N: usize, const H: usize> Seed for HyperDual<N, H> {
    const LANES: usize = N;
    fn unit(v: f64, lane: usize) -> Self {
        let mut d = [0.0f64; N];
        d[lane] = 1.0;
        // A coordinate is linear in itself, so its own second derivative is
        // zero — the packed block starts (and, for an unused padding lane,
        // stays) all-zero.
        HyperDual { v, d, h: [0.0; H] }
    }
    fn dslice(&self) -> &[f64] {
        &self.d
    }
}

/// Packed second-derivative lane access, on top of `Seed`'s first-derivative
/// one. A separate trait rather than a `Seed` method: `Dual<N>` also
/// implements `Seed` and has no `h` field to back it.
trait SeedHessian: Seed {
    /// Packed lower triangle, `h[i*(i+1)/2 + j] = ∂²value/∂p_i∂p_j` for `i >= j`.
    fn hslice(&self) -> &[f64];
}

impl<const N: usize, const H: usize> SeedHessian for HyperDual<N, H> {
    fn hslice(&self) -> &[f64] {
        &self.h
    }
}

/// One evaluation of the objective the derivative passes differentiate, routed
/// as `deviance::laplace_deviance` routes it: AGQ (`agq_deviance` at
/// `q_p == 1`, `agq_deviance_vec` above) when `agq::agq_eligible` holds on a
/// design without extras, the structured Laplace kernel on an extras design,
/// the blocked one otherwise. `None` when a Laplace kernel's PIRLS did not
/// converge; AGQ has no `conv` flag, its failure is a non-finite value, which
/// the caller checks. `T = f64` is the mode solve (`dual: None`), a dual `T`
/// the lanes pass. `laplace_deviance` is the `f64` fit-path copy of this
/// router — change together.
///
/// `pattern` is the `f64` workspace's own pattern, shared rather than mirrored
/// at `T`, so its `structured_schur` holds the cached sparse LLT here. That is
/// harmless at a dual `T`: the cached factor is `f64`-only (faer's
/// `SparseColMat<usize, f64>`), and `TailKernel`'s default
/// `tail_factor`/`tail_downdate`/`tail_solve` bodies (the only ones
/// `Dual`/`HyperDual` implement) ignore their `ss` argument outright, so the
/// tail always takes the dense generic path. The `f64` mode solves take the
/// factor out around the call themselves.
#[allow(clippy::too_many_arguments)]
fn eval_objective<T: TailKernel>(
    data: &FitData,
    nb_theta: f64,
    params: &[T],
    beta: &mut [T],
    pirls: &mut PirlsScratch<T>,
    structured: &mut StructuredScratch<T>,
    pattern: &mut StructuredPattern,
    dual: Option<&mut DualStep<T>>,
    wx: &mut Mat<f64>,
    nagq: u8,
    tol: f64,
    idx: &ClusterRowIndex,
    counters: &mut crate::counters::EvalCounters,
) -> Option<T> {
    let FitData {
        family,
        groupings,
        layout: _,
        x,
        y,
        prior_w,
        weighted,
        cluster_ids,
        extra_ids,
        z_buf,
        offset,
        n,
        p: _,
    } = *data;
    let extras = !groupings.extra_offsets.is_empty();
    if agq_eligible(family, nagq, groupings.primary_q) && !extras {
        let agq = if groupings.primary_q == 1 {
            agq_deviance::<T>
        } else {
            agq_deviance_vec::<T>
        };
        return Some(agq(
            family,
            nb_theta,
            groupings,
            params,
            beta,
            pirls,
            z_buf,
            x,
            y,
            prior_w,
            weighted,
            cluster_ids,
            dual,
            wx,
            nagq,
            Some(tol),
            n,
            Some(idx),
            offset,
            counters,
        ));
    }
    let (o, _pen_dev, conv, _raw_finite) = if extras {
        structured_laplace_deviance::<T>(
            family,
            nb_theta,
            groupings,
            params,
            z_buf,
            extra_ids,
            cluster_ids,
            pirls,
            structured,
            pattern,
            x,
            y,
            prior_w,
            weighted,
            beta,
            BetaStep::Fixed,
            dual,
            wx,
            offset,
            Some(tol),
            n,
            counters,
        )
    } else {
        blocked_laplace_deviance::<T>(
            family,
            nb_theta,
            groupings,
            params,
            beta,
            pirls,
            z_buf,
            x,
            y,
            prior_w,
            weighted,
            cluster_ids,
            dual,
            wx,
            BetaStep::Fixed,
            offset,
            Some(tol),
            n,
            counters,
        )
    };
    conv.then_some(o)
}

/// θ/β-lane seeding, the (zero-lane) mode seeding, the dual
/// `blocked_laplace_deviance::<T>` (no extras) or
/// `structured_laplace_deviance::<T>` (extras) call — one exact-Hessian solve
/// of two steps (`pirls::DualStep`), re-entered until the returned lanes settle
/// only when the solve was not exact — and the gradient copy into the caller's
/// buffer. The whole seed-call-read body `laplace_gradient`'s per-`N` match
/// arms hand a typed buffer set to.
///
/// `m = n_theta + p` may exceed `T::LANES`: the body runs `⌈m / T::LANES⌉`
/// passes, each seeding its own chunk of coordinates and writing that chunk's
/// slice of `grad`. One pass when `m <= T::LANES`, which is every shape the
/// ladder covers exactly.
///
/// `u_mode` is the `f64` PIRLS mode `laplace_gradient` already converged on
/// (its lanes start at zero — see `Seed::unit`'s own doc comment); `ws_params` is
/// `ws.params[..n_theta + p]`, read only to build the unit-lane seeds.
///
/// No production caller: [`laplace_gradient`]'s body, which has none either.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn run_gradient<T: Seed>(
    bufs: &mut GlmmDualBufs<T>,
    data: &FitData,
    nb_theta: f64,
    // θ-independent structured index pattern — the `f64` workspace's own
    // `ws.pattern`, shared rather than mirrored at `T`. **The pattern is
    // shared; the values are not, and the two can disagree in width.** The
    // pin skip in `build_packed_m` and the pin mask in
    // `structured_laplace_deviance` are both `f64`-only, so at a θ̂ with a
    // pinned crossed grouping the pattern a DUAL call builds is WIDER than
    // the one an `f64` call builds — while `cross_theta` lives in
    // `GlmmDualBufs<T>` and stays dual-private. On return from a derivative
    // call `pattern.cross_col`/`pattern.n_cross` can therefore name slots
    // `bufs.structured.cross_theta` never filled.
    //
    // The rule that keeps that safe: **every `f64` reader of the
    // `cross_col`/`n_cross`/`cross_theta` triple must run an `f64` deviance
    // evaluation of its own first.** There is one such reader,
    // `se::structured_schur_fill`, and it is reached only after
    // `joint_hessian_cov`'s `fallback!()` central `fd_eval` or after the
    // pinned re-eval in `pin::post_pin` — both of which re-pack the triple at
    // γ̂. Breaking the rule does not panic: the reader would pick up a stale
    // `cross_theta` left by an earlier optimizer trial and return a plausible
    // wrong SE.
    pattern: &mut StructuredPattern,
    wx: &mut Mat<f64>,
    ws_params: &[f64],
    u_mode: &[f64],
    n_theta: usize,
    tol: f64,
    nagq: u8,
    cluster_rows: &ClusterRowIndex,
    grad: &mut [f64],
) -> DerivStatus {
    let FitData { family, p, .. } = *data;
    let m = n_theta + p;
    let k = u_mode.len();
    // One pass seeds at most `T::LANES` coordinates; coordinate `base + j`
    // takes lane `j` and every other coordinate is a plain value with zero
    // lanes. Forward-mode lanes are independent — `d[j]` of every operation
    // depends only on lane `j` and the value parts, and control flow branches
    // on `.value()` alone — so the concatenation of the passes' slices is bit
    // for bit what one `Dual<m>` pass would produce.
    let lanes = T::LANES;
    let n_chunks = m.div_ceil(lanes.max(1));
    // Canonical links: `A = MᵀWM + I` IS the exact `½h_uu` at the mode, so one
    // kernel call's lanes are exact. Non-canonical links (probit, cloglog,
    // Gamma-log, NB-log) only get a Fisher-weighted approximation to `h_uu`
    // from `A`, so the kernel is told to step with the observed-information
    // `A_obs` instead (`DualStep::observed`), which makes the lanes exact in
    // one step there too. Two steps per call either way: the first moves the
    // lanes, the second reads `dev`/`log|A|` at the moved `u`.
    //
    // Both the blocked and the structured-extras kernel take this step:
    // `pirls_solve_blocked_extras` packs its twin as the core-block + Schur
    // split the Fisher factor already uses, since a crossed-tail column
    // couples every cluster and there is no per-cluster block to solve alone.
    // What is left for the refinement loop below to do any work on, on either
    // path: a row on one of the kernel's clamps, which holds on canonical
    // links too, and a non-PD twin factor.
    //
    // `observed` is set whatever the route (`eval_objective`), so a
    // probit/cloglog AGQ model takes the observed step too.
    let canonical = crate::family::is_canonical(family);
    bufs.dual.observed = !canonical;
    // `MAX_DUAL_REFINEMENTS` counted AS THE TOTAL KERNEL-CALL CAP here
    // (`max_calls` calls total): the settle check only runs from the 2nd
    // call on, so this gives up to `MAX_DUAL_REFINEMENTS - 1` refinement
    // compares. `run_hessian` counts the same constant differently — see its
    // `max_reads` comment; mirrors this one, change together.
    let max_calls = MAX_DUAL_REFINEMENTS;
    // The objective value is the same in every pass — only the seeded lanes
    // move — so the last pass's value is the function's answer.
    let mut last_value = f64::NAN;
    for chunk in 0..n_chunks {
        // Not named `offset` — that is this function's own `Option<&[f64]>`
        // model-offset argument.
        let base = chunk * lanes;
        let width = lanes.min(m - base);
        // Every pass re-enters at the SAME `f64` mode with zero lanes: the
        // kernel moves `bufs.pirls.u` (value and lanes) in place, so without
        // this a later pass would start from the previous pass's moved `u`
        // and differentiate at a different point.
        #[allow(clippy::needless_range_loop)]
        for c in 0..k {
            bufs.pirls.u[c] = T::from_f64(u_mode[c]);
        }
        bufs.dual.min_iters = 0;
        // Throwaway: the dual evaluation is not a fit-path PIRLS solve, so
        // it must not reach `ws.counters` — mirrors `se.rs`'s `fd_eval`
        // discipline.
        let mut counters = crate::counters::EvalCounters::new();
        // Sized at the top rung, not at `m`: a chunk is never wider than
        // `T::LANES <= MAX_DUAL_N`, so this holds any pass's lanes.
        let mut prev_d = [0.0f64; MAX_DUAL_N];
        let mut have_prev = false;
        // `Some` once this pass's lanes are the answer; still `None` after the
        // refinement loop means the cap was hit without settling.
        let mut pass_value = None;
        for _ in 0..max_calls {
            #[allow(clippy::needless_range_loop)]
            for j in 0..m {
                bufs.params[j] = if j >= base && j < base + width {
                    T::unit(ws_params[j], j - base)
                } else {
                    T::from_f64(ws_params[j])
                };
            }
            for i in 0..p {
                bufs.beta[i] = bufs.params[n_theta + i];
            }
            let Some(obj) = eval_objective::<T>(
                data,
                nb_theta,
                &bufs.params[..m],
                &mut bufs.beta[..p],
                &mut bufs.pirls,
                &mut bufs.structured,
                pattern,
                Some(&mut bufs.dual),
                wx,
                nagq,
                tol,
                cluster_rows,
                &mut counters,
            ) else {
                return DerivStatus::NotConverged;
            };
            // AGQ has no `conv` flag — a bare `+∞` value IS the failure signal
            // (`agq.rs`: `agq_deviance`/`agq_deviance_vec` return
            // `T::from_f64(f64::INFINITY)` on internal PIRLS non-convergence).
            // Checked uniformly for both branches, so a blocked-path non-finite
            // `dev` (already excluded by its own `!conv` above in practice) is
            // still caught.
            if !obj.value().is_finite() {
                return DerivStatus::NotConverged;
            }
            let d = obj.dslice();
            // Every step of this call was an exact-Hessian step
            // (`DualStep::exact`), so its lanes are the answer. What leaves the
            // contraction below to do the work instead, on either path: a row
            // on one of the kernel's clamps, or a non-PD observed factor — a
            // block on the blocked path, the whole crossed-tail Schur on the
            // extras one. Either way the loop re-enters from the returned `u`
            // with its lanes.
            if bufs.dual.exact {
                grad[base..base + width].copy_from_slice(&d[..width]);
                pass_value = Some(obj.value());
                break;
            }
            if have_prev {
                let settled = (0..width)
                    .all(|j| (d[j] - prev_d[j]).abs() < DUAL_SETTLE_REL * (1.0 + d[j].abs()));
                if settled {
                    grad[base..base + width].copy_from_slice(&d[..width]);
                    pass_value = Some(obj.value());
                    break;
                }
            }
            prev_d[..width].copy_from_slice(&d[..width]);
            have_prev = true;
        }
        match pass_value {
            Some(v) => last_value = v,
            None => return DerivStatus::NotConverged, // refinement cap hit without settling
        }
    }
    DerivStatus::Ok(last_value)
}

/// Gradient of the joint Laplace deviance with respect to `ws.params = [θ |
/// β]`, at the parameters currently in `ws.params`. Solves PIRLS at `f64`
/// first (tightened tolerance, throwaway counters), then differentiates at
/// that converged mode via one dual kernel call entered at the mode with
/// zero-lane `u` (several when the solve is not exact, see
/// `MAX_DUAL_REFINEMENTS`; several more when `m` is above the top rung and the
/// gradient chunks — see `run_gradient`). Writes `m = ws.n_theta + p`
/// entries into `grad`. Restores `ws.pirls.u` to what it found; `eta`, `prob`, `w`,
/// `mu`, `beta_rhs` and the block factors are left at the internal solve's
/// values, not the caller's — correctness rests on the caller re-evaluating
/// at the pinned γ̂ afterwards (the pinned re-eval in `pin::post_pin`).
///
/// Which objective is differentiated is `eval_objective`'s routing, which
/// mirrors `laplace_deviance`'s own: extras present ⇒
/// `structured_laplace_deviance`; otherwise AGQ (`nagq > 1` and
/// `agq::agq_eligible`) ⇒ `agq_deviance`/`agq_deviance_vec`; every other shape
/// (including `nagq == 1`, which IS the Laplace objective) ⇒
/// `blocked_laplace_deviance`. The shapes with no exact derivative at all are
/// [`supports_shape`]'s business, not this routing's.
///
/// No production caller: this stays as the exact gradient the FD and
/// chunked-Hessian gates compare against.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn laplace_gradient(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    // Per-row extra-grouping level ids, the same slice `laplace_deviance`
    // takes — read only on the structured route; unread when
    // `groupings.extra_offsets` is empty.
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    grad: &mut [f64],
) -> DerivStatus {
    // At the fitted Gamma dispersion, differentiate the fixed-φ GLMM the fit
    // minimized (`GlmmWorkspace::at_fixed_dispersion`); inside, `gamma_phi` is 1
    // and this guard does not fire again.
    if ws.dispersion_scaled() {
        return ws.at_fixed_dispersion(n, |ws| {
            laplace_gradient(ws, x, y, cluster_ids, extra_ids, p, n, grad)
        });
    }
    // Routing gate: [`supports_shape`], the single owner of the question.
    // Checked before `m`, so a shape with no exact derivative never allocates
    // scratch. Shape is the whole question — a pinned crossed θ̂ is carried by
    // the `f64`-only pin skip (`workspace.rs`/`deviance.rs`), not refused here.
    if !supports_shape(ws.layout, &ws.groupings) {
        return DerivStatus::Unsupported;
    }
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let nl = NLanes::pick(m, false).expect("the gradient ladder covers every m");

    // Reuse policy: `for_shape` only runs on the first request for this
    // shape/order, or when the stored scratch's order or shape doesn't match
    // this call.
    let (k, s, q_p, q_core, e, nagq) = (
        ws.k,
        ws.groupings.n_primary,
        ws.groupings.primary_q,
        ws.groupings.primary_q + ws.groupings.nested_per_parent,
        ws.groupings.k_crossed(),
        ws.nagq,
    );
    // Sizes the `DualStep` twins as well as pinning them in the reuse check —
    // same condition `bufs.dual.observed` takes below.
    let observed = !crate::family::is_canonical(ws.family);
    let extras = !ws.groupings.extra_offsets.is_empty();
    let coup_pairs = dual_coupling_pairs(&ws.pattern, s);
    let need_build = ws.dual_scratch.as_deref().is_none_or(|sc| {
        sc.lanes() != nl
            // `supports_shape` above refused `GlmmLayout::Packed`, so the
            // packed twin is never sized on this path.
            || !sc.matches_shape(
                m, p, k, n, s, q_p, q_core, e, extras, nagq, observed, 0, coup_pairs,
            )
    });
    if need_build {
        ws.dual_scratch = Some(Box::new(GlmmDualScratch::for_shape(
            nl,
            m,
            p,
            k,
            n,
            s,
            q_p,
            q_core,
            e,
            extras,
            nagq,
            observed,
            // `supports_shape` above refused `GlmmLayout::Packed`, so the
            // packed twin is never sized on this path.
            0,
            cluster_ids,
            coup_pairs,
        )));
    }

    let family = ws.family;
    // Never the fit's own exit tolerance — see "Tolerance handling":
    // `ws.fd.pirls_tol_override` if the caller set one, `pirls_tol_fd(family)`
    // otherwise. The SAME value is passed to every dual kernel call below.
    let tol = ws
        .fd
        .pirls_tol_override
        .unwrap_or_else(|| super::pirls_tol_fd(family));

    let kk = k.max(1);
    let nb_theta = ws.nb_theta;
    let weighted = ws.weighted;
    // Fixed-mode β transient (mirrors `laplace_deviance`'s own copy step,
    // reproduced by hand here since the mode solve calls `eval_objective`
    // rather than the `laplace_deviance` router).
    ws.beta_rhs[..p].copy_from_slice(&ws.params[n_theta..m]);

    // ONE destructure covers both the f64 mode solve and the dual
    // evaluation(s) below — splitting it in two would need a second reborrow
    // of `*ws` that the borrow checker cannot prove disjoint from the first
    // (an already-live `ws.offset.as_deref()` borrow versus a fresh `&mut
    // *ws`), even though the two borrows never touch the same field.
    let GlmmWorkspace {
        groupings,
        layout,
        params: prm,
        beta_rhs,
        z_buf,
        prior_w,
        pirls,
        structured,
        pattern,
        wx,
        offset: offset_field,
        dual_scratch,
        ..
    } = ws;
    let offset = offset_field.as_deref();

    // Read-only design view `run_gradient` takes below — see [`FitData`].
    let data = FitData {
        family,
        groupings,
        layout: *layout,
        x,
        y,
        prior_w: &prior_w[..n],
        weighted,
        cluster_ids,
        extra_ids,
        z_buf,
        offset,
        n,
        p,
    };

    // --- f64 mode solve, at `tol`, THROWAWAY counters. The
    // `pirls_hist`-sum == `n_eval` invariant holds only if a mode solve run
    // for a derivative request never reaches `ws.counters` — mirrors
    // `se.rs`'s `fd_eval` discipline exactly. `u` is mutated in place by the
    // solve (it is the PIRLS `u` buffer); snapshotted here and restored
    // below so the workspace's f64 fit state comes back as found.
    //
    // `eval_objective` routes it as it routes `run_gradient`'s dual calls, so
    // the mode is solved on the objective the lanes differentiate. On AGQ the
    // converged `u` is the same Laplace mode (both kernels run the identical
    // `pirls_solve_blocked` call, `agq.rs`); only the failure signal differs,
    // a non-finite value with no `conv` flag. ---
    // `saved_u`/`u_mode` are the dual scratch's own `GlmmModeBufs` (sized once
    // in `for_shape`), not fresh `Vec`s — a per-call `to_vec()` here is
    // exactly what the zero-alloc gate (`dual_gradient_repeat_calls_allocate_nothing`)
    // caught; see `GlmmModeBufs`'s doc comment.
    let scratch = dual_scratch
        .as_deref_mut()
        .expect("just built or confirmed present above");
    scratch.mode.saved_u[..kk].copy_from_slice(&pirls.u[..kk]);
    let mut mode_counters = crate::counters::EvalCounters::new();
    // Keep the mode solve off the cached sparse factor, which only the
    // structured arm reads: a derivative request must never overwrite the
    // converged factors `se::structured_schur_fill` reuses, and the mode this
    // solve hands the dual kernel below must come from the same dense tail the
    // dual kernel runs (see `eval_objective`'s doc). `pattern.structured_schur`
    // is taken out and restored around this one f64 call so its CSR refresh
    // (`cross_col`/`n_cross`/`coup_cols`/`coup_ptr`/`coup_mask`) still lands
    // in the real, shared `pattern`.
    let saved_schur = pattern.structured_schur.take();
    let mode_ok = eval_objective::<f64>(
        &data,
        nb_theta,
        &prm[..m],
        beta_rhs,
        pirls,
        structured,
        pattern,
        None,
        wx,
        nagq,
        tol,
        &scratch.idx,
        &mut mode_counters,
    )
    .is_some_and(|dev| dev.is_finite());
    pattern.structured_schur = saved_schur;
    if !mode_ok {
        pirls.u[..kk].copy_from_slice(&scratch.mode.saved_u[..kk]);
        return DerivStatus::NotConverged;
    }
    scratch.mode.u_mode[..k].copy_from_slice(&pirls.u[..k]);
    pirls.u[..kk].copy_from_slice(&scratch.mode.saved_u[..kk]); // restore — leave ws.pirls.u as found

    // --- dual evaluation(s), entered at the mode ---
    let GlmmDualScratch { lanes, idx, mode } = scratch;
    // One arm body, five instantiations — the same local-macro shape
    // `for_shape`/`matches_shape` use, so the argument list is written once
    // and a new rung is one line.
    macro_rules! grad_arm {
        ($bufs:expr, $idx:expr, $mode:expr) => {
            run_gradient(
                $bufs,
                &data,
                nb_theta,
                pattern,
                wx,
                &prm[..m],
                &$mode.u_mode[..k],
                n_theta,
                tol,
                nagq,
                $idx,
                grad,
            )
        };
    }
    match lanes {
        DualLanes::D4(bufs) => grad_arm!(bufs, idx, mode),
        DualLanes::D5(bufs) => grad_arm!(bufs, idx, mode),
        DualLanes::D6(bufs) => grad_arm!(bufs, idx, mode),
        DualLanes::D8(bufs) => grad_arm!(bufs, idx, mode),
        DualLanes::D12(bufs) => grad_arm!(bufs, idx, mode),
        // Unreachable: `ws.dual_scratch` is written only from `pick(m,
        // false)`, so this slot only ever holds a `Dual` variant — a
        // `HyperDual` arm here can't happen. The arm exists because the match
        // is exhaustive over the one shared enum; fall back to `Unsupported`
        // rather than panic.
        DualLanes::H4(..)
        | DualLanes::H5(..)
        | DualLanes::H6(..)
        | DualLanes::H8(..)
        | DualLanes::H12(..) => DerivStatus::Unsupported,
    }
}

/// Unpack the packed lower triangle `h[i*(i+1)/2 + j]` (`i >= j`, `i, j <
/// m`) into both triangles of `hess`. The packing enumerates rows `i = 0, 1,
/// 2, …` with row `i` holding `i+1` entries, so for `m <= N` the entries with
/// `i < m` are exactly the first `m*(m+1)/2` slots of the full `N`-sized
/// packed array — `h[..m*(m+1)/2]` is what both this and the settle check in
/// `run_hessian` read, never the tail belonging to padding rows `i >= m`.
pub(crate) fn unpack_hessian(hess: &mut Mat<f64>, h: &[f64], m: usize) {
    for i in 0..m {
        for j in 0..=i {
            let v = h[i * (i + 1) / 2 + j];
            hess[(i, j)] = v;
            hess[(j, i)] = v;
        }
    }
}

/// θ/β-lane seeding, the (zero-lane) mode seeding, and the dual
/// `blocked_laplace_deviance::<T>` / `structured_laplace_deviance::<T>` call —
/// one exact-Hessian solve of THREE steps (see the comment at the call site
/// below), falling into the refinement loop only when the solve was inexact.
/// The whole seed-call-read body `laplace_hessian`'s per-`N` match arms hand
/// a typed buffer set to.
///
/// `u_mode` is the `f64` PIRLS mode `laplace_hessian` already converged on
/// (its lanes start at zero); `ws_params` is `ws.params[..n_theta + p]`, read
/// only to build the unit-lane seeds. Mirrors `run_gradient`'s structure
/// exactly, plus the step floor and the `h`-block settle check.
#[allow(clippy::too_many_arguments)]
fn run_hessian<T: SeedHessian>(
    bufs: &mut GlmmDualBufs<T>,
    data: &FitData,
    nb_theta: f64,
    // The `f64` workspace's shared structured pattern — see `run_gradient`'s
    // doc comment on the same parameter.
    pattern: &mut StructuredPattern,
    wx: &mut Mat<f64>,
    ws_params: &[f64],
    u_mode: &[f64],
    n_theta: usize,
    tol: f64,
    nagq: u8,
    cluster_rows: &ClusterRowIndex,
    grad: &mut [f64],
    hess: &mut Mat<f64>,
) -> DerivStatus {
    let FitData { family, p, .. } = *data;
    let m = n_theta + p;
    let hlen = m * (m + 1) / 2;
    let k = u_mode.len();
    #[allow(clippy::needless_range_loop)]
    for c in 0..k {
        bufs.pirls.u[c] = T::from_f64(u_mode[c]);
    }
    let canonical = crate::family::is_canonical(family);
    // Throwaway, same discipline as `run_gradient`.
    let mut counters = crate::counters::EvalCounters::new();

    macro_rules! seed_params {
        () => {{
            #[allow(clippy::needless_range_loop)]
            for j in 0..m {
                bufs.params[j] = T::unit(ws_params[j], j);
            }
            for i in 0..p {
                bufs.beta[i] = T::unit(ws_params[n_theta + i], n_theta + i);
            }
        }};
    }
    // Produces a bare `T`, the convergence/finiteness check baked in (so every
    // call site gets the same early `NotConverged` return `run_gradient`'s
    // loop applies) — mirrors `run_gradient`'s per-call branch exactly, one
    // level up since `run_hessian` calls this from two sites (the mandatory
    // call 1 and the read loop) instead of one.
    macro_rules! call_kernel {
        () => {{
            let Some(obj_val) = eval_objective::<T>(
                data,
                nb_theta,
                &bufs.params[..m],
                &mut bufs.beta[..p],
                &mut bufs.pirls,
                &mut bufs.structured,
                pattern,
                Some(&mut bufs.dual),
                wx,
                nagq,
                tol,
                cluster_rows,
                &mut counters,
            ) else {
                return DerivStatus::NotConverged;
            };
            // Same uniform finiteness check as `run_gradient` — see its
            // comment at the same spot for why AGQ has no separate `conv`.
            if !obj_val.value().is_finite() {
                return DerivStatus::NotConverged;
            }
            obj_val
        }};
    }

    // One solve of at least three steps, which on a cell where those steps are
    // exact-Hessian ones is the whole answer (two-pass Newton-map argument;
    // the `DualStep::exact` guard below is what says they were):
    // step 1 makes `u`'s first-order lanes exact, step 2 its
    // second-order lanes, and step 3 is the read — `dev` and `log|A|` are
    // evaluated at step 3's INPUT `u` (the step-2 output) and `‖u‖²` at its
    // output, all with exact lanes. Neither `∂dev/∂u` nor `∂logdet/∂u`
    // vanishes at the mode, so reading them one step earlier (at the
    // zero-second-order-lane input of step 2) would be wrong. The value part
    // sits at the mode throughout, so the mixed-deviance exit would fire
    // after step 2 without the floor.
    // Same routing `run_gradient` gives: both the blocked and the
    // structured-extras kernel take the observed step on a non-canonical
    // link, the latter through its core-block + Schur twin.
    bufs.dual.observed = !canonical;
    bufs.dual.min_iters = 3;
    seed_params!();
    let obj: T = call_kernel!();
    if bufs.dual.exact {
        grad[..m].copy_from_slice(&obj.dslice()[..m]);
        unpack_hessian(hess, &obj.hslice()[..hlen], m);
        return DerivStatus::Ok(obj.value());
    }

    // Fallback: the solve was not exact — a row on one of the kernel's clamps,
    // or a non-PD observed factor (a block on the blocked path, the whole
    // crossed-tail Schur on the extras one) whose step took a Fisher factor
    // instead — so the lanes only contracted toward the IFT answer, they did
    // not reach it. Re-enter from the returned `u` (lanes included) until the
    // objective's `d` and `h` settle, or until two successive calls were both
    // exact — second-order lanes are exact once the last two steps were exact
    // ones, and the read call's objective is evaluated at its input `u`, which
    // the previous call produced.
    //
    // `MAX_DUAL_REFINEMENTS` counted AS READ CALLS ONLY here (`max_reads`
    // calls, i.e. the second kernel call onward) — the first call above sits
    // OUTSIDE this count, so the actual total is `1 + MAX_DUAL_REFINEMENTS`
    // kernel calls: one more than `run_gradient`'s own use of this same
    // constant, which counts its loop as the total. Both give up to
    // `MAX_DUAL_REFINEMENTS - 1` refinement compares (the first read call has
    // no previous read to compare against) — mirrors `run_gradient`'s
    // `max_calls` comment, change together.
    bufs.dual.min_iters = 0;
    let mut prev_exact = false;
    let mut prev_d = [0.0f64; MAX_DUAL_N];
    let mut prev_h = [0.0f64; MAX_DUAL_H];
    let mut have_prev = false;
    let max_reads = MAX_DUAL_REFINEMENTS;
    for _ in 0..max_reads {
        seed_params!();
        let obj: T = call_kernel!();
        let d = obj.dslice();
        let h = obj.hslice();
        let exact = bufs.dual.exact;
        if prev_exact && exact {
            grad[..m].copy_from_slice(&d[..m]);
            unpack_hessian(hess, &h[..hlen], m);
            return DerivStatus::Ok(obj.value());
        }
        prev_exact = exact;
        if have_prev {
            // `DUAL_SETTLE_REL` on every gradient entry and every packed `h`
            // entry — a settled gradient does not imply a settled Hessian, so
            // both must be checked before either is trusted.
            let grad_settled =
                (0..m).all(|j| (d[j] - prev_d[j]).abs() < DUAL_SETTLE_REL * (1.0 + d[j].abs()));
            let hess_settled = (0..hlen)
                .all(|idx| (h[idx] - prev_h[idx]).abs() < DUAL_SETTLE_REL * (1.0 + h[idx].abs()));
            if grad_settled && hess_settled {
                grad[..m].copy_from_slice(&d[..m]);
                unpack_hessian(hess, &h[..hlen], m);
                return DerivStatus::Ok(obj.value());
            }
        }
        prev_d[..m].copy_from_slice(&d[..m]);
        prev_h[..hlen].copy_from_slice(&h[..hlen]);
        have_prev = true;
    }
    DerivStatus::NotConverged // refinement cap hit without settling
}

/// Gradient and exact Hessian of the same joint Laplace deviance
/// `laplace_gradient` differentiates, with respect to `ws.params = [θ | β]`.
/// Same contract as `laplace_gradient` (`f64` PIRLS solve first, at
/// `ws.fd.pirls_tol_override` or `pirls_tol_fd(family)`; only `ws.pirls.u` restored,
/// `eta`/`prob`/`w`/`mu`/`beta_rhs`/block factors left at the internal
/// solve's values) plus one structural difference: its single dual kernel call runs
/// ONE MORE PIRLS step than the gradient's before its objective is
/// trustworthy at second order — see `run_hessian`'s doc comment for why.
///
/// Writes `m = ws.n_theta + p` entries into `grad`, and both triangles of an
/// `m×m` `hess` (same shape as `ws.inference.hess_scratch`) from
/// the packed lower triangle `h[i*(i+1)/2 + j]`, `i >= j`.
///
/// `hess` is the **deviance** Hessian, not the information matrix: the
/// information matrix is `hess / 2` — the same factor of 2 `rx_cov_into`'s
/// doc states. Halving (or not) this output
/// wrongly is a 2× error in every downstream standard error.
///
/// AGQ (`ws.nagq > 1`): same routing as `laplace_gradient` — see its doc
/// comment.
#[allow(clippy::too_many_arguments)]
pub(crate) fn laplace_hessian(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    grad: &mut [f64],
    hess: &mut Mat<f64>,
) -> DerivStatus {
    // At the fitted Gamma dispersion, differentiate the fixed-φ GLMM the fit
    // minimized (`GlmmWorkspace::at_fixed_dispersion`); inside, `gamma_phi` is 1
    // and this guard does not fire again.
    if ws.dispersion_scaled() {
        return ws.at_fixed_dispersion(n, |ws| {
            laplace_hessian(ws, x, y, cluster_ids, extra_ids, p, n, grad, hess)
        });
    }
    // Routing gate: same as `laplace_gradient`.
    if !supports_shape(ws.layout, &ws.groupings) {
        return DerivStatus::Unsupported;
    }
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let Some(nl) = NLanes::pick(m, true) else {
        return DerivStatus::Unsupported;
    };

    // Reuse policy: same as `laplace_gradient` — a request at the same
    // `(order, N)` and shape reuses the stored scratch, a different one
    // reallocates once.
    let (k, s, q_p, q_core, e, nagq) = (
        ws.k,
        ws.groupings.n_primary,
        ws.groupings.primary_q,
        ws.groupings.primary_q + ws.groupings.nested_per_parent,
        ws.groupings.k_crossed(),
        ws.nagq,
    );
    // Same twin sizing/pinning `laplace_gradient` uses.
    let observed = !crate::family::is_canonical(ws.family);
    let extras = !ws.groupings.extra_offsets.is_empty();
    let coup_pairs = dual_coupling_pairs(&ws.pattern, s);
    let need_build = ws.hyper_scratch.as_deref().is_none_or(|sc| {
        sc.lanes() != nl
            // `supports_shape` above refused `GlmmLayout::Packed`, so the
            // packed twin is never sized on this path.
            || !sc.matches_shape(
                m, p, k, n, s, q_p, q_core, e, extras, nagq, observed, 0, coup_pairs,
            )
    });
    if need_build {
        ws.hyper_scratch = Some(Box::new(GlmmDualScratch::for_shape(
            nl,
            m,
            p,
            k,
            n,
            s,
            q_p,
            q_core,
            e,
            extras,
            nagq,
            observed,
            // `supports_shape` above refused `GlmmLayout::Packed`, so the
            // packed twin is never sized on this path.
            0,
            cluster_ids,
            coup_pairs,
        )));
    }

    let family = ws.family;
    // Never the fit's own exit tolerance — see `laplace_gradient`. The SAME
    // value is passed to every dual kernel call below.
    let tol = ws
        .fd
        .pirls_tol_override
        .unwrap_or_else(|| super::pirls_tol_fd(family));

    let kk = k.max(1);
    let nb_theta = ws.nb_theta;
    let weighted = ws.weighted;
    ws.beta_rhs[..p].copy_from_slice(&ws.params[n_theta..m]);

    // ONE destructure, same borrow-checker reason as `laplace_gradient`.
    let GlmmWorkspace {
        groupings,
        layout,
        params: prm,
        beta_rhs,
        z_buf,
        prior_w,
        pirls,
        structured,
        pattern,
        wx,
        offset: offset_field,
        hyper_scratch,
        ..
    } = ws;
    let offset = offset_field.as_deref();

    // Read-only design view `run_hessian` takes below — see [`FitData`].
    let data = FitData {
        family,
        groupings,
        layout: *layout,
        x,
        y,
        prior_w: &prior_w[..n],
        weighted,
        cluster_ids,
        extra_ids,
        z_buf,
        offset,
        n,
        p,
    };

    // --- f64 mode solve, at `tol`, THROWAWAY counters — identical to
    // `laplace_gradient`'s, AGQ routing included (see its comment there). ---
    // `saved_u`/`u_mode` are the dual scratch's own `GlmmModeBufs` — see
    // `laplace_gradient`'s comment at the same spot.
    let scratch = hyper_scratch
        .as_deref_mut()
        .expect("just built or confirmed present above");
    scratch.mode.saved_u[..kk].copy_from_slice(&pirls.u[..kk]);
    let mut mode_counters = crate::counters::EvalCounters::new();
    // Off the cached sparse factor — see `laplace_gradient`'s own mode solve
    // for why `pattern.structured_schur` is taken out and restored around
    // this one f64 call.
    let saved_schur = pattern.structured_schur.take();
    let mode_ok = eval_objective::<f64>(
        &data,
        nb_theta,
        &prm[..m],
        beta_rhs,
        pirls,
        structured,
        pattern,
        None,
        wx,
        nagq,
        tol,
        &scratch.idx,
        &mut mode_counters,
    )
    .is_some_and(|dev| dev.is_finite());
    pattern.structured_schur = saved_schur;
    if !mode_ok {
        pirls.u[..kk].copy_from_slice(&scratch.mode.saved_u[..kk]);
        return DerivStatus::NotConverged;
    }
    scratch.mode.u_mode[..k].copy_from_slice(&pirls.u[..k]);
    pirls.u[..kk].copy_from_slice(&scratch.mode.saved_u[..kk]); // restore — leave ws.pirls.u as found

    // --- dual evaluation(s), entered at the mode ---
    let GlmmDualScratch { lanes, idx, mode } = scratch;
    // One arm body, five instantiations — the same local-macro shape
    // `for_shape`/`matches_shape` use, so the argument list is written once
    // and a new rung is one line.
    macro_rules! hess_arm {
        ($bufs:expr, $idx:expr, $mode:expr) => {
            run_hessian(
                $bufs,
                &data,
                nb_theta,
                pattern,
                wx,
                &prm[..m],
                &$mode.u_mode[..k],
                n_theta,
                tol,
                nagq,
                $idx,
                grad,
                hess,
            )
        };
    }
    match lanes {
        DualLanes::H4(bufs) => hess_arm!(bufs, idx, mode),
        DualLanes::H5(bufs) => hess_arm!(bufs, idx, mode),
        DualLanes::H6(bufs) => hess_arm!(bufs, idx, mode),
        DualLanes::H8(bufs) => hess_arm!(bufs, idx, mode),
        DualLanes::H12(bufs) => hess_arm!(bufs, idx, mode),
        // Unreachable: `ws.hyper_scratch` is written only from `pick(m,
        // true)`, so this slot only ever holds a `HyperDual` variant — a
        // `Dual` arm here can't happen. The arm exists because the match is
        // exhaustive over the one shared enum; fall back to `Unsupported`
        // rather than panic.
        DualLanes::D4(..)
        | DualLanes::D5(..)
        | DualLanes::D6(..)
        | DualLanes::D8(..)
        | DualLanes::D12(..) => DerivStatus::Unsupported,
    }
}

#[cfg(test)]
#[path = "derivative_tests.rs"]
mod tests;
