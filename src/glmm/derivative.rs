//! GLMM dual-arithmetic scratch and the `N` dispatch.
//!
//! Sizes and owns the θ-dependent PIRLS/AGQ buffers at a non-`f64` scalar
//! (`Dual<N>` for a gradient, `HyperDual<N, H>` for a gradient + Hessian), so
//! `laplace_gradient`/`laplace_hessian` can differentiate the joint
//! Laplace deviance without touching the `f64` fit path's own buffers. `N` is
//! a compile-time lane count; the runtime model dimension `m = n_theta + p`
//! picks the smallest instantiated `N` that covers it via [`GlmmDualScratch`],
//! an enum over the six instantiated variants.
//!
//! ## When a derivative request falls back
//!
//! `laplace_gradient` and `laplace_hessian` return `DerivStatus::Unsupported`
//! — a routing answer, not an error — in exactly three cases. Each names the
//! caller's own fallback, unchanged from what runs today:
//!
//! - **(a) `ws.groupings.extra_offsets` non-empty** — every nested or crossed
//!   design. These route to `pirls_solve_blocked_extras`
//!   (`pirls/blocked_extras.rs:332`): W8's generic structured tail below the
//!   dense boundary, or W10's hand adjoint above it — neither is W2's. Checked
//!   first in both entry points' routing gate, before `m` is even computed.
//!   Caller fallback: the FD Hessian (`se::fd_hessian_cov`, `se.rs:244`) for
//!   W3's SE path, BOBYQA on the objective for W5's optimizer.
//! - **(b) `m = n_theta + p > MAX_DUAL_N` (12)** — for the gradient and the
//!   Hessian alike (`NLanes::pick`). Same two caller fallbacks as (a).
//! - **(c) the dense `pirls_solve` fallback route** (`deviance.rs`'s
//!   "Non-eligible extras (oversized core)" arm, `deviance.rs:427-429`) — is
//!   **unreachable from this module.** That arm only runs when
//!   `extra_offsets` is non-empty AND `groupings.structured_extras_eligible()`
//!   is false; (a)'s guard above already returns `Unsupported` on the
//!   `extra_offsets` check alone, before eligibility is ever evaluated, so no
//!   model reaches route (c) through `laplace_gradient`/`laplace_hessian`. No
//!   separate guard is added for it — (a)'s condition is implied by (c)'s, so
//!   the existing check covers it structurally. Named here so the fallback
//!   table stays complete against `deviance.rs`'s own routing, though no test
//!   constructs it: there is nothing (a) doesn't already prove.
//!
//! **Memory is never a fallback reason.** There is no byte budget (see
//! `MAX_DUAL_N`'s own doc comment) — bigger buffers are allowed wherever they
//! make the code faster or simpler; only compute shape, (a) and (b), routes
//! to `Unsupported`.

// No non-test caller until W3 wires the SE path or W5 the optimizer through
// this module's entry points — remove this allow then. The crate's blanket
// `not(feature = "loop_advanced")` allow (`lib.rs`) does not cover a
// `loop_advanced` build, where this whole module would otherwise warn as
// dead code (precedent: the scoped allows in `dual.rs`).
#![cfg_attr(feature = "loop_advanced", allow(dead_code))]

use super::agq::{agq_deviance, agq_deviance_vec, ClusterRowIndex};
use super::deviance::blocked_laplace_deviance;
use super::pirls::BetaStep;
use super::workspace::GlmmWorkspace;
use crate::dual::{Dual, HyperDual};
use crate::lmm::LmmGroupings;
use crate::scalar::Scalar;
use crate::spec::Family;
use faer::{Mat, MatRef};

/// Outcome of a derivative request. `Unsupported` is a routing answer, not an
/// error: the caller falls back to the finite-difference Hessian (`se.rs`) or
/// to BOBYQA on the objective, exactly as it does today.
pub(crate) enum DerivStatus {
    /// Objective value at the seeded parameters; the gradient (and Hessian, if
    /// requested) has been written to the caller's buffers.
    Ok(f64),
    /// PIRLS did not converge, or the objective is non-finite. On the AGQ route
    /// the non-finite objective IS the failure signal — `agq_deviance` returns a
    /// bare `+∞` and carries no `conv` flag. Buffers are untouched.
    NotConverged,
    /// Not on the blocked path, or `m` exceeds the largest instantiated `N`.
    Unsupported,
}

/// The θ-dependent buffers `blocked_laplace_deviance` and the AGQ kernels write,
/// at a non-`f64` scalar. Mirrors the `GlmmWorkspace` fields of the same names —
/// change together with the workspace's sizing block (`workspace.rs:495-571`).
pub(crate) struct GlmmDualBufs<T: Scalar> {
    params: Vec<T>,    // m (seeded per call)
    beta: Vec<T>,      // p
    lam: Vec<T>,       // q_p²
    m_buf: Vec<T>,     // rows · q_p
    eta: Vec<T>,       // rows
    prob: Vec<T>,      // rows
    w: Vec<T>,         // rows
    u: Vec<T>,         // k
    u_prev: Vec<T>,    // k.max(1)
    eta_fixed: Vec<T>, // rows
    a_blocks: Vec<T>,  // s · q_p²
    a_rhs: Vec<T>,     // k
    // Same shape rule as the f64 `ws.agq_scratch` (`workspace.rs:415-426`):
    // `2·s + nagq^q_p·(q_p+1)` at q_p ≥ 2, `4·s` at q_p == 1, `.max(1)`.
    // Read by `agq_deviance`/`agq_deviance_vec` when `ws.nagq > 1`;
    // allocated but untouched on every model shape that takes the blocked
    // path instead.
    agq_scratch: Vec<T>,
}

/// `f64` mode-transfer buffers for one dual-scratch variant, sized once in
/// `for_shape` alongside its `GlmmDualBufs<T>` and kept as a SEPARATE tuple
/// field of [`GlmmDualScratch`] rather than folded into `GlmmDualBufs<T>`
/// itself: `laplace_gradient`/`laplace_hessian` need to read `u_mode` while
/// `bufs` is borrowed `&mut` for the dual kernel call below, and matching on
/// `GlmmDualScratch`'s variant gives `bufs` and `mode` as disjoint bindings
/// the borrow checker accepts — folding `u_mode` into `GlmmDualBufs` would
/// make it alias the same struct `bufs` already borrows whole.
///
/// Replaces two per-call `Vec::to_vec()` allocations the zero-alloc gate
/// caught (`dual_gradient_repeat_calls_allocate_nothing`, `tests.rs`):
/// before this, `laplace_gradient`/`laplace_hessian` built a fresh `saved_u`
/// and `u_mode` `Vec` on every single call.
pub(crate) struct GlmmModeBufs {
    /// `f64` snapshot of `ws.u[..k.max(1)]` taken right before the mode solve
    /// mutates `ws.u` in place, and copied back after so the workspace's fit
    /// state comes back as found — same role the removed local `saved_u`
    /// played.
    saved_u: Vec<f64>,
    /// The converged PIRLS mode `ws.u[..k]`, copied out of `ws.u` once per
    /// call before the dual kernel(s) below read it as `run_gradient`'s /
    /// `run_hessian`'s `u_mode` argument — same role the removed local
    /// `u_mode` played.
    u_mode: Vec<f64>,
}

impl GlmmModeBufs {
    fn for_shape(k: usize) -> GlmmModeBufs {
        GlmmModeBufs {
            saved_u: vec![0.0; k.max(1)],
            u_mode: vec![0.0; k],
        }
    }
}

/// The θ-dependent buffers, at a non-`f64` scalar, over the instantiated
/// lane counts. Each variant also carries the `ClusterRowIndex` the AGQ
/// cluster-outer arm needs, built once when the variant is (re)allocated,
/// and the `GlmmModeBufs` the mode-transfer snapshot/restore uses.
///
/// `pub(crate)` rather than module-private: `GlmmWorkspace::dual_scratch`
/// (`workspace.rs`) names this type in its field, so it — and, transitively,
/// `GlmmDualBufs` in its variants — must be nameable from a sibling module
/// (`private_interfaces` requires a variant's field types be at least as
/// visible as the variant itself).
pub(crate) enum GlmmDualScratch {
    D4(GlmmDualBufs<Dual<4>>, ClusterRowIndex, GlmmModeBufs),
    D8(GlmmDualBufs<Dual<8>>, ClusterRowIndex, GlmmModeBufs),
    D12(GlmmDualBufs<Dual<12>>, ClusterRowIndex, GlmmModeBufs),
    H4(
        GlmmDualBufs<HyperDual<4, 10>>,
        ClusterRowIndex,
        GlmmModeBufs,
    ),
    H8(
        GlmmDualBufs<HyperDual<8, 36>>,
        ClusterRowIndex,
        GlmmModeBufs,
    ),
    H12(
        GlmmDualBufs<HyperDual<12, 78>>,
        ClusterRowIndex,
        GlmmModeBufs,
    ),
}

impl GlmmDualScratch {
    /// Which `NLanes` member this scratch was built for — half of the
    /// reuse-policy check every entry point runs before calling `for_shape`
    /// (the other half is [`Self::matches_shape`]): a request at the same
    /// `(order, N)` and shape reuses the stored scratch, a different one
    /// reallocates once (the zero-alloc-on-repeat gate is about
    /// repeat calls at the SAME shape; a shape change is not a repeat call).
    fn lanes(&self) -> NLanes {
        match self {
            GlmmDualScratch::D4(..) => NLanes::D4,
            GlmmDualScratch::D8(..) => NLanes::D8,
            GlmmDualScratch::D12(..) => NLanes::D12,
            GlmmDualScratch::H4(..) => NLanes::H4,
            GlmmDualScratch::H8(..) => NLanes::H8,
            GlmmDualScratch::H12(..) => NLanes::H12,
        }
    }

    /// The `ClusterRowIndex` this scratch's variant carries, regardless of
    /// order (`Dual` or `HyperDual`). The AGQ dual entry hands this
    /// to `agq_deviance`/`agq_deviance_vec` as `Some(idx)` EXPLICITLY, and the
    /// f64 mode solve reads the same index so it runs the identical
    /// cluster-outer arm before the dual kernel ever sees the mode — never
    /// `ws.cluster_rows`, which is populated only under the `parallel`
    /// feature (`mod.rs:376`) and would silently drop to the node-outer arm
    /// on a serial build.
    fn cluster_rows(&self) -> &ClusterRowIndex {
        match self {
            GlmmDualScratch::D4(_, idx, _)
            | GlmmDualScratch::D8(_, idx, _)
            | GlmmDualScratch::D12(_, idx, _)
            | GlmmDualScratch::H4(_, idx, _)
            | GlmmDualScratch::H8(_, idx, _)
            | GlmmDualScratch::H12(_, idx, _) => idx,
        }
    }

    /// The `GlmmModeBufs` this scratch's variant carries, regardless of
    /// order — the mode-transfer snapshot/restore in `laplace_gradient`/
    /// `laplace_hessian` runs before either function knows which order it
    /// resolved to (the type-specific match comes after), so it reaches the
    /// buffers through this accessor rather than the final match's `mode`
    /// binding.
    fn mode_bufs_mut(&mut self) -> &mut GlmmModeBufs {
        match self {
            GlmmDualScratch::D4(_, _, mode)
            | GlmmDualScratch::D8(_, _, mode)
            | GlmmDualScratch::D12(_, _, mode)
            | GlmmDualScratch::H4(_, _, mode)
            | GlmmDualScratch::H8(_, _, mode)
            | GlmmDualScratch::H12(_, _, mode) => mode,
        }
    }
}

/// Largest instantiated lane count. `m` above this is `DerivStatus::Unsupported`
/// — a compute cap, not a memory one. Mirrors the `GlmmDualScratch` variants —
/// change together, along with `lmm::kernel::LmmDualScratch`/
/// `LmmHyperScratch::for_groupings`, which hardcode the same `{4, 8, 12}`
/// lane set.
pub(crate) const MAX_DUAL_N: usize = 12;

/// Packed length of the `N = MAX_DUAL_N` second-derivative block
/// (`12*13/2`), sized to hold any instantiated `HyperDual<N, H>`'s `h` — the
/// settle-check and prev-call scratch in `run_hessian` are sized once at this
/// bound rather than per-`N`.
const MAX_DUAL_H: usize = MAX_DUAL_N * (MAX_DUAL_N + 1) / 2;

/// Cap on the dual re-entries a non-canonical link may take before the returned
/// derivatives stop moving. The Fisher weight makes `A` an approximation to
/// `h_uu` there, so the IFT lanes are reached by contraction rather than in one
/// step. Hitting the cap is `DerivStatus::NotConverged`, not a silently
/// truncated gradient.
///
/// Measured 2026-09-01 on the gradient FD gate (`dual_gradient_matches_central_fd_per_family_and_shape`,
/// 10 draws × `{int1, q2s}` per non-canonical family): observed settle counts
/// were Binomial-probit 5, Binomial-cloglog 6–7, Gamma-log 6–7,
/// NegativeBinomial-log 5, max 7 across every cell run. 8 keeps one call of
/// headroom above that measured max.
///
/// Also measured 2026-09-01 on the Hessian FD gate
/// (`dual_hessian_matches_central_fd_of_gradient_per_family_and_shape`), whose
/// settle check additionally covers every packed `h` entry: total kernel-call
/// counts (the mandatory 2 plus refinement) were Binomial-probit 5–6,
/// Binomial-cloglog 7, Gamma-log 6–7, NegativeBinomial-log 5–6, max 7 —
/// inside the same cap, so this constant is shared unchanged.
///
/// **Counted differently at its two use sites — read the comment at each
/// before changing either:** `run_gradient`'s `max_calls` treats this as the
/// TOTAL kernel-call cap for the non-canonical loop. `run_hessian`'s
/// `max_reads` treats it as the cap on READ calls only, counted after the
/// mandatory unread call 1 that sits outside the loop — so the Hessian's true
/// total is `1 + MAX_DUAL_REFINEMENTS` kernel calls, one more than the
/// gradient's own use of this same constant. Both give up to
/// `MAX_DUAL_REFINEMENTS - 1` refinement compares (the first read call has no
/// previous call to compare against).
pub(crate) const MAX_DUAL_REFINEMENTS: usize = 8;

/// Which instantiated `(order, N)` pair a derivative request resolves to.
/// [`GlmmDualScratch::for_shape`] picks the smallest member whose `N` covers
/// `m`, for the requested order — `Dual` for a gradient-only request,
/// `HyperDual` when a Hessian is also wanted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NLanes {
    D4,
    D8,
    D12,
    H4,
    H8,
    H12,
}

impl NLanes {
    /// Smallest instantiated lane count at or above `m`, for the given order.
    /// `hessian == true` selects `HyperDual` (second derivatives wanted);
    /// `false` selects `Dual` (gradient only). `None` iff `m > MAX_DUAL_N` —
    /// the same guard every entry point runs first, exposed here so a caller
    /// that only needs the routing decision (not the allocation) can make it.
    pub(crate) fn pick(m: usize, hessian: bool) -> Option<NLanes> {
        if m > MAX_DUAL_N {
            return None;
        }
        Some(match (hessian, m) {
            (false, 0..=4) => NLanes::D4,
            (false, 5..=8) => NLanes::D8,
            (false, _) => NLanes::D12,
            (true, 0..=4) => NLanes::H4,
            (true, 5..=8) => NLanes::H8,
            (true, _) => NLanes::H12,
        })
    }
}

/// Scratch length for `agq_scratch`, mirroring the f64 `ws.agq_scratch` field's
/// shape rule exactly (`workspace.rs:415-426`): the vector kernel (`q_p ≥ 2`)
/// needs `2·s` (center loglik | running log-sum) plus a per-eval product-grid
/// node table of `nagq^q_p · (q_p+1)`; the scalar kernel (`q_p == 1`) needs
/// `4·s` (center loglik | node u_cj | per-node loglik | running log-sum).
fn agq_len(s: usize, q_p: usize, nagq: u8) -> usize {
    if q_p >= 2 {
        let kq = (nagq as usize).pow(q_p as u32);
        (2 * s + kq * (q_p + 1)).max(1)
    } else {
        (4 * s).max(1)
    }
}

/// Shape terms of the AGQ routing gate: family/nagq/q_p. `deviance.rs`'s
/// `laplace_deviance` gate is this AND `groupings.extra_offsets.is_empty()`;
/// the derivative-path callers rely on their routing gate having already
/// returned `Unsupported` on extras, so they call this alone.
pub(super) fn agq_eligible(family: Family, nagq: u8, primary_q: usize) -> bool {
    nagq > 1
        && (1..=3).contains(&primary_q)
        && matches!(family, Family::Binomial { .. } | Family::Poisson { .. })
}

/// True iff this buffer set was sized for exactly this shape — every
/// shape-determining length `for_shape` chose is re-derived and compared
/// (checking `eta` covers `prob`/`w`/`eta_fixed`, allocated together at the
/// same `rows`; `u` pins `k` and with it `u_prev`, `a_rhs`, and the
/// `GlmmModeBufs`). Lengths only: the `ClusterRowIndex` built from
/// `cluster_ids` is not covered — same-shape data with different cluster
/// assignment is still the caller's responsibility.
#[allow(clippy::too_many_arguments)]
fn bufs_match_shape<T: Scalar>(
    b: &GlmmDualBufs<T>,
    m: usize,
    p: usize,
    k: usize,
    rows: usize,
    s: usize,
    q_p: usize,
    nagq: u8,
) -> bool {
    b.params.len() == m
        && b.beta.len() == p
        && b.lam.len() == q_p * q_p
        && b.m_buf.len() == rows * q_p
        && b.eta.len() == rows
        && b.u.len() == k
        && b.a_blocks.len() == s * q_p * q_p
        && b.agq_scratch.len() == agq_len(s, q_p, nagq)
}

impl GlmmDualScratch {
    /// Allocate the dual scratch for one model shape at one lane count. Row
    /// buffers (`eta`, `prob`, `w`, `eta_fixed`, `m_buf`) are `rows`-length on
    /// every route — `rows` is `n`, the global row count: the row passes and
    /// the AGQ kernels both index by global row (see the module doc on
    /// `GlmmDualBufs`'s row buffers). Infallible: there is no size ceiling
    /// here — memory is not a routing reason, only `m > MAX_DUAL_N` is, and
    /// that guard runs before this is called.
    ///
    /// Builds its own `ClusterRowIndex` from `cluster_ids` unconditionally
    /// (`ClusterRowIndex::build`), rather than reusing `ws.cluster_rows` when
    /// the `parallel` feature has already populated it for this fit — that
    /// reuse would need `ClusterRowIndex: Clone`, a change to `agq.rs`, which
    /// is out of scope here. `for_shape` only runs on the first
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
        nagq: u8,
        cluster_ids: &[u32],
    ) -> GlmmDualScratch {
        let idx = ClusterRowIndex::build(cluster_ids, s);
        macro_rules! build {
            ($T:ty, $variant:ident) => {
                GlmmDualScratch::$variant(
                    GlmmDualBufs::<$T> {
                        params: vec![<$T as Scalar>::ZERO; m],
                        beta: vec![<$T as Scalar>::ZERO; p],
                        lam: vec![<$T as Scalar>::ZERO; q_p * q_p],
                        m_buf: vec![<$T as Scalar>::ZERO; rows * q_p],
                        eta: vec![<$T as Scalar>::ZERO; rows],
                        prob: vec![<$T as Scalar>::ZERO; rows],
                        w: vec![<$T as Scalar>::ZERO; rows],
                        u: vec![<$T as Scalar>::ZERO; k],
                        u_prev: vec![<$T as Scalar>::ZERO; k.max(1)],
                        eta_fixed: vec![<$T as Scalar>::ZERO; rows],
                        a_blocks: vec![<$T as Scalar>::ZERO; s * q_p * q_p],
                        a_rhs: vec![<$T as Scalar>::ZERO; k],
                        agq_scratch: vec![<$T as Scalar>::ZERO; agq_len(s, q_p, nagq)],
                    },
                    idx,
                    GlmmModeBufs::for_shape(k),
                )
            };
        }
        match n {
            NLanes::D4 => build!(Dual<4>, D4),
            NLanes::D8 => build!(Dual<8>, D8),
            NLanes::D12 => build!(Dual<12>, D12),
            NLanes::H4 => build!(HyperDual<4, 10>, H4),
            NLanes::H8 => build!(HyperDual<8, 36>, H8),
            NLanes::H12 => build!(HyperDual<12, 78>, H12),
        }
    }

    /// Shape half of the reuse-policy check (see [`Self::lanes`]) —
    /// [`bufs_match_shape`] against the stored buffers, whatever the variant.
    #[allow(clippy::too_many_arguments)]
    fn matches_shape(
        &self,
        m: usize,
        p: usize,
        k: usize,
        rows: usize,
        s: usize,
        q_p: usize,
        nagq: u8,
    ) -> bool {
        match self {
            GlmmDualScratch::D4(b, ..) => bufs_match_shape(b, m, p, k, rows, s, q_p, nagq),
            GlmmDualScratch::D8(b, ..) => bufs_match_shape(b, m, p, k, rows, s, q_p, nagq),
            GlmmDualScratch::D12(b, ..) => bufs_match_shape(b, m, p, k, rows, s, q_p, nagq),
            GlmmDualScratch::H4(b, ..) => bufs_match_shape(b, m, p, k, rows, s, q_p, nagq),
            GlmmDualScratch::H8(b, ..) => bufs_match_shape(b, m, p, k, rows, s, q_p, nagq),
            GlmmDualScratch::H12(b, ..) => bufs_match_shape(b, m, p, k, rows, s, q_p, nagq),
        }
    }
}

/// Per-coordinate unit-derivative seeding and derivative-lane access for the
/// dual scalar types. Not part of `Scalar` itself — the `f64` kernel has no
/// concept of a derivative lane, and only the derivative entry points below
/// need this, so it lives here rather than widening the kernel's own trait.
/// Both `Dual<N>` and `HyperDual<N, H>` implement it, below.
trait Seed: Scalar {
    /// `v` with first-derivative lane `lane` set to 1 and every other lane
    /// zero — lane `j` for `j` in `0..n_theta` seeds `θ_j`; lane `n_theta +
    /// i` seeds `β_i`.
    fn unit(v: f64, lane: usize) -> Self;
    /// First-derivative lanes, `d[j] = ∂value/∂p_j`.
    fn dslice(&self) -> &[f64];
}

impl<const N: usize> Seed for Dual<N> {
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

/// θ/β-lane seeding, the (zero-lane) mode seeding, the dual
/// `blocked_laplace_deviance::<T>` call — repeated on a non-canonical link
/// until the returned lanes settle (canonical links get the exact `H_uu`;
/// non-canonical links need a refinement loop) — and the
/// settled-gradient copy into the caller's buffer. The whole seed-call-read
/// body `laplace_gradient`'s per-`N` match arms hand a typed buffer set to.
///
/// `u_mode` is the `f64` PIRLS mode `laplace_gradient` already converged on
/// (its lanes start at zero — see `Seed::unit`'s own doc comment); `ws_params` is
/// `ws.params[..n_theta + p]`, read only to build the unit-lane seeds.
#[allow(clippy::too_many_arguments)]
fn run_gradient<T: Seed>(
    bufs: &mut GlmmDualBufs<T>,
    family: Family,
    nb_theta: f64,
    groupings: &LmmGroupings,
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    cluster_ids: &[u32],
    z_buf: &[f64],
    offset: Option<&[f64]>,
    wx: &mut Mat<f64>,
    ws_params: &[f64],
    u_mode: &[f64],
    n_theta: usize,
    p: usize,
    n: usize,
    tol: f64,
    nagq: u8,
    cluster_rows: &ClusterRowIndex,
    grad: &mut [f64],
) -> DerivStatus {
    let m = n_theta + p;
    let k = u_mode.len();
    #[allow(clippy::needless_range_loop)]
    for c in 0..k {
        bufs.u[c] = T::from_f64(u_mode[c]);
    }
    // Canonical links: `A = MᵀWM + I` IS the exact `½h_uu` at the mode, so the
    // single kernel call's lanes are exact — no refinement loop. Non-canonical
    // links (probit, cloglog, Gamma-log, NB-log) only get a Fisher-weighted
    // approximation to `h_uu` from `A`, so the lanes are a step toward the IFT
    // answer rather than the answer, and re-entering from the returned `u`
    // (lanes included) contracts toward it.
    let canonical = crate::family::is_canonical(family);
    // AGQ routing: `extra_offsets.is_empty()` is already guaranteed by the
    // caller's own routing gate (`laplace_gradient` returns `Unsupported`
    // before this is ever called), so only the shape terms are checked. Any
    // Binomial link (not just the canonical logit) can satisfy this gate —
    // `canonical` above is computed independently and still governs the
    // refinement loop below, so a probit/cloglog AGQ model still gets it.
    let agq_eligible = agq_eligible(family, nagq, groupings.primary_q);
    let scalar_agq = groupings.primary_q == 1;
    // `MAX_DUAL_REFINEMENTS` counted AS THE TOTAL KERNEL-CALL CAP here
    // (`max_calls` calls total): the settle check only runs from the 2nd
    // call on, so this gives up to `MAX_DUAL_REFINEMENTS - 1` refinement
    // compares. `run_hessian` counts the same constant differently — see its
    // `max_reads` comment; mirrors this one, change together.
    let max_calls = if canonical { 1 } else { MAX_DUAL_REFINEMENTS };
    // Throwaway (W0): the dual evaluation is not a fit-path PIRLS solve, so it
    // must not reach `ws.counters` — mirrors `se.rs`'s `fd_eval` discipline.
    let mut counters = crate::counters::EvalCounters::new();
    let mut prev_d = [0.0f64; MAX_DUAL_N];
    let mut have_prev = false;
    for _ in 0..max_calls {
        #[allow(clippy::needless_range_loop)]
        for j in 0..m {
            bufs.params[j] = T::unit(ws_params[j], j);
        }
        for i in 0..p {
            bufs.beta[i] = T::unit(ws_params[n_theta + i], n_theta + i);
        }
        let obj: T = if agq_eligible {
            if scalar_agq {
                agq_deviance::<T>(
                    family,
                    nb_theta,
                    groupings,
                    &bufs.params[..m],
                    &mut bufs.beta[..p],
                    &mut bufs.lam,
                    z_buf,
                    &mut bufs.m_buf,
                    x,
                    y,
                    prior_w,
                    weighted,
                    cluster_ids,
                    &mut bufs.eta,
                    &mut bufs.prob,
                    &mut bufs.w,
                    &mut bufs.u,
                    &mut bufs.u_prev,
                    &mut bufs.eta_fixed,
                    &mut bufs.a_blocks,
                    &mut bufs.a_rhs,
                    wx,
                    &mut bufs.agq_scratch,
                    nagq,
                    Some(tol),
                    n,
                    Some(cluster_rows),
                    offset,
                    &mut counters,
                )
            } else {
                agq_deviance_vec::<T>(
                    family,
                    nb_theta,
                    groupings,
                    &bufs.params[..m],
                    &mut bufs.beta[..p],
                    &mut bufs.lam,
                    z_buf,
                    &mut bufs.m_buf,
                    x,
                    y,
                    prior_w,
                    weighted,
                    cluster_ids,
                    &mut bufs.eta,
                    &mut bufs.prob,
                    &mut bufs.w,
                    &mut bufs.u,
                    &mut bufs.u_prev,
                    &mut bufs.eta_fixed,
                    &mut bufs.a_blocks,
                    &mut bufs.a_rhs,
                    wx,
                    &mut bufs.agq_scratch,
                    nagq,
                    Some(tol),
                    n,
                    Some(cluster_rows),
                    offset,
                    &mut counters,
                )
            }
        } else {
            let (o, conv, _raw_finite) = blocked_laplace_deviance::<T>(
                family,
                nb_theta,
                groupings,
                &bufs.params[..m],
                &mut bufs.beta[..p],
                &mut bufs.lam,
                z_buf,
                &mut bufs.m_buf,
                x,
                y,
                prior_w,
                weighted,
                cluster_ids,
                &mut bufs.eta,
                &mut bufs.prob,
                &mut bufs.w,
                &mut bufs.u,
                &mut bufs.u_prev,
                &mut bufs.eta_fixed,
                &mut bufs.a_blocks,
                &mut bufs.a_rhs,
                wx,
                BetaStep::Fixed,
                offset,
                Some(tol),
                p,
                n,
                &mut counters,
            );
            if !conv {
                return DerivStatus::NotConverged;
            }
            o
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
        if canonical {
            grad[..m].copy_from_slice(&d[..m]);
            return DerivStatus::Ok(obj.value());
        }
        if have_prev {
            // Same band shape as the kernel's own mixed-deviance convergence
            // check (`pirls_solve_blocked`'s `tol * (1.0 + mixed.abs())`):
            // relative, with an absolute floor so a near-zero lane still
            // settles.
            let settled = (0..m).all(|j| (d[j] - prev_d[j]).abs() < 1e-10 * (1.0 + d[j].abs()));
            if settled {
                grad[..m].copy_from_slice(&d[..m]);
                return DerivStatus::Ok(obj.value());
            }
        }
        prev_d[..m].copy_from_slice(&d[..m]);
        have_prev = true;
    }
    DerivStatus::NotConverged // refinement cap hit without settling
}

/// Gradient of the joint Laplace deviance with respect to `ws.params = [θ |
/// β]`, at the parameters currently in `ws.params`. Solves PIRLS at `f64`
/// first (tightened tolerance, throwaway counters), then differentiates at
/// that converged mode via one
/// (canonical) or several (non-canonical, refinement loop) dual kernel calls
/// entered at the mode with zero-lane `u`. Writes `m = ws.n_theta + p`
/// entries into `grad`. Leaves the workspace's `f64` fit state as it found
/// it — same contract as `se::fd_hessian_cov`.
///
/// AGQ (`ws.nagq > 1`): routes to `agq_deviance`/`agq_deviance_vec` under the
/// same gate `laplace_deviance` uses (`deviance.rs`'s AGQ gate: `nagq > 1 &&
/// extra_offsets.is_empty() && (1..=3).contains(&primary_q) &&
/// Binomial|Poisson`) — mirrored, not called through, since the mode solve
/// and the dual kernel calls need their own typed buffers. Every other shape
/// (including `nagq == 1`, which IS the Laplace objective) takes the blocked
/// arm unchanged.
pub(crate) fn laplace_gradient(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    p: usize,
    n: usize,
    grad: &mut [f64],
) -> DerivStatus {
    // Routing gate: mirrors `deviance.rs`'s blocked-arm condition
    // (`groupings.extra_offsets.is_empty()`, `deviance.rs:312`) — change
    // together. Extras (crossed or nested) route to
    // `pirls_solve_blocked_extras`, which is W10's hand adjoint, not W2's.
    if !ws.groupings.extra_offsets.is_empty() {
        return DerivStatus::Unsupported;
    }
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let Some(nl) = NLanes::pick(m, false) else {
        return DerivStatus::Unsupported;
    };

    // Reuse policy: `for_shape` only runs on the first request for this
    // shape/order, or when the stored scratch's order or shape doesn't match
    // this call (e.g. a Hessian call left a `HyperDual` variant behind).
    let (k, s, q_p, nagq) = (
        ws.k,
        ws.groupings.n_primary,
        ws.groupings.primary_q,
        ws.nagq,
    );
    let need_build = ws
        .dual_scratch
        .as_deref()
        .is_none_or(|sc| sc.lanes() != nl || !sc.matches_shape(m, p, k, n, s, q_p, nagq));
    if need_build {
        ws.dual_scratch = Some(Box::new(GlmmDualScratch::for_shape(
            nl,
            m,
            p,
            k,
            n,
            s,
            q_p,
            nagq,
            cluster_ids,
        )));
    }

    let family = ws.family;
    // Never the fit's own exit tolerance — see "Tolerance handling":
    // `ws.pirls_tol_override` if the caller set one, `pirls_tol_fd(family)`
    // otherwise. The SAME value is passed to every dual kernel call below.
    let tol = ws
        .pirls_tol_override
        .unwrap_or_else(|| super::pirls_tol_fd(family));

    let kk = k.max(1);
    let nb_theta = ws.nb_theta;
    let weighted = ws.weighted;
    // Fixed-mode β transient (mirrors `laplace_deviance`'s own copy step,
    // reproduced by hand here since the mode solve calls
    // `blocked_laplace_deviance` directly rather than through the AGQ-gated
    // `laplace_deviance` router — see the AGQ note above).
    ws.beta_rhs[..p].copy_from_slice(&ws.params[n_theta..m]);

    // ONE destructure covers both the f64 mode solve and the dual
    // evaluation(s) below — splitting it in two would need a second reborrow
    // of `*ws` that the borrow checker cannot prove disjoint from the first
    // (an already-live `ws.offset.as_deref()` borrow versus a fresh `&mut
    // *ws`), even though the two borrows never touch the same field.
    let GlmmWorkspace {
        groupings,
        params: prm,
        beta_rhs,
        lam,
        z_buf,
        m_buf,
        prior_w,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        a_blocks,
        a_rhs,
        wx,
        agq_scratch,
        offset: offset_field,
        dual_scratch,
        ..
    } = ws;
    let offset = offset_field.as_deref();

    // AGQ routing: `extra_offsets.is_empty()` is already guaranteed by the
    // routing gate above, so only the shape terms are checked.
    let agq_eligible = agq_eligible(family, nagq, groupings.primary_q);

    // --- f64 mode solve, at `tol`, THROWAWAY counters. W0's
    // `pirls_hist`-sum == `n_eval` invariant holds only if a mode solve run
    // for a derivative request never reaches `ws.counters` — mirrors
    // `se.rs`'s `fd_eval` discipline exactly. `u` is mutated in place by the
    // solve (it is the PIRLS `u` buffer); snapshotted here and restored
    // below so the workspace's f64 fit state comes back as found.
    //
    // Must evaluate the SAME objective the dual kernel below differentiates:
    // at `agq_eligible` that is `agq_deviance`/`agq_deviance_vec`, not
    // `blocked_laplace_deviance`. Both kernels run the identical
    // `pirls_solve_blocked` call internally
    // (`agq.rs`), so the converged `u` IS the same Laplace mode either way —
    // only the convergence signal differs: AGQ has no `conv` flag, so a
    // non-finite return value alone is the failure signal (mirrors
    // `run_gradient`'s own per-call check). ---
    // `saved_u`/`u_mode` are the dual scratch's own `GlmmModeBufs` (sized once
    // in `for_shape`), not fresh `Vec`s — a per-call `to_vec()` here is
    // exactly what the zero-alloc gate (`dual_gradient_repeat_calls_allocate_nothing`)
    // caught; see `GlmmModeBufs`'s doc comment.
    dual_scratch
        .as_deref_mut()
        .expect("just built or confirmed present above")
        .mode_bufs_mut()
        .saved_u[..kk]
        .copy_from_slice(&u[..kk]);
    let mut mode_counters = crate::counters::EvalCounters::new();
    let mode_ok = if agq_eligible {
        let idx = dual_scratch
            .as_deref()
            .expect("just built or confirmed present above")
            .cluster_rows();
        let dev = if groupings.primary_q == 1 {
            agq_deviance::<f64>(
                family,
                nb_theta,
                groupings,
                &prm[..m],
                beta_rhs,
                lam,
                z_buf,
                m_buf,
                x,
                y,
                &prior_w[..n],
                weighted,
                cluster_ids,
                eta,
                prob,
                w,
                u,
                u_prev,
                eta_fixed,
                a_blocks,
                a_rhs,
                wx,
                agq_scratch,
                nagq,
                Some(tol),
                n,
                Some(idx),
                offset,
                &mut mode_counters,
            )
        } else {
            agq_deviance_vec::<f64>(
                family,
                nb_theta,
                groupings,
                &prm[..m],
                beta_rhs,
                lam,
                z_buf,
                m_buf,
                x,
                y,
                &prior_w[..n],
                weighted,
                cluster_ids,
                eta,
                prob,
                w,
                u,
                u_prev,
                eta_fixed,
                a_blocks,
                a_rhs,
                wx,
                agq_scratch,
                nagq,
                Some(tol),
                n,
                Some(idx),
                offset,
                &mut mode_counters,
            )
        };
        dev.is_finite()
    } else {
        let (dev, conv, _raw_finite) = blocked_laplace_deviance::<f64>(
            family,
            nb_theta,
            groupings,
            &prm[..m],
            beta_rhs,
            lam,
            z_buf,
            m_buf,
            x,
            y,
            &prior_w[..n],
            weighted,
            cluster_ids,
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            a_blocks,
            a_rhs,
            wx,
            BetaStep::Fixed,
            offset,
            Some(tol),
            p,
            n,
            &mut mode_counters,
        );
        conv && dev.is_finite()
    };
    if !mode_ok {
        u[..kk].copy_from_slice(
            &dual_scratch
                .as_deref_mut()
                .expect("just built or confirmed present above")
                .mode_bufs_mut()
                .saved_u[..kk],
        );
        return DerivStatus::NotConverged;
    }
    {
        let mode = dual_scratch
            .as_deref_mut()
            .expect("just built or confirmed present above")
            .mode_bufs_mut();
        mode.u_mode[..k].copy_from_slice(&u[..k]);
        u[..kk].copy_from_slice(&mode.saved_u[..kk]); // restore — leave ws.u as found
    }

    // --- dual evaluation(s), entered at the mode ---
    let scratch = dual_scratch
        .as_deref_mut()
        .expect("just built or confirmed present above");
    match scratch {
        GlmmDualScratch::D4(bufs, idx, mode) => run_gradient(
            bufs,
            family,
            nb_theta,
            groupings,
            x,
            y,
            &prior_w[..n],
            weighted,
            cluster_ids,
            z_buf,
            offset,
            wx,
            &prm[..m],
            &mode.u_mode[..k],
            n_theta,
            p,
            n,
            tol,
            nagq,
            idx,
            grad,
        ),
        GlmmDualScratch::D8(bufs, idx, mode) => run_gradient(
            bufs,
            family,
            nb_theta,
            groupings,
            x,
            y,
            &prior_w[..n],
            weighted,
            cluster_ids,
            z_buf,
            offset,
            wx,
            &prm[..m],
            &mode.u_mode[..k],
            n_theta,
            p,
            n,
            tol,
            nagq,
            idx,
            grad,
        ),
        GlmmDualScratch::D12(bufs, idx, mode) => run_gradient(
            bufs,
            family,
            nb_theta,
            groupings,
            x,
            y,
            &prior_w[..n],
            weighted,
            cluster_ids,
            z_buf,
            offset,
            wx,
            &prm[..m],
            &mode.u_mode[..k],
            n_theta,
            p,
            n,
            tol,
            nagq,
            idx,
            grad,
        ),
        // `nl` above is always a `Dual` member (`NLanes::pick(m, false)`), and
        // the reuse-policy guard rebuilds whenever the stored order doesn't
        // match, so a `HyperDual` variant here means a Hessian call raced this
        // one on the same workspace (not a supported usage — GlmmWorkspace is
        // `&mut`-borrowed, single-threaded, per-call). Unreachable in
        // practice; fall back rather than panic.
        GlmmDualScratch::H4(..) | GlmmDualScratch::H8(..) | GlmmDualScratch::H12(..) => {
            DerivStatus::Unsupported
        }
    }
}

/// Unpack the packed lower triangle `h[i*(i+1)/2 + j]` (`i >= j`, `i, j <
/// m`) into both triangles of `hess`. The packing enumerates rows `i = 0, 1,
/// 2, …` with row `i` holding `i+1` entries, so for `m <= N` the entries with
/// `i < m` are exactly the first `m*(m+1)/2` slots of the full `N`-sized
/// packed array — `h[..m*(m+1)/2]` is what both this and the settle check in
/// `run_hessian` read, never the tail belonging to padding rows `i >= m`.
/// Re-exported through `glmm`'s `mod.rs` so `lmm::kernel`'s REML Hessian
/// entry can share the packing convention instead of duplicating it.
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
/// `blocked_laplace_deviance::<T>` calls — reads the objective off the
/// SECOND kernel call, unconditionally (see the comment at the call site
/// below), then continues the non-canonical refinement loop from there. The
/// whole seed-call-read body `laplace_hessian`'s per-`N` match arms hand a
/// typed buffer set to.
///
/// `u_mode` is the `f64` PIRLS mode `laplace_hessian` already converged on
/// (its lanes start at zero); `ws_params` is `ws.params[..n_theta + p]`, read
/// only to build the unit-lane seeds. Mirrors `run_gradient`'s structure
/// exactly, plus the mandatory extra call and the `h`-block settle check.
#[allow(clippy::too_many_arguments)]
fn run_hessian<T: SeedHessian>(
    bufs: &mut GlmmDualBufs<T>,
    family: Family,
    nb_theta: f64,
    groupings: &LmmGroupings,
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    cluster_ids: &[u32],
    z_buf: &[f64],
    offset: Option<&[f64]>,
    wx: &mut Mat<f64>,
    ws_params: &[f64],
    u_mode: &[f64],
    n_theta: usize,
    p: usize,
    n: usize,
    tol: f64,
    nagq: u8,
    cluster_rows: &ClusterRowIndex,
    grad: &mut [f64],
    hess: &mut Mat<f64>,
) -> DerivStatus {
    let m = n_theta + p;
    let hlen = m * (m + 1) / 2;
    let k = u_mode.len();
    #[allow(clippy::needless_range_loop)]
    for c in 0..k {
        bufs.u[c] = T::from_f64(u_mode[c]);
    }
    let canonical = crate::family::is_canonical(family);
    // AGQ routing: `groupings.extra_offsets` is already guaranteed empty by
    // `laplace_hessian`'s routing gate, so only the shape terms are checked.
    let agq_eligible = agq_eligible(family, nagq, groupings.primary_q);
    let scalar_agq = groupings.primary_q == 1;
    // Throwaway (W0), same discipline as `run_gradient`.
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
            let obj_val: T = if agq_eligible {
                if scalar_agq {
                    agq_deviance::<T>(
                        family,
                        nb_theta,
                        groupings,
                        &bufs.params[..m],
                        &mut bufs.beta[..p],
                        &mut bufs.lam,
                        z_buf,
                        &mut bufs.m_buf,
                        x,
                        y,
                        prior_w,
                        weighted,
                        cluster_ids,
                        &mut bufs.eta,
                        &mut bufs.prob,
                        &mut bufs.w,
                        &mut bufs.u,
                        &mut bufs.u_prev,
                        &mut bufs.eta_fixed,
                        &mut bufs.a_blocks,
                        &mut bufs.a_rhs,
                        wx,
                        &mut bufs.agq_scratch,
                        nagq,
                        Some(tol),
                        n,
                        Some(cluster_rows),
                        offset,
                        &mut counters,
                    )
                } else {
                    agq_deviance_vec::<T>(
                        family,
                        nb_theta,
                        groupings,
                        &bufs.params[..m],
                        &mut bufs.beta[..p],
                        &mut bufs.lam,
                        z_buf,
                        &mut bufs.m_buf,
                        x,
                        y,
                        prior_w,
                        weighted,
                        cluster_ids,
                        &mut bufs.eta,
                        &mut bufs.prob,
                        &mut bufs.w,
                        &mut bufs.u,
                        &mut bufs.u_prev,
                        &mut bufs.eta_fixed,
                        &mut bufs.a_blocks,
                        &mut bufs.a_rhs,
                        wx,
                        &mut bufs.agq_scratch,
                        nagq,
                        Some(tol),
                        n,
                        Some(cluster_rows),
                        offset,
                        &mut counters,
                    )
                }
            } else {
                let (o, conv, _raw_finite) = blocked_laplace_deviance::<T>(
                    family,
                    nb_theta,
                    groupings,
                    &bufs.params[..m],
                    &mut bufs.beta[..p],
                    &mut bufs.lam,
                    z_buf,
                    &mut bufs.m_buf,
                    x,
                    y,
                    prior_w,
                    weighted,
                    cluster_ids,
                    &mut bufs.eta,
                    &mut bufs.prob,
                    &mut bufs.w,
                    &mut bufs.u,
                    &mut bufs.u_prev,
                    &mut bufs.eta_fixed,
                    &mut bufs.a_blocks,
                    &mut bufs.a_rhs,
                    wx,
                    BetaStep::Fixed,
                    offset,
                    Some(tol),
                    p,
                    n,
                    &mut counters,
                );
                if !conv {
                    return DerivStatus::NotConverged;
                }
                o
            };
            // Same uniform finiteness check as `run_gradient` — see its
            // comment at the same spot for why AGQ has no separate `conv`.
            if !obj_val.value().is_finite() {
                return DerivStatus::NotConverged;
            }
            obj_val
        }};
    }

    // Kernel call 1 — unconditional, canonical links take it too. `u` (and
    // `u_prev`) are `&mut` and mutated in place by the kernel, so this call's
    // OUTPUT `u` (correct first- AND second-order lanes, per the two-pass
    // Newton-map argument) is left sitting in `bufs.u`
    // for the next call to read as its input — nothing is copied by hand.
    // The returned objective itself is discarded: `dev` and `logdet` were
    // read at this call's INPUT `u` (the zero-second-order-lane mode), and
    // neither `∂dev/∂u` nor `∂logdet/∂u` vanishes at the mode, so the wrong
    // input `S` does not cancel out of them the way it does out of the
    // gradient.
    seed_params!();
    let _: T = call_kernel!(); // discarded — see the doc comment above

    let mut prev_d = [0.0f64; MAX_DUAL_N];
    let mut prev_h = [0.0f64; MAX_DUAL_H];
    let mut have_prev = false;
    // Kernel call 2 is the mandatory re-entry whose objective is read (loop
    // iteration 0 below); further iterations are the non-canonical
    // refinement loop, re-entering from each call's own output `u` the same
    // way call 2 re-entered from call 1's.
    //
    // `MAX_DUAL_REFINEMENTS` counted AS READ CALLS ONLY here (`max_reads`
    // calls, i.e. kernel call 2 onward) — the mandatory, unread call 1 above
    // sits OUTSIDE this count, so the actual total is
    // `1 + MAX_DUAL_REFINEMENTS` kernel calls: one more than
    // `run_gradient`'s own use of this same constant, which counts its loop
    // as the total. Both give up to `MAX_DUAL_REFINEMENTS - 1` refinement
    // compares (the first read call has no previous read to compare
    // against) — mirrors `run_gradient`'s `max_calls` comment, change
    // together.
    let max_reads = if canonical { 1 } else { MAX_DUAL_REFINEMENTS };
    for _ in 0..max_reads {
        seed_params!();
        let obj: T = call_kernel!();
        let d = obj.dslice();
        let h = obj.hslice();
        if canonical {
            grad[..m].copy_from_slice(&d[..m]);
            unpack_hessian(hess, &h[..hlen], m);
            return DerivStatus::Ok(obj.value());
        }
        if have_prev {
            // Same band shape as `run_gradient`'s settle check, extended to
            // cover every packed `h` entry — a settled gradient does not
            // imply a settled Hessian, so both must be checked before either
            // is trusted.
            let grad_settled =
                (0..m).all(|j| (d[j] - prev_d[j]).abs() < 1e-10 * (1.0 + d[j].abs()));
            let hess_settled =
                (0..hlen).all(|idx| (h[idx] - prev_h[idx]).abs() < 1e-10 * (1.0 + h[idx].abs()));
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
/// `ws.pirls_tol_override` or `pirls_tol_fd(family)`; workspace left as
/// found) plus one structural difference: this takes ONE MORE dual kernel
/// call than the gradient before its objective is trustworthy at second
/// order — see `run_hessian`'s doc comment for why.
///
/// Writes `m = ws.n_theta + p` entries into `grad`, and both triangles of an
/// `m×m` `hess` (same shape as `ws.hess_scratch`, `workspace.rs:268`) from
/// the packed lower triangle `h[i*(i+1)/2 + j]`, `i >= j`.
///
/// `hess` is the **deviance** Hessian, not the information matrix: the
/// information matrix is `hess / 2` — the same factor of 2 `rx_cov_into`
/// documents at `se.rs:121` ("NO factor of 2; that factor only applies to
/// the deviance Hessian, where info = H_dev/2"). Halving (or not) this output
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
    p: usize,
    n: usize,
    grad: &mut [f64],
    hess: &mut Mat<f64>,
) -> DerivStatus {
    // Routing gate: same as `laplace_gradient` — change together.
    if !ws.groupings.extra_offsets.is_empty() {
        return DerivStatus::Unsupported;
    }
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let Some(nl) = NLanes::pick(m, true) else {
        return DerivStatus::Unsupported;
    };

    // Reuse policy: same as `laplace_gradient` — a request at the same
    // `(order, N)` and shape reuses the stored scratch, a different one (e.g.
    // a gradient call left a `Dual` variant behind) reallocates once.
    let (k, s, q_p, nagq) = (
        ws.k,
        ws.groupings.n_primary,
        ws.groupings.primary_q,
        ws.nagq,
    );
    let need_build = ws
        .dual_scratch
        .as_deref()
        .is_none_or(|sc| sc.lanes() != nl || !sc.matches_shape(m, p, k, n, s, q_p, nagq));
    if need_build {
        ws.dual_scratch = Some(Box::new(GlmmDualScratch::for_shape(
            nl,
            m,
            p,
            k,
            n,
            s,
            q_p,
            nagq,
            cluster_ids,
        )));
    }

    let family = ws.family;
    // Never the fit's own exit tolerance — see `laplace_gradient`. The SAME
    // value is passed to every dual kernel call below.
    let tol = ws
        .pirls_tol_override
        .unwrap_or_else(|| super::pirls_tol_fd(family));

    let kk = k.max(1);
    let nb_theta = ws.nb_theta;
    let weighted = ws.weighted;
    ws.beta_rhs[..p].copy_from_slice(&ws.params[n_theta..m]);

    // ONE destructure, same borrow-checker reason as `laplace_gradient`.
    let GlmmWorkspace {
        groupings,
        params: prm,
        beta_rhs,
        lam,
        z_buf,
        m_buf,
        prior_w,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        a_blocks,
        a_rhs,
        wx,
        agq_scratch,
        offset: offset_field,
        dual_scratch,
        ..
    } = ws;
    let offset = offset_field.as_deref();

    // AGQ routing: same as `laplace_gradient` — extras already excluded by
    // the routing gate above.
    let agq_eligible = agq_eligible(family, nagq, groupings.primary_q);

    // --- f64 mode solve, at `tol`, THROWAWAY counters — identical to
    // `laplace_gradient`'s, AGQ routing included (see its comment there). ---
    // `saved_u`/`u_mode` are the dual scratch's own `GlmmModeBufs` — see
    // `laplace_gradient`'s comment at the same spot.
    dual_scratch
        .as_deref_mut()
        .expect("just built or confirmed present above")
        .mode_bufs_mut()
        .saved_u[..kk]
        .copy_from_slice(&u[..kk]);
    let mut mode_counters = crate::counters::EvalCounters::new();
    let mode_ok = if agq_eligible {
        let idx = dual_scratch
            .as_deref()
            .expect("just built or confirmed present above")
            .cluster_rows();
        let dev = if groupings.primary_q == 1 {
            agq_deviance::<f64>(
                family,
                nb_theta,
                groupings,
                &prm[..m],
                beta_rhs,
                lam,
                z_buf,
                m_buf,
                x,
                y,
                &prior_w[..n],
                weighted,
                cluster_ids,
                eta,
                prob,
                w,
                u,
                u_prev,
                eta_fixed,
                a_blocks,
                a_rhs,
                wx,
                agq_scratch,
                nagq,
                Some(tol),
                n,
                Some(idx),
                offset,
                &mut mode_counters,
            )
        } else {
            agq_deviance_vec::<f64>(
                family,
                nb_theta,
                groupings,
                &prm[..m],
                beta_rhs,
                lam,
                z_buf,
                m_buf,
                x,
                y,
                &prior_w[..n],
                weighted,
                cluster_ids,
                eta,
                prob,
                w,
                u,
                u_prev,
                eta_fixed,
                a_blocks,
                a_rhs,
                wx,
                agq_scratch,
                nagq,
                Some(tol),
                n,
                Some(idx),
                offset,
                &mut mode_counters,
            )
        };
        dev.is_finite()
    } else {
        let (dev, conv, _raw_finite) = blocked_laplace_deviance::<f64>(
            family,
            nb_theta,
            groupings,
            &prm[..m],
            beta_rhs,
            lam,
            z_buf,
            m_buf,
            x,
            y,
            &prior_w[..n],
            weighted,
            cluster_ids,
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            a_blocks,
            a_rhs,
            wx,
            BetaStep::Fixed,
            offset,
            Some(tol),
            p,
            n,
            &mut mode_counters,
        );
        conv && dev.is_finite()
    };
    if !mode_ok {
        u[..kk].copy_from_slice(
            &dual_scratch
                .as_deref_mut()
                .expect("just built or confirmed present above")
                .mode_bufs_mut()
                .saved_u[..kk],
        );
        return DerivStatus::NotConverged;
    }
    {
        let mode = dual_scratch
            .as_deref_mut()
            .expect("just built or confirmed present above")
            .mode_bufs_mut();
        mode.u_mode[..k].copy_from_slice(&u[..k]);
        u[..kk].copy_from_slice(&mode.saved_u[..kk]); // restore — leave ws.u as found
    }

    // --- dual evaluation(s), entered at the mode ---
    let scratch = dual_scratch
        .as_deref_mut()
        .expect("just built or confirmed present above");
    match scratch {
        GlmmDualScratch::H4(bufs, idx, mode) => run_hessian(
            bufs,
            family,
            nb_theta,
            groupings,
            x,
            y,
            &prior_w[..n],
            weighted,
            cluster_ids,
            z_buf,
            offset,
            wx,
            &prm[..m],
            &mode.u_mode[..k],
            n_theta,
            p,
            n,
            tol,
            nagq,
            idx,
            grad,
            hess,
        ),
        GlmmDualScratch::H8(bufs, idx, mode) => run_hessian(
            bufs,
            family,
            nb_theta,
            groupings,
            x,
            y,
            &prior_w[..n],
            weighted,
            cluster_ids,
            z_buf,
            offset,
            wx,
            &prm[..m],
            &mode.u_mode[..k],
            n_theta,
            p,
            n,
            tol,
            nagq,
            idx,
            grad,
            hess,
        ),
        GlmmDualScratch::H12(bufs, idx, mode) => run_hessian(
            bufs,
            family,
            nb_theta,
            groupings,
            x,
            y,
            &prior_w[..n],
            weighted,
            cluster_ids,
            z_buf,
            offset,
            wx,
            &prm[..m],
            &mode.u_mode[..k],
            n_theta,
            p,
            n,
            tol,
            nagq,
            idx,
            grad,
            hess,
        ),
        // Mirrors `laplace_gradient`'s own unreachable-in-practice arm: `nl`
        // above is always a `HyperDual` member (`NLanes::pick(m, true)`), and
        // the reuse-policy guard rebuilds whenever the stored order doesn't
        // match. A `Dual` variant here would mean a gradient call raced this
        // one on the same `&mut` workspace — not supported usage. Fall back
        // rather than panic.
        GlmmDualScratch::D4(..) | GlmmDualScratch::D8(..) | GlmmDualScratch::D12(..) => {
            DerivStatus::Unsupported
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `for_shape` at `q_p == 1` (the scalar-intercept shape, `agq_scratch`'s
    /// `4·s` arm): every `GlmmDualBufs` field's `len()` must equal the same
    /// expression `for_shape` used to allocate it.
    #[test]
    fn for_shape_buffer_lengths_match_at_q_p_1() {
        let (m, p, k, rows, s, q_p, nagq) = (5usize, 3usize, 7usize, 40usize, 7usize, 1usize, 7u8);
        let cluster_ids: Vec<u32> = (0..rows as u32).map(|i| i % s as u32).collect();
        let scratch =
            GlmmDualScratch::for_shape(NLanes::D8, m, p, k, rows, s, q_p, nagq, &cluster_ids);
        match scratch {
            GlmmDualScratch::D8(bufs, _idx, mode) => {
                assert_eq!(bufs.params.len(), m);
                assert_eq!(bufs.beta.len(), p);
                assert_eq!(bufs.lam.len(), q_p * q_p);
                assert_eq!(bufs.m_buf.len(), rows * q_p);
                assert_eq!(bufs.eta.len(), rows);
                assert_eq!(bufs.prob.len(), rows);
                assert_eq!(bufs.w.len(), rows);
                assert_eq!(bufs.u.len(), k);
                assert_eq!(bufs.u_prev.len(), k.max(1));
                assert_eq!(bufs.eta_fixed.len(), rows);
                assert_eq!(bufs.a_blocks.len(), s * q_p * q_p);
                assert_eq!(bufs.a_rhs.len(), k);
                assert_eq!(bufs.agq_scratch.len(), agq_len(s, q_p, nagq));
                assert_eq!(bufs.agq_scratch.len(), 4 * s);
                assert_eq!(mode.saved_u.len(), k.max(1));
                assert_eq!(mode.u_mode.len(), k);
            }
            _ => panic!("expected D8 variant"),
        }
    }

    /// Same check at `q_p == 2` (the vector-RE shape, `agq_scratch`'s
    /// `2·s + nagq^q_p·(q_p+1)` arm), on the `HyperDual` order.
    #[test]
    fn for_shape_buffer_lengths_match_at_q_p_2() {
        let (m, p, k, rows, s, q_p, nagq) = (6usize, 2usize, 18usize, 50usize, 9usize, 2usize, 5u8);
        let cluster_ids: Vec<u32> = (0..rows as u32).map(|i| i % s as u32).collect();
        let scratch =
            GlmmDualScratch::for_shape(NLanes::H8, m, p, k, rows, s, q_p, nagq, &cluster_ids);
        match scratch {
            GlmmDualScratch::H8(bufs, _idx, mode) => {
                assert_eq!(bufs.params.len(), m);
                assert_eq!(bufs.beta.len(), p);
                assert_eq!(bufs.lam.len(), q_p * q_p);
                assert_eq!(bufs.m_buf.len(), rows * q_p);
                assert_eq!(bufs.eta.len(), rows);
                assert_eq!(bufs.prob.len(), rows);
                assert_eq!(bufs.w.len(), rows);
                assert_eq!(bufs.u.len(), k);
                assert_eq!(bufs.u_prev.len(), k.max(1));
                assert_eq!(bufs.eta_fixed.len(), rows);
                assert_eq!(bufs.a_blocks.len(), s * q_p * q_p);
                assert_eq!(bufs.a_rhs.len(), k);
                let kq = (nagq as usize).pow(q_p as u32);
                assert_eq!(bufs.agq_scratch.len(), agq_len(s, q_p, nagq));
                assert_eq!(bufs.agq_scratch.len(), 2 * s + kq * (q_p + 1));
                assert_eq!(mode.saved_u.len(), k.max(1));
                assert_eq!(mode.u_mode.len(), k);
            }
            _ => panic!("expected H8 variant"),
        }
    }

    /// `m = ws.n_theta + p > MAX_DUAL_N` returns `Unsupported` through
    /// `laplace_gradient`'s own guard without panicking — no scratch is built
    /// and `ws.dual_scratch` stays untouched.
    #[test]
    fn laplace_gradient_m_above_cap_is_unsupported() {
        assert!(NLanes::pick(MAX_DUAL_N + 1, false).is_none());

        // A single-intercept binomial workspace (m = n_theta(1) + p) — `p` is
        // padded to push `m` past `MAX_DUAL_N` (12) without needing a real
        // 12-column design; the guard fires before `x`/`y` are ever read.
        let mut model = crate::test_support::intercept_only_spec(crate::Sizing::FixedClusters {
            n_clusters: 3,
        });
        model.family = crate::Family::Binomial {
            link: crate::BinomialLink::Logit,
        };
        let n = 3;
        let p = MAX_DUAL_N; // m = 1 + 12 = 13 > MAX_DUAL_N
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
        let cluster_ids: [u32; 3] = [0, 1, 2];
        let x = faer::Mat::<f64>::zeros(n, p);
        let y = vec![0.0f64; n];
        let mut grad = vec![0.0f64; p + 1];
        let status = laplace_gradient(&mut ws, x.as_ref(), &y, &cluster_ids, p, n, &mut grad);
        assert!(matches!(status, DerivStatus::Unsupported));
        assert!(ws.dual_scratch.is_none());
    }

    /// A model with a non-empty `extra_offsets` (crossed or nested — both
    /// route to `pirls_solve_blocked_extras`) returns `Unsupported` through
    /// `laplace_gradient`'s routing gate without panicking. Mirrors
    /// `dense_glmm_entry_rejects_extra_slopes`'s (`glmm/tests.rs`) hand-built
    /// crossed-grouping spec.
    #[test]
    fn laplace_gradient_extras_are_unsupported() {
        let model = crate::ModelSpec {
            family: crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            re: Some(crate::ReStructure {
                sizing: crate::Sizing::FixedClusters { n_clusters: 2 },
                slopes: vec![],
                extra_groupings: vec![crate::Grouping {
                    relation: crate::GroupingRelation::Crossed { n_clusters: 2 },
                    slopes: vec![],
                }],
            }),
        };
        let n = 8;
        let p = 1;
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
        assert!(
            !ws.groupings.extra_offsets.is_empty(),
            "fixture must carry an extra grouping"
        );
        let x = faer::Mat::<f64>::zeros(n, p);
        let y = vec![0.0f64, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let cluster_ids = vec![0u32, 0, 1, 1, 0, 0, 1, 1];
        let mut grad = vec![0.0f64; ws.n_theta + p];
        let status = laplace_gradient(&mut ws, x.as_ref(), &y, &cluster_ids, p, n, &mut grad);
        assert!(matches!(status, DerivStatus::Unsupported));
    }

    /// `laplace_hessian`'s own cap guard — mirrors
    /// `laplace_gradient_m_above_cap_is_unsupported`.
    #[test]
    fn laplace_hessian_m_above_cap_is_unsupported() {
        assert!(NLanes::pick(MAX_DUAL_N + 1, true).is_none());

        let mut model = crate::test_support::intercept_only_spec(crate::Sizing::FixedClusters {
            n_clusters: 3,
        });
        model.family = crate::Family::Binomial {
            link: crate::BinomialLink::Logit,
        };
        let n = 3;
        let p = MAX_DUAL_N; // m = 1 + 12 = 13 > MAX_DUAL_N
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
        let cluster_ids: [u32; 3] = [0, 1, 2];
        let x = faer::Mat::<f64>::zeros(n, p);
        let y = vec![0.0f64; n];
        let mut grad = vec![0.0f64; p + 1];
        let mut hess = Mat::<f64>::zeros(p + 1, p + 1);
        let status = laplace_hessian(
            &mut ws,
            x.as_ref(),
            &y,
            &cluster_ids,
            p,
            n,
            &mut grad,
            &mut hess,
        );
        assert!(matches!(status, DerivStatus::Unsupported));
        assert!(ws.dual_scratch.is_none());
    }

    /// `laplace_hessian`'s own extras-routing guard — mirrors
    /// `laplace_gradient_extras_are_unsupported`.
    #[test]
    fn laplace_hessian_extras_are_unsupported() {
        let model = crate::ModelSpec {
            family: crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            re: Some(crate::ReStructure {
                sizing: crate::Sizing::FixedClusters { n_clusters: 2 },
                slopes: vec![],
                extra_groupings: vec![crate::Grouping {
                    relation: crate::GroupingRelation::Crossed { n_clusters: 2 },
                    slopes: vec![],
                }],
            }),
        };
        let n = 8;
        let p = 1;
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
        assert!(
            !ws.groupings.extra_offsets.is_empty(),
            "fixture must carry an extra grouping"
        );
        let x = faer::Mat::<f64>::zeros(n, p);
        let y = vec![0.0f64, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let cluster_ids = vec![0u32, 0, 1, 1, 0, 0, 1, 1];
        let m = ws.n_theta + p;
        let mut grad = vec![0.0f64; m];
        let mut hess = Mat::<f64>::zeros(m, m);
        let status = laplace_hessian(
            &mut ws,
            x.as_ref(),
            &y,
            &cluster_ids,
            p,
            n,
            &mut grad,
            &mut hess,
        );
        assert!(matches!(status, DerivStatus::Unsupported));
    }
}
