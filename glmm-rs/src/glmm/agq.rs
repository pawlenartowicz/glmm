//! Adaptive Gauss–Hermite quadrature (`nAGQ>1`) for the single scalar-intercept
//! binomial / Poisson / negative-binomial / Gamma GLMM — the only shape where the marginal likelihood is a
//! product of independent 1-D cluster integrals, so AGQ is a per-cluster k-node
//! sum with no curse of dimensionality. Replaces the Laplace `+log|A|` curvature
//! term with a Liu–Pierce (1994) adaptive GH sum centered at each cluster's PIRLS
//! mode/curvature. The sibling kernel [`agq_deviance_vec`] extends this one
//! dimension up — a **vector** RE (`q_p ∈ 2..=3`) on a single grouping factor,
//! integrated over a `k^q` product grid.
//!
//! `nAGQ=1` is the Laplace term exactly (the k=1 node sits at the mode with GH
//! weight √π), so the production path routes `nagq==1` to `laplace_deviance`
//! verbatim; only `nagq>1` reaches here — see the gate in
//! `deviance::laplace_deviance`. The white-box k=1≡Laplace reduction is asserted
//! to `rel<1e-12` in `glmm::tests` (different op order, not bit-equality).

use faer::MatRef;

use super::pirls::{pirls_solve_blocked, BetaStep, DualStep};
use super::workspace::PirlsScratch;
use crate::lmm::LmmGroupings;
use crate::scalar::Scalar;
use crate::spec::Family;

/// Per-cluster row index (CSR): `ptr[c]..ptr[c+1]` slices `rows` down to cluster `c`'s
/// row indices, in ASCENDING original-row order. That ordering is load-bearing, not
/// cosmetic: the cluster-outer loop in [`agq_deviance`] visits each cluster's rows
/// in the same order the node-outer loop does (row `0..n` ascending), so the two
/// agree bit for bit — same operands, same summation order. `build` gets this
/// for free (see below), so no separate sort step is required, unlike the crossed-
/// extras coupling CSR this mirrors (`pirls::build_coupling_csr`) — that one needs a
/// sort+dedup because a row can repeat across a cluster's crossed columns; here a row
/// belongs to exactly one cluster, so no duplicates are possible.
///
/// Wired into [`agq_deviance`]'s `cluster_rows` parameter: `fit_glmm` builds it
/// in a `parallel` build when `FitOptions::parallel_inner` is set, and in a
/// serial build for vector AGQ (`q_p ≥ 2`, no extra groupings).
/// Built ONCE per fit in `fit_glmm` (`cluster_ids` doesn't change across BOBYQA
/// evals), not once per eval.
#[derive(Default)]
pub(crate) struct ClusterRowIndex {
    /// Length `s+1`.
    ptr: Vec<u32>,
    /// Length `n` — row indices grouped by cluster, ascending within each cluster.
    rows: Vec<u32>,
}

impl ClusterRowIndex {
    /// `cluster_ids[i]` = row `i`'s cluster (`0..s`); `s` = cluster count
    /// (`groupings.n_primary`). Counting-sort CSR: counts → prefix → fill. Rows are
    /// visited `0..n` ascending to fill `rows`, so each cluster's slice comes out
    /// ascending automatically — no sort step needed (see the struct doc comment).
    pub(crate) fn build(cluster_ids: &[u32], s: usize) -> Self {
        let n = cluster_ids.len();
        let mut ptr = vec![0u32; s + 1];
        for &c in cluster_ids {
            ptr[c as usize + 1] += 1;
        }
        for c in 0..s {
            ptr[c + 1] += ptr[c];
        }
        let mut cursor = ptr.clone();
        let mut rows = vec![0u32; n];
        for (i, &c) in cluster_ids.iter().enumerate() {
            rows[cursor[c as usize] as usize] = i as u32;
            cursor[c as usize] += 1;
        }
        ClusterRowIndex { ptr, rows }
    }

    /// [`ClusterRowIndex::build`] into this index's own buffers, the first time
    /// it is called on them: a workspace's ids never change, so an index built
    /// for this `n` and `s` is the one the ids would give. The same counting
    /// sort, so the same `ptr`/`rows`.
    pub(crate) fn ensure(&mut self, cluster_ids: &[u32], s: usize) {
        let n = cluster_ids.len();
        if self.rows.len() == n && self.ptr.len() == s + 1 {
            return;
        }
        self.ptr.clear();
        self.ptr.resize(s + 1, 0);
        for &c in cluster_ids {
            self.ptr[c as usize + 1] += 1;
        }
        for c in 0..s {
            self.ptr[c + 1] += self.ptr[c];
        }
        self.rows.clear();
        self.rows.resize(n, 0);
        // The per-cluster write cursor runs in `ptr` itself (`ptr[c]` ends at
        // the start of cluster `c + 1`) and is shifted back one slot after the
        // pass, so no second buffer.
        for (i, &c) in cluster_ids.iter().enumerate() {
            let cur = &mut self.ptr[c as usize];
            self.rows[*cur as usize] = i as u32;
            *cur += 1;
        }
        for c in (1..=s).rev() {
            self.ptr[c] = self.ptr[c - 1];
        }
        self.ptr[0] = 0;
    }

    /// Cluster `c`'s row indices, ascending original-row order.
    pub(crate) fn cluster_rows(&self, c: usize) -> &[u32] {
        &self.rows[self.ptr[c] as usize..self.ptr[c + 1] as usize]
    }

    /// Cluster `c`'s positions in [`ClusterRowIndex::rows`], the range
    /// [`ClusterRowIndex::cluster_rows`] slices.
    pub(crate) fn cluster_range(&self, c: usize) -> std::ops::Range<usize> {
        self.ptr[c] as usize..self.ptr[c + 1] as usize
    }

    /// Every row index, grouped by cluster (the CSR column array).
    pub(crate) fn rows(&self) -> &[u32] {
        &self.rows
    }
}

/// The families adaptive quadrature takes: binomial, Poisson, negative
/// binomial and Gamma, any link. Why these qualify is written at
/// `fit::common::check_model_shape`.
pub(crate) fn agq_family(family: Family) -> bool {
    matches!(
        family,
        Family::Binomial { .. }
            | Family::Poisson { .. }
            | Family::NegativeBinomial { .. }
            | Family::Gamma { .. }
    )
}

/// Shape terms of the AGQ routing gate: family/nagq/q_p. The full gate is this
/// AND `groupings.extra_offsets.is_empty()` — an extras design takes the
/// structured Laplace arm whatever `nagq` says. Every caller
/// (`deviance::laplace_deviance`, `derivative::eval_objective`, `assembled.rs`
/// and `pin.rs`) spells out that second half itself, so this stays the shape
/// half alone.
pub(super) fn agq_eligible(family: Family, nagq: u8, primary_q: usize) -> bool {
    nagq > 1 && (1..=crate::consts::MAX_AGQ_Q).contains(&primary_q) && agq_family(family)
}

/// Scratch length for `agq_scratch`, mirroring the f64 `ws.pirls.agq_scratch` field's
/// shape rule exactly: the vector kernel (`q_p ≥ 2`) needs `2·s` (data part of
/// `ℓ_c(ũ_c)` | running node sum `Σ_j W_j·expm1(x_cj)`) plus a per-eval
/// product-grid node table of `nagq^q_p · (q_p+1)`; the scalar kernel
/// (`q_p == 1`) needs `7·s` (data part of `ℓ_c(ũ_c)` | node shift `h_cj` | node
/// data sum | running node sum `Σ_j w_j·expm1(x_cj)` | the compressed route's
/// two sums and max |η|). `nagq=1` shapes never reach the vector kernel, so
/// their `k^q=1` table is a harmless `q+1` slots. Sizes the splits in
/// [`agq_deviance`] and [`agq_deviance_vec`] — change together.
pub(super) fn agq_len(s: usize, q_p: usize, nagq: u8) -> usize {
    if q_p >= 2 {
        let kq = (nagq as usize).pow(q_p as u32);
        (2 * s + kq * (q_p + 1)).max(1)
    } else {
        (7 * s).max(1)
    }
}

/// Largest `|η|` (center plus node shift) the compressed AGQ route accepts. It
/// sits below the Gamma-log tail switch (`family::LOG_TAIL_ETA`, 230) and the
/// log-link clamp (`family::ETA_MAX`, 700), so no row of a compressed cluster is
/// clamped or on a tail form at any node, and a margin of 30 covers the round-off
/// between `η_c + δ` and the row kernel's own `η_fix + λ·u_cj`. At `e^200` the
/// aggregated sums stay finite for any realistic weight total.
const COMPRESS_ETA_MAX: f64 = 200.0;

/// The families whose scalar-intercept AGQ node sums compress to two per-cluster
/// sums: the log-link Poisson and Gamma (`μ = e^η`, `y·ln μ` and `y/μ` linear in
/// `e^{±δ}`). Every other family keeps the row kernel.
pub(super) fn compressible(family: Family) -> bool {
    matches!(
        family,
        Family::Poisson {
            link: crate::spec::PoissonLink::Log
        } | Family::Gamma {
            link: crate::spec::GammaLink::Log
        }
    )
}

/// Whether a cluster may take the compressed route: its largest center `|η|`
/// plus the largest node shift `|λ|·√2·σ_c·max|z_j|` stays at or below
/// [`COMPRESS_ETA_MAX`] (so no node reaches a clamp or a tail form), and both
/// sums are finite. False on any NaN, so a broken cluster falls back to the row
/// kernel and fails there as before.
pub(super) fn compress_ok(eta_reach: f64, shift_reach: f64, sum_a: f64, sum_b: f64) -> bool {
    eta_reach + shift_reach <= COMPRESS_ETA_MAX && sum_a.is_finite() && sum_b.is_finite()
}

/// `Σ_j w_j·expm1(x_j)`, `x_j = z_j² + ℓ_c(u_cj) − ℓ_c(ũ_c)`, for one cluster of a
/// scalar-intercept log-link Poisson or Gamma GLMM, from two per-cluster sums
/// instead of a pass over the rows per node. With `h_j = √2·σ_c·z_j = u_cj − ũ_c`,
/// `δ_j = λ·h_j`, the row weights `a_i` (`prior_w`, already `/φ` on Gamma) and
/// `μ0_i = e^{η_i}` at the mode, the data part of `x_j` (`−½·a_i·d_i` summed) is
/// ```text
///   Poisson:  Y·δ − M·expm1(δ),        Y = Σ a_i y_i,  M = Σ a_i μ0_i
///   Gamma:   −R·expm1(−δ) − W·δ,       W = Σ a_i,      R = Σ a_i y_i/μ0_i
/// ```
/// (unit deviances `2[y ln(y/μ) − (y−μ)]` and `2[−ln(y/μ) + (y−μ)/μ]` with
/// `μ = μ0·e^δ`; the `y ln y`, `ln y` and saturated terms cancel in the
/// difference), and the prior part with the Liu–Pierce `z_j²` is
/// `z_j² − ½(u_cj² − ũ_c²) = −h_j·(ũ_c − ½·(A_c − 1)·h_j)`, since
/// `z_j² = ½·A_c·h_j²` (`excess_c = A_c − 1`, kept apart from `A_c`).
/// `sum_a` is `Y` or `W`, `sum_b` is `M` or `R`. Exact for the unclamped,
/// off-tail likelihood; [`compress_ok`] keeps every cluster it is used on there.
/// This is the generic (dual) route; `f64` evaluations take the same identity
/// inside `agq_fast::scalar_sums`.
#[allow(clippy::too_many_arguments)]
fn compressed_cluster_sum<T: Scalar>(
    gamma: bool,
    sum_a: T,
    sum_b: T,
    u_c: T,
    sigma_c: T,
    excess_c: T,
    lambda: T,
    nodes: &[f64],
    wts: &[f64],
) -> T {
    let mut acc_sum = T::ZERO;
    for (&zj, &wj) in nodes.iter().zip(wts) {
        let h = sigma_c.scale(std::f64::consts::SQRT_2).scale(zj);
        let delta = lambda * h;
        let data = if gamma {
            -(sum_b * (-delta).exp_m1()) - sum_a * delta
        } else {
            sum_a * delta - sum_b * delta.exp_m1()
        };
        let prior = -(h * (u_c - excess_c.scale(0.5) * h));
        acc_sum += (prior + data).exp_m1().scale(wj);
    }
    acc_sum
}

/// The generic (dual) route's compression gather for Poisson-log and Gamma-log
/// (`agq_deviance`): fills each cluster's two sums for [`compressed_cluster_sum`]
/// (`csum_a`, `csum_b`) and its reach `eta_reach`, the largest `|η|` at the mode
/// over its rows, or ∞ when one of its rows fails the `f64` route's row guard
/// (`agq_fast::fast_consts` at the value parts, with the node-shift bound
/// `shift_reach(c)`), so [`compress_ok`] refuses that cluster. One slot per
/// cluster in each output; `cluster_ids` has one entry per row. Mirrors the
/// `f64` gate in `agq_fast::scalar_sums` (`nslow[c] == 0 && compress_ok(..)`) —
/// change together.
#[allow(clippy::too_many_arguments)]
fn compression_gather<T: Scalar>(
    family: Family,
    nb_theta: f64,
    y: &[f64],
    prior_w: &[f64],
    cluster_ids: &[u32],
    eta: &[T],
    prob: &[T],
    shift_reach: &impl Fn(usize) -> f64,
    csum_a: &mut [T],
    csum_b: &mut [T],
    eta_reach: &mut [T],
) {
    let gamma = matches!(family, Family::Gamma { .. });
    csum_a.fill(T::ZERO);
    csum_b.fill(T::ZERO);
    eta_reach.fill(T::ZERO);
    for (i, &cid) in cluster_ids.iter().enumerate() {
        let c = cid as usize;
        let a = prior_w[i];
        if gamma {
            csum_a[c] += T::from_f64(a); // W_c
            csum_b[c] += T::from_f64(a * y[i]) / prob[i]; // R_c
        } else {
            csum_a[c] += T::from_f64(a * y[i]); // Y_c
            csum_b[c] += prob[i].scale(a); // M_c
        }
        let b = shift_reach(c);
        // Both compressible families take the `Lin` row terms.
        let g = super::agq_fast::mag_factor(super::agq_fast::Kind::Lin, b);
        let fast_row = super::agq_fast::fast_consts(
            family,
            nb_theta,
            y[i],
            a,
            eta[i].value(),
            prob[i].value(),
            b,
            g,
        )
        .is_some();
        let e = if fast_row {
            eta[i].value().abs()
        } else {
            f64::INFINITY
        };
        if e > eta_reach[c].value() {
            eta_reach[c] = T::from_f64(e);
        }
    }
}

/// AGQ deviance for the scalar-intercept (`q_p==1`, no extras) binomial/Poisson/NB/Gamma
/// GLMM at `(θ,β)=params`. Converges the per-cluster conditional modes `ũ_c` and
/// curvatures `A_c` via the same blocked PIRLS the Laplace path uses, then for
/// each cluster integrates the conditional likelihood with `k=nagq` adaptive GH
/// nodes `u_cj = ũ_c + √2·σ_c·z_j` (`σ_c = 1/√A_c`), summed in the bracket below.
///
/// Returns the AGQ deviance `−2·Σ_c log L_c` with the saturated and `2π` constants
/// dropped (the deviance convention), so it equals `laplace_deviance` at `k=1`.
/// Non-convergence ⇒ `f64::INFINITY` (the module's failure surface).
///
/// Math — standardized u-scale (prior `u ~ N(0,1)`, `η = Xβ + λ·u` with `λ` the
/// scalar primary Λ_p factor, so `ũ_c`/`σ_c` from PIRLS are on the u-scale and the
/// integrand must match — a b-scale prior `−½u²/τ²` or dropping `λ` breaks the
/// k=1≡Laplace reduction for `τ²≠1`; on NB the θ-dependent saturated term is
/// constant in u, so it stays out of `ℓ_c` and the driver adds it back as `Dispersion::term`):
/// ```text
///   ℓ_c(u)     = Σ_{i∈c} −½·w_i·dev_resid(family, y_i, g⁻¹(η_fix,i + λ·u)) − ½u²
///   x_cj       = z_j² + ℓ_c(u_cj) − ℓ_c(ũ_c)
///              = −h_cj·(ũ_c − ½·(A_c − 1)·h_cj) + [data part of ℓ_c(u_cj) − ℓ_c(ũ_c)],
///                h_cj = u_cj − ũ_c = √2·σ_c·z_j
///   −2 log L_c = −2·ℓ_c(ũ_c) + ln_1p(A_c − 1) − 2·ln_1p[ Σ_j w_j·expm1(x_cj) / Σ_j w_j ]
/// ```
/// The `z_j²` term un-weights the GH `e^{−z²}` kernel (Liu–Pierce adaptive GHQ);
/// the second form of `x_cj` uses `z_j² = ½·A_c·h_cj²`. Since
/// `Σ_j w_j·e^{x} = Σ_j w_j + Σ_j w_j·expm1(x)`, the bracket is
/// `log[(1/√π)·Σ_j w_j·e^{x_cj}]` less the constant `ln Σ_j w_j − ln √π`: zero for
/// the exact GH weights, which sum to √π, and an ulp-level rounding of the `f64`
/// table, the same at every θ and β, so it is dropped. The node sum is centred on
/// the mode, not max-shifted: each node enters the bracket as `w_j·expm1(x_cj)`.
/// It stays finite because the converged mode maximises `ℓ_c` on these
/// log-concave families, so every `x_cj` is at most the node's own `z_j²`
/// (bounded by the grid), and an odd `k` puts a node at the mode (`x = 0`), so
/// the `ln_1p` argument stays above −1. Every term of `x_cj` is of
/// the size of the data part and of `A_c − 1` (`PirlsScratch::a_excess`, kept apart
/// from `A_c`), which at prior weights far below 1 are both far below ε; formed this
/// way, and summed through `expm1` and `ln_1p`, none of their digits is rounded
/// against a 1. The data part is the row-term difference of `agq_fast`, or on the
/// generic loop the node's data sum minus the center's. At `k=1` the node is
/// `z=0, w=√π`, so `x = 0`, the bracket is 0 and what remains is
/// `−2ℓ_c(ũ_c) + log A_c` — the Laplace term. Validated against frozen
/// `glmer(nAGQ=k)` goldens (`fit::tests::fit_glmm_{binomial,poisson}_agq_*`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn agq_deviance<T: Scalar>(
    family: Family,
    nb_theta: f64,
    groupings: &LmmGroupings,
    params: &[T],
    beta: &mut [T],
    scratch: &mut PirlsScratch<T>,
    z_buf: &[f64],
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    cluster_ids: &[u32],
    dual: Option<&mut DualStep<T>>,
    // AGQ is always `BetaStep::Fixed`, so `pirls_solve_blocked`'s Profile-only
    // C = X'WX GEMM never runs here — this is just uniform plumbing across the
    // three PIRLS variants.
    wx: &mut faer::Mat<f64>,
    nagq: u8,
    pirls_tol_override: Option<f64>,
    n: usize,
    cluster_rows: Option<&ClusterRowIndex>,
    // Per-row linear-predictor offset (`Data::offset`), forwarded to the
    // inner `pirls_solve_blocked` call; the node loop below reads it back off
    // the `eta_fixed` that fill leaves, so nothing else here changes. `None` ⇒
    // no offset.
    offset: Option<&[f64]>,
    counters: &mut crate::counters::EvalCounters,
) -> T {
    // β length p is `beta.len()` (the Fixed-mode buffer the caller filled) — the
    // blocked PIRLS derives p from it, so no separate `p` param is threaded here.
    let n_theta = groupings.n_theta();
    let s = groupings.n_primary;
    // Converge ũ_c / A_c via the same blocked PIRLS the Laplace path uses (β is the
    // caller's Fixed-mode buffer = params[n_theta..]; leaves prob = g⁻¹ at the mode,
    // a_blocks[c] = √A_c). AGQ is always β-fixed, so pass `BetaStep::Fixed` explicitly.
    // This solve (and its seed, set by the caller `laplace_deviance_at`) is
    // mirrored by `agq_gradient::mode_at`; change together.
    crate::lmm::primary_lambda(&params[..n_theta], groupings.primary_q, &mut scratch.lam);
    // `weighted` threads through to PIRLS so the converged mode ũ_c and curvature
    // A_c fold in the prior weights: `pirls_solve_blocked`'s μ/W pass is
    // `Scalar::family_pass`, whose fused Bernoulli-logit kernel is gated on
    // `!weighted` (`simd_transcendental.rs` at `f64`,
    // `scalar::generic_family_pass` on the dual twins), so a wrong unweighted
    // flag would skip `prior_w` entirely and give an unweighted mode/scale —
    // wrong for aggregated-binomial cells. Poisson/probit fold `prior_w`
    // regardless, but the flag must still be correct.
    let (_dev, _pen, _logdet, conv) = pirls_solve_blocked(
        family,
        nb_theta,
        groupings,
        cluster_ids,
        x,
        y,
        prior_w,
        weighted,
        beta,
        BetaStep::Fixed,
        scratch,
        z_buf,
        dual,
        wx,
        offset,
        pirls_tol_override,
        n,
        counters,
    );
    if !conv {
        return T::from_f64(f64::INFINITY);
    }
    let PirlsScratch {
        lam,
        prob,
        eta,
        u,
        eta_fixed,
        a_blocks,
        a_half_logdet,
        a_excess,
        agq_scratch,
        agq_rows,
        agq_slow,
        ..
    } = scratch;
    // Read-only from here; shared borrows also let rayon's `par_iter_mut`
    // closure below capture them (`&mut [T]` is not `Sync`).
    let (u, eta_fixed, a_blocks, a_excess): (&[T], &[T], &[T], &[T]) =
        (u, eta_fixed, a_blocks, a_excess);
    let lambda = lam[0]; // scalar Λ_p (q_p == 1, gated)
    let k = nagq as usize;
    let blk = crate::consts::GH_OFFSETS[(k - 1) / 2];
    let nodes = &crate::consts::GH_NODES[blk..blk + k];
    let wts = &crate::consts::GH_WEIGHTS[blk..blk + k];
    // The bracket's normalisation `Σ_j w_j` (see the doc on the dropped constant).
    let wsum = T::from_f64(wts.iter().sum());

    // Per-cluster scratch (size s each): the data part of ℓ_c(ũ_c) | node shift
    // h_cj | node data sum | running Σ_j w_j·expm1(x_cj) | the generic compressed
    // route's two sums | its max |η| at the mode. Sized by `agq_len` — change
    // together.
    let (ctr, rest) = agq_scratch.split_at_mut(s);
    let (ucj, rest) = rest.split_at_mut(s);
    let (acc, rest) = rest.split_at_mut(s);
    let (sum, rest) = rest.split_at_mut(s);
    let (csum_a, rest) = rest.split_at_mut(s);
    let (csum_b, rest) = rest.split_at_mut(s);
    let eta_reach = &mut rest[..s];

    // ctr_c = Σ_{i∈c} −½·dev_resid at the converged mode (eta[i]/prob[i] already
    // hold η_fix,i + λ·ũ_c and its g⁻¹), the data part of ℓ_c(ũ_c); the prior
    // −½ũ_c² joins it in the deviance sum below.
    for c in 0..s {
        ctr[c] = T::ZERO;
        sum[c] = T::ZERO;
    }
    for i in 0..n {
        let c = cluster_ids[i] as usize;
        // w_i·dev_resid: prior_w[i] is exactly 1.0 on the unweighted path (workspace
        // init), and x·1.0 is bit-exact, so unweighted stays byte-identical.
        ctr[c] -= crate::family::dev_resid_at(family, nb_theta, y[i], eta[i], prob[i])
            .scale(0.5 * prior_w[i]);
    }

    // `f64` evaluations on the log links, logit and probit take the row-term
    // node sums of `agq_fast` (same `sum`, every other quantity unchanged).
    let mut done = false;
    let fast_kind = super::agq_fast::kind(family);
    let probit = super::agq_fast::is_probit(family);
    if fast_kind.is_some() || probit {
        if let (Some(eta_f), Some(prob_f), Some(ef_f), Some(u_f), Some(ab_f), Some(ex_f)) = (
            T::as_f64s(eta),
            T::as_f64s(prob),
            T::as_f64s(eta_fixed),
            T::as_f64s(u),
            T::as_f64s(a_blocks),
            T::as_f64s(a_excess),
        ) {
            if let (Some(dl), Some(acc_f), Some(sum_f)) = (
                T::as_f64s_mut(ucj),
                T::as_f64s_mut(acc),
                T::as_f64s_mut(sum),
            ) {
                if let Some(kind) = fast_kind {
                    super::agq_fast::scalar_sums(
                        kind,
                        family,
                        nb_theta,
                        y,
                        prior_w,
                        cluster_ids,
                        n,
                        eta_f,
                        prob_f,
                        ef_f,
                        u_f,
                        ab_f,
                        ex_f,
                        lambda.value(),
                        nodes,
                        wts,
                        agq_rows,
                        agq_slow,
                        dl,
                        acc_f,
                        sum_f,
                        cluster_rows,
                    );
                } else {
                    super::agq_fast::probit_sums(
                        y,
                        prior_w,
                        cluster_ids,
                        n,
                        eta_f,
                        prob_f,
                        ef_f,
                        u_f,
                        ab_f,
                        ex_f,
                        lambda.value(),
                        nodes,
                        wts,
                        agq_rows,
                        agq_slow,
                        dl,
                        acc_f,
                        sum_f,
                        cluster_rows,
                    );
                }
                done = true;
            }
        }
    }

    // Cluster compression on the generic (dual) route, Poisson-log and
    // Gamma-log, cluster-outer arm: every row of cluster c moves by the same
    // δ = λ·(u − ũ_c), so ℓ_c(u) − ℓ_c(ũ_c) needs two per-cluster sums (see
    // `compressed_cluster_sum`), gathered in one pass over the rows by
    // `compression_gather`. `eta_reach[c]` is max |η| at the mode over the
    // cluster's rows (the stored, clamped η: a clamped row sits at ±ETA_MAX and
    // fails `compress_ok`), or ∞ when a row fails the guard of the `f64` route
    // (`agq_fast::scalar_sums`, which compresses there): the node range reaches
    // the clamp or the tail, or the row's two parts exceed `TERM_MAG_MAX`. So
    // both routes compress the same clusters.
    let compress = !done && compressible(family) && cluster_rows.is_some();
    let gamma = matches!(family, Family::Gamma { .. });
    let z_max = nodes.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    let lambda_abs = lambda.value().abs();
    // Bound on |δ| over cluster c's nodes, formed as `agq_fast::scalar_sums` forms it.
    let shift_reach =
        |c: usize| lambda_abs * std::f64::consts::SQRT_2 * (1.0 / a_blocks[c].value()) * z_max;
    if compress {
        compression_gather(
            family,
            nb_theta,
            y,
            prior_w,
            &cluster_ids[..n],
            eta,
            prob,
            &shift_reach,
            csum_a,
            csum_b,
            eta_reach,
        );
    }

    let ctr: &[T] = ctr;
    let csum_a_ro: &[T] = csum_a;
    let csum_b_ro: &[T] = csum_b;
    // Whether cluster c takes the compressed route (see `compress_ok`).
    let fast = |c: usize, eta_reach_c: T| {
        compress
            && compress_ok(
                eta_reach_c.value(),
                shift_reach(c),
                csum_a_ro[c].value(),
                csum_b_ro[c].value(),
            )
    };
    let compressed = |c: usize| {
        compressed_cluster_sum(
            gamma,
            csum_a_ro[c],
            csum_b_ro[c],
            u[c],
            T::ONE / a_blocks[c],
            a_excess[c],
            lambda,
            nodes,
            wts,
        )
    };
    // x_cj from the node shift h and the node's data sum (the doc's second form).
    let node_term = |c: usize, wj: f64, h: T, data: T| {
        let prior = -(h * (u[c] - a_excess[c].scale(0.5) * h));
        (prior + (data - ctr[c])).exp_m1().scale(wj)
    };
    // GH sum over nodes. σ_c = 1/√A_c = 1/a_blocks[c] (the 1×1 Cholesky factor).
    if !done {
        match cluster_rows {
            // Cluster-outer: per cluster, same node order and (via the CSR's
            // ascending-row guarantee) same per-accumulator operand order as the
            // node-outer loop below, so the two agree bit for bit on every
            // cluster the row loop sums. A Poisson-log or Gamma-log cluster that
            // compresses here has no node-outer twin (that arm never
            // compresses); no dual caller reaches the node-outer arm. Rows stay
            // cache-hot per cluster instead of k full n-sweeps.
            Some(idx) => {
                let eta_reach_ro: &[T] = eta_reach;
                // Per-cluster closure: reads shared slices, writes only its own sum slot —
                // deterministic under any thread schedule, so parallel == serial bitwise.
                let per_cluster = |c: usize, sum_c: &mut T| {
                    if fast(c, eta_reach_ro[c]) {
                        *sum_c = compressed(c);
                        return;
                    }
                    let rows = idx.cluster_rows(c);
                    let sigma_c = T::ONE / a_blocks[c];
                    let mut acc_sum = T::ZERO;
                    for (&zj, &wj) in nodes.iter().zip(wts) {
                        let h = sigma_c.scale(std::f64::consts::SQRT_2).scale(zj);
                        let u_cj = u[c] + h;
                        let mut data = T::ZERO;
                        for &i in rows {
                            let i = i as usize;
                            data -= crate::family::dev_resid_eta(
                                family,
                                nb_theta,
                                y[i],
                                eta_fixed[i] + lambda * u_cj,
                            )
                            .scale(0.5 * prior_w[i]);
                        }
                        acc_sum += node_term(c, wj, h, data);
                    }
                    *sum_c = acc_sum;
                };
                #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
                {
                    use rayon::prelude::*;
                    sum[..s]
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(c, sc)| per_cluster(c, sc));
                }
                #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
                for (c, sc) in sum[..s].iter_mut().enumerate() {
                    per_cluster(c, sc);
                }
            }
            // Node-outer: taken when no `ClusterRowIndex` is handed in, on the
            // `f64` families without row terms (binomial cloglog, Gamma
            // inverse).
            None => {
                for (&zj, &wj) in nodes.iter().zip(wts) {
                    for c in 0..s {
                        ucj[c] = (T::ONE / a_blocks[c])
                            .scale(std::f64::consts::SQRT_2)
                            .scale(zj);
                        acc[c] = T::ZERO;
                    }
                    for i in 0..n {
                        let c = cluster_ids[i] as usize;
                        acc[c] -= crate::family::dev_resid_eta(
                            family,
                            nb_theta,
                            y[i],
                            eta_fixed[i] + lambda * (u[c] + ucj[c]),
                        )
                        .scale(0.5 * prior_w[i]);
                    }
                    for c in 0..s {
                        sum[c] += node_term(c, wj, ucj[c], acc[c]);
                    }
                }
            }
        }
    }

    // −2·Σ_c [ ℓ_c(ũ_c) + log σ_c + ln_1p(Σ_j w_j·expm1(x_cj) / Σ_j w_j) ].
    let mut dev = T::ZERO;
    for c in 0..s {
        let ctr_c = ctr[c] - T::from_f64(0.5) * u[c] * u[c]; // ℓ_c(ũ_c)
        let log_sigma_c = -a_half_logdet[c]; // log σ_c = −½·ln_1p(A_c − 1)
        dev += T::from_f64(-2.0) * (ctr_c + log_sigma_c + (sum[c] / wsum).ln_1p());
    }
    dev
}

/// Multivariate AGQ deviance for the **vector**-RE (`q_p ∈ 2..=3`, single
/// grouping factor, no extras) binomial/Poisson/NB/Gamma GLMM at `(θ,β)=params` — the
/// dimension-up sibling of [`agq_deviance`], reached via the widened
/// `deviance.rs` gate for `q_p ≥ 2`. Converges each cluster's `q_p`-vector
/// conditional mode `ũ_c` and curvature `A_c` with the same blocked PIRLS the
/// Laplace path uses, then integrates the conditional likelihood over a `k=nagq`
/// **product** Gauss–Hermite grid (`k^q` nodes/cluster), summed in the bracket
/// below.
///
/// Returns the AGQ deviance `−2·Σ_c log L_c` with the saturated and `2π`
/// constants dropped (the deviance convention), so it equals `laplace_deviance`
/// at `k=1`. Non-convergence ⇒ `f64::INFINITY`.
///
/// **Convention.** Multivariate Liu–Pierce (1994) adaptive GHQ, standardized
/// u-scale (prior `u_c ~ N(0, I_q)`, `η = Xβ + Z_c·Λ_p·u_c` with `Λ_p` the
/// lower-triangular `primary_lambda` factor), **product** grid only (`k^q`
/// nodes; Smolyak deferred), **odd** `k` (a node at the mode ⇒ the k=1≡Laplace
/// reduction). Binomial/Poisson/NB/Gamma (see
/// `fit::common::assert_model_shape`); Gamma arrives on prior weights `wᵢ/φ`. Adaptive transform per
/// cluster: `u_cj = ũ_c + √2·L_cᵀ⁻¹·z_j`, `L_c` = the `q×q` Cholesky factor of
/// `A_c` (`glmm_block_chol`'s per-block output, reused as-is). Per node the
/// generalizations from the scalar path are all direct — `η_i` per row is
/// `eta_fixed[i] + z_i·(Λ_p·u_cj)`; the node weight is the product
/// `W_j = Π_d w_{j_d}` and the Liu–Pierce reweight the sum `‖z_j‖² = Σ_d z_{j_d}²`;
/// the scale term is `−½ log|A_c|`. With `B_c = A_c − I` (`PirlsScratch::a_excess`,
/// kept apart from `A_c`) and `h_cj = u_cj − ũ_c = √2·L_cᵀ⁻¹·z_j`,
/// `‖z_j‖² = ½·h_cjᵀ A_c h_cj`, so
/// ```text
///   ℓ_c(u)     = Σ_{i∈c} −½·w_i·dev_resid(family, y_i, g⁻¹(η_i(u))) − ½‖u‖²
///   x_cj       = ‖z_j‖² + ℓ_c(u_cj) − ℓ_c(ũ_c)
///              = −h_cj·(ũ_c − ½·B_c h_cj) + [data part of ℓ_c(u_cj) − ℓ_c(ũ_c)]
///   −2 log L_c = −2·ℓ_c(ũ_c) + log|A_c| − 2·ln_1p[ Σ_j W_j·expm1(x_cj) / Σ_j W_j ]
/// ```
/// The bracket drops `q·(ln Σ_j w_j − ln √π)` and keeps the digits of `x_cj` for
/// the reasons [`agq_deviance`] gives. The node sum is centred on the mode, not
/// max-shifted: each node enters the bracket as `W_j·expm1(x_cj)`. It stays
/// finite because the converged mode maximises `ℓ_c` on these log-concave
/// families, so every `x_cj` is at most the node's own `‖z_j‖²` (bounded by the
/// grid), and an odd `k` puts a node at the mode (`x = 0`), so the `ln_1p`
/// argument stays above −1. At `k=1` the single node is `z=0, W=√π^q`,
/// so `x = 0`, the bracket is 0 and what remains is `−2ℓ_c(ũ_c)+log|A_c|` — the
/// Laplace term. The k=1≡Laplace reduction is asserted `rel<1e-12` for q=2 and q=3 in
/// `glmm::tests`; serial≡parallel and k-convergence self-consistency likewise.
///
/// **Oracle.** Validated against **GLMMadaptive** (`mixed_model(nAGQ=k)`) — lme4
/// `glmer` refuses `nAGQ>1` for vector REs, so it covers only the scalar rungs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn agq_deviance_vec<T: Scalar>(
    family: Family,
    nb_theta: f64,
    groupings: &LmmGroupings,
    params: &[T],
    beta: &mut [T],
    scratch: &mut PirlsScratch<T>,
    z_buf: &[f64],
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    cluster_ids: &[u32],
    dual: Option<&mut DualStep<T>>,
    wx: &mut faer::Mat<f64>,
    nagq: u8,
    pirls_tol_override: Option<f64>,
    n: usize,
    cluster_rows: Option<&ClusterRowIndex>,
    // Per-row linear-predictor offset (`Data::offset`), forwarded to the
    // inner `pirls_solve_blocked` call — the node loop below reads it back off
    // the `eta_fixed` that fill leaves. `None` ⇒ no offset.
    offset: Option<&[f64]>,
    counters: &mut crate::counters::EvalCounters,
) -> T {
    let n_theta = groupings.n_theta();
    let s = groupings.n_primary;
    let q = groupings.primary_q; // 2 or 3 (gate-enforced)
                                 // Converge ũ_c / A_c via the same blocked PIRLS the Laplace path uses; leaves
                                 // prob = g⁻¹ at the mode, a_blocks[c] = the q×q Cholesky factor L_c of A_c.
                                 // AGQ is always β-fixed; `weighted` folds prior weights into the mode/curvature
                                 // (see agq_deviance for why the flag must be correct on the logit fast path).
    crate::lmm::primary_lambda(&params[..n_theta], q, &mut scratch.lam);
    let (_dev, _pen, _logdet, conv) = pirls_solve_blocked(
        family,
        nb_theta,
        groupings,
        cluster_ids,
        x,
        y,
        prior_w,
        weighted,
        beta,
        BetaStep::Fixed,
        scratch,
        z_buf,
        dual,
        wx,
        offset,
        pirls_tol_override,
        n,
        counters,
    );
    if !conv {
        return T::from_f64(f64::INFINITY);
    }
    let PirlsScratch {
        lam,
        prob,
        eta,
        u,
        eta_fixed,
        a_blocks,
        a_half_logdet,
        a_excess,
        agq_scratch,
        agq_rows,
        ..
    } = scratch;
    let k = nagq as usize;
    let blk = crate::consts::GH_OFFSETS[(k - 1) / 2];
    let nodes = &crate::consts::GH_NODES[blk..blk + k];
    let wts = &crate::consts::GH_WEIGHTS[blk..blk + k];
    let kq = k.pow(q as u32);

    // Scratch: the data part of ℓ_c(ũ_c) (s) | running Σ_j W_j·expm1(x_cj) (s) |
    // product-grid node table (k^q·(q+1): the q z-vector components + the
    // product weight W_j per node). Per-cluster temporaries (h_cj, v_cj) are
    // [T;3] stack arrays. Sized by `agq_len` — change together.
    let (ctr, rest) = agq_scratch.split_at_mut(s);
    let (sum, node_tbl) = rest.split_at_mut(s);
    let node_tbl = &mut node_tbl[..kq * (q + 1)];

    // Hoist the k^q node table once per eval (parameter-independent within an
    // eval), shared read-only across clusters. Multi-index t = (j_0,…,j_{q-1})
    // base k; store the z-vector then W_t = Π_d w_{j_d}, and sum the W_t for the
    // bracket's normalisation.
    let mut wsum = 0.0f64;
    for t in 0..kq {
        let base = t * (q + 1);
        let mut rem = t;
        let mut wt = 1.0f64;
        for d in 0..q {
            let jd = rem % k;
            rem /= k;
            node_tbl[base + d] = T::from_f64(nodes[jd]);
            wt *= wts[jd];
        }
        node_tbl[base + q] = T::from_f64(wt);
        wsum += wt;
    }

    // ctr_c = Σ_{i∈c} −½·dev_resid at the mode (eta[i]/prob[i] already hold the
    // converged η and its g⁻¹), the data part of ℓ_c(ũ_c); the prior −½‖ũ_c‖²
    // joins it in the deviance sum below.
    for c in 0..s {
        ctr[c] = T::ZERO;
        sum[c] = T::ZERO;
    }
    for i in 0..n {
        let c = cluster_ids[i] as usize;
        // w_i·dev_resid; prior_w[i]==1.0 exactly on the unweighted path ⇒ byte-identical.
        ctr[c] -= crate::family::dev_resid_at(family, nb_theta, y[i], eta[i], prob[i])
            .scale(0.5 * prior_w[i]);
    }

    // `f64` evaluations on the log links and logit take the row-term node
    // sums of `agq_fast` (same `sum`, every other quantity unchanged).
    let mut done = false;
    if let Some(kind) = super::agq_fast::kind(family) {
        if let (
            Some(eta_f),
            Some(prob_f),
            Some(ef_f),
            Some(u_f),
            Some(ab_f),
            Some(ex_f),
            Some(lam_f),
            Some(tbl_f),
        ) = (
            T::as_f64s(eta),
            T::as_f64s(prob),
            T::as_f64s(eta_fixed),
            T::as_f64s(u),
            T::as_f64s(a_blocks),
            T::as_f64s(a_excess),
            T::as_f64s(lam),
            T::as_f64s(node_tbl),
        ) {
            if let Some(sum_f) = T::as_f64s_mut(sum) {
                // A handed-in index runs the clusters on rayon, the transient one
                // serially, as the generic loop below does.
                let parallel = cluster_rows.is_some();
                let owned;
                let idx = match cluster_rows {
                    Some(idx) => idx,
                    None => {
                        owned = ClusterRowIndex::build(cluster_ids, s);
                        &owned
                    }
                };
                super::agq_fast::vector_sums(
                    kind, family, nb_theta, y, prior_w, z_buf, q, n, eta_f, prob_f, ef_f, u_f,
                    ab_f, ex_f, lam_f, nodes, tbl_f, agq_rows, sum_f, idx, parallel,
                );
                done = true;
            }
        }
    }

    // Shared read-only reborrows for the parallel closure (rayon needs Sync
    // captures; `&mut [T]` isn't Sync). The per-cluster body only reads these.
    let lam_ro: &[T] = lam;
    let eta_fixed_ro: &[T] = eta_fixed;
    let u_ro: &[T] = u;
    let a_blocks_ro: &[T] = a_blocks;
    let a_excess_ro: &[T] = a_excess;
    let ctr_ro: &[T] = ctr;
    let node_tbl_ro: &[T] = node_tbl;
    // Per-cluster integral: reads shared slices, writes only its own `sum` slot —
    // deterministic under any thread schedule ⇒ parallel == serial bitwise.
    let per_cluster = |idx: &ClusterRowIndex, c: usize, sum_c: &mut T| {
        let rows = idx.cluster_rows(c);
        let lblk = &a_blocks_ro[c * q * q..c * q * q + q * q]; // L_c (row-major lower)
        let bblk = &a_excess_ro[c * q * q..c * q * q + q * q]; // B_c (row-major lower)
        let ubase = c * q;
        let mut acc_sum = T::ZERO;
        for t in 0..kq {
            let nbase = t * (q + 1);
            let znode = &node_tbl_ro[nbase..nbase + q];
            // Stored lifted, so its lanes are zero.
            let wt = node_tbl_ro[nbase + q].value();
            // h_cj = √2·L_cᵀ⁻¹·z_j: back-solve L_cᵀ·w = z (upper-tri), then scale.
            // L_cᵀ[r][cc] = L_c[cc][r] = lblk[cc·q + r].
            let mut h = [T::ZERO; 3];
            for r in (0..q).rev() {
                let mut v = znode[r];
                for cc in (r + 1)..q {
                    v -= lblk[cc * q + r] * h[cc];
                }
                h[r] = v / lblk[r * q + r];
            }
            for hr in &mut h[..q] {
                *hr = hr.scale(std::f64::consts::SQRT_2);
            }
            // v_cj = Λ_p·(ũ_c + h_cj) (Λ_p lower-tri row-major = lam).
            let mut v_cj = [T::ZERO; 3];
            for r in 0..q {
                let mut vv = T::ZERO;
                for cc in 0..=r {
                    vv += lam_ro[r * q + cc] * (u_ro[ubase + cc] + h[cc]);
                }
                v_cj[r] = vv;
            }
            // The data part at η_i = eta_fixed[i] + z_i·v_cj (z_i = [1, z_buf row]).
            let mut acc_c = T::ZERO;
            for &i in rows {
                let i = i as usize;
                let mut eta_i = eta_fixed_ro[i] + v_cj[0]; // z_i0 = 1
                for d in 0..q - 1 {
                    eta_i += v_cj[d + 1].scale(z_buf[i * (q - 1) + d]);
                }
                acc_c -= crate::family::dev_resid_eta(family, nb_theta, y[i], eta_i)
                    .scale(0.5 * prior_w[i]);
            }
            let prior = vec_prior(&h, &u_ro[ubase..ubase + q], bblk, q);
            acc_sum += (prior + (acc_c - ctr_ro[c])).exp_m1().scale(wt);
        }
        *sum_c = acc_sum;
    };
    if !done {
        match cluster_rows {
            // Prebuilt CSR (`fit_glmm` builds it once per fit: in a `parallel` build
            // when `parallel_inner` is set, in a serial build always; a derivative
            // scratch hands in its own): rayon-parallel under the feature, serial
            // otherwise. Cluster-outer either way.
            Some(idx) => {
                #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
                {
                    use rayon::prelude::*;
                    sum[..s]
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(c, sc)| per_cluster(idx, c, sc));
                }
                #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
                for (c, sc) in sum[..s].iter_mut().enumerate() {
                    per_cluster(idx, c, sc);
                }
            }
            // Fallback with no prebuilt CSR (FD-Hessian workers, a `parallel` build
            // with `parallel_inner` off).
            // The vector kernel is cluster-outer only (no node-outer variant: at
            // 49–729 nodes/cluster it is the sole sensible shape), so build a
            // transient index here — O(n), negligible against the k^q·n node sweeps.
            // Same `build` ⇒ same ascending-row ordering as the `Some` arm, so the
            // result is bit-identical.
            None => {
                let idx = ClusterRowIndex::build(cluster_ids, s);
                for (c, sc) in sum[..s].iter_mut().enumerate() {
                    per_cluster(&idx, c, sc);
                }
            }
        }
    }

    // −2·Σ_c [ ℓ_c(ũ_c) − ½ log|A_c| + ln_1p(Σ_j W_j·expm1(x_cj) / Σ_j W_j) ].
    let mut dev = T::ZERO;
    let wsum = T::from_f64(wsum);
    for c in 0..s {
        let ubase = c * q;
        let mut ctr_c = ctr[c]; // ℓ_c(ũ_c)
        for r in 0..q {
            ctr_c -= T::from_f64(0.5) * u[ubase + r] * u[ubase + r];
        }
        let log_scale = -a_half_logdet[c]; // −½ log|A_c|
        dev += T::from_f64(-2.0) * (ctr_c + log_scale + (sum[c] / wsum).ln_1p());
    }
    dev
}

/// The prior part of the vector node exponent, `−h·(ũ − ½·B h)`, from the node
/// shift `h`, the mode `ũ` and `B = A − I` stored as its lower triangle
/// row-major (`agq_deviance_vec`'s doc). Shared by the generic loop and
/// `agq_fast::vector_sums`.
#[inline]
pub(super) fn vec_prior<T: Scalar>(h: &[T; 3], u: &[T], b: &[T], q: usize) -> T {
    let mut acc = T::ZERO;
    for r in 0..q {
        let mut bh = T::ZERO;
        for (cc, &hc) in h[..q].iter().enumerate() {
            let (hi, lo) = if cc > r { (cc, r) } else { (r, cc) };
            bh += b[hi * q + lo] * hc;
        }
        acc -= h[r] * (u[r] - bh.scale(0.5));
    }
    acc
}

/// Test-only: drive `agq_deviance`/`agq_deviance_vec` from a `&[f64]` params
/// (mirrors `deviance::glmm_laplace_deviance`). Routes by `q_p` exactly as the
/// production gate does (`q_p==1`→scalar, `q_p∈2..=3`→vector). Forces the AGQ
/// path at the given `nagq` regardless of the production gate, so the
/// k=1≡Laplace reduction can be asserted directly against `glmm_laplace_deviance`.
#[cfg(test)]
pub(crate) fn glmm_agq_deviance(
    params: &[f64],
    ws: &mut super::workspace::GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    n: usize,
    nagq: u8,
) -> f64 {
    ws.params[..params.len()].copy_from_slice(params);
    super::workspace::fill_z_f64(&ws.groupings, x, x.subcols(0, 0), &mut ws.z_buf, n);
    for v in ws.pirls.u.iter_mut() {
        *v = 0.0; // self-contained PIRLS seed (mode is point-determined; seed only shifts iterates)
    }
    let family = ws.family;
    let nb_theta = ws.nb_theta;
    let weighted = ws.weighted;
    let offset = ws.offset.as_deref();
    let super::workspace::GlmmWorkspace {
        groupings,
        params: prm,
        beta_rhs,
        p,
        pirls,
        z_buf,
        prior_w,
        wx,
        cluster_rows,
        ..
    } = ws;
    // Fixed-mode β copy (mirrors `laplace_deviance`): value-exact `params[n_theta..]`
    // → `beta_rhs`, the transient β buffer `agq_deviance` reads (NOT `betas`, which
    // is the fit's reported output).
    let nt = groupings.n_theta();
    beta_rhs[..*p].copy_from_slice(&prm[nt..nt + *p]);
    let kernel = if groupings.primary_q == 1 {
        agq_deviance::<f64>
    } else {
        agq_deviance_vec::<f64>
    };
    kernel(
        family,
        nb_theta,
        groupings,
        &prm[..],
        beta_rhs,
        pirls,
        z_buf,
        x,
        y,
        &prior_w[..n],
        weighted,
        cluster_ids,
        None,
        wx,
        nagq,
        None,
        n,
        cluster_rows.as_ref(),
        offset,
        &mut crate::counters::EvalCounters::new(),
    )
}

#[cfg(test)]
#[path = "agq_tests.rs"]
mod tests;
