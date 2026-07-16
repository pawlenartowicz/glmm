//! Adaptive Gauss–Hermite quadrature (`nAGQ>1`) for the single scalar-intercept
//! binomial / Poisson GLMM — the only shape where the marginal likelihood is a
//! product of independent 1-D cluster integrals, so AGQ is a per-cluster k-node
//! sum with no curse of dimensionality. Replaces the Laplace `+log|A|` curvature
//! term with a Liu–Pierce (1994) adaptive GH sum centered at each cluster's PIRLS
//! mode/curvature. The sibling kernel [`agq_deviance_vec`] extends this one
//! dimension up — a **vector** RE (`q_p ∈ 2..=3`) on a single grouping factor,
//! integrated over a `k^q` product grid; the scalar kernel stays untouched.
//!
//! `nAGQ=1` is the Laplace term exactly (the k=1 node sits at the mode with GH
//! weight √π), so the production path routes `nagq==1` to `laplace_deviance`
//! verbatim; only `nagq>1` reaches here — see the gate in
//! `deviance::laplace_deviance`. The white-box k=1≡Laplace reduction is asserted
//! to `rel<1e-12` in `glmm::tests` (different op order, not bit-equality).

use faer::MatRef;

use super::pirls::{pirls_solve_blocked, BetaStep};
use crate::lmm::LmmGroupings;
use crate::spec::Family;

/// Per-cluster row index (CSR): `ptr[c]..ptr[c+1]` slices `rows` down to cluster `c`'s
/// row indices, in ASCENDING original-row order. That ordering is load-bearing, not
/// cosmetic: a cluster-outer restructuring of the node loop in [`agq_deviance`] must
/// visit each cluster's rows in the same order the current node-outer loop does (row
/// `0..n` ascending) to stay bit-identical — same operands, same summation order, no
/// re-validation against the frozen `glmer(nAGQ=k)` goldens needed. `build` gets this
/// for free (see below), so no separate sort step is required, unlike the crossed-
/// extras coupling CSR this mirrors (`pirls::build_coupling_csr`) — that one needs a
/// sort+dedup because a row can repeat across a cluster's crossed columns; here a row
/// belongs to exactly one cluster, so no duplicates are possible.
///
/// Wired into [`agq_deviance`]'s `cluster_rows` parameter, gated by
/// `FitOptions::parallel_inner`.
/// Built ONCE per fit in `fit_glmm` (`cluster_ids` doesn't change across BOBYQA
/// evals), not once per eval.
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

    /// Cluster `c`'s row indices, ascending original-row order.
    pub(crate) fn cluster_rows(&self, c: usize) -> &[u32] {
        &self.rows[self.ptr[c] as usize..self.ptr[c + 1] as usize]
    }
}

/// AGQ deviance for the scalar-intercept (`q_p==1`, no extras) binomial/Poisson
/// GLMM at `(θ,β)=params`. Converges the per-cluster conditional modes `ũ_c` and
/// curvatures `A_c` via the same blocked PIRLS the Laplace path uses, then for
/// each cluster integrates the conditional likelihood with `k=nagq` adaptive GH
/// nodes `u_cj = ũ_c + √2·σ_c·z_j` (`σ_c = 1/√A_c`), combined by log-sum-exp.
///
/// Returns the AGQ deviance `−2·Σ_c log L_c` with the saturated and `2π` constants
/// dropped (the deviance convention), so it equals `laplace_deviance` at `k=1`.
/// Non-convergence ⇒ `f64::INFINITY` (the module's failure surface).
///
/// Math — standardized u-scale (prior `u ~ N(0,1)`, `η = Xβ + λ·u` with `λ` the
/// scalar primary Λ_p factor, so `ũ_c`/`σ_c` from PIRLS are on the u-scale and the
/// integrand must match — a b-scale prior `−½u²/τ²` or dropping `λ` breaks the
/// k=1≡Laplace reduction for `τ²≠1`):
/// ```text
///   ℓ_c(u)     = Σ_{i∈c} −½·w_i·dev_resid(family, y_i, g⁻¹(η_fix,i + λ·u)) − ½u²
///   −2 log L_c = −2·ℓ_c(ũ_c) − 2·log σ_c
///              − 2·log[ (1/√π)·Σ_j w_j·exp(z_j² + ℓ_c(u_cj) − ℓ_c(ũ_c)) ]
/// ```
/// The `z_j²` term un-weights the GH `e^{−z²}` kernel (Liu–Pierce adaptive GHQ);
/// at `k=1` the node is `z=0, w=√π`, so the bracket is `log((1/√π)·√π) = 0` and
/// what remains is `−2ℓ_c(ũ_c) + log A_c` — the Laplace term. Validated against
/// frozen `glmer(nAGQ=k)` goldens (`fit::tests::fit_glmm_{binomial,poisson}_agq_*`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn agq_deviance(
    family: Family,
    nb_theta: f64,
    groupings: &LmmGroupings,
    params: &[f64],
    beta: &mut [f64],
    lam: &mut [f64],
    z_buf: &[f64],
    m_buf: &mut [f64],
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    cluster_ids: &[u32],
    eta: &mut [f64],
    prob: &mut [f64],
    w: &mut [f64],
    u: &mut [f64],
    u_prev: &mut [f64],
    eta_fixed: &mut [f64],
    a_blocks: &mut [f64],
    a_rhs: &mut [f64],
    // AGQ is always `BetaStep::Fixed`, so `pirls_solve_blocked`'s Profile-only
    // C = X'WX GEMM never runs here — this is just uniform plumbing across the
    // three PIRLS variants.
    wx: &mut faer::Mat<f64>,
    agq_scratch: &mut [f64],
    nagq: u8,
    pirls_tol_override: Option<f64>,
    n: usize,
    cluster_rows: Option<&ClusterRowIndex>,
) -> f64 {
    // β length p is `beta.len()` (the Fixed-mode buffer the caller filled) — the
    // blocked PIRLS derives p from it, so no separate `p` param is threaded here.
    let n_theta = groupings.n_theta();
    let s = groupings.n_primary;
    // Converge ũ_c / A_c via the same blocked PIRLS the Laplace path uses (β is the
    // caller's Fixed-mode buffer = params[n_theta..]; leaves prob = g⁻¹ at the mode,
    // a_blocks[c] = √A_c). AGQ is always β-fixed, so pass `BetaStep::Fixed` explicitly.
    crate::lmm::primary_lambda(&params[..n_theta], groupings.primary_q, lam);
    // `weighted` threads through to PIRLS so the converged mode ũ_c and curvature
    // A_c fold in the prior weights: on the logit-binomial fast path (a `!weighted`
    // match arm in pirls.rs) an unweighted flag would skip `prior_w` entirely and
    // give an unweighted mode/scale, wrong for aggregated-binomial cells. Poisson/
    // probit fold `prior_w` regardless, but the flag must still be correct.
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
        lam,
        z_buf,
        m_buf,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        a_blocks,
        a_rhs,
        wx,
        pirls_tol_override,
        n,
    );
    if !conv {
        return f64::INFINITY;
    }
    let lambda = lam[0]; // scalar Λ_p (q_p == 1, gated)
    let k = nagq as usize;
    let blk = crate::consts::GH_OFFSETS[(k - 1) / 2];
    let nodes = &crate::consts::GH_NODES[blk..blk + k];
    let wts = &crate::consts::GH_WEIGHTS[blk..blk + k];
    let ln_sqrt_pi = 0.5 * std::f64::consts::PI.ln();

    // Per-cluster scratch (size s each): ℓ_c(ũ_c) | node u_cj | ℓ_c(u_cj) | running Σ.
    let (ctr, rest) = agq_scratch.split_at_mut(s);
    let (ucj, rest) = rest.split_at_mut(s);
    let (acc, sum) = rest.split_at_mut(s);

    // ctr_c = ℓ_c(ũ_c): RE prior −½ũ_c², then Σ_{i∈c} −½·dev_resid at the converged
    // mode (prob[i] already holds g⁻¹(η_fix,i + λ·ũ_c)).
    for c in 0..s {
        ctr[c] = -0.5 * u[c] * u[c];
        sum[c] = 0.0;
    }
    for i in 0..n {
        let c = cluster_ids[i] as usize;
        // w_i·dev_resid: prior_w[i] is exactly 1.0 on the unweighted path (workspace
        // init), and x·1.0 is bit-exact, so unweighted stays byte-identical.
        ctr[c] -= 0.5 * prior_w[i] * crate::family::dev_resid(family, nb_theta, y[i], prob[i]);
    }

    // GH sum over nodes. σ_c = 1/√A_c = 1/a_blocks[c] (the 1×1 Cholesky factor).
    // Shared reborrows for the `Some` arm's closure below: rayon's `par_iter_mut`
    // needs `Sync` captures, and `&mut [f64]` isn't `Sync` — downgrade to `&[f64]`
    // since the per-cluster body only ever reads these buffers, never writes them.
    // Kept as separate bindings (not a shadow) so the `None` arm and the dev-sum
    // loop below still use the original mutable names unchanged.
    let eta_fixed_ro: &[f64] = eta_fixed;
    let u_ro: &[f64] = u;
    let a_blocks_ro: &[f64] = a_blocks;
    let ctr_ro: &[f64] = ctr;
    match cluster_rows {
        // Cluster-outer: per cluster, same node order and (via the CSR's
        // ascending-row guarantee) same per-accumulator operand order as the
        // node-outer loop below — bit-identical by construction, so the frozen
        // glmer(nAGQ=k) goldens re-validate nothing. Rows stay cache-hot per
        // cluster instead of k full n-sweeps.
        Some(idx) => {
            // ln(w_j·e^{z_j²}) per node — the Liu–Pierce reweight, a function of
            // the GH table only. Filled once (k ≤ MAX_NAGQ, stack buffer — no
            // alloc on the hot path) so this arm reads it instead of
            // recomputing per (cluster, node): that recomputation cost s·k
            // ln() calls per eval and was the dominant cluster-outer overhead
            // on many-tiny-cluster shapes. Same operands, deterministic —
            // bit-identical to the inline form (only reader — the node-outer
            // `None` arm below recomputes `ln_wj` per node instead).
            let mut ln_wj_buf = [0.0f64; crate::consts::MAX_NAGQ as usize];
            for (j, (&zj, &wj)) in nodes.iter().zip(wts).enumerate() {
                ln_wj_buf[j] = wj.ln() + zj * zj;
            }
            let ln_wj_ro: &[f64] = &ln_wj_buf[..k];
            // Per-cluster closure: reads shared slices, writes only its own sum slot —
            // deterministic under any thread schedule, so parallel == serial bitwise.
            let per_cluster = |c: usize, sum_c: &mut f64| {
                let rows = idx.cluster_rows(c);
                let sigma_c = 1.0 / a_blocks_ro[c];
                let mut acc_sum = 0.0;
                for (j, &zj) in nodes.iter().enumerate() {
                    let ln_wj = ln_wj_ro[j];
                    let u_cj = u_ro[c] + std::f64::consts::SQRT_2 * sigma_c * zj;
                    let mut acc_c = -0.5 * u_cj * u_cj;
                    for &i in rows {
                        let i = i as usize;
                        let mu = crate::family::link_inv(family, eta_fixed_ro[i] + lambda * u_cj);
                        acc_c -=
                            0.5 * prior_w[i] * crate::family::dev_resid(family, nb_theta, y[i], mu);
                    }
                    acc_sum += (ln_wj + acc_c - ctr_ro[c]).exp();
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
        // Node-outer original (parallel_inner == false): unchanged, verbatim.
        None => {
            for (&zj, &wj) in nodes.iter().zip(wts) {
                let ln_wj = wj.ln() + zj * zj; // log(w_j·e^{z_j²}) — the Liu–Pierce reweight
                for c in 0..s {
                    let sigma_c = 1.0 / a_blocks[c];
                    ucj[c] = u[c] + std::f64::consts::SQRT_2 * sigma_c * zj;
                    acc[c] = -0.5 * ucj[c] * ucj[c]; // RE prior at the node
                }
                for i in 0..n {
                    let c = cluster_ids[i] as usize;
                    let mu = crate::family::link_inv(family, eta_fixed[i] + lambda * ucj[c]);
                    acc[c] -=
                        0.5 * prior_w[i] * crate::family::dev_resid(family, nb_theta, y[i], mu);
                }
                for c in 0..s {
                    sum[c] += (ln_wj + acc[c] - ctr[c]).exp();
                }
            }
        }
    }

    // −2·Σ_c [ ℓ_c(ũ_c) + log σ_c + log((1/√π)·Σ_j …) ].
    let mut dev = 0.0;
    for c in 0..s {
        let log_sigma_c = -a_blocks[c].ln(); // log σ_c = −log √A_c
        dev += -2.0 * (ctr[c] + log_sigma_c + sum[c].ln() - ln_sqrt_pi);
    }
    dev
}

/// Multivariate AGQ deviance for the **vector**-RE (`q_p ∈ 2..=3`, single
/// grouping factor, no extras) binomial/Poisson GLMM at `(θ,β)=params` — the
/// dimension-up sibling of [`agq_deviance`], reached via the widened
/// `deviance.rs` gate for `q_p ≥ 2`. Converges each cluster's `q_p`-vector
/// conditional mode `ũ_c` and curvature `A_c` with the same blocked PIRLS the
/// Laplace path uses, then integrates the conditional likelihood over a `k=nagq`
/// **product** Gauss–Hermite grid (`k^q` nodes/cluster), combined by
/// log-sum-exp.
///
/// Returns the AGQ deviance `−2·Σ_c log L_c` with the saturated and `2π`
/// constants dropped (the deviance convention), so it equals `laplace_deviance`
/// at `k=1`. Non-convergence ⇒ `f64::INFINITY`.
///
/// **Convention.** Multivariate Liu–Pierce (1994) adaptive GHQ, standardized
/// u-scale (prior `u_c ~ N(0, I_q)`, `η = Xβ + Z_c·Λ_p·u_c` with `Λ_p` the
/// lower-triangular `primary_lambda` factor), **product** grid only (`k^q`
/// nodes; Smolyak deferred), **odd** `k` (a node at the mode ⇒ the k=1≡Laplace
/// reduction). Binomial/Poisson only (NB/Gamma deferred). Adaptive transform per
/// cluster: `u_cj = ũ_c + √2·L_cᵀ⁻¹·z_j`, `L_c` = the `q×q` Cholesky factor of
/// `A_c` (`glmm_block_chol`'s per-block output, reused as-is). Per node the
/// generalizations from the scalar path are all direct — `η_i` per row is
/// `eta_fixed[i] + z_i·(Λ_p·u_cj)`; the Liu–Pierce reweight is the log-space sum
/// `Σ_d (ln w_{j_d} + z_{j_d}²)`; the normalization is `q·ln√π`; the scale term
/// is `−Σ_r ln L_c[r,r] = −½ log|A_c|`.
/// ```text
///   ℓ_c(u)     = Σ_{i∈c} −½·w_i·dev_resid(family, y_i, g⁻¹(η_i(u))) − ½‖u‖²
///   −2 log L_c = −2·ℓ_c(ũ_c) − 2·(−Σ_r ln L_c[r,r])
///              − 2·log[ (1/√π)^q · Σ_j exp( ln_wj + ℓ_c(u_cj) − ℓ_c(ũ_c) ) ]
/// ```
/// At `k=1` the single node is `z=0, ln_wj=q·ln√π`, so the bracket is
/// `log((1/√π)^q·√π^q)=0` and what remains is `−2ℓ_c(ũ_c)+log|A_c|` — the Laplace
/// term. The k=1≡Laplace reduction is asserted `rel<1e-12` for q=2 and q=3 in
/// `glmm::tests`; serial≡parallel and k-convergence self-consistency likewise.
///
/// **Oracle.** Validated against **GLMMadaptive** (`mixed_model(nAGQ=k)`) — lme4
/// `glmer` refuses `nAGQ>1` for vector REs, so it covers only the scalar rungs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn agq_deviance_vec(
    family: Family,
    nb_theta: f64,
    groupings: &LmmGroupings,
    params: &[f64],
    beta: &mut [f64],
    lam: &mut [f64],
    z_buf: &[f64],
    m_buf: &mut [f64],
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    cluster_ids: &[u32],
    eta: &mut [f64],
    prob: &mut [f64],
    w: &mut [f64],
    u: &mut [f64],
    u_prev: &mut [f64],
    eta_fixed: &mut [f64],
    a_blocks: &mut [f64],
    a_rhs: &mut [f64],
    wx: &mut faer::Mat<f64>,
    agq_scratch: &mut [f64],
    nagq: u8,
    pirls_tol_override: Option<f64>,
    n: usize,
    cluster_rows: Option<&ClusterRowIndex>,
) -> f64 {
    let n_theta = groupings.n_theta();
    let s = groupings.n_primary;
    let q = groupings.primary_q; // 2 or 3 (gate-enforced)
                                 // Converge ũ_c / A_c via the same blocked PIRLS the Laplace path uses; leaves
                                 // prob = g⁻¹ at the mode, a_blocks[c] = the q×q Cholesky factor L_c of A_c.
                                 // AGQ is always β-fixed; `weighted` folds prior weights into the mode/curvature
                                 // (see agq_deviance for why the flag must be correct on the logit fast path).
    crate::lmm::primary_lambda(&params[..n_theta], q, lam);
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
        lam,
        z_buf,
        m_buf,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        a_blocks,
        a_rhs,
        wx,
        pirls_tol_override,
        n,
    );
    if !conv {
        return f64::INFINITY;
    }
    let k = nagq as usize;
    let blk = crate::consts::GH_OFFSETS[(k - 1) / 2];
    let nodes = &crate::consts::GH_NODES[blk..blk + k];
    let wts = &crate::consts::GH_WEIGHTS[blk..blk + k];
    let ln_sqrt_pi = 0.5 * std::f64::consts::PI.ln();
    let kq = k.pow(q as u32);

    // Scratch: ctr_c = ℓ_c(ũ_c) (s) | running Σ_j (s) | product-grid node table
    // (k^q·(q+1): the q z-vector components + the summed Liu–Pierce reweight per
    // node). Per-cluster temporaries (u_cj, v_cj) are [f64;3] stack arrays.
    let (ctr, rest) = agq_scratch.split_at_mut(s);
    let (sum, node_tbl) = rest.split_at_mut(s);
    let node_tbl = &mut node_tbl[..kq * (q + 1)];

    // Hoist the k^q node table once per eval (parameter-independent within an
    // eval), shared read-only across clusters — generalizes the scalar ln_wj_buf
    // hoist. Multi-index t = (j_0,…,j_{q-1}) base k; store the z-vector then
    // ln_wj = Σ_d (ln w_{j_d} + z_{j_d}²) (the product rule is a log-space sum).
    for t in 0..kq {
        let base = t * (q + 1);
        let mut rem = t;
        let mut ln_wj = 0.0;
        for d in 0..q {
            let jd = rem % k;
            rem /= k;
            let zj = nodes[jd];
            node_tbl[base + d] = zj;
            ln_wj += wts[jd].ln() + zj * zj;
        }
        node_tbl[base + q] = ln_wj;
    }

    // ctr_c = ℓ_c(ũ_c): RE prior −½‖ũ_c‖², then Σ_{i∈c} −½·dev_resid at the mode
    // (prob[i] already holds g⁻¹ at the converged η).
    for c in 0..s {
        let ubase = c * q;
        let mut acc = 0.0;
        for r in 0..q {
            acc -= 0.5 * u[ubase + r] * u[ubase + r];
        }
        ctr[c] = acc;
        sum[c] = 0.0;
    }
    for i in 0..n {
        let c = cluster_ids[i] as usize;
        // w_i·dev_resid; prior_w[i]==1.0 exactly on the unweighted path ⇒ byte-identical.
        ctr[c] -= 0.5 * prior_w[i] * crate::family::dev_resid(family, nb_theta, y[i], prob[i]);
    }

    // Shared read-only reborrows for the parallel closure (rayon needs Sync
    // captures; `&mut [f64]` isn't Sync). The per-cluster body only reads these.
    let lam_ro: &[f64] = lam;
    let eta_fixed_ro: &[f64] = eta_fixed;
    let u_ro: &[f64] = u;
    let a_blocks_ro: &[f64] = a_blocks;
    let ctr_ro: &[f64] = ctr;
    let node_tbl_ro: &[f64] = node_tbl;
    // Per-cluster integral: reads shared slices, writes only its own `sum` slot —
    // deterministic under any thread schedule ⇒ parallel == serial bitwise.
    let per_cluster = |idx: &ClusterRowIndex, c: usize, sum_c: &mut f64| {
        let rows = idx.cluster_rows(c);
        let lblk = &a_blocks_ro[c * q * q..c * q * q + q * q]; // L_c (row-major lower)
        let ubase = c * q;
        let mut acc_sum = 0.0;
        for t in 0..kq {
            let nbase = t * (q + 1);
            let znode = &node_tbl_ro[nbase..nbase + q];
            let ln_wj = node_tbl_ro[nbase + q];
            // u_cj = ũ_c + √2·L_cᵀ⁻¹·z_j: back-solve L_cᵀ·w = z (upper-tri), then
            // shift/scale. L_cᵀ[r][cc] = L_c[cc][r] = lblk[cc·q + r].
            let mut u_cj = [0.0f64; 3];
            for r in (0..q).rev() {
                let mut v = znode[r];
                for cc in (r + 1)..q {
                    v -= lblk[cc * q + r] * u_cj[cc];
                }
                u_cj[r] = v / lblk[r * q + r];
            }
            for r in 0..q {
                u_cj[r] = u_ro[ubase + r] + std::f64::consts::SQRT_2 * u_cj[r];
            }
            // v_cj = Λ_p·u_cj (Λ_p lower-tri row-major = lam): v[r] = Σ_{cc≤r} lam[r·q+cc]·u_cj[cc].
            let mut v_cj = [0.0f64; 3];
            for r in 0..q {
                let mut vv = 0.0;
                for cc in 0..=r {
                    vv += lam_ro[r * q + cc] * u_cj[cc];
                }
                v_cj[r] = vv;
            }
            // ℓ_c(u_cj): RE prior −½‖u_cj‖², then the per-row deviance at η_i =
            // eta_fixed[i] + z_i·v_cj (z_i = [1, z_buf row]).
            let mut acc_c = 0.0;
            for &uc in &u_cj[..q] {
                acc_c -= 0.5 * uc * uc;
            }
            for &i in rows {
                let i = i as usize;
                let mut eta_i = eta_fixed_ro[i] + v_cj[0]; // z_i0 = 1
                for d in 0..q - 1 {
                    eta_i += z_buf[i * (q - 1) + d] * v_cj[d + 1];
                }
                let mu = crate::family::link_inv(family, eta_i);
                acc_c -= 0.5 * prior_w[i] * crate::family::dev_resid(family, nb_theta, y[i], mu);
            }
            acc_sum += (ln_wj + acc_c - ctr_ro[c]).exp();
        }
        *sum_c = acc_sum;
    };
    match cluster_rows {
        // Prebuilt CSR (parallel_inner): rayon-parallel under the feature, serial
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
        // Serial fallback (no prebuilt CSR — FD-Hessian workers, serial builds).
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

    // −2·Σ_c [ ℓ_c(ũ_c) − Σ_r ln L_c[r,r] + log((1/√π)^q·Σ_j …) ].
    let mut dev = 0.0;
    let q_ln_sqrt_pi = q as f64 * ln_sqrt_pi;
    for c in 0..s {
        let lblk = &a_blocks[c * q * q..c * q * q + q * q];
        let mut log_scale = 0.0; // −Σ_r ln L_c[r,r] = −½ log|A_c|
        for r in 0..q {
            log_scale -= lblk[r * q + r].ln();
        }
        dev += -2.0 * (ctr[c] + log_scale + sum[c].ln() - q_ln_sqrt_pi);
    }
    dev
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
    super::workspace::fill_z_f64(&ws.groupings, x, &mut ws.z_buf, n);
    for v in ws.u.iter_mut() {
        *v = 0.0; // self-contained PIRLS seed (mode is point-determined; seed only shifts iterates)
    }
    let family = ws.family;
    let nb_theta = ws.nb_theta;
    let weighted = ws.weighted;
    let super::workspace::GlmmWorkspace {
        groupings,
        params: prm,
        beta_rhs,
        p,
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
        cluster_rows,
        ..
    } = ws;
    // Fixed-mode β copy (mirrors `laplace_deviance`): value-exact `params[n_theta..]`
    // → `beta_rhs`, the transient β buffer `agq_deviance` reads (NOT `betas`, which
    // is the fit's reported output).
    let nt = groupings.n_theta();
    beta_rhs[..*p].copy_from_slice(&prm[nt..nt + *p]);
    let kernel = if groupings.primary_q == 1 {
        agq_deviance
    } else {
        agq_deviance_vec
    };
    kernel(
        family,
        nb_theta,
        groupings,
        &prm[..],
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
        None,
        n,
        cluster_rows.as_ref(),
    )
}
