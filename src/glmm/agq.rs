//! Adaptive Gauss–Hermite quadrature (`nAGQ>1`) for the single scalar-intercept
//! binomial / Poisson GLMM — the only shape where the marginal likelihood is a
//! product of independent 1-D cluster integrals, so AGQ is a per-cluster k-node
//! sum with no curse of dimensionality. Replaces the Laplace `+log|A|` curvature
//! term with a Liu–Pierce (1994) adaptive GH sum centered at each cluster's PIRLS
//! mode/curvature.
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
///   ℓ_c(u)     = Σ_{i∈c} −½·dev_resid(family, y_i, g⁻¹(η_fix,i + λ·u)) − ½u²
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
    cluster_ids: &[u32],
    eta: &mut [f64],
    prob: &mut [f64],
    w: &mut [f64],
    u: &mut [f64],
    u_prev: &mut [f64],
    eta_fixed: &mut [f64],
    a_blocks: &mut [f64],
    a_rhs: &mut [f64],
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
    // AGQ (nagq>1) is gated closed for weighted fits (see the boundary-gate
    // capability map in `fit.rs`) — `weighted` is unconditionally `false` here;
    // `prior_w` still threads through so the callee's signature is uniform
    // across the three PIRLS variants.
    let (_dev, _pen, _logdet, conv) = pirls_solve_blocked(
        family,
        nb_theta,
        groupings,
        cluster_ids,
        x,
        y,
        prior_w,
        false,
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
        ctr[c] -= 0.5 * crate::family::dev_resid(family, nb_theta, y[i], prob[i]);
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
    // ln(w_j·e^{z_j²}) per node — the Liu–Pierce reweight, a function of the GH
    // table only. Filled once (k ≤ MAX_NAGQ, stack buffer — no alloc on the hot
    // path) so the cluster-outer arm reads it instead of recomputing per
    // (cluster, node): that recomputation cost s·k ln() calls per eval and was
    // the dominant cluster-outer overhead on many-tiny-cluster shapes. Same
    // operands, deterministic — bit-identical to the inline form.
    let mut ln_wj_buf = [0.0f64; crate::consts::MAX_NAGQ as usize];
    for (j, (&zj, &wj)) in nodes.iter().zip(wts).enumerate() {
        ln_wj_buf[j] = wj.ln() + zj * zj;
    }
    let ln_wj_ro: &[f64] = &ln_wj_buf[..k];
    match cluster_rows {
        // Cluster-outer: per cluster, same node order and (via the CSR's
        // ascending-row guarantee) same per-accumulator operand order as the
        // node-outer loop below — bit-identical by construction, so the frozen
        // glmer(nAGQ=k) goldens re-validate nothing. Rows stay cache-hot per
        // cluster instead of k full n-sweeps.
        Some(idx) => {
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
                        acc_c -= 0.5 * crate::family::dev_resid(family, nb_theta, y[i], mu);
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
                    acc[c] -= 0.5 * crate::family::dev_resid(family, nb_theta, y[i], mu);
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

/// Test-only: drive `agq_deviance` from a `&[f64]` params (mirrors
/// `deviance::glmm_laplace_deviance`). Forces the AGQ path at the given `nagq`
/// regardless of the production gate, so the k=1≡Laplace reduction can be
/// asserted directly against `glmm_laplace_deviance`.
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
        agq_scratch,
        cluster_rows,
        ..
    } = ws;
    // Fixed-mode β copy (mirrors `laplace_deviance`): value-exact `params[n_theta..]`
    // → `beta_rhs`, the transient β buffer `agq_deviance` reads (NOT `betas`, which
    // is the fit's reported output).
    let nt = groupings.n_theta();
    beta_rhs[..*p].copy_from_slice(&prm[nt..nt + *p]);
    agq_deviance(
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
        cluster_ids,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        a_blocks,
        a_rhs,
        agq_scratch,
        nagq,
        None,
        n,
        cluster_rows.as_ref(),
    )
}
