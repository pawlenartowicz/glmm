use faer::dyn_stack::MemBuffer;
use faer::{Mat, MatRef};

use super::pirls::{
    build_coupling_csr, pirls_solve, pirls_solve_blocked, pirls_solve_blocked_extras, BetaStep,
};
#[cfg(test)]
use super::workspace::fill_z_f64;
use super::workspace::{apply_lambda, build_packed_m, GlmmWorkspace, StructuredSchur};
use crate::lmm::LmmGroupings;
use crate::spec::Family;

/// Laplace deviance at (θ, β): rebuild M = ZΛ, solve the PIRLS conditional
/// modes, then return `d(y,ũ) + ‖ũ‖² + log|A|` (A = M'WM + I at ũ). The +I in A
/// is the same ridge the penalty `‖ũ‖²` carries — this is glmer's nAGQ=1 Laplace
/// objective. Convention: the `d(y,ũ)` term is the family `aic` (glmer's own
/// substitution), not the bare deviance — for binomial/Poisson `aic = D + const`
/// (same minimizer, kept as `D` for byte-identity), but Gamma profiles the
/// dispersion as `D/n` (`family::gamma_aic`), the sole route by which dispersion
/// shifts glmer's β̂/τ̂. Non-convergence / Cholesky failure ⇒ `f64::INFINITY` (the
/// module's failure surface, mirrors `lmm::reml_deviance`). `pirls_solve` returns
/// `log|A|` off its converged factor, so there is no re-factor here.
/// The blocked branch requires `z_buf` pre-filled for this fit's `x`
/// (`fill_z_f64`); the dense branch ignores `z_buf`/`m_buf`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn laplace_deviance(
    family: Family,
    nb_theta: f64,
    nagq: u8,
    groupings: &LmmGroupings,
    params: &[f64],
    beta: &mut [f64],
    z: MatRef<f64>,
    m: &mut Mat<f64>,
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
    mu: &mut [f64],
    wm: &mut Mat<f64>,
    wx: &mut Mat<f64>,
    a: &mut Mat<f64>,
    // Copy-then-factor target for `a`'s Cholesky (ws.a_chol) — dense-fallback
    // branch only (`pirls_solve`); inert on the blocked/structured branches.
    a_chol: &mut Mat<f64>,
    // Persistent scratch for `a_chol`'s in-place `cholesky_in_place` (ws.a_llt_mem) —
    // dense-fallback branch only (`pirls_solve`); inert on the blocked/structured
    // branches, which never factor `a`.
    a_llt_mem: &mut MemBuffer,
    a_rhs: &mut [f64],
    a_blocks: &mut [f64],
    core_blocks: &mut [f64],
    coupling: &mut [f64],
    schur_blk: &mut [f64],
    m_core_buf: &mut [f64],
    cross_val: &mut [f64],
    cross_col: &mut [u32],
    n_cross: &mut [u8],
    coup_cols: &mut [u32],
    coup_ptr: &mut [u32],
    coup_mask: &mut Option<u32>,
    structured_schur: Option<&mut StructuredSchur>,
    force_dense_schur: bool,
    agq_scratch: &mut [f64],
    // Profile-mode (β-profiling / stage-1) scratch — the β-Schur border buffers
    // each PIRLS variant's Profile δβ step reads (mirrors `dense/blocked/structured
    // _schur_fill` in se.rs). All inert when `profile_beta == false`. `beta_step_rhs`
    // is the caller-owned δβ RHS/solution scratch (BetaStep::Profile.beta_rhs) and
    // MUST be a distinct buffer from `beta` — Fixed callers pass `ws.beta_prof` here
    // (spare) and `ws.beta_rhs` as `beta`; the Profile caller passes `ws.beta_rhs`
    // here and `ws.beta_prof` as `beta`.
    xtwx: &mut Mat<f64>,
    xtwm: &mut Mat<f64>,
    ainv_mtwx: &mut Mat<f64>,
    schur: &mut Mat<f64>,
    // Persistent scratch for `schur`'s in-place `cholesky_in_place` (ws.schur_llt_mem),
    // threaded into `BetaStep::Profile` — inert under `BetaStep::Fixed`.
    schur_llt_mem: &mut MemBuffer,
    beta_step_rhs: &mut [f64],
    beta_prev: &mut [f64],
    // β mode: `true` builds `BetaStep::Profile` (joint (u,β) PQL step, converged β
    // written back through `beta`); `false` builds `BetaStep::Fixed` (β held at the
    // caller's input — the FD-Hessian and stage-2 contract). No default: every call
    // site chooses.
    profile_beta: bool,
    // PIRLS exit-tol override, forwarded verbatim to whichever PIRLS variant (or
    // `agq_deviance`) runs. `Some(PIRLS_TOL_REL_FD)` only under the FD-Hessian SE
    // evals (`ws.pirls_tol_override`, set by `fd_hessian_cov`); `None` on the fit
    // path, which therefore stays bit-identical.
    pirls_tol_override: Option<f64>,
    p: usize,
    n: usize,
    // Cluster-outer AGQ substrate (`agq::ClusterRowIndex`), forwarded verbatim to
    // `agq_deviance`'s early return below; `None` on every non-AGQ path (unread).
    cluster_rows: Option<&super::agq::ClusterRowIndex>,
    // Per-row linear-predictor offset (`FitOptions::offset`), forwarded to every
    // PIRLS/AGQ variant's `eta_fixed` fill. `None` ⇒ no offset.
    offset: Option<&[f64]>,
) -> f64 {
    let n_theta = groupings.n_theta();
    // Fixed-mode β: a value-exact copy of `params[n_theta..n_theta+p]` into the
    // caller's β buffer. β is never sliced out of `params` below — the PIRLS
    // variants (and `agq_deviance`) read it from `beta`, and every call is
    // `BetaStep::Fixed`, so β is read-only and this stays bit-identical to the
    // pre-plumbing path. The Fixed-mode callers pass `ws.beta_rhs` (a transient
    // scratch), NOT `ws.betas`: `betas` is the fit's reported β output, and the
    // FD-Hessian SE path re-evals this fn many times — clobbering `betas` would
    // corrupt the reported coefficients. In Profile mode `beta` is the caller's β
    // in/out state (the stage-1 warm-start buffer, `ws.beta_prof`): it must NOT be
    // reseeded from params — the joint (u,β) step drives it, so this copy is gated
    // `!profile_beta`.
    if !profile_beta {
        beta[..p].copy_from_slice(&params[n_theta..n_theta + p]);
    }
    // Profile mode is only defined on the PIRLS path below. The nAGQ>1 early-return
    // bypasses PIRLS entirely, so Profile there is undefined; the two-stage
    // optimizer's stage-1 gating routes around it (`two_stage && nagq == 1`).
    debug_assert!(!profile_beta || nagq == 1);
    // AGQ (nagq>1) only on a single grouping factor (no extras), q_p ≤ 3,
    // binomial/Poisson — the shapes where the marginal likelihood factorizes into
    // independent per-cluster q-D integrals (a k^q product quadrature). Route by
    // q_p: scalar (q_p==1) → agq_deviance (verbatim, frozen goldens), vector
    // (q_p∈2..=3) → agq_deviance_vec. Every other shape (and nagq==1) uses the
    // Laplace path below unchanged (nagq==1 IS Laplace, so it is bit-identical).
    if nagq > 1
        && groupings.extra_offsets.is_empty()
        && (1..=3).contains(&groupings.primary_q)
        && matches!(family, Family::Binomial { .. } | Family::Poisson { .. })
    {
        let kernel = if groupings.primary_q == 1 {
            super::agq::agq_deviance
        } else {
            super::agq::agq_deviance_vec
        };
        return kernel(
            family,
            nb_theta,
            groupings,
            params,
            beta,
            lam,
            z_buf,
            m_buf,
            x,
            y,
            prior_w,
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
            pirls_tol_override,
            n,
            cluster_rows,
            offset,
        );
    }
    let k = groupings.k_total;
    // One BetaStep, moved into whichever PIRLS branch runs (the branches are
    // mutually exclusive). Fixed leaves the border buffers untouched.
    let beta_step = if profile_beta {
        BetaStep::Profile {
            xtwx,
            xtwm,
            ainv_mtwx,
            schur,
            beta_rhs: beta_step_rhs,
            beta_prev,
            schur_llt_mem,
        }
    } else {
        BetaStep::Fixed
    };
    let (dev, pen, logdet, conv) = if groupings.extra_offsets.is_empty() {
        // No extras ⇒ A is block-diagonal: reconstruct mᵢ per row, never build Z/M.
        crate::lmm::primary_lambda(&params[..n_theta], groupings.primary_q, lam);
        pirls_solve_blocked(
            family,
            nb_theta,
            groupings,
            cluster_ids,
            x,
            y,
            prior_w,
            weighted,
            beta,
            beta_step,
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
            offset,
            pirls_tol_override,
            n,
        )
    } else if groupings.structured_extras_eligible() {
        // Intercept-only crossed/nested ⇒ block-diagonal core + Schur on the
        // crossed width. The M = ZΛ nonzeros are packed once here (core slice +
        // crossed entries) instead of materializing the dense n×k M every eval; the
        // structured passes read the packed buffers. `z`/`m` are untouched on this
        // path now (the dense `m` only feeds the genuinely-dense fallback below).
        build_packed_m(
            groupings,
            params,
            z,
            lam,
            cluster_ids,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            n,
        );
        // CSR cache: pattern = f(design, pinning mask). Rebuild only when the set
        // of θ-pinned crossed groupings changes (see build_coupling_csr's contract).
        debug_assert!(groupings.crossed.len() <= 32);
        let mut pin_mask: u32 = 0;
        for (gi, cf) in groupings.crossed.iter().enumerate() {
            if params[cf.vech_start] == 0.0 {
                pin_mask |= 1 << gi;
            }
        }
        if *coup_mask != Some(pin_mask) {
            build_coupling_csr(
                cluster_ids,
                cross_col,
                n_cross,
                groupings.n_primary,
                n,
                coup_cols,
                coup_ptr,
            );
            *coup_mask = Some(pin_mask);
        }
        pirls_solve_blocked_extras(
            family,
            nb_theta,
            groupings,
            cluster_ids,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            x,
            y,
            prior_w,
            weighted,
            beta,
            beta_step,
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            mu,
            core_blocks,
            coupling,
            schur_blk,
            coup_cols,
            coup_ptr,
            structured_schur,
            force_dense_schur,
            a_rhs,
            wx,
            offset,
            pirls_tol_override,
            n,
        )
    } else {
        // Non-eligible extras (oversized core) ⇒ A genuinely dense: dense fallback.
        apply_lambda(groupings, params, z, m, lam, n);
        pirls_solve(
            family,
            nb_theta,
            k,
            p,
            m.as_ref(),
            x,
            y,
            prior_w,
            weighted,
            beta,
            beta_step,
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            mu,
            wm,
            wx,
            a,
            a_chol,
            a_rhs,
            a_llt_mem,
            offset,
            pirls_tol_override,
            n,
        )
    };
    if !conv || !dev.is_finite() {
        return f64::INFINITY;
    }
    // glmer substitutes the family `aic` for the bare deviance in the Laplace
    // objective. For binomial/Poisson `aic = D + const` (same minimizer — kept as
    // `dev` for byte-identity), but Gamma's `aic` profiles the dispersion as `D/n`,
    // making it a nonlinear function of `D` — the sole route by which the dispersion
    // shifts glmer's β̂/τ̂ (see `family::gamma_aic`). `prob` holds μ̂ at the mode.
    let data_term = if matches!(family, Family::Gamma { .. }) {
        crate::family::gamma_aic(y, prob, dev, n, Some(prior_w))
    } else {
        dev
    };
    data_term + pen + 2.0 * logdet
}

/// Evaluate the joint Laplace deviance at the params CURRENTLY in `ws.params`
/// (the FD loop in `fd_hessian_cov` writes them before each call). Borrow-split
/// twin of `fit_glmm`'s BOBYQA-closure body: destructures the workspace into the
/// disjoint borrows `laplace_deviance` needs (z read; m/lam/a/etc. written) and
/// calls it. Seeds the PIRLS conditional modes from û = 0 each call — UNLESS
/// `ws.warm_seed_active`, in which case it seeds from the fixed shared `ws.u_seed`
/// (the fitted mode û(γ̂), set by `fd_hessian_cov`). Either way the seed is a
/// constant independent of evaluation order, so each `f(γ)` depends only on γ and
/// the FD second differences stay valid — only a *chained* seed (eval k from eval
/// k−1's mode) would make `f(γ)` order-dependent and corrupt them.
///
/// Caller must have filled `ws.z_buf` for this fit's `x` (blocked path) — `x` is
/// constant across all FD perturbations, so fill it ONCE before the FD loop, not
/// per eval (`fd_hessian_cov` does; `glmm_laplace_deviance` does it inline).
pub(crate) fn laplace_deviance_at(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    n: usize,
) -> f64 {
    let kk = ws.k.max(1);
    if ws.warm_seed_active {
        ws.u[..kk].copy_from_slice(&ws.u_seed[..kk]);
    } else {
        for v in ws.u[..kk].iter_mut() {
            *v = 0.0;
        }
    }
    // Fixed mode: β = ws.beta_rhs (transient scratch, never ws.betas — see
    // laplace_deviance's doc). `beta_step_rhs` just needs a distinct spare
    // buffer (inert under Fixed) — ws.beta_prof is it.
    laplace_deviance_ws(ws, x, y, cluster_ids, n, false)
}

/// Shared borrow-split body of `laplace_deviance_at` and (test-only)
/// `glmm_laplace_deviance_profile`: destructures the workspace into the
/// disjoint borrows `laplace_deviance` needs and calls it. `profile_beta`
/// selects both `laplace_deviance`'s β mode AND which workspace buffer plays
/// β vs. the spare `beta_step_rhs` (Fixed: β = `beta_rhs`, spare = `beta_prof`;
/// Profile: β = `beta_prof`, spare = `beta_rhs` — the two must never alias).
/// Callers own all u/β seeding — this helper seeds nothing.
fn laplace_deviance_ws(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    n: usize,
    profile_beta: bool,
) -> f64 {
    let family = ws.family;
    let nb_theta = ws.nb_theta;
    let nagq = ws.nagq;
    let force_dense_schur = ws.force_dense_schur;
    let pirls_tol_override = ws.pirls_tol_override;
    let weighted = ws.weighted;
    let offset = ws.offset.as_deref();
    let GlmmWorkspace {
        groupings,
        params: prm,
        beta_rhs,
        p,
        z,
        m,
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
        mu,
        wm,
        wx,
        a,
        a_chol,
        a_llt_mem,
        a_rhs,
        a_blocks,
        core_blocks,
        coupling,
        schur_blk,
        m_core_buf,
        cross_val,
        cross_col,
        n_cross,
        coup_cols,
        coup_ptr,
        coup_mask,
        structured_schur,
        agq_scratch,
        cluster_rows,
        xtwx,
        xtwm,
        ainv_mtwx,
        schur,
        schur_llt_mem,
        beta_prof,
        beta_prev,
        ..
    } = ws;
    let (beta, beta_step_rhs): (&mut [f64], &mut [f64]) = if profile_beta {
        (beta_prof, beta_rhs)
    } else {
        (beta_rhs, beta_prof)
    };
    laplace_deviance(
        family,
        nb_theta,
        nagq,
        groupings,
        &prm[..],
        beta,
        z.as_ref(),
        m,
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
        mu,
        wm,
        wx,
        a,
        a_chol,
        a_llt_mem,
        a_rhs,
        a_blocks,
        core_blocks,
        coupling,
        schur_blk,
        m_core_buf,
        cross_val,
        cross_col,
        n_cross,
        coup_cols,
        coup_ptr,
        coup_mask,
        structured_schur.as_mut(),
        force_dense_schur,
        agq_scratch,
        xtwx,
        xtwm,
        ainv_mtwx,
        schur,
        schur_llt_mem,
        beta_step_rhs,
        beta_prev,
        profile_beta,
        pirls_tol_override,
        *p,
        n,
        cluster_rows.as_ref(),
        offset,
    )
}

/// Workspace-bound wrapper for `laplace_deviance`: copies `params` into the
/// workspace, fills `z_buf`, then delegates to the shared `laplace_deviance_at`.
/// Test-only entry point — the production fit (`fit_glmm`) destructures the
/// workspace and calls `laplace_deviance` directly (the BOBYQA closure and the
/// pinned-γ̂ re-eval both inline it), so this exists purely to drive the deviance
/// from a `&[f64]` in tests.
#[cfg(test)]
pub(crate) fn glmm_laplace_deviance(
    params: &[f64],
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    n: usize,
) -> f64 {
    ws.params[..params.len()].copy_from_slice(params);
    fill_z_f64(&ws.groupings, x, &mut ws.z_buf, n);
    laplace_deviance_at(ws, x, y, cluster_ids, n)
}

/// Test-only Profile twin of `glmm_laplace_deviance`: drives `laplace_deviance`
/// with `profile_beta = true` and `beta = ws.beta_prof` (the stage-1 in/out β),
/// so it evaluates the PQL objective at θ and leaves the profiled β̂(θ) in
/// `ws.beta_prof`. Seeds BOTH the conditional mode (`ws.u`) and `beta_prof` at 0
/// each call, making the result depend only on `params` — the determinism (BOBYQA
/// objective-consistency) the two-stage optimizer needs. This is the stage-1
/// production call shape (`laplace_deviance(profile_beta = true, beta =
/// &mut ws.beta_prof, …)`) exercised from a `&[f64]` in tests.
#[cfg(test)]
pub(crate) fn glmm_laplace_deviance_profile(
    params: &[f64],
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    n: usize,
) -> f64 {
    ws.params[..params.len()].copy_from_slice(params);
    fill_z_f64(&ws.groupings, x, &mut ws.z_buf, n);
    let kk = ws.k.max(1);
    for v in ws.u[..kk].iter_mut() {
        *v = 0.0;
    }
    for v in ws.beta_prof.iter_mut() {
        *v = 0.0;
    }
    laplace_deviance_ws(ws, x, y, cluster_ids, n, true)
}
