use faer::{Mat, MatRef};

use super::pirls::{
    build_coupling_csr, fill_m_vals, pirls_solve_blocked, pirls_solve_blocked_extras,
    pirls_solve_packed, BetaMode, BetaStep, DualStep, TailKernel,
};
#[cfg(test)]
use super::workspace::fill_z_f64;
use super::workspace::{
    build_packed_m, BorderScratch, FitData, GlmmLayout, GlmmWorkspace, PackedScratch, PirlsScratch,
    StructuredPattern, StructuredScratch,
};
use crate::lmm::LmmGroupings;
use crate::scalar::Scalar;
use crate::spec::Family;

/// The blocked (`extra_offsets` empty) Laplace objective at (θ, β), generic
/// over the scalar: build Λ_p, solve the blocked PIRLS conditional modes, and
/// return `d(y,ũ) + ‖ũ‖² + log|A|` — the same three terms `laplace_deviance`
/// assembles, extracted so a non-f64 scalar has an entry point that does not
/// carry the packed and structured branches' buffers.
/// Non-convergence / Cholesky failure ⇒ `+∞`. The third element of the
/// return is the raw PIRLS deviance's finiteness, read before the `+∞` fold,
/// so the router's `pirls_exhausted` counter can keep telling an
/// iteration-cap exhaustion (finite raw deviance) apart from a hard failure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn blocked_laplace_deviance<T: Scalar>(
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
    wx: &mut Mat<f64>,
    beta_step: BetaStep,
    offset: Option<&[f64]>,
    pirls_tol_override: Option<f64>,
    // Unused on the blocked path (`pirls_solve_blocked` reads `p` off
    // `beta.len()`) — kept so the signature matches the packed/structured arms'
    // shape.
    _p: usize,
    n: usize,
    counters: &mut crate::counters::EvalCounters,
) -> (T, bool, bool) {
    crate::lmm::primary_lambda(
        &params[..groupings.n_theta()],
        groupings.primary_q,
        &mut scratch.lam,
    );
    let (dev, pen, logdet, conv) = pirls_solve_blocked(
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
        scratch,
        z_buf,
        dual,
        wx,
        offset,
        pirls_tol_override,
        n,
        counters,
    );
    let raw_dev_finite = dev.value().is_finite();
    if !conv || !raw_dev_finite {
        return (T::from_f64(f64::INFINITY), conv, raw_dev_finite);
    }
    // Gamma substitutes its AIC-style objective for the bare deviance —
    // rationale in `laplace_deviance`'s doc comment (`family::gamma_aic`);
    // mirrors that branch, change together.
    let data_term = if matches!(family, Family::Gamma { .. }) {
        crate::family::gamma_aic(y, &scratch.prob, dev, n, Some(prior_w))
    } else {
        dev
    };
    (
        data_term + pen + T::from_f64(2.0) * logdet,
        conv,
        raw_dev_finite,
    )
}

/// The structured (`extra_offsets` non-empty, `structured_extras_eligible()`)
/// Laplace objective at (θ, β), generic over the scalar: pack `M = ZΛ`'s
/// nonzeros, refresh the coupling CSR when the pin mask changed, solve the
/// block+Schur PIRLS conditional modes, and return `d(y,ũ) + ‖ũ‖² + log|A|` —
/// the same three terms `laplace_deviance` assembles on this arm.
/// Non-convergence / non-PD core or Schur ⇒ `+∞`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn structured_laplace_deviance<T: TailKernel>(
    family: Family,
    nb_theta: f64,
    groupings: &LmmGroupings,
    params: &[T],
    z_buf: &[f64],
    extra_ids: &[Vec<u32>],
    cluster_ids: &[u32],
    scratch: &mut PirlsScratch<T>,
    structured: &mut StructuredScratch<T>,
    pattern: &mut StructuredPattern,
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    beta: &mut [T],
    beta_step: BetaStep,
    dual: Option<&mut DualStep<T>>,
    wx: &mut Mat<f64>,
    offset: Option<&[f64]>,
    pirls_tol_override: Option<f64>,
    n: usize,
    counters: &mut crate::counters::EvalCounters,
) -> (T, bool, bool) {
    // Intercept-only crossed/nested ⇒ block-diagonal core + Schur on the
    // crossed width. The M = ZΛ nonzeros are packed once here (core slice +
    // crossed entries) instead of materializing a dense n×k M every eval; the
    // structured passes read the packed buffers.
    build_packed_m(
        groupings,
        params,
        z_buf,
        extra_ids,
        &mut scratch.lam,
        cluster_ids,
        &mut structured.m_core_buf,
        &mut structured.cross_val,
        &mut pattern.cross_col,
        &mut pattern.n_cross,
        n,
    );
    // CSR cache: pattern = f(design, pinning mask). Rebuild only when the set
    // of θ-pinned crossed groupings changes (see build_coupling_csr's contract).
    debug_assert!(groupings.crossed.len() <= 32);
    let mut pin_mask: u32 = 0;
    for (gi, cf) in groupings.crossed.iter().enumerate() {
        // Mirrors `build_packed_m`'s pin skip — change together: both are
        // `f64`-only, so an `f64` call at a pinned θ̂ keys the cache on the
        // narrow mask and a dual call keys it on 0 (= full pattern). The two
        // keys differ exactly where the two patterns differ, which is what
        // makes the cache correct across the alternating calls inside one
        // derivative request. Dropping the `T::IS_F64` here while keeping it
        // there is a silent wrong answer: the CSR would not be rebuilt for the
        // widened pattern, so the retained column would never be read back.
        if T::IS_F64 && params[cf.vech_start].value() == 0.0 {
            pin_mask |= 1 << gi;
        }
    }
    if pattern.coup_mask != Some(pin_mask) {
        build_coupling_csr(
            cluster_ids,
            &pattern.cross_col,
            &pattern.n_cross,
            groupings.n_primary,
            n,
            &mut pattern.coup_cols,
            &mut pattern.coup_ptr,
        );
        pattern.coup_mask = Some(pin_mask);
    }
    let (dev, pen, logdet, conv) = pirls_solve_blocked_extras(
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
        scratch,
        structured,
        pattern,
        dual,
        wx,
        offset,
        pirls_tol_override,
        n,
        counters,
    );
    let raw_dev_finite = dev.value().is_finite();
    if !conv || !raw_dev_finite {
        return (T::from_f64(f64::INFINITY), conv, raw_dev_finite);
    }
    // Gamma substitutes its AIC-style objective for the bare deviance —
    // rationale in `laplace_deviance`'s doc comment (`family::gamma_aic`);
    // mirrors that branch, change together.
    let data_term = if matches!(family, Family::Gamma { .. }) {
        crate::family::gamma_aic(y, &scratch.prob, dev, n, Some(prior_w))
    } else {
        dev
    };
    (
        data_term + pen + T::from_f64(2.0) * logdet,
        conv,
        raw_dev_finite,
    )
}

/// Laplace deviance at (θ, β): rebuild M = ZΛ, solve the PIRLS conditional
/// modes, then return `d(y,ũ) + ‖ũ‖² + log|A|` (A = M'WM + I at ũ). The +I in A
/// is the same ridge the penalty `‖ũ‖²` carries — this is glmer's nAGQ=1 Laplace
/// objective. Convention: the `d(y,ũ)` term is the family `aic` (glmer's own
/// substitution), not the bare deviance — for binomial/Poisson `aic = D + const`
/// (same minimizer, kept as `D` for byte-identity), but Gamma profiles the
/// dispersion as `D/n` (`family::gamma_aic`), the sole route by which dispersion
/// shifts glmer's β̂/τ̂. Non-convergence / Cholesky failure ⇒ `f64::INFINITY` (the
/// module's failure surface, mirrors `lmm::reml_deviance`). Every PIRLS variant
/// returns `log|L|` off its converged factor (L the Cholesky factor of A, so
/// `log|A| = 2 log|L|`) — the caller below doubles it to get `log|A|`, so there
/// is no re-factor here.
/// The blocked AND structured branches require `data.z_buf` pre-filled for
/// this fit's `x` (`fill_z_f64`) — `build_packed_m`'s primary-core reduction
/// reads it the same way `pirls_solve_blocked`'s does; the packed branch reads
/// `x` directly through `fill_m_vals`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn laplace_deviance(
    data: &FitData,
    nb_theta: f64,
    nagq: u8,
    params: &[f64],
    beta: &mut [f64],
    wx: &mut Mat<f64>,
    // Row- and RE-sized PIRLS scratch every route writes — see [`PirlsScratch`].
    pirls: &mut PirlsScratch<f64>,
    // Structured crossed/nested route scratch — see [`StructuredScratch`].
    structured: &mut StructuredScratch<f64>,
    // Structured route's θ-independent index pattern — see [`StructuredPattern`].
    pattern: &mut StructuredPattern,
    // Packed-row layout scratch — see [`PackedScratch`]. Zero-length on the
    // blocked/structured branches, which never read it.
    packed: &mut PackedScratch,
    // Profile-mode (β-profiling / stage-1) scratch — the β-Schur border buffers
    // each PIRLS variant's Profile δβ step reads (mirrors `packed/blocked/structured
    // _schur_fill` in se.rs). All inert when `beta_mode == BetaMode::Fixed`.
    border: &mut BorderScratch,
    // The caller-owned δβ RHS/solution scratch (BetaStep::Profile.beta_rhs) and
    // MUST be a distinct buffer from `beta` — Fixed callers pass `ws.beta_prof` here
    // (spare) and `ws.beta_rhs` as `beta`; the Profile caller passes `ws.beta_rhs`
    // here and `ws.beta_prof` as `beta`.
    beta_step_rhs: &mut [f64],
    // Exact-profile scratch (`pirls::ExactProfileBufs`), threaded into
    // `BetaStep::Profile { exact: .. }` only under `BetaMode::ProfileExact` — inert
    // (unread) under `Fixed`/`ProfilePql`.
    exact_prof: &mut super::pirls::ExactProfileBufs,
    // β mode: `ProfilePql`/`ProfileExact` build `BetaStep::Profile` (joint (u,β) step,
    // converged β written back through `beta`); `Fixed` builds `BetaStep::Fixed` (β
    // held at the caller's input — the FD-Hessian and stage-2 contract). No default:
    // every call site chooses.
    beta_mode: BetaMode,
    // PIRLS exit-tol override, forwarded verbatim to whichever PIRLS variant (or
    // `agq_deviance`) runs. `Some(pirls_tol_fd(family))` only under the FD-Hessian
    // SE evals (`ws.fd.pirls_tol_override`, set by `joint_hessian_cov`); `None` on the fit
    // path, which therefore stays bit-identical.
    pirls_tol_override: Option<f64>,
    // Cluster-outer AGQ substrate (`agq::ClusterRowIndex`), forwarded verbatim to
    // `agq_deviance`'s early return below; `None` on every non-AGQ path (unread).
    cluster_rows: Option<&super::agq::ClusterRowIndex>,
    // Observation-only: incremented below when a fit-path PIRLS solve (the
    // Laplace branch only — AGQ's own per-cluster PIRLS, above, is not
    // instrumented) runs the full iteration cap without converging. Never read
    // by anything on the numeric path; see `Note::PirlsExhausted`.
    pirls_exhausted: &mut u32,
    // Observation-only optimizer counters (`counters` feature; a zero-sized
    // no-op otherwise). The FD-Hessian SE path passes its own discard value,
    // which is how SE evals stay out of every count.
    counters: &mut crate::counters::EvalCounters,
) -> f64 {
    let FitData {
        family,
        groupings,
        layout,
        x,
        y,
        prior_w,
        weighted,
        cluster_ids,
        extra_ids,
        z_buf,
        offset,
        n,
        p,
    } = *data;
    let BorderScratch {
        xtwx,
        xtwm,
        ainv_mtwx,
        schur,
        schur_llt_mem,
        beta_prev,
    } = border;
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
    // `beta_mode == BetaMode::Fixed`.
    if beta_mode == BetaMode::Fixed {
        beta[..p].copy_from_slice(&params[n_theta..n_theta + p]);
    }
    // Profile mode is only defined on the PIRLS path below. The nAGQ>1 early-return
    // bypasses PIRLS entirely, so Profile there is undefined; the driver's
    // `stage1_mode.filter(|_| nagq == 1)` gate routes around it.
    debug_assert!(beta_mode == BetaMode::Fixed || nagq == 1);
    // AGQ (nagq>1) only on a single grouping factor (no extras), q_p ≤ 3,
    // binomial/Poisson — the shapes where the marginal likelihood factorizes into
    // independent per-cluster q-D integrals (a k^q product quadrature). The
    // family/nagq/q_p terms are `derivative::agq_eligible`, shared with the
    // derivative path. Route by
    // q_p: scalar (q_p==1) → agq_deviance (verbatim, frozen goldens), vector
    // (q_p∈2..=3) → agq_deviance_vec. Every other shape (and nagq==1) uses the
    // Laplace path below unchanged (nagq==1 IS Laplace, so it is bit-identical).
    if super::derivative::agq_eligible(family, nagq, groupings.primary_q)
        && groupings.extra_offsets.is_empty()
    {
        let kernel = if groupings.primary_q == 1 {
            super::agq::agq_deviance
        } else {
            super::agq::agq_deviance_vec
        };
        // AGQ cost per evaluation: one q-dimensional product grid per cluster,
        // `nagq^q` nodes each (`agq_deviance_vec`'s `kq`; `nagq` itself in the
        // scalar q == 1 kernel). Recorded here rather than inside the node loop
        // so the hot loop is untouched.
        counters.record_agq_eval(
            groupings.n_primary as u64 * (nagq as u64).pow(groupings.primary_q as u32),
        );
        let dev = kernel(
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
            None,
            wx,
            nagq,
            pirls_tol_override,
            n,
            cluster_rows,
            offset,
            counters,
        );
        counters.commit_pirls_iters();
        return dev;
    }
    let k = groupings.k_total;
    // One BetaStep, moved into whichever PIRLS branch runs (the branches are
    // mutually exclusive). Fixed leaves the border buffers untouched.
    let beta_step = match beta_mode {
        BetaMode::Fixed => BetaStep::Fixed,
        BetaMode::ProfilePql | BetaMode::ProfileExact => BetaStep::Profile {
            exact: (beta_mode == BetaMode::ProfileExact).then_some(exact_prof),
            xtwx,
            xtwm,
            ainv_mtwx,
            schur,
            beta_rhs: beta_step_rhs,
            beta_prev,
            schur_llt_mem,
        },
    };
    if layout == GlmmLayout::Blocked {
        // No extras ⇒ A is block-diagonal: reconstruct mᵢ per row, never build M.
        let (obj, conv, raw_dev_finite) = blocked_laplace_deviance(
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
            None,
            wx,
            beta_step,
            offset,
            pirls_tol_override,
            p,
            n,
            counters,
        );
        if !conv && raw_dev_finite && pirls_tol_override.is_none() {
            *pirls_exhausted += 1;
        }
        counters.commit_pirls_iters();
        return obj;
    }
    if layout == GlmmLayout::Structured {
        let (obj, conv, raw_dev_finite) = structured_laplace_deviance(
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
            beta_step,
            None,
            wx,
            offset,
            pirls_tol_override,
            n,
            counters,
        );
        // Same iteration-cap-exhaustion discriminator as the shared tail below
        // (see its comment): `raw_dev_finite` alongside `!conv` is the natural
        // exhaustion case, not a hard (NaN) failure.
        if !conv && raw_dev_finite && pirls_tol_override.is_none() {
            *pirls_exhausted += 1;
        }
        counters.commit_pirls_iters();
        return obj;
    }
    // The packed-row layout: Λ and the packed `M` values at this θ, then the
    // dense `k×k` PIRLS over the fixed-width rows.
    crate::sparse::fill_lambda_small(&params[..n_theta], groupings, &mut packed.lam_small);
    fill_m_vals(packed, groupings, x, n);
    // Fit-path evals (`pirls_tol_override == None`) carry the previous call's
    // converged û forward as the starting point — fewer iterations to
    // reconverge, same fixed point. Tight-tol evals (the FD-Hessian stencil)
    // cold-seed û = 0, overriding the `u_seed` that `laplace_deviance_at` copied
    // in. Both seeds are constant over the grid, so either makes every cell a
    // pure function of `(γ̂, steps, design)` whatever order it runs in; the cold
    // one is the one whose second differences match the assembled Hessian.
    // Measured on `sim_sparse_gamma` (Gamma-log), diagonal entry of the first β
    // coordinate: assembled 60.971162, stencil from û = 0 60.972707, stencil
    // from `u_seed` −94.968163, which is indefinite and costs the fit its
    // Hessian SE. The cold f0 there is 3427.047583 against the fit's 3427.047586.
    if pirls_tol_override.is_some() {
        pirls.u.fill(0.0);
    }
    let (dev, pen, logdet, conv) = pirls_solve_packed(
        family,
        nb_theta,
        k,
        p,
        x,
        y,
        prior_w,
        weighted,
        beta,
        beta_step,
        pirls,
        packed,
        wx,
        offset,
        pirls_tol_override,
        n,
        counters,
    );
    // `!conv` with a FINITE `dev` is exactly the natural iteration-cap
    // exhaustion: the three PIRLS variants return `(NaN, NaN, NaN, false)` from
    // every OTHER failure path (halving exhausted, non-PD Cholesky), so a
    // finite `dev` alongside `!conv` can only mean the `for` loop ran out its
    // `PIRLS_MAX_ITERS` iterations still inside a rising/falling accepted
    // sequence. Gated on `pirls_tol_override.is_none()` — the FD-Hessian SE
    // evals run their own tight tolerance and must not count — the same
    // fit-path-vs-FD-eval discriminator the cold seed above uses.
    if !conv && dev.is_finite() && pirls_tol_override.is_none() {
        *pirls_exhausted += 1;
    }
    counters.commit_pirls_iters();
    if !conv || !dev.is_finite() {
        return f64::INFINITY;
    }
    // glmer substitutes the family `aic` for the bare deviance in the Laplace
    // objective. For binomial/Poisson `aic = D + const` (same minimizer — kept as
    // `dev` for byte-identity), but Gamma's `aic` profiles the dispersion as `D/n`,
    // making it a nonlinear function of `D` — the sole route by which the dispersion
    // shifts glmer's β̂/τ̂ (see `family::gamma_aic`). `prob` holds μ̂ at the mode.
    let data_term = if matches!(family, Family::Gamma { .. }) {
        crate::family::gamma_aic(y, &pirls.prob, dev, n, Some(prior_w))
    } else {
        dev
    };
    data_term + pen + 2.0 * logdet
}

/// Evaluate the joint Laplace deviance at the params CURRENTLY in `ws.params`
/// (the FD loop in `joint_hessian_cov` writes them before each call). Borrow-split
/// twin of `fit_glmm`'s BOBYQA-closure body: destructures the workspace into the
/// disjoint borrows `laplace_deviance` needs and calls it. Seeds the PIRLS conditional modes from û = 0 each call — UNLESS
/// `ws.fd.warm_seed_active`, in which case it seeds from the fixed shared `ws.u_seed`
/// (the fitted mode û(γ̂), set by `joint_hessian_cov`). Same fixed-seed FD-derivative
/// invariant as `joint_hessian_cov` in se.rs — see there for the derivation.
///
/// Caller must have filled `ws.z_buf` for this fit's `x` (blocked path) — `x` is
/// constant across all FD perturbations, so fill it ONCE before the FD loop, not
/// per eval (`joint_hessian_cov` does; `glmm_laplace_deviance` does it inline).
pub(crate) fn laplace_deviance_at(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
    counters: &mut crate::counters::EvalCounters,
) -> f64 {
    let kk = ws.k.max(1);
    if ws.fd.warm_seed_active {
        ws.pirls.u[..kk].copy_from_slice(&ws.u_seed[..kk]);
    } else {
        for v in ws.pirls.u[..kk].iter_mut() {
            *v = 0.0;
        }
    }
    // Fixed mode: β = ws.beta_rhs (transient scratch, never ws.betas — see
    // laplace_deviance's doc). `beta_step_rhs` just needs a distinct spare
    // buffer (inert under Fixed) — ws.beta_prof is it.
    laplace_deviance_ws(
        ws,
        x,
        y,
        cluster_ids,
        extra_ids,
        n,
        BetaMode::Fixed,
        counters,
    )
}

/// Shared borrow-split body of `laplace_deviance_at` and (test-only)
/// `glmm_laplace_deviance_profile`: destructures the workspace into the
/// disjoint borrows `laplace_deviance` needs and calls it. `beta_mode`
/// selects both `laplace_deviance`'s β mode AND which workspace buffer plays
/// β vs. the spare `beta_step_rhs` (Fixed: β = `beta_rhs`, spare = `beta_prof`;
/// Profile: β = `beta_prof`, spare = `beta_rhs` — the two must never alias).
/// Callers own all u/β seeding — this helper seeds nothing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn laplace_deviance_ws(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
    beta_mode: BetaMode,
    counters: &mut crate::counters::EvalCounters,
) -> f64 {
    let family = ws.family;
    let nb_theta = ws.nb_theta;
    let nagq = ws.nagq;
    let pirls_tol_override = ws.fd.pirls_tol_override;
    let weighted = ws.weighted;
    let offset = ws.offset.as_deref();
    let GlmmWorkspace {
        groupings,
        layout,
        params: prm,
        beta_rhs,
        p,
        packed,
        z_buf,
        prior_w,
        pirls,
        wx,
        structured,
        pattern,
        cluster_rows,
        border,
        beta_prof,
        exact_prof,
        pirls_exhausted,
        ..
    } = ws;
    let (beta, beta_step_rhs): (&mut [f64], &mut [f64]) = if beta_mode == BetaMode::Fixed {
        (beta_rhs, beta_prof)
    } else {
        (beta_prof, beta_rhs)
    };
    // Read-only design view `laplace_deviance` takes below — see [`FitData`].
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
        p: *p,
    };
    laplace_deviance(
        &data,
        nb_theta,
        nagq,
        &prm[..],
        beta,
        wx,
        pirls,
        structured,
        pattern,
        packed,
        border,
        beta_step_rhs,
        exact_prof,
        beta_mode,
        pirls_tol_override,
        cluster_rows.as_ref(),
        pirls_exhausted,
        counters,
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
    extra_ids: &[Vec<u32>],
    n: usize,
) -> f64 {
    ws.params[..params.len()].copy_from_slice(params);
    fill_z_f64(&ws.groupings, x, &mut ws.z_buf, n);
    let mut counters = crate::counters::EvalCounters::new();
    laplace_deviance_at(ws, x, y, cluster_ids, extra_ids, n, &mut counters)
}

/// Test-only Profile twin of `glmm_laplace_deviance`: drives `laplace_deviance`
/// with `beta_mode = BetaMode::ProfilePql` and `beta = ws.beta_prof` (the stage-1
/// in/out β), so it evaluates the PQL objective at θ and leaves the profiled β̂(θ) in
/// `ws.beta_prof`. Seeds BOTH the conditional mode (`ws.pirls.u`) and `beta_prof` at 0
/// each call, making the result depend only on `params` — the determinism (BOBYQA
/// objective-consistency) the two-stage optimizer needs. This is the stage-1
/// production call shape (`laplace_deviance(beta_mode = BetaMode::ProfilePql, beta =
/// &mut ws.beta_prof, …)`) exercised from a `&[f64]` in tests.
#[cfg(test)]
pub(crate) fn glmm_laplace_deviance_profile(
    params: &[f64],
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
) -> f64 {
    ws.params[..params.len()].copy_from_slice(params);
    fill_z_f64(&ws.groupings, x, &mut ws.z_buf, n);
    let kk = ws.k.max(1);
    for v in ws.pirls.u[..kk].iter_mut() {
        *v = 0.0;
    }
    for v in ws.beta_prof.iter_mut() {
        *v = 0.0;
    }
    let mut counters = crate::counters::EvalCounters::new();
    laplace_deviance_ws(
        ws,
        x,
        y,
        cluster_ids,
        extra_ids,
        n,
        BetaMode::ProfilePql,
        &mut counters,
    )
}
