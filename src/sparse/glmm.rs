//! Sparse-Z non-Gaussian GLMM half of `sparse::` — see `super` (`sparse/mod.rs`)
//! for the Gaussian LMM half this composes with.
//!
//! On designs outside the dense-solver envelope, this module returns a
//! NaN-filled `Fit { converged: false, ... }` instead of panicking — tested by
//! `fit_over_envelope_non_gaussian_never_panics`. The dense/sparse routing
//! decision is made by `fit::classify_design` (see `fit/mod.rs:609`).

use crate::lmm::LmmGroupings;
use bobyqa::Status;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{
    cholesky_in_place, cholesky_in_place_scratch, LltRegularization,
};
use faer::linalg::cholesky::llt::solve::solve_in_place;
use faer::{Mat, MatRef, Par, Spec};

use super::fill_lambda_small;

// ---------------------------------------------------------------------------
// Sparse-Z non-Gaussian GLMM
// ---------------------------------------------------------------------------
//
// The over-envelope non-Gaussian close: a sparse PIRLS driver composing the
// sparse-Z half (per-row Z scatter / block-diagonal Λ, cap-free heap sizing —
// this module) with the family half (per-family IRLS weights/working residual
// and the joint θ+β Laplace deviance — `family.rs` / the dense `glmm` kernel).
// Per BOBYQA eval the packed M = ZΛ row values are refilled at θ and PIRLS
// iterates the conditional modes: each inner step re-weights the k×k system
// A = M'WM + I from the ≤(q_p + Σq_g) nonzeros per row and re-solves through a
// dense heap LLT, whose log-det at the converged mode feeds the Laplace
// deviance. The dense k×k factor (not the Gaussian blocked kernel) is
// deliberate: the Gaussian path's packed Λ'Z'ZΛ streams are θ-independent and
// packed once, but W changes every inner iteration, so the packing would be
// rebuilt per step anyway — the O(n·nnz²) weighted Gram accumulation dominates
// and stays sparse; only the k×k factor is dense, and k (RE columns) is
// moderate for every over-envelope shape this path serves. Perf retuning is
// explicitly out of scope (YAGNI).
//
// The outer optimizer is the single joint [θ | β] BOBYQA — the dense kernel's
// `two_stage = false` shape, which the A/B gate keeps converging to the same
// Laplace optimum. The θ-only PQL stage 1 is an accelerant only and is not
// replicated here.

/// Per-fit workspace for the sparse non-Gaussian GLMM path. Everything is
/// heap-sized off the design (cap-free); the packed M rows have a FIXED width
/// `q_p + Σ q_g` (every row loads exactly one level of every grouping), so the
/// per-row scatter is two flat arrays, no CSR offsets.
pub(crate) struct SparseGlmmWorkspace {
    pub(crate) g: LmmGroupings,
    /// `lam_small` offsets per extra DECLARATION (parallel to `g.extra_offsets`);
    /// the primary block is at 0. Maps `fill_lambda_small`'s
    /// `[primary | nested | crossed]` layout back to declaration order.
    lam_off_decl: Vec<usize>,
    /// Concatenated per-grouping `q×q` Λ factors (row-major lower-tri), refilled
    /// once per θ eval by `fill_lambda_small`.
    lam_small: Vec<f64>,
    /// Packed M = ZΛ nonzeros, fixed row width: row `i`'s entries at
    /// `[i·width, (i+1)·width)`. Columns (`m_cols`, design-fixed, filled once)
    /// follow `super::for_each_z_entry`'s layout — slope-major primary
    /// (component c at `c·n_primary + f`), level-major extras
    /// (`extra_offsets[e] + level·q_g + c`) — change together. Values
    /// (`m_vals`) are the Λ-folded z entries, refilled per θ eval.
    width: usize,
    m_cols: Vec<u32>,
    m_vals: Vec<f64>,
    // PIRLS state, length n / k — mirrors the dense `GlmmWorkspace` fields of
    // the same names (`pirls_solve` is the reference implementation).
    eta_fixed: Vec<f64>,
    eta: Vec<f64>,
    prob: Vec<f64>,
    w: Vec<f64>,
    mu: Vec<f64>,
    /// Per-row prior weights `wᵢ` (`FitOptions::weights`; all-1 when absent —
    /// zero behavioral change). Enter as `W̃ᵢ ← wᵢ·W̃ᵢ` on the working weight,
    /// `wᵢ·devᵢ` on the deviance, and `wᵢ·ρᵢ` on the score — ρ here is the
    /// PRODUCT W̃·r_working (not R's bare working residual, which prior weights
    /// leave untouched), so it carries the weight. Everything downstream
    /// (A/RHS scatter, β border, Rx Schur) reads `w`/ρ and inherits it.
    /// Every family is wired (Task 7): Gamma's profiled dispersion
    /// (`family::gamma_aic`, called with `Some(&ws.prior_w)`) and its
    /// `vcov(use.hessian=FALSE)` scale (`family::glmm_sigma_sq`) both take
    /// `Σwᵢ`/`wᵢ` in place of `n`/1; its post-fit Pearson φ̂ moment
    /// (`fit_glmm_sparse`'s `dispersion` arm) sums `wᵢrᵢ²` over the raw `n−p`
    /// df. NB's marginal-θ profile (`fit_glmm_nb_sparse`) passes
    /// `opts.weights` straight into `nb_profile_loglik`.
    pub(super) prior_w: Vec<f64>,
    u: Vec<f64>,
    u_prev: Vec<f64>,
    /// `A = M'WM + I` (k×k, full symmetric — the per-row scatter writes both
    /// triangles); left holding the FINAL iterate's A after a converged PIRLS,
    /// which the Rx Schur fill re-factors (the `dense_schur_fill` contract).
    /// `pirls` must therefore never factor THIS field in place — see `a_chol`.
    a: Mat<f64>,
    /// Copy-then-factor target for `a`'s Cholesky (k×k): `pirls` copies `a`'s
    /// lower triangle in here (mirroring `.llt(Side::Lower)`'s internal
    /// `copy_from_triangular_lower`) and factors THIS buffer in place, leaving
    /// `a` itself untouched for `sparse_glmm_schur` to re-read.
    a_chol: Mat<f64>,
    /// Scratch for `a_chol`'s in-place `cholesky_in_place` (k×k) — avoids the
    /// per-PIRLS-iteration `.llt(Side::Lower)` allocation on the `pirls` hot loop.
    a_llt_mem: MemBuffer,
    a_rhs: Vec<f64>,
    /// PIRLS β state, length p: `beta` is the current β (input for a Fixed
    /// solve, in/out for a Profile solve — the sparse twin of the dense
    /// `BetaStep` split); `beta_prev` its step-halving backtrack twin;
    /// `beta_rhs` the Profile δβ RHS/solution scratch. The Profile border
    /// matrices (`xtwx`/`xtwm`/`ainv_mtwx`/`schur`) mirror `BetaStep::Profile`'s.
    beta: Vec<f64>,
    beta_prev: Vec<f64>,
    beta_rhs: Vec<f64>,
    xtwx: Mat<f64>,
    /// `WX = diag(w)·X` (n×p) scratch for the Profile-mode `xtwx = Xᵀ(WX)`
    /// weighted gemm — refilled each PIRLS iteration (W changes) before the
    /// matmul.
    wx: Mat<f64>,
    xtwm: Mat<f64>,
    ainv_mtwx: Mat<f64>,
    schur: Mat<f64>,
    /// Scratch for `schur`'s in-place `cholesky_in_place` (p×p) — avoids the
    /// per-PIRLS-iteration `.llt(Side::Lower)` allocation on the Profile-mode
    /// β-Schur border step.
    schur_llt_mem: MemBuffer,
    k: usize,
    p: usize,
    /// PIRLS exit-tol override read by `pirls` — the sparse twin of the dense
    /// `GlmmWorkspace::pirls_tol_override`. `Some(PIRLS_TOL_REL_FD)` only around
    /// the `WaldSe::Hessian` FD evals (and the RX-fallback central re-eval);
    /// `None` on the fit path, which therefore stays bit-identical.
    pirls_tol_override: Option<f64>,
    /// Per-row linear-predictor offset (`FitOptions::offset`), added into
    /// `eta_fixed` by `refresh_eta_fixed`. `None` ⇒ no offset, byte-identical.
    pub(super) offset: Option<Vec<f64>>,
}

impl SparseGlmmWorkspace {
    pub(crate) fn new(
        g: &LmmGroupings,
        cluster_ids: &[u32],
        extra_ids: &[Vec<u32>],
        n: usize,
        p: usize,
    ) -> Self {
        let q_p = g.primary_q;
        // lam_small layout mirrors `fill_lambda_small` — primary, nested, crossed.
        let mut lam_len = q_p * q_p;
        let mut lam_off_decl = vec![0usize; g.extra_offsets.len()];
        if let Some(nf) = g.nested {
            lam_off_decl[nf.decl] = lam_len;
            lam_len += nf.q * nf.q;
        }
        for cf in &g.crossed {
            lam_off_decl[cf.decl] = lam_len;
            lam_len += cf.q * cf.q;
        }
        let width = q_p + g.extra_q.iter().sum::<usize>();
        // m_cols is design-fixed: fill once from the ids (values are θ-dependent
        // and filled per eval by `fill_m_vals`).
        let mut m_cols = vec![0u32; n * width];
        for i in 0..n {
            let mut t = i * width;
            let f = cluster_ids[i] as usize;
            for c in 0..q_p {
                m_cols[t] = (c * g.n_primary + f) as u32;
                t += 1;
            }
            for (e, ids_e) in extra_ids.iter().enumerate() {
                let q_g = g.extra_q[e];
                let base = g.extra_offsets[e] + ids_e[i] as usize * q_g;
                for c in 0..q_g {
                    m_cols[t] = (base + c) as u32;
                    t += 1;
                }
            }
        }
        let k = g.k_total;
        SparseGlmmWorkspace {
            g: g.clone(),
            lam_off_decl,
            lam_small: vec![0.0; lam_len.max(1)],
            width,
            m_cols,
            m_vals: vec![0.0; n * width],
            eta_fixed: vec![0.0; n.max(1)],
            eta: vec![0.0; n.max(1)],
            prob: vec![0.0; n.max(1)],
            w: vec![0.0; n.max(1)],
            mu: vec![0.0; n.max(1)],
            prior_w: vec![1.0; n.max(1)],
            u: vec![0.0; k.max(1)],
            u_prev: vec![0.0; k.max(1)],
            a: Mat::zeros(k.max(1), k.max(1)),
            a_chol: Mat::zeros(k.max(1), k.max(1)),
            a_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                k.max(1),
                Par::Seq,
                Spec::default(),
            )),
            a_rhs: vec![0.0; k.max(1)],
            beta: vec![0.0; p.max(1)],
            beta_prev: vec![0.0; p.max(1)],
            beta_rhs: vec![0.0; p.max(1)],
            xtwx: Mat::zeros(p.max(1), p.max(1)),
            wx: Mat::zeros(n.max(1), p.max(1)),
            xtwm: Mat::zeros(p.max(1), k.max(1)),
            ainv_mtwx: Mat::zeros(k.max(1), p.max(1)),
            schur: Mat::zeros(p.max(1), p.max(1)),
            schur_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                p.max(1),
                Par::Seq,
                Spec::default(),
            )),
            k,
            p,
            pirls_tol_override: None,
            offset: None,
        }
    }

    /// Fresh per-thread clone for one FD-Hessian worker: independently-sized, shares
    /// no mutable state with `self`. The design-fixed fields (`g`, `lam_off_decl`,
    /// `width`, `m_cols`, `prior_w`) and the tol override are carried over; the
    /// scratch (`lam_small`/`m_vals`/PIRLS buffers) is cloned only for its SIZE —
    /// every eval refills Λ and M and cold-seeds û = 0 (`sparse_glmm_deviance` with
    /// `pirls_tol_override == Some`), so each grid cell is a pure function of
    /// `(gamma_hat, steps, design)` and reproduces the serial value bit-for-bit. The
    /// two `MemBuffer`s can't be cloned (not `Clone`); they are re-sized from `k`/`p`
    /// exactly as `new` does.
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    fn clone_worker(&self) -> SparseGlmmWorkspace {
        // Exhaustive destructure (no `..`) so a future field addition to
        // `SparseGlmmWorkspace` fails compilation here instead of silently
        // sharing state across FD-Hessian worker threads.
        let Self {
            g,
            lam_off_decl,
            lam_small,
            width,
            m_cols,
            m_vals,
            eta_fixed,
            eta,
            prob,
            w,
            mu,
            prior_w,
            u,
            u_prev,
            a,
            a_chol,
            a_llt_mem: _,
            a_rhs,
            beta,
            beta_prev,
            beta_rhs,
            xtwx,
            wx,
            xtwm,
            ainv_mtwx,
            schur,
            schur_llt_mem: _,
            k,
            p,
            pirls_tol_override,
            offset,
        } = self;
        SparseGlmmWorkspace {
            g: g.clone(),
            lam_off_decl: lam_off_decl.clone(),
            lam_small: lam_small.clone(),
            width: *width,
            m_cols: m_cols.clone(),
            m_vals: m_vals.clone(),
            eta_fixed: eta_fixed.clone(),
            eta: eta.clone(),
            prob: prob.clone(),
            w: w.clone(),
            mu: mu.clone(),
            prior_w: prior_w.clone(),
            u: u.clone(),
            u_prev: u_prev.clone(),
            a: a.clone(),
            a_chol: a_chol.clone(),
            a_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                (*k).max(1),
                Par::Seq,
                Spec::default(),
            )),
            a_rhs: a_rhs.clone(),
            beta: beta.clone(),
            beta_prev: beta_prev.clone(),
            beta_rhs: beta_rhs.clone(),
            xtwx: xtwx.clone(),
            wx: wx.clone(),
            xtwm: xtwm.clone(),
            ainv_mtwx: ainv_mtwx.clone(),
            schur: schur.clone(),
            schur_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                (*p).max(1),
                Par::Seq,
                Spec::default(),
            )),
            k: *k,
            p: *p,
            pirls_tol_override: *pirls_tol_override,
            offset: offset.clone(),
        }
    }

    /// Refill the packed M values at the current Λ (`lam_small` must be filled
    /// for this θ): entry c of a row's block is the lower-tri fold
    /// `Σ_{r≥c} z[r]·Λ[r,c]` with `z = [1, x[slope cols…]]` — the same sandwich
    /// `apply_lambda` writes densely on the in-envelope GLMM path.
    fn fill_m_vals(&mut self, x: MatRef<f64>, n: usize) {
        let g = &self.g;
        let q_p = g.primary_q;
        for i in 0..n {
            let mut t = i * self.width;
            for c in 0..q_p {
                let mut acc = 0.0;
                for r in c..q_p {
                    let z = if r == 0 {
                        1.0
                    } else {
                        x[(i, g.primary_slope_cols[r - 1])]
                    };
                    acc += z * self.lam_small[r * q_p + c];
                }
                self.m_vals[t] = acc;
                t += 1;
            }
            for e in 0..g.extra_offsets.len() {
                let q_g = g.extra_q[e];
                let lo = self.lam_off_decl[e];
                for c in 0..q_g {
                    let mut acc = 0.0;
                    for r in c..q_g {
                        let z = if r == 0 {
                            1.0
                        } else {
                            x[(i, g.extra_slope_cols[e][r - 1])]
                        };
                        acc += z * self.lam_small[lo + r * q_g + c];
                    }
                    self.m_vals[t] = acc;
                    t += 1;
                }
            }
        }
    }

    /// Refill `eta_fixed[i] = offset[i] + Σ_j x[i,j]·β[j]` from `self.beta` — the
    /// sparse twin of `pirls::refresh_eta_fixed`. Called at PIRLS entry and, in
    /// Profile mode, after every β update (δβ step and each β halving).
    fn refresh_eta_fixed(&mut self, x: MatRef<f64>, n: usize) {
        for i in 0..n {
            let mut e = 0.0;
            for (j, &b) in self.beta[..self.p].iter().enumerate() {
                e += x[(i, j)] * b;
            }
            self.eta_fixed[i] = e;
        }
        if let Some(o) = &self.offset {
            for (e, &ov) in self.eta_fixed[..n].iter_mut().zip(o) {
                *e += ov;
            }
        }
    }

    /// Penalized-IRLS inner solve on the packed sparse M rows — the sparse twin
    /// of `glmm::pirls_solve`, with the SAME two β modes: `profile = false`
    /// holds `self.beta` fixed (the FD-Hessian / joint stage-2 contract);
    /// `profile = true` adds the joint δβ Schur-border step each iteration
    /// (PQL — β̂(θ) written back through `self.beta`), backtracked in lockstep
    /// with u. Same discipline verbatim: trial evaluation at the current u,
    /// band-tolerant retrospective step-halving (lme4 `pwrssUpdate`), the mixed
    /// `dev(uⱼ) + ‖uⱼ₊₁‖²` convergence rule, `log|A|` off the factor that
    /// produced the returned u. Every family takes the general Fisher-scoring
    /// branch through `family.rs` (no fused-SIMD logit shortcut here — for
    /// canonical links the general weight/residual reduce to the same
    /// quantities, and this path has no byte-identity gate to a prior
    /// implementation). Returns `(dev, ‖ũ‖², log|A|, converged)`; a non-PD
    /// A/S_β or exhausted halvings surface as `(NaN, NaN, NaN, false)`.
    /// Iterates from whatever `self.u` holds on entry — `pirls` itself never
    /// decides reset vs. warm-start; that call is `sparse_glmm_deviance`'s
    /// (its caller), which cold-seeds `self.u = 0` for FD-Hessian/tight-tol
    /// evals and otherwise leaves the previous eval's converged `u` in place
    /// as a warm start.
    fn pirls(
        &mut self,
        family: crate::Family,
        nb_theta: f64,
        x: MatRef<f64>,
        y: &[f64],
        n: usize,
        profile: bool,
    ) -> (f64, f64, f64, bool) {
        let (k, p, width) = (self.k, self.p, self.width);
        self.refresh_eta_fixed(x, n);
        let tol = self
            .pirls_tol_override
            .unwrap_or_else(|| crate::glmm::pirls_tol(family));
        // Backtrack seeds for the FIRST trial iterate (which has no accepted
        // predecessor): u_prev = 0 so an infeasible first trial halves toward
        // η = eta_fixed (the canonical cold seed), beta_prev = the caller's β.
        // Dead for the overshoot trigger — it cannot fire before an accept —
        // so only the domain-infeasibility trigger ever reads these seeds
        // (mirrors `pirls_solve`).
        self.u_prev[..k].fill(0.0);
        if profile {
            let (head, _) = self.beta.split_at(p);
            self.beta_prev[..p].copy_from_slice(head);
        }
        let mut pen_accepted = f64::INFINITY;
        let mut mixed_prev = f64::INFINITY;
        let mut halvings = 0usize;
        let mut converged = false;
        let mut dev = f64::NAN;
        let mut pen = f64::NAN;
        let mut logdet = 0.0;
        for _ in 0..crate::glmm::PIRLS_MAX_ITERS {
            // Trial evaluation at the current u: (Mu)ᵢ, then η/μ/W/deviance.
            // `infeasible` flags any raw η outside the link's open domain
            // (Gamma-inverse only — mirrors `pirls_solve`).
            dev = 0.0;
            let mut infeasible = false;
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let base = i * width;
                let mut acc = 0.0;
                for t in base..base + width {
                    acc += self.m_vals[t] * self.u[self.m_cols[t] as usize];
                }
                self.mu[i] = acc;
                let raw = self.eta_fixed[i] + acc;
                infeasible |= crate::family::eta_infeasible(family, raw);
                let e = crate::family::clamp_eta(family, raw);
                self.eta[i] = e;
                // Canonical-link shortcut (Poisson-log) lives inside this call — see
                // `irls_weight_and_resid`'s doc comment.
                let (mui, wi, _) = crate::family::irls_weight_and_resid(family, nb_theta, y[i], e);
                self.prob[i] = mui;
                self.w[i] = (self.prior_w[i] * wi).max(crate::glm::WEIGHT_CLAMP);
                dev += self.prior_w[i] * crate::family::dev_resid(family, nb_theta, y[i], mui);
            }
            // Band-tolerant retrospective step-halving (mirror `pirls_solve` —
            // see its in-loop comment for why the band must not converge, and
            // for why a domain-infeasible trial halves regardless of the band
            // and only from an accepted feasible iterate). In Profile mode the
            // trial point is the JOINT (u, β) step, so β halves toward
            // `beta_prev` in lockstep with u.
            let pen_u: f64 = self.u[..k].iter().map(|v| v * v).sum();
            let penalized = dev + pen_u;
            if infeasible || penalized - pen_accepted > tol * (1.0 + penalized.abs()) {
                if halvings < crate::glmm::PIRLS_MAX_HALVINGS {
                    halvings += 1;
                    for c in 0..k {
                        self.u[c] = 0.5 * (self.u[c] + self.u_prev[c]);
                    }
                    if profile {
                        for j in 0..p {
                            self.beta[j] = 0.5 * (self.beta[j] + self.beta_prev[j]);
                        }
                        self.refresh_eta_fixed(x, n);
                    }
                    continue;
                }
                return (f64::NAN, f64::NAN, f64::NAN, false);
            }
            halvings = 0;
            pen_accepted = penalized;
            self.u_prev[..k].copy_from_slice(&self.u[..k]);
            if profile {
                self.beta_prev[..p].copy_from_slice(&self.beta[..p]);
            }
            // A = M'WM + I and rhs = M'(W·Mu + W·r), accumulated from each row's
            // ≤width nonzeros (full-symmetric A — both triangles written, the
            // `SparseLmmWorkspace` Z'Z convention). Profile additionally
            // accumulates the β-gradient X'ρ (ρ = the effective residual) into
            // `beta_rhs` — the joint system's bottom-block RHS.
            for c in 0..k {
                for r in 0..k {
                    self.a[(r, c)] = 0.0;
                }
                self.a_rhs[c] = 0.0;
            }
            if profile {
                for v in self.beta_rhs[..p].iter_mut() {
                    *v = 0.0;
                }
            }
            for i in 0..n {
                let wi = self.w[i];
                let dmu = crate::family::mu_eta(family, self.eta[i]);
                let v = crate::family::variance(family, nb_theta, self.prob[i]);
                let rho = self.prior_w[i] * dmu * (y[i] - self.prob[i]) / v;
                let q_i = wi * self.mu[i] + rho;
                let base = i * width;
                for ta in base..base + width {
                    let ca = self.m_cols[ta] as usize;
                    let va = self.m_vals[ta];
                    let wva = wi * va;
                    for tb in base..base + width {
                        let cb = self.m_cols[tb] as usize;
                        let vb = self.m_vals[tb];
                        self.a[(ca, cb)] += wva * vb;
                    }
                    self.a_rhs[ca] += va * q_i;
                }
                if profile {
                    for j in 0..p {
                        self.beta_rhs[j] += x[(i, j)] * rho;
                    }
                }
            }
            for r in 0..k {
                self.a[(r, r)] += 1.0;
            }
            // Copy A's lower triangle into the persistent `a_chol` scratch (mirrors
            // `.llt(Side::Lower)`'s own `copy_from_triangular_lower`), then factor
            // THAT in place — `self.a` must come out of this call unmutated (see
            // its field doc; `sparse_glmm_schur` re-reads it post-fit).
            self.a_chol.copy_from_triangular_lower(self.a.as_ref());
            if cholesky_in_place(
                self.a_chol.as_mut(),
                LltRegularization::default(),
                Par::Seq,
                MemStack::new(&mut self.a_llt_mem),
                Spec::default(),
            )
            .is_err()
            {
                return (f64::NAN, f64::NAN, f64::NAN, false);
            }
            logdet = 0.0;
            for r in 0..k {
                logdet += self.a_chol[(r, r)].ln();
            }
            solve_in_place(
                self.a_chol.as_ref(),
                faer::MatMut::from_column_major_slice_mut(&mut self.a_rhs[..k], k, 1),
                Par::Seq,
                MemStack::new(&mut self.a_llt_mem),
            );
            pen = 0.0;
            for c in 0..k {
                self.u[c] = self.a_rhs[c];
                pen += self.u[c] * self.u[c];
            }
            // Profile-mode joint δβ step (β-Schur border), taken while `ac` is
            // alive — mirrors `pirls_solve`'s Profile block: T = A⁻¹B,
            // S_β = C − B'T, δβ = S_β⁻¹(X'ρ − B'·δu₀), then β += δβ and
            // u ← u_new − T·δβ.
            if profile {
                // B' = X'WM (p×k) via the packed rows; C = X'WX (p×p).
                for r in 0..p {
                    for c in 0..k {
                        self.xtwm[(r, c)] = 0.0;
                    }
                }
                for i in 0..n {
                    let wi = self.w[i];
                    let base = i * width;
                    for r in 0..p {
                        let xw = x[(i, r)] * wi;
                        for t in base..base + width {
                            self.xtwm[(r, self.m_cols[t] as usize)] += xw * self.m_vals[t];
                        }
                    }
                }
                // C = X'WX = Xᵀ diag(w) X via one weighted gemm, replacing the
                // O(p²·n) per-pair loop. Recomputed each PIRLS iteration because
                // W changes with the working weights — same per-iteration
                // invariant as the X'WM assembly just above. WX = diag(w)·X is
                // formed into `wx`, then xtwx = Xᵀ·WX (full p×p, kept
                // full-symmetric as the downstream border reads it).
                for r in 0..p {
                    for i in 0..n {
                        self.wx[(i, r)] = self.w[i] * x[(i, r)];
                    }
                }
                faer::linalg::matmul::matmul(
                    self.xtwx.as_mut(),
                    faer::Accum::Replace,
                    x.transpose(),
                    self.wx.as_ref(),
                    1.0,
                    Par::Seq,
                );
                for r in 0..k {
                    for c in 0..p {
                        self.ainv_mtwx[(r, c)] = self.xtwm[(c, r)];
                    }
                }
                solve_in_place(
                    self.a_chol.as_ref(),
                    self.ainv_mtwx.as_mut(),
                    Par::Seq,
                    MemStack::new(&mut self.a_llt_mem),
                );
                for r in 0..p {
                    for c in 0..p {
                        let mut s = self.xtwx[(r, c)];
                        for j in 0..k {
                            s -= self.xtwm[(r, j)] * self.ainv_mtwx[(j, c)];
                        }
                        self.schur[(r, c)] = s;
                    }
                }
                // rhs = X'ρ − B'·δu₀ (δu₀ = u − u_prev).
                for r in 0..p {
                    let mut acc = 0.0;
                    for c in 0..k {
                        acc += self.xtwm[(r, c)] * (self.u[c] - self.u_prev[c]);
                    }
                    self.beta_rhs[r] -= acc;
                }
                if cholesky_in_place(
                    self.schur.as_mut(),
                    LltRegularization::default(),
                    Par::Seq,
                    MemStack::new(&mut self.schur_llt_mem),
                    Spec::default(),
                )
                .is_err()
                {
                    return (f64::NAN, f64::NAN, f64::NAN, false);
                }
                solve_in_place(
                    self.schur.as_ref(),
                    faer::MatMut::from_column_major_slice_mut(&mut self.beta_rhs[..p], p, 1),
                    Par::Seq,
                    MemStack::new(&mut self.schur_llt_mem),
                );
                for j in 0..p {
                    self.beta[j] += self.beta_rhs[j];
                }
                for c in 0..k {
                    let mut acc = 0.0;
                    for j in 0..p {
                        acc += self.ainv_mtwx[(c, j)] * self.beta_rhs[j];
                    }
                    self.u[c] -= acc;
                }
                self.refresh_eta_fixed(x, n);
                pen = 0.0;
                for c in 0..k {
                    pen += self.u[c] * self.u[c];
                }
            }
            let mixed = dev + pen;
            if (mixed - mixed_prev).abs() < tol * (1.0 + mixed.abs()) {
                converged = true;
                break;
            }
            mixed_prev = mixed;
        }
        (dev, pen, logdet, converged)
    }
}

/// Joint Laplace deviance at `params = [θ | β]` on the sparse path — the sparse
/// twin of `glmm::laplace_deviance`: refill Λ and the packed M values at θ,
/// seed û (fit-path evals, `pirls_tol_override == None`, warm-start from
/// whatever `ws.u` holds on entry — the previous call's converged mode, fewer
/// PIRLS iterations to reconverge; FD-Hessian/tight-tol evals cold-seed û = 0,
/// order-free as those evals require a seed independent of evaluation order),
/// run the sparse PIRLS, and return
/// `data + ‖ũ‖² + log|A|²` with Gamma's `aic` substitution
/// (`family::gamma_aic`) exactly as the dense objective does. β mode mirrors
/// the dense call sites: `profile_beta = false` copies `params[n_theta..]`
/// into `ws.beta` and holds it fixed (the stage-2 / FD-Hessian contract);
/// `profile_beta = true` reads only the θ prefix (`params` may be a θ-only
/// slice) and lets the PQL δβ step drive `ws.beta` from the CALLER's
/// pre-seeded value (the stage-1 objective — seed `ws.beta` to a fixed β₀
/// before each eval so the objective stays a function of θ alone).
/// Non-convergence / Cholesky failure ⇒ `f64::INFINITY`.
#[allow(clippy::too_many_arguments)]
pub(super) fn sparse_glmm_deviance(
    family: crate::Family,
    nb_theta: f64,
    params: &[f64],
    ws: &mut SparseGlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    n: usize,
    profile_beta: bool,
) -> f64 {
    let n_theta = ws.g.n_theta();
    let p = ws.p;
    fill_lambda_small(&params[..n_theta], &ws.g, &mut ws.lam_small);
    ws.fill_m_vals(x, n);
    if !profile_beta {
        ws.beta[..p].copy_from_slice(&params[n_theta..n_theta + p]);
    }
    // Fit-path evals (pirls_tol_override == None) carry the previous call's
    // converged û forward as PIRLS's starting point — fewer iterations to
    // reconverge, same fixed point (seed-independence: PIRLS converges to the
    // same conditional mode from any start, only iteration count differs; see
    // the dense analogue `warm_start_objective_is_seed_independent`,
    // src/glmm/tests.rs). FD-Hessian/tight-tol evals (Some(...)) still cold-seed
    // û = 0, preserving the order-free property `sparse_fd_hessian_cov` relies on.
    if ws.pirls_tol_override.is_some() {
        for v in ws.u.iter_mut() {
            *v = 0.0;
        }
    }
    let (dev, pen, logdet, conv) = ws.pirls(family, nb_theta, x, y, n, profile_beta);
    if !conv || !dev.is_finite() {
        return f64::INFINITY;
    }
    let data_term = if matches!(family, crate::Family::Gamma { .. }) {
        crate::family::gamma_aic(y, &ws.prob[..n], dev, n, Some(&ws.prior_w[..n]))
    } else {
        dev
    };
    data_term + pen + 2.0 * logdet
}

/// Rx (closed-form Schur) fixed-effect information at the converged state —
/// the sparse twin of `glmm::se::dense_schur_fill`: `S_β = X'W̃X − X'W̃M·A⁻¹M'W̃X`
/// from the final PIRLS iterate's W̃ (`ws.w`), packed M rows, and A (`ws.a`).
/// Returns `None` on a non-PD A. Local allocations are fine — this is a
/// once-per-fit cold path, not the optimizer loop.
fn sparse_glmm_schur(ws: &mut SparseGlmmWorkspace, x: MatRef<f64>, n: usize) -> Option<Mat<f64>> {
    use faer::linalg::solvers::Solve;
    let (k, p, width) = (ws.k, ws.p, ws.width);
    let mut xtwx = Mat::<f64>::zeros(p, p);
    for r in 0..p {
        for c in 0..=r {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, r)] * ws.w[i] * x[(i, c)];
            }
            xtwx[(r, c)] = s;
            xtwx[(c, r)] = s;
        }
    }
    // X'W̃M (p×k) by per-row scatter over the packed nonzeros.
    let mut xtwm = Mat::<f64>::zeros(p, k);
    for i in 0..n {
        let wi = ws.w[i];
        let base = i * width;
        for r in 0..p {
            let xw = x[(i, r)] * wi;
            for t in base..base + width {
                xtwm[(r, ws.m_cols[t] as usize)] += xw * ws.m_vals[t];
            }
        }
    }
    let ac = match ws.a.as_ref().llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return None,
    };
    let mut ainv_mtwx = Mat::<f64>::zeros(k, p);
    for r in 0..k {
        for c in 0..p {
            ainv_mtwx[(r, c)] = xtwm[(c, r)];
        }
    }
    ac.solve_in_place(ainv_mtwx.as_mut());
    let mut schur = Mat::<f64>::zeros(p, p);
    for r in 0..p {
        for c in 0..p {
            let mut s = xtwx[(r, c)];
            for j in 0..k {
                s -= xtwm[(r, j)] * ainv_mtwx[(j, c)];
            }
            schur[(r, c)] = s;
        }
    }
    Some(schur)
}

/// Relative FD step for the SPARSE joint-deviance Hessian — deliberately NOT
/// the dense `glmm::FD_STEP_REL` (1e-2): the two paths sit on opposite sides
/// of the truncation-vs-noise trade. On the weighted sparse Gamma golden
/// (`sim_sparse_gamma`, weight-perturbed cell), the flat intercept direction's
/// FD-step plateau (true curvature, found by scanning h) sits at
/// h ∈ [5e-5, 2e-4]; h = 1e-3 falls outside that plateau and biases se(β₀)
/// high, while h = 1e-4 lands on the plateau for both the weighted cell
/// (0.18915 vs lme4 0.18909) and its unweighted sibling (0.18919, unchanged
/// from the 1e-3 step — that golden was already inside the plateau). h = 1e-4
/// also sits inside the dense path's already-validated [1e-4, 1e-2] band. The
/// DENSE path is the mirror image: at h = 1e-3 its FD noise blows the curated
/// se_hess gates (sim_gamma 1e-2, cbpp_probit 2e-3 vs the 1e-3 band) while
/// h = 1e-2 holds them at ~1e-4 — so the dense constant stays 1e-2 and this
/// one must not be folded back into it.
const SPARSE_FD_STEP_REL: f64 = 1e-4;

/// FD-Hessian joint (θ,β) covariance on the sparse path — mirrors
/// `glmm::fd_hessian_cov`'s scheme exactly (single-step central differences, no
/// Richardson extrapolation, step `h_k = SPARSE_FD_STEP_REL·max(1, |γ̂_k|)`
/// (sparse-calibrated, see the constant above),
/// `cov = 2·(H_dev⁻¹)_ββ`, θ SE from the θ diagonal) minus the warm-seed
/// machinery: every eval here cold-seeds û = 0 inside `sparse_glmm_deviance`,
/// which is a constant seed and therefore order-free by the same argument.
/// Returns `None` on a non-finite perturbed deviance or non-PD joint Hessian —
/// the caller falls back to the Rx Schur (the `NonPdFellBackToRx` shape).
/// Tolerance contract: the CALLER sets `ws.pirls_tol_override =
/// Some(PIRLS_TOL_REL_FD)` around this call (and its fallback re-eval) and
/// resets it after — set/reset can't live here because the `?` early returns
/// would skip the reset. Same rationale as the dense `fd_hessian_cov`: at the
/// canonical fit tol the FD is not step-invariant; at the tight tol it is, by
/// construction.
/// One Hessian entry `(i, j)` of the sparse FD grid via the shared stencils:
/// diagonal → `fd_second_diff`, off-diagonal → `fd_mixed_diff`. Returns `None`
/// on any non-finite eval so both the serial (`?`) and rayon (grid-wide check)
/// arms share identical per-entry logic. Generic over the eval closure (each arm
/// binds its own workspace) — monomorphized, no dyn.
fn fd_hess_entry(
    i: usize,
    j: usize,
    steps: &[f64],
    f0: f64,
    ev: &mut impl FnMut(&[usize], &[f64]) -> f64,
) -> Option<f64> {
    let h = if i == j {
        crate::glmm::fd_second_diff(ev, i, steps[i], f0)
    } else {
        crate::glmm::fd_mixed_diff(ev, i, j, steps[i], steps[j])
    };
    h.is_finite().then_some(h)
}

#[allow(clippy::too_many_arguments)]
fn sparse_fd_hessian_cov(
    family: crate::Family,
    nb_theta: f64,
    gamma_hat: &[f64],
    ws: &mut SparseGlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    n: usize,
    parallel_inner: bool,
) -> Option<(Mat<f64>, Vec<f64>)> {
    use faer::linalg::solvers::Solve;
    let m = gamma_hat.len();
    let n_theta = ws.g.n_theta();
    let p = ws.p;
    let f0 = sparse_glmm_deviance(family, nb_theta, gamma_hat, ws, x, y, n, false);
    if !f0.is_finite() {
        return None;
    }
    let steps: Vec<f64> = gamma_hat
        .iter()
        .map(|&g| SPARSE_FD_STEP_REL * g.abs().max(1.0))
        .collect();
    // Each grid cell cold-seeds û = 0 (constant seed), so every eval is a pure
    // function of (gamma_hat, steps, design) — no frozen-seed discipline is needed
    // at all, and per-thread workspaces reproduce the serial values bitwise.
    let mut hess = Mat::<f64>::zeros(m, m);
    let use_par = cfg!(all(feature = "parallel", not(target_arch = "wasm32"))) && parallel_inner;
    if use_par {
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        {
            use rayon::prelude::*;
            let cells: Vec<(usize, usize)> =
                (0..m).flat_map(|i| (i..m).map(move |j| (i, j))).collect();
            let ws_ro: &SparseGlmmWorkspace = ws;
            let steps = &steps;
            let results: Vec<(usize, usize, Option<f64>)> = cells
                .par_iter()
                .map_init(
                    || (ws_ro.clone_worker(), gamma_hat.to_vec()),
                    |(wws, pt), &(i, j)| {
                        let mut ev = |coords: &[usize], deltas: &[f64]| -> f64 {
                            pt.copy_from_slice(gamma_hat);
                            for (&c, &d) in coords.iter().zip(deltas) {
                                pt[c] += d;
                            }
                            sparse_glmm_deviance(family, nb_theta, pt, wws, x, y, n, false)
                        };
                        (i, j, fd_hess_entry(i, j, steps, f0, &mut ev))
                    },
                )
                .collect();
            // Serial arm returns None on the FIRST non-finite eval; here the whole
            // grid ran, then we check — same destination (Rx fallback), extra work
            // only on the already-failing path.
            if results.iter().any(|(_, _, h)| h.is_none()) {
                return None;
            }
            for (i, j, h) in results {
                let h = h.expect("checked all-Some above");
                hess[(i, j)] = h;
                hess[(j, i)] = h;
            }
        }
    } else {
        // Diagonal cells are single-step central second differences (no Richardson —
        // see the doc comment above `fd_hessian_cov`). Serial arm returns None on the
        // FIRST non-finite eval via `?`; the same per-entry stencil the rayon arm uses.
        let mut pt = gamma_hat.to_vec();
        for i in 0..m {
            for j in i..m {
                let mut ev = |coords: &[usize], deltas: &[f64]| -> f64 {
                    pt.copy_from_slice(gamma_hat);
                    for (&c, &d) in coords.iter().zip(deltas) {
                        pt[c] += d;
                    }
                    sparse_glmm_deviance(family, nb_theta, &pt, ws, x, y, n, false)
                };
                let hij = fd_hess_entry(i, j, &steps, f0, &mut ev)?;
                hess[(i, j)] = hij;
                hess[(j, i)] = hij;
            }
        }
    }
    let chol = hess.as_ref().llt(faer::Side::Lower).ok()?;
    let mut inv = Mat::<f64>::identity(m, m);
    chol.solve_in_place(inv.as_mut());
    let mut cov = Mat::<f64>::zeros(p, p);
    for a in 0..p {
        for b in 0..p {
            cov[(a, b)] = 2.0 * inv[(n_theta + a, n_theta + b)];
        }
    }
    let theta_se: Vec<f64> = (0..n_theta)
        .map(|kk| (2.0 * inv[(kk, kk)]).max(0.0).sqrt())
        .collect();
    Some((cov, theta_se))
}

/// The non-converged NaN `Fit` for the sparse GLMM path (mirrors the dense
/// adapters' NaN-fill shape; `dispersion` NaN, `tau2` NaN per θ coordinate).
fn sparse_glmm_nan_fit(p: usize, n_theta: usize) -> crate::Fit {
    crate::Fit {
        beta: vec![f64::NAN; p],
        se: vec![f64::NAN; p],
        vcov: crate::fit::nan_vcov(p),
        tau2: vec![f64::NAN; n_theta],
        dispersion: f64::NAN,
        converged: false,
        varcorr: vec![],
        stddev_se: vec![],
        aliased: vec![false; p],
        n_eval: 0,
        deviance: f64::NAN,
        singular: false,
        loglik: f64::NAN,
        df: 0,
        reml: false,
        fitted: vec![],
        ranef: vec![],
        ranef_levels: vec![],
    }
}

/// Sparse-Z non-Gaussian GLMM end-to-end fit: the
/// over-envelope sibling of the dense `fit::fit_glmm` adapter, serving
/// Binomial / Poisson / Gamma (and, via `fit_glmm_nb_sparse`, NB) designs that
/// exceed the NoZ envelope. Single joint [θ | β] BOBYQA over the sparse Laplace
/// deviance (the dense kernel's `two_stage = false` shape), θ/β seeding and
/// ρ schedule mirroring `GlmmWorkspace::for_cluster_spec` + `glmm::fit_glmm`
/// (blind THETA0 θ₀ or a warm start floored at `THETA_TRUTH_FLOOR`; β from the
/// no-RE GLM warm start or the caller's `start`), diagonal-θ pin at
/// `PIN_THETA`, and a pinned-γ̂ re-eval whose finite deviance is the
/// convergence witness (the same degenerate-fit guard as the dense kernel).
///
/// SE: both `WaldSe` arms, exactly as the dense path emits them — `Hessian`
/// (default) via the joint FD-Hessian (`sparse_fd_hessian_cov`), falling back
/// to the Rx Schur on a non-PD Hessian; `Rx` via the closed-form Schur
/// conditional on θ̂ (`sparse_glmm_schur`). `tau2`/`dispersion`/`varcorr`
/// mirror `fit::fit_glmm`'s mapping (Gamma's pwrss/n τ² scale and Pearson
/// dispersion included). Returns the mapped `Fit` plus the minimized marginal
/// Laplace deviance (the NB marginal-θ objective kernel); non-NB callers take
/// `.0`. On failure (non-convergence, rank-deficiency, or numeric failure),
/// returns a NaN-filled `Fit { converged: false, ... }` constructed by
/// `sparse_glmm_nan_fit`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_glmm_sparse(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &crate::ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    nb_theta: f64,
    start: Option<&crate::StartValues>,
    opts: &crate::FitOptions,
) -> (crate::Fit, f64) {
    let re = model
        .re
        .as_ref()
        .expect("fit_glmm_sparse requires a mixed model (re: Some)");
    let family = model.family;
    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();
    let extra_slope_cols: Vec<Vec<usize>> = re
        .extra_groupings
        .iter()
        .map(|g| g.slopes.iter().map(|&c| c as usize).collect())
        .collect();
    let g = LmmGroupings::from_cluster_spec_ext(model, n, &slope_cols, &extra_slope_cols);
    let n_theta = g.n_theta();
    if n == 0 || p == 0 {
        return (sparse_glmm_nan_fit(p, n_theta), f64::INFINITY);
    }
    let xm = MatRef::from_row_major_slice(x, n, p);
    let mut ws = SparseGlmmWorkspace::new(&g, cluster_ids, extra_ids, n, p);
    if let Some(w) = &opts.weights {
        ws.prior_w[..n].copy_from_slice(w);
    }
    ws.offset = opts.offset.clone();

    // Joint [θ | β] parameter vector, seeds, and boxes. The θ cold start is the
    // structure-aware blind seed from `blind_theta_and_bounds` — diagonal vech
    // entries at THETA0, OFF-DIAGONAL entries at 0 — the same shape the LMM
    // cold starts adopted in the 2026-07-11 basin fix: with a wide vech block (the
    // over-width q_g=5 shape) all-ones off-diagonals give a badly mis-scaled Λ
    // (D diagonals up to q·THETA0²) and the joint BOBYQA stalls in that basin —
    // measured on sim_sparse_gamma, where the all-THETA0 start converged ~240
    // deviance units above the lme4 optimum with θ̂ ≈ θ₀. A warm start is floored
    // at THETA_TRUTH_FLOOR on every coordinate (mirror `glmm::fit_glmm`). The β
    // portion mirrors `fit::fit_glmm`: caller start verbatim, else the no-RE GLM
    // warm start; clamped into the ±BETA_BOX box.
    let (theta0, mut lower, mut upper) = g.blind_theta_and_bounds();
    let mut params = vec![0.0f64; n_theta + p];
    match start {
        Some(s) => {
            for (t, &v) in params[..n_theta].iter_mut().zip(&s.theta) {
                *t = v.max(crate::lmm::THETA_TRUTH_FLOOR);
            }
        }
        None => params[..n_theta].copy_from_slice(&theta0),
    }
    let beta_start = match start {
        Some(s) => s.beta.clone(),
        None => {
            crate::fit::glm_warm_start_beta(family, nb_theta, xm, y, n, p, opts.offset.as_deref())
        }
    };
    for (slot, &b) in params[n_theta..].iter_mut().zip(&beta_start) {
        *slot = b.clamp(-crate::glmm::BETA_BOX, crate::glmm::BETA_BOX);
    }
    lower.extend(std::iter::repeat_n(-crate::glmm::BETA_BOX, p));
    upper.extend(std::iter::repeat_n(crate::glmm::BETA_BOX, p));

    // ρ schedule: mirror `GlmmWorkspace::for_cluster_spec` — ρ_begin ≤ RHO_BEGIN
    // and ≤ 0.1·min diagonal θ₀ (= 0.1·THETA0 on the blind start), ρ_end the
    // GLMM-calibrated GLMM_RHO_END, PRIMA-default npt for the joint dimension.
    let rho_begin = (0.1 * crate::lmm::THETA0).min(crate::lmm::RHO_BEGIN);

    // STAGE 1 — θ-only BOBYQA on the PQL objective (β profiled inside PIRLS),
    // mirroring the dense two-stage optimizer: an accelerant that warm-starts
    // the joint stage 2 and never gates convergence. Not optional garnish here:
    // on the over-width gamma golden the single-stage joint solve (dim
    // n_theta + p = 21) stalled ~0.24 deviance units short of the optimum along
    // the weakly-identified intercept↔RE direction; profiling β collapses that
    // valley and stage 2 polishes from the PQL point to the Laplace optimum.
    // Each eval re-seeds `ws.beta` from the same fixed β₀, so the stage-1
    // objective is a deterministic function of θ alone (the order-free
    // requirement the dense stage 1 meets through its incumbent snapshots).
    let n_eval_stage1;
    {
        let npt1 = if n_theta >= 3 {
            (3 * n_theta).div_ceil(2) + 1
        } else {
            2 * n_theta + 1
        };
        // MIRRORS `config_stage1` in `GlmmWorkspace::from_groupings` — both
        // feed through the shared `apply_campaign_overrides` tail.
        let mut config1 = bobyqa::Config {
            rho_begin,
            rho_end: crate::lmm::GLMM_RHO_END,
            npt: npt1,
            ..bobyqa::Config::new(n_theta)
        };
        crate::lmm::apply_campaign_overrides(&mut config1, n_theta);
        let mut solver1 = bobyqa::Bobyqa::new(n_theta, config1)
            .expect("BOBYQA config constants are valid by construction");
        let beta0: Vec<f64> = params[n_theta..].to_vec();
        let mut theta1: Vec<f64> = params[..n_theta].to_vec();
        let out1 = solver1.minimize(
            |theta| {
                ws.beta[..p].copy_from_slice(&beta0);
                sparse_glmm_deviance(family, nb_theta, theta, &mut ws, xm, y, n, true)
            },
            &mut theta1,
            &lower[..n_theta],
            &upper[..n_theta],
        );
        n_eval_stage1 = out1.n_eval;
        // Warm-start stage 2 at (θ̂₁, β̂(θ̂₁)): one more Profile eval at the
        // incumbent θ̂₁ leaves the profiled β in ws.beta. A non-finite eval
        // (never seen at an incumbent) just keeps the stage-1-independent seed.
        ws.beta[..p].copy_from_slice(&beta0);
        let d1 = sparse_glmm_deviance(family, nb_theta, &theta1, &mut ws, xm, y, n, true);
        if d1.is_finite() {
            params[..n_theta].copy_from_slice(&theta1);
            for (slot, &b) in params[n_theta..].iter_mut().zip(&ws.beta[..p]) {
                *slot = b.clamp(-crate::glmm::BETA_BOX, crate::glmm::BETA_BOX);
            }
        }
    }

    // STAGE 2 — joint [θ | β] polish on the true Laplace objective (β-Fixed
    // per eval), the dense kernel's stage-2 shape. Only this stage's status
    // feeds `converged`. MIRRORS the joint config in
    // `GlmmWorkspace::from_groupings` — both feed through the shared
    // `apply_campaign_overrides` tail.
    let mut config = bobyqa::Config {
        rho_begin,
        rho_end: crate::lmm::GLMM_RHO_END,
        ..bobyqa::Config::new(n_theta + p)
    };
    crate::lmm::apply_campaign_overrides(&mut config, n_theta + p);
    let mut solver = bobyqa::Bobyqa::new(n_theta + p, config)
        .expect("BOBYQA config constants are valid by construction");
    let out = solver.minimize(
        |gamma| sparse_glmm_deviance(family, nb_theta, gamma, &mut ws, xm, y, n, false),
        &mut params,
        &lower,
        &upper,
    );
    debug_assert!(out.status != Status::InvalidArgs);
    let mut ok = matches!(out.status, Status::Converged);

    // Diagonal-θ pin (mirror `glmm::fit_glmm`; β never pins).
    let mut pinned = false;
    if ok {
        for &ti in g.diagonal_theta() {
            if params[ti] <= crate::lmm::PIN_THETA {
                params[ti] = 0.0;
                pinned = true;
            }
        }
    }
    let n_eval = n_eval_stage1 + out.n_eval;
    // Pinned-γ̂ re-eval: refreshes W̃/û/μ̂/A for the inference reads below, and its
    // finite deviance is the degenerate-fit witness (dense kernel's guard).
    let mut final_deviance = f64::INFINITY;
    if ok {
        final_deviance = sparse_glmm_deviance(family, nb_theta, &params, &mut ws, xm, y, n, false);
        ok = final_deviance.is_finite();
    }
    if !ok {
        return (sparse_glmm_nan_fit(p, n_theta), f64::INFINITY);
    }

    let beta: Vec<f64> = params[n_theta..].to_vec();

    // tau2 / dispersion / varcorr off the converged state, BEFORE the FD-Hessian
    // perturbs the workspace (mirrors `fit::fit_glmm`'s mapping, including
    // Gamma's pwrss/n τ² scale and Pearson dispersion).
    let sigma_sq = crate::family::glmm_sigma_sq(
        family,
        &y[..n],
        &ws.prob[..n],
        &ws.u[..ws.k],
        Some(&ws.prior_w[..n]),
    );
    let tau2: Vec<f64> = params[..n_theta]
        .iter()
        .map(|&t| t * t * sigma_sq)
        .collect();
    let dispersion = match family {
        crate::Family::Gamma { .. } => match opts.dispersion {
            Some(v) => v,
            None => crate::family::pearson_dispersion(
                &y[..n],
                &ws.prob[..n],
                family,
                nb_theta,
                n,
                p,
                Some(&ws.prior_w[..n]),
            ),
        },
        _ => 1.0,
    };
    // σ̂²-scaled like tau2 above (lme4 VarCorr; σ̂² ≡ 1 for φ≡1 families —
    // mirrors `fit::fit_glmm`'s varcorr, change together).
    let varcorr = crate::fit::assemble_varcorr(&params[..n_theta], &g, sigma_sq);
    // μ̂/û/loglik captured HERE, off the same converged state as tau2 above —
    // the FD-Hessian arm below perturbs ws.prob/ws.u and only its fallback
    // restores them (mirrors `fit::fit_glmm`'s read discipline).
    let fitted = ws.prob[..n].to_vec();
    let ranef = crate::fit::assemble_ranef_sparse(&params[..n_theta], &g, &ws.u[..ws.k]);
    let loglik = crate::fit::glmm_loglik(
        family,
        nb_theta,
        final_deviance,
        &y[..n],
        Some(&ws.prior_w[..n]),
    );

    // SE per WaldSe arm. The Rx Schur reads the converged W̃/A the re-eval left;
    // the FD-Hessian perturbs the workspace, so its Rx FALLBACK re-evals at γ̂
    // first to restore that state.
    let mut se = vec![f64::NAN; p];
    let mut vcov = crate::fit::nan_vcov(p);
    let mut stddev_se = vec![f64::NAN; n_theta];
    let cov_from_schur = |schur: Mat<f64>, se: &mut [f64], vcov: &mut Vec<Vec<f64>>| -> bool {
        // σ̂²·(S_β)⁻¹ from chol(S_β) (mirror the dense Rx arm, including Gamma's
        // σ̂² on the RX vcov — lme4 `vcov(use.hessian=FALSE)`). SE is its
        // diagonal, so the shared helper's forward solve serves both.
        let sc = match schur.as_ref().llt(faer::Side::Lower) {
            Ok(c) => c,
            Err(_) => return false,
        };
        *vcov = crate::fit::vcov_from_chol(sc.L(), p, &opts.target_indices, sigma_sq);
        for &tj in &opts.target_indices {
            let tj = tj as usize;
            let vd = vcov[tj][tj];
            if vd.is_finite() && vd >= 0.0 {
                se[tj] = vd.sqrt();
            }
        }
        true
    };
    match opts.wald_se {
        crate::WaldSe::Rx => {
            let schur = match sparse_glmm_schur(&mut ws, xm, n) {
                Some(s) => s,
                None => return (sparse_glmm_nan_fit(p, n_theta), f64::INFINITY),
            };
            if !cov_from_schur(schur, &mut se, &mut vcov) {
                return (sparse_glmm_nan_fit(p, n_theta), f64::INFINITY);
            }
        }
        crate::WaldSe::Hessian => {
            // FD evals (and the fallback's central re-eval below) converge PIRLS at
            // the FD-only tight tol; reset right after the match so the returned-fit
            // workspace never leaks it (see sparse_fd_hessian_cov's contract).
            ws.pirls_tol_override = Some(crate::glmm::PIRLS_TOL_REL_FD);
            match sparse_fd_hessian_cov(
                family,
                nb_theta,
                &params,
                &mut ws,
                xm,
                y,
                n,
                opts.parallel_inner,
            ) {
                Some((cov, tse)) => {
                    for &tj in &opts.target_indices {
                        let tj = tj as usize;
                        let vd = cov[(tj, tj)];
                        if vd.is_finite() && vd >= 0.0 {
                            se[tj] = vd.sqrt();
                        }
                    }
                    // `cov` is Cov(β̂) in full — keep the target block, not only
                    // the diagonal just read. Mirrors the dense Hessian arm,
                    // including taking one value per pair: `cov` is a solve
                    // against an identity, so (a,b)/(b,a) differ in the last
                    // bits and a verbatim copy would not be exactly symmetric.
                    for &ta in &opts.target_indices {
                        for &tb in &opts.target_indices {
                            let (a, b) = (ta as usize, tb as usize);
                            if b > a {
                                continue;
                            }
                            vcov[a][b] = cov[(a, b)];
                            vcov[b][a] = cov[(a, b)];
                        }
                    }
                    stddev_se.copy_from_slice(&tse);
                }
                None => {
                    // RX fallback (the dense `NonPdFellBackToRx` shape): restore the
                    // converged workspace state the FD loop perturbed, then Schur. The
                    // re-eval runs cold-seeded/tight-tol (see `sparse_glmm_deviance`'s
                    // doc comment), so it is not guaranteed to land back at a finite
                    // deviance for a near-degenerate design — mirrors the dense
                    // `fallback!()` macro (`glmm/se.rs`), which also discards this
                    // return value and gates correctness on the Schur PD check below (a
                    // double failure there already routes to `sparse_glmm_nan_fit`).
                    let _ =
                        sparse_glmm_deviance(family, nb_theta, &params, &mut ws, xm, y, n, false);
                    let schur = match sparse_glmm_schur(&mut ws, xm, n) {
                        Some(s) => s,
                        None => return (sparse_glmm_nan_fit(p, n_theta), f64::INFINITY),
                    };
                    if !cov_from_schur(schur, &mut se, &mut vcov) {
                        return (sparse_glmm_nan_fit(p, n_theta), f64::INFINITY);
                    }
                    // No joint Hessian ⇒ no θ-block SE (stays NaN), as the dense
                    // fallback reports. `vcov` IS filled here — the Schur inverse
                    // is a full p×p, same as the dense fallback's `rx_cov_into`.
                }
            }
            ws.pirls_tol_override = None; // never leak the FD tight tol past the SE step
        }
    }

    let mut fit = crate::Fit {
        beta,
        se,
        vcov,
        tau2,
        dispersion,
        converged: true,
        varcorr,
        stddev_se,
        aliased: vec![false; p],
        n_eval,
        deviance: final_deviance,
        singular: pinned,
        loglik,
        df: crate::fit::model_df(family, p, n_theta, opts.dispersion.is_some()),
        reml: false,
        fitted,
        ranef,
        ranef_levels: crate::fit::ranef_level_counts(&g),
    };
    fit.singular = fit.singular || fit.has_negligible_component();
    (fit, final_deviance)
}

/// Sparse-Z negative-binomial GLMM: the over-envelope sibling of
/// `fit::fit_glmm_nb`, same **marginal-θ** profile (`lme4::glmer.nb`) — for
/// each candidate θ the inner `fit_glmm_sparse` re-fits the full GLMM at that
/// fixed θ and its minimized marginal Laplace deviance feeds
/// `logL_marginal(θ) = −½·D(θ) + nb_profile_loglik(y, y, θ, weights)`, maximized
/// over `ln θ` by the shared golden-section bracket (mirrors `fit::fit_glmm_nb`,
/// fit.rs:1806). The spec is θ-free (the NB shape is threaded explicitly per
/// candidate); a warm `start` is irrelevant to the global bracket search,
/// exactly as on the dense path. `dispersion = θ̂`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_glmm_nb_sparse(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &crate::ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    _start: Option<&crate::StartValues>,
    opts: &crate::FitOptions,
) -> crate::Fit {
    let nb_spec = crate::ModelSpec {
        family: crate::Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        },
        re: model.re.clone(),
    };
    let theta = crate::fit::golden_max_ln_theta(|t| {
        let th = t.exp();
        let (_fit, dev) =
            fit_glmm_sparse(x, y, n, p, &nb_spec, cluster_ids, extra_ids, th, None, opts);
        -0.5 * dev + crate::fit::nb_profile_loglik(y, y, th, opts.weights.as_deref())
    });
    let mut fit_result = fit_glmm_sparse(
        x,
        y,
        n,
        p,
        &nb_spec,
        cluster_ids,
        extra_ids,
        theta,
        None,
        opts,
    )
    .0;
    fit_result.dispersion = theta;
    fit_result
}
