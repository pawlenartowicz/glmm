//! PIRLS module root: `BetaStep`, `refresh_eta_fixed`, `build_coupling_csr`, and the re-exports that keep `crate::glmm::pirls::*` paths stable across the `packed`/`blocked`/`blocked_extras` solve variants.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, LltRegularization};
use faer::linalg::cholesky::llt::solve::solve_in_place;
use faer::{Mat, MatMut, MatRef, Par, Spec};

use super::workspace::{
    glmm_block_chol, glmm_block_solve, PackedScratch, PirlsScratch, StructuredPattern,
    StructuredSchur, StructuredScratch,
};
use super::{PIRLS_MAX_HALVINGS, PIRLS_MAX_ITERS, PIRLS_OSC_RATIO, PIRLS_OSC_TRIGGER};
use crate::scalar::Scalar;
use crate::spec::{BinomialLink, Family};

mod blocked;
mod blocked_extras;
mod packed;

pub(crate) use blocked::pirls_solve_blocked;
pub(crate) use blocked_extras::{
    pirls_solve_blocked_extras, structured_ainv_solve, structured_factor, TailKernel,
};
pub(crate) use packed::{fill_m_vals, packed_m_vals_theta_deriv, pirls_solve_packed};

/// What one `laplace_deviance` call does with β. `Fixed` = β is the caller's input
/// (every SE, derivative and joint-BOBYQA eval). `ProfilePql` = the PQL border, stage 1.
/// `ProfileExact` = the exact-profile border with the log|A| correction and the Laplace
/// merit — the objective is then the exact Laplace β-profile at θ.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BetaMode {
    Fixed,
    ProfilePql,
    ProfileExact,
}

/// β handling for one PIRLS solve. `Fixed` = the behavior verbatim (β is an
/// immutable input; the FD-Hessian path and BOBYQA stage 2 REQUIRE this so the
/// objective stays a function of the caller's β). `Profile` = PQL/stage-1 mode:
/// a δβ Schur-border update runs each iteration and the converged β is written
/// back through `beta`. No default — every call site chooses explicitly.
pub(crate) enum BetaStep<'a> {
    Fixed,
    Profile {
        // `Some` = exact Laplace profile (`BetaMode::ProfileExact`); `None` = the PQL
        // border, verbatim. Set only by `laplace_deviance`.
        exact: Option<&'a mut ExactProfileBufs>,
        xtwx: &'a mut Mat<f64>,      // p×p  C = X'WX          (ws.border.xtwx)
        xtwm: &'a mut Mat<f64>,      // p×k  B' = X'WM         (ws.border.xtwm)
        ainv_mtwx: &'a mut Mat<f64>, // k×p  T = A⁻¹B          (ws.border.ainv_mtwx)
        schur: &'a mut Mat<f64>,     // p×p  S_β               (ws.border.schur)
        beta_rhs: &'a mut [f64],     // len p: X'ρ, then rhs, then δβ in place (ws.beta_rhs)
        beta_prev: &'a mut [f64], // len p: last-accepted β for the halving backtrack (ws.border.beta_prev)
        // Persistent scratch for `schur`'s in-place `cholesky_in_place`, sized once
        // for p×p at workspace construction (ws.border.schur_llt_mem) — avoids the
        // `.llt(Side::Lower)` per-iteration heap allocation on this hot β-Schur step.
        schur_llt_mem: &'a mut MemBuffer,
    },
}

/// Length to allocate for an observed-information twin buffer: `len` where the
/// fit takes the observed step, nothing where it does not. Every read and write
/// of a [`DualStep`] twin sits behind `dual.observed`, the same condition
/// `observed` carries here, so a canonical link never touches one. On a large
/// crossed shape the twins are the biggest buffers in the scratch
/// (`obs_coupling` is `q_core·s·e` elements, 45 `f64` each at
/// `HyperDual<8, 36>`), and sizing them off the fit keeps that allocation and
/// its first-touch page faults off every canonical fit.
pub(crate) fn obs_len(observed: bool, len: usize) -> usize {
    if observed {
        len
    } else {
        0
    }
}

/// Per-solve controls the dual-scalar derivative kernels (`derivative.rs`'s
/// `run_gradient`/`run_hessian`) hand `pirls_solve_blocked` and
/// `pirls_solve_blocked_extras`; `None` on every `f64` fit-path call, which
/// takes the Newton step on its own weights instead
/// (`observed_weights_in_place` before the scatter).
///
/// **Observed-information step (`observed`).** Each PIRLS step solves
/// `u_new = A_obs⁻¹((A_obs − I)u + g)` with `A_obs = M'W_obs M + I`, `W_obs`
/// the observed (Newton) weight `family::observed_weight`, while the in-loop
/// factor and the convergence test stay on the Fisher `A` — the fixed point
/// `ũ` is unchanged, only the path the iterate takes to it; the objective's
/// `log|A|` is the exit refresh's (`evaluate_at_mode`), exact on every link. Why: at the mode the lane fixed-point map
/// `du ← (I − A⁻¹H_uu)·du + b` contracts by `‖I − A⁻¹H_uu‖`, which is 0 for
/// `A_obs = H_uu` (lanes exact in one step, as on a canonical link) but only
/// 0.2–0.5 for the Fisher `A` on a non-canonical link, where the refinement
/// loop needed 6–10 kernel calls per gradient (measured 2026-09-02,
/// cbpp_probit / sim_gamma / sim_probit_large). Canonical links pass
/// `observed = false`: their Fisher `A` already IS `½h_uu`.
///
/// The blocked kernel packs the twin as `s` independent `q_p×q_p` blocks
/// (`obs_blocks`); the structured-extras kernel packs it as the same
/// core-block + crossed-tail Schur split the Fisher factor uses
/// (`obs_core_blocks`/`obs_coupling`/`obs_schur_blk`), because a crossed-tail
/// column couples every cluster and there is no per-cluster block to solve
/// alone. Both packings solve through the same `A_obs⁻¹` step. Every twin
/// below is allocated through [`obs_len`], so a canonical link carries none of
/// them at all.
///
/// **Step floor (`min_iters`).** The mixed-deviance exit fires after two
/// steps once the `f64` value sits at the mode, but the second-order lanes
/// need two steps to become exact and the objective must then be read at
/// that `u` — three steps in one solve, where the value test alone would
/// stop at two and force a second kernel call (two more steps) just to read
/// it. `0` leaves the exit rule untouched.
pub(crate) struct DualStep<T> {
    /// Take the observed-information step (non-canonical links).
    pub(crate) observed: bool,
    /// `s·q_p²` scratch for the observed blocked-path blocks, same layout as
    /// `a_blocks`; untouched when `observed` is false, and untouched on the
    /// structured-extras path, which packs its twin into the three buffers
    /// below plus `obs_rhs` instead.
    pub(crate) obs_blocks: Vec<T>,
    /// `(q_core² · s).max(1)` twin of the structured kernel's `core_blocks`,
    /// same per-cluster lower-triangle layout, scattered from `W_obs`.
    /// Untouched when `observed` is false and on the blocked path.
    pub(crate) obs_core_blocks: Vec<T>,
    /// `(q_core · s · e).max(1)` twin of `coupling`, `C_obs[f·q_core·e + local·e + b]`.
    /// Shares the `coup_cols`/`coup_ptr` CSR pattern with the Fisher coupling:
    /// the pattern is a function of the design and the θ-pin mask, not of `W`.
    pub(crate) obs_coupling: Vec<T>,
    /// `(e²).max(1)` twin of `schur_blk`, lower triangle.
    pub(crate) obs_schur_blk: Vec<T>,
    /// len `k_total`. The observed right-hand side in the `a_rhs` packing —
    /// `[f·q_core + local | k_family + b]` on the structured-extras path,
    /// `f·q_p + local` on the blocked one — then `u_obs = A_obs⁻¹ rhs` in
    /// place. Both paths stage the twin's right-hand side here so the Fisher
    /// `a_rhs` is free to carry its own fold.
    pub(crate) obs_rhs: Vec<T>,
    /// len `n`. Per-row observed IRLS residual `w_obs,i·(Mu)ᵢ + W·working_residᵢ`.
    /// The structured kernel folds `(A − I)u` into its residual before the
    /// scatter, so the observed right-hand side cannot be recovered from
    /// `a_rhs` and needs its own residual, formed while `mu` still holds `(Mu)ᵢ`.
    pub(crate) obs_resid: Vec<T>,
    /// Do not exit before this many steps have run.
    pub(crate) min_iters: usize,
    /// Written by the solve: true iff every step of every block was an
    /// exact-Hessian step, so the caller may read the lanes after this one
    /// call. A non-PD observed factor on a non-canonical link (the observed
    /// weight can go negative on an outlying row) takes it back: that step
    /// fell back to its Fisher factor. On the structured-extras path a non-PD
    /// twin downgrades the WHOLE iteration, not one block: the crossed Schur
    /// couples every cluster, so there is no per-cluster fallback to take.
    ///
    /// `false` means the lanes have only contracted toward the answer, and the
    /// caller's refinement loop (`derivative.rs`'s `run_gradient` /
    /// `run_hessian`) re-enters until they stop moving.
    pub(crate) exact: bool,
}

/// One Laplace `A`-layout: how `M = ZΛ` is stored, how `A = M'WM + I` and
/// `a_rhs = M'r` are assembled from the working weights, how `A` is factored
/// and `log|A|` read, and how `A⁻¹` is applied. Implemented by the blocked,
/// structured-extras and packed-row layouts over borrowed workspace buffers.
/// The three PIRLS loops call these where their inline assembly sits; the exit
/// refresh calls the same two assembly methods after the loop, so the
/// objective's three terms and the factor the Schur fillers inherit describe
/// one iterate.
pub(crate) trait LaplaceFactor<T: Scalar> {
    /// `eta[i] = eta_fixed[i] + (M u)_i` for `i < n`, together with every
    /// layout-owned per-row buffer this layout's [`LaplaceFactor::scatter`]
    /// consumes. The invariant the exit refresh rests on: after
    /// `eta_from_mode(u)`, a family pass over the `eta` it wrote, and
    /// `scatter`, the layout's `A` and `a_rhs` describe the iterate `u`.
    ///
    /// Returns `Σᵢ yᵢηᵢ` off the RAW η, before the family pass clamps it in
    /// place — the fused-identity deviance branch inside that pass consumes
    /// it. Accumulated inside this row pass so no caller needs a second one.
    fn eta_from_mode(&mut self, u: &[T], eta_fixed: &[T], eta: &mut [T], y: &[f64], n: usize) -> T;
    /// The row pass that fills `A` (without `+I`) and `a_rhs` from `w`, `prob`,
    /// `eta`, `y` and the layout's `M`; `dual` is the observed-information twin
    /// target the blocked and structured loops fill in the same pass (`None`
    /// from the exit refresh, always `None` on the packed layout).
    // The row-pass inputs plus the observed-twin target are this trait's
    // contract; the layout owns everything else, so there is nothing left to
    // bundle.
    #[allow(clippy::too_many_arguments)]
    fn scatter(
        &mut self,
        w: &[T],
        prob: &[T],
        eta: &[T],
        y: &[f64],
        prior_w: &[f64],
        weighted: bool,
        n: usize,
        dual: Option<&mut DualStep<T>>,
    );
    /// `+I`, factor in place, return `log|A|`; `None` when not PD. Leaves the
    /// factor for [`LaplaceFactor::solve_in_place`] and for the `se.rs` Schur
    /// fill.
    fn factor_logdet(&mut self) -> Option<T>;
    /// `a_rhs ← A⁻¹ a_rhs` using the factor `factor_logdet` left.
    fn solve_in_place(&mut self);
}

/// One Laplace evaluation at the returned PIRLS iterate: η, μ, W and the
/// deviance at the returned `(u, β)` (β is already in `eta_fixed`), then
/// `A = M'WM + I` from that W, factored, with `log|A|` off the new factor.
/// `D(û) + ‖û‖²` is stationary in u at the mode, `log|A(u)|` is not, so a
/// factor left one Newton step behind puts a first-order error in the
/// objective and in every lane differentiated through it. Returns
/// `(dev, logdet)`, `None` when the refreshed factor is not PD or when a raw η
/// at the refreshed point sits past the link's clamp bounds.
// `eta`, `prob`, `w`, `eta_fixed` and `u` are `PirlsScratch` fields the
// `layout` value passed beside them already holds disjoint `&mut` borrows of
// other fields of, so passing the scratch struct itself would conflict with
// `layout` at every call site.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_at_mode<T: Scalar, L: LaplaceFactor<T>>(
    layout: &mut L,
    family: Family,
    nb_theta: f64,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    eta_fixed: &[T],
    u: &[T],
    eta: &mut [T],
    prob: &mut [T],
    w: &mut [T],
    n: usize,
) -> Option<(T, T)> {
    let yeta = layout.eta_from_mode(u, eta_fixed, eta, y, n);
    let (dev, infeasible) = T::family_pass(
        family,
        nb_theta,
        &mut eta[..n],
        &y[..n],
        &prior_w[..n],
        weighted,
        yeta,
        &mut prob[..n],
        &mut w[..n],
        &mut [],
    );
    // The refreshed point is the loop's POST-step iterate: each loop takes a
    // step (and, in Profile mode, the δβ border move) after accepting an
    // iterate and only then tests the exit band, so no trial evaluation has
    // passed on this η. A raw η past the link's clamp bounds
    // (`family::eta_infeasible`) therefore reaches here. Refusing it hands the
    // solve its failure surface; returning the deviance would report the
    // `clamp_eta`-projected boundary point as the converged answer.
    if infeasible {
        return None;
    }
    // Exact Laplace curvature. `log|A|` must be the curvature of the conditional
    // log-density at the mode, `A_obs = M'W_obs M + I` with
    // `W_obs = −∂²ℓᵢ/∂ηᵢ²` (`family::observed_weight`), not the Fisher `A`:
    // the two coincide on a canonical link and differ on probit, cloglog,
    // Gamma/log and NB/log (`family::exact_curvature_differs`). The `f64` loops
    // already step with `W_obs`; a dual kernel steps through its observed twin
    // and leaves the Fisher weight in `w`, so the refresh converts here. `w`
    // is left holding `W_obs` and the factor is `A_obs`'s, so the Rx Schur
    // fill (the observed information) and the AGQ node scale read the same
    // curvature. No floor: for these links `W_obs ≥ 0` wherever the
    // log-likelihood is concave in η, and a non-PD `A_obs` is the `None`
    // (objective `+∞`) that `factor_logdet` already returns (TMB's convention).
    observed_weights_in_place(family, nb_theta, y, prior_w, eta, prob, w, n);
    layout.scatter(&w[..], &prob[..], &eta[..], y, prior_w, weighted, n, None);
    let logdet = layout.factor_logdet()?;
    Some((dev, logdet))
}

/// Exact-profile scratch for the exact Laplace β-profile inside `pirls_solve_blocked`'s and
/// `pirls_solve_blocked_extras`'s Profile mode. `f64` throughout — the β border
/// is `f64`-only. Sized once per workspace, so the warm path allocates nothing.
pub(crate) struct ExactProfileBufs {
    /// len `k_total`. Holds `g_u = ∂log|A|/∂u` after the row pass, then
    /// `v = Ã⁻¹ g_u` in place after the block solve. On the structured-extras
    /// path both live in the `a_rhs` packing (`[f·q_core + local | k_family + b]`),
    /// NOT the RE-column order `u` uses.
    pub(crate) logdet_u: Vec<f64>,
    /// len p. `c_β = d log|A|/dβ` (direct part, then minus the û path).
    pub(crate) logdet_beta: Vec<f64>,
    /// len `k_total`. Last ACCEPTED `u` (RE-column order, as `u` itself) — the
    /// halving target once the accept decision moves after the block sweep
    /// (`u_prev` then holds the trial).
    pub(crate) u_acc: Vec<f64>,
    /// `e×e` column-major `S⁻¹` (`tail_inv[b·e + a] = (S⁻¹)_{a,b}`), the dense
    /// inverse of the structured path's crossed-tail Schur complement. Rebuilt
    /// every exact-mode structured iteration by `TailKernel::tail_inverse`,
    /// which writes only the blocks of the tail's components; the entries
    /// between components are zeroed once per fit (`prep_glmm_design`) and
    /// stay zero. Length 1 (unread) when `e == 0` and on the blocked path.
    pub(crate) tail_inv: Vec<f64>,
    /// len `e`. Per-row crossed residual `r_i = C_f'(A_f⁻¹ m_c) − m_x`, indexed
    /// by crossed column. Filled and read only under `cfg(test)`, by the row
    /// site's equality oracle, which needs `r_i` as a vector; the shipped row
    /// term is the three-term form over `tail_g` and `tail_h`, so a release
    /// build allocates this and never touches it. Only cluster `f`'s coupling
    /// columns are ever written or read in one row, and each is written before
    /// it is read, so it needs no clearing between rows. A stack array cannot
    /// serve: `e` is a data dimension (181 on grouseticks), not a compile-time
    /// cap.
    pub(crate) tail_r: Vec<f64>,
    /// `s·q_core²`, cluster `f` at `f·q_core²`, row-major: `G_f = C_f S⁻¹ C_f'`,
    /// the full square rather than a triangle (`q_core ≤ MAX_PRIMARY_Q`, and a
    /// triangle would cost the row loop a branch). Rebuilt with `tail_inv`,
    /// every exact-mode structured iteration.
    pub(crate) tail_g: Vec<f64>,
    /// `s·q_core·e`, `H_f[local][b]` at `f·q_core·e + local·e + b` — the same
    /// indexing as the structured kernel's `coupling`, so the row site reuses
    /// that arithmetic verbatim: `H_f = C_f S⁻¹`, written and read only on
    /// cluster `f`'s coupling columns. Rebuilt with `tail_inv`.
    pub(crate) tail_h: Vec<f64>,
    /// f64 mirror of THIS iterate's per-cluster factor, filled once per exact
    /// block and read by every pass below it: `s·q²` in `a_blocks` layout on the
    /// blocked path, `s·q_core²` in `core_blocks` layout on the structured one
    /// (sized for the wider, `q_core ≥ q`). Those buffers are generic `&[T]` (the
    /// derivative kernels instantiate `T = Dual<N>`) while exact mode is f64-only,
    /// and `block_leverage`/`glmm_block_solve` need a plain `&[f64]`; a transmute
    /// is not allowed. Filled per CLUSTER, not per row: `cluster_ids` is the
    /// caller's row order and is not contiguous by cluster, so a per-row copy
    /// cannot be hoisted by a last-seen-cluster check.
    pub(crate) fac_f64: Vec<f64>,
    /// p×p. `½·d²log|A|/dβ²` along the mode path ũ(β), written by
    /// [`logdet_beta_curvature`] and added to the border's `S_β`; zero on a
    /// border step that does not compute it ([`BorderTrust::wants_curvature`]).
    pub(crate) logdet_hess: Mat<f64>,
    /// p×p. The border's `S_β` before that curvature is added, for the trust
    /// region's damped step and, where the sum is not positive definite
    /// before any radius is set, the first radius ([`border_solve`]).
    pub(crate) schur_plain: Mat<f64>,
    /// len p each: the border's right-hand side `g`, kept for the trust
    /// region's predicted change and its damped re-solves, and the damped
    /// solve's scratch ([`border_solve`]).
    pub(crate) trust_g: Vec<f64>,
    pub(crate) trust_q: Vec<f64>,
    /// len `n` each, written by pass A and pass C of the exact block for
    /// [`logdet_beta_curvature`]: the leverage `hᵢ` and `mᵢ'v`.
    pub(crate) curv_h: Vec<f64>,
    pub(crate) curv_sdot: Vec<f64>,
    /// The scratch of [`logdet_beta_curvature`], laid out there: `k·p` rows
    /// of `A⁻¹M'WX` (`curv_tt`), one `p×p` sum (`curv_acc`), one `p` row
    /// (`curv_a`) and `s·p·q_core(q_core+1)/2` cluster sums `P_f`
    /// (`curv_cc`).
    pub(crate) curv_tt: Vec<f64>,
    pub(crate) curv_acc: Vec<f64>,
    pub(crate) curv_a: Vec<f64>,
    pub(crate) curv_cc: Vec<f64>,
    /// The fit's memory of whether the border needs [`logdet_beta_curvature`]:
    /// set when a solve's trouble switched it on, cleared when a later solve
    /// measures that it barely changes the step ([`BorderTrust::wants_curvature`]).
    /// Reset per fit, so a fit's path never depends on an earlier fit that used
    /// the same workspace.
    pub(crate) curv_memory: bool,
    /// p×p factor and len-p step of the plain `S_β` step, for that measurement
    /// ([`border_solve`]).
    pub(crate) plain_fac: Mat<f64>,
    pub(crate) plain_step: Vec<f64>,
}

/// On a link where the exact curvature differs from Fisher
/// (`family::exact_curvature_differs`), overwrite the Fisher working weights a
/// family pass left in `w` with the observed weights `W_obs` the Laplace
/// objective's `A` is built from (`family::observed_weight`, which takes the η
/// form on a tail row). `W_obs ≥ 0` wherever the row's log-likelihood is
/// concave in η — every row of probit, cloglog, Gamma/log and NB/log, and an
/// inverse-Gaussian/log row with μ < 2y — so the `A` built from these weights
/// is positive definite there. A no-op on every
/// other link. Shared by [`evaluate_at_mode`], the three `f64` PIRLS loops
/// (their Newton step), and the packed assembled engine's dual rebuild of the
/// mode state, which must describe the same `A`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn observed_weights_in_place<T: Scalar>(
    family: Family,
    nb_theta: f64,
    y: &[f64],
    prior_w: &[f64],
    eta: &[T],
    prob: &[T],
    w: &mut [T],
    n: usize,
) {
    if !crate::family::exact_curvature_differs(family) {
        return;
    }
    for i in 0..n {
        w[i] = crate::family::observed_weight(
            family, nb_theta, y[i], prior_w[i], eta[i], prob[i], w[i],
        );
    }
}

/// Pass A's `dW/dη` for `c_β` at one row, for the weight `W` the exact
/// border's `A` is built from: `W_obs` where `exact_obj` (its η form on a tail
/// row, as [`observed_weights_in_place`] leaves it), the Fisher weight
/// otherwise. Plain `f64` closed forms at the row's `prob`, the μ the family
/// pass already holds, and at `w`, pass A's prior-weighted step weight, so the
/// row pays no second inverse link — on logit that would be a scalar `exp`
/// per row per iteration beside the family pass's SIMD `sigmoid`, about a
/// quarter of an exact-profile logit fit. A Fisher tail row keeps the
/// `Dual<1>` derivative of `irls_weight_and_resid`, whose η-form tail weight
/// `weight_eta_deriv` does not reproduce on the canonical links.
/// [`row_weight_eta_curv`] forms the curvature's `W′`, `W″` of this same
/// weight — change together.
#[allow(clippy::too_many_arguments)]
pub(crate) fn row_weight_eta_deriv(
    family: Family,
    nb_theta: f64,
    exact_obj: bool,
    y: f64,
    prior_w: f64,
    eta: f64,
    prob: f64,
    w: f64,
) -> f64 {
    use crate::family::{in_tail, weight_eta_deriv};
    if exact_obj {
        // `w_eta` is read only where `W_obs` equals the Fisher weight, which
        // `exact_obj` excludes.
        return crate::family::observed_weight_eta_deriv(
            family, nb_theta, y, prior_w, eta, prob, 0.0,
        );
    }
    if in_tail(family, eta, prob) {
        use crate::dual::Dual;
        let e = Dual::<1> { v: eta, d: [1.0] };
        let (_, wd, _) = crate::family::irls_weight_and_resid(family, nb_theta, y, e);
        return prior_w * wd.d[0];
    }
    // `w` already carries the prior weight, and `weight_eta_deriv` is linear
    // in `w` on every arm.
    weight_eta_deriv(family, nb_theta, eta, prob, w)
}

/// `(dW/dη, d²W/dη²)` at one row, for [`logdet_beta_curvature`], of the same
/// weight as [`row_weight_eta_deriv`]. One `Dual<1>` pass through the closed
/// forms of `dW/dη`, so `d²W/dη²` needs no closed form of its own: on `W_obs`
/// at this row's `prob` with `dμ/dη` as μ's derivative lane, on the Fisher
/// weight through `weight_eta_deriv` at the `Dual<1>` μ and `w` of
/// `irls_weight_and_resid`.
pub(crate) fn row_weight_eta_curv(
    family: Family,
    nb_theta: f64,
    exact_obj: bool,
    y: f64,
    prior_w: f64,
    eta: f64,
    prob: f64,
) -> (f64, f64) {
    use crate::dual::Dual;
    let e = Dual::<1> { v: eta, d: [1.0] };
    let c = if exact_obj {
        let m = Dual::<1> {
            v: prob,
            d: [crate::family::mu_eta(family, eta)],
        };
        crate::family::observed_weight_eta_deriv(family, nb_theta, y, prior_w, e, m, Dual::ZERO)
    } else {
        let (m, w, _) = crate::family::irls_weight_and_resid(family, nb_theta, y, e);
        Dual::<1>::from_f64(prior_w) * crate::family::weight_eta_deriv(family, nb_theta, e, m, w)
    };
    (c.v, c.d[0])
}

/// The three terms of a structured row's crossed-tail leverage
/// `rᵢ'S⁻¹rᵢ = yᵢ'G_f yᵢ − 2·yᵢ'(H_f tᵢ) + tᵢ'S⁻¹tᵢ`, returned as
/// `(yᵢ'G_f yᵢ, yᵢ'(H_f tᵢ), tᵢ'S⁻¹tᵢ)`: `yc = A_f⁻¹m_c`, `tail_g_f` cluster
/// `f`'s `G_f` (`q_core×q_core`), `tail_h` the `H_f` buffer with `coup` its
/// cluster offset, `cols`/`vals` the row's own crossed entries `tᵢ`. The
/// derivation and the index conventions are at pass A of
/// `pirls_solve_blocked_extras`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn row_tail_terms<T: Scalar>(
    yc: &[f64],
    qc: usize,
    tail_g_f: &[f64],
    tail_h: &[f64],
    coup: usize,
    e: usize,
    cols: &[u32],
    vals: &[T],
    tail_inv: &[f64],
) -> (f64, f64, f64) {
    let mut t_gg = 0.0;
    for l1 in 0..qc {
        let mut inner = 0.0;
        for l2 in 0..qc {
            inner += tail_g_f[l1 * qc + l2] * yc[l2];
        }
        t_gg += yc[l1] * inner;
    }
    let mut t_ght = 0.0;
    for (&b, v) in cols.iter().zip(vals) {
        let b = b as usize;
        let mut inner = 0.0;
        for l in 0..qc {
            inner += yc[l] * tail_h[coup + l * e + b];
        }
        t_ght += v.value() * inner;
    }
    let mut t_tst = 0.0;
    for (&b1, v1) in cols.iter().zip(vals) {
        let b1 = b1 as usize;
        let mut inner = 0.0;
        for (&b2, v2) in cols.iter().zip(vals) {
            inner += v2.value() * tail_inv[b1 * e + b2 as usize];
        }
        t_tst += v1.value() * inner;
    }
    (t_gg, t_ght, t_tst)
}

/// The crossed tail of the structured layout as [`logdet_beta_curvature`]
/// reads it: the packed crossed nonzeros of `M` (`cross_col`/`cross_val`,
/// `n_cross` live per row).
pub(crate) struct CrossedTail<'a, T> {
    pub(crate) cross_col: &'a [u32],
    pub(crate) cross_val: &'a [T],
    pub(crate) n_cross: &'a [u8],
    /// Crossed width.
    pub(crate) e: usize,
}

/// `½·d²log|A|/dβ²` along the mode path ũ(β), into `ex.logdet_hess`: the part
/// of the Laplace profile's curvature in β that `S_β` leaves out. The border
/// step is Newton on `L(β) = D(ũ(β), β) + log|A(ũ(β), β)|`, `D = dev + ‖u‖²`;
/// `D`'s half of `½·d²L/dβ²` is `S_β` exactly (ũ is stationary in D), and
/// without this half the step overshoots wherever log|A| dominates the
/// curvature (a few clusters at a large θ, where `S_β` is small in the
/// direction the random effects nearly span).
///
/// Derivation. `A = I + M'WM`, `W = W(η)`, `η = Xβ + Mu`. Along the mode,
/// `dũ/dβ = −A⁻¹M'WX`, so `dη/dβ = X̃ = X − M·A⁻¹M'WX` (the border's `T` is
/// `A⁻¹M'WX`). Differentiating the mode equation `M'ρ(η) = u` twice
/// (`dρ/dη = −W`) gives `d²ũ/dβⱼdβₖ = −A⁻¹M'(W'∘X̃ⱼ∘X̃ₖ)`. With
/// `hᵢ = mᵢ'A⁻¹mᵢ`, `c_β = d log|A|/dβ = X̃'(W'∘h)` (pass A–C of the border)
/// and
/// ```text
///   d²log|A|/dβⱼdβₖ = Σᵢ (W''ᵢhᵢ − W'ᵢ·mᵢ'v)·X̃ᵢⱼX̃ᵢₖ − tr(A⁻¹PⱼA⁻¹Pₖ),
///   v = A⁻¹M'(W'∘h),  Pⱼ = M'diag(W'∘X̃ⱼ)M,
/// ```
/// the first sum from `W''` and from `d²ũ/dβ²` (through `v`, pass B's adjoint
/// solve), the trace from the change of `A⁻¹` inside `c_β`. With `zᵢ = L⁻¹mᵢ`
/// (`LL' = A`) and `aᵢⱼ = W'ᵢX̃ᵢⱼ`, `tr(A⁻¹PⱼA⁻¹Pₖ) = ⟨Rⱼ, Rₖ⟩_F`,
/// `Rⱼ = Σᵢ aᵢⱼzᵢzᵢ'`. On the blocked layout `L` is block-diagonal and
/// `Rⱼ` splits into per-cluster `q×q` blocks `CC_fj = Σ_{i∈f} aᵢⱼcᵢcᵢ'`,
/// `cᵢ = L_f⁻¹mᵢ`, so the trace is `Σ_f ⟨CC_fj, CC_fk⟩` and the result is
/// exact. On the structured layout `zᵢ = [cᵢ; −L_S⁻¹rᵢ]` (`S` the
/// crossed-tail Schur complement) and the product gains two crossed-tail
/// terms, `2·Σ_f tr(V_fj S⁻¹ V_fk')` and `tr(S⁻¹UⱼS⁻¹Uₖ)` (`V_fj`, `Uⱼ` the
/// coupling and tail blocks of `Rⱼ`). They are left out. Each is a Gram
/// matrix over `j`, so positive semidefinite, and it enters with a minus
/// sign: without them the result is an upper bound on the exact curvature
/// (in the Loewner order), exact where no cluster couples to the tail. An
/// upper bound keeps the step no longer than the exact Newton step, which
/// is the side that matters: an overstated curvature only slows the
/// iteration (its rate is `1 − H_true/H_model`), while one below half the
/// true curvature makes plain Newton diverge. Their cost was `p·e³ + p²·e²`
/// per iteration for the dense `S⁻¹Uⱼ` products plus `p·Σ_f q·e_f²` for the
/// `V` blocks, several times the solve's own tail work; measured on the fits
/// where log|A| dominates, the bound needs no more PIRLS iterations than the
/// exact form.
///
/// `W'`, `W''` are those of the step weight (`exact_obj` picks `W_obs`),
/// formed here per row by [`row_weight_eta_curv`] at the `eta`/`prob` pass A
/// read, so an iteration that skips the curvature never pays for them. `hᵢ`
/// (with its crossed-tail part) and `mᵢ'v` come per row from pass A and pass C
/// of the border's exact block (`ex.curv_h`, `curv_sdot`), and the per-cluster
/// factors `L_f` (`ex.fac_f64`) are this iteration's, left by the same block.
/// Exact at the mode; off it, it is the same linearization the rest of the
/// border step makes.
///
/// Order of work. The row pass runs in the rows' own order, so every per-row
/// read streams, and forms no `cᵢ`: it adds `aᵢⱼ·m_cᵢm_cᵢ'` (`m_cᵢ` the
/// core part of `mᵢ`) into per-cluster sums `P_fj`, and one pass over the
/// clusters turns them into `CC_fj = L_f⁻¹P_fjL_f⁻ᵀ`.
///
/// Cost per iteration, `q = q_core`: `n·(p·q + p²/2 + p·q²/2)` for the row
/// pass, plus `p` per crossed nonzero, and `s·(p·q³ + p²·q²/2)` for the
/// cluster pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn logdet_beta_curvature<T: Scalar>(
    ex: &mut ExactProfileBufs,
    x: MatRef<f64>,
    ainv_mtwx: MatRef<f64>,
    m_core: &[T],
    cluster_ids: &[u32],
    g: &crate::lmm::LmmGroupings,
    tail: Option<&CrossedTail<T>>,
    family: Family,
    nb_theta: f64,
    exact_obj: bool,
    y: &[f64],
    prior_w: &[f64],
    eta: &[T],
    prob: &[T],
    n: usize,
    p: usize,
) {
    use crate::consts::MAX_PRIMARY_Q;
    let q = g.primary_q;
    let np = g.nested_per_parent;
    let qc = q + np;
    let s = g.n_primary;
    let prim_width = q * s;
    let k_family = qc * s;
    let e = tail.map_or(0, |t| t.e);
    let qq = qc * (qc + 1) / 2;
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    // RE column of core-block-local column `local` of cluster `f`: the
    // structured kernel's `core_col` (np = 0 on the blocked layout reduces it
    // to `f·q + local`).
    let core_col = |f: usize, local: usize| -> usize {
        if local < q {
            f * q + local
        } else {
            prim_width + f * np + (local - q)
        }
    };
    let ExactProfileBufs {
        fac_f64: fac,
        logdet_hess: hess,
        curv_h: hrow,
        curv_sdot: sdrow,
        curv_tt: tt,
        curv_acc: acc,
        curv_a: av,
        curv_cc: pall,
        ..
    } = ex;
    // Rows of `T = A⁻¹M'WX` in the `a_rhs` packing, each `p` long and
    // contiguous, so a row's `X̃ᵢ = xᵢ − T'mᵢ` is a few axpys.
    for j in 0..p {
        for f in 0..s {
            for local in 0..qc {
                tt[(f * qc + local) * p + j] = ainv_mtwx[(core_col(f, local), j)];
            }
        }
        for b in 0..e {
            tt[(k_family + b) * p + j] = ainv_mtwx[(k_family + b, j)];
        }
    }
    // `acc` accumulates d²log|A|/dβ² on its lower triangle, row-major.
    let acc = &mut acc[..p * p];
    acc.fill(0.0);
    // `P_fj` at `pall[(f·qq + ix)·p + j]`, `ix` packed lower, so one row's
    // update is contiguous in `j`.
    let pall = &mut pall[..s * qq * p];
    pall.fill(0.0);
    let av = &mut av[..p];
    let mut mc = [0.0_f64; MAX_PRIMARY_Q];
    let mut nzl = [0usize; MAX_PRIMARY_Q];
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let mut nnz = 0;
        for local in 0..qc {
            let v = m_core[i * qc + local].value();
            mc[local] = v;
            if v != 0.0 {
                nzl[nnz] = local;
                nnz += 1;
            }
        }
        for (a, xv) in av.iter_mut().zip(x.row(i).iter()) {
            *a = *xv;
        }
        for &local in &nzl[..nnz] {
            let w = mc[local];
            let tr = &tt[(f * qc + local) * p..(f * qc + local + 1) * p];
            for (a, tv) in av.iter_mut().zip(tr) {
                *a -= w * tv;
            }
        }
        if let Some(t) = tail {
            let base = i * g_cap;
            for z in 0..t.n_cross[i] as usize {
                let w = t.cross_val[base + z].value();
                let b = t.cross_col[base + z] as usize;
                let tr = &tt[(k_family + b) * p..(k_family + b + 1) * p];
                for (a, tv) in av.iter_mut().zip(tr) {
                    *a -= w * tv;
                }
            }
        }
        let (wp, wpp) = row_weight_eta_curv(
            family,
            nb_theta,
            exact_obj,
            y[i],
            prior_w[i],
            eta[i].value(),
            prob[i].value(),
        );
        let d = wpp * hrow[i] - wp * sdrow[i];
        for j in 0..p {
            let dj = d * av[j];
            for (h, xl) in acc[j * p..j * p + j + 1].iter_mut().zip(&av[..=j]) {
                *h += dj * xl;
            }
        }
        // `aᵢ = W'ᵢX̃ᵢ` in place of `X̃ᵢ`.
        for a in av.iter_mut() {
            *a *= wp;
        }
        let pf = &mut pall[f * qq * p..(f + 1) * qq * p];
        for a1 in 0..nnz {
            let r = nzl[a1];
            for &cl in &nzl[..=a1] {
                let coef = mc[r] * mc[cl];
                let ix = r * (r + 1) / 2 + cl;
                for (o, aj) in pf[ix * p..(ix + 1) * p].iter_mut().zip(av.iter()) {
                    *o += coef * aj;
                }
            }
        }
    }
    let mut w1 = [0.0_f64; MAX_PRIMARY_Q * MAX_PRIMARY_Q];
    let mut colv = [0.0_f64; MAX_PRIMARY_Q];
    let mut colo = [0.0_f64; MAX_PRIMARY_Q];
    for f in 0..s {
        let fl = &fac[f * qc * qc..(f + 1) * qc * qc];
        // CC_fj = L_f⁻¹P_fjL_f⁻ᵀ, packed lower in place of P_fj.
        let cc = &mut pall[f * qq * p..(f + 1) * qq * p];
        if qc == 1 {
            let r2 = 1.0 / (fl[0] * fl[0]);
            for v in cc.iter_mut() {
                *v *= r2;
            }
        } else {
            for j in 0..p {
                // W = L⁻¹P column by column (`w1[r·qc + c] = W[r,c]`), then
                // row r of CC = W L⁻ᵀ is L⁻¹ applied to row r of W.
                for cl in 0..qc {
                    for (r, cv) in colv.iter_mut().enumerate().take(qc) {
                        let (hi, lo) = if r >= cl { (r, cl) } else { (cl, r) };
                        *cv = cc[(hi * (hi + 1) / 2 + lo) * p + j];
                    }
                    block_forward_solve(fl, qc, &colv[..qc], &mut colo[..qc]);
                    for r in 0..qc {
                        w1[r * qc + cl] = colo[r];
                    }
                }
                for r in 0..qc {
                    block_forward_solve(fl, qc, &w1[r * qc..(r + 1) * qc], &mut colo[..qc]);
                    for cl in 0..=r {
                        cc[(r * (r + 1) / 2 + cl) * p + j] = colo[cl];
                    }
                }
            }
        }
        // −Σ_f ⟨CC_fj, CC_fk⟩: off-diagonal packed entries count twice.
        let mut ix = 0;
        for r in 0..qc {
            for cl in 0..=r {
                let wgt = if cl == r { 1.0 } else { 2.0 };
                let row = &cc[ix * p..(ix + 1) * p];
                for j in 0..p {
                    let cj = wgt * row[j];
                    for (h, rl) in acc[j * p..j * p + j + 1].iter_mut().zip(&row[..=j]) {
                        *h -= cj * rl;
                    }
                }
                ix += 1;
            }
        }
    }
    for j in 0..p {
        for l in 0..=j {
            let h = 0.5 * acc[j * p + l];
            hess[(j, l)] = h;
            hess[(l, j)] = h;
        }
    }
}

/// Trust region on the exact β-profile's border step (Nocedal & Wright,
/// Numerical Optimization, 2nd ed., ch. 4). The border step is Newton on the
/// Laplace profile `L(β)`, whose quadratic model with the exact curvature
/// `H = S_β + ½·d²log|A|/dβ²` and the border's right-hand side `g` predicts
/// the merit's decrease along a step `δ` as `pred = 2g'δ − δ'Hδ` (`L`'s
/// gradient is `−2g` and its Hessian `2H`). Far from the mode that model can be far off:
/// where the profile is nearly flat in β (a large θ, few clusters) a full step
/// moved β₀ by 2.6e4 (4-cluster NB-log fit with a 3-level crossed factor at
/// θ = 1000). So each trial a border step produced is judged against its
/// model, `ρ = (merit before − merit at the trial)/pred`, and the radius on
/// `‖δ‖` follows Algorithm 4.1: `ρ < ¼` shrinks it to a quarter of the step
/// taken, `ρ > ¾` on a step the radius cut doubles it. A step longer than the
/// radius, or one on an `H` that is not positive definite, is the damped
/// `(H + λI)δ = g` with `‖δ‖` on the radius ([`trust_region_step`]).
///
/// The radius starts infinite, so a solve whose steps agree with their model
/// takes the plain Newton step throughout. The merits the solve compares carry
/// their own error (the mode-consistency slack of the accept test), so a
/// disagreement no larger than the accept test's allowance counts as
/// agreement: without that, round-off near the mode would shrink the radius
/// of every solve.
pub(crate) struct BorderTrust {
    /// Radius on `‖δβ‖`; `∞` until a trial first disagrees with its model.
    radius: f64,
    /// A border step is waiting for its trial to be judged.
    pending: bool,
    /// `g'δ` and `δ'Hδ` of that step, its length, and whether the radius cut
    /// it.
    gd: f64,
    dhd: f64,
    len: f64,
    cut: bool,
    /// The fraction of `δ` the iterate took: `1`, less where the damped solve
    /// ended past the radius, halved by the period-2 relaxed step.
    frac: f64,
    /// This solve's border steps carry the log|A| curvature, and how many
    /// border steps it has taken ([`BorderTrust::wants_curvature`]).
    curv: bool,
    steps: u32,
    /// The radius shrank in this solve: a judged trial disagreed with its model
    /// or could not be evaluated.
    shrunk: bool,
    /// The curvature came from the fit's memory, so the solve's first step
    /// measures its effect `‖δ_H − δ_S‖/‖δ_H‖` ([`border_solve`]) into `effect`.
    check: bool,
    effect: f64,
}

/// The border computes the log|A| curvature only where it changes the step
/// ([`BorderTrust::wants_curvature`]). Its row pass is about one more pass
/// over the rows per call, and on most solves it moves the step by a few
/// percent. Without it the plain `S_β` step contracts the β error by about
/// `‖S_β⁻¹·½d²log|A|‖` per iteration, so it matters only where that factor is
/// near or above one (a few clusters at a large θ, most binary fits with few
/// clusters), where the plain step overshoots and cycles. A solve still running
/// after this many iterations is slow for some reason, and the curvature is
/// then cheap next to the iterations it can save; converged exact-profile solves
/// take four to five.
const CURV_SLOW_ITERS: usize = 8;
/// A solve whose curvature came from the fit's memory drops it (and clears the
/// memory) when its first step changes by less than this fraction with it. At
/// that size the plain step converges at least tenfold per iteration, so the
/// curvature saves about one iteration per solve, and one call costs at most
/// about that.
const CURV_RELEASE: f64 = 0.1;

/// The trust-region constants of Nocedal & Wright's Algorithm 4.1 (shrink
/// below `¼`, grow above `¾`, by `¼` and `2`) and Moré & Sorensen's stopping
/// band on `‖δ‖` (within 10 % of the radius).
const TR_SHRINK_BELOW: f64 = 0.25;
const TR_GROW_ABOVE: f64 = 0.75;
const TR_BOUNDARY_TOL: f64 = 0.1;
/// Cap on the damped solve's λ iterations; two or three are usual (N&W §4.3).
const TR_MAX_ITERS: usize = 30;

impl BorderTrust {
    pub(crate) fn new() -> Self {
        BorderTrust {
            radius: f64::INFINITY,
            pending: false,
            gd: 0.0,
            dhd: 0.0,
            len: 0.0,
            cut: false,
            frac: 1.0,
            curv: false,
            steps: 0,
            shrunk: false,
            check: false,
            effect: f64::INFINITY,
        }
    }

    /// Whether this border step computes [`logdet_beta_curvature`]; called once
    /// per border step, before it. The curvature switches on for the rest of the
    /// solve at the first sign that the plain `S_β` model is wrong: the radius
    /// shrank (a judged trial fell short of its model), careful mode (the merit
    /// could not be trusted), the period-2 damping (the step overshoots by about
    /// a factor two), or `CURV_SLOW_ITERS` iterations without converging. Each
    /// switch sets the fit's `memory`, so the next solve starts with the
    /// curvature on: at nearby θ the next solve needs it too, and waiting for
    /// trouble again would cost iterations on every solve of a hard fit. That
    /// solve's first step measures what the curvature does to the step
    /// ([`border_solve`]); below `CURV_RELEASE` the solve drops it and clears
    /// the memory, so a fit that had trouble at one θ pays nothing once it
    /// moves where the plain step is good. A rejected trial is not a trigger of
    /// its own: a judged one always shrinks the radius, and an unjudged one is
    /// the u half of a first step overshooting, which the β model cannot help
    /// (see [`BorderTrust::judge`]).
    pub(crate) fn wants_curvature(
        &mut self,
        it: usize,
        careful: bool,
        damp: bool,
        memory: &mut bool,
    ) -> bool {
        self.steps += 1;
        if self.curv {
            if self.check && self.steps == 2 && self.effect < CURV_RELEASE {
                self.curv = false;
                *memory = false;
            }
        } else if *memory {
            self.curv = true;
            self.check = true;
        } else if self.shrunk || careful || damp || it >= CURV_SLOW_ITERS {
            self.curv = true;
            *memory = true;
        }
        self.curv
    }

    fn set_step(&mut self, gd: f64, lam: f64, len: f64, cut: bool, frac: f64) {
        self.pending = true;
        self.gd = gd;
        // `(H + λI)δ = g` gives `δ'Hδ = g'δ − λ‖δ‖²`.
        self.dhd = gd - lam * len * len;
        self.len = len;
        self.cut = cut;
        self.frac = frac;
    }

    /// The iterate takes only `t` of the pending step.
    pub(crate) fn scale(&mut self, t: f64) {
        self.frac *= t;
    }

    /// Judge the pending step's trial: `l_from` is the merit the step started
    /// from, `from_judged` whether that point was itself accepted by a
    /// comparison, `l_trial` the trial's merit, `allow` the accept test's
    /// allowance for their error. A step from a point accepted unjudged (a
    /// solve's first trial, or the first after careful mode drops its stored
    /// merit) is not judged: from a cold or far warm start the joint step's u
    /// half can overshoot its mode by far more than the β model can say
    /// anything about, and charging that to the β radius would cut the steps
    /// of solves whose β model is sound.
    pub(crate) fn judge(&mut self, l_from: f64, from_judged: bool, l_trial: f64, allow: f64) {
        if !std::mem::take(&mut self.pending) || !from_judged || !l_from.is_finite() {
            return;
        }
        let t = self.frac;
        let pred = 2.0 * t * self.gd - t * t * self.dhd;
        let actual = l_from - l_trial;
        let rho = if !l_trial.is_finite() {
            f64::NEG_INFINITY
        } else if (actual - pred).abs() <= allow {
            1.0
        } else {
            actual / pred
        };
        if rho < TR_SHRINK_BELOW {
            self.shrink_to(TR_SHRINK_BELOW * t * self.len);
        } else if rho > TR_GROW_ABOVE && self.cut {
            self.radius *= 2.0;
        }
    }

    /// A step of zero or non-finite length leaves nothing to shrink to.
    fn shrink_to(&mut self, r: f64) {
        if r > 0.0 && r.is_finite() {
            self.radius = r;
            self.shrunk = true;
        }
    }

    /// The pending step's trial could not be evaluated (a domain-infeasible
    /// η, a non-finite value, or an `A` that is not positive definite): the
    /// step was too long whatever the merit would have said. `from_judged` as
    /// for [`BorderTrust::judge`].
    pub(crate) fn fail(&mut self, from_judged: bool) {
        if std::mem::take(&mut self.pending) && from_judged {
            self.shrink_to(TR_SHRINK_BELOW * self.frac * self.len);
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn llt_factor(m: &mut Mat<f64>, mem: &mut MemBuffer) -> bool {
    cholesky_in_place(
        m.as_mut(),
        LltRegularization::default(),
        Par::Seq,
        MemStack::new(mem),
        Spec::default(),
    )
    .is_ok()
}

fn llt_solve(fac: &Mat<f64>, rhs: &mut [f64], mem: &mut MemBuffer) {
    let p = rhs.len();
    solve_in_place(
        fac.as_ref(),
        MatMut::from_column_major_slice_mut(rhs, p, 1),
        Par::Seq,
        MemStack::new(mem),
    );
}

/// `δβ` in place of `rhs` for the border step; `schur` holds `S_β` on entry
/// and is overwritten. Without `curv` (the PQL border) it is `S_β⁻¹·rhs`.
/// With it (the exact border) the step is Newton on
/// `H = S_β + ½·d²log|A|/dβ²` ([`logdet_beta_curvature`]) while that is
/// positive definite and its step fits the trust radius ([`BorderTrust`]);
/// otherwise the damped step on the radius. Where `H` is not positive definite
/// before any radius is set, the first radius is the length of the `S_β` step,
/// the positive definite part of the curvature. Returns `false` when the step
/// needs `S_β` and `S_β` is not positive definite.
pub(crate) fn border_solve(
    schur: &mut Mat<f64>,
    rhs: &mut [f64],
    llt_mem: &mut MemBuffer,
    curv: Option<(&mut ExactProfileBufs, &mut BorderTrust)>,
) -> bool {
    let p = rhs.len();
    let Some((ex, tr)) = curv else {
        if !llt_factor(schur, llt_mem) {
            return false;
        }
        llt_solve(schur, rhs, llt_mem);
        return true;
    };
    // The curvature's effect on the step, measured on a solve's first step
    // when that solve took the curvature from the fit's memory
    // (`BorderTrust::wants_curvature`). A plain step that cannot be formed
    // counts as a large effect.
    let measure = tr.check && tr.steps == 1;
    if measure {
        ex.plain_fac.copy_from(&*schur);
        let d = &mut ex.plain_step[..p];
        d.copy_from_slice(rhs);
        if llt_factor(&mut ex.plain_fac, llt_mem) {
            llt_solve(&ex.plain_fac, d, llt_mem);
        } else {
            d.fill(f64::NAN);
        }
    }
    ex.schur_plain.copy_from(&*schur);
    for c in 0..p {
        for r in 0..p {
            schur[(r, c)] += ex.logdet_hess[(r, c)];
        }
    }
    let g = &mut ex.trust_g[..p];
    g.copy_from_slice(rhs);
    if llt_factor(schur, llt_mem) {
        llt_solve(schur, rhs, llt_mem);
        if measure {
            let diff: f64 = rhs
                .iter()
                .zip(&ex.plain_step[..p])
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let effect = diff.sqrt() / dot(rhs, rhs).sqrt();
            // NaN (a plain step that could not be formed, or a zero step)
            // keeps the curvature.
            tr.effect = if effect.is_nan() {
                f64::INFINITY
            } else {
                effect
            };
        }
        let len = dot(rhs, rhs).sqrt();
        // A non-finite step is taken as it is: its trial fails, and
        // `BorderTrust::fail` has nothing finite to shrink to.
        if len <= tr.radius || !len.is_finite() {
            tr.set_step(dot(g, rhs), 0.0, len, false, 1.0);
            return true;
        }
    } else if tr.radius == f64::INFINITY {
        schur.copy_from(&ex.schur_plain);
        if !llt_factor(schur, llt_mem) {
            return false;
        }
        rhs.copy_from_slice(g);
        llt_solve(schur, rhs, llt_mem);
        let len = dot(rhs, rhs).sqrt();
        // A zero or non-finite `S_β` step sets no radius; it is taken unjudged.
        if !(len > 0.0 && len.is_finite()) {
            return true;
        }
        tr.radius = len;
    }
    let (lam, len) = trust_region_step(
        &ex.schur_plain,
        &ex.logdet_hess,
        g,
        tr.radius,
        schur,
        llt_mem,
        rhs,
        &mut ex.trust_q[..p],
    );
    let frac = if len > tr.radius {
        tr.radius / len
    } else {
        1.0
    };
    tr.set_step(dot(g, rhs), lam, len, true, frac);
    for v in rhs.iter_mut() {
        *v *= frac;
    }
    true
}

/// The trust-region subproblem `min −2g'δ + δ'Hδ` subject to `‖δ‖ ≤ Δ`, for a
/// Newton step `H⁻¹g` longer than `Δ` or an `H` that is not positive definite:
/// `δ = (H + λI)⁻¹g` with `λ ≥ 0` chosen so `‖δ‖ = Δ`, found by Newton's
/// method on `1/‖δ(λ)‖ − 1/Δ` (Nocedal & Wright, Algorithm 4.3), safeguarded
/// inside `[λ_L, λ_U]` and stopped within 10 % of `Δ` (Moré & Sorensen,
/// "Computing a trust region step", SIAM J. Sci. Stat. Comput. 4(3), 1983).
/// `H = s_beta + logdet_hess`, summed as [`border_solve`] sums it; `fac` and
/// `q` are scratch; `step` receives `δ`. Returns `(λ, ‖δ‖)` of the
/// last positive definite `H + λI`. The hard case (`g` orthogonal to `H`'s
/// lowest eigenvector) is not completed with an eigenvector move: the step
/// then stays inside the radius, which only makes it shorter.
#[allow(clippy::too_many_arguments)]
fn trust_region_step(
    s_beta: &Mat<f64>,
    logdet_hess: &Mat<f64>,
    g: &[f64],
    delta: f64,
    fac: &mut Mat<f64>,
    mem: &mut MemBuffer,
    step: &mut [f64],
    q: &mut [f64],
) -> (f64, f64) {
    let p = g.len();
    let gn = dot(g, g).sqrt();
    if !gn.is_finite() {
        // A non-finite right-hand side gives a non-finite step, whose trial
        // fails.
        step.copy_from_slice(g);
        return (0.0, f64::NAN);
    }
    let h = |r: usize, c: usize| s_beta[(r, c)] + logdet_hess[(r, c)];
    let mut h1 = 0.0_f64;
    let mut neg_diag = 0.0_f64;
    for c in 0..p {
        h1 = h1.max((0..p).map(|r| h(r, c).abs()).sum());
        neg_diag = neg_diag.max(-h(c, c));
    }
    let shifted = |fac: &mut Mat<f64>, lam: f64| {
        for c in 0..p {
            for r in 0..p {
                fac[(r, c)] = h(r, c);
            }
            fac[(c, c)] += lam;
        }
    };
    let mut lo = neg_diag.max(gn / delta - h1).max(0.0);
    let mut hi = gn / delta + h1;
    let mut lam = lo;
    let mut last = (f64::NAN, f64::NAN);
    for _ in 0..TR_MAX_ITERS {
        shifted(fac, lam);
        if !llt_factor(fac, mem) {
            lo = lam;
            lam = (lo * hi).sqrt().max(lo + 0.01 * (hi - lo));
            continue;
        }
        step.copy_from_slice(g);
        llt_solve(fac, step, mem);
        let len = dot(step, step).sqrt();
        last = (lam, len);
        if (len - delta).abs() <= TR_BOUNDARY_TOL * delta || (lam == 0.0 && len <= delta) {
            break;
        }
        if len < delta {
            hi = lam;
        } else {
            lo = lam;
        }
        // q = L⁻¹δ with LL' = H + λI, the lower factor `llt_factor` left.
        for r in 0..p {
            let mut v = step[r];
            for c in 0..r {
                v -= fac[(r, c)] * q[c];
            }
            q[r] = v / fac[(r, r)];
        }
        let next = lam + (len / dot(q, q).sqrt()).powi(2) * (len - delta) / delta;
        lam = if next > lo && next < hi {
            next
        } else {
            (lo * hi).sqrt().max(lo + 0.01 * (hi - lo))
        };
    }
    if last.0.is_nan() {
        // Every λ tried was below `−λ_min(H)`; `λ_U` exceeds it, since
        // `‖H‖₁ ≥ |λ_min(H)|`.
        shifted(fac, hi);
        step.copy_from_slice(g);
        if llt_factor(fac, mem) {
            llt_solve(fac, step, mem);
        }
        last = (hi, dot(step, step).sqrt());
    }
    last
}

/// `h = ‖L⁻¹ m‖²` for one row: the forward half of `glmm_block_solve` on the
/// row-major lower factor `l` (q×q), `m` the row's `q` RE-design entries.
pub(crate) fn block_leverage(l: &[f64], q: usize, m: &[f64]) -> f64 {
    let mut t = [0.0_f64; crate::consts::MAX_PRIMARY_Q];
    for r in 0..q {
        let mut v = m[r];
        for c in 0..r {
            v -= l[r * q + c] * t[c];
        }
        t[r] = v / l[r * q + r];
    }
    t[..q].iter().map(|x| x * x).sum()
}

/// Forward half of `glmm_block_solve` on the row-major lower factor `l` (q×q):
/// `t = L⁻¹m`, written into `t`. `block_leverage` is `‖t‖²` at `f64`; a
/// bilinear form `mᵢ'A⁻¹mⱼ` over the same block is `tᵢ·tⱼ`, which is why the
/// assembled gradient needs the vector and not only its norm.
pub(crate) fn block_forward_solve<T: Scalar>(l: &[T], q: usize, m: &[T], t: &mut [T]) {
    for r in 0..q {
        let mut v = m[r];
        for c in 0..r {
            v -= l[r * q + c] * t[c];
        }
        t[r] = v / l[r * q + r];
    }
}

/// Refill `eta_fixed[i] = offset[i] + Σ_j x[i,j]·β[j]` (the fixed-effect linear
/// predictor). Called once at entry of each PIRLS solve and, in `BetaStep::Profile`,
/// after every β update (the accepted δβ step and each β halving) — the trial
/// evaluation at the top of the loop reads `eta_fixed`, so it must track the
/// current β. `offset` is `FitOptions::offset` (`None` ⇒ this function is
/// byte-identical to the pre-offset version).
fn refresh_eta_fixed<T: crate::scalar::Scalar>(
    x: MatRef<f64>,
    beta: &[T],
    eta_fixed: &mut [T],
    n: usize,
    p: usize,
    offset: Option<&[f64]>,
) {
    for i in 0..n {
        let mut e = T::ZERO;
        for j in 0..p {
            e += T::from_f64(x[(i, j)]) * beta[j];
        }
        eta_fixed[i] = e;
    }
    if let Some(o) = offset {
        for i in 0..n {
            eta_fixed[i] += T::from_f64(o[i]);
        }
    }
}

/// Per-cluster crossed-column pattern (CSR over f): the union of cluster f's
/// rows' cross_col entries — exactly the nonzero-column support of C_f.
/// Counting-sort CSR: counts → prefix → fill (coup_ptr doubles as the write
/// cursors) → shift cursors back → per-cluster sort + dedup-compact (a
/// cluster's rows repeat the same crossed level; a duplicate in the list
/// would double-subtract in the Schur build). e = 0 (nested only) degenerates
/// to an all-empty CSR — n_cross is all zero.
///
/// The pattern is a function of the design AND the θ-pinning mask
/// (`build_packed_m` drops θ=0 crossed groupings from `cross_col`/`n_cross`,
/// but only at `T = f64`; a dual `T` keeps them, so the dual pattern can be
/// wider), so it is fit-invariant only while the pinning mask is: the caller
/// (deviance.rs structured branch) caches it keyed on that mask — which is
/// `f64`-only for the same reason — and rebuilds on transitions, not per
/// eval and not blindly per fit.
pub(crate) fn build_coupling_csr(
    cluster_ids: &[u32],
    cross_col: &[u32],
    n_cross: &[u8],
    s: usize,
    n: usize,
    coup_cols: &mut [u32],
    coup_ptr: &mut [u32],
) {
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    for v in coup_ptr[..s + 1].iter_mut() {
        *v = 0;
    }
    for i in 0..n {
        coup_ptr[cluster_ids[i] as usize + 1] += n_cross[i] as u32;
    }
    for f in 0..s {
        coup_ptr[f + 1] += coup_ptr[f];
    }
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let cbase = i * g_cap;
        for z in 0..n_cross[i] as usize {
            coup_cols[coup_ptr[f] as usize] = cross_col[cbase + z];
            coup_ptr[f] += 1;
        }
    }
    for f in (1..=s).rev() {
        coup_ptr[f] = coup_ptr[f - 1];
    }
    coup_ptr[0] = 0;
    {
        let mut write = 0usize;
        let mut start = 0usize;
        for f in 0..s {
            let end = coup_ptr[f + 1] as usize;
            coup_cols[start..end].sort_unstable();
            coup_ptr[f] = write as u32;
            let mut prev = u32::MAX; // crossed indices are < e ≪ u32::MAX
            for idx in start..end {
                let v = coup_cols[idx];
                if v != prev {
                    coup_cols[write] = v;
                    write += 1;
                    prev = v;
                }
            }
            start = end;
        }
        coup_ptr[s] = write as u32;
    }
}
