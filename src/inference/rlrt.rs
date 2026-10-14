//! The exact restricted likelihood ratio test (RLRT) of one random intercept
//! against the model without it (Crainiceanu & Ruppert 2004): the design's
//! eigenvalues, the observed statistic, the refusals a caller reports in place
//! of an exact reference, the grid kernel that evaluates one null draw, the
//! simulated null with its p-value, and [`exact_rlrt`], which composes them.
//!
//! The model is `y = Xβ + Zb + ε`, `Z` the `n × K` level indicators of one
//! grouping factor, `b ~ N(0, σ_b² I_K)`, `ε ~ N(0, σ² I_n)`, and the null is
//! `λ = σ_b²/σ² = 0` (`λ = θ²` in the crate's parameterization). Under it the
//! RLRT has the distribution of a supremum over `λ` of a function of
//! independent `χ²` draws whose only design inputs are `n − p`, `K` and the
//! eigenvalues `μ` of `ZᵀP₀Z`, `P₀ = I − X(XᵀX)⁻¹Xᵀ` (Crainiceanu & Ruppert
//! 2004, the RLRT representation that follows Theorem 1). The response does
//! not enter, so the null depends on the design alone.
//!
//! Everything here takes numbers: the design summary a `(1 | g)` fit keeps
//! (`crate::fit::DesignSummary`) and the two REML log-likelihoods. It reads no
//! `Fit`, no `x` and no `y`. The method, with a worked example, is in
//! `documentation/algorithms-lmm.md` (Exact RLRT for one random intercept).
//!
//! Crainiceanu, C. M. & Ruppert, D. (2004). Likelihood ratio tests in linear
//! mixed models with one variance component. *Journal of the Royal Statistical
//! Society: Series B (Statistical Methodology)*, 66(1), 165-185.
//!
//! Phipson, B. & Smyth, G. K. (2010). Permutation p-values should never be
//! zero: calculating exact p-values when permutations are randomly drawn.
//! *Statistical Applications in Genetics and Molecular Biology*, 9(1),
//! Article 39.
//!
//! Scheipl, F., Greven, S. & Küchenhoff, H. (2008). Size and power of tests
//! for a zero random effect variance or polynomial regression in additive and
//! linear mixed models. *Computational Statistics & Data Analysis*, 52(7),
//! 3283-3299.

// The exact test's only caller is the two-fit comparison of the formula
// frontend (`formula::anova::compare`), so a build without `formula` compiles
// it unused.
#![cfg_attr(not(feature = "formula"), allow(dead_code))]

use commonstats::dist::{ChiSquared, Normal, Sampler};
use commonstats::rng::{CommonStatsRng, STREAM_TAG_RLRT};
use faer::linalg::matmul::matmul;
use faer::linalg::triangular_solve::solve_lower_triangular_in_place;
use faer::{Accum, Mat, Par};
use std::sync::LazyLock;

use crate::inference::evd::{sym_eigen, EvdError};
use crate::linalg::SeqLlt;

/// A normalized eigenvalue below this is zero (`ε_μ`, relative to `μ_max`):
/// faer's EVD has an absolute error of order `K·ε·μ_max`, and a dropped
/// `μ = 1e-10` moves the profile by at most `e⁸·1e-10 ≈ 3e-7` at the top of
/// the `λ` grid. Also the scale of an all-zero `ZᵀP₀Z` (see [`rlrt_design`]).
pub(crate) const MU_ZERO_REL: f64 = 1e-10;

/// Why the exact RLRT cannot be computed from a design summary. The caller
/// falls back to another reference and reports [`RlrtRefusal::reason`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RlrtRefusal {
    /// `n − p_eff < 1` after the aliased columns are dropped.
    NoResidualDf,
    /// Every `μ` is zero: the fixed effects absorb the groups.
    NoGroupVariation,
    /// The eigendecomposition of `ZᵀP₀Z` failed.
    Eigen(EvdError),
    /// The summary is not one a fit gives: sizes that disagree, a non-finite
    /// entry, a count that is not a row count, a `c` the Cholesky refuses, or
    /// more nonzero `μ` than `n − p_eff`.
    MalformedSummary,
}

impl RlrtRefusal {
    /// One sentence for the caller's note, in plain ASCII.
    pub(crate) fn reason(&self) -> &'static str {
        match self {
            Self::NoResidualDf => {
                "the design leaves no residual degrees of freedom (n - p < 1 after aliased columns are dropped)"
            }
            Self::NoGroupVariation => {
                "the fixed effects absorb the grouping factor, so the random intercept has no variation left to test"
            }
            Self::Eigen(_) => "the eigendecomposition of the design's Z'P0Z failed",
            Self::MalformedSummary => {
                "the design summary is not one a fit gives (sizes that disagree, a non-finite entry, a count that is not a row count, an X'X that is not positive definite, or more nonzero eigenvalues than n - p)"
            }
        }
    }
}

/// The design's part of the RLRT null.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RlrtDesign {
    /// The nonzero eigenvalues of `ZᵀP₀Z` divided by the largest, descending:
    /// `mu[0] == 1`, every value at least [`MU_ZERO_REL`]. Its length is `k`.
    pub(crate) mu: Vec<f64>,
    /// `n − p_eff`, the residual degrees of freedom of the reduced model.
    pub(crate) n_minus_p: usize,
    /// `n − p_eff − k`, the degrees of freedom of the tail `χ²` that also
    /// takes every dropped zero eigenvalue's draw. May be 0.
    pub(crate) tail_df: usize,
}

/// The eigenvalues of `ZᵀP₀Z` from a `(1 | g)` fit's design summary
/// (`crate::fit::DesignSummary`): `counts` (length `K`), `s` (`p_eff × K`,
/// column-major) and `c` (`p_eff × p_eff`, column-major), with `K` from
/// `counts`, `p_eff` from the size of `c` and `n = Σ counts` (inside the
/// envelope there are no weights, so `counts` holds row counts).
///
/// **Why `Z` is never formed.** With `S = XᵀZ` (column `a` the sum of `X`
/// over the rows of level `a`) and `ZᵀZ = diag(n_a)`,
///
/// `ZᵀP₀Z = ZᵀZ − ZᵀX(XᵀX)⁻¹XᵀZ = diag(n_a) − Sᵀ(XᵀX)⁻¹S`.
///
/// `P₀` depends on `X` only through its column space, and the summary is
/// accumulated on `X̃ = X·B·U⁻¹` (`B` the column standardization, `U` the
/// fixed-effect factor, both invertible), which spans the same space, so `s`
/// and `c` in place of `S` and `XᵀX` give the same matrix. With `c = LLᵀ`
/// and `G = L⁻¹s`, `ZᵀP₀Z = diag(counts) − GᵀG`: `K·p_eff` numbers and a
/// `K × K` EVD in place of an `n × K` design.
///
/// A level with no rows has a zero row and column, hence a zero eigenvalue,
/// so `M` is formed over the non-empty levels only. The eigenvalues are
/// clamped at 0 (round-off negatives), sorted descending, divided by the
/// largest, and those below [`MU_ZERO_REL`] dropped: a zero `μ` adds its
/// `χ²₁` to the tail and nothing else (with an intercept in `X`, one for the
/// intercept and one per group-level covariate), so it merges exactly into a
/// tail of `n − p_eff − k` degrees of freedom.
///
/// Refusals, never a panic, in this order: [`RlrtRefusal::MalformedSummary`]
/// for sizes, non-finite entries and counts; [`RlrtRefusal::NoResidualDf`];
/// [`RlrtRefusal::MalformedSummary`] for a `c` the Cholesky refuses;
/// [`RlrtRefusal::Eigen`]; [`RlrtRefusal::NoGroupVariation`] when the largest
/// eigenvalue is at most [`MU_ZERO_REL`] times the largest count
/// (`ZᵀP₀Z ⪯ diag(counts)`, so that is a zero matrix up to round-off); and
/// [`RlrtRefusal::MalformedSummary`] for more nonzero `μ` than `n − p_eff`.
pub(crate) fn rlrt_design(counts: &[f64], s: &[f64], c: &[f64]) -> Result<RlrtDesign, RlrtRefusal> {
    use RlrtRefusal::*;
    let levels = counts.len();
    let p = c.len().isqrt();
    let is_count = |v: f64| v.is_finite() && v >= 0.0 && v.fract() == 0.0;
    if p * p != c.len()
        || s.len() != p * levels
        || !counts.iter().all(|&v| is_count(v))
        || !s.iter().chain(c).all(|v| v.is_finite())
    {
        return Err(MalformedSummary);
    }
    // Row counts, so the sum is exact below 2⁵³.
    let n = counts.iter().sum::<f64>() as usize;
    let n_minus_p = n.checked_sub(p).filter(|&d| d >= 1).ok_or(NoResidualDf)?;

    let chol =
        SeqLlt::new(Mat::from_fn(p, p, |i, j| c[j * p + i]).as_ref()).ok_or(MalformedSummary)?;
    let seen: Vec<usize> = (0..levels).filter(|&a| counts[a] > 0.0).collect();
    let k_seen = seen.len();
    let mut g = Mat::<f64>::from_fn(p, k_seen, |j, b| s[seen[b] * p + j]);
    solve_lower_triangular_in_place(chol.L(), g.as_mut(), Par::Seq);
    let mut m = Mat::<f64>::zeros(k_seen, k_seen);
    matmul(
        m.as_mut(),
        Accum::Replace,
        g.transpose(),
        g.as_ref(),
        -1.0,
        Par::Seq,
    );
    for b in 0..k_seen {
        m[(b, b)] += counts[seen[b]];
        for a in 0..b {
            m[(a, b)] = m[(b, a)];
        }
    }

    let mut mu = sym_eigen(m.as_ref(), false).map_err(Eigen)?.values;
    for v in &mut mu {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    mu.sort_by(|a, b| b.total_cmp(a));
    let top = mu.first().copied().unwrap_or(0.0);
    let largest_count = counts.iter().copied().fold(0.0, f64::max);
    if top <= MU_ZERO_REL * largest_count {
        return Err(NoGroupVariation);
    }
    for v in &mut mu {
        *v /= top;
    }
    mu.retain(|&v| v >= MU_ZERO_REL);
    // `rank(ZᵀP₀Z) ≤ rank(P₀) = n − p_eff`, so only a summary no design
    // gives has more.
    let tail_df = n_minus_p.checked_sub(mu.len()).ok_or(MalformedSummary)?;
    Ok(RlrtDesign {
        mu,
        n_minus_p,
        tail_df,
    })
}

/// The observed RLRT from the REML log-likelihood of the `(1 | g)` fit
/// (`loglik_full`) and of the OLS fit with the same `X` made with
/// `reml = Some(true)` (`loglik_reduced`, the value at `θ = 0`), and whether
/// the `(1 | g)` fit pinned its one component at zero.
///
/// NaN, pinned or not, when either log-likelihood is NaN or `−∞` (a fit that
/// failed or collapsed), never `0.0`. Else `0.0` when `pinned`, by rule: on a
/// pinned fit the two log-likelihoods agree to round-off, and the sign of
/// that round-off must not decide a test whose null has an atom at 0.
/// Otherwise `2·(loglik_full − loglik_reduced)` floored at `0.0`: a search
/// that ended at a lower log-likelihood than at `θ = 0` still tests as 0, and
/// so does a finite `loglik_full` against `loglik_reduced = +∞` (an exact OLS
/// fit). `loglik_full = +∞` against a finite `loglik_reduced` gives `+∞`
/// (`0.0` when pinned), though no `(1 | g)` fit reports `+∞`: `lmm_loglik`
/// maps a non-finite deviance to NaN. So on an exact fit (`y` in the column
/// space of `X`), where the `(1 | g)` fit's deviance is non-finite, the pair
/// tests as NaN.
pub(crate) fn rlrt_statistic(loglik_full: f64, loglik_reduced: f64, pinned: bool) -> f64 {
    let failed = |l: f64| l.is_nan() || l == f64::NEG_INFINITY;
    if failed(loglik_full) || failed(loglik_reduced) {
        return f64::NAN;
    }
    if pinned {
        return 0.0;
    }
    let t = 2.0 * (loglik_full - loglik_reduced);
    // `f64::max` would turn a NaN (`+∞ − +∞`) into 0.0.
    if t > 0.0 || t.is_nan() {
        t
    } else {
        0.0
    }
}

/// The `λ` grid of the null's supremum, in units of `1/μ_max` (the design's
/// `μ` are divided by their largest, and the supremum is invariant to a
/// common rescaling of `μ`, which `λ` absorbs): `λ = 0`, then `exp` of 199
/// points equally spaced on `[−10, 8]`. These are RLRsim 3.1-9's
/// `make.lambdagrid` defaults (Scheipl, Greven & Küchenhoff 2008),
/// `c(0, exp(seq(-10, 8, length = 199)))`, built with R's `seq` arithmetic:
/// index 0 holds 0 and index `j ≥ 1` holds `exp(−10 + (j − 1)·(18/198))`,
/// with the exponents at indices 1 and 199 exactly −10 and 8, so the grid
/// matches RLRsim's to `exp`'s rounding. The supremum is taken on the grid
/// with no refinement beyond it.
pub(crate) static LAMBDA_GRID: LazyLock<[f64; 200]> = LazyLock::new(|| {
    let (lo, hi, n1) = (-10.0f64, 8.0f64, 198);
    let step = (hi - lo) / n1 as f64;
    let mut grid = [0.0; 200];
    for (i, g) in grid[1..].iter_mut().enumerate() {
        let log_lambda = match i {
            0 => lo,
            i if i == n1 => hi,
            i => lo + i as f64 * step,
        };
        *g = log_lambda.exp();
    }
    grid
});

/// The grid-dependent part of the profile for one design, so that a draw
/// costs two dot products per grid point. Row `i` (grid point `λ_i`) of the
/// `g × k` row-major tables holds `fn_[i·k + s] = λ_iμ_s/(1 + λ_iμ_s)` and
/// `fd[i·k + s] = 1/(1 + λ_iμ_s)`, and `sl[i] = Σ_s log1p(λ_iμ_s)`.
pub(crate) struct GridTables {
    fn_: Vec<f64>,
    fd: Vec<f64>,
    sl: Vec<f64>,
}

impl GridTables {
    /// The tables of eigenvalues `mu` on `grid` (zeros allowed).
    pub(crate) fn new(mu: &[f64], grid: &[f64]) -> Self {
        let size = grid.len() * mu.len();
        let (mut fn_, mut fd) = (Vec::with_capacity(size), Vec::with_capacity(size));
        let mut sl = Vec::with_capacity(grid.len());
        for &lambda in grid {
            let mut log_sum = 0.0;
            for &m in mu {
                let x = lambda * m;
                fn_.push(x / (1.0 + x));
                fd.push(1.0 / (1.0 + x));
                log_sum += x.ln_1p();
            }
            sl.push(log_sum);
        }
        Self { fn_, fd, sl }
    }

    /// The profile at grid point `i` for one draw (`chi1[s]` the `χ²₁` of
    /// `μ_s`, `chi_tail` the tail `χ²`):
    /// `f_i = (n − p)·log1p(N_i/D_i) − SL[i]` with `N_i = Σ fN[i][s]·χ_s` and
    /// `D_i = Σ fD[i][s]·χ_s + χ_tail` (Crainiceanu & Ruppert 2004, the RLRT
    /// representation that follows Theorem 1). At `λ = 0`, `N = 0` and
    /// `SL = 0`, so `f = +0.0` exactly whenever `D > 0`.
    fn at(&self, i: usize, chi1: &[f64], chi_tail: f64, n_minus_p: f64) -> f64 {
        let k = chi1.len();
        let row = i * k..(i + 1) * k;
        let (mut num, mut den) = (0.0, 0.0);
        for ((&a, &b), &c) in self.fn_[row.clone()].iter().zip(&self.fd[row]).zip(chi1) {
            num += a * c;
            den += b * c;
        }
        n_minus_p * (num / (den + chi_tail)).ln_1p() - self.sl[i]
    }
}

/// One null draw's statistic: the global maximum of the profile over the
/// grid (the definition of the supremum, never a walk that stops at the
/// first decrease) and its grid index. Ties keep the first index. On a grid
/// that starts at `λ = 0` the value is at least `+0.0`, and an index-0
/// maximum is exactly `+0.0`, an exact zero of the null; index
/// `grid.len() − 1` is an edge hit. `chi1` has one value per `μ` of the
/// tables, and `n_minus_p` is `n − p_eff`.
pub(crate) fn max_on_grid(
    tables: &GridTables,
    chi1: &[f64],
    chi_tail: f64,
    n_minus_p: usize,
) -> (f64, usize) {
    debug_assert_eq!(tables.fn_.len(), tables.sl.len() * chi1.len());
    let n_minus_p = n_minus_p as f64;
    let mut best = (tables.at(0, chi1, chi_tail, n_minus_p), 0);
    for i in 1..tables.sl.len() {
        let f = tables.at(i, chi1, chi_tail, n_minus_p);
        if f > best.0 {
            best = (f, i);
        }
    }
    best
}

/// The settings of one simulated null: `nsim` draws from `seed`, both chosen
/// by the caller (this module keeps no RNG state; the defaults the ports
/// take, `orchestrate::DEFAULT_NSIM` draws from `orchestrate::DEFAULT_SEED`,
/// live in the `orchestrate` module), and whether the draws may run over
/// rayon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RlrtConfig {
    /// Draws of the null. `0` is a caller error: a positive statistic then
    /// gets a NaN p-value.
    pub(crate) nsim: usize,
    /// The run seed of every draw's stream.
    pub(crate) seed: u64,
    /// Split the draws over rayon in fixed chunks. Live only in a `parallel`
    /// build on a non-`wasm32` target, and bit-identical to the serial loop.
    pub(crate) parallel_inner: bool,
}

/// The simulated null distribution of the RLRT on one design, built from the
/// design alone by [`rlrt_null`], so any number of observed statistics on
/// that design read their p-value from it ([`RlrtNull::p_value`]). The
/// fields no caller reads (`n_zero`, `seed`, `edge_hits`) exist in tests only.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RlrtNull {
    /// The positive draws, sorted ascending.
    pub(crate) draws_pos: Vec<f64>,
    /// Draws exactly zero (the profile's maximum at `λ = 0`).
    #[cfg(test)]
    pub(crate) n_zero: usize,
    /// Draws in all, `n_zero + draws_pos.len()`.
    pub(crate) nsim: usize,
    /// The seed the draws came from.
    #[cfg(test)]
    pub(crate) seed: u64,
    /// Draws whose maximum sits on the last grid point, where the supremum
    /// may lie beyond the grid.
    #[cfg(test)]
    pub(crate) edge_hits: usize,
}

impl RlrtNull {
    /// The Monte Carlo p-value of an observed statistic `t_obs` (Phipson &
    /// Smyth 2010): `(1 + #{b : T_b ≥ t_obs}) / (nsim + 1)` for `t_obs > 0`,
    /// so never below `1/(nsim + 1)`, and exactly `1` for `t_obs ≤ 0` (the
    /// null's atom at zero holds at least the observed value). NaN for a NaN
    /// `t_obs`, and for a positive `t_obs` against a null of no draws.
    pub(crate) fn p_value(&self, t_obs: f64) -> f64 {
        if t_obs.is_nan() {
            return f64::NAN;
        }
        if t_obs <= 0.0 {
            return 1.0;
        }
        if self.nsim == 0 {
            return f64::NAN;
        }
        let at_least = self.draws_pos.len() - self.draws_pos.partition_point(|&t| t < t_obs);
        (1 + at_least) as f64 / (self.nsim + 1) as f64
    }
}

/// The draws of one null, `null_draws(design, cfg)`, in fixed chunks of this
/// many: the chunk boundaries depend on `nsim` alone, never on the threads.
const NULL_CHUNK: usize = 1024;

/// The null's draws in draw order: draw `b` is `(T_b, i_b)`, the maximum of
/// its profile on [`LAMBDA_GRID`] and the grid index of that maximum
/// ([`max_on_grid`]).
///
/// Draw `b` reads only its own stream,
/// `CommonStatsRng::new_tagged(seed, b, STREAM_TAG_RLRT)`: first one standard
/// normal per `μ`, in the design's descending-`μ` order, each squared to a
/// `χ²₁`; then one `χ²` of `tail_df` degrees of freedom when `tail_df > 0`,
/// else a tail of `0.0` and no draw (`ChiSquared::new(0.0)` is an error). So a
/// draw is a pure function of `(seed, b)` and the design: the first `B` draws
/// of a longer run are the same values, and the rayon split (fixed chunks,
/// each writing its own slice of the output) is bitwise the serial loop.
pub(crate) fn null_draws(design: &RlrtDesign, cfg: &RlrtConfig) -> Vec<(f64, usize)> {
    let tables = GridTables::new(&design.mu, &LAMBDA_GRID[..]);
    let normal = Normal::new(0.0, 1.0).expect("N(0, 1) is a valid normal");
    let tail = (design.tail_df > 0)
        .then(|| ChiSquared::new(design.tail_df as f64).expect("a positive df is valid"));
    // No draw needs a guard against `D = 0` (a NaN profile): every `χ²₁` is
    // positive, as `Normal`'s sampler is `norm_quantile((w + 0.5)/2³²)` for a
    // 32-bit word `w`, never at 1/2, and the two words nearest 1/2 give
    // `|z| = 2.9e-10` (commonstats 0.2.2), so `z² ≥ 8.5e-20`; a design has
    // `k ≥ 1` (or, with `k = 0`, a positive `tail_df`), so `D_i > 0`.
    let fill = |first: usize, out: &mut [(f64, usize)]| {
        let mut chi1 = vec![0.0; design.mu.len()];
        for (b, draw) in (first..).zip(out.iter_mut()) {
            let mut rng = CommonStatsRng::new_tagged(cfg.seed, b as u64, STREAM_TAG_RLRT);
            for c in &mut chi1 {
                let z = normal.sample(&mut rng);
                *c = z * z;
            }
            let chi_tail = tail.as_ref().map_or(0.0, |t| t.sample(&mut rng));
            *draw = max_on_grid(&tables, &chi1, chi_tail, design.n_minus_p);
        }
    };
    let mut draws = vec![(0.0, 0); cfg.nsim];
    let use_par =
        cfg!(all(feature = "parallel", not(target_arch = "wasm32"))) && cfg.parallel_inner;
    if use_par {
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        {
            use rayon::prelude::*;
            draws
                .par_chunks_mut(NULL_CHUNK)
                .enumerate()
                .for_each(|(c, out)| fill(c * NULL_CHUNK, out));
        }
    } else {
        for (c, out) in draws.chunks_mut(NULL_CHUNK).enumerate() {
            fill(c * NULL_CHUNK, out);
        }
    }
    draws
}

/// The simulated null of `design` ([`null_draws`]): the positive draws
/// sorted, the zeros and the grid-edge hits counted, and the `nsim` and seed
/// of `cfg` carried.
pub(crate) fn rlrt_null(design: &RlrtDesign, cfg: &RlrtConfig) -> RlrtNull {
    let draws = null_draws(design, cfg);
    // An index-0 maximum is exactly `+0.0` and any other is positive.
    let mut draws_pos: Vec<f64> = draws.iter().map(|d| d.0).filter(|&t| t > 0.0).collect();
    draws_pos.sort_unstable_by(f64::total_cmp);
    RlrtNull {
        #[cfg(test)]
        n_zero: cfg.nsim - draws_pos.len(),
        draws_pos,
        nsim: cfg.nsim,
        #[cfg(test)]
        seed: cfg.seed,
        #[cfg(test)]
        edge_hits: draws
            .iter()
            .filter(|d| d.1 == LAMBDA_GRID.len() - 1)
            .count(),
    }
}

/// The exact RLRT of one fit, as [`exact_rlrt`] returns it. The fields no
/// caller reads yet (`mass_at_zero`, `mc_se`, `k_eigen`, `edge_hits`) exist
/// in tests only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RlrtResult {
    /// The observed statistic ([`rlrt_statistic`]).
    pub(crate) statistic: f64,
    /// The p-value: [`RlrtNull::p_value`] on a positive statistic, exactly
    /// `1` on a zero one, NaN on a NaN one.
    pub(crate) p_value: f64,
    /// Draws made: the caller's `nsim` on a positive statistic, `0` when
    /// none were made.
    pub(crate) nsim: usize,
    /// The caller's seed.
    pub(crate) seed: u64,
    /// The null's share of exact zeros, `n_zero / nsim`; NaN when no draws
    /// were made.
    #[cfg(test)]
    pub(crate) mass_at_zero: f64,
    /// The Monte Carlo standard error of the p-value, `√(p(1 − p)/nsim)`;
    /// `0` on a zero statistic, whose p-value is exact.
    #[cfg(test)]
    pub(crate) mc_se: f64,
    /// The design's nonzero eigenvalues, `k`.
    #[cfg(test)]
    pub(crate) k_eigen: usize,
    /// Draws whose maximum sits on the last grid point.
    #[cfg(test)]
    pub(crate) edge_hits: usize,
}

/// The exact RLRT of a `(1 | g)` fit against the OLS fit with the same `X`:
/// [`rlrt_design`] on the fit's design summary (`counts`, `s`, `c`), then
/// [`rlrt_statistic`] on `loglik_full` (the REML log-likelihood of the
/// `(1 | g)` fit), `loglik_reduced` (that of the OLS fit made with
/// `reml = Some(true)`, the value at `θ = 0`) and `pinned` (the `(1 | g)`
/// fit pinned its one component at zero), and, only when the statistic is
/// positive, the null of `cfg` ([`rlrt_null`]) and its p-value.
///
/// A zero statistic (pinned, or the full fit no better than `θ = 0`) gives
/// `p = 1` with no draws: `nsim = 0` (and, in tests, `mass_at_zero` NaN and
/// `mc_se = 0`), so no caller reports a draw count that was never made. A
/// NaN statistic gives a NaN p-value, also with no draws. `cfg.nsim = 0` on
/// a positive statistic is a caller error that gives a NaN p-value. A
/// refusal comes before the statistic is read, so a pinned fit on a design
/// the test cannot handle is still refused.
pub(crate) fn exact_rlrt(
    counts: &[f64],
    s: &[f64],
    c: &[f64],
    loglik_full: f64,
    loglik_reduced: f64,
    pinned: bool,
    cfg: &RlrtConfig,
) -> Result<RlrtResult, RlrtRefusal> {
    let design = rlrt_design(counts, s, c)?;
    let statistic = rlrt_statistic(loglik_full, loglik_reduced, pinned);
    // Zero or NaN: no draws. The statistic is never negative.
    if statistic.is_nan() || statistic <= 0.0 {
        let p_value = if statistic == 0.0 { 1.0 } else { f64::NAN };
        return Ok(RlrtResult {
            statistic,
            p_value,
            nsim: 0,
            seed: cfg.seed,
            #[cfg(test)]
            mass_at_zero: f64::NAN,
            #[cfg(test)]
            mc_se: if statistic == 0.0 { 0.0 } else { f64::NAN },
            #[cfg(test)]
            k_eigen: design.mu.len(),
            #[cfg(test)]
            edge_hits: 0,
        });
    }
    let null = rlrt_null(&design, cfg);
    let p_value = null.p_value(statistic);
    Ok(RlrtResult {
        statistic,
        p_value,
        nsim: cfg.nsim,
        seed: cfg.seed,
        #[cfg(test)]
        mass_at_zero: null.n_zero as f64 / cfg.nsim as f64,
        #[cfg(test)]
        mc_se: (p_value * (1.0 - p_value) / cfg.nsim as f64).sqrt(),
        #[cfg(test)]
        k_eigen: design.mu.len(),
        #[cfg(test)]
        edge_hits: null.edge_hits,
    })
}

/// The whole profile `f_0, …, f_{g−1}` of one draw for any explicit `mu`
/// (zeros allowed, as RLRsim keeps them) and `grid`, through the same
/// arithmetic as [`max_on_grid`], so a test can apply another rule for the
/// maximum.
#[cfg(test)]
pub(crate) fn profile(
    mu: &[f64],
    grid: &[f64],
    chi1: &[f64],
    chi_tail: f64,
    n_minus_p: usize,
) -> Vec<f64> {
    assert_eq!(mu.len(), chi1.len(), "one χ²₁ per μ");
    let tables = GridTables::new(mu, grid);
    (0..grid.len())
        .map(|i| tables.at(i, chi1, chi_tail, n_minus_p as f64))
        .collect()
}
