//! Loop-tier benchmark harness (cargo example `loop_bench`, needs the
//! `loop_advanced` feature): builds one workspace per shape and refits many
//! simulated datasets on it, the build-once / fit-many pattern of a power
//! simulation. Not a gate; it asserts nothing about speed.
//!
//! Every draw is a new response (and, in the `redraw` variant, new covariates)
//! on a fixed group structure, generated here from fixed seeds and known β and
//! θ. The fit path stays RNG-free: the generator is a small inline xoshiro256**
//! with its own normal, Poisson, gamma and negative-binomial draws, so the file
//! needs no RNG crate and builds against older versions of the crate.
//!
//! Output: JSON Lines, one `header` record per process, one `block` record per
//! (shape, size, variant, start), one `draw` record per fit and one
//! `block_end` record. Field list: the speed-campaign README, section
//! "loop_bench".
//!
//! Every call into the crate sits in `mod api` below. A build against another
//! version of the crate swaps that module only.
//!
//! Usage: `loop_bench --help`.

use std::fmt::Write as _;
use std::io::Write as _;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Harness-side model description (crate-independent)
// ---------------------------------------------------------------------------

/// Response family and link of a shape.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Fam {
    Gaussian,
    Logit,
    Probit,
    Poisson,
    NegBin,
    GammaLog,
}

impl Fam {
    fn name(self) -> &'static str {
        match self {
            Fam::Gaussian => "gaussian",
            Fam::Logit => "binomial_logit",
            Fam::Probit => "binomial_probit",
            Fam::Poisson => "poisson_log",
            Fam::NegBin => "negbin_log",
            Fam::GammaLog => "gamma_log",
        }
    }
}

/// One grouping factor: its level count, the X columns it reads as random
/// slopes, and the generating covariance (standard deviations per component,
/// one correlation for every pair).
#[derive(Clone, Debug)]
struct Grp {
    name: &'static str,
    levels: usize,
    /// `Some(k)`: nested within the primary grouping with `k` children per
    /// parent; `None`: crossed (or the primary itself).
    nested_per_parent: Option<usize>,
    slopes: Vec<usize>,
    sd: Vec<f64>,
    corr: f64,
    /// Per slope, the covariate value Σ is defined at: a draw adds
    /// `b₀ + Σₛ bₛ·(xₛ − centerₛ)` with `b ~ N(0, Σ)`. The covariate's mean
    /// ([`RAW_CENTER`]) on the `raw` variant, 0 everywhere else.
    center: Vec<f64>,
}

impl Grp {
    fn q(&self) -> usize {
        1 + self.slopes.len()
    }
    /// The generating covariance Σ, row-major q×q.
    fn cov(&self) -> Vec<f64> {
        let q = self.q();
        let mut s = vec![0.0; q * q];
        for i in 0..q {
            for j in 0..q {
                let r = if i == j { 1.0 } else { self.corr };
                s[i * q + j] = r * self.sd[i] * self.sd[j];
            }
        }
        s
    }
    /// Lower Cholesky factor of Σ, row-major q×q: what a draw multiplies its
    /// standard normals by.
    fn chol(&self) -> Vec<f64> {
        chol_psd(&self.cov(), self.q())
    }
    /// Lower Cholesky factor, row-major q×q, of the covariance of the
    /// coefficients at covariate value 0, the origin of the caller's design
    /// and so of the crate's θ. `b₀ + Σₛ bₛ(xₛ − cₛ) = (b₀ − Σₛ cₛbₛ) + Σₛ bₛxₛ`,
    /// so the origin coefficients are `N·b`, with `N` the identity except
    /// `N₀,₁₊ₛ = −cₛ`, and their covariance is `N·Σ·Nᵀ`. Equal to
    /// [`Grp::chol`] when every `cₛ` is 0.
    fn chol_origin(&self) -> Vec<f64> {
        if self.center.iter().all(|&c| c == 0.0) {
            return self.chol();
        }
        let q = self.q();
        let sig = self.cov();
        let mut nm = vec![0.0; q * q];
        for i in 0..q {
            nm[i * q + i] = 1.0;
        }
        for (s, &c) in self.center.iter().enumerate() {
            nm[1 + s] = -c;
        }
        // N·Σ, then (N·Σ)·Nᵀ.
        let ns: Vec<f64> = (0..q * q)
            .map(|ij| {
                (0..q)
                    .map(|k| nm[(ij / q) * q + k] * sig[k * q + ij % q])
                    .sum()
            })
            .collect();
        let a: Vec<f64> = (0..q * q)
            .map(|ij| {
                (0..q)
                    .map(|k| ns[(ij / q) * q + k] * nm[(ij % q) * q + k])
                    .sum()
            })
            .collect();
        chol_psd(&a, q)
    }
}

/// Lower Cholesky factor of a symmetric positive semidefinite q×q matrix
/// (row-major). A zero pivot leaves its column zero, so a singular Σ (a
/// variance at the boundary) still factors.
fn chol_psd(a: &[f64], q: usize) -> Vec<f64> {
    let mut l = vec![0.0; q * q];
    for j in 0..q {
        let mut d = a[j * q + j];
        for k in 0..j {
            d -= l[j * q + k] * l[j * q + k];
        }
        let ljj = if d > 0.0 { d.sqrt() } else { 0.0 };
        l[j * q + j] = ljj;
        for i in (j + 1)..q {
            let mut s = a[i * q + j];
            for k in 0..j {
                s -= l[i * q + k] * l[j * q + k];
            }
            l[i * q + j] = if ljj > 0.0 { s / ljj } else { 0.0 };
        }
    }
    l
}

/// Column-major lower-triangular vech of a row-major q×q lower factor:
/// `(0,0),(1,0),…,(q-1,0),(1,1),…`.
fn vech(l: &[f64], q: usize) -> Vec<f64> {
    let mut v = Vec::with_capacity(q * (q + 1) / 2);
    for c in 0..q {
        for r in c..q {
            v.push(l[r * q + c]);
        }
    }
    v
}

/// Inverse of [`vech`]: the row-major q×q lower factor.
fn unvech(v: &[f64], q: usize) -> Vec<f64> {
    let mut l = vec![0.0; q * q];
    let mut k = 0;
    for c in 0..q {
        for r in c..q {
            l[r * q + c] = v[k];
            k += 1;
        }
    }
    l
}

/// Covariate scheme of a shape.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Scheme {
    /// `[1, x1 ~ N(0,1), x2 ~ N(0,1), b ~ Bernoulli(0.5)]` (+ `x3 ~ N(0,1)` at p = 5).
    Std,
    /// Research scales nobody centers: `[1, age ~ U(18,70) per subject,
    /// cond ∈ {0,1} within subject, score ~ N(100, 15)]`.
    Raw,
}

/// Row layout of a shape's group structure.
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// No random effects.
    Flat,
    /// One grouping; `balanced` gives every cluster n/G rows, otherwise sizes
    /// vary between about half and one and a half times the mean.
    Single { balanced: bool },
    /// Subjects × items, one row per pair; `drop` is the share of pairs left out.
    Crossed { drop: f64 },
    /// Parents × children × reps.
    Nested { reps: usize },
}

/// Everything about one (shape, size, variant): the model, the fixed group
/// structure and covariates, and the generating parameters.
#[derive(Clone, Debug)]
struct Design {
    shape: &'static str,
    size: &'static str,
    variant: &'static str,
    fam: Fam,
    nagq: u8,
    scheme: Scheme,
    layout: Layout,
    redraw: bool,
    n: usize,
    p: usize,
    /// `groups[0]` is the primary grouping, the rest are extras in
    /// declaration order. Empty on OLS/GLM.
    groups: Vec<Grp>,
    ids_primary: Vec<u32>,
    ids_extra: Vec<Vec<u32>>,
    /// Row-major n×p design of the fixed draws (redrawn per draw in `redraw`).
    x: Vec<f64>,
    beta: Vec<f64>,
    /// Gaussian residual SD (1 elsewhere).
    sigma: f64,
    nb_size: f64,
    gamma_shape: f64,
    /// Generating standard deviation of the slope components was shrunk to
    /// sit near the boundary.
    boundary: bool,
}

impl Design {
    fn n_theta(&self) -> usize {
        self.groups.iter().map(|g| g.q() * (g.q() + 1) / 2).sum()
    }

    /// θ of the generating model in the crate's start convention: per
    /// grouping (declaration order) the column-major vech of the lower
    /// Cholesky factor of Σ_g at covariate value 0 ([`Grp::chol_origin`]),
    /// divided by σ on the Gaussian (relative factor) and absolute on every
    /// GLMM family.
    fn theta_true(&self) -> Vec<f64> {
        let scale = if self.fam == Fam::Gaussian {
            1.0 / self.sigma
        } else {
            1.0
        };
        let mut t = Vec::new();
        for g in &self.groups {
            for v in vech(&g.chol_origin(), g.q()) {
                t.push(v * scale);
            }
        }
        t
    }

    /// The kernel's blind θ₀ in the same convention: `THETA0` on every
    /// diagonal, 0 off it. A GLMM start at exactly this θ skips the
    /// warm-start guard.
    fn theta_blind(&self) -> Vec<f64> {
        let mut t = Vec::new();
        for g in &self.groups {
            let q = g.q();
            let mut l = vec![0.0; q * q];
            for i in 0..q {
                l[i * q + i] = glmm::THETA0;
            }
            t.extend(vech(&l, q));
        }
        t
    }

    /// Exactly-specified expectations of which outer search the crate runs.
    fn route_expected(&self) -> &'static str {
        if self.groups.is_empty() {
            return if self.fam == Fam::Gaussian {
                "ols"
            } else {
                "irls"
            };
        }
        if self.fam == Fam::Gaussian {
            return if self.sparse_lmm() {
                "lmm_sparse"
            } else {
                "lmm_dense"
            };
        }
        if self.nagq > 1 || self.fam == Fam::GammaLog {
            "joint"
        } else {
            "exact_profile"
        }
    }

    /// Extra-grouping random slopes route the crate's sparse solver.
    fn sparse_lmm(&self) -> bool {
        self.groups.iter().skip(1).any(|g| !g.slopes.is_empty())
    }

    /// A vector-AGQ shape (nagq > 1, more than one random effect per group):
    /// its deviance moves by the quadrature difference under a change of
    /// standardization, so the answer check reads the wider grid band there.
    fn agq_vec(&self) -> bool {
        self.nagq > 1 && self.groups.first().is_some_and(|g| g.q() > 1)
    }
}

// ---------------------------------------------------------------------------
// The inline generator
// ---------------------------------------------------------------------------

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// FNV-1a, to turn a stream label into seed material.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// The variant name the data streams are seeded with: `raw-origin` draws the
/// same covariates and the same standard normals as `raw`, so the two differ
/// only in where Σ is defined.
fn stream_label(variant: &'static str) -> &'static str {
    if variant == "raw-origin" {
        "raw"
    } else {
        variant
    }
}

/// A seed for one named stream: every draw of every shape gets its own, so a
/// draw's data depend only on (seed, shape, size, variant, draw), never on the
/// start mode, the warm-up length or which shapes ran before it. `raw-origin`
/// uses the streams of `raw` ([`stream_label`]).
fn stream_seed(seed: u64, parts: &[&str], index: u64) -> u64 {
    let mut x = seed;
    let mut h = splitmix64(&mut x);
    for p in parts {
        let mut y = h ^ fnv1a(p);
        h = splitmix64(&mut y);
    }
    let mut y = h ^ index;
    splitmix64(&mut y)
}

/// xoshiro256** (Blackman & Vigna 2018), seeded through splitmix64.
struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut x = seed;
        Rng {
            s: [
                splitmix64(&mut x),
                splitmix64(&mut x),
                splitmix64(&mut x),
                splitmix64(&mut x),
            ],
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform on [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform on (0, 1).
    fn uniform_open(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal, Box–Muller (both values used).
    fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let r = (-2.0 * self.uniform_open().ln()).sqrt();
        let a = std::f64::consts::TAU * self.uniform();
        self.spare = Some(r * a.sin());
        r * a.cos()
    }

    /// Poisson(λ): Knuth's product method on pieces of at most 25 (a sum of
    /// independent Poissons is Poisson), so it is exact at any λ.
    fn poisson(&mut self, lambda: f64) -> f64 {
        let mut left = lambda;
        let mut k = 0u64;
        while left > 0.0 {
            let piece = left.min(25.0);
            left -= piece;
            let floor = (-piece).exp();
            let mut prod = self.uniform_open();
            while prod > floor {
                k += 1;
                prod *= self.uniform_open();
            }
        }
        k as f64
    }

    /// Gamma(shape, scale), Marsaglia & Tsang (2000); shape < 1 by the
    /// `U^(1/shape)` boost.
    fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        if shape < 1.0 {
            let u = self.uniform_open();
            return self.gamma(shape + 1.0, scale) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.uniform_open();
            if u < 1.0 - 0.0331 * x * x * x * x || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v * scale;
            }
        }
    }

    /// Negative binomial with mean μ and size k, as the gamma–Poisson mixture.
    fn neg_binomial(&mut self, mu: f64, size: f64) -> f64 {
        let lam = self.gamma(size, mu / size);
        self.poisson(lam)
    }
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// Every shape name, in run order.
const SHAPES: &[&str] = &[
    "ols",
    "glm_logit",
    "lmm_ri",
    "lmm_slope",
    "lmm_crossed",
    "lmm_sparse",
    "glmm_logit_ri",
    "glmm_probit_ri",
    "glmm_pois_ri",
    "glmm_logit_slope",
    "glmm_crossed",
    "glmm_nested",
    "nb_ri",
    "gamma_ri",
    "agq_slope",
];

const VARIANTS: &[&str] = &["base", "redraw", "raw", "raw-origin", "boundary"];

fn has_slope(shape: &str) -> bool {
    matches!(
        shape,
        "lmm_slope" | "lmm_sparse" | "glmm_logit_slope" | "agq_slope"
    )
}

/// Whether `variant` exists for `shape`. `base` and `redraw` exist everywhere,
/// `raw` on every mixed shape, `raw-origin` and `boundary` on the random-slope
/// shapes (without a slope, `raw-origin` would be `raw` itself).
fn variant_ok(shape: &str, variant: &str) -> bool {
    match variant {
        "base" | "redraw" => true,
        "raw" => !matches!(shape, "ols" | "glm_logit"),
        "raw-origin" | "boundary" => has_slope(shape),
        _ => false,
    }
}

/// The raw scheme's covariate means, per design column: the intercept, age
/// U(18, 70) → 44, the alternating 0/1 condition → 0.5, score N(100, 15) →
/// 100. `raw` defines its random-effect Σ there, and [`raw_beta`] its mean
/// response.
const RAW_CENTER: [f64; 4] = [0.0, 44.0, 0.5, 100.0];

/// Raw-scale (research) effect sizes and the intercept that puts the mean
/// response at its target with the covariates at their means (age 44,
/// cond 0.5, score 100).
fn raw_beta(fam: Fam) -> Vec<f64> {
    let (b_age, b_cond, b_score, target) = match fam {
        // RT in ms: 2 ms per year, 30 ms for the condition, -1 ms per score
        // point; mean RT 670 ms (Brysbaert & Stevens 2018, J Cognition 1:9).
        Fam::Gaussian => (2.0, 30.0, -1.0, 670.0),
        // Odds ratio 1.2 per decade of age, 1.5 for the condition, 1.3 per SD
        // of score; prevalence 25% at the means.
        Fam::Logit => (
            1.2f64.ln() / 10.0,
            1.5f64.ln(),
            1.3f64.ln() / 15.0,
            (0.25f64 / 0.75).ln(),
        ),
        // The logit effects divided by 1.7; P(y = 1) = 25% at the means
        // (Φ⁻¹(0.25) = -0.6745).
        Fam::Probit => (
            1.2f64.ln() / 17.0,
            1.5f64.ln() / 1.7,
            1.3f64.ln() / 25.5,
            -0.674_489_750_196_081_7,
        ),
        // Counts with mean 3 at the means.
        Fam::Poisson | Fam::NegBin => (0.005, 0.2, 0.005, 3.0f64.ln()),
        // A positive skewed response (durations in ms) with mean 670.
        Fam::GammaLog => (0.003, 0.05, -0.002, 670.0f64.ln()),
    };
    let b0 = target - b_age * RAW_CENTER[1] - b_cond * RAW_CENTER[2] - b_score * RAW_CENTER[3];
    vec![b0, b_age, b_cond, b_score]
}

/// Builds one (shape, size, variant) design with its fixed group structure
/// and covariates, from the `design` stream of `seed`.
fn make_design(
    shape: &'static str,
    size: &'static str,
    variant: &'static str,
    seed: u64,
) -> Design {
    let large = size == "large";
    let scheme = if matches!(variant, "raw" | "raw-origin") {
        Scheme::Raw
    } else {
        Scheme::Std
    };
    let raw = scheme == Scheme::Raw;
    // `raw` states Σ at the covariate means, `raw-origin` at covariate value 0.
    let re_center = variant == "raw";
    let boundary = variant == "boundary";
    let pick = |s: usize, l: usize| if large { l } else { s };
    // Single-grouping sizes: (rows, clusters).
    let (n_lmm, g_lmm) = (pick(200, 4000), pick(20, 100));
    let (n_glmm, g_glmm) = (pick(200, 2000), pick(20, 50));
    let single = Layout::Single { balanced: !raw };
    let crossed_drop = if raw { 0.1 } else { 0.0 };

    // Generating parameters per family and scheme. `ri` the random-intercept
    // SD, `sl` the slope SD (per unit of the slope column), `rho` their
    // correlation.
    let fam = match shape {
        "ols" | "lmm_ri" | "lmm_slope" | "lmm_crossed" | "lmm_sparse" => Fam::Gaussian,
        "glm_logit" | "glmm_logit_ri" | "glmm_logit_slope" | "glmm_crossed" | "agq_slope" => {
            Fam::Logit
        }
        "glmm_probit_ri" => Fam::Probit,
        "glmm_pois_ri" | "glmm_nested" => Fam::Poisson,
        "nb_ri" => Fam::NegBin,
        "gamma_ri" => Fam::GammaLog,
        _ => unreachable!("unknown shape {shape}"),
    };
    let p = if shape == "ols" { 5 } else { 4 };
    let beta: Vec<f64> = if raw {
        raw_beta(fam)
    } else {
        match fam {
            Fam::Gaussian if p == 5 => vec![0.0, 0.15, 0.10, 0.25, 0.10],
            Fam::Gaussian => vec![0.0, 0.20, 0.12, 0.30],
            Fam::Logit if shape == "glm_logit" => vec![-0.85, 0.30, 0.20, 0.40],
            Fam::Logit => vec![-0.85, 0.35, 0.20, 0.45],
            Fam::Probit => vec![-0.50, 0.20, 0.12, 0.26],
            Fam::Poisson | Fam::NegBin | Fam::GammaLog => vec![3.0f64.ln(), 0.20, 0.10, 0.25],
        }
    };
    // Gaussian residual SD: 1 on the standard scheme, 200 ms on the raw one
    // (DeBruine & Barr 2021, AMPPS 4(1); 150 to 300 ms across RT studies).
    let sigma = if fam == Fam::Gaussian && raw {
        200.0
    } else {
        1.0
    };
    // Random-intercept SD of the primary grouping (response or link scale).
    // Raw: subject SD 100 ms on RT (DeBruine & Barr 2021; Brysbaert & Stevens
    // 2018), σ_u = 1 on the logit scale (latent ICC 0.23; DeBruine & Barr).
    let ri = match (fam, raw) {
        (Fam::Gaussian, false) => 0.5,
        (Fam::Gaussian, true) => 100.0,
        (Fam::Logit, false) => 0.822f64.sqrt(),
        (Fam::Logit, true) => 1.0,
        (Fam::Probit, false) => 0.53,
        (Fam::Probit, true) => 0.5,
        (Fam::Poisson | Fam::NegBin, _) => 0.5,
        (Fam::GammaLog, false) => 0.3,
        (Fam::GammaLog, true) => 0.15,
    };
    // Slope column and slope SD. Standard scheme: slope on x1 (column 1).
    // Raw scheme: on the uncentered score (column 3) for a single grouping,
    // given per SD of score (15 points) and converted per point; on the 0/1
    // condition (column 2) for the crossed sparse shape. Raw slope SDs: 20 ms
    // (inside the 5 to 40 ms of the RT studies), 0.5 on the logit scale.
    // On `raw` these SDs and the correlation hold at the covariate means
    // (`RAW_CENTER`), where the studies report them; `raw-origin` puts the
    // same values at covariate value 0, 6.7 score SDs outside the data, which
    // at the means is an intercept SD of 3.8 against 1 and a correlation of
    // 0.97 on the logit shapes (189 ms and 0.86 on RT): a near-boundary
    // stress test, not typical data.
    let shrink = if boundary { 0.02 } else { 1.0 };
    let (slope_col, slope_sd) = match (fam, raw, shape) {
        (Fam::Gaussian, false, _) => (1, 0.1f64.sqrt()),
        (Fam::Gaussian, true, "lmm_sparse") => (2, 20.0),
        (Fam::Gaussian, true, _) => (3, 20.0 / 15.0),
        (_, false, _) => (1, 0.2f64.sqrt()),
        (_, true, _) => (3, 0.5 / 15.0),
    };
    let slope_sd = slope_sd * shrink;
    let rho = 0.3;
    let grp = |name: &'static str, levels: usize, slopes: Vec<usize>, sd: Vec<f64>| Grp {
        name,
        levels,
        nested_per_parent: None,
        center: slopes
            .iter()
            .map(|&c| if re_center { RAW_CENTER[c] } else { 0.0 })
            .collect(),
        slopes,
        sd,
        corr: rho,
    };

    let (layout, n_rows_hint, groups): (Layout, usize, Vec<Grp>) = match shape {
        "ols" => (Layout::Flat, pick(100, 5000), vec![]),
        "glm_logit" => (Layout::Flat, pick(200, 5000), vec![]),
        "lmm_ri" => (single, n_lmm, vec![grp("subject", g_lmm, vec![], vec![ri])]),
        "lmm_slope" => (
            single,
            n_lmm,
            vec![grp("subject", g_lmm, vec![slope_col], vec![ri, slope_sd])],
        ),
        "lmm_crossed" | "lmm_sparse" => {
            let (s, i) = if raw {
                (pick(30, 100), pick(20, 40))
            } else {
                (pick(20, 100), pick(10, 40))
            };
            // Item intercept SD 60 ms on RT: between Brysbaert & Stevens'
            // 44 ms and DeBruine & Barr's 80 ms.
            let item_ri = if raw { 60.0 } else { 0.3 };
            if shape == "lmm_crossed" {
                (
                    Layout::Crossed { drop: crossed_drop },
                    s * i,
                    vec![
                        grp("subject", s, vec![], vec![ri]),
                        grp("item", i, vec![], vec![item_ri]),
                    ],
                )
            } else {
                let item_sl = if raw { 10.0 } else { 0.2 } * shrink;
                (
                    Layout::Crossed { drop: crossed_drop },
                    s * i,
                    vec![
                        grp("subject", s, vec![slope_col], vec![ri, slope_sd]),
                        grp("item", i, vec![slope_col], vec![item_ri, item_sl]),
                    ],
                )
            }
        }
        "glmm_logit_ri" | "glmm_probit_ri" | "glmm_pois_ri" | "nb_ri" | "gamma_ri" => (
            single,
            n_glmm,
            vec![grp("subject", g_glmm, vec![], vec![ri])],
        ),
        "glmm_logit_slope" | "agq_slope" => (
            single,
            n_glmm,
            vec![grp("subject", g_glmm, vec![slope_col], vec![ri, slope_sd])],
        ),
        "glmm_crossed" => {
            // Declared items first, subjects second, so the crate's size rule
            // swaps them (more subjects than items): this shape exercises the
            // grouping reorder of starts and θ̂ on the base scheme.
            let (s, i) = if raw {
                (pick(30, 50), pick(20, 40))
            } else {
                (pick(20, 50), pick(10, 40))
            };
            (
                Layout::Crossed { drop: crossed_drop },
                s * i,
                vec![
                    grp("item", i, vec![], vec![if raw { 1.0 } else { 0.5 }]),
                    grp("subject", s, vec![], vec![ri]),
                ],
            )
        }
        "glmm_nested" => {
            let (loc, per, reps) = (pick(10, 50), 4, pick(5, 10));
            let mut child = grp("location:brood", loc * per, vec![], vec![0.4]);
            child.nested_per_parent = Some(per);
            (
                Layout::Nested { reps },
                loc * per * reps,
                vec![grp("location", loc, vec![], vec![ri]), child],
            )
        }
        _ => unreachable!(),
    };

    let mut rng = Rng::new(stream_seed(
        seed,
        &[shape, size, stream_label(variant), "design"],
        0,
    ));
    let (ids_primary, ids_extra) = build_rows(layout, &groups, n_rows_hint, &mut rng);
    let n = if groups.is_empty() {
        n_rows_hint
    } else {
        ids_primary.len()
    };
    let mut d = Design {
        shape,
        size,
        variant,
        fam,
        nagq: if shape == "agq_slope" { 7 } else { 1 },
        scheme,
        layout,
        redraw: variant == "redraw",
        n,
        p,
        groups,
        ids_primary,
        ids_extra,
        x: vec![0.0; n * p],
        beta,
        sigma,
        nb_size: 2.0,
        gamma_shape: if raw { 5.0 } else { 4.0 },
        boundary,
    };
    let mut x = std::mem::take(&mut d.x);
    fill_x(&d, &mut x, &mut rng);
    d.x = x;
    d
}

/// The fixed row → level structure of a layout.
fn build_rows(
    layout: Layout,
    groups: &[Grp],
    n_hint: usize,
    rng: &mut Rng,
) -> (Vec<u32>, Vec<Vec<u32>>) {
    match layout {
        Layout::Flat => (vec![], vec![]),
        Layout::Single { balanced } => {
            let g = groups[0].levels;
            let sizes: Vec<usize> = if balanced {
                (0..g)
                    .map(|k| n_hint / g + usize::from(k < n_hint % g))
                    .collect()
            } else {
                // Weights U(0.5, 1.5), scaled to the row total, at least 2 rows each.
                let w: Vec<f64> = (0..g).map(|_| 0.5 + rng.uniform()).collect();
                let tot: f64 = w.iter().sum();
                let mut s: Vec<usize> = w
                    .iter()
                    .map(|wi| ((wi / tot) * n_hint as f64).floor().max(2.0) as usize)
                    .collect();
                let mut k = 0;
                while s.iter().sum::<usize>() < n_hint {
                    s[k % g] += 1;
                    k += 1;
                }
                s
            };
            let mut prim = Vec::with_capacity(n_hint);
            for (c, &sz) in sizes.iter().enumerate() {
                prim.extend(std::iter::repeat_n(c as u32, sz));
            }
            (prim, vec![])
        }
        Layout::Crossed { drop } => {
            // Primary and the first extra are the two crossed factors, in
            // declaration order; glmm_crossed declares items first.
            let (a, b) = (groups[0].levels, groups[1].levels);
            let mut prim = Vec::new();
            let mut extra = Vec::new();
            for i in 0..a {
                for j in 0..b {
                    if drop > 0.0 && rng.uniform() < drop {
                        continue;
                    }
                    prim.push(i as u32);
                    extra.push(j as u32);
                }
            }
            (prim, vec![extra])
        }
        Layout::Nested { reps } => {
            let per = groups[1].nested_per_parent.expect("nested child");
            let mut prim = Vec::new();
            let mut extra = Vec::new();
            for loc in 0..groups[0].levels {
                for c in 0..per {
                    for _ in 0..reps {
                        prim.push(loc as u32);
                        extra.push((loc * per + c) as u32);
                    }
                }
            }
            (prim, vec![extra])
        }
    }
}

/// Fills the row-major design: column 0 the intercept, the rest per scheme.
fn fill_x(d: &Design, x: &mut [f64], rng: &mut Rng) {
    let (n, p) = (d.n, d.p);
    match d.scheme {
        Scheme::Std => {
            for i in 0..n {
                let row = &mut x[i * p..(i + 1) * p];
                row[0] = 1.0;
                row[1] = rng.normal();
                row[2] = rng.normal();
                row[3] = if rng.uniform() < 0.5 { 1.0 } else { 0.0 };
                if p == 5 {
                    row[4] = rng.normal();
                }
            }
        }
        Scheme::Raw => {
            // Age per subject (the primary level, or the subject factor of a
            // crossed design); the condition alternates within subject (and
            // within item on crossed designs, a Latin square); the score
            // varies by row.
            let subj_of = |i: usize| -> usize {
                match d.layout {
                    Layout::Crossed { .. } if d.groups[0].name == "item" => {
                        d.ids_extra[0][i] as usize
                    }
                    _ if d.groups.is_empty() => i,
                    _ => d.ids_primary[i] as usize,
                }
            };
            let n_subj = match d.layout {
                Layout::Crossed { .. } if d.groups[0].name == "item" => d.groups[1].levels,
                _ if d.groups.is_empty() => n,
                _ => d.groups[0].levels,
            };
            let ages: Vec<f64> = (0..n_subj).map(|_| 18.0 + 52.0 * rng.uniform()).collect();
            let mut within = vec![0usize; n_subj];
            for i in 0..n {
                let s = subj_of(i);
                let cond = match d.layout {
                    Layout::Crossed { .. } => {
                        (d.ids_primary[i] as usize + d.ids_extra[0][i] as usize) % 2
                    }
                    _ => {
                        within[s] += 1;
                        within[s] % 2
                    }
                };
                let row = &mut x[i * p..(i + 1) * p];
                row[0] = 1.0;
                row[1] = ages[s];
                row[2] = cond as f64;
                row[3] = 100.0 + 15.0 * rng.normal();
            }
        }
    }
}

/// One draw's response from the generating model (and, in `redraw`, new
/// covariates first), from the draw's own stream.
fn draw_data(d: &Design, x: &mut [f64], y: &mut [f64], rng: &mut Rng) {
    let (n, p) = (d.n, d.p);
    if d.redraw {
        fill_x(d, x, rng);
    }
    let mut eta: Vec<f64> = (0..n)
        .map(|i| {
            let row = &x[i * p..(i + 1) * p];
            row.iter().zip(&d.beta).map(|(a, b)| a * b).sum()
        })
        .collect();
    for (gi, g) in d.groups.iter().enumerate() {
        let q = g.q();
        let l = g.chol();
        let mut b = vec![0.0; g.levels * q];
        for lev in 0..g.levels {
            let z: Vec<f64> = (0..q).map(|_| rng.normal()).collect();
            for r in 0..q {
                b[lev * q + r] = (0..=r).map(|c| l[r * q + c] * z[c]).sum();
            }
        }
        let ids = if gi == 0 {
            &d.ids_primary
        } else {
            &d.ids_extra[gi - 1]
        };
        for i in 0..n {
            let lev = ids[i] as usize;
            let mut e = b[lev * q];
            for (s, &col) in g.slopes.iter().enumerate() {
                e += b[lev * q + 1 + s] * (x[i * p + col] - g.center[s]);
            }
            eta[i] += e;
        }
    }
    for i in 0..n {
        let e = eta[i];
        y[i] = match d.fam {
            Fam::Gaussian => e + d.sigma * rng.normal(),
            Fam::Logit => {
                let pr = 1.0 / (1.0 + (-e).exp());
                if rng.uniform() < pr {
                    1.0
                } else {
                    0.0
                }
            }
            // Latent-variable form of the probit: exact, no normal CDF needed.
            Fam::Probit => {
                if e + rng.normal() > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Fam::Poisson => rng.poisson(e.exp()),
            Fam::NegBin => rng.neg_binomial(e.exp(), d.nb_size),
            Fam::GammaLog => rng.gamma(d.gamma_shape, e.exp() / d.gamma_shape),
        };
    }
}

// ---------------------------------------------------------------------------
// The C arm: data handed over already standardized
// ---------------------------------------------------------------------------

/// Per-draw column statistics of the C arm, by the crate's own rule: the
/// intercept, 0/1 columns and constant columns are left as they are; every
/// other column is centered at its mean and divided by its population SD.
/// The same two passes in the same order as the crate, so the standardized
/// columns come out as the base arm's internal ones.
struct Prestd {
    center: Vec<f64>,
    scale: Vec<f64>,
}

impl Prestd {
    fn compute(x: &[f64], n: usize, p: usize) -> Self {
        let first = &x[..p];
        let mut binary = vec![true; p];
        let mut ones = vec![true; p];
        let mut constant = vec![true; p];
        let mut sum = vec![0.0; p];
        for row in x[..n * p].chunks_exact(p) {
            for (j, &v) in row.iter().enumerate() {
                binary[j] &= v == 0.0 || v == 1.0;
                ones[j] &= v == 1.0;
                constant[j] &= v == first[j];
                sum[j] += v;
            }
        }
        let intercept = ones.iter().position(|&o| o);
        let sw = n as f64;
        let mut center = vec![0.0; p];
        let mut scale = vec![1.0; p];
        let kept: Vec<bool> = (0..p)
            .map(|j| binary[j] || (intercept.is_some() && constant[j]))
            .collect();
        let m: Vec<f64> = (0..p)
            .map(|j| {
                if kept[j] || intercept.is_none() {
                    0.0
                } else {
                    sum[j] / sw
                }
            })
            .collect();
        let mut ss = vec![0.0; p];
        for row in x[..n * p].chunks_exact(p) {
            for (j, &v) in row.iter().enumerate() {
                let dd = v - m[j];
                ss[j] += dd * dd;
            }
        }
        for j in 0..p {
            if kept[j] {
                continue;
            }
            let s = (ss[j] / sw).sqrt();
            if s > 0.0 && s.is_finite() && m[j].is_finite() {
                center[j] = m[j];
                scale[j] = s;
            }
        }
        Prestd { center, scale }
    }

    fn apply(&self, x: &mut [f64], n: usize, p: usize) {
        for j in 0..p {
            let (m, s) = (self.center[j], self.scale[j]);
            if m == 0.0 && s == 1.0 {
                continue;
            }
            for i in 0..n {
                let v = &mut x[i * p + j];
                *v = (*v - m) / s;
            }
        }
    }

    /// β in the caller's (raw) units → β on the standardized design.
    fn beta_to_std(&self, beta: &[f64]) -> Vec<f64> {
        let mut b: Vec<f64> = beta.iter().zip(&self.scale).map(|(b, s)| b * s).collect();
        b[0] += beta
            .iter()
            .zip(&self.center)
            .skip(1)
            .map(|(b, m)| b * m)
            .sum::<f64>();
        b
    }

    /// β on the standardized design → the caller's (raw) units.
    fn beta_to_raw(&self, bt: &[f64]) -> Vec<f64> {
        let mut b: Vec<f64> = bt.iter().zip(&self.scale).map(|(b, s)| b / s).collect();
        let shift: f64 = (1..bt.len()).map(|j| self.center[j] * b[j]).sum();
        b[0] -= shift;
        b
    }

    /// θ vech on one design → the other, per grouping: the random effects of a
    /// block with slope columns S map as `b̃ = N·b`, `N₀ₛ = mₛ`, `Nₛₛ = sₛ`, so
    /// `Σ̃ = N Σ Nᵀ` and θ̃ is the vech of its lower Cholesky factor. `to_std`
    /// picks the direction (`N` or `N⁻¹`). A relative factor maps the same way.
    fn map_theta(&self, d: &Design, theta: &[f64], to_std: bool) -> Vec<f64> {
        let mut out = Vec::with_capacity(theta.len());
        let mut off = 0;
        for g in &d.groups {
            let q = g.q();
            let len = q * (q + 1) / 2;
            let l = unvech(&theta[off..off + len], q);
            off += len;
            let mut nmat = vec![0.0; q * q];
            nmat[0] = 1.0;
            for (k, &c) in g.slopes.iter().enumerate() {
                let (m, s) = (self.center[c], self.scale[c]);
                if to_std {
                    nmat[k + 1] = m;
                    nmat[(k + 1) * q + k + 1] = s;
                } else {
                    nmat[k + 1] = -m / s;
                    nmat[(k + 1) * q + k + 1] = 1.0 / s;
                }
            }
            // M = N·L, Σ̃ = M Mᵀ.
            let mut mm = vec![0.0; q * q];
            for i in 0..q {
                for j in 0..q {
                    mm[i * q + j] = (0..q).map(|k| nmat[i * q + k] * l[k * q + j]).sum();
                }
            }
            let mut sig = vec![0.0; q * q];
            for i in 0..q {
                for j in 0..q {
                    sig[i * q + j] = (0..q).map(|k| mm[i * q + k] * mm[j * q + k]).sum();
                }
            }
            out.extend(vech(&chol_psd(&sig, q), q));
        }
        out
    }

    /// A flattened per-grouping covariance vech (Σ̂ in the standardized
    /// design's units, as `Fit::varcorr` reports it there) back to raw units,
    /// through its Cholesky factor and [`Self::map_theta`]. Empty stays empty.
    fn map_cov(&self, d: &Design, cov: &[f64]) -> Vec<f64> {
        let want: usize = d.groups.iter().map(|g| g.q() * (g.q() + 1) / 2).sum();
        if cov.len() != want {
            return cov.to_vec();
        }
        let mut chol = Vec::with_capacity(cov.len());
        let mut off = 0;
        for g in &d.groups {
            let q = g.q();
            let len = q * (q + 1) / 2;
            let l = unvech(&cov[off..off + len], q);
            off += len;
            // `unvech` filled the lower triangle; mirror it to a full Σ.
            let mut sig = l.clone();
            for i in 0..q {
                for j in (i + 1)..q {
                    sig[i * q + j] = l[j * q + i];
                }
            }
            chol.extend(vech(&chol_psd(&sig, q), q));
        }
        let mapped = self.map_theta(d, &chol, false);
        let mut out = Vec::with_capacity(cov.len());
        let mut off = 0;
        for g in &d.groups {
            let q = g.q();
            let len = q * (q + 1) / 2;
            let l = unvech(&mapped[off..off + len], q);
            off += len;
            let mut sig = vec![0.0; q * q];
            for i in 0..q {
                for j in 0..q {
                    sig[i * q + j] = (0..q).map(|k| l[i * q + k] * l[j * q + k]).sum();
                }
            }
            out.extend(vech(&sig, q));
        }
        out
    }

    /// `2·Σⱼ ln sⱼ`: added to an LMM's REML deviance on the standardized
    /// design, it gives the REML deviance of the raw design (REML is not
    /// invariant to rescaling X; ML is).
    fn reml_shift(&self) -> f64 {
        2.0 * self.scale.iter().map(|s| s.ln()).sum::<f64>()
    }
}

// ---------------------------------------------------------------------------
// Measurement probes (crate-independent)
// ---------------------------------------------------------------------------

/// What `api::fit` brackets around the one `fit_on` call: wall time, thread
/// CPU time, and (macOS) instructions retired and cycles from
/// `proc_pid_rusage`.
#[derive(Default)]
struct Probe {
    full: bool,
    t0: Option<Instant>,
    cpu0: u64,
    ic0: (u64, u64),
    wall_ns: u64,
    cpu_ns: Option<u64>,
    instr: Option<u64>,
    cycles: Option<u64>,
}

impl Probe {
    fn begin(&mut self) {
        if self.full {
            self.ic0 = sys::instr_cycles().unwrap_or((0, 0));
            self.cpu0 = sys::thread_cpu_ns().unwrap_or(0);
        }
        self.t0 = Some(Instant::now());
    }
    fn end(&mut self) {
        let t1 = Instant::now();
        self.wall_ns = t1.duration_since(self.t0.take().expect("begin")).as_nanos() as u64;
        if self.full {
            self.cpu_ns = sys::thread_cpu_ns().map(|c| c.saturating_sub(self.cpu0));
            if let Some((i, c)) = sys::instr_cycles() {
                self.instr = Some(i.saturating_sub(self.ic0.0));
                self.cycles = Some(c.saturating_sub(self.ic0.1));
            }
        }
    }
}

/// OS counters (the `os_counters` feature): plain libc calls with
/// zero-initialized out-structs, the only `unsafe` in this package. Without
/// the feature the stub module below reads every counter as `None`.
#[cfg(feature = "os_counters")]
mod sys {
    #![allow(unsafe_code)]
    /// (instructions retired, cycles) of this process so far (macOS only).
    #[cfg(target_os = "macos")]
    pub fn instr_cycles() -> Option<(u64, u64)> {
        let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V4,
                (&mut info as *mut libc::rusage_info_v4).cast(),
            )
        };
        (rc == 0).then_some((info.ri_instructions, info.ri_cycles))
    }
    #[cfg(not(target_os = "macos"))]
    pub fn instr_cycles() -> Option<(u64, u64)> {
        None
    }

    /// CPU time of the calling thread, ns.
    #[cfg(unix)]
    pub fn thread_cpu_ns() -> Option<u64> {
        let mut ts: libc::timespec = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        (rc == 0).then_some(ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64)
    }
    #[cfg(not(unix))]
    pub fn thread_cpu_ns() -> Option<u64> {
        None
    }

    /// Process (user µs, system µs, peak RSS bytes) from `getrusage`.
    #[cfg(unix)]
    pub fn rusage() -> Option<(u64, u64, u64)> {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
        let us = |t: libc::timeval| t.tv_sec as u64 * 1_000_000 + t.tv_usec as u64;
        // ru_maxrss is bytes on macOS, KiB on Linux.
        let rss = if cfg!(target_os = "macos") {
            ru.ru_maxrss as u64
        } else {
            ru.ru_maxrss as u64 * 1024
        };
        (rc == 0).then_some((us(ru.ru_utime), us(ru.ru_stime), rss))
    }
    #[cfg(not(unix))]
    pub fn rusage() -> Option<(u64, u64, u64)> {
        None
    }
}

/// Without `os_counters`: no counters, no `unsafe`, no libc calls. The records
/// carry wall time; `cpu_ns`, `instr`, `cycles` and the block's `getrusage`
/// fields are null.
#[cfg(not(feature = "os_counters"))]
mod sys {
    pub fn instr_cycles() -> Option<(u64, u64)> {
        None
    }
    pub fn thread_cpu_ns() -> Option<u64> {
        None
    }
    pub fn rusage() -> Option<(u64, u64, u64)> {
        None
    }
}

// ---------------------------------------------------------------------------
// api: every call into the crate. A per-version shim replaces this module.
// ---------------------------------------------------------------------------

/// Start values in the caller's units and declaration order.
#[derive(Clone, Debug, Default)]
struct Start {
    beta: Vec<f64>,
    theta: Vec<f64>,
}

/// What one fit reports back to the harness.
#[derive(Default)]
struct Out {
    converged: bool,
    n_eval: usize,
    theta: Vec<f64>,
    beta: Vec<f64>,
    boundary_hit: u8,
    pinned_mask: u64,
    ill_conditioned: bool,
    pirls_exhausted: u32,
    final_pirls_exhausted: bool,
    hessian_fallback: bool,
    exact_profile_fallback: bool,
    /// Filled by the full readout (`into_fit`) only.
    full: Option<FullOut>,
}

#[derive(Default)]
struct FullOut {
    deviance: f64,
    loglik: f64,
    singular: bool,
    boundary: &'static str,
    dispersion: f64,
    n_eval: usize,
    /// RE covariance per grouping (vech of Σ̂, declaration order), flattened.
    varcorr: Vec<f64>,
    counters: Option<Counters>,
}

/// The `counters` feature's per-fit counts.
#[derive(Default)]
struct Counters {
    stage_evals: [u32; 2],
    shrink_evals: [u32; 2],
    pirls_hist: Vec<u32>,
    agq_evals: u32,
    agq_node_evals: u64,
    nb_nodes: u32,
    nb_evals_total: u64,
}

mod api {
    use super::{Counters, Design, Fam, FullOut, Out, Probe, Start};
    use glmm::loop_advanced::{build_workspace, fit_on, FitWorkspace};
    use glmm::{
        BinomialLink, Data, Family, FitOptions, GammaLink, GroupIds, Grouping, ModelSpec,
        NegBinomialLink, PoissonLink, ReStructure, SeFor, StartValues,
    };

    /// The crate's version this module was written against.
    pub const API: &str = "glmm 0.4.x loop_advanced (build_workspace / fit_on / FitView)";

    pub fn features() -> Vec<&'static str> {
        let mut f = vec!["loop_advanced"];
        if cfg!(feature = "counters") {
            f.push("counters");
        }
        f
    }

    fn family(f: Fam) -> Family {
        match f {
            Fam::Gaussian => Family::Gaussian,
            Fam::Logit => Family::Binomial {
                link: BinomialLink::Logit,
            },
            Fam::Probit => Family::Binomial {
                link: BinomialLink::Probit,
            },
            Fam::Poisson => Family::Poisson {
                link: PoissonLink::Log,
            },
            Fam::NegBin => Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            Fam::GammaLog => Family::Gamma {
                link: GammaLink::Log,
            },
        }
    }

    /// The caller's declaration-order spec of a design.
    fn model(d: &Design) -> ModelSpec {
        let Some(g0) = d.groups.first() else {
            return ModelSpec::fixed(family(d.fam));
        };
        let primary = ReStructure::intercept().slopes(g0.slopes.iter().map(|&c| c as u32));
        let re = d.groups[1..].iter().fold(primary, |re, g| {
            let grouping = match g.nested_per_parent {
                Some(_) => Grouping::nested(),
                None => Grouping::crossed(),
            };
            re.extra(grouping.slopes(g.slopes.iter().map(|&c| c as u32)))
        });
        ModelSpec::mixed(family(d.fam), re)
    }

    /// One shape's built workspace plus everything `fit_on` and `into_fit`
    /// need alongside it.
    pub struct Bench {
        ws: FitWorkspace,
        model: ModelSpec,
        opts: FitOptions,
        start: StartValues,
        n: usize,
        p: usize,
    }

    /// `build_workspace`, once per shape. Every
    /// non-intercept column is a target (the per-coefficient Wald reads a
    /// power simulation makes).
    pub fn build(d: &Design, prestandardized: bool) -> Bench {
        let model = model(d);
        let ids = d
            .ids_extra
            .iter()
            .fold(GroupIds::new(d.ids_primary.clone()), |ids, e| {
                ids.extra(e.clone())
            });
        let mut opts = FitOptions::default();
        opts.se_for = SeFor::Only((1..d.p as u32).collect());
        opts.nagq = d.nagq;
        let mut ws = build_workspace(&model, &ids, d.n, d.p, &opts);
        if prestandardized {
            ws.set_prestandardized(true);
        }
        Bench {
            ws,
            model,
            opts,
            start: StartValues::new(Vec::with_capacity(d.p), Vec::with_capacity(64)),
            n: d.n,
            p: d.p,
        }
    }

    /// One `fit_on` call, bracketed by `probe`, then the readout. `full`
    /// maps the view through `into_fit` (deviance, singular, counters) after
    /// the probe has stopped.
    pub fn fit(
        b: &mut Bench,
        x: &[f64],
        y: &[f64],
        start: Option<&Start>,
        full: bool,
        probe: &mut Probe,
    ) -> Out {
        let Bench {
            ws,
            model,
            opts,
            start: sv,
            n,
            p,
        } = b;
        if let Some(s) = start {
            sv.beta.clear();
            sv.beta.extend_from_slice(&s.beta);
            sv.theta.clear();
            sv.theta.extend_from_slice(&s.theta);
        }
        let sv: Option<&StartValues> = start.map(|_| &*sv);
        probe.begin();
        let view = fit_on(ws, &Data::new(x, y, *n, *p), sv, opts);
        probe.end();
        let diag = view.diagnostics();
        let mut out = Out {
            converged: diag.converged,
            n_eval: view.n_eval(),
            theta: view.theta().to_vec(),
            beta: view.betas().to_vec(),
            boundary_hit: diag.boundary_hit,
            pinned_mask: diag.pinned_components,
            ill_conditioned: diag.ill_conditioned,
            pirls_exhausted: diag.pirls_exhausted,
            final_pirls_exhausted: diag.final_pirls_exhausted,
            hessian_fallback: diag.hessian_fallback,
            exact_profile_fallback: diag.exact_profile_fallback,
            full: None,
        };
        if full {
            let fit = view.into_fit(&Data::new(x, y, *n, *p), model, opts);
            out.full = Some(FullOut {
                deviance: fit.deviance,
                loglik: fit.loglik,
                singular: fit.singular(),
                boundary: match fit.diagnostics.boundary {
                    glmm::Boundary::Interior => "interior",
                    glmm::Boundary::AtBoundary => "at_boundary",
                    glmm::Boundary::NoOptimum => "no_optimum",
                    glmm::Boundary::UpperLimit => "upper_limit",
                    _ => "other",
                },
                dispersion: fit.dispersion,
                n_eval: fit.n_eval,
                varcorr: fit.varcorr.iter().flatten().copied().collect(),
                counters: counters(&fit),
            });
        }
        out
    }

    #[cfg(feature = "counters")]
    fn counters(fit: &glmm::Fit) -> Option<Counters> {
        use glmm::Stage;
        let c = &fit.counters;
        let last = c
            .pirls_hist
            .iter()
            .rposition(|&v| v > 0)
            .map_or(0, |i| i + 1);
        Some(Counters {
            stage_evals: c.stage_evals,
            shrink_evals: [
                c.evals_after_last_improve(Stage::One),
                c.evals_after_last_improve(Stage::Two),
            ],
            pirls_hist: c.pirls_hist[..last].to_vec(),
            agq_evals: c.agq_evals,
            agq_node_evals: c.agq_node_evals,
            nb_nodes: c.nb_nodes,
            nb_evals_total: c.nb_evals_total,
        })
    }
    #[cfg(not(feature = "counters"))]
    fn counters(_fit: &glmm::Fit) -> Option<Counters> {
        None
    }
}

// ---------------------------------------------------------------------------
// JSON output (hand-written: no serde, so the file builds against any tag)
// ---------------------------------------------------------------------------

struct Rec(String);

impl Rec {
    fn new(kind: &str) -> Self {
        let mut r = Rec(String::from("{"));
        r.s("type", kind);
        r
    }
    fn key(&mut self, k: &str) {
        if self.0.len() > 1 {
            self.0.push(',');
        }
        let _ = write!(self.0, "\"{k}\":");
    }
    fn s(&mut self, k: &str, v: &str) -> &mut Self {
        self.key(k);
        self.0.push('"');
        for c in v.chars() {
            match c {
                '"' => self.0.push_str("\\\""),
                '\\' => self.0.push_str("\\\\"),
                '\n' => self.0.push_str("\\n"),
                c if (c as u32) < 0x20 => {
                    let _ = write!(self.0, "\\u{:04x}", c as u32);
                }
                c => self.0.push(c),
            }
        }
        self.0.push('"');
        self
    }
    fn f(&mut self, k: &str, v: f64) -> &mut Self {
        self.key(k);
        push_f64(&mut self.0, v);
        self
    }
    fn u(&mut self, k: &str, v: u64) -> &mut Self {
        self.key(k);
        let _ = write!(self.0, "{v}");
        self
    }
    fn opt_u(&mut self, k: &str, v: Option<u64>) -> &mut Self {
        match v {
            Some(v) => self.u(k, v),
            None => self.null(k),
        }
    }
    fn b(&mut self, k: &str, v: bool) -> &mut Self {
        self.key(k);
        self.0.push_str(if v { "true" } else { "false" });
        self
    }
    fn null(&mut self, k: &str) -> &mut Self {
        self.key(k);
        self.0.push_str("null");
        self
    }
    fn fs(&mut self, k: &str, v: &[f64]) -> &mut Self {
        self.key(k);
        self.0.push('[');
        for (i, &x) in v.iter().enumerate() {
            if i > 0 {
                self.0.push(',');
            }
            push_f64(&mut self.0, x);
        }
        self.0.push(']');
        self
    }
    fn us<T: std::fmt::Display>(&mut self, k: &str, v: &[T]) -> &mut Self {
        self.key(k);
        self.0.push('[');
        for (i, x) in v.iter().enumerate() {
            if i > 0 {
                self.0.push(',');
            }
            let _ = write!(self.0, "{x}");
        }
        self.0.push(']');
        self
    }
    fn raw(&mut self, k: &str, json: &str) -> &mut Self {
        self.key(k);
        self.0.push_str(json);
        self
    }
    fn done(mut self) -> String {
        self.0.push('}');
        self.0
    }
}

/// Shortest round-trip form; non-finite values as `null`.
fn push_f64(s: &mut String, v: f64) {
    if v.is_finite() {
        let _ = write!(s, "{v:?}");
    } else {
        s.push_str("null");
    }
}

// ---------------------------------------------------------------------------
// Header: machine and build state
// ---------------------------------------------------------------------------

fn cmd(prog: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(prog).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Median of what the probe itself reads around an empty bracket, over 1001
/// brackets: subtract it from a per-fit value when reading the sub-10 µs
/// shapes (OLS, GLM small).
fn probe_overhead(full: bool) -> String {
    let mut p = Probe {
        full,
        ..Probe::default()
    };
    let (mut w, mut c, mut i, mut y) = (vec![], vec![], vec![], vec![]);
    for _ in 0..1001 {
        p.begin();
        p.end();
        w.push(p.wall_ns);
        c.extend(p.cpu_ns);
        i.extend(p.instr);
        y.extend(p.cycles);
    }
    // The median, or null for a counter this build or OS does not have.
    let med = |v: &mut Vec<u64>| -> String {
        if v.is_empty() {
            return "null".into();
        }
        v.sort_unstable();
        v[v.len() / 2].to_string()
    };
    format!(
        "{{\"wall_ns\":{},\"cpu_ns\":{},\"instr\":{},\"cycles\":{}}}",
        med(&mut w),
        med(&mut c),
        med(&mut i),
        med(&mut y)
    )
}

fn header(cli: &Cli) -> String {
    let mut r = Rec::new("header");
    r.s("harness", "loop_bench")
        .s("api", api::API)
        .s("arm", &cli.arm)
        .s("source", &cli.source)
        .s("validation_pkg_version", env!("CARGO_PKG_VERSION"))
        .s("rustc", &cmd("rustc", &["-V"]).unwrap_or_default())
        .us(
            "features",
            &api::features()
                .iter()
                .map(|f| format!("\"{f}\""))
                .collect::<Vec<_>>(),
        )
        .b("prestandardized", cli.prestd)
        .s("readout", if cli.full_readout { "full" } else { "lean" })
        .s("probe", if cli.probe_full { "full" } else { "wall" })
        .u("seed", cli.seed)
        .u("draws", cli.draws as u64)
        .u("warmup", cli.warmup as u64)
        .b("dry", cli.dry)
        .raw("probe_overhead", &probe_overhead(cli.probe_full))
        .s("argv", &std::env::args().collect::<Vec<_>>().join(" "))
        .s(
            "rayon_num_threads",
            &std::env::var("RAYON_NUM_THREADS").unwrap_or_default(),
        );
    // Machine state, macOS only; every probe tolerates failure.
    let batt = cmd("pmset", &["-g", "batt"]).unwrap_or_default();
    let power = batt
        .lines()
        .next()
        .and_then(|l| l.split('\'').nth(1))
        .unwrap_or("")
        .to_string();
    let lpm = cmd("pmset", &["-g"])
        .unwrap_or_default()
        .lines()
        .find(|l| l.contains("lowpowermode"))
        .and_then(|l| l.split_whitespace().last().map(str::to_string))
        .unwrap_or_default();
    r.s("power_source", &power)
        .s("pmset_batt", &batt)
        .s("low_power_mode", &lpm)
        .s(
            "loadavg",
            &cmd("sysctl", &["-n", "vm.loadavg"]).unwrap_or_default(),
        )
        .s(
            "cpu",
            &cmd("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default(),
        )
        .s("host", &cmd("hostname", &["-s"]).unwrap_or_default())
        .s(
            "date",
            &cmd("date", &["+%Y-%m-%dT%H:%M:%S%z"]).unwrap_or_default(),
        );
    r.done()
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const USAGE: &str = "\
loop_bench: build one workspace per shape, refit D simulated draws on it.

  --shape S[,S..]    shape names, or `all` (default all); --list prints them
  --size small|large|both            (default both)
  --variant V[,V..]  base|redraw|raw|raw-origin|boundary|all (default base);
                     a variant a shape lacks is skipped
  --start M[,M..]    cold|truth|carry|theta0|all (default cold,truth,carry);
                     theta0 = MCPower's start with no truth or carry: theta
                     [THETA0; n_theta] in caller units, beta the generating
                     beta (MCPower's effect sizes)
  --draws D          measured draws per block (default 20)
  --warmup W         warm-up draws per block, recorded with warmup:true,
                     own data stream (default 3)
  --seed S           base seed (default 46)
  --out PATH         JSONL output, appended; `-` for stdout (default -)
  --arm NAME         arm name for the header (default env LOOP_BENCH_ARM or `unnamed`)
  --source TEXT      tree / commit / tag of the arm (default env LOOP_BENCH_SOURCE)
  --prestandardized  C arm: standardize the data here, set_prestandardized(true);
                     β, θ and the deviance are reported back in raw units
  --readout full|lean  full maps every fit through into_fit after the timer
                     (deviance, singular, counters); lean reads FitView only
  --probe full|wall  full also reads thread CPU time and instructions/cycles
                     per fit; wall reads Instant only (default full)
  --dry              generate data and build workspaces, fit nothing
  --list             print shapes x variants with n, p, n_theta and exit
";

struct Cli {
    shapes: Vec<&'static str>,
    sizes: Vec<&'static str>,
    variants: Vec<&'static str>,
    starts: Vec<&'static str>,
    draws: usize,
    warmup: usize,
    seed: u64,
    out: String,
    arm: String,
    source: String,
    prestd: bool,
    full_readout: bool,
    probe_full: bool,
    dry: bool,
    list: bool,
}

fn pick_list(arg: &str, all: &[&'static str], what: &str) -> Vec<&'static str> {
    if arg == "all" {
        return all.to_vec();
    }
    arg.split(',')
        .map(|s| {
            *all.iter()
                .find(|a| **a == s)
                .unwrap_or_else(|| die(&format!("unknown {what} `{s}` (one of {all:?})")))
        })
        .collect()
}

fn die(msg: &str) -> ! {
    eprintln!("loop_bench: {msg}\n\n{USAGE}");
    std::process::exit(2)
}

fn parse_cli() -> Cli {
    let mut cli = Cli {
        shapes: SHAPES.to_vec(),
        sizes: vec!["small", "large"],
        variants: vec!["base"],
        starts: vec!["cold", "truth", "carry"],
        draws: 20,
        warmup: 3,
        seed: 46,
        out: "-".into(),
        arm: std::env::var("LOOP_BENCH_ARM").unwrap_or_else(|_| "unnamed".into()),
        source: std::env::var("LOOP_BENCH_SOURCE").unwrap_or_default(),
        prestd: false,
        full_readout: true,
        probe_full: true,
        dry: false,
        list: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || {
            args.next()
                .unwrap_or_else(|| die(&format!("{a} needs a value")))
        };
        match a.as_str() {
            "--shape" => cli.shapes = pick_list(&val(), SHAPES, "shape"),
            "--size" => {
                cli.sizes = match val().as_str() {
                    "both" => vec!["small", "large"],
                    "small" => vec!["small"],
                    "large" => vec!["large"],
                    s => die(&format!("unknown size `{s}`")),
                }
            }
            "--variant" => cli.variants = pick_list(&val(), VARIANTS, "variant"),
            "--start" => {
                cli.starts = pick_list(&val(), &["cold", "truth", "carry", "theta0"], "start")
            }
            "--draws" => cli.draws = val().parse().unwrap_or_else(|_| die("bad --draws")),
            "--warmup" => cli.warmup = val().parse().unwrap_or_else(|_| die("bad --warmup")),
            "--seed" => cli.seed = val().parse().unwrap_or_else(|_| die("bad --seed")),
            "--out" => cli.out = val(),
            "--arm" => cli.arm = val(),
            "--source" => cli.source = val(),
            "--prestandardized" => cli.prestd = true,
            "--readout" => {
                cli.full_readout = match val().as_str() {
                    "full" => true,
                    "lean" => false,
                    s => die(&format!("unknown readout `{s}`")),
                }
            }
            "--probe" => {
                cli.probe_full = match val().as_str() {
                    "full" => true,
                    "wall" => false,
                    s => die(&format!("unknown probe `{s}`")),
                }
            }
            "--dry" => cli.dry = true,
            "--list" => cli.list = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0)
            }
            _ => die(&format!("unknown argument `{a}`")),
        }
    }
    cli
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

/// The start a draw is fitted from, in the arm's own design units.
fn start_for(
    mode: &str,
    d: &Design,
    carry: &Option<Start>,
    std: Option<&Prestd>,
) -> (Option<Start>, &'static str) {
    let truth = || Start {
        beta: d.beta.clone(),
        theta: d.theta_true(),
    };
    let raw = match mode {
        "cold" => return (None, "cold"),
        "truth" => (truth(), "truth"),
        // MCPower's GLMM start when it has neither a truth nor a carried
        // start: θ = [THETA0; n] in caller units, β = the generating β (its
        // effect sizes).
        "theta0" => (
            Start {
                beta: d.beta.clone(),
                theta: vec![glmm::THETA0; d.n_theta()],
            },
            "theta0",
        ),
        _ => match carry {
            Some(c) => (c.clone(), "carry"),
            None => (truth(), "truth(carry-init)"),
        },
    };
    let (s, label) = raw;
    let s = match std {
        // Starts are kept in raw units and mapped onto this draw's
        // standardized design, so a carry means the same in both arms.
        Some(ps) => Start {
            beta: ps.beta_to_std(&s.beta),
            theta: ps.map_theta(d, &s.theta, true),
        },
        None => s,
    };
    (Some(s), label)
}

struct BlockCtx<'a> {
    d: &'a Design,
    start: &'static str,
    cli: &'a Cli,
}

/// One draw's values in the caller's raw units (mapped back on the C arm).
struct RawVals {
    beta: Vec<f64>,
    theta: Vec<f64>,
    varcorr: Vec<f64>,
    /// Added to the reported deviance (the C arm's REML shift; else 0).
    dev_shift: f64,
    guard: u32,
}

fn draw_record(
    ctx: &BlockCtx,
    (draw, warmup, start_used): (usize, bool, &str),
    probe: &Probe,
    out: &Out,
    v: &RawVals,
) -> String {
    let d = ctx.d;
    let (beta_raw, theta_raw, dev_shift, guard) = (&v.beta, &v.theta, v.dev_shift, v.guard);
    let mut r = Rec::new("draw");
    r.s("shape", d.shape)
        .s("size", d.size)
        .s("variant", d.variant)
        .s("start", ctx.start)
        .u("seed", ctx.cli.seed)
        .u("draw", draw as u64)
        .b("warmup", warmup)
        .s("start_used", start_used)
        .u("n", d.n as u64)
        .u("p", d.p as u64)
        .u("n_theta", d.n_theta() as u64)
        .u("wall_ns", probe.wall_ns)
        .opt_u("cpu_ns", probe.cpu_ns)
        .opt_u("instr", probe.instr)
        .opt_u("cycles", probe.cycles)
        .u("n_eval", out.n_eval as u64)
        .b("converged", out.converged)
        .u("boundary_hit", u64::from(out.boundary_hit))
        .u("pinned_mask", out.pinned_mask)
        .b("ill_conditioned", out.ill_conditioned)
        .u("pirls_exhausted", u64::from(out.pirls_exhausted))
        .b("final_pirls_exhausted", out.final_pirls_exhausted)
        .b("hessian_fallback", out.hessian_fallback)
        .b("exact_profile_fallback", out.exact_profile_fallback)
        // The warm-start guard's two Laplace evaluations on every GLMM fit
        // handed a θ start other than the blind θ₀ (a start at the blind θ₀
        // skips it); derived, not counted by the kernel. When the search takes
        // the guard's winning solve over as its first evaluation (a hand-over
        // hit), one of the two is also n_eval's first evaluation; otherwise
        // both are outside n_eval.
        .u("guard_evals_derived", u64::from(guard));
    if let Some(f) = &out.full {
        // `objective`: the value the answer check compares. The deviance
        // (REML on the LMM, Laplace/AGQ on the GLMM) in the raw design's
        // units, else −2·loglik on OLS/GLM, whose deviance is NaN.
        let dev = f.deviance + dev_shift;
        let objective = if dev.is_finite() {
            dev
        } else {
            -2.0 * f.loglik
        };
        r.f("deviance", dev)
            .f("deviance_reported", f.deviance)
            .f("loglik", f.loglik)
            .f("objective", objective)
            .b("singular", f.singular)
            .s("boundary", f.boundary)
            .f("dispersion", f.dispersion)
            .fs("varcorr", &v.varcorr);
        if f.n_eval != out.n_eval {
            r.u("n_eval_fit", f.n_eval as u64);
        }
        if let Some(c) = &f.counters {
            let iters: u64 = c
                .pirls_hist
                .iter()
                .enumerate()
                .map(|(i, &v)| i as u64 * u64::from(v))
                .sum();
            // The LMM routes record every eval as stage 2; their route is
            // the one the design picks (dense or sparse REML).
            let route = match (c.stage_evals[0] > 0, c.stage_evals[1] > 0) {
                _ if d.fam == Fam::Gaussian => d.route_expected(),
                (true, false) => "exact_profile",
                (false, true) => "joint",
                (true, true) => "two_stage",
                (false, false) => "none",
            };
            r.us("stage_evals", &c.stage_evals)
                .us("shrink_evals", &c.shrink_evals)
                .us("pirls_hist", &c.pirls_hist)
                .u("pirls_iters", iters)
                .u("agq_evals", u64::from(c.agq_evals))
                .u("agq_node_evals", c.agq_node_evals)
                .u("nb_nodes", u64::from(c.nb_nodes))
                .u("nb_evals_total", c.nb_evals_total)
                .s("route", route);
        }
    }
    r.fs("beta", beta_raw).fs("theta", theta_raw);
    r.done()
}

fn run_block(d: &Design, start: &'static str, cli: &Cli, sink: &mut dyn std::io::Write) {
    let n_theta = d.n_theta();
    let mut rec = Rec::new("block");
    rec.s("shape", d.shape)
        .s("size", d.size)
        .s("variant", d.variant)
        .s("start", start)
        .u("seed", cli.seed)
        .s("family", d.fam.name())
        .u("nagq", u64::from(d.nagq))
        .u("n", d.n as u64)
        .u("p", d.p as u64)
        .u("n_theta", n_theta as u64)
        .us("levels", &d.groups.iter().map(|g| g.levels).collect::<Vec<_>>())
        .us(
            "groupings",
            &d.groups
                .iter()
                .map(|g| {
                    let rel = match g.nested_per_parent {
                        Some(k) => format!("nested/{k}"),
                        None => "crossed".into(),
                    };
                    format!(
                        "{{\"name\":\"{}\",\"q\":{},\"levels\":{},\"relation\":\"{rel}\",\"slopes\":{:?}}}",
                        g.name,
                        g.q(),
                        g.levels,
                        g.slopes
                    )
                })
                .collect::<Vec<_>>(),
        )
        .s(
            "scheme",
            if d.scheme == Scheme::Raw { "raw" } else { "std" },
        )
        .b("redraw", d.redraw)
        .b("boundary_variant", d.boundary)
        .b("agq_vec", d.agq_vec())
        .s("route_expected", d.route_expected())
        .fs("beta_true", &d.beta)
        .fs("theta_true", &d.theta_true())
        .f("sigma", d.sigma)
        .b("prestandardized", cli.prestd);
    let t_build = Instant::now();
    let mut bench = api::build(d, cli.prestd);
    rec.u("build_ns", t_build.elapsed().as_nanos() as u64);
    let _ = writeln!(sink, "{}", rec.done());

    let mut x = d.x.clone();
    let mut y = vec![0.0; d.n];
    let mut x_fit = vec![0.0; d.n * d.p];
    let ctx = BlockCtx { d, start, cli };
    let theta_blind = d.theta_blind();
    let mut probe = Probe {
        full: cli.probe_full,
        ..Probe::default()
    };
    let mut wall_sum = 0u64;
    let mut cpu_sum = Some(0u64);
    let mut fits = 0u64;
    let mut ru0 = None;
    let mut t_block = Instant::now();
    for phase in 0..2 {
        let warm = phase == 0;
        let count = if warm { cli.warmup } else { cli.draws };
        // The carry chain restarts at the measured block, so measured draw 0
        // starts from the truth in every run regardless of the warm-up length.
        let mut carry: Option<Start> = None;
        if !warm {
            ru0 = sys::rusage();
            t_block = Instant::now();
        }
        for k in 0..count {
            let stream = if warm { "warmup" } else { "draw" };
            let mut rng = Rng::new(stream_seed(
                cli.seed,
                &[d.shape, d.size, stream_label(d.variant), stream],
                k as u64,
            ));
            draw_data(d, &mut x, &mut y, &mut rng);
            if cli.dry {
                continue;
            }
            x_fit.copy_from_slice(&x);
            let pstd = cli.prestd.then(|| {
                let ps = Prestd::compute(&x, d.n, d.p);
                ps.apply(&mut x_fit, d.n, d.p);
                ps
            });
            let (s, label) = start_for(start, d, &carry, pstd.as_ref());
            // The kernel skips the guard when the start θ is its blind θ₀ bit
            // for bit (mirrors `fit_glmm`'s skip; change together). This test
            // reads θ in the caller's units; the kernel reads it on the fitted
            // design, after random-slope rescaling and `THETA_TRUTH_FLOOR`. The
            // two agree for the starts this harness builds, not for a hand-made
            // start on a random-slope model.
            let guard = match &s {
                Some(s)
                    if !s.theta.is_empty()
                        && d.fam != Fam::Gaussian
                        && !d.groups.is_empty()
                        && !s
                            .theta
                            .iter()
                            .zip(&theta_blind)
                            .all(|(a, b)| a.to_bits() == b.to_bits()) =>
                {
                    2
                }
                _ => 0,
            };
            let out = api::fit(
                &mut bench,
                &x_fit,
                &y,
                s.as_ref(),
                cli.full_readout,
                &mut probe,
            );
            // Back to raw units (C arm) for the record and the carry.
            let varcorr = out
                .full
                .as_ref()
                .map_or_else(Vec::new, |f| f.varcorr.clone());
            let (beta_raw, theta_raw, varcorr, dev_shift) = match &pstd {
                Some(ps) => (
                    ps.beta_to_raw(&out.beta),
                    if out.theta.is_empty() {
                        vec![]
                    } else {
                        ps.map_theta(d, &out.theta, false)
                    },
                    ps.map_cov(d, &varcorr),
                    if d.fam == Fam::Gaussian && !d.groups.is_empty() {
                        ps.reml_shift()
                    } else {
                        0.0
                    },
                ),
                None => (out.beta.clone(), out.theta.clone(), varcorr, 0.0),
            };
            // MCPower's carry rule: only a converged θ̂ is carried; a view
            // that exposes no θ̂ (the NB route) keeps the last good carry.
            if out.converged && beta_raw.iter().all(|b| b.is_finite()) {
                let theta = if theta_raw.is_empty() {
                    carry
                        .as_ref()
                        .map_or_else(|| d.theta_true(), |c| c.theta.clone())
                } else {
                    theta_raw.clone()
                };
                carry = Some(Start {
                    beta: beta_raw.clone(),
                    theta,
                });
            }
            let vals = RawVals {
                beta: beta_raw,
                theta: theta_raw,
                varcorr,
                dev_shift,
                guard,
            };
            let line = draw_record(&ctx, (k, warm, label), &probe, &out, &vals);
            let _ = writeln!(sink, "{line}");
            if !warm {
                wall_sum += probe.wall_ns;
                cpu_sum = cpu_sum.zip(probe.cpu_ns).map(|(a, b)| a + b);
                fits += 1;
            }
        }
    }
    let block_wall = t_block.elapsed().as_nanos() as u64;
    let ru1 = sys::rusage();
    let mut end = Rec::new("block_end");
    end.s("shape", d.shape)
        .s("size", d.size)
        .s("variant", d.variant)
        .s("start", start)
        .u("fits", fits)
        .u("fit_wall_ns_sum", wall_sum)
        .opt_u("fit_cpu_ns_sum", cpu_sum)
        .u("block_wall_ns", block_wall);
    match (ru0, ru1) {
        (Some((u0, s0, _)), Some((u1, s1, rss))) => {
            end.u("block_cpu_user_us", u1 - u0)
                .u("block_cpu_sys_us", s1 - s0)
                .u("maxrss_bytes", rss);
        }
        _ => {
            end.null("block_cpu_user_us");
        }
    }
    let _ = writeln!(sink, "{}", end.done());
    let _ = sink.flush();
}

fn main() {
    let cli = parse_cli();
    if cli.list {
        println!("shape\tsize\tvariant\tfamily\tn\tp\tn_theta\tlevels\troute_expected");
        for &sh in SHAPES {
            for &sz in &["small", "large"] {
                for &v in VARIANTS {
                    if !variant_ok(sh, v) {
                        continue;
                    }
                    let d = make_design(sh, sz, v, cli.seed);
                    println!(
                        "{sh}\t{sz}\t{v}\t{}\t{}\t{}\t{}\t{:?}\t{}",
                        d.fam.name(),
                        d.n,
                        d.p,
                        d.n_theta(),
                        d.groups.iter().map(|g| g.levels).collect::<Vec<_>>(),
                        d.route_expected()
                    );
                }
            }
        }
        return;
    }
    let mut sink: Box<dyn std::io::Write> = if cli.out == "-" {
        Box::new(std::io::stdout().lock())
    } else {
        if let Some(dir) = std::path::Path::new(&cli.out).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        Box::new(std::io::BufWriter::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&cli.out)
                .unwrap_or_else(|e| die(&format!("cannot open {}: {e}", cli.out))),
        ))
    };
    let _ = writeln!(sink, "{}", header(&cli));
    for &shape in &cli.shapes {
        for &variant in &cli.variants {
            if !variant_ok(shape, variant) {
                eprintln!(
                    "loop_bench: skip {shape}/{variant} (variant not defined for this shape)"
                );
                continue;
            }
            for &size in &cli.sizes {
                let d = make_design(shape, size, variant, cli.seed);
                for &start in &cli.starts {
                    run_block(&d, start, &cli, &mut *sink);
                }
            }
        }
    }
    let _ = sink.flush();
}
