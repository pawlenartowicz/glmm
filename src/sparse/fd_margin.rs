//! FD-Hessian noise margin: what the PIRLS exit band costs the FD Hessian,
//! measured across the whole GLMM corpus on both solver paths.
//!
//! A finite-difference second derivative divides by δ², so whatever error PIRLS
//! leaves in the deviance at its exit reaches the Hessian amplified by 1/δ². The
//! candidate margin was
//!
//! ```text
//! M = (PIRLS residual deviance error AT γ̂) / (H·δ²)
//! ```
//!
//! with `H` the joint-Hessian diagonal entry for the coordinate and δ its FD
//! step; `H·δ²` is exactly the raw central-difference NUMERATOR
//! `f(+δ) − 2·f(0) + f(−δ)`, so no curvature has to be estimated separately.
//!
//! **Measured 2026-08-24, M does not predict step-invariance and is not a
//! criterion.** The second difference annihilates whatever part of the exit
//! error is constant or linear in γ, so only the error's CURVATURE across the
//! stencil reaches the Hessian, while `M` divides by an error LEVEL. The two
//! come apart in both directions on this corpus: on the sparse Gamma-log rung
//! `M = 6.3` — nominally past the whole budget — while the δ and δ/2 standard
//! errors agree to 2.0e-5 and show no noise growth at δ/4; on the dense probit
//! rung `M = 2.8e-11` while those same SEs agree only to 1.4e-4 and the gap
//! grows 4.7× per halving, the noise signature. Eleven orders of M separate two
//! rungs whose step-invariance is ordered the other way round, so no threshold
//! on M rescues it.
//!
//! What does track step-invariance is `M*`, the same ratio built from the error
//! that actually reaches the second difference — the shipped-band numerator
//! against the reference-band numerator — which this module measures alongside
//! `M` so the two can be compared. It is reported, not enforced: adopting it is
//! a separate decision.
//!
//! The reference deviance comes from PIRLS with its exit band shrunk until the
//! penalized deviance stops changing, never from "the same solve a few decades
//! tighter": an over-tight band makes PIRLS exit on an iteration count that
//! flickers from one γ to the next, turning one large step in the deviance into
//! hash, which would calibrate against a corrupted baseline. The reference is
//! therefore verified step-free before use — scanned over ±3δ on a grid well
//! below δ, its grid second differences must show no jump that is large next to
//! `H·δ²` — and a coordinate where that cannot be established is reported as
//! such rather than quietly given a number.
//!
//! Both paths are driven here. The measurement lives under `sparse` because the
//! sparse deviance evaluator is private to this module tree, while the dense
//! one is `pub(crate)` and reachable from anywhere.

use faer::linalg::solvers::Solve;
use faer::{Mat, MatRef};
use serde_json::Value;

use super::glmm::{
    sparse_glmm_deviance, SparseGlmmWorkspace, GAMMA_HAT_CAPTURE, SPARSE_FD_STEP_REL,
};
use crate::formula::{lower, Column, Table};
use crate::glmm::{
    build_z, fd_mixed_diff, fd_second_diff, glmm_laplace_deviance, pirls_tol_fd, GlmmWorkspace,
    StructuredSchur, FD_STEP_BASE,
};
use crate::lmm::LmmGroupings;
use crate::{
    BinomialLink, Family, FitOptions, GammaLink, GroupIds, ModelSpec, NegBinomialLink, PoissonLink,
    WaldSe,
};

const MANIFEST: &str = include_str!("../../validation/manifest.json");

/// Exit-band ladder for the REFERENCE deviance, tried in order until one returns
/// a finite value. `0.0` is the literal fixed point — iterate until the
/// penalized deviance stops changing at f64 resolution — but PIRLS's overshoot
/// test shares this band, so at 0.0 a rise of one ULP near the mode counts as an
/// overshoot and burns a step-halving; on a solve that opens AT the mode that can
/// exhaust the halving cap and return the honest NaN. Measured 2026-08-24: `0.0`
/// qualified on no rung of the corpus, 17 landed at 1e-15 or 1e-14, and the
/// sparse Gamma rung needed 1e-11. The ladder stops at 1e-10 because a reference
/// looser than two decades under the shipped band measures nothing. Which rung
/// was used is reported per cell, and the reference is separately checked for
/// step-freeness before any M is computed from it.
const REF_TOL_LADDER: [f64; 7] = [0.0, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10];

/// Half-width of the reference step-freeness scan, in units of δ, and the grid
/// divisor below δ. ±3δ covers the whole stencil with margin; δ/8 resolves a
/// step that the stencil would straddle.
const SCAN_HALF_WIDTH: i32 = 3;
const SCAN_GRID_DIV: i32 = 8;

// ── corpus ───────────────────────────────────────────────────────────────────

/// One GLMM rung, lowered and ready to fit.
struct Rung {
    name: String,
    rung: u64,
    family: Family,
    x: Vec<f64>,
    y: Vec<f64>,
    n: usize,
    p: usize,
    model: ModelSpec,
    ids: GroupIds,
    opts: FitOptions,
}

fn suite_dir() -> String {
    format!("{}/validation", env!("CARGO_MANIFEST_DIR"))
}

/// Unquoted-header, `,`-split CSV read — the validation corpus's own format.
/// Mirrors `validation/engines/common.rs`'s reader; the crate cannot depend on
/// the (publish = false) validation package, so the twenty lines are repeated
/// rather than shared.
fn read_csv(path: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read csv {path}: {e}"));
    let mut lines = raw.lines().filter(|l| !l.trim().is_empty());
    let unq = |s: &str| s.trim().trim_matches('"').to_string();
    let header: Vec<String> = lines.next().unwrap().split(',').map(unq).collect();
    let rows: Vec<Vec<String>> = lines
        .map(|l| l.split(',').map(&unq).collect::<Vec<String>>())
        .collect();
    (header, rows)
}

/// Manifest `factors` become `Column::Factor`; a column that fails to parse as
/// `f64` anywhere falls back to `Factor` (the corpus carries categorical helper
/// columns no formula references). Mirrors the validation harness's `build_table`.
fn build_table(header: &[String], rows: &[Vec<String>], factors: &[String]) -> Table {
    let n = rows.len();
    let columns = header
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let is_factor = factors.iter().any(|f| f == name)
                || rows.iter().any(|r| r[j].parse::<f64>().is_err());
            let col = if is_factor {
                let labels: Vec<String> = rows.iter().map(|r| r[j].clone()).collect();
                Column::factor_from_labels(&labels)
            } else {
                Column::Numeric(rows.iter().map(|r| r[j].parse().unwrap()).collect())
            };
            (name.replace('.', "_"), col)
        })
        .collect();
    Table { columns, n }
}

fn family_of(spec: &Value) -> Family {
    let link = spec["link"].as_str();
    match spec["family"].as_str().unwrap() {
        "binomial" => Family::Binomial {
            link: match link {
                None | Some("logit") => BinomialLink::Logit,
                Some("probit") => BinomialLink::Probit,
                Some(o) => panic!("binomial link {o}"),
            },
        },
        "poisson" => Family::Poisson {
            link: PoissonLink::Log,
        },
        "gamma" => Family::Gamma {
            link: match link {
                None | Some("log") => GammaLink::Log,
                Some("inverse") => GammaLink::Inverse,
                Some(o) => panic!("gamma link {o}"),
            },
        },
        "negbin" => Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        other => panic!("family {other}"),
    }
}

/// `jl_formula` when present (guaranteed `cbind`-free), else `r_formula` with an
/// aggregated-binomial response rewritten onto the synthesized `prop` column.
/// The two rewrites after that are the frontend's dialect: `&` is Julia's
/// interaction-grouping operator where this parser uses `:`, and the parser
/// treats the intercept as implicit so a literal leading `1` has no term.
fn formula_of(spec: &Value) -> String {
    let f = match spec["jl_formula"].as_str() {
        Some(jl) => jl
            .strip_prefix("@formula(")
            .and_then(|s| s.strip_suffix(')'))
            .unwrap()
            .to_string(),
        None => {
            let r = spec["r_formula"].as_str().unwrap();
            match r.split_once('~') {
                Some((resp, rhs)) if resp.trim_start().starts_with("cbind(") => {
                    format!("prop ~{rhs}")
                }
                _ => r.to_string(),
            }
        }
    };
    f.replace(" & ", ":").replacen(" ~ 1 + ", " ~ ", 1)
}

/// Every mixed non-Gaussian rung in `validation/manifest.json`, lowered.
fn glmm_corpus() -> Vec<Rung> {
    let manifest: Value = serde_json::from_str(MANIFEST).expect("manifest parses");
    let suite = suite_dir();
    let mut out = vec![];
    for spec in manifest["datasets"].as_array().unwrap() {
        let family_str = spec["family"].as_str().unwrap();
        let r_formula = spec["r_formula"].as_str().unwrap_or("");
        if family_str == "gaussian" || !r_formula.contains('|') {
            continue;
        }
        let name = spec["name"].as_str().unwrap().to_string();
        let source = if spec["source"].as_str() == Some("sim") {
            "simulated"
        } else {
            "empirical"
        };
        let data_name = spec["data"].as_str().unwrap_or(&name);
        let (header, rows) = read_csv(&format!("{suite}/data/{source}/{data_name}.csv"));
        let factors: Vec<String> = spec["factors"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect())
            .unwrap_or_default();
        let family = family_of(spec);
        let formula = formula_of(spec);

        // Aggregated binomial (`weights` = trial-count column): synthesize the
        // `prop` response the jl_formula names, and pass the sizes as prior
        // weights — the lowering the validation harness performs.
        let mut table = build_table(&header, &rows, &factors);
        let agg_sizes = spec["weights"].as_str().map(|w_name| {
            let idx = |c: &str| header.iter().position(|h| h == c).unwrap();
            let (w_idx, inc_idx) = (idx(w_name), idx("incidence"));
            let sizes: Vec<f64> = rows.iter().map(|r| r[w_idx].parse().unwrap()).collect();
            let prop: Vec<f64> = rows
                .iter()
                .zip(&sizes)
                .map(|(r, s)| r[inc_idx].parse::<f64>().unwrap() / s)
                .collect();
            table.columns.push(("prop".into(), Column::Numeric(prop)));
            sizes
        });

        let mut lo = lower(&formula, &table, family).unwrap_or_else(|e| panic!("{name}: {e}"));
        if let Some(sizes) = agg_sizes {
            lo.opts.weights = Some(sizes);
        }
        if let Some(wc) = spec["weights_col"].as_str() {
            let w_idx = header.iter().position(|h| h == wc).unwrap();
            lo.opts.weights = Some(rows.iter().map(|r| r[w_idx].parse().unwrap()).collect());
        }
        if let Some(oc) = spec["offset"].as_str() {
            let o_idx = header.iter().position(|h| h == oc).unwrap();
            lo.opts.offset = Some(rows.iter().map(|r| r[o_idx].parse().unwrap()).collect());
        }
        lo.opts.wald_se = WaldSe::Hessian;
        lo.opts.nagq = 1;
        out.push(Rung {
            name,
            rung: spec["rung"].as_u64().unwrap(),
            family,
            x: lo.x,
            y: lo.y,
            n: lo.n,
            p: lo.p,
            model: lo.model,
            ids: lo.ids,
            opts: lo.opts,
        });
    }
    out
}

// ── measurement ──────────────────────────────────────────────────────────────

/// What one rung's FD pass measures. Per-coordinate vectors are indexed by the
/// joint γ = [θ | β] coordinate; `n_theta` splits them.
struct Measured {
    n_theta: usize,
    /// Shipped FD step per coordinate.
    steps: Vec<f64>,
    /// Exit band the reference deviance was taken at (a `REF_TOL_LADDER` rung).
    /// `None` when no rung of the ladder gave a finite deviance across the whole
    /// scan range — that cell has NO usable baseline and gets no M.
    ref_tol: Option<f64>,
    /// |dev(γ̂, shipped FD band) − dev(γ̂, reference band)| — the PIRLS residual
    /// deviance error the stencil's numerator carries. `None` with `ref_tol`.
    eps: Option<f64>,
    /// Largest grid second-difference jump the reference showed in the ±3δ scan,
    /// per coordinate, as a fraction of that coordinate's `H·δ²`. A reference
    /// that is not step-free reads ≳ 1 here.
    ref_jump_rel: Vec<f64>,
    /// `H·δ²` at δ, δ/2 and δ/4 — the raw stencil numerators.
    num: [Vec<f64>; 3],
    /// The δ-step numerators recomputed with every eval at the REFERENCE band.
    /// `num[FULL] − num_ref` is the error that actually reaches the second
    /// difference, as opposed to `eps`, which is the error's LEVEL at γ̂ and is
    /// annihilated by the stencil to the extent it is constant or linear in γ.
    /// `None` with `ref_tol`.
    num_ref: Option<Vec<f64>>,
    /// β standard errors rebuilt from the joint Hessian at δ, δ/2 and δ/4.
    /// `None` at a step whose Hessian came out non-PD.
    se: [Option<Vec<f64>>; 3],
    /// `se` the shipped fit reported, for the harness's own self-check.
    se_shipped: Vec<f64>,
}

/// Step index into [`Measured::num`] / [`Measured::se`].
const FULL: usize = 0;
const HALF: usize = 1;
const QUARTER: usize = 2;

impl Measured {
    /// M per coordinate at step index `s`. Empty when there is no baseline.
    fn m_at(&self, s: usize) -> Vec<f64> {
        match self.eps {
            Some(e) => self.num[s].iter().map(|&v| e / v.abs()).collect(),
            None => vec![],
        }
    }
    /// Worst (largest) M over all coordinates at step index `s`; NaN with no
    /// baseline.
    fn worst_m(&self, s: usize) -> f64 {
        self.m_at(s).into_iter().fold(f64::NAN, f64::max)
    }
    fn worst_ref_jump(&self) -> f64 {
        self.ref_jump_rel.iter().copied().fold(0.0, f64::max)
    }
    /// Coordinates whose reference scan hit a non-finite deviance somewhere
    /// strictly inside `±3δ` (the two ends are finite by `pick_ref_tol`), so
    /// step-freeness could not be established there.
    fn unscannable_coords(&self) -> Vec<String> {
        (0..self.ref_jump_rel.len())
            .filter(|&k| !self.ref_jump_rel[k].is_finite())
            .map(|k| self.coord_label(k))
            .collect()
    }
    /// The margin M was meant to stand in for, measured directly: the relative
    /// error the shipped exit band puts INTO the second difference, per
    /// coordinate. NaN with no baseline.
    fn worst_m_star(&self) -> f64 {
        match &self.num_ref {
            Some(nr) => self.num[FULL]
                .iter()
                .zip(nr)
                .map(|(&a, &b)| (a - b).abs() / b.abs().max(f64::MIN_POSITIVE))
                .fold(f64::NAN, f64::max),
            None => f64::NAN,
        }
    }
    /// Worst relative gap between the standard errors at two step indices.
    /// `None` when either Hessian was non-PD.
    fn se_gap(&self, a: usize, b: usize) -> Option<f64> {
        let (u, v) = (self.se[a].as_ref()?, self.se[b].as_ref()?);
        Some(
            u.iter()
                .zip(v)
                .map(|(&x, &y)| (x - y).abs() / x.abs().max(f64::MIN_POSITIVE))
                .fold(0.0, f64::max),
        )
    }
    /// Richardson limit of the β standard errors from a coarse/fine step PAIR,
    /// assuming the residual is `O(δ²)` — the order of a central second
    /// difference, which is what every stencil here uses, carried through the
    /// inversion and the square root as a smooth function of the Hessian. With
    /// `SE(δ) = S + Cδ²` and `SE(δ/2) = S + Cδ²/4`, the limit is
    /// `S = (4·SE(δ/2) − SE(δ))/3`. `None` when either Hessian was non-PD.
    ///
    /// The formula is arithmetic, not evidence: it returns a number on a noisy
    /// sequence too. `gap_ratio` and `h0_limits_disagree` are what say whether
    /// the sequence is in the regime where the number means anything.
    fn h0_limit(&self, coarse: usize, fine: usize) -> Option<Vec<f64>> {
        let (c, f) = (self.se[coarse].as_ref()?, self.se[fine].as_ref()?);
        Some(
            c.iter()
                .zip(f)
                .map(|(&a, &b)| (4.0 * b - a) / 3.0)
                .collect(),
        )
    }
    /// Worst relative distance from the standard errors at step `s` to the
    /// Richardson limit built from the `(coarse, fine)` pair.
    fn h0_gap(&self, s: usize, coarse: usize, fine: usize) -> Option<f64> {
        let (u, lim) = (self.se[s].as_ref()?, self.h0_limit(coarse, fine)?);
        Some(
            u.iter()
                .zip(&lim)
                .map(|(&x, &l)| (x - l).abs() / l.abs().max(f64::MIN_POSITIVE))
                .fold(0.0, f64::max),
        )
    }
    /// Worst relative disagreement between the two Richardson limits this
    /// three-step sequence supports — (δ, δ/2) against (δ/2, δ/4). In the
    /// convergent regime both estimate the same `S` and this is small; where
    /// noise dominates below δ the two extrapolate different things and this
    /// blows up, which is the signal that no h→0 limit has been established.
    fn h0_limits_disagree(&self) -> Option<f64> {
        let (a, b) = (self.h0_limit(FULL, HALF)?, self.h0_limit(HALF, QUARTER)?);
        Some(
            a.iter()
                .zip(&b)
                .map(|(&x, &y)| (x - y).abs() / y.abs().max(f64::MIN_POSITIVE))
                .fold(0.0, f64::max),
        )
    }
    /// How the δ↔δ/2 gap evolves into the δ/2↔δ/4 gap. Truncation error is
    /// `O(δ²)`, so a truncation-dominated gap shrinks by ~4× per halving
    /// (ratio ≈ 0.25); FD noise is `O(ε/δ²)`, so a noise-dominated gap GROWS by
    /// ~4× (ratio ≈ 4). This is what separates the two sides of the window
    /// without assuming which one is binding.
    fn gap_ratio(&self) -> Option<f64> {
        let g12 = self.se_gap(FULL, HALF)?;
        let g24 = self.se_gap(HALF, QUARTER)?;
        Some(g24 / g12.max(f64::MIN_POSITIVE))
    }
    fn coord_label(&self, k: usize) -> String {
        if k < self.n_theta {
            format!("theta[{k}]")
        } else {
            format!("beta[{}]", k - self.n_theta)
        }
    }
    /// Coordinate carrying `worst_m(FULL)`; 0 with no baseline.
    fn worst_coord(&self) -> usize {
        let m = self.m_at(FULL);
        (0..m.len()).fold(0, |best, k| if m[k] > m[best] { k } else { best })
    }
}

/// Build the joint FD Hessian with `eval` at `steps`, returning the β standard
/// errors `sqrt(2·(H⁻¹)_ββ diagonal)` (None on a non-PD Hessian) and the raw
/// stencil numerators `H_kk·δ_k²`. Same stencil helpers, factor of 2 and
/// inversion the shipped FD-Hessian arms use.
fn fd_hessian_probe(
    m: usize,
    n_theta: usize,
    p: usize,
    steps: &[f64],
    f0: f64,
    eval: &mut impl FnMut(&[usize], &[f64]) -> f64,
) -> (Option<Vec<f64>>, Vec<f64>) {
    let mut h = Mat::<f64>::zeros(m, m);
    for i in 0..m {
        h[(i, i)] = fd_second_diff(eval, i, steps[i], f0);
        for j in (i + 1)..m {
            let v = fd_mixed_diff(eval, i, j, steps[i], steps[j]);
            h[(i, j)] = v;
            h[(j, i)] = v;
        }
    }
    let numerators: Vec<f64> = (0..m).map(|i| h[(i, i)] * steps[i] * steps[i]).collect();
    let se = h.as_ref().llt(faer::Side::Lower).ok().map(|c| {
        let mut inv = Mat::<f64>::identity(m, m);
        c.solve_in_place(inv.as_mut());
        (0..p)
            .map(|j| (2.0 * inv[(n_theta + j, n_theta + j)]).max(0.0).sqrt())
            .collect()
    });
    (se, numerators)
}

/// Diagonal-only stencil numerators `H_kk·δ_k²` — the `fd_hessian_probe`
/// diagonal without the `O(m²)` mixed partials, for the reference-band rerun
/// that only the diagonal is compared on.
fn diag_numerators(
    m: usize,
    steps: &[f64],
    f0: f64,
    eval: &mut impl FnMut(&[usize], &[f64]) -> f64,
) -> Vec<f64> {
    (0..m)
        .map(|k| fd_second_diff(eval, k, steps[k], f0) * steps[k] * steps[k])
        .collect()
}

/// Largest jump in the reference's grid second differences over `±3δ` at `δ/8`,
/// per coordinate. A smooth function has near-constant grid second differences
/// (`≈ H·grid²`); a step in the deviance shows up as one grid triple whose second
/// difference departs from the rest. Returned relative to `H·δ²` — the quantity
/// the stencil actually divides by — so a value ≳ 1 means the reference itself
/// carries a step the stencil would straddle.
fn reference_step_scan(
    m: usize,
    steps: &[f64],
    numerators: &[f64],
    eval: &mut impl FnMut(usize, f64) -> f64,
) -> Vec<f64> {
    let mut out = vec![0.0; m];
    let pts = 2 * SCAN_HALF_WIDTH * SCAN_GRID_DIV + 1;
    for k in 0..m {
        let grid = steps[k] / SCAN_GRID_DIV as f64;
        let f: Vec<f64> = (0..pts)
            .map(|i| eval(k, (i - SCAN_HALF_WIDTH * SCAN_GRID_DIV) as f64 * grid))
            .collect();
        if f.iter().any(|v| !v.is_finite()) {
            out[k] = f64::INFINITY;
            continue;
        }
        let mut second: Vec<f64> = (1..f.len() - 1)
            .map(|i| f[i - 1] - 2.0 * f[i] + f[i + 1])
            .collect();
        second.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = second[second.len() / 2];
        let jump = second
            .iter()
            .map(|&s| (s - median).abs())
            .fold(0.0, f64::max);
        out[k] = jump / numerators[k].abs();
    }
    out
}

/// The tightest `REF_TOL_LADDER` rung that returns a finite deviance at γ̂ AND at
/// both ends of every coordinate's scan range. Requiring the whole range keeps a
/// baseline that dies on a perturbed evaluation from being adopted at γ̂ and then
/// reported as an infinite step-scan. `None` when no rung qualifies.
fn pick_ref_tol(
    m: usize,
    steps: &[f64],
    dev: &mut impl FnMut(f64, &[usize], &[f64]) -> f64,
) -> Option<f64> {
    REF_TOL_LADDER.into_iter().find(|&t| {
        dev(t, &[], &[]).is_finite()
            && (0..m).all(|k| {
                let d = SCAN_HALF_WIDTH as f64 * steps[k];
                dev(t, &[k], &[d]).is_finite() && dev(t, &[k], &[-d]).is_finite()
            })
    })
}

/// The path-independent half of the measurement: given a deviance evaluator
/// `dev(exit_band, coords, deltas)` that is anchored at γ̂ and reproduces the
/// shipped stencil's own evaluation contract, take the reference baseline, the
/// three step Hessians, and the reference step-freeness scan.
fn probe(
    m: usize,
    n_theta: usize,
    p: usize,
    steps: Vec<f64>,
    ship_tol: f64,
    se_shipped: Vec<f64>,
    dev: &mut impl FnMut(f64, &[usize], &[f64]) -> f64,
) -> Measured {
    let f0 = dev(ship_tol, &[], &[]);
    assert!(f0.is_finite(), "shipped-band deviance at γ̂ must be finite");
    let ref_tol = pick_ref_tol(m, &steps, dev);
    let eps = ref_tol.map(|t| (f0 - dev(t, &[], &[])).abs());

    let mut se = [None, None, None];
    let mut num = [vec![], vec![], vec![]];
    for (s, div) in [(FULL, 1.0), (HALF, 2.0), (QUARTER, 4.0)] {
        let scaled: Vec<f64> = steps.iter().map(|&v| v / div).collect();
        let mut eval = |coords: &[usize], deltas: &[f64]| dev(ship_tol, coords, deltas);
        let (se_s, num_s) = fd_hessian_probe(m, n_theta, p, &scaled, f0, &mut eval);
        se[s] = se_s;
        num[s] = num_s;
    }

    let num_ref = ref_tol.map(|t| {
        let f0r = dev(t, &[], &[]);
        let mut eval = |coords: &[usize], deltas: &[f64]| dev(t, coords, deltas);
        diag_numerators(m, &steps, f0r, &mut eval)
    });

    let ref_jump_rel = match ref_tol {
        Some(t) => {
            let mut scan = |k: usize, d: f64| dev(t, &[k], &[d]);
            reference_step_scan(m, &steps, &num[FULL], &mut scan)
        }
        None => vec![f64::INFINITY; m],
    };

    Measured {
        n_theta,
        steps,
        ref_tol,
        eps,
        ref_jump_rel,
        num,
        num_ref,
        se,
        se_shipped,
    }
}

/// Dense (`Solver::NoZ`) arm: fit through the dense GLMM kernel, then difference
/// its own deviance evaluator at exactly the point and seed the shipped FD pass
/// differences. Mirrors `fit_glmm_build`'s workspace construction — the FD grid
/// is only the shipped one if the design, scales and Z are built the same way.
fn measure_dense(r: &Rung, sized: &ModelSpec, ids: &GroupIds) -> Measured {
    let (n, p) = (r.n, r.p);
    let re = sized.re.as_ref().expect("mixed rung");
    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();
    let mut ws = GlmmWorkspace::for_cluster_spec(p, sized, n, &slope_cols, 1);
    if let Some(w) = &r.opts.weights {
        ws.prior_w[..n].copy_from_slice(w);
        ws.weighted = true;
    }
    ws.offset = r.opts.offset.clone();
    let x_mat = Mat::<f64>::from_fn(n, p, |i, j| r.x[i * p + j]);
    ws.groupings
        .set_slope_scales(x_mat.as_ref(), r.opts.weights.as_deref());
    build_z(&mut ws, x_mat.as_ref(), &ids.primary, &ids.extra, n);
    ws.structured_schur = if ws.groupings.structured_extras_eligible() {
        StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n)
    } else {
        None
    };
    let beta_start = crate::fit::glm_warm_start_beta(
        sized.family,
        f64::NAN,
        x_mat.as_ref(),
        &r.y,
        n,
        p,
        r.opts.offset.as_deref(),
    );
    let targets: Vec<u32> = (0..p as u32).collect();
    // Force the FD arm: the probe MEASURES the FD stencil's margin, and since
    // W3 the blocked path ships the exact hyper-dual Hessian instead — without
    // the flag the `se_shipped` self-check below would compare the probe's FD
    // numbers against exact ones and fail at the size of the stencil's own
    // error, which is the quantity being measured.
    ws.force_fd_hessian = true;
    let fit = crate::glmm::fit_glmm(
        &mut ws,
        x_mat.as_ref(),
        &r.y,
        &ids.primary,
        &ids.extra,
        &targets,
        None,
        &beta_start,
        n,
        WaldSe::Hessian,
    );
    assert!(fit.converged, "{}: dense fit must converge", r.name);
    let se_shipped: Vec<f64> = (0..p).map(|j| ws.var_diag[j].sqrt()).collect();

    // Freeze the FD grid on the fit's own converged mode, as `joint_hessian_cov`
    // does: every eval below warm-starts from this one seed, so each f(γ) is a
    // function of γ alone.
    let m = ws.params.len();
    let n_theta = ws.n_theta;
    let gamma: Vec<f64> = ws.params[..m].to_vec();
    let kk = ws.k.max(1);
    let seed: Vec<f64> = ws.u[..kk].to_vec();
    ws.u_seed[..kk].copy_from_slice(&seed);
    ws.warm_seed_active = true;
    let steps: Vec<f64> = (0..m)
        .map(|k| {
            if k < n_theta {
                FD_STEP_BASE
            } else {
                FD_STEP_BASE * gamma[k].abs().max(1.0)
            }
        })
        .collect();

    let (xr, y, cid, eid) = (x_mat.as_ref(), &r.y, &ids.primary, &ids.extra);
    let mut scratch = vec![0.0f64; m];
    let mut dev = |tol: f64, coords: &[usize], deltas: &[f64]| {
        scratch.copy_from_slice(&gamma);
        for (&c, &d) in coords.iter().zip(deltas) {
            scratch[c] += d;
        }
        ws.pirls_tol_override = Some(tol);
        glmm_laplace_deviance(&scratch, &mut ws, xr, y, cid, eid, n)
    };
    let out = probe(
        m,
        n_theta,
        p,
        steps,
        pirls_tol_fd(r.family),
        se_shipped,
        &mut dev,
    );
    ws.pirls_tol_override = None;
    ws.warm_seed_active = false;
    out
}

/// Sparse (`Solver::Sparse`) arm. Every sparse deviance eval cold-seeds û = 0,
/// so it is a pure function of γ and no seed discipline is needed; γ̂ comes from
/// the shipped fit through `GAMMA_HAT_CAPTURE` (see its comment for why it
/// cannot be read off the returned `Fit`).
fn measure_sparse(r: &Rung, sized: &ModelSpec, ids: &GroupIds) -> Measured {
    let (n, p) = (r.n, r.p);
    let re = sized.re.as_ref().expect("mixed rung");
    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();
    let extra_slope_cols: Vec<Vec<usize>> = re
        .extra_groupings
        .iter()
        .map(|g| g.slopes.iter().map(|&c| c as usize).collect())
        .collect();

    GAMMA_HAT_CAPTURE.with(|s| *s.borrow_mut() = Some(vec![]));
    let fit = crate::fit_cold(&r.x, &r.y, n, p, &r.model, &r.ids, &r.opts);
    let gamma = GAMMA_HAT_CAPTURE
        .with(|s| s.borrow_mut().take())
        .expect("capture armed");
    assert!(fit.converged(), "{}: sparse fit must converge", r.name);
    let se_shipped = fit.se.clone();

    let xm = MatRef::from_row_major_slice(&r.x, n, p);
    let mut g = LmmGroupings::from_cluster_spec_ext(sized, n, &slope_cols, &extra_slope_cols);
    g.set_slope_scales(xm, r.opts.weights.as_deref());
    let n_theta = g.n_theta();
    let m = n_theta + p;
    assert_eq!(gamma.len(), m, "{}: captured γ̂ width", r.name);
    let mut ws = SparseGlmmWorkspace::new(&g, &ids.primary, &ids.extra, n, p);
    if let Some(w) = &r.opts.weights {
        ws.prior_w[..n].copy_from_slice(w);
    }
    ws.offset = r.opts.offset.clone();

    let steps: Vec<f64> = gamma
        .iter()
        .map(|&v| SPARSE_FD_STEP_REL * v.abs().max(1.0))
        .collect();
    let (y, family) = (&r.y, r.family);
    let mut scratch = vec![0.0f64; m];
    let mut dev = |tol: f64, coords: &[usize], deltas: &[f64]| {
        scratch.copy_from_slice(&gamma);
        for (&c, &d) in coords.iter().zip(deltas) {
            scratch[c] += d;
        }
        ws.pirls_tol_override = Some(tol);
        sparse_glmm_deviance(family, f64::NAN, &scratch, &mut ws, xm, y, n, false)
    };
    let out = probe(
        m,
        n_theta,
        p,
        steps,
        pirls_tol_fd(family),
        se_shipped,
        &mut dev,
    );
    ws.pirls_tol_override = None;
    out
}

/// Which of the four report cells a rung falls in.
fn cell_of(sized: &ModelSpec, family: Family) -> (&'static str, &'static str) {
    let path = match crate::fit::classify_design(sized, 1) {
        crate::fit::Solver::Sparse => "sparse",
        crate::fit::Solver::NoZ => "dense",
    };
    let link = if crate::family::is_canonical(family) {
        "canonical"
    } else {
        "non-canonical"
    };
    (path, link)
}

/// Measure one rung end to end, self-checking that the harness's own δ-step
/// Hessian reproduces the SEs the shipped FD pass reported — if it does not, the
/// probe is not differencing the shipped stencil and nothing below it means
/// anything.
fn measure_rung(r: &Rung) -> (&'static str, &'static str, Measured) {
    let (sized, sids, _perm) = crate::fit::spec_sized_from_ids_pub(&r.model, &r.ids);
    let (path, link) = cell_of(&sized, r.family);
    let meas = if path == "sparse" {
        measure_sparse(r, &sized, &sids)
    } else {
        measure_dense(r, &sized, &sids)
    };
    if let Some(se) = &meas.se[FULL] {
        for (j, (&a, &b)) in se.iter().zip(&meas.se_shipped).enumerate() {
            let rel = (a - b).abs() / b.abs().max(f64::MIN_POSITIVE);
            assert!(
                rel < 1e-10,
                "{}: harness se[{j}] {a} vs shipped {b} (rel {rel:.2e}) — the probe is not \
                 differencing the shipped stencil",
                r.name
            );
        }
    }
    (path, link, meas)
}

// ── the measurement run ──────────────────────────────────────────────────────

/// Per-rung M measurement plus the δ vs δ/2 step-invariance check, over every
/// mixed non-Gaussian rung in the validation corpus. `#[ignore]`: it refits the
/// whole GLMM corpus and rebuilds each fit's joint Hessian three times, which is
/// far outside the default suite's cost. Run it explicitly:
///
/// ```text
/// cargo test fd_hessian_margin_corpus -- --ignored --nocapture
/// ```
///
/// Prints one block per rung and the worst cell per dense/sparse ×
/// canonical/non-canonical — the worst, never a median, because the pooled
/// number hides the corner the constants have to cover.
#[test]
#[ignore]
fn fd_hessian_margin_corpus_measurement() {
    // Serialized under alloc-tests so its allocations can't land in a concurrent
    // dhat profiler window on an `-- --ignored` run (the contract every
    // `#[ignore]` lib test shares — see `test_support::alloc_test_guard`).
    #[cfg(feature = "alloc-tests")]
    let _serial = crate::test_support::alloc_test_guard();
    let corpus = glmm_corpus();
    // (path, link, worst M, h→0 gap at the shipped δ, limit disagreement, label)
    let mut rows: Vec<(String, String, f64, f64, f64, String)> = vec![];
    for r in &corpus {
        let (path, link, meas) = measure_rung(r);
        let fmt = |o: Option<f64>| {
            o.map(|v| format!("{v:.3e}"))
                .unwrap_or_else(|| "non-PD".into())
        };
        println!(
            "{:<28} #{:<3} {:>7} {:>14}  ref_tol {:<8}  eps {:>10}  M(δ) {:>10}  M(δ/2) {:>10}",
            r.name,
            r.rung,
            path,
            link,
            meas.ref_tol
                .map(|t| format!("{t:.0e}"))
                .unwrap_or_else(|| "none".into()),
            fmt(meas.eps),
            format!("{:.3e}", meas.worst_m(FULL)),
            format!("{:.3e}", meas.worst_m(HALF)),
        );
        println!(
            "    worst coord {} (δ={:.3e}, H·δ²={:.6e}); M* {:.3e}; ref step-scan worst \
             {:.2e}×H·δ²; se gap δ→δ/2 {}, δ/2→δ/4 {}, ratio {}",
            meas.coord_label(meas.worst_coord()),
            meas.steps[meas.worst_coord()],
            meas.num[FULL][meas.worst_coord()],
            meas.worst_m_star(),
            meas.worst_ref_jump(),
            fmt(meas.se_gap(FULL, HALF)),
            fmt(meas.se_gap(HALF, QUARTER)),
            fmt(meas.gap_ratio()),
        );
        println!(
            "    h→0 (Richardson, O(δ²)): |SE(δ)−limit|/limit = {} from (δ,δ/2), {} from \
             (δ/2,δ/4); the two limits disagree by {}",
            fmt(meas.h0_gap(FULL, FULL, HALF)),
            fmt(meas.h0_gap(FULL, HALF, QUARTER)),
            fmt(meas.h0_limits_disagree()),
        );
        let unscannable = meas.unscannable_coords();
        if !unscannable.is_empty() {
            println!(
                "    reference not scannable (non-finite inside ±3δ) at: {}",
                unscannable.join(", ")
            );
        }
        rows.push((
            path.into(),
            link.into(),
            meas.worst_m(FULL),
            meas.h0_gap(FULL, HALF, QUARTER).unwrap_or(f64::NAN),
            meas.h0_limits_disagree().unwrap_or(f64::NAN),
            format!(
                "{} — M(δ/2)={:.3e}, M*={:.3e}, se gap δ→δ/2 {}, ratio {}, ref_jump {:.2e}",
                r.name,
                meas.worst_m(HALF),
                meas.worst_m_star(),
                fmt(meas.se_gap(FULL, HALF)),
                fmt(meas.gap_ratio()),
                meas.worst_ref_jump()
            ),
        ));
    }
    println!("\n== worst M per cell (dense/sparse × canonical/non-canonical) ==");
    for path in ["dense", "sparse"] {
        for link in ["canonical", "non-canonical"] {
            let hit = rows
                .iter()
                .filter(|w| w.0 == path && w.1 == link)
                .max_by(|a, b| a.2.total_cmp(&b.2));
            match hit {
                Some((_, _, m, _, _, extra)) => {
                    println!("{path:>7} × {link:<14} worst M = {m:.3e}  [{extra}]")
                }
                None => println!("{path:>7} × {link:<14} no rung in the corpus"),
            }
        }
    }
    println!("\n== worst h→0 gap per cell: |SE(δ) − Richardson limit(δ/2,δ/4)| / limit ==");
    for path in ["dense", "sparse"] {
        for link in ["canonical", "non-canonical"] {
            let hit = rows
                .iter()
                .filter(|w| w.0 == path && w.1 == link)
                .max_by(|a, b| a.3.total_cmp(&b.3));
            match hit {
                Some((_, _, _, g, d, extra)) => println!(
                    "{path:>7} × {link:<14} worst h→0 gap = {g:.3e} (limits disagree {d:.3e})  \
                     [{extra}]"
                ),
                None => println!("{path:>7} × {link:<14} no rung in the corpus"),
            }
        }
    }
}
