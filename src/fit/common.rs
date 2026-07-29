//! Shared helpers used by 2+ estimator dispatch files (`ols.rs`/`lmm.rs`/
//! `glm.rs`/`glmm.rs`): θ-width/varcorr bookkeeping, RE-count sizing from
//! `GroupIds`, rank-deficiency salvage, and the two `Fit::se` fill conventions
//! (target-compact vs predictor-indexed). None of this touches a kernel
//! directly — it is `fit`'s own marshalling logic, factored out of `mod.rs`
//! because every estimator's dispatch needs at least one of these.

use faer::Mat;

use crate::{
    Family, GroupIds, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing, StartValues,
};

use super::{Fit, FitOptions};

/// RE θ length for a mixed model, from topology ALONE (independent of level
/// counts): primary vech `q_p(q_p+1)/2` plus a `vech(Λ_g)` block `q_g(q_g+1)/2`
/// per extra grouping. Equals `LmmGroupings::n_theta()` for the same spec; used to
/// validate a `StartValues.theta` at the stable boundary before the workspace
/// (which is where `n_theta` otherwise first materializes) is built. `None`
/// (fixed-only) → 0.
pub(super) fn theta_width(re: Option<&ReStructure>) -> usize {
    let Some(re) = re else { return 0 };
    let q_p = 1 + re.slopes.len();
    let mut w = q_p * (q_p + 1) / 2;
    for g in &re.extra_groupings {
        let q_g = 1 + g.slopes.len();
        w += q_g * (q_g + 1) / 2;
    }
    w
}

/// vech(column-major lower-tri) of `D̂ = scale·Λ̂Λ̂'` for one `q×q` grouping.
/// `theta_block` is that grouping's column-major lower-tri `vech(Λ)` prefix
/// (as `primary_lambda` reads it). `D[r][c] = Σ_{k≤min(r,c)} Λ[r][k]·Λ[c][k]`.
pub(super) fn varcorr_block(theta_block: &[f64], q: usize, scale: f64) -> Vec<f64> {
    let mut lam = vec![0.0f64; q * q];
    crate::lmm::primary_lambda(theta_block, q, &mut lam); // Λ lower-tri, row-major
    let mut vech = Vec::with_capacity(q * (q + 1) / 2);
    for c in 0..q {
        for r in c..q {
            let mut d = 0.0;
            for k in 0..=c {
                d += lam[r * q + k] * lam[c * q + k];
            }
            vech.push(scale * d);
        }
    }
    vech
}

/// Assemble `Fit::varcorr`: one vech-packed `D̂ = scale·Λ̂Λ̂'` block
/// per grouping, declaration order (primary, then each extra). `scale` is σ̂²
/// (LMM) or 1.0 (GLMM link scale). Path-independent — a function of θ̂ only.
/// The primary-then-extras vech walk mirrors the `vech_start` layout assigned
/// in `LmmGroupings::from_cluster_spec_ext` (`src/lmm.rs`) — change together.
/// `pub(crate)` so the sparse-Z path (`sparse::fit_mle_sparse`) recovers varcorr
/// from θ̂ through the same path-independent assembly as the NoZ `fit_mle`.
pub(crate) fn assemble_varcorr(
    theta: &[f64],
    groupings: &crate::lmm::LmmGroupings,
    scale: f64,
) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(1 + groupings.extra_q.len());
    let mut cursor = 0usize;
    for &q in std::iter::once(&groupings.primary_q).chain(groupings.extra_q.iter()) {
        let vech = q * (q + 1) / 2;
        out.push(varcorr_block(&theta[cursor..cursor + vech], q, scale));
        cursor += vech;
    }
    out
}
/// [`Fit::loglik`] for the Gaussian LMM paths: the REML criterion
/// `−½(deviance + (n−p)(1 + ln 2π))` — the stripped constant restored (see
/// `Fit::deviance`'s LMM contract), then onto the `logLik()` scale. Call AFTER
/// any weighted `−Σlog wᵢ` deviance correction (the correction is part of
/// lme4's REMLcrit, so it belongs inside the criterion). NaN in ⇒ NaN out.
pub(crate) fn lmm_loglik(deviance: f64, n: usize, p: usize) -> f64 {
    // The NaN gate also protects the `n − p` residual df from underflowing on
    // the degenerate n ≤ p short-circuit (which always reports a NaN deviance).
    if !deviance.is_finite() || n <= p {
        return f64::NAN;
    }
    -0.5 * (deviance + (n - p) as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln()))
}

/// [`Fit::loglik`] for the GLMM paths (dense and sparse, Laplace and AGQ):
/// restore the data-only saturated constant the marginal deviance drops.
/// Binomial/Poisson/NB: `−½·deviance + saturated_loglik` (the deviance's data
/// term is the weighted `dev_resid` sum). Gamma: `−½·deviance` with NO
/// correction — lme4's `logLik(glmer fit)` is literally `−devfun/2`, and the
/// `+2` its `Gamma()$aic` data term carries stays inside (so glmer's Gamma
/// logLik sits 1 below `Σwᵢ·log f`; R's `logLik.glm` subtracts that 2 back
/// out, glmer does not — pinned against the sim_gamma validation loglik). NaN in
/// ⇒ NaN out (the non-converged contract).
pub(crate) fn glmm_loglik(
    family: Family,
    nb_theta: f64,
    deviance: f64,
    y: &[f64],
    prior_w: Option<&[f64]>,
) -> f64 {
    if !deviance.is_finite() {
        return f64::NAN;
    }
    match family {
        Family::Gamma { .. } => -0.5 * deviance,
        _ => -0.5 * deviance + crate::family::saturated_loglik(family, nb_theta, y, prior_w),
    }
}

/// [`Fit::df`] for a converged fit: retained fixed effects + RE θ parameters +
/// 1 if the family estimates a dispersion/scale. `dispersion_fixed` is
/// `FitOptions::dispersion.is_some()` (Gamma's hold-φ-fixed directive; every
/// other family ignores it).
pub(crate) fn model_df(
    family: Family,
    p_retained: usize,
    n_theta: usize,
    dispersion_fixed: bool,
) -> usize {
    let scale = match family {
        Family::Gaussian | Family::NegativeBinomial { .. } => 1,
        Family::Gamma { .. } => usize::from(!dispersion_fixed),
        Family::Binomial { .. } | Family::Poisson { .. } => 0,
    };
    p_retained + n_theta + scale
}

/// Extra grouping `e`'s θ vech offset — nested and crossed factors store it on
/// their own descriptors, keyed by declaration index.
fn extra_vech_start(g: &crate::lmm::LmmGroupings, e: usize) -> usize {
    if let Some(nf) = g.nested {
        if nf.decl == e {
            return nf.vech_start;
        }
    }
    g.crossed
        .iter()
        .find(|cf| cf.decl == e)
        .map(|cf| cf.vech_start)
        .expect("an extra grouping is either nested or crossed")
}

/// [`Fit::ranef_levels`]: level count per grouping, declaration order (primary,
/// then each extra). A nested grouping spans the padded
/// `n_primary·nested_per_parent` child grid (see `Fit::ranef`'s layout doc).
pub(crate) fn ranef_level_counts(g: &crate::lmm::LmmGroupings) -> Vec<usize> {
    let mut counts = Vec::with_capacity(1 + g.extra_q.len());
    counts.push(g.n_primary);
    for e in 0..g.extra_q.len() {
        let is_nested = g.nested.map(|nf| nf.decl) == Some(e);
        counts.push(if is_nested {
            g.n_primary * g.nested_per_parent
        } else {
            g.crossed
                .iter()
                .find(|cf| cf.decl == e)
                .map(|cf| cf.n_levels)
                .expect("an extra grouping is either nested or crossed")
        });
    }
    counts
}

/// [`Fit::ranef`] from the DENSE GLMM workspace's spherical modes `u`
/// (`glmm::build_z` layout: primary block level-major `lvl·q_p + c`, then each
/// extra's scalar indicator columns at its absolute `extra_offsets[e]` — this
/// path carries intercept-only extras exclusively, see `apply_lambda`).
/// `b = Λ̂û` per block: the primary level's `q_p`-vector through the lower-tri
/// `Λ_p`, each extra level through its scalar θ. Output is `Fit::ranef`'s
/// public layout (per grouping, level-major) — for this path the primary block
/// is already level-major, so it maps through 1:1.
pub(crate) fn assemble_ranef_dense(
    theta: &[f64],
    g: &crate::lmm::LmmGroupings,
    u: &[f64],
) -> Vec<f64> {
    let q = g.primary_q;
    let mut lam = vec![0.0f64; q * q];
    crate::lmm::primary_lambda(theta, q, &mut lam);
    let mut out = Vec::with_capacity(g.k_total);
    for lvl in 0..g.n_primary {
        let base = lvl * q;
        for r in 0..q {
            let mut b = 0.0;
            for c in 0..=r {
                b += lam[r * q + c] * u[base + c];
            }
            out.push(b);
        }
    }
    debug_assert!(!g.extra_slopes_any, "dense GLMM extras are intercept-only");
    let counts = ranef_level_counts(g);
    for (e, &off) in g.extra_offsets.iter().enumerate() {
        let theta_e = theta[extra_vech_start(g, e)];
        for l in 0..counts[e + 1] {
            out.push(theta_e * u[off + l]);
        }
    }
    out
}

/// [`Fit::ranef`] from the SPARSE workspace's spherical modes `u`
/// (`sparse::for_each_z_entry` layout: primary block component-major
/// `d·n_primary + f`, each extra level-major `extra_offsets[e] + l·q_g + c`
/// with a full `q_g×q_g` `Λ_g` block). Output is `Fit::ranef`'s public layout
/// (per grouping, level-major) — the primary block is transposed into it here.
pub(crate) fn assemble_ranef_sparse(
    theta: &[f64],
    g: &crate::lmm::LmmGroupings,
    u: &[f64],
) -> Vec<f64> {
    let q = g.primary_q;
    let s = g.n_primary;
    let mut lam = vec![0.0f64; q * q];
    crate::lmm::primary_lambda(theta, q, &mut lam);
    let mut out = Vec::with_capacity(g.k_total);
    for f in 0..s {
        for r in 0..q {
            let mut b = 0.0;
            for c in 0..=r {
                b += lam[r * q + c] * u[c * s + f];
            }
            out.push(b);
        }
    }
    let counts = ranef_level_counts(g);
    for (e, &off) in g.extra_offsets.iter().enumerate() {
        let q_g = g.extra_q[e];
        let mut lam_g = vec![0.0f64; q_g * q_g];
        crate::lmm::primary_lambda(&theta[extra_vech_start(g, e)..], q_g, &mut lam_g);
        for l in 0..counts[e + 1] {
            let base = off + l * q_g;
            for r in 0..q_g {
                let mut b = 0.0;
                for c in 0..=r {
                    b += lam_g[r * q_g + c] * u[base + c];
                }
                out.push(b);
            }
        }
    }
    out
}

/// A `ModelSpec` copy whose RE **counts** are derived from the supplied `GroupIds`
/// (`n_primary = max(primary)+1`; per crossed grouping `n_clusters =
/// max(extra[g])+1`; per nested grouping `n_per_parent = ⌈children/n_primary⌉`,
/// `children = max(extra[g])+1`). Topology tags (`Crossed`/`NestedWithin`),
/// family, and slope columns are preserved. This is the data path's mechanism for
/// "derive every level count from the ids": the sizing-corrected copy
/// feeds the existing workspace builders unchanged, so no builder/kernel signature
/// changes. `re: None` is returned as-is (fixed-only carries no counts).
pub(super) fn spec_sized_from_ids(model: &ModelSpec, ids: &GroupIds) -> ModelSpec {
    let Some(re) = model.re.as_ref() else {
        return model.clone();
    };
    let level_count = |v: &[u32]| v.iter().copied().max().map(|m| m as usize + 1).unwrap_or(1);
    let n_primary = level_count(&ids.primary);
    let extra_groupings: Vec<Grouping> = re
        .extra_groupings
        .iter()
        .enumerate()
        .map(|(g, gr)| {
            let relation = match gr.relation {
                GroupingRelation::Crossed { .. } => {
                    let children = level_count(&ids.extra[g]);
                    GroupingRelation::Crossed {
                        n_clusters: children as u32,
                    }
                }
                GroupingRelation::NestedWithin { .. } => {
                    // Per-parent distinct-child-id sets, counted directly from the
                    // (primary, extra) row pairs — layout-agnostic (works whether
                    // or not the frontend's ids are contiguous per-parent blocks),
                    // unlike deriving from a single global `max(extra)+1` divided
                    // by `n_primary`, which under-sizes any parent with an
                    // above-average child count (the true fix for unbalanced
                    // nesting: `n_per_parent` must be the TRUE max, not a
                    // `children.div_ceil(n_primary)` global average).
                    let mut per_parent: Vec<std::collections::HashSet<u32>> =
                        vec![Default::default(); n_primary];
                    for (&p, &c) in ids.primary.iter().zip(&ids.extra[g]) {
                        per_parent[p as usize].insert(c);
                    }
                    let n_per_parent = per_parent.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
                    GroupingRelation::NestedWithin {
                        n_per_parent: n_per_parent as u32,
                    }
                }
            };
            Grouping {
                relation,
                slopes: gr.slopes.clone(),
            }
        })
        .collect();
    ModelSpec {
        family: model.family,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_primary as u32,
            },
            slopes: re.slopes.clone(),
            extra_groupings,
        }),
    }
}
/// Lower-tri Gram `G = XᵀX` (from row-major X) → `aliased_columns` mask. `p` is
/// tiny, so the extra `O(N·p²)` reduction for the drop decision is negligible
/// (the reduced fit recomputes its own suff stats).
pub(super) fn detect_aliased(x: &[f64], n: usize, p: usize) -> Vec<bool> {
    let mut gram = Mat::<f64>::zeros(p, p);
    for i in 0..n {
        for a in 0..p {
            let xa = x[i * p + a];
            for b in 0..=a {
                gram[(a, b)] += xa * x[i * p + b];
            }
        }
    }
    crate::ols::aliased_columns(gram.as_ref(), p, crate::ols::ALIAS_EPS)
}

/// Remap a spec's RE slope x-column indices through the kept-columns map
/// (`to_reduced[orig]` = reduced index, or `usize::MAX` if that column was
/// dropped). An aliased column that is ALSO an RE slope is a rank-deficient
/// random slope — out of scope for #4; fault honestly rather than silently
/// mis-index.
fn remap_spec_slopes(model: &ModelSpec, to_reduced: &[usize]) -> ModelSpec {
    let Some(re) = model.re.as_ref() else {
        return model.clone();
    };
    let remap = |cols: &[u32]| -> Vec<u32> {
        cols.iter()
            .map(|&c| {
                let r = to_reduced[c as usize];
                assert!(
                    r != usize::MAX,
                    "rank-deficient random-slope column {c}: an aliased fixed column is used as an RE slope (unsupported)"
                );
                r as u32
            })
            .collect()
    };
    let extra_groupings = re
        .extra_groupings
        .iter()
        .map(|g| Grouping {
            relation: g.relation.clone(),
            slopes: remap(&g.slopes),
        })
        .collect();
    ModelSpec {
        family: model.family,
        re: Some(ReStructure {
            sizing: re.sizing.clone(),
            slopes: remap(&re.slopes),
            extra_groupings,
        }),
    }
}

/// Fit the reduced (aliased-columns-dropped) model and scatter β/se back to full
/// width: retained slots take the reduced fit, aliased slots are NaN,
/// `converged` follows the reduced fit, `tau2`/`varcorr`/`dispersion` pass through
/// (the RE structure is unchanged; only fixed-column indices are remapped). The
/// reduced design is full-rank, so the recursive `fit_warm` never re-enters here.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_rank_deficient(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    start: Option<&StartValues>,
    opts: &FitOptions,
    aliased: &[bool],
) -> Fit {
    let kept: Vec<usize> = (0..p).filter(|&j| !aliased[j]).collect();
    let pk = kept.len();
    let mut to_reduced = vec![usize::MAX; p];
    for (r, &orig) in kept.iter().enumerate() {
        to_reduced[orig] = r;
    }
    // Reduced design (drop aliased columns), row-major.
    let mut xr = vec![0.0f64; n * pk];
    for i in 0..n {
        for (r, &orig) in kept.iter().enumerate() {
            xr[i * pk + r] = x[i * p + orig];
        }
    }
    let model_r = remap_spec_slopes(model, &to_reduced);
    // StartValues.beta is p-wide → reduce it; theta is RE-only, unchanged.
    let start_r = start.map(|s| StartValues {
        beta: kept.iter().map(|&o| s.beta[o]).collect(),
        theta: s.theta.clone(),
    });
    // Targets: drop aliased targets, reindex survivors into the reduced design.
    let targets_r: Vec<u32> = opts
        .target_indices
        .iter()
        .filter(|&&t| !aliased[t as usize])
        .map(|&t| to_reduced[t as usize] as u32)
        .collect();
    // Per-row weights are unaffected by dropping fixed-effect COLUMNS, so the
    // rest of `opts` carries over unchanged via `..opts.clone()`.
    let opts_r = FitOptions {
        target_indices: targets_r,
        ..opts.clone()
    };
    let fr = super::fit_warm(&xr, y, n, pk, &model_r, ids, start_r.as_ref(), &opts_r);
    // Scatter reduced β/se to full width; aliased slots stay NaN.
    let mut beta = vec![f64::NAN; p];
    let mut se = vec![f64::NAN; p];
    for (r, &orig) in kept.iter().enumerate() {
        beta[orig] = fr.beta[r];
        se[orig] = fr.se[r];
    }
    // Same scatter in two dimensions: an aliased column has no coefficient, so
    // it has no covariance with anything — its whole row and column stay NaN,
    // exactly as its `se` slot does.
    let mut vcov = nan_vcov(p);
    for (ri, &oi) in kept.iter().enumerate() {
        for (rj, &oj) in kept.iter().enumerate() {
            vcov[oi][oj] = fr.vcov[ri][rj];
        }
    }
    Fit {
        beta,
        se,
        vcov,
        tau2: fr.tau2,
        dispersion: fr.dispersion,
        converged: fr.converged,
        varcorr: fr.varcorr,
        stddev_se: fr.stddev_se,
        aliased: aliased.to_vec(),
        n_eval: fr.n_eval,
        deviance: fr.deviance,
        singular: fr.singular,
        // Scalars/per-row/RE-shaped fields pass through: loglik and df come
        // from the REDUCED fit (aliased columns carry no parameter — lme4's
        // NA-coefficient df), fitted is per-row (the reduced model's means are
        // the model's means), ranef is RE-shaped (unchanged by fixed-column
        // drops).
        loglik: fr.loglik,
        df: fr.df,
        reml: fr.reml,
        fitted: fr.fitted,
        ranef: fr.ranef,
        ranef_levels: fr.ranef_levels,
    }
}
/// Engine-invariant shape check for the data-shaped ids (mirrors `fit_grouped`'s
/// former `cluster_ids.len()==n` panic): the primary id vector is length `n`, the
/// extra vectors align 1:1 with the declared extra groupings, and each is length
/// `n`. A malformed `GroupIds` is an engine invariant violation, so this panics
/// (consistent with the `fit` panic convention), not a `converged:false` return.
pub(super) fn assert_group_ids(re: &ReStructure, ids: &GroupIds, n: usize) {
    assert_eq!(
        ids.primary.len(),
        n,
        "GroupIds.primary must have n elements"
    );
    assert_eq!(
        ids.extra.len(),
        re.extra_groupings.len(),
        "GroupIds.extra must align 1:1 with re.extra_groupings (declaration order)"
    );
    for (g, e) in ids.extra.iter().enumerate() {
        assert_eq!(e.len(), n, "GroupIds.extra[{g}] must have n elements");
    }
}

/// Engine invariant checks for the standalone `fit` path: the kernel's stack
/// scratch is sized off `MAX_PRIMARY_Q`/`MAX_EXTRA_Q`, so a
/// `q` over the cap would overflow it, and every slope column must index into the
/// `p`-wide design. A malformed spec is an engine invariant violation (see the
/// `fit` panic convention), so this asserts rather than returning a `Fit`.
/// Fixed-only models (`re: None`) carry no RE caps to check.
pub(super) fn assert_model_shape(model: &ModelSpec, p: usize, nagq: u8) {
    // nAGQ: odd, 1..=MAX_NAGQ; >1 only on a single grouping factor (no extras),
    // q_p ≤ 3, binomial/Poisson GLMM — the shapes whose marginal likelihood is a
    // product of independent per-cluster q-D integrals. Checked before the RE
    // early-return so even fixed-only specs can't smuggle a bad nagq through.
    // Sourced from `FitOptions` (M3.5), not the spec. Mirrors the Python layer's
    // warn-and-strip boundary (`glmm.fit`) — change together.
    assert!(
        (1..=crate::consts::MAX_NAGQ).contains(&nagq) && nagq % 2 == 1,
        "nagq={} must be odd in 1..={}",
        nagq,
        crate::consts::MAX_NAGQ
    );
    if nagq > 1 {
        let re = model
            .re
            .as_ref()
            .expect("nagq>1 requires a mixed model (re: Some)");
        let single_factor = re.extra_groupings.is_empty();
        let agq_family = matches!(
            model.family,
            Family::Binomial { .. } | Family::Poisson { .. }
        );
        assert!(
            single_factor && agq_family,
            "nagq>1 legal only on a single grouping factor, binomial/Poisson GLMM"
        );
        // q_p = 1 + #primary slopes. The q_p ≥ 4 refusal is a TEMPORARY cost /
        // oracle-coverage boundary (the k^q product grid and the dimension-generic
        // loop have no q limit), not a code limit — lifting it later is deleting
        // this one check. Locked decision: cap AGQ to q_p≤3 until oracle coverage
        // extends past it, rather than let uncapped q_p through untested.
        let q_p = 1 + re.slopes.len();
        assert!(
            q_p <= 3,
            "nagq>1 with q_p={q_p} random effects per group exceeds the temporary \
             q_p≤3 cap (a cost/oracle-coverage boundary, not a code limit)"
        );
    }
    let Some(re) = model.re.as_ref() else {
        return;
    };
    // The RE-envelope caps (extra-grouping count, q_p, q_g) that used to panic
    // here are now `classify_design`'s routing boundary — over-envelope
    // designs route to the sparse path instead of aborting. Only the column-
    // bounds validity asserts remain below (they hold regardless of solver).
    for &col in &re.slopes {
        assert!(
            (col as usize) < p,
            "primary slope column {col} out of range (p={p})"
        );
    }
    for g in &re.extra_groupings {
        for &col in &g.slopes {
            assert!(
                (col as usize) < p,
                "extra-grouping slope column {col} out of range (p={p})"
            );
        }
    }
    // The kernel holds ONE nested slot (`LmmGroupings.nested`) and
    // `from_cluster_spec_ext` would silently let a later `NestedWithin` extra
    // overwrite an earlier one (last wins). The formula frontend detects at most
    // one; this guards the explicit `parent:child` route (defense in depth).
    let n_nested = re
        .extra_groupings
        .iter()
        .filter(|g| matches!(g.relation, GroupingRelation::NestedWithin { .. }))
        .count();
    assert!(
        n_nested <= 1,
        "at most one NestedWithin extra grouping is supported (got {n_nested})"
    );
}
/// Test-only re-export of [`assert_model_shape`] so `spec.rs` can exercise the
/// `nagq` shape-check without routing through a full `fit` call.
#[cfg(test)]
pub(crate) fn assert_model_shape_pub(model: &ModelSpec, p: usize, nagq: u8) {
    assert_model_shape(model, p, nagq);
}

/// Crate-internal re-export of [`spec_sized_from_ids`]: the sparse-Z path's
/// equivalence test (`sparse::tests`) sizes a spec from ids exactly as the stable
/// `fit_warm` entry does, and the `loop_advanced` surface hands it to loop-tier
/// consumers (MCPower) so they normalize RE level counts the same validated way
/// before [`build_workspace`] rather than reimplementing the count derivation.
#[cfg(any(test, feature = "loop_advanced"))]
pub fn spec_sized_from_ids_pub(model: &ModelSpec, ids: &GroupIds) -> ModelSpec {
    spec_sized_from_ids(model, ids)
}
/// Fills `dst[0..n, 0..p]` from row-major `x` (n·p, unweighted). Factored out
/// of `to_col_major` so the `fit_on` hot-path arms (`Ols`/`Glm`/`LmmDense`) can
/// fill an `n_max`-sized buffer allocated once at `build_workspace`, instead of
/// allocating a fresh `n×p` `Mat` every call — the same buffer-reuse pattern
/// `FitKind::GlmmDense`'s `x_mat` already uses. `dst` must be at least `n×p`;
/// rows/columns past `n`/`p` are left untouched.
pub(super) fn fill_col_major(dst: &mut Mat<f64>, x: &[f64], n: usize, p: usize) {
    for i in 0..n {
        for j in 0..p {
            dst[(i, j)] = x[i * p + j];
        }
    }
}

/// Converts row-major `x` (n·p, unweighted) into a column-major faer matrix.
/// Shared by every unweighted site; `fit_ols`'s √wᵢ-scaled loop is a distinct
/// variant and is not routed through this helper. Thin wrapper over
/// [`fill_col_major`] for the call sites that build a throwaway `Mat` once per
/// call regardless (test-only paths, `fit_glm`, `fit_glm_nb`, `fit_glmm_build`)
/// — not on the `fit_on` hot path, so the allocation here is not worth removing.
pub(super) fn to_col_major(x: &[f64], n: usize, p: usize) -> Mat<f64> {
    let mut x_mat = Mat::<f64>::zeros(n.max(1), p.max(1));
    fill_col_major(&mut x_mat, x, n, p);
    x_mat
}

/// Fills `se[target_indices[i]]` from `var_diag[i]` — the target-COMPACT
/// convention (OLS/GLM: `var_diag` is packed `[0..t]` in target order).
/// `se` must already be NaN-filled; entries fail the `>= 0` guard stay NaN
/// (numerically singular target).
pub(super) fn fill_se_compact(var_diag: &[f64], target_indices: &[u32], se: &mut [f64]) {
    for (i, &ti) in target_indices.iter().enumerate() {
        let vd = var_diag[i];
        if vd.is_finite() && vd >= 0.0 {
            se[ti as usize] = vd.sqrt();
        }
    }
}

/// Fills `se[j]` from `var_diag[j]` — the predictor-INDEXED convention
/// (LMM/GLMM: `var_diag` is full-width `[0..p]`, read at each target index).
/// `se` must already be NaN-filled; entries failing the `>= 0` guard stay NaN
/// (numerically singular target).
pub(super) fn fill_se_by_predictor(var_diag: &[f64], target_indices: &[u32], se: &mut [f64]) {
    for &ti in target_indices {
        let vd = var_diag[ti as usize];
        if vd.is_finite() && vd >= 0.0 {
            se[ti as usize] = vd.sqrt();
        }
    }
}

/// An all-NaN `p×p` [`Fit::vcov`] — the shape every path starts from, and the
/// whole of what a non-converged fit reports (mirroring how `se` NaN-fills).
pub(crate) fn nan_vcov(p: usize) -> Vec<Vec<f64>> {
    vec![vec![f64::NAN; p]; p]
}

/// Full `p×p` [`Fit::vcov`] from `l`, the lower-triangular Cholesky factor of a
/// precision matrix (`X'X` for OLS, `X'WX` for GLM, `X'V⁻¹X` for LMM), scaled by
/// `scale`: `vcov = scale · (L L')⁻¹`.
///
/// No new numerics — this is the SE forward solve, kept instead of discarded.
/// `(L L')⁻¹ = L⁻ᵀ L⁻¹`, so `vcov[i][j] = scale · (uᵢ · uⱼ)` where `uⱼ = L⁻¹eⱼ`
/// is exactly the vector each `se` target already forward-solves for
/// (`var_diag_j = scale·‖uⱼ‖²` is this matrix's diagonal — see
/// `ols::fit_suff_stats_t_sq`). Solving each target's column once and forming
/// the Gram costs the `p` solves `se` already pays plus the dot products.
///
/// Only `targets` rows/columns are solved: outside that block there is no `u`
/// to dot and the result stays NaN, so `vcov` is finite exactly where `se` is
/// (the `targets`-subset carve-out). `l` must be
/// the converged factor — every caller gates on `converged` first, the same
/// staleness contract [`crate::ols::OlsFitView::factor`] documents.
pub(crate) fn vcov_from_chol(
    l: faer::MatRef<'_, f64>,
    p: usize,
    target_indices: &[u32],
    scale: f64,
) -> Vec<Vec<f64>> {
    let mut vcov = nan_vcov(p);
    // Column j of L⁻¹, for each target j: forward-solve L·u = e_j, reading
    // L[(i,k)] directly (already lower-triangular, no transposed access).
    let mut cols: Vec<(usize, Vec<f64>)> = Vec::with_capacity(target_indices.len());
    for &tj in target_indices {
        let tj = tj as usize;
        if tj >= p {
            continue;
        }
        let mut u = vec![0.0f64; p];
        for i in 0..p {
            let mut acc = if i == tj { 1.0 } else { 0.0 };
            for k in 0..i {
                acc -= l[(i, k)] * u[k];
            }
            let l_ii = l[(i, i)];
            u[i] = if l_ii == 0.0 { f64::NAN } else { acc / l_ii };
        }
        cols.push((tj, u));
    }
    for (a, (i, ui)) in cols.iter().enumerate() {
        for (j, uj) in cols[a..].iter() {
            let v = scale * ui.iter().zip(uj).map(|(x, y)| x * y).sum::<f64>();
            vcov[*i][*j] = v;
            vcov[*j][*i] = v;
        }
    }
    vcov
}
