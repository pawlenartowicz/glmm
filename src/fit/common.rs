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
/// The primary-then-extras vech walk mirrors `LmmGroupings::from_cluster_spec_ext`'s
/// `vech_start` layout (`lmm.rs:252-267`) — change together.
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
    Fit {
        beta,
        se,
        tau2: fr.tau2,
        dispersion: fr.dispersion,
        converged: fr.converged,
        varcorr: fr.varcorr,
        stddev_se: fr.stddev_se,
        aliased: aliased.to_vec(),
        n_eval: fr.n_eval,
        deviance: fr.deviance,
        singular: fr.singular,
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

/// Mirror of MCPower's contract invariants 19/21 for the standalone `fit` path:
/// the kernel's stack scratch is sized off `MAX_PRIMARY_Q`/`MAX_EXTRA_Q`, so a
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
        // this one check. See the AGQ spec's locked decision #2.
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
    // d1 #2: the RE-envelope caps (extra-grouping count, q_p, q_g) that used to
    // panic here are now `classify_design`'s routing boundary — over-envelope
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

/// Test-only re-export of [`spec_sized_from_ids`] so the sparse-Z path's
/// equivalence test (`sparse::tests`) can size a spec from ids exactly as the
/// stable `fit_warm` entry does, then force the sparse path directly.
#[cfg(test)]
pub(crate) fn spec_sized_from_ids_pub(model: &ModelSpec, ids: &GroupIds) -> ModelSpec {
    spec_sized_from_ids(model, ids)
}
/// Converts row-major `x` (n·p, unweighted) into a column-major faer matrix.
/// Shared by every unweighted site; `fit_ols`'s √wᵢ-scaled loop is a distinct
/// variant and is not routed through this helper.
pub(super) fn to_col_major(x: &[f64], n: usize, p: usize) -> Mat<f64> {
    let mut x_mat = Mat::<f64>::zeros(n.max(1), p.max(1));
    for i in 0..n {
        for j in 0..p {
            x_mat[(i, j)] = x[i * p + j];
        }
    }
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
