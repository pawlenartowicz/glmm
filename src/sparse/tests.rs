//! Tests for the sparse-Z LMM/GLMM solver (`super` + `super::glmm`).

use super::*;
// faer sparse Cholesky call sequence — verified against faer 0.24.4 source.
use faer::sparse::linalg::cholesky::{factorize_symbolic_cholesky, CholeskySymbolicParams};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMat, Triplet};
// Real path for LltRegularization (the sparse cholesky module only uses it privately).
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::{Conj, Mat, Par, Side, Spec};
// Solve trait: blanket impl for SolveCore<T>; must be in scope to call .solve().
use faer::linalg::solvers::Solve;
// AsMatMut: gives as_mat_mut() → MatMut<'_, T>; as_mut() gives &mut Mat, wrong type.
use faer::mat::AsMatMut;

use crate::fit::common_tests::assert_pinned;
use crate::{Family, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing};

/// One extra grouping's per-row level ids, packed as the `extra_ids` shape
/// (`&[Vec<u32>]`) `fit_mle_sparse` expects.
fn cr_as_extra(cr: &[u32]) -> Vec<Vec<u32>> {
    vec![cr.to_vec()]
}

/// Run `f` with the sparse-tail branch forced on (workspaces built inside
/// take the fill-reducing factor regardless of `TAIL_SPARSE_MIN`). Restores
/// the flag before returning; each #[test] has its own thread, so a
/// panicked test cannot leak the flag into another.
fn with_forced_sparse_tail<T>(f: impl FnOnce() -> T) -> T {
    super::FORCE_SPARSE_TAIL.with(|c| c.set(true));
    let out = f();
    super::FORCE_SPARSE_TAIL.with(|c| c.set(false));
    out
}

/// fit_mle_sparse on an in-envelope crossed LMM matches the NoZ fit_mle on
/// β, varcomp, and SE — the superset property. This is the
/// unit-level seed of the both-paths cross-check harness.
#[test]
fn fit_mle_sparse_matches_noz_in_envelope() {
    use faer::Mat;
    let n = 24;
    let p = 2;
    let mut xflat = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut cl = vec![0u32; n];
    let mut cr = vec![0u32; n];
    let mut st = 5u64;
    for i in 0..n {
        let cov = super::test_lcg(&mut st);
        xflat[i * p] = 1.0;
        xflat[i * p + 1] = cov;
        cl[i] = (i % 4) as u32;
        cr[i] = (i % 3) as u32;
        y[i] = 1.0 + 0.5 * cov + super::test_lcg(&mut st);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 4 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 3 },
                slopes: vec![],
            }],
        }),
    };
    let ids = crate::GroupIds {
        primary: cl.clone(),
        extra: vec![cr.clone()],
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };

    let noz = crate::fit_cold(&xflat, &y, n, p, &model, &ids, &opts); // in-envelope ⇒ NoZ
                                                                      // Force the sparse path directly (bypassing classify_design's NoZ route).
    let x = Mat::<f64>::from_fn(n, p, |i, j| xflat[i * p + j]);
    let (sized, _ids, _perm) = crate::fit::spec_sized_from_ids_pub(&model, &ids);
    let sp = super::fit_mle_sparse(
        &xflat,
        &y,
        n,
        p,
        &sized,
        &cl,
        &cr_as_extra(&cr),
        None,
        &opts,
    );

    assert!(sp.converged() && noz.converged());
    for j in 0..p {
        assert!(
            (sp.beta[j] - noz.beta[j]).abs() < 1e-6,
            "β{j} sparse {} vs noz {}",
            sp.beta[j],
            noz.beta[j]
        );
        assert!(
            (sp.se[j] - noz.se[j]).abs() < 1e-6,
            "se{j} sparse {} vs noz {}",
            sp.se[j],
            noz.se[j]
        );
    }
    assert_eq!(sp.varcorr.len(), noz.varcorr.len());
    for (a, b) in sp
        .varcorr
        .iter()
        .flatten()
        .zip(noz.varcorr.iter().flatten())
    {
        assert!((a - b).abs() < 1e-6, "varcorr {a} vs {b}");
    }
    let _ = (x, model);
}

/// Route invariance for detected flat nesting: when every child level falls
/// under a single parent, the nested (padded per-parent ids, `NestedWithin`)
/// and crossed (flat global ids, `Crossed`) parameterizations are the SAME
/// statistical model, so REML deviance and β must agree across routes. This
/// is the correctness lever behind the frontend's inflation-bound detection
/// (`detect_flat_nesting`): whichever way a near-balanced factor is routed,
/// the answer cannot change. Near-balanced shape: children-per-parent
/// {3,2,3} over 3 parents (8 child levels, padded dim 9). Run for both
/// tail branches (see `sparse_deviance_equals_dense_crossed`).
#[test]
fn nested_route_matches_forced_crossed_sparse() {
    run_nested_route_matches_forced_crossed_sparse();
}
#[test]
fn nested_route_matches_forced_crossed_sparse_sparse_tail() {
    with_forced_sparse_tail(run_nested_route_matches_forced_crossed_sparse);
}
fn run_nested_route_matches_forced_crossed_sparse() {
    // 8 children × 6 obs each = 48 rows. Parent of child c: 0,0,0,1,1,2,2,2.
    let parent_of_child: [u32; 8] = [0, 0, 0, 1, 1, 2, 2, 2];
    // Padded per-parent layout (W = 3): child c → parent·3 + local index.
    let padded_of_child: [u32; 8] = [0, 1, 2, 3, 4, 6, 7, 8];
    let n = 48;
    let p = 2;
    let mut xflat = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut cl = vec![0u32; n];
    let mut flat = vec![0u32; n];
    let mut padded = vec![0u32; n];
    let mut st = 11u64;
    let parent_eff = [0.8, -0.3, 0.1];
    let child_eff: Vec<f64> = (0..8).map(|_| 0.5 * super::test_lcg(&mut st)).collect();
    for i in 0..n {
        let c = (i % 8) as u32;
        let cov = super::test_lcg(&mut st);
        xflat[i * p] = 1.0;
        xflat[i * p + 1] = cov;
        cl[i] = parent_of_child[c as usize];
        flat[i] = c;
        padded[i] = padded_of_child[c as usize];
        y[i] = 1.0
            + 0.5 * cov
            + parent_eff[cl[i] as usize]
            + child_eff[c as usize]
            + 0.3 * super::test_lcg(&mut st);
    }
    let spec = |relation: GroupingRelation| ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 3 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation,
                slopes: vec![],
            }],
        }),
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };
    // Detection-nested route: fit_cold routes it NoZ (nested levels don't
    // count toward the crossed cap) and eliminates per family.
    let nested_model = spec(GroupingRelation::NestedWithin { n_per_parent: 1 });
    let nested_ids = crate::GroupIds {
        primary: cl.clone(),
        extra: vec![padded.clone()],
    };
    let nf = crate::fit_cold(&xflat, &y, n, p, &nested_model, &nested_ids, &opts);
    // Forced-crossed Sparse route on the SAME data, flat ids.
    let crossed_model = spec(GroupingRelation::Crossed { n_clusters: 1 });
    let crossed_ids = crate::GroupIds {
        primary: cl.clone(),
        extra: vec![flat.clone()],
    };
    // Ids from the sizing step, not the hand-built pair: the size rule makes the
    // 8-level child the primary here, and the kernel is fed whatever order the
    // sized spec describes.
    let (sized, crossed_ids, _perm) =
        crate::fit::spec_sized_from_ids_pub(&crossed_model, &crossed_ids);
    let sp = super::fit_mle_sparse(
        &xflat,
        &y,
        n,
        p,
        &sized,
        &crossed_ids.primary,
        &crossed_ids.extra,
        None,
        &opts,
    );
    assert!(nf.converged() && sp.converged());
    assert!(
        (nf.deviance - sp.deviance).abs() < 1e-6 * sp.deviance.abs().max(1.0),
        "deviance nested {} vs forced-crossed sparse {}",
        nf.deviance,
        sp.deviance
    );
    for j in 0..p {
        assert!(
            (nf.beta[j] - sp.beta[j]).abs() < 1e-5,
            "β{j} nested {} vs crossed {}",
            nf.beta[j],
            sp.beta[j]
        );
    }
}

/// A >32-variance-component design (over-envelope-by-count ⇒ sparse) fits
/// through the sparse-Z path without the two grouping-cap panics the shared
/// NoZ structures would hit: `add_rows_multi`'s fixed `[usize; 1+MAX_EXTRA_
/// GROUPINGS]` gid array (dropped from `SparseLmmWorkspace::new`)
/// and `from_cluster_spec_ext`'s `n_extras <= MAX_EXTRA_GROUPINGS` guard
/// (removed). 33 components (1 primary + 32 crossed) also exceeds
/// the old 32-bit `pinned_components` ceiling (u64 now).
#[test]
fn sparse_over_32_components_no_overflow() {
    const N_EXTRA: usize = 32; // > MAX_EXTRA_GROUPINGS=6 ⇒ Sparse; +1 primary ⇒ 33 comps
    let n = 60;
    let p = 1;
    let xflat = vec![1.0f64; n * p]; // intercept-only fixed block
    let mut y = vec![0.0f64; n];
    let mut st = 11u64;
    // Each extra grouping g has (2 + g % 2) levels — modest, well-populated.
    let extra: Vec<Vec<u32>> = (0..N_EXTRA)
        .map(|g| {
            let levels = 2 + (g % 2) as u32;
            (0..n).map(|i| (i as u32) % levels).collect()
        })
        .collect();
    let primary: Vec<u32> = (0..n).map(|i| (i % 4) as u32).collect();
    for yi in y.iter_mut() {
        *yi = 1.0 + 0.5 * super::test_lcg(&mut st);
    }

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 4 },
            slopes: vec![],
            extra_groupings: (0..N_EXTRA)
                .map(|g| Grouping {
                    relation: GroupingRelation::Crossed {
                        n_clusters: 2 + (g % 2) as u32,
                    },
                    slopes: vec![],
                })
                .collect(),
        }),
    };
    // Over-envelope-by-count ⇒ classify_design routes to Sparse.
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse
    ));

    let ids = crate::GroupIds { primary, extra };
    let opts = crate::FitOptions {
        target_indices: vec![0],
        ..crate::FitOptions::default()
    };
    let f = crate::fit_cold(&xflat, &y, n, p, &model, &ids, &opts);

    // The bar is: no panic/overflow through the sparse path, and a finite fit.
    assert!(f.converged(), "33-component sparse fit converged");
    assert!(f.beta[0].is_finite(), "β finite");
    assert!(f.se[0].is_finite(), "se finite");

    // Grouping-permutation invariance: reverse the 32 extra id-vectors AND
    // their matching Groupings together (same index in both, so id-vector g
    // still pairs with its own level-count) and refit. The fitted model is
    // the same 33-way variance partition regardless of declaration order, so
    // β0/se0/deviance must agree — but not bitwise, since BOBYQA's θ-space
    // walk (and the sparse factor's fill-in pattern) differs at a different
    // extra-grouping order.
    let re = model.re.as_ref().unwrap();
    let model_rev = ModelSpec {
        family: model.family,
        re: Some(ReStructure {
            sizing: re.sizing.clone(),
            slopes: re.slopes.clone(),
            extra_groupings: re.extra_groupings.iter().rev().cloned().collect(),
        }),
    };
    let ids_rev = crate::GroupIds {
        primary: ids.primary.clone(),
        extra: ids.extra.iter().rev().cloned().collect(),
    };
    let f_rev = crate::fit_cold(&xflat, &y, n, p, &model_rev, &ids_rev, &opts);
    assert!(f_rev.converged(), "reversed-grouping-order fit converged");
    assert!(
        (f_rev.beta[0] - f.beta[0]).abs() < 1e-6 * f.beta[0].abs().max(1.0),
        "reversed β0 {} vs original {}",
        f_rev.beta[0],
        f.beta[0]
    );
    assert!(
        (f_rev.se[0] - f.se[0]).abs() < 1e-6 * f.se[0].abs().max(1.0),
        "reversed se0 {} vs original {}",
        f_rev.se[0],
        f.se[0]
    );
    assert!(
        (f_rev.deviance - f.deviance).abs() < 1e-6 * f.deviance.abs().max(1.0),
        "reversed deviance {} vs original {}",
        f_rev.deviance,
        f.deviance
    );
}

/// Spike: prove the faer 0.24 sparse-LLT call sequence + logdet-off-the-CSC
/// convention on a hand-checked 3×3 SPD matrix. Locks the API the whole
/// sparse path is built on. det(A)=18 ⇒ logdet=ln 18.
#[test]
fn sparse_llt_spike_logdet_and_solve() {
    let n = 3usize;
    // Lower triangle of A = [[4,1,0],[1,3,1],[0,1,2]].
    let tri = [
        Triplet::new(0usize, 0usize, 4.0f64),
        Triplet::new(1, 0, 1.0),
        Triplet::new(1, 1, 3.0),
        Triplet::new(2, 1, 1.0),
        Triplet::new(2, 2, 2.0),
    ];
    let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &tri).unwrap();

    let params = CholeskySymbolicParams {
        supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
        ..Default::default()
    };
    let symbolic = factorize_symbolic_cholesky(
        a.symbolic(),
        Side::Lower,
        Default::default(), // fill-reducing ordering (AMD-family default)
        params,
    )
    .expect("symbolic factorization");

    let mut l_values = vec![0.0f64; symbolic.len_val()];
    let fac_req = symbolic.factorize_numeric_llt_scratch::<f64>(Par::Seq, Spec::default());
    let mut fac_mem = MemBuffer::new(fac_req);
    let llt = symbolic
        .factorize_numeric_llt(
            &mut l_values,
            a.as_ref(),
            Side::Lower,
            LltRegularization::default(),
            Par::Seq,
            MemStack::new(&mut fac_mem),
            Spec::default(),
        )
        .expect("numeric LLT (A is SPD)");

    // Solve A x = b first: llt holds &'out [T] into l_values; must end that
    // borrow before taking &l_values for logdet_llt. `let _ = llt` is the
    // last use of llt (LltRef is Copy) so NLL ends the borrow on the next line.
    let mut rhs = Mat::<f64>::from_fn(3, 1, |i, _| [1.0, 2.0, 3.0][i]);
    let solve_req = symbolic.solve_in_place_scratch::<f64>(1, Par::Seq);
    let mut solve_mem = MemBuffer::new(solve_req);
    llt.solve_in_place_with_conj(
        Conj::No,
        rhs.as_mat_mut(),
        Par::Seq,
        MemStack::new(&mut solve_mem),
    );
    let _ = llt; // ends &'out borrow on l_values

    // Verify logdet: det([[4,1,0],[1,3,1],[0,1,2]]) = 18 ⇒ log det = ln 18.
    let logdet = logdet_llt(&symbolic, &l_values);
    assert!(
        (logdet - 18.0f64.ln()).abs() < 1e-10,
        "logdet {logdet} vs ln 18"
    );

    // Verify solve against the dense LLT of the same matrix.
    let dense = Mat::<f64>::from_fn(3, 3, |i, j| {
        [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]][i][j]
    });
    let bref = Mat::<f64>::from_fn(3, 1, |i, _| [1.0, 2.0, 3.0][i]);
    let x_dense = dense.llt(Side::Lower).unwrap().solve(bref.as_ref());
    for i in 0..3 {
        assert!(
            (rhs[(i, 0)] - x_dense[(i, 0)]).abs() < 1e-10,
            "x[{i}] {} vs dense",
            rhs[(i, 0)]
        );
    }
}

/// `logdet_llt`'s supernodal arm on the spike test's fixture: same 3×3 SPD
/// matrix, `FORCE_SUPERNODAL` symbolic — the diagonal now lives in dense
/// supernode panels, not the simplicial CSC. Same oracle (det = 18) ⇒ the
/// two arms agree; zeroed `l_values` (every diagonal 0) exercises the
/// non-PD `+INFINITY` sentinel on the supernodal read.
#[test]
fn sparse_llt_supernodal_logdet() {
    let n = 3usize;
    // Lower triangle of A = [[4,1,0],[1,3,1],[0,1,2]] (spike test's matrix).
    let tri = [
        Triplet::new(0usize, 0usize, 4.0f64),
        Triplet::new(1, 0, 1.0),
        Triplet::new(1, 1, 3.0),
        Triplet::new(2, 1, 1.0),
        Triplet::new(2, 2, 2.0),
    ];
    let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &tri).unwrap();

    let params = CholeskySymbolicParams {
        supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SUPERNODAL,
        ..Default::default()
    };
    let symbolic = factorize_symbolic_cholesky(
        a.symbolic(),
        Side::Lower,
        Default::default(), // fill-reducing ordering (AMD-family default)
        params,
    )
    .expect("symbolic factorization");
    assert!(
        matches!(
            symbolic.raw(),
            faer::sparse::linalg::cholesky::SymbolicCholeskyRaw::Supernodal(_)
        ),
        "FORCE_SUPERNODAL produced a supernodal symbolic factor"
    );

    let mut l_values = vec![0.0f64; symbolic.len_val()];
    let fac_req = symbolic.factorize_numeric_llt_scratch::<f64>(Par::Seq, Spec::default());
    let mut fac_mem = MemBuffer::new(fac_req);
    symbolic
        .factorize_numeric_llt(
            &mut l_values,
            a.as_ref(),
            Side::Lower,
            LltRegularization::default(),
            Par::Seq,
            MemStack::new(&mut fac_mem),
            Spec::default(),
        )
        .expect("numeric LLT (A is SPD)");

    // Same oracle as the simplicial arm: det(A) = 18 ⇒ logdet = ln 18.
    let logdet = logdet_llt(&symbolic, &l_values);
    assert!(
        (logdet - 18.0f64.ln()).abs() < 1e-10,
        "supernodal logdet {logdet} vs ln 18"
    );

    // Non-PD sentinel: zeroed factor values ⇒ first diagonal read is 0 ⇒ +∞.
    let zeros = vec![0.0f64; symbolic.len_val()];
    assert_eq!(logdet_llt(&symbolic, &zeros), f64::INFINITY);
}

/// Blocked-kernel logdet oracle: `sparse_schur_factor`'s log|L_ZZ|² at an
/// identity-Λ θ (all scalar components 1.0) matches the dense
/// `logdet(Z'Z + I)` — the θ=identity-Λ case, where A = Z'Z + I exactly.
#[test]
fn blocked_logdet_matches_dense_ztz_plus_i() {
    use faer::Mat;
    let n = 4;
    let p = 1;
    let x = Mat::<f64>::from_fn(n, p, |_, _| 1.0);
    let cluster_ids = [0u32, 0, 1, 1];
    let extra_ids = vec![vec![0u32, 1, 0, 1]];
    let y = [1.0f64, 2.0, 3.0, 4.0];
    let model = crate::ModelSpec {
        family: crate::Family::Gaussian,
        re: Some(crate::ReStructure {
            sizing: crate::Sizing::FixedClusters { n_clusters: 2 },
            slopes: vec![],
            extra_groupings: vec![crate::Grouping {
                relation: crate::GroupingRelation::Crossed { n_clusters: 2 },
                slopes: vec![],
            }],
        }),
    };
    let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &[vec![]]);
    let z = super::build_sparse_z(&g, x.as_ref(), &cluster_ids, &extra_ids, n);

    let mut ws =
        super::SparseLmmWorkspace::new(&g, x.as_ref(), &cluster_ids, &extra_ids, &y, n, p, None);
    // θ = [1, 1] (primary scalar, crossed scalar) ⇒ Λ = I ⇒ A = Z'Z + I.
    let ld = super::sparse_schur_factor(&[1.0, 1.0], &mut ws).expect("Z'Z + I is SPD");

    // Dense reference: logdet(Z'Z + I).
    let zd = z.to_dense();
    let mut ztz = zd.transpose() * &zd;
    for d in 0..g.k_total {
        ztz[(d, d)] += 1.0;
    }
    let dense_ld = {
        let l = ztz.llt(faer::Side::Lower).unwrap();
        let ld_mat = l.L();
        let mut s = 0.0;
        for d in 0..g.k_total {
            s += ld_mat[(d, d)].ln();
        }
        2.0 * s
    };
    assert!(
        (ld - dense_ld).abs() < 1e-9,
        "blocked logdet {ld} vs dense {dense_ld}"
    );
}

/// build_sparse_z lays columns out in no-Z RE-column order
/// `[primary | crossed]` with a 1 per row in its primary level and its
/// crossed level (scalar intercepts). Checked against the dense pattern.
#[test]
fn sparse_z_matches_dense_crossed_intercept() {
    use faer::Mat;
    let n = 4;
    let p = 1;
    // Intercept-only X (unused for intercept RE columns, but the signature takes it).
    let x = Mat::<f64>::from_fn(n, p, |_, _| 1.0);
    let cluster_ids = [0u32, 0, 1, 1]; // 2 primary levels
    let extra_ids = vec![vec![0u32, 1, 0, 1]]; // 1 crossed grouping, 2 levels

    let model = crate::ModelSpec {
        family: crate::Family::Gaussian,
        re: Some(crate::ReStructure {
            sizing: crate::Sizing::FixedClusters { n_clusters: 2 },
            slopes: vec![],
            extra_groupings: vec![crate::Grouping {
                relation: crate::GroupingRelation::Crossed { n_clusters: 2 },
                slopes: vec![],
            }],
        }),
    };
    let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &[vec![]]);
    let z = super::build_sparse_z(&g, x.as_ref(), &cluster_ids, &extra_ids, n);

    assert_eq!(z.nrows(), n);
    assert_eq!(z.ncols(), g.k_total);
    // Densify and compare to the expected [primary(2) | crossed(2)] pattern.
    let dense = z.to_dense();
    let expect = [
        [1.0, 0.0, 1.0, 0.0], // row0: primary lvl0, crossed lvl0
        [1.0, 0.0, 0.0, 1.0], // row1: primary lvl0, crossed lvl1
        [0.0, 1.0, 1.0, 0.0], // row2: primary lvl1, crossed lvl0
        [0.0, 1.0, 0.0, 1.0], // row3: primary lvl1, crossed lvl1
    ];
    for i in 0..n {
        for j in 0..g.k_total {
            assert!(
                (dense[(i, j)] - expect[i][j]).abs() < 1e-12,
                "Z[{i},{j}] {}",
                dense[(i, j)]
            );
        }
    }
}

/// sparse_reml_deviance equals lmm::reml_deviance at the same θ on an
/// in-envelope design — the free cross-check at the deviance-value level.
/// If these disagree, exactly one path is wrong. Run for BOTH tail
/// branches: bare (dense tail, e=3 ≤ TAIL_SPARSE_MIN) and forced-sparse
/// (the fill-reducing factor at the same tolerance — AMD reordering is a
/// sanctioned reassociation).
#[test]
fn sparse_deviance_equals_dense_crossed() {
    run_sparse_deviance_equals_dense_crossed();
}
#[test]
fn sparse_deviance_equals_dense_crossed_sparse_tail() {
    with_forced_sparse_tail(run_sparse_deviance_equals_dense_crossed);
}
fn run_sparse_deviance_equals_dense_crossed() {
    use crate::{Family, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing};
    use faer::Mat;
    // Small crossed LMM: primary (2 levels) + 1 crossed grouping (3 levels),
    // scalar intercepts, p=2 fixed (intercept + 1 covariate).
    let n = 12;
    let p = 2;
    let mut x = Mat::<f64>::zeros(n, p);
    let mut y = vec![0.0f64; n];
    let mut cl = vec![0u32; n];
    let mut cr = vec![0u32; n];
    let mut st = 11u64;
    for i in 0..n {
        let cov = super::test_lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = cov;
        cl[i] = (i % 2) as u32;
        cr[i] = (i % 3) as u32;
        y[i] = 1.0 + 0.5 * cov + super::test_lcg(&mut st);
    }
    let extra_ids = vec![cr.clone()];
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 2 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 3 },
                slopes: vec![],
            }],
        }),
    };
    let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &[vec![]]);

    // Dense reference workspace.
    let mut suff = crate::lmm::LmmSuffStats::with_groupings(p, g.clone());
    suff.add_rows_multi(x.as_ref(), &y, &cl, &extra_ids, None);
    let mut fit = crate::lmm::LmmFitScratch::with_groupings(p, &g);

    let mut ws = super::SparseLmmWorkspace::new(&g, x.as_ref(), &cl, &extra_ids, &y, n, p, None);

    // θ has 2 scalar components here (primary intercept var, crossed var).
    for theta in [[0.5f64, 0.7], [1.0, 0.2], [0.1, 1.3]] {
        let dense = crate::lmm::reml_deviance(&theta, &suff, &mut fit);
        let sparse = super::sparse_reml_deviance(&theta, &mut ws);
        assert!(
            (dense - sparse).abs() < 1e-8 * (1.0 + dense.abs()),
            "θ={theta:?}: dense {dense} vs sparse {sparse}"
        );
    }
}

/// Regression for the structural-pattern seeding fix: a PRIMARY RANDOM SLOPE
/// (q_p=2) design whose slope covariate is balanced ±1 within each primary
/// cluster, so `Σx = 0` exactly over every cluster's rows → the intercept×slope
/// cross-Gram entry `Z'Z[(n_prim+f, f)]` is EXACTLY 0.0. Under the old numeric
/// seeding (`v != 0.0`) that off-diagonal within-block slot was never reserved,
/// so a non-diagonal Λ's fill there (Λ'GΛ has a nonzero at that slot) was
/// silently dropped → wrong A → wrong deviance with no error. With structural
/// seeding (`|Z|ᵀ|Z| > 0.0`) the slot exists and the deviance matches the dense
/// oracle. θ is chosen with all three vech components nonzero so Λ is genuinely
/// non-diagonal. Random-continuous data can't hit the exact zero — it must be
/// constructed. Companion to `sparse_deviance_equals_dense_crossed` (scalar Λ).
#[test]
fn sparse_deviance_equals_dense_primary_slope_balanced_zero() {
    use crate::{Family, ModelSpec, ReStructure, Sizing};
    use faer::Mat;
    // 3 primary clusters × 4 rows; within each, 2 rows x=+1 and 2 rows x=-1.
    let n = 12;
    let p = 2; // intercept + the ±1 slope covariate as fixed effects
    let n_clusters = 3u32;
    let mut x = Mat::<f64>::zeros(n, p);
    let mut y = vec![0.0f64; n];
    let mut cl = vec![0u32; n];
    let mut st = 7u64;
    for i in 0..n {
        let slope_cov = if i % 4 < 2 { 1.0 } else { -1.0 }; // Σ over each cluster = 0
        x[(i, 0)] = 1.0;
        x[(i, 1)] = slope_cov;
        cl[i] = (i / 4) as u32;
        y[i] = 0.5 + 0.3 * slope_cov + super::test_lcg(&mut st);
    }
    let extra_ids: Vec<Vec<u32>> = vec![]; // no extra groupings
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters },
            slopes: vec![1], // primary random slope on x column 1
            extra_groupings: vec![],
        }),
    };
    // Primary slope x-col = 1; no extra groupings.
    let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[1], &[]);
    assert_eq!(g.primary_q, 2, "design must be a q_p=2 primary slope");

    // Confirm the target cross-Gram entry is EXACTLY 0.0 (would be dropped
    // under old numeric seeding): intercept col f vs slope col n_prim+f.
    let z = super::build_sparse_z(&g, x.as_ref(), &cl, &extra_ids, n);
    let ztz = z.to_dense().transpose() * &z.to_dense();
    let n_prim = g.n_primary;
    for f in 0..n_prim {
        assert_eq!(
            ztz[(n_prim + f, f)],
            0.0,
            "cross-Gram (slope,intercept) at cluster {f} must be exactly 0"
        );
    }

    // Dense reference workspace (the oracle).
    let mut suff = crate::lmm::LmmSuffStats::with_groupings(p, g.clone());
    suff.add_rows_multi(x.as_ref(), &y, &cl, &extra_ids, None);
    let mut fit = crate::lmm::LmmFitScratch::with_groupings(p, &g);

    let mut ws = super::SparseLmmWorkspace::new(&g, x.as_ref(), &cl, &extra_ids, &y, n, p, None);

    // θ = vech(Λ) with all three components nonzero → Λ genuinely non-diagonal.
    for theta in [[0.8f64, 0.3, 0.6], [1.0, 0.5, 0.4], [0.2, 0.7, 0.9]] {
        let dense = crate::lmm::reml_deviance(&theta, &suff, &mut fit);
        let sparse = super::sparse_reml_deviance(&theta, &mut ws);
        assert!(
            (dense - sparse).abs() < 1e-8 * (1.0 + dense.abs()),
            "θ={theta:?}: dense {dense} vs sparse {sparse}"
        );
    }
}

/// Sparse-tail S22 pattern, pinned on a hand-built two-family fixture with
/// known cliques: 2 primary families over a 5-level crossed factor
/// (declared `n_clusters: 5`), family 0 touching levels {0,1}, family 1
/// {1,2}; levels 3–4 UNOBSERVED (spec count > observed ids — reachable,
/// `n_levels` is unclamped by the ids). Expected scalar pattern (lower):
/// the full diagonal plus exactly the within-clique couplings (1,0) and
/// (2,1) — nothing couples 0↔2 (different families), and the unobserved
/// columns are diagonal-only (their `+I` slot must still exist). Also pins
/// the numerics: forced-sparse deviance equals the dense-tail branch's at
/// several θ, and the full fit matches, unobserved levels included.
#[test]
fn sparse_tail_pattern_two_family_cliques_unobserved_level() {
    use faer::Mat;
    let n = 12;
    let p = 1;
    let x = Mat::<f64>::from_fn(n, p, |_, _| 1.0);
    let xflat = vec![1.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut st = 23u64;
    for v in y.iter_mut() {
        *v = 1.0 + super::test_lcg(&mut st);
    }
    // 3 replicates of each (family, crossed) incidence pair.
    let cl: Vec<u32> = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    let cr: Vec<u32> = vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2];
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 2 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 5 },
                slopes: vec![],
            }],
        }),
    };
    let extra_ids = cr_as_extra(&cr);
    let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &[vec![]]);
    assert_eq!(g.k_crossed(), 5, "unobserved levels counted from the spec");

    let mut ws_sparse = with_forced_sparse_tail(|| {
        super::SparseLmmWorkspace::new(&g, x.as_ref(), &cl, &extra_ids, &y, n, p, None)
    });
    let tail = ws_sparse.tail.as_ref().expect("forced sparse tail");
    let sym = tail.axx.symbolic();
    assert_eq!(
        sym.col_ptr(),
        &[0usize, 2, 4, 5, 6, 7],
        "clique-exact col_ptr"
    );
    assert_eq!(
        sym.row_idx(),
        &[0usize, 1, 1, 2, 2, 3, 4],
        "clique-exact row_idx"
    );

    // Deviance equality vs the dense-tail branch at several θ …
    let mut ws_dense =
        super::SparseLmmWorkspace::new(&g, x.as_ref(), &cl, &extra_ids, &y, n, p, None);
    assert!(ws_dense.tail.is_none(), "e=5 stays on the dense tail bare");
    for theta in [[0.5f64, 0.7], [1.0, 0.2], [0.1, 1.3]] {
        let d = super::sparse_reml_deviance(&theta, &mut ws_dense);
        let s = super::sparse_reml_deviance(&theta, &mut ws_sparse);
        assert!(
            (d - s).abs() < 1e-8 * (1.0 + d.abs()),
            "θ={theta:?}: dense-tail {d} vs sparse-tail {s}"
        );
    }
    // … and full-fit equality (deviance/β/se), unobserved levels included.
    let opts = crate::FitOptions {
        target_indices: vec![0],
        ..crate::FitOptions::default()
    };
    let fd = super::fit_mle_sparse(&xflat, &y, n, p, &model, &cl, &extra_ids, None, &opts);
    let fs = with_forced_sparse_tail(|| {
        super::fit_mle_sparse(&xflat, &y, n, p, &model, &cl, &extra_ids, None, &opts)
    });
    assert!(fd.converged() && fs.converged());
    assert!(
        (fd.deviance - fs.deviance).abs() < 1e-8 * (1.0 + fd.deviance.abs()),
        "deviance dense-tail {} vs sparse-tail {}",
        fd.deviance,
        fs.deviance
    );
    assert!((fd.beta[0] - fs.beta[0]).abs() < 1e-6);
    assert!((fd.se[0] - fs.se[0]).abs() < 1e-6);
}

/// One natural fixture over the cutover (e = 150 > TAIL_SPARSE_MIN): the
/// un-overridden workspace takes the sparse tail, and the fit matches the
/// NoZ oracle (150 crossed levels stay under MAX_CROSSED_LEVELS, so
/// fit_cold routes dense NoZ) — the superset property of
/// `fit_mle_sparse_matches_noz_in_envelope`, across the branch boundary.
#[test]
fn sparse_tail_natural_over_cutover_matches_noz() {
    use faer::Mat;
    let n = 600;
    let p = 2;
    let e_levels = 150u32;
    assert!((e_levels as usize) > super::TAIL_SPARSE_MIN);
    let mut xflat = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut cl = vec![0u32; n];
    let mut cr = vec![0u32; n];
    let mut st = 31u64;
    for i in 0..n {
        let cov = super::test_lcg(&mut st);
        xflat[i * p] = 1.0;
        xflat[i * p + 1] = cov;
        cl[i] = (i % 4) as u32;
        cr[i] = (i as u32) % e_levels;
        y[i] = 1.0 + 0.5 * cov + super::test_lcg(&mut st);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 4 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed {
                    n_clusters: e_levels,
                },
                slopes: vec![],
            }],
        }),
    };
    let ids = crate::GroupIds {
        primary: cl.clone(),
        extra: vec![cr.clone()],
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };
    let noz = crate::fit_cold(&xflat, &y, n, p, &model, &ids, &opts);
    // `model`'s declared counts are already the data's exact counts, so the
    // normalizer is bypassed on the sparse leg deliberately: it would make the
    // 150-level factor the primary (all-scalar all-crossed size rule) and leave
    // a 4-level extra, which is the one arrangement that does NOT build the
    // sparse tail this fixture exists to cross. Both legs still report in
    // declaration order, so the varcorr comparison below is like-for-like.
    // Branch sanity: this design builds a sparse tail without any override.
    {
        let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &[vec![]]);
        let x = Mat::<f64>::from_fn(n, p, |i, j| xflat[i * p + j]);
        let ws =
            super::SparseLmmWorkspace::new(&g, x.as_ref(), &cl, &cr_as_extra(&cr), &y, n, p, None);
        assert!(ws.tail.is_some(), "e=150 routes sparse naturally");
    }
    let sp = super::fit_mle_sparse(
        &xflat,
        &y,
        n,
        p,
        &model,
        &cl,
        &cr_as_extra(&cr),
        None,
        &opts,
    );
    assert!(noz.converged() && sp.converged());
    assert!(
        (sp.deviance - noz.deviance).abs() < 1e-6 * (1.0 + noz.deviance.abs()),
        "deviance sparse {} vs noz {}",
        sp.deviance,
        noz.deviance
    );
    for j in 0..p {
        assert!(
            (sp.beta[j] - noz.beta[j]).abs() < 1e-6,
            "β{j} sparse {} vs noz {}",
            sp.beta[j],
            noz.beta[j]
        );
        assert!(
            (sp.se[j] - noz.se[j]).abs() < 1e-6,
            "se{j} sparse {} vs noz {}",
            sp.se[j],
            noz.se[j]
        );
    }
    for (a, b) in sp
        .varcorr
        .iter()
        .flatten()
        .zip(noz.varcorr.iter().flatten())
    {
        assert!((a - b).abs() < 1e-6, "varcorr {a} vs {b}");
    }
}

/// Deterministic builder for one RE-topology case in the cross-check table.
/// Returns `(xflat, y, n, p, model, ids, opts)`. All designs are
/// in-envelope (q_p ≤ 8, extras ≤ 6, q_g ≤ 4) so `fit_cold` routes to NoZ
/// and `fit_mle_sparse` is a valid superset. `test_lcg` seeds are chosen
/// unique per case so the designs are independent deterministic instances.
/// Shared LCG covariate/response/primary-id fill for `build_case`'s five
/// n=24, p=2 topology cases: column 0 = intercept, column 1 = LCG
/// covariate, `y[i] = c0 + c1·cov + noise`, `pid[i] = i % n_primary`. Any
/// extra grouping id (crossed/nested) is a deterministic function of `i`
/// (and, for nested, `pid`) — no further LCG draws — so callers compute
/// it in a follow-up pass over the returned `pid`, keeping RE-topology
/// setup local to each `build_case` arm.
fn build_case_fill(seed: u64, n_primary: u32, c0: f64, c1: f64) -> (Vec<f64>, Vec<f64>, Vec<u32>) {
    let n = 24;
    let p = 2;
    let mut xflat = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut st = seed;
    for i in 0..n {
        let cov = super::test_lcg(&mut st);
        xflat[i * p] = 1.0;
        xflat[i * p + 1] = cov;
        pid[i] = (i as u32) % n_primary;
        y[i] = c0 + c1 * cov + super::test_lcg(&mut st);
    }
    (xflat, y, pid)
}

fn build_case(
    label: &str,
) -> (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    ModelSpec,
    crate::GroupIds,
    crate::FitOptions,
) {
    match label {
        "scalar_intercept_primary" => {
            // (1 | g): intercept-only primary RE, no extras. q_p=1, n_theta=1.
            let n = 24;
            let p = 2;
            let (xflat, y, pid) = build_case_fill(13, 4, 1.0, 0.5);
            let model = ModelSpec {
                family: Family::Gaussian,
                re: Some(ReStructure {
                    sizing: Sizing::FixedClusters { n_clusters: 4 },
                    slopes: vec![],
                    extra_groupings: vec![],
                }),
            };
            let ids = crate::GroupIds {
                primary: pid,
                extra: vec![],
            };
            let opts = crate::FitOptions {
                target_indices: vec![0, 1],
                ..crate::FitOptions::default()
            };
            (xflat, y, n, p, model, ids, opts)
        }
        "primary_random_slope_q2" => {
            // (1 + x | g): q_p=2 primary with random slope on col 1. The key
            // q_p>1 runtime gate — exercises the non-diagonal primary Λ block
            // in both the dense (reml_deviance_blocked) and sparse paths.
            let n = 24;
            let p = 2;
            let (xflat, y, pid) = build_case_fill(17, 4, 0.5, 0.3);
            let model = ModelSpec {
                family: Family::Gaussian,
                re: Some(ReStructure {
                    sizing: Sizing::FixedClusters { n_clusters: 4 },
                    slopes: vec![1], // random slope on col 1; q_p = 2
                    extra_groupings: vec![],
                }),
            };
            let ids = crate::GroupIds {
                primary: pid,
                extra: vec![],
            };
            let opts = crate::FitOptions {
                target_indices: vec![0, 1],
                ..crate::FitOptions::default()
            };
            (xflat, y, n, p, model, ids, opts)
        }
        "crossed_two_intercepts" => {
            // (1 | g1) + (1 | g2): primary (3 levels) + one crossed extra (4 levels).
            // Periods 3 and 4 are coprime so every (primary, crossed) cell is
            // populated in n=24 rows (lcm(3,4)=12 → 2 full cycles). n_theta=2.
            let n = 24;
            let p = 2;
            let (xflat, y, pid) = build_case_fill(23, 3, 1.0, 0.4);
            let eid: Vec<u32> = (0..n as u32).map(|i| i % 4).collect();
            let model = ModelSpec {
                family: Family::Gaussian,
                re: Some(ReStructure {
                    sizing: Sizing::FixedClusters { n_clusters: 3 },
                    slopes: vec![],
                    extra_groupings: vec![Grouping {
                        relation: GroupingRelation::Crossed { n_clusters: 4 },
                        slopes: vec![],
                    }],
                }),
            };
            let ids = crate::GroupIds {
                primary: pid,
                extra: vec![eid],
            };
            let opts = crate::FitOptions {
                target_indices: vec![0, 1],
                ..crate::FitOptions::default()
            };
            (xflat, y, n, p, model, ids, opts)
        }
        "nested_intercept" => {
            // (1 | g1/g2): primary (4 levels) + nested extra (2 children per
            // parent → 8 global children). Global nested id formula mirrors
            // `ids.rs::extra_level_of_row` for FixedClusters + NestedWithin:
            //   global_id = pid * n_per_parent + (i / n_primary) % n_per_parent
            // giving contiguous [0,2,4,6,1,3,5,7,...] coverage over n=24 rows.
            // k_total = 4 + 4*2 = 12; n_theta = 2 (primary + nested child).
            let n = 24;
            let p = 2;
            let (xflat, y, pid) = build_case_fill(31, 4, 0.8, 0.3);
            // Global nested id: parent·n_per_parent + within_child.
            let cid: Vec<u32> = (0..n).map(|i| pid[i] * 2 + ((i / 4) % 2) as u32).collect();
            let model = ModelSpec {
                family: Family::Gaussian,
                re: Some(ReStructure {
                    sizing: Sizing::FixedClusters { n_clusters: 4 },
                    slopes: vec![],
                    extra_groupings: vec![Grouping {
                        relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
                        slopes: vec![],
                    }],
                }),
            };
            let ids = crate::GroupIds {
                primary: pid,
                extra: vec![cid],
            };
            let opts = crate::FitOptions {
                target_indices: vec![0, 1],
                ..crate::FitOptions::default()
            };
            (xflat, y, n, p, model, ids, opts)
        }
        "primary_slope_plus_crossed" => {
            // (1 + x | g1) + (1 | g2): q_p=2 primary + crossed extra (3 levels).
            // Exercises the non-diagonal Λ_p alongside a crossed tail in both paths.
            // k_total = 4*2 + 3 = 11; n_theta = 3 + 1 = 4.
            let n = 24;
            let p = 2;
            let (xflat, y, pid) = build_case_fill(41, 4, 0.6, 0.4);
            let eid: Vec<u32> = (0..n as u32).map(|i| i % 3).collect();
            let model = ModelSpec {
                family: Family::Gaussian,
                re: Some(ReStructure {
                    sizing: Sizing::FixedClusters { n_clusters: 4 },
                    slopes: vec![1], // random slope on col 1; q_p = 2
                    extra_groupings: vec![Grouping {
                        relation: GroupingRelation::Crossed { n_clusters: 3 },
                        slopes: vec![],
                    }],
                }),
            };
            let ids = crate::GroupIds {
                primary: pid,
                extra: vec![eid],
            };
            let opts = crate::FitOptions {
                target_indices: vec![0, 1],
                ..crate::FitOptions::default()
            };
            (xflat, y, n, p, model, ids, opts)
        }
        other => panic!("unknown cross-check label: {other}"),
    }
}

/// Parses `validation/data/simulated/sim_wide_crossed.csv` into the over-cap
/// `y ~ 1 + x + (1|g1) + (1|c1)+...+(1|c7)` design — 7 crossed intercept
/// extras exceed `MAX_EXTRA_GROUPINGS=6`, so `fit_cold` routes to the
/// sparse-Z path. Shared by the lme4 golden gate and the warm-start A/B
/// test below. Returns `(x row-major, y, n, p, model, ids)`.
fn wide_crossed_design() -> (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    crate::ModelSpec,
    crate::GroupIds,
) {
    let csv = include_str!("../../validation/data/simulated/sim_wide_crossed.csv");
    let mut y = Vec::<f64>::new();
    let mut xcol = Vec::<f64>::new();
    let mut g1_raw = Vec::<String>::new();
    let mut c1_raw = Vec::<String>::new();
    let mut c2_raw = Vec::<String>::new();
    let mut c3_raw = Vec::<String>::new();
    let mut c4_raw = Vec::<String>::new();
    let mut c5_raw = Vec::<String>::new();
    let mut c6_raw = Vec::<String>::new();
    let mut c7_raw = Vec::<String>::new();
    // Columns: y, x, g1, c1, c2, c3, c4, c5, c6, c7 (indices 0..9).
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        xcol.push(f[1].parse().unwrap());
        g1_raw.push(f[2].to_string());
        c1_raw.push(f[3].to_string());
        c2_raw.push(f[4].to_string());
        c3_raw.push(f[5].to_string());
        c4_raw.push(f[6].to_string());
        c5_raw.push(f[7].to_string());
        c6_raw.push(f[8].to_string());
        c7_raw.push(f[9].to_string());
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = xcol[i];
    }

    // Map string factor labels to dense 0-based ids (first-seen order).
    // Inner fn mirrors `dense_str` from fit.rs test module — same pattern.
    fn dense_str(raw: &[String]) -> (Vec<u32>, usize) {
        use std::collections::HashMap;
        let mut map: HashMap<String, u32> = HashMap::new();
        let mut next = 0u32;
        let ids: Vec<u32> = raw
            .iter()
            .map(|r| {
                *map.entry(r.clone()).or_insert_with(|| {
                    let v = next;
                    next += 1;
                    v
                })
            })
            .collect();
        (ids, next as usize)
    }
    let (g1, _) = dense_str(&g1_raw);
    let (c1, _) = dense_str(&c1_raw);
    let (c2, _) = dense_str(&c2_raw);
    let (c3, _) = dense_str(&c3_raw);
    let (c4, _) = dense_str(&c4_raw);
    let (c5, _) = dense_str(&c5_raw);
    let (c6, _) = dense_str(&c6_raw);
    let (c7, _) = dense_str(&c7_raw);

    // n_clusters: 1 placeholders — fit_cold derives true sizes from ids via
    // spec_sized_from_ids. Topology and family are preserved.
    let model = crate::ModelSpec {
        family: crate::Family::Gaussian,
        re: Some(crate::ReStructure {
            sizing: crate::Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![
                crate::Grouping {
                    relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                }, // c1
                crate::Grouping {
                    relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                }, // c2
                crate::Grouping {
                    relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                }, // c3
                crate::Grouping {
                    relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                }, // c4
                crate::Grouping {
                    relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                }, // c5
                crate::Grouping {
                    relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                }, // c6
                crate::Grouping {
                    relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                }, // c7
            ],
        }),
    };
    let ids = crate::GroupIds {
        primary: g1,
        extra: vec![c1, c2, c3, c4, c5, c6, c7],
    };
    (x, y, n, p, model, ids)
}

/// OVER-CAP sparse LMM on the `wide_crossed_design` above: 7 crossed extras
/// exceed `MAX_EXTRA_GROUPINGS`, so the design routes to the sparse solver.
/// The eight scalar variance components are pinned in glmm's own order
/// (primary first, extras in declaration order) — a permuted Z layout moves
/// them, which is what this rung is for.
///
/// Values recorded from glmm. They are validated by `sim_wide_crossed_lmm`,
/// whose cross-engine cell checks the same fit against lme4 and pairs the
/// variance components by group NAME (lme4's `VarCorr` order is descending
/// level count and matches ours only by luck).
///
/// Relative-tolerance, not bit-equal. These values reproduce BIT-EXACTLY on the
/// anchor machine (see `fit::common_tests::assert_pinned`, "which machine the
/// pins are frozen on"); `BAND` is margin for aarch64-apple-darwin, which drifts
/// 1.44e-7 (`se[0]`) from architecture-dependent SIMD/FMA contraction on this
/// kernel's long reductions. ~35x that: loose enough to absorb cross-arch
/// reassociation, tight enough that a real change in the fit still trips it.
#[test]
fn fit_wide_crossed_sparse_is_pinned() {
    const BAND: f64 = 5e-6;
    const REF_BETA: [f64; 2] = [1.7525795303530547, 0.808767308253792];
    const REF_SE: [f64; 2] = [0.6809497161215503, 0.03212803911232081];
    // Eight q=1 blocks, glmm order [g1 | c1..c7]. Variances, not stddevs.
    const REF_VAR: [f64; 8] = [
        0.9096743051917926,
        1.0157902991959213,
        0.3140283552167859,
        0.39010531978955515,
        0.3554082353151701,
        0.45603559620535206,
        0.37148878217578546,
        0.19148041401408306,
    ];
    // Gaussian dispersion = REML σ̂².
    const REF_SIGMA2: f64 = 0.3836290814464871;

    let (x, y, n, p, model, ids) = wide_crossed_design();
    // 7 extras > MAX_EXTRA_GROUPINGS=6 → over-envelope-by-count ⇒ Sparse.
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse,
    ));
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);

    assert!(f.converged(), "sparse wide-crossed fit must converge");
    assert_pinned(&f.beta, &REF_BETA, BAND, "beta");
    assert_pinned(&f.se, &REF_SE, BAND, "se");
    assert_eq!(f.varcorr.len(), 8, "8 scalar varcomp blocks (g1 + c1..c7)");
    let vars: Vec<f64> = f.varcorr.iter().map(|b| b[0]).collect();
    assert_pinned(&vars, &REF_VAR, BAND, "varcorr");
    assert_pinned(&[f.dispersion], &[REF_SIGMA2], BAND, "sigma2");
}

/// Pin the family-downdate route for one workspace construction, mirroring
/// `with_forced_sparse_tail`. `None` is restored so no later test inherits it.
fn with_forced_dd_route<T>(dense: bool, f: impl FnOnce() -> T) -> T {
    super::FORCE_DD_ROUTE.with(|c| c.set(Some(dense)));
    let out = f();
    super::FORCE_DD_ROUTE.with(|c| c.set(None));
    out
}

/// The two family-downdate routes (`FamDowndate`) must produce the same
/// deviance surface. They are the same arithmetic in a different summation
/// order — the scatter applies each family's contribution straight into `vals`
/// (`((s − c₁) − c₂) − c₃`), the dense accumulator sums them first
/// (`s − (c₁ + c₂ + c₃)`) — so they agree to reassociation noise, not bit for
/// bit, and the band below is set at the level that difference actually
/// reaches, not at a tolerance chosen to pass.
///
/// Both arms are exercised at both contraction widths, because the dense route
/// splits on `w`: `w == 1` folds the rank-1 update straight into the
/// accumulator and never touches `dd_temp` (every `(1|g)` crossed design),
/// while `w ≥ 2` keeps faer's triangular matmul and then transfers through the
/// row map. Comparison is at fixed θ, so no optimizer-path divergence enters —
/// a whole-fit comparison would only reproduce this at the optimizer's band.
#[test]
fn family_downdate_dense_route_matches_scatter_route() {
    use faer::Mat;

    // Compare the two routes' deviance over `n_theta`-wide random θ, returning
    // the worst relative gap. `build` constructs a workspace under whatever
    // route/tail overrides are active when it runs.
    fn worst_rel(
        n_theta: usize,
        seed: u64,
        build: impl Fn() -> super::SparseLmmWorkspace,
    ) -> (f64, bool, bool) {
        let mut ws_s = with_forced_dd_route(false, &build);
        let mut ws_d = with_forced_dd_route(true, &build);
        let route_of = |ws: &super::SparseLmmWorkspace| {
            matches!(
                ws.tail
                    .as_ref()
                    .expect("fixture must reach the sparse tail")
                    .fam_dd,
                super::FamDowndate::Dense { .. }
            )
        };
        let (rs, rd) = (route_of(&ws_s), route_of(&ws_d));
        let mut st = seed;
        let mut worst = 0.0f64;
        for _ in 0..8 {
            let theta: Vec<f64> = (0..n_theta)
                .map(|_| 0.3 + 0.6 * super::test_lcg(&mut st))
                .collect();
            let a = super::sparse_reml_deviance(&theta, &mut ws_s);
            let b = super::sparse_reml_deviance(&theta, &mut ws_d);
            worst = worst.max((a - b).abs() / (1.0 + a.abs()));
        }
        (worst, rs, rd)
    }

    // w == 1: the wide-crossed fixture (7 crossed groupings). Its e = 56 is
    // under `TAIL_SPARSE_MIN` — the whole in-crate corpus is, which is why the
    // sparse-tail tests all force it — so `FORCE_SPARSE_TAIL` puts it on the
    // tail the routes live in.
    let (xflat, y, n, p, mut model, ids) = wide_crossed_design();
    let x = Mat::<f64>::from_fn(n, p, |i, j| xflat[i * p + j]);
    // `wide_crossed_design`'s spec carries placeholder level counts (`fit_cold`
    // fills them from the ids); building `LmmGroupings` by hand needs the real
    // ones, or the primary collapses to one cluster.
    let lv = |v: &[u32]| v.iter().max().map_or(1, |m| *m + 1);
    {
        let re = model.re.as_mut().unwrap();
        re.sizing = crate::Sizing::FixedClusters {
            n_clusters: lv(&ids.primary),
        };
        for (gi, gs) in re.extra_groupings.iter_mut().enumerate() {
            gs.relation = crate::GroupingRelation::Crossed {
                n_clusters: lv(&ids.extra[gi]),
            };
        }
    }
    let g1 = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &vec![vec![]; 7]);
    assert_eq!(g1.primary_q, 1, "w == 1 arm needs a scalar primary");
    let (w1, r1s, r1d) = with_forced_sparse_tail(|| {
        worst_rel(8, 0xA51, || {
            super::SparseLmmWorkspace::new(
                &g1,
                x.as_ref(),
                &ids.primary,
                &ids.extra,
                &y,
                n,
                p,
                None,
            )
        })
    });
    assert!(
        !r1s && r1d,
        "w==1: the override must actually split the routes"
    );
    assert!(
        w1 < 1e-12,
        "w==1 routes disagree beyond reassociation: {w1:.3e}"
    );

    // w == 2: primary random slope over a crossed extra, forced onto the sparse
    // tail so a small fixture exercises the faer + row-map-transfer arm.
    let nn = 60;
    let pp = 2;
    let mut xs = Mat::<f64>::zeros(nn, pp);
    let mut ys = vec![0.0f64; nn];
    let mut cl = vec![0u32; nn];
    let mut cr = vec![0u32; nn];
    let mut st = 4242u64;
    for i in 0..nn {
        let cov = super::test_lcg(&mut st);
        xs[(i, 0)] = 1.0;
        xs[(i, 1)] = cov;
        cl[i] = (i % 5) as u32;
        cr[i] = (i % 7) as u32;
        ys[i] = 1.0 + 0.4 * cov + super::test_lcg(&mut st);
    }
    let model2 = crate::ModelSpec {
        family: crate::Family::Gaussian,
        re: Some(crate::ReStructure {
            sizing: crate::Sizing::FixedClusters { n_clusters: 5 },
            slopes: vec![1],
            extra_groupings: vec![crate::Grouping {
                relation: crate::GroupingRelation::Crossed { n_clusters: 7 },
                slopes: vec![],
            }],
        }),
    };
    let extra2 = vec![cr.clone()];
    let g2 = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model2, nn, &[1], &[vec![]]);
    assert_eq!(g2.primary_q, 2, "w == 2 arm needs a primary slope");
    let (w2, r2s, r2d) = with_forced_sparse_tail(|| {
        worst_rel(4, 0xB77, || {
            super::SparseLmmWorkspace::new(&g2, xs.as_ref(), &cl, &extra2, &ys, nn, pp, None)
        })
    });
    assert!(
        !r2s && r2d,
        "w==2: the override must actually split the routes"
    );
    assert!(
        w2 < 1e-12,
        "w==2 routes disagree beyond reassociation: {w2:.3e}"
    );
}

/// Warm-start A/B on the sparse-routed wide-crossed LMM: a warm fit from
/// the frozen lme4 θ̂ ("from the truth" — `Fit` doesn't expose θ̂, and
/// Gaussian tau2/varcorr are both σ²-scaled so θ can't be recovered from
/// the cold fit; θ̂_k = stddev_k/σ̂ from the golden) and one from a
/// perturbed θ must land on the cold optimum — β, SE, varcomp stddevs —
/// and must never degrade convergence status. The sparse sibling of the
/// fit.rs `fit_warm_*_matches_cold_optimum` pair; β start is irrelevant on
/// the LMM path (β is solved exactly given θ).
#[test]
fn fit_warm_sparse_wide_crossed_matches_cold_optimum() {
    let (x, y, n, p, model, ids) = wide_crossed_design();
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse,
    ));
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };
    let cold = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(
        cold.converged(),
        "cold sparse wide-crossed fit must converge"
    );

    // Frozen lme4 golden (sim_wide_crossed_lmm.json): per-grouping stddev
    // in glmm declaration order [g1, c1..c7], and σ̂; θ̂_k = stddev_k/σ̂
    // (scalar blocks).
    const REF_SD: [f64; 8] = [
        0.95374359126349,
        1.00779577183308,
        0.560396926386321,
        0.624586780829176,
        0.596163063210671,
        0.675316308597726,
        0.609496256365909,
        0.437601153218947,
    ];
    const REF_SIGMA: f64 = 0.619378289188346;
    let truth: Vec<f64> = REF_SD.iter().map(|sd| sd / REF_SIGMA).collect();
    let starts = [
        (
            "truth",
            crate::StartValues {
                beta: cold.beta.clone(),
                theta: truth,
            },
        ),
        // θ̂ spans ≈ [0.7, 1.6]; 3.0 everywhere is well off every coordinate.
        (
            "perturbed",
            crate::StartValues {
                beta: vec![0.0; p],
                theta: vec![3.0; 8],
            },
        ),
    ];
    for (label, start) in &starts {
        let warm = crate::fit_warm(&x, &y, n, p, &model, &ids, Some(start), &opts);
        assert!(
            warm.converged(),
            "{label}: warm must not degrade convergence"
        );
        for j in 0..p {
            let rel = (warm.beta[j] - cold.beta[j]).abs() / cold.beta[j].abs();
            assert!(
                rel < 1e-3,
                "{label}: β[{j}] warm {} vs cold {} (rel {rel})",
                warm.beta[j],
                cold.beta[j]
            );
            let rel = (warm.se[j] - cold.se[j]).abs() / cold.se[j];
            assert!(
                rel < 1e-3,
                "{label}: se[{j}] warm {} vs cold {} (rel {rel})",
                warm.se[j],
                cold.se[j]
            );
        }
        // 8 scalar varcomp blocks [g1, c1..c7].
        for k in 0..8 {
            let (w, c) = (warm.varcorr[k][0].sqrt(), cold.varcorr[k][0].sqrt());
            let rel = (w - c).abs() / c;
            assert!(
                rel < 1e-3,
                "{label}: varcomp[{k}] stddev warm {w} vs cold {c} (rel {rel})"
            );
        }
    }
}

/// Cross-check: force Sparse on in-envelope designs and diff every
/// output against NoZ. A mismatch is a bug in exactly one path (NoZ is the
/// oracle). Spans the five RE-topology axes: scalar-intercept, primary slope
/// (q_p=2 runtime gate), crossed, nested, slope+crossed. Run for both tail
/// branches — the forced-sparse pass covers the non-diagonal-Λ crossed
/// tail (`primary_slope_plus_crossed`) through the fill-reducing factor;
/// the e=0 topologies are unaffected by the flag (no tail exists).
#[test]
fn sparse_vs_noz_cross_check_table() {
    run_sparse_vs_noz_cross_check_table();
}
#[test]
fn sparse_vs_noz_cross_check_table_sparse_tail() {
    with_forced_sparse_tail(run_sparse_vs_noz_cross_check_table);
}
fn run_sparse_vs_noz_cross_check_table() {
    let cases: &[&str] = &[
        "scalar_intercept_primary",   // (1 | g)
        "primary_random_slope_q2",    // (1 + x | g), q_p=2 runtime gate
        "crossed_two_intercepts",     // (1 | g1) + (1 | g2)
        "nested_intercept",           // (1 | g1/g2)
        "primary_slope_plus_crossed", // (1 + x | g1) + (1 | g2)
    ];
    for label in cases {
        let (xflat, y, n, p, model, ids, opts) = build_case(label);
        let noz = crate::fit_cold(&xflat, &y, n, p, &model, &ids, &opts);
        let (sized, ids, perm) = crate::fit::spec_sized_from_ids_pub(&model, &ids);
        let mut sp = super::fit_mle_sparse(
            &xflat,
            &y,
            n,
            p,
            &sized,
            &ids.primary,
            &ids.extra,
            None,
            &opts,
        );
        // `noz` came through `fit_cold`, which reports grouping-indexed results
        // in declaration order; the kernel called directly here reports them in
        // the sized spec's slot order.
        perm.swap_slots(&mut sp.varcorr);
        assert!(
            noz.converged() && sp.converged(),
            "{label}: both paths must converge"
        );
        for j in 0..p {
            assert!(
                (sp.beta[j] - noz.beta[j]).abs() < 1e-6,
                "{label} β[{j}]: sparse={} noz={}",
                sp.beta[j],
                noz.beta[j]
            );
            assert!(
                (sp.se[j] - noz.se[j]).abs() < 1e-6,
                "{label} se[{j}]: sparse={} noz={}",
                sp.se[j],
                noz.se[j]
            );
        }
        assert_eq!(
            sp.varcorr.len(),
            noz.varcorr.len(),
            "{label}: varcorr block count"
        );
        for (a, b) in sp
            .varcorr
            .iter()
            .flatten()
            .zip(noz.varcorr.iter().flatten())
        {
            assert!((a - b).abs() < 1e-6, "{label} varcorr: sparse={a} noz={b}");
        }
    }
}

// ── NoZ↔Sparse crossover grid ──────────────────────────────
// Generalizes `build_case`/`sparse_vs_noz_cross_check_table` from 5 hand-
// built topologies to a programmatic sweep of the whole NoZ overlap
// envelope, shared by the accuracy gate (`noz_sparse_grid_agrees`) and the
// timed sweep (`noz_sparse_crossover_timed`).

/// One cell of the crossover grid. All structural cells sit inside the NoZ
/// overlap envelope (`q_p ≤ MAX_PRIMARY_Q`, `n_extra ≤ MAX_EXTRA_GROUPINGS`,
/// `q_g ≤ MAX_EXTRA_Q`) so both kernels are valid on every cell; each side
/// is forced directly (`fit_mle_noz_pub` / `fit_mle_sparse`), bypassing
/// `classify_design` — whose q_g performance boundary these sweeps set.
/// Widths include the intercept: `q_p = 1 + primary slopes`,
/// `q_g = 1 + per-extra slopes`.
#[derive(Clone, Copy)]
struct GridCell {
    /// Rows. Structural cells carry the timing size
    /// (`TIMING_ROWS_PER_RE_COL · re_cols`); the accuracy gate overrides it
    /// with the smaller `ACCURACY_ROWS_PER_RE_COL` sizing. The timed
    /// N-control cells carry their swept value as-is.
    n: usize,
    n_primary: usize,
    q_p: usize,
    n_extra: usize,
    q_g: usize,
}

/// Estimability safety factor `k` in `n ≥ k · total_re_cols`.
/// 4 rows per RE column keeps every cell non-singular (the existing
/// hand-built cases run at ~2.2) while keeping the sweeps cheap. Timing
/// uses the same sizing: the structural cells measure shape, not N — the
/// N-control slice verifies N cancels in the NoZ/Sparse ratio.
const ACCURACY_ROWS_PER_RE_COL: usize = 4;
const TIMING_ROWS_PER_RE_COL: usize = 4;

/// Level count of extra grouping `g`. Distinct per grouping (5, 6, 7, …)
/// so no two crossed factors are identical or mutually confounded.
fn extra_levels(g: usize) -> usize {
    5 + g
}

/// Total random-effect columns of a cell — the estimability driver.
fn re_cols(c: &GridCell) -> usize {
    c.n_primary * c.q_p
        + (0..c.n_extra)
            .map(|g| extra_levels(g) * c.q_g)
            .sum::<usize>()
}

/// Cells too slow for the default `cargo test` gate; they run in the
/// `#[ignore]`d `noz_sparse_grid_agrees_heavy`. Cut from the 2026-07-01
/// release calibration timings: these eight cells cost 27–242s each in
/// release (wide-θ BOBYQA — q_g=4 puts 10 vech entries per extra grouping —
/// or big dense primary patches); the remaining 13 cells total ~4s release.
/// The default subset still spans every axis endpoint except q_g=4, whose
/// only always-on coverage is the over-width lme4 golden
/// (`fit_wide_slopes_sparse_matches_lme4`, q_g=5) vs Sparse — that anchors
/// Sparse to lme4, not Sparse to NoZ, so the on-demand heavy run is the
/// only place the NoZ↔Sparse comparison reaches q_g=4.
fn is_heavy_cell(c: &GridCell) -> bool {
    c.q_g >= 4 || (c.q_p >= 4 && c.n_primary >= 200) || (c.q_p >= 6 && c.n_primary >= 50)
}

/// The structural grid: a 2D `q_p × n_primary` patch
/// (`n_extra = 0`) catching the interaction pure OAT would miss, plus a
/// crossed slice (`n_extra × q_g` at `q_p=2, n_primary=50`). 21 cells; the
/// N-control slice lives in the timed test only (it adds no structure).
fn crossover_structures() -> Vec<GridCell> {
    let mut cells = Vec::new();
    for &q_p in &[1usize, 2, 4, 6, 8] {
        for &n_primary in &[10usize, 50, 200] {
            cells.push(GridCell {
                n: 0,
                n_primary,
                q_p,
                n_extra: 0,
                q_g: 1,
            });
        }
    }
    for &q_g in &[1usize, 4] {
        for &n_extra in &[2usize, 4, 6] {
            cells.push(GridCell {
                n: 0,
                n_primary: 50,
                q_p: 2,
                n_extra,
                q_g,
            });
        }
    }
    for c in cells.iter_mut() {
        c.n = TIMING_ROWS_PER_RE_COL * re_cols(c);
    }
    cells
}

/// Parametric generalization of `build_case`: one deterministic synthetic
/// design per cell. With `w = max(q_p, q_g)` the fixed design is
/// `[1, cov₁…cov_{w−1}]` (`p = w`), so slope indices `1..q_p` / `1..q_g`
/// always reference existing columns. True per-level effects (primary
/// intercept+slopes, extras likewise, amplitude 0.5·LCG) are injected so
/// fitted variance components sit off the θ = 0 boundary. Covariates,
/// effects, and noise all come from `test_lcg(seed)` — unique seed per
/// cell, no wall clock, no `rand`.
fn build_grid_case(
    cell: &GridCell,
    seed: u64,
) -> (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    ModelSpec,
    crate::GroupIds,
    crate::FitOptions,
) {
    let GridCell {
        n,
        n_primary,
        q_p,
        n_extra,
        q_g,
    } = *cell;
    let p = q_p.max(q_g);
    let mut st = seed;

    let pid: Vec<u32> = (0..n).map(|i| (i % n_primary) as u32).collect();
    // Extra ids stride by g+1 so each grouping's level pattern differs from
    // the others' and from the primary's `i % n_primary`; integer division
    // still visits every level for n ≫ (g+1)·levels.
    let extra: Vec<Vec<u32>> = (0..n_extra)
        .map(|g| {
            (0..n)
                .map(|i| ((i / (g + 1)) % extra_levels(g)) as u32)
                .collect()
        })
        .collect();

    let prim_eff: Vec<f64> = (0..n_primary * q_p)
        .map(|_| 0.5 * super::test_lcg(&mut st))
        .collect();
    let extra_eff: Vec<Vec<f64>> = (0..n_extra)
        .map(|g| {
            (0..extra_levels(g) * q_g)
                .map(|_| 0.5 * super::test_lcg(&mut st))
                .collect()
        })
        .collect();

    let mut xflat = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        xflat[i * p] = 1.0;
        for j in 1..p {
            xflat[i * p + j] = super::test_lcg(&mut st);
        }
        let mut mu = 1.0;
        for j in 1..p {
            mu += 0.5 * xflat[i * p + j];
        }
        let c = pid[i] as usize;
        mu += prim_eff[c * q_p];
        for k in 1..q_p {
            mu += prim_eff[c * q_p + k] * xflat[i * p + k];
        }
        for g in 0..n_extra {
            let l = extra[g][i] as usize;
            mu += extra_eff[g][l * q_g];
            for k in 1..q_g {
                mu += extra_eff[g][l * q_g + k] * xflat[i * p + k];
            }
        }
        y[i] = mu + super::test_lcg(&mut st);
    }

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_primary as u32,
            },
            slopes: (1..q_p as u32).collect(),
            extra_groupings: (0..n_extra)
                .map(|g| Grouping {
                    relation: GroupingRelation::Crossed {
                        n_clusters: extra_levels(g) as u32,
                    },
                    slopes: (1..q_g as u32).collect(),
                })
                .collect(),
        }),
    };
    let ids = crate::GroupIds {
        primary: pid,
        extra,
    };
    let opts = crate::FitOptions {
        target_indices: (0..p as u32).collect(),
        ..crate::FitOptions::default()
    };
    (xflat, y, n, p, model, ids, opts)
}

/// The cells both grid-agreement gates draw from: the 21 structural cells
/// plus two appended q_g ∈ {2,3} boundary cells.
///
/// The position of a cell in this vector is load-bearing — it seeds that
/// cell's data (`build_grid_case`), so the list is built whole and then
/// filtered by weight. A cell therefore gets the same data whichever of the
/// two gates runs it, and appending (never inserting) keeps the original 21
/// on the indices their 2026-07-01 calibration was measured at.
fn grid_agreement_cells() -> Vec<GridCell> {
    let mut cells = crossover_structures();
    // q_g ∈ {2,3} coverage at the routing boundary: these widths now route
    // to Sparse (`classify_design`'s slope-extra clause), so NoZ=Sparse
    // parity must hold here too. n_extra=2 keeps both cheap enough
    // (~4 s release combined) for the default gate.
    for &q_g in &[2usize, 3] {
        cells.push(GridCell {
            n: 0,
            n_primary: 50,
            q_p: 2,
            n_extra: 2,
            q_g,
        });
    }
    cells
}

/// Accuracy gate: NoZ = Sparse across the overlap envelope.
/// A disagreement is a bug in exactly one path, never a tuning knob (NoZ is
/// the oracle for the overlap; Sparse is separately anchored to lme4 by the
/// over-cap goldens). Relative bound `|Δ| ≤ TOL·(1 + |ref|)` because the
/// two paths are not identical arithmetic: the deviances agree only to
/// rel ~1e-8 (`sparse_deviance_matches_dense_lmm`) and each side is its own
/// BOBYQA minimization, so θ* can legitimately differ slightly. TOL may be
/// loosened later ONLY with a documented numerical reason.
///
/// The gate's claim is about the OBJECTIVE the two routes share, not about
/// BOBYQA landing at the same point on it. At n_primary=10, q_p=8 (36 θ
/// parameters, effectively flat: only 10 clusters constrain an 8×8 vech
/// block) the objective surface is genuinely multimodal, and a difference at
/// the noise floor is enough to send the two independent minimizations into
/// different basins. Measured on that cell: at four matched-θ probes (flat
/// 1.0, flat 0.5, flat 2.0, a ramp) the dense and sparse profiled-REML
/// objectives agree to ≤1.3e-14 relative — essentially the same function.
/// Yet cold, dense reaches deviance −207.553008925 in 11101 evaluations
/// while sparse reaches −202.142398734 in 1543; warm-restarting the sparse
/// route at the dense route's own θ̂ takes it to −207.553008934 in 1442
/// evaluations, so the sparse cold endpoint is a second, worse basin on the
/// same surface, not an early stop. Four other seeds of the same cell
/// (`0x5eed_000c + k*7919`, k = 1..4) funnel to the same basin and agree on
/// β to ≤5e-7, on deviance to ≤1e-9 — so a basin split is the multimodal
/// exception, not the rule, on this shape of cell.
///
/// So the per-quantity checks below only run when the two routes' deviances
/// agree (same basin, `DEV_TOL`); on a split, β/se/varcorr comparison would
/// be comparing two different fits and is skipped, but the split itself is
/// recorded against `expected_splits` — an unexpected split (new cell splits,
/// or a listed one stops splitting) still fails the test, so this scoping
/// can't turn into a silent pass. What still holds unconditionally, split or
/// not, is the ≤1.3e-14 matched-θ objective agreement this scoping rests on.
///
/// `heavy` picks which side of `is_heavy_cell` to sweep; the two callers
/// partition the grid between them, so every cell runs in exactly one.
/// `expected_splits` is the exact, frozen set of cell indices (within this
/// `heavy` half) allowed to basin-split.
fn run_grid_agreement(heavy: bool, label: &str, expected_splits: &[usize]) {
    // Frozen after the 2026-07-01 full-grid calibration run (release):
    // observed max rel |Δ| = 2.37e-5, at the q_g=4 crossed cells whose
    // θ-space is 23–63-dimensional. That exceeds the 1e-6 starting bound,
    // so the worst cell was investigated before loosening:
    // `crossover_worst_cell_deviance_parity` shows dense and sparse
    // deviance agree there to rel ~1e-15 at arbitrary θ, so the gap is
    // purely BOBYQA termination scatter between two independent
    // high-dimensional minimizations — not a path bug. 1e-4 gives ~4×
    // margin over the observed max. Loosen further ONLY with a comparable
    // documented numerical reason.
    //
    // Shared by both weights deliberately: the bound is a statement about the
    // two kernels, not about cell size, and the cheap gate having slack under
    // it is not a reason to give the two gates different bars.
    const TOL: f64 = 1e-4;
    // The agreeing cells' deviances match to ~1e-9 relative (see the four
    // reseeded probes in the doc comment above), so 1e-6 leaves three orders
    // of margin before "same basin" tips into "different basin" — nowhere
    // near the ~2e-2 relative gap the actual split cell shows.
    const DEV_TOL: f64 = 1e-6;
    let mut max_rel = 0f64;
    let mut worst = String::new();
    let cells = grid_agreement_cells();
    let mut checked = 0usize;
    let mut splits = Vec::new();
    for (idx, c) in cells
        .iter()
        .enumerate()
        .filter(|(_, c)| is_heavy_cell(c) == heavy)
    {
        checked += 1;
        let cell = GridCell {
            n: ACCURACY_ROWS_PER_RE_COL * re_cols(c),
            ..*c
        };
        let (xflat, y, n, p, model, ids, opts) = build_grid_case(&cell, 0x5eed_0000 + idx as u64);
        let t0 = std::time::Instant::now();
        let (sized, ids, _perm) = crate::fit::spec_sized_from_ids_pub(&model, &ids);
        let noz = crate::fit::fit_mle_noz_pub(
            &xflat,
            &y,
            n,
            p,
            &sized,
            &ids.primary,
            &ids.extra,
            None,
            &opts,
        );
        let sp = super::fit_mle_sparse(
            &xflat,
            &y,
            n,
            p,
            &sized,
            &ids.primary,
            &ids.extra,
            None,
            &opts,
        );
        eprintln!(
            "cell {idx}: n={n} n_primary={} q_p={} n_extra={} q_g={} — {:.1}s",
            cell.n_primary,
            cell.q_p,
            cell.n_extra,
            cell.q_g,
            t0.elapsed().as_secs_f64()
        );
        let tag = format!(
            "cell {idx} (n={n}, n_primary={}, q_p={}, n_extra={}, q_g={})",
            cell.n_primary, cell.q_p, cell.n_extra, cell.q_g
        );
        assert!(
            noz.converged() && sp.converged(),
            "{tag}: both paths must converge"
        );
        let dev_rel = (sp.deviance - noz.deviance).abs() / (1.0 + noz.deviance.abs());
        if dev_rel > DEV_TOL {
            eprintln!(
                "{tag}: BASIN SPLIT — sparse deviance={:.9} noz deviance={:.9} rel={dev_rel:.3e} (skipping β/se/varcorr)",
                sp.deviance, noz.deviance
            );
            splits.push(idx);
            continue;
        }
        let mut check = |a: f64, b: f64, what: String| {
            let rel = (a - b).abs() / (1.0 + b.abs());
            if rel > max_rel {
                max_rel = rel;
                worst = format!("{tag} {what}");
            }
            assert!(rel <= TOL, "{tag} {what}: sparse={a} noz={b} rel={rel:.3e}");
        };
        for j in 0..p {
            check(sp.beta[j], noz.beta[j], format!("β[{j}]"));
            check(sp.se[j], noz.se[j], format!("se[{j}]"));
        }
        assert_eq!(
            sp.varcorr.len(),
            noz.varcorr.len(),
            "{tag}: varcorr block count"
        );
        for (bi, (sb, nb)) in sp.varcorr.iter().zip(noz.varcorr.iter()).enumerate() {
            assert_eq!(sb.len(), nb.len(), "{tag}: varcorr[{bi}] len");
            for (ei, (a, b)) in sb.iter().zip(nb.iter()).enumerate() {
                check(*a, *b, format!("varcorr[{bi}][{ei}]"));
            }
        }
    }
    // Report the real margin on success, not just "under the bar".
    eprintln!(
        "{label}: {checked} cells checked, {} basin splits, max rel |Δ| = {max_rel:.3e} at {worst}",
        splits.len()
    );
    // A basin split is not a silent skip: the exact set that's allowed to
    // split is frozen here, so an unexpected split (new cell, or a listed
    // one converging to one basin again) fails the test instead of passing
    // quietly.
    assert_eq!(
        splits, expected_splits,
        "{label}: basin-split cell set changed from the frozen list — investigate before updating it"
    );
}

/// The always-on half of the accuracy gate: the 15 cheap cells, ~4s release.
/// Spans every axis endpoint except q_g=4 — see `is_heavy_cell` for what
/// covers that width instead. Cell 12 (n_primary=10, q_p=8, n_extra=0,
/// q_g=1) is a frozen basin split — see `run_grid_agreement`'s doc comment
/// for the warm-restart evidence that it's a multimodal-surface artifact,
/// not a path bug.
#[test]
fn noz_sparse_grid_agrees() {
    run_grid_agreement(false, "noz_sparse_grid_agrees", &[12]);
}

/// The other half: the 8 cells of `is_heavy_cell`, 27–242s each in release.
/// `#[ignore]`d purely on cost — nothing here needs a feature or a locked
/// machine, it is the same comparison as the cheap half on wider cells.
/// Run it when a change could move the NoZ or Sparse θ-search:
///
/// ```sh
/// cargo test --release noz_sparse_grid_agrees_heavy -- --ignored --nocapture
/// ```
///
/// Cell 20 (n_primary=50, q_p=2, n_extra=6, q_g=4) is a frozen basin split,
/// measured 2026-09-01 on x86_64: sparse converges to deviance
/// −734.101866542, NoZ to −735.990789395, rel 2.563e-3. It is the grid's
/// widest θ space — 63 coordinates, 3 primary vech entries plus 10 per extra
/// grouping — so it is the extreme of the same multimodality
/// `run_grid_agreement`'s doc comment measures at cell 12; the other seven
/// heavy cells stay in band, at max rel 2.32e-5 (cell 18, varcorr).
///
/// The four probes that separate "two basins" from "one route is wrong",
/// all on this cell:
/// 1. At 8 random matched θ the dense and sparse profiled-REML objectives
///    agree to rel ≤ 2.1e-12 — one surface, as at cell 18
///    (`crossover_worst_cell_deviance_parity`).
/// 2. Both cold runs report BOBYQA `Converged`, dense in 6354 evaluations
///    and sparse in 5442, so neither endpoint is an evaluation-cap stop.
/// 3. Each route re-evaluates the OTHER route's endpoint to that endpoint's
///    own value (dense at θ̂_sparse → −734.101866542, sparse at θ̂_dense →
///    −735.990789392): both points are points of the shared objective, not
///    of two different ones.
/// 4. Warm-restarted at the other route's θ̂, sparse reaches −735.990789389
///    in 1671 evaluations, while dense STAYS at −734.101866533 in 1096. The
///    worse point is a genuine local optimum that either route settles into
///    once seeded there — what differs is only which one the cold seed
///    funnels to.
#[test]
#[ignore = "8 heavy cells, 27–242s each — run on demand (see doc-comment)"]
fn noz_sparse_grid_agrees_heavy() {
    // Serialized under alloc-tests so its allocations can't land in a
    // concurrent dhat profiler window on an `-- --ignored` run.
    #[cfg(feature = "alloc-tests")]
    let _serial = crate::test_support::alloc_test_guard();
    run_grid_agreement(true, "noz_sparse_grid_agrees_heavy", &[20]);
}

/// Conditional-mode parity on the blocked dense path. `classify_design`
/// sends any design with an extra-grouping slope to the sparse solver, so
/// `recover_ranef_blocked` is unreachable from `fit_cold` — the only way in
/// is the forced-NoZ entry point, exactly as `noz_sparse_grid_agrees` reaches
/// the blocked deviance. The sparse side is the reference here because
/// `tests/lmm_ranef.rs` anchors it to a brute-force dense BLUP solve.
///
/// Same relative bound and same reason as `noz_sparse_grid_agrees`: the two
/// paths are separate BOBYQA minimizations, so θ* — and with it b̂ — may
/// legitimately differ a little more than round-off.
#[test]
fn noz_sparse_ranef_agrees() {
    const TOL: f64 = 1e-4;
    let c = GridCell {
        n: 0,
        n_primary: 20,
        q_p: 2,
        n_extra: 1,
        q_g: 2,
    };
    let cell = GridCell {
        n: ACCURACY_ROWS_PER_RE_COL * re_cols(&c),
        ..c
    };
    let (xflat, y, n, p, model, ids, opts) = build_grid_case(&cell, 0x5eed_ba1d);
    let (sized, ids, _perm) = crate::fit::spec_sized_from_ids_pub(&model, &ids);
    let noz = crate::fit::fit_mle_noz_pub(
        &xflat,
        &y,
        n,
        p,
        &sized,
        &ids.primary,
        &ids.extra,
        None,
        &opts,
    );
    let sp = super::fit_mle_sparse(
        &xflat,
        &y,
        n,
        p,
        &sized,
        &ids.primary,
        &ids.extra,
        None,
        &opts,
    );
    assert!(
        noz.converged() && sp.converged(),
        "both paths must converge"
    );
    assert_eq!(
        noz.ranef_levels, sp.ranef_levels,
        "blocked and sparse must report the same per-grouping level counts"
    );
    assert_eq!(noz.ranef.len(), sp.ranef.len(), "ranef length");
    assert!(!noz.ranef.is_empty(), "blocked path must recover ranef");
    let mut max_rel = 0f64;
    for (k, (a, b)) in noz.ranef.iter().zip(sp.ranef.iter()).enumerate() {
        let rel = (a - b).abs() / (1.0 + b.abs());
        max_rel = max_rel.max(rel);
        assert!(rel <= TOL, "ranef[{k}]: noz={a} sparse={b} rel={rel:.3e}");
    }
    assert_eq!(noz.fitted.len(), n, "blocked path must report fitted");
    for (i, (a, b)) in noz.fitted.iter().zip(sp.fitted.iter()).enumerate() {
        let rel = (a - b).abs() / (1.0 + b.abs());
        max_rel = max_rel.max(rel);
        assert!(rel <= TOL, "fitted[{i}]: noz={a} sparse={b} rel={rel:.3e}");
    }
    eprintln!("noz_sparse_ranef_agrees: max rel |Δ| = {max_rel:.3e}");
}

/// Deviance-level parity on the grid's worst-disagreeing cell (q_p=2,
/// n_extra=2, q_g=4 — the max-|Δ| cell of the 2026-07-01 calibration run).
/// Dense and sparse deviance agree here to rel ~1e-15 at arbitrary θ, which
/// is the evidence behind `noz_sparse_grid_agrees`' frozen 1e-4 tolerance:
/// the fit-level gap on this cell (2.4e-5) is BOBYQA termination scatter,
/// not a path bug. Cheap (8 deviance evals, no optimization) — stays in the
/// default gate as the standing witness for that calibration argument.
#[test]
fn crossover_worst_cell_deviance_parity() {
    use faer::Mat;
    let c = GridCell {
        n: 0,
        n_primary: 50,
        q_p: 2,
        n_extra: 2,
        q_g: 4,
    };
    let cell = GridCell {
        n: ACCURACY_ROWS_PER_RE_COL * re_cols(&c),
        ..c
    };
    let (xflat, y, n, p, model, ids, _opts) = build_grid_case(&cell, 0x5eed_0000 + 18);
    let x = Mat::<f64>::from_fn(n, p, |i, j| xflat[i * p + j]);
    let prim_slopes: Vec<usize> = (1..cell.q_p).collect();
    let extra_slopes: Vec<Vec<usize>> =
        (0..cell.n_extra).map(|_| (1..cell.q_g).collect()).collect();
    let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &prim_slopes, &extra_slopes);
    let mut suff = crate::lmm::LmmSuffStats::with_groupings(p, g.clone());
    suff.add_rows_multi(x.as_ref(), &y, &ids.primary, &ids.extra, None);
    let mut fit = crate::lmm::LmmFitScratch::with_groupings(p, &g);
    let mut ws =
        super::SparseLmmWorkspace::new(&g, x.as_ref(), &ids.primary, &ids.extra, &y, n, p, None);
    let n_theta = 3 + 2 * 10; // vech(2x2) + 2·vech(4x4)
    let mut st = 99u64;
    let mut max_rel = 0.0f64;
    for t in 0..8 {
        let theta: Vec<f64> = (0..n_theta)
            .map(|_| 0.3 + 0.6 * super::test_lcg(&mut st))
            .collect();
        let dense = crate::lmm::reml_deviance(&theta, &suff, &mut fit);
        let sparse = super::sparse_reml_deviance(&theta, &mut ws);
        let rel = (dense - sparse).abs() / (1.0 + dense.abs());
        eprintln!("θ set {t}: dense={dense:.12e} sparse={sparse:.12e} rel={rel:.3e}");
        max_rel = max_rel.max(rel);
    }
    eprintln!("worst-cell deviance parity: max rel = {max_rel:.3e}");
    assert!(
        max_rel < 1e-8,
        "deviance functions disagree — real path bug"
    );
}

/// Min elapsed µs over an adaptive rep count: one probe call (discarded —
/// it is the cold pass: cache + frequency ramp) sets
/// `reps ≈ target_loop_s / t_probe`, clamped to [1, 30]; the reported min
/// is over the following warm calls. Min because timing noise is one-sided
/// — interference only ever slows a run. Floor 1: it only engages on fits
/// slower than the loop budget (seconds+), where interference is
/// proportionally negligible and more reps would cost minutes per cell.
fn min_time_us<F: FnMut()>(target_loop_s: f64, mut f: F) -> f64 {
    let t0 = std::time::Instant::now();
    f();
    let probe_s = t0.elapsed().as_secs_f64();
    let reps = ((target_loop_s / probe_s.max(1e-9)) as usize).clamp(1, 30);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        f();
        let dt = t0.elapsed().as_secs_f64() * 1e6;
        if dt < best {
            best = dt;
        }
    }
    best
}

/// Machine-state guard: read (never write) the pstate/governor sysfs
/// and report LOCKED/UNLOCKED. A run whose header does not say LOCKED is
/// noise — do not record it.
fn machine_lock_header() {
    let read = |path: &str| {
        std::fs::read_to_string(path)
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "<unreadable>".into())
    };
    let no_turbo = read("/sys/devices/system/cpu/intel_pstate/no_turbo");
    let gov = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    let state = if no_turbo == "1" && gov == "performance" {
        "LOCKED"
    } else {
        "UNLOCKED"
    };
    println!("machine: no_turbo={no_turbo} cpu0_governor={gov} -> {state}");
}

/// The two cells whose single fits run ~100 s (43/63-dim BOBYQA at
/// q_g=4, n_extra ≥ 4): ~15 min of sweep on their own, so they live in
/// `noz_sparse_crossover_heavy_timed` and the main sweep stays ~5 min.
fn is_ultra_heavy_cell(c: &GridCell) -> bool {
    c.q_g >= 4 && c.n_extra >= 4
}

/// Shared driver for the timed sweeps: machine header, then one table row
/// per cell — min-of-N µs per path (adaptive N, probe pass discarded,
/// design built outside the timed region).
fn run_timed_sweep(cells: &[GridCell]) {
    // Rep budget per (cell, path): reps ≈ TARGET_LOOP_S / t_probe,
    // clamped to [1, 30].
    const TARGET_LOOP_S: f64 = 2.0;
    machine_lock_header();
    println!(
        "{:>6} {:>9} {:>4} {:>7} {:>4} {:>12} {:>12} {:>7}  winner",
        "N", "n_prim", "q_p", "n_extra", "q_g", "t_noz_us", "t_sparse_us", "ratio"
    );
    for (idx, cell) in cells.iter().enumerate() {
        let (xflat, y, n, p, model, ids, opts) = build_grid_case(cell, 0x71ED_0000 + idx as u64);
        let (sized, ids, _perm) = crate::fit::spec_sized_from_ids_pub(&model, &ids);
        let t_noz = min_time_us(TARGET_LOOP_S, || {
            std::hint::black_box(crate::fit::fit_mle_noz_pub(
                &xflat,
                &y,
                n,
                p,
                &sized,
                &ids.primary,
                &ids.extra,
                None,
                &opts,
            ));
        });
        let t_sparse = min_time_us(TARGET_LOOP_S, || {
            std::hint::black_box(super::fit_mle_sparse(
                &xflat,
                &y,
                n,
                p,
                &sized,
                &ids.primary,
                &ids.extra,
                None,
                &opts,
            ));
        });
        let ratio = t_sparse / t_noz;
        let winner = if t_noz <= t_sparse { "NoZ" } else { "Sparse" };
        println!(
            "{:>6} {:>9} {:>4} {:>7} {:>4} {:>12.1} {:>12.1} {:>7.2}  {}",
            n, cell.n_primary, cell.q_p, cell.n_extra, cell.q_g, t_noz, t_sparse, ratio, winner
        );
    }
}

/// Timed crossover sweep, main slice (~5 min): all structural cells
/// except the two ultra-heavy ones, plus the N-control slice.
/// `#[ignore]`d — run explicitly, only
/// after the machine is locked (user's call), pinned to one P-core:
///
/// ```sh
/// taskset -c 0 cargo test --release noz_sparse_crossover_timed -- \
///     --ignored --nocapture --test-threads=1
/// ```
#[test]
#[ignore = "timed sweep — run pinned on a user-locked machine (see doc-comment)"]
fn noz_sparse_crossover_timed() {
    // Serialized under alloc-tests so its allocations can't land in a
    // concurrent dhat profiler window on an `-- --ignored` run.
    #[cfg(feature = "alloc-tests")]
    let _serial = crate::test_support::alloc_test_guard();
    let mut cells: Vec<GridCell> = crossover_structures()
        .into_iter()
        .filter(|c| !is_ultra_heavy_cell(c))
        .collect();
    // N control at the baseline structure (q_p=2, n_primary=50, n_extra=0):
    // both paths pay the same shared suff-stat accumulation, so N should
    // cancel in the ratio — this slice verifies that.
    for &n in &[500usize, 2000, 8000] {
        cells.push(GridCell {
            n,
            n_primary: 50,
            q_p: 2,
            n_extra: 0,
            q_g: 1,
        });
    }
    run_timed_sweep(&cells);
}

/// Follow-up sweep tightening the q_g crossover locus: the main grid swept
/// `q_g ∈ {1, 4}` and found NoZ↔Sparse flips between them, so this measures
/// the two skipped widths at the same crossed slice (`q_p=2, n_primary=50`,
/// `n_extra ∈ {2,4,6}`). Own cell list — `crossover_structures()` stays
/// untouched because its cell indices are seed-bound and cited by the
/// calibration comments. Same invocation as the main sweep:
///
/// ```sh
/// taskset -c 0 cargo test --release noz_sparse_crossover_qg23_timed -- \
///     --ignored --nocapture --test-threads=1
/// ```
#[test]
#[ignore = "timed sweep (q_g ∈ {2,3}) — run pinned on a user-locked machine"]
fn noz_sparse_crossover_qg23_timed() {
    // Serialized under alloc-tests so its allocations can't land in a
    // concurrent dhat profiler window on an `-- --ignored` run.
    #[cfg(feature = "alloc-tests")]
    let _serial = crate::test_support::alloc_test_guard();
    let mut cells = Vec::new();
    for &q_g in &[2usize, 3] {
        for &n_extra in &[2usize, 4, 6] {
            cells.push(GridCell {
                n: 0,
                n_primary: 50,
                q_p: 2,
                n_extra,
                q_g,
            });
        }
    }
    for c in cells.iter_mut() {
        c.n = TIMING_ROWS_PER_RE_COL * re_cols(c);
    }
    run_timed_sweep(&cells);
}

/// Timed crossover sweep, ultra-heavy slice (~15 min: two cells whose
/// single fits run ~100 s). Named so neither sweep's filter substring-matches
/// the other. Same invocation as the main sweep, run it when the q_g=4
/// crossed corner matters:
///
/// ```sh
/// taskset -c 0 cargo test --release noz_sparse_crossover_heavy_timed -- \
///     --ignored --nocapture --test-threads=1
/// ```
#[test]
#[ignore = "timed sweep (ultra-heavy cells) — run pinned on a user-locked machine"]
fn noz_sparse_crossover_heavy_timed() {
    // Serialized under alloc-tests so its allocations can't land in a
    // concurrent dhat profiler window on an `-- --ignored` run.
    #[cfg(feature = "alloc-tests")]
    let _serial = crate::test_support::alloc_test_guard();
    let cells: Vec<GridCell> = crossover_structures()
        .into_iter()
        .filter(is_ultra_heavy_cell)
        .collect();
    run_timed_sweep(&cells);
}

/// OVER-WIDTH lme4 REML golden: `y ~ 1 + x1+x2+x3+x4 + (1|gp) + (1 + x1+x2+x3+x4 | ge)`.
/// `ge` carries q_g = 5 (intercept + 4 slopes) > `MAX_EXTRA_Q = 4`, so the design is
/// over-envelope by a single grouping's WIDTH — the one over-cap axis with NO dense
/// (NoZ) twin, since NoZ physically cannot run q_g>4. The sparse-vs-NoZ cross-check
/// (`sparse_vs_noz_cross_check_table`) therefore never reaches q_g>1, so this rung is
/// the *sole* check on the extras LEVEL-MAJOR multi-slope Z column layout at width 5.
/// A wrong layout corrupts the whole fit, so the full 15-element vech of the `ge`
/// block is pinned rather than just its diagonal: a column permutation moves the
/// off-diagonals even where it happens to leave the variances alone.
///
/// Values recorded from glmm. They are validated by `sim_wide_slopes_lmm`, whose
/// cross-engine cell checks the same fit against lme4.
///
/// Re-pinned 2026-08-23 with random-effect design column scaling: `ge` carries
/// four real column scales, so this fit sits in the reassociation band the change
/// allows (worst move here 1.2e-4 on the `ge` off-diagonal covariance).
///
/// Relative-tolerance, not bit-equal. These values reproduce BIT-EXACTLY on the
/// anchor machine (see `fit::common_tests::assert_pinned`, "which machine the
/// pins are frozen on"); `BAND` is margin for aarch64-apple-darwin, which drifts
/// 1.38e-6 (`se[1]`) from architecture-dependent SIMD/FMA contraction on this
/// kernel's long reductions. ~36x that: loose enough to absorb cross-arch
/// reassociation, tight enough that a real change in the fit still trips it.
#[test]
fn fit_wide_slopes_sparse_is_pinned() {
    const BAND: f64 = 5e-5;
    const REF_BETA: [f64; 5] = [
        1.7059457447877522,
        0.6799307380087812,
        -0.5337786880029339,
        0.3961595342142479,
        -0.23725708532053053,
    ];
    const REF_SE: [f64; 5] = [
        0.2635797634299826,
        0.13544570755514324,
        0.11236950513293584,
        0.06849216626722299,
        0.04723930439453695,
    ];
    // gp: scalar block. ge: q=5, column-major lower-triangle vech of D̂.
    const REF_VC_GP: f64 = 0.83171497209491;
    const REF_VC_GE: [f64; 15] = [
        1.0998838508246986,
        -0.02567559548875538,
        0.22422096561536567,
        0.0968135518710221,
        0.08265911370742436,
        0.7159780677389519,
        0.2979603145366615,
        0.02504493141157207,
        0.007685066803612515,
        0.488219498511271,
        0.07347543147914619,
        0.041592437917156784,
        0.17210525013901,
        0.008381294965910347,
        0.07263556319841555,
    ];
    const REF_SIGMA2: f64 = 0.37644138278272;

    let csv = include_str!("../../validation/data/simulated/sim_wide_slopes.csv");
    // Columns: y, x1, x2, x3, x4, gp, ge (indices 0..7).
    let mut y = Vec::<f64>::new();
    let mut xc: [Vec<f64>; 4] = [vec![], vec![], vec![], vec![]];
    let mut gp_raw = Vec::<String>::new();
    let mut ge_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        for k in 0..4 {
            xc[k].push(f[1 + k].parse().unwrap());
        }
        gp_raw.push(f[5].to_string());
        ge_raw.push(f[6].to_string());
    }
    let n = y.len();
    let p = 5; // intercept + x1..x4
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        for k in 0..4 {
            x[i * p + 1 + k] = xc[k][i];
        }
    }

    // Map string factor labels to dense 0-based ids (first-seen order); mirrors
    // `dense_str` in the wide-crossed golden test above.
    fn dense_str(raw: &[String]) -> Vec<u32> {
        use std::collections::HashMap;
        let mut map: HashMap<String, u32> = HashMap::new();
        let mut next = 0u32;
        raw.iter()
            .map(|r| {
                *map.entry(r.clone()).or_insert_with(|| {
                    let v = next;
                    next += 1;
                    v
                })
            })
            .collect()
    }
    let gp = dense_str(&gp_raw);
    let ge = dense_str(&ge_raw);

    // primary gp intercept-only (q_p=1); extra ge with slopes on x1..x4 (q_g=5).
    // n_clusters: 1 placeholders — fit_cold derives true sizes from ids.
    let model = crate::ModelSpec {
        family: crate::Family::Gaussian,
        re: Some(crate::ReStructure {
            sizing: crate::Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![crate::Grouping {
                relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![1, 2, 3, 4], // q_g = 5 > MAX_EXTRA_Q=4 ⇒ over-WIDTH
            }],
        }),
    };
    // q_g=5 over the NoZ envelope WIDTH ⇒ Sparse (over-width, no NoZ twin).
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse,
    ));
    let ids = crate::GroupIds {
        primary: gp,
        extra: vec![ge],
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1, 2, 3, 4],
        ..crate::FitOptions::default()
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);

    assert!(f.converged(), "sparse over-width fit must converge");
    assert_pinned(&f.beta, &REF_BETA, BAND, "beta");
    assert_pinned(&f.se, &REF_SE, BAND, "se");
    // glmm's varcorr order is declaration order [gp(primary,q=1), ge(extra,q=5)],
    // each block the column-major lower-triangle vech of D̂ = σ̂²Λ̂Λ̂'.
    assert_eq!(f.varcorr.len(), 2, "two varcomp blocks (gp + ge)");
    assert_pinned(&f.varcorr[0], &[REF_VC_GP], BAND, "gp varcorr");
    assert_pinned(&f.varcorr[1], &REF_VC_GE, BAND, "ge varcorr");
    assert_pinned(&[f.dispersion], &[REF_SIGMA2], BAND, "sigma2");
}

// ── Sparse non-Gaussian goldens (gamma over-width, NB over-count) ─

/// Shared serde shape for the two sparse gamma goldens (goldens_agq.R's
/// glmm schema). serde ignores unread fields (loglik, corr, …). Gated with
/// the tests that read it — those are the only two cross-engine checks left
/// in this file, everything else here is pinned against glmm's own values.
#[cfg(feature = "oracle-tests")]
#[derive(serde::Deserialize)]
struct SgVcBlock {
    group: String,
    stddev: Vec<f64>,
}
#[cfg(feature = "oracle-tests")]
#[derive(serde::Deserialize)]
struct SgEst {
    beta: Vec<f64>,
    se_hessian: Vec<f64>,
    se_rx: Vec<f64>,
    varcomp: Vec<SgVcBlock>,
    dispersion: Option<f64>,
}
#[cfg(feature = "oracle-tests")]
#[derive(serde::Deserialize)]
struct SgGolden {
    estimates: SgEst,
}

/// Map string factor labels to dense 0-based ids (first-seen order) —
/// the `dense_str` pattern shared by the over-cap golden tests above.
fn dense_ids(raw: &[String]) -> Vec<u32> {
    use std::collections::HashMap;
    let mut map: HashMap<String, u32> = HashMap::new();
    let mut next = 0u32;
    raw.iter()
        .map(|r| {
            *map.entry(r.clone()).or_insert_with(|| {
                let v = next;
                next += 1;
                v
            })
        })
        .collect()
}

/// OVER-WIDTH gamma GLMM golden: `y ~ 1 + x1..x4 + (1|gp) + (1 + x1..x4 | ge)`,
/// gamma/log — `ge` carries q_g = 5 > MAX_EXTRA_Q, so the design routes to the
/// sparse non-Gaussian PIRLS (`fit_glmm_sparse`); no dense twin exists.
/// Gated against frozen `glmer(Gamma("log"))` (`validation/goldens/sim_sparse_gamma.json`).
/// The oracle is sacred.
///
/// Both SE arms are lme4-gated: **Hessian** (glmm's default) against
/// `se_hessian` (the like-for-like pairing the sim_gamma_glmm golden
/// settled), and **Rx** against `se_rx` — glmm's Gamma Rx carries lme4's
/// σ̂² = pwrss/n like `vcov(use.hessian=FALSE)` (`family::glmm_sigma_sq`;
/// unscaled, the two differ by exactly σ̂ on this dataset).
/// Tier 2 (cross-engine): compiled only under `oracle-tests`. It used to skip
/// at runtime under default features and still report PASS, which is strictly
/// worse than `#[ignore]` — a golden that asserted nothing while being counted
/// as covered. A compile-time gate makes its absence visible in the test count.
/// Also ~8 min release (n=1200, 21-dim joint BOBYQA + FD-Hessian SE), so the
/// off-by-default tier is where the cost belongs anyway.
///
/// Not subsumed by the corpus-driven Tier 2 cell of the same name: that one
/// goes through the formula frontend, which orders the slope RE first and so
/// routes the DENSE kernel (`manifest.json` `//gamma_rungs`). This builds the
/// `ModelSpec` by hand to pin the SPARSE orientation. Different claim, both
/// worth having.
#[cfg(feature = "oracle-tests")]
#[test]
fn fit_sparse_gamma_glmm_matches_lme4() {
    let raw = include_str!("../../validation/goldens/sim_sparse_gamma.json");
    let gold: SgGolden = serde_json::from_str(raw).expect("golden JSON parses");

    let csv = include_str!("../../validation/data/simulated/sim_sparse_gamma.csv");
    // Columns: y, x1..x4, gp, ge (indices 0..6).
    let mut y = Vec::<f64>::new();
    let mut xc: [Vec<f64>; 4] = [vec![], vec![], vec![], vec![]];
    let (mut gp_raw, mut ge_raw) = (Vec::<String>::new(), Vec::<String>::new());
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        for k in 0..4 {
            xc[k].push(f[1 + k].parse().unwrap());
        }
        gp_raw.push(f[5].to_string());
        ge_raw.push(f[6].to_string());
    }
    let n = y.len();
    let p = 5;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        for k in 0..4 {
            x[i * p + 1 + k] = xc[k][i];
        }
    }
    let model = crate::ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // sized from ids
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![1, 2, 3, 4], // q_g = 5 > MAX_EXTRA_Q ⇒ Sparse
            }],
        }),
    };
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse
    ));
    let ids = crate::GroupIds {
        primary: dense_ids(&gp_raw),
        extra: vec![dense_ids(&ge_raw)],
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1, 2, 3, 4],
        ..crate::FitOptions::default() // default WaldSe::Hessian (see doc)
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(f.converged(), "sparse gamma GLMM must converge");
    // Exactly one stage-1 trial point exhausts PIRLS_MAX_ITERS on this rung
    // (eval 29/267), and the final re-eval at θ̂ converges cleanly in 5
    // iterations — a rejected trial point, not a truncated returned fit.
    assert_eq!(
        f.diagnostics.notes.len(),
        1,
        "expected exactly one PirlsExhausted note, got {:?}",
        f.diagnostics.notes
    );
    match &f.diagnostics.notes[0] {
        crate::Note::PirlsExhausted { evals, final_eval } => {
            assert_eq!(*evals, 1, "exactly one stage-1 trial point exhausts");
            assert!(
                !final_eval,
                "the final re-eval at θ̂ converges in 5 iterations"
            );
        }
        other => panic!("expected PirlsExhausted, got {other:?}"),
    }

    // β: 2e-2 relative (the over-cap phase-1 band the wide-slopes golden
    // uses); se_hessian at the FD-Hessian floor 3e-2 (compare.R's
    // se_hessian_rel).
    for j in 0..p {
        let rb = gold.estimates.beta[j];
        let rs = gold.estimates.se_hessian[j];
        assert!(
            (f.beta[j] - rb).abs() / rb.abs().max(1e-6) < 2e-2,
            "β[{j}] glmm={} lme4={rb}",
            f.beta[j]
        );
        assert!(
            (f.se[j] - rs).abs() / rs.abs().max(1e-6) < 3e-2,
            "se[{j}] glmm={} lme4={rs}",
            f.se[j]
        );
    }
    // Dispersion: post-fit Pearson moment, same estimator as the golden's
    // hand-computed Σpearson²/(n−p) (the sim_gamma_glmm precedent).
    let rd = gold
        .estimates
        .dispersion
        .expect("gamma golden carries dispersion");
    assert!(
        (f.dispersion - rd).abs() / rd < 3e-2,
        "φ̂ glmm={} lme4={rd}",
        f.dispersion
    );
    // Varcomp via stddev_corr — varcorr is σ̂²-scaled like tau2 (B1 fix),
    // directly lme4's Gamma VarCorr stddev scale. glmm order
    // [gp (primary), ge (extra)]; lme4's VarCorr order is descending level
    // count [ge(40), gp(20)] — map by group NAME.
    let gold_of = |name: &str| {
        gold.estimates
            .varcomp
            .iter()
            .find(|b| b.group == name)
            .expect("golden block")
    };
    let (gp_sds, _) = f.stddev_corr(0);
    let gp_ref = gold_of("gp").stddev[0];
    assert!(
        (gp_sds[0] - gp_ref).abs() / gp_ref.max(1e-6) < 3e-2,
        "gp stddev glmm={:.6} lme4={gp_ref:.6}",
        gp_sds[0]
    );
    let (ge_sds, _) = f.stddev_corr(1);
    let ge_ref = gold_of("ge");
    for (t, &got) in ge_sds.iter().enumerate() {
        let rf = ge_ref.stddev[t];
        assert!(
            (got - rf).abs() / rf.max(1e-6) < 5e-2,
            "ge stddev[{t}] glmm={got:.6} lme4={rf:.6}"
        );
    }

    // Rx arm vs the golden's `se_rx` (σ̂²-scaled, see doc). A second full fit —
    // cheap relative to the Hessian arm's FD sweep on this 21-dim design.
    let f_rx = crate::fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &crate::FitOptions {
            target_indices: vec![0, 1, 2, 3, 4],
            wald_se: crate::WaldSe::Rx,
            ..crate::FitOptions::default()
        },
    );
    assert!(f_rx.converged(), "sparse gamma GLMM (Rx) must converge");
    for j in 0..p {
        let rs = gold.estimates.se_rx[j];
        assert!(
            (f_rx.se[j] - rs).abs() / rs.abs().max(1e-6) < 3e-2,
            "rx se[{j}] glmm={} lme4={rs}",
            f_rx.se[j]
        );
    }
}

/// Rung 46 (`sim_sparse_binomial_bigsd`) must reach the SPARSE solver when it
/// is fitted the way the validation harness fits it — from the formula
/// string, not from a hand-built `ModelSpec`.
///
/// This is a separate assertion from the hand-built ones above on purpose.
/// Random-effect lowering order decides which grouping becomes primary, and a
/// sparse rung was once mis-routed to the dense kernel for a whole release
/// because the frontend extracted slope random effects before intercept ones.
/// Rung 46 is all-intercept, so that particular reordering cannot change the
/// answer — seven extras stay seven extras, over `MAX_EXTRA_GROUPINGS` either
/// way — but "cannot" is the claim this test exists to check rather than assert
/// in prose.
///
/// The fixture is Bernoulli, not Poisson: a Poisson design in this large-θ̂,
/// seven-crossed-extra regime pushes counts into the tens of thousands, and
/// at that scale the deviance-sum rounding noise dominates the FD Hessian's
/// step regardless of solver tuning (see the fixture's own comment block in
/// `validation/prep/gen_large_theta_data.R`, block R4). A Bernoulli response
/// keeps the working weight bounded (`μ(1−μ) ≤ 1/4`) at any θ̂, which removes
/// that noise floor at its source.
///
/// The formula string below is character-for-character the rung's `r_formula`
/// in `validation/manifest.json`. If one changes, both change.
#[cfg(feature = "formula")]
#[test]
fn sparse_binomial_bigsd_formula_routes_sparse() {
    use crate::formula::{Column, Table};

    let csv = include_str!("../../validation/data/simulated/sim_sparse_binomial_bigsd.csv");
    // Columns: y, x, z, g1, c1..c7 (indices 0..10).
    let rows: Vec<Vec<String>> = csv
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .map(|s| s.trim().trim_matches('"').to_string())
                .collect()
        })
        .collect();
    let numeric =
        |col: usize| Column::Numeric(rows.iter().map(|r| r[col].parse().unwrap()).collect());
    let factor = |col: usize| {
        let labels: Vec<String> = rows.iter().map(|r| r[col].clone()).collect();
        Column::factor_from_labels(&labels)
    };
    let mut columns = vec![
        ("y".to_string(), numeric(0)),
        ("x".to_string(), numeric(1)),
        ("z".to_string(), numeric(2)),
        ("g1".to_string(), factor(3)),
    ];
    for k in 0..7 {
        columns.push((format!("c{}", k + 1), factor(4 + k)));
    }
    let table = Table {
        n: rows.len(),
        columns,
    };

    // The manifest's r_formula, character-for-character. Lower it the same way
    // validation/engines/glmm.rs does: that harness strips the literal "1 + "
    // intercept token before calling `lower()`, because this crate's parser
    // treats the intercept as always-implicit and has no
    // term for a literal `1` — `engines/glmm.rs`'s
    // `formula_str.replacen(" ~ 1 + ", " ~ ", 1)`. Then size the spec from the
    // ids before classifying — the crossed-level clause reads
    // `Crossed { n_clusters }`, and frontend placeholders carry 1.
    let r_formula = "y ~ 1 + x + z + (1 | g1) + (1 | c1) + (1 | c2) + (1 | c3) + (1 | c4) + (1 | c5) + (1 | c6) + (1 | c7)";
    let formula_str = r_formula.replacen(" ~ 1 + ", " ~ ", 1);
    let lo = crate::formula::lower(
        &formula_str,
        &table,
        crate::Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
    )
    .unwrap();
    let (sized, _ids, _perm) = crate::fit::spec_sized_from_ids_pub(&lo.model, &lo.ids);
    assert!(
        matches!(
            crate::fit::classify_design_pub(&sized, 1),
            crate::fit::Solver::Sparse
        ),
        "rung 46 must route sparse through the formula frontend"
    );
    assert_eq!(
        lo.model.re.as_ref().unwrap().extra_groupings.len(),
        7,
        "the sparse trigger is the extra-grouping count; if this is not 7 the \
         route is being reached for some other reason"
    );
}

/// Default-tier in-crate pin of `sim_sparse_gamma`'s `WaldSe::Hessian` arm —
/// glmm's default SE method. The cross-engine comparison
/// against lme4 already exists just above
/// (`fit_sparse_gamma_glmm_matches_lme4`), but that test is
/// `#[cfg(feature = "oracle-tests")]` and does not run under plain
/// `cargo test`. The rejected 0.1.4 FD-Hessian seeding change moved THIS
/// fixture's `se_hessian` by −27% and its `stddev_se` to NaN while the
/// entire default tier stayed green, because this cell had no default-tier
/// Hessian coverage at all — not absent everywhere, just missing here. This
/// pin closes that hole. It is self-referential (glmm's own values, not
/// lme4's), so it needs no oracle and catches movement in `cargo test`
/// alone.
///
/// Same design/data/`ModelSpec` construction as
/// `fit_sparse_gamma_glmm_matches_lme4` above (hand-built to force the
/// sparse orientation — `ge` carries q_g = 5 > `MAX_EXTRA_Q`).
///
/// Values are the x86_64 anchor's (`x86_64-unknown-linux-gnu`, Intel Core
/// Ultra 7 265H — see `fit::common_tests::assert_pinned`, "which machine the
/// pins are frozen on"), re-anchored 2026-08-05. First frozen on
/// `aarch64-apple-darwin` with a "re-pin on anchor" marker on every `REF_*`
/// below; both pin sets passed on the anchor unchanged, so this swap is a
/// re-freeze, not a regression fix.
///
/// **Band derivation, measured not assumed.** This fixture had no prior
/// default-tier Hessian pin, so there was no existing cross-platform figure
/// to reuse (unlike the NB test below, which already documents one). The
/// substitute measured here is the same mechanism that produced the NB
/// fixture's own documented drift: a NEON-vs-scalar-forced-pulp lane-width
/// swap on this host, via the committed harness
/// (`validation/lanewidth/run_lanewidth.sh`, its `pulp-0.22.2-scalar-force.patch`
/// and scratch-tree procedure — run here through a scratch-local probe that
/// refits this exact design under `WaldSe::Hessian` instead of the
/// committed probe's `WaldSe::Rx`; the harness itself was not modified).
/// Measured worst movement over both quantities: `se_hessian` 3.85e-4
/// (β[1]'s SE), `stddev_se` 1.20e-3 (θ coordinate 1 of 16 — the `ge`
/// random-slope block). `BAND_HESSIAN = 1.5e-2` clears the measured worst
/// (1.20e-3) by ~12.5×: normal headroom, not the NB pin's deliberately
/// thinner 3e-3 (that number belongs to a different, noisier quantity on a
/// 2-parameter fit — copying it here without its own measurement would be
/// arbitrary, which is exactly what this comment exists to avoid). At
/// 1.5e-2 the pin still catches the 0.1.4 regression class (−27% on
/// `se_hessian`, NaN on `stddev_se`) by 18×–∞.
#[test]
fn fit_sparse_gamma_hessian_is_pinned() {
    // Band unchanged on re-pin (2026-08-05) — its derivation above measured
    // headroom, not a machine-specific value.
    const BAND_HESSIAN: f64 = 1.5e-2;
    // Re-anchored 2026-08-05 on x86_64-unknown-linux-gnu; was aarch64-apple-darwin.
    const REF_SE_HESSIAN: [f64; 5] = [
        0.18919738058439473,
        0.09146492656440869,
        0.08363897681514539,
        0.058861416253009266,
        0.047327399530816265,
    ];
    // Re-anchored 2026-08-05 on x86_64-unknown-linux-gnu; was aarch64-apple-darwin.
    // θ-scale SE, 16 coordinates: gp's 1 (scalar intercept) then ge's 15
    // (vech of the 5×5 slope block).
    const REF_STDDEV_SE: [f64; 16] = [
        0.09870430140167974,
        0.09180799701884287,
        0.09242606966375577,
        0.08329298246933324,
        0.058600841435353594,
        0.04946625171071949,
        0.06731917309870179,
        0.08208207780133939,
        0.05728301432143014,
        0.05109452639817435,
        0.0581195005290471,
        0.055103112849115336,
        0.05128458872417365,
        0.04223593518280066,
        0.05601701110304042,
        0.0410549179101112,
    ];

    let csv = include_str!("../../validation/data/simulated/sim_sparse_gamma.csv");
    // Columns: y, x1..x4, gp, ge (indices 0..6).
    let mut y = Vec::<f64>::new();
    let mut xc: [Vec<f64>; 4] = [vec![], vec![], vec![], vec![]];
    let (mut gp_raw, mut ge_raw) = (Vec::<String>::new(), Vec::<String>::new());
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        for k in 0..4 {
            xc[k].push(f[1 + k].parse().unwrap());
        }
        gp_raw.push(f[5].to_string());
        ge_raw.push(f[6].to_string());
    }
    let n = y.len();
    let p = 5;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        for k in 0..4 {
            x[i * p + 1 + k] = xc[k][i];
        }
    }
    let model = crate::ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // sized from ids
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![1, 2, 3, 4], // q_g = 5 > MAX_EXTRA_Q ⇒ Sparse
            }],
        }),
    };
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse
    ));
    let ids = crate::GroupIds {
        primary: dense_ids(&gp_raw),
        extra: vec![dense_ids(&ge_raw)],
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1, 2, 3, 4],
        ..crate::FitOptions::default() // default WaldSe::Hessian
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(f.converged(), "sparse gamma GLMM (Hessian) must converge");
    assert_pinned(&f.se, &REF_SE_HESSIAN, BAND_HESSIAN, "se_hessian");
    assert_pinned(&f.stddev_se, &REF_STDDEV_SE, BAND_HESSIAN, "stddev_se");
}

/// Pins rung 46's (`sim_sparse_binomial_bigsd`) `se`/`stddev_se` under the
/// default `WaldSe::Hessian` arm in the DEFAULT test tier.
///
/// Why it exists: this is the sparse solver's only large-θ̂ cell, and the FD
/// Hessian's θ step is built per-component as a relative step clamped at 1,
/// so it is the first cell any recalibration of that step will move. Before
/// this fixture existed, the sparse arm's behavior in this regime was
/// untested rather than merely uncalibrated. It is a PIN, not an oracle: it
/// catches movement. The lme4 agreement for this rung lives in the
/// validation harness (`validation/tol.R`'s `sim_sparse_binomial_bigsd`
/// override), not here.
///
/// Design provenance: 3600 rows, one primary grouping `g1` (300 levels × 12)
/// plus seven crossed intercept-only groupings `c1..c7` (8 levels each).
/// Seven extras is over `MAX_EXTRA_GROUPINGS`, which is what routes the fit
/// to the sparse solver; every block is scalar on purpose, which is what
/// keeps `stddev_se` a reported quantity on the lme4 side. Data from
/// `validation/prep/gen_large_theta_data.R` block R4, a Bernoulli design
/// (chosen over an earlier Poisson candidate — see that block's comment for
/// why) tuned so the fitted θ̂ on `g1` lands near 3.9 while the seven extras
/// sit in 0.25..0.67.
///
/// Anchor provenance: values frozen 2026-08-06 on
/// `x86_64-unknown-linux-gnu` (see `fit::common_tests::assert_pinned`,
/// "which machine the pins are frozen on").
///
/// **Band derivation, measured not assumed, and wider than first hoped.** A
/// 1-ULP input sweep (K = 64 draws on `x`; the 0/1 response has no
/// meaningful 1-ULP nudge that stays inside `{0,1}`) measured worst relative
/// movement `se` 2.017e-4, `beta` 2.726e-5, `stddev_se` 3.059e-5, `var`
/// 1.199e-5 — noisier than the ≤1e-5 target the conditioning gate
/// was written for, though the same order of magnitude as an earlier
/// investigation's own measurement on this exact design under a different
/// probe (a 3e-6 post-convergence γ̂ nudge, 8.4e-5 on `se(β₀)`). Neither
/// figure is close to the blow-ups that condemned the alternatives measured
/// during that investigation (a rejected FD-seeding change moved
/// `sim_sparse_gamma`'s `se_hessian` −27% and `sim_sparse_nb`'s −61% —
/// three to four orders of magnitude worse), and the self-noise here (2e-4)
/// is still below the corpus-wide `se_hessian_rel` cross-engine band
/// (1e-3, `validation/tol.R`), so it is read as "noisier than hoped, not
/// pathological." `BAND_HESSIAN` below is 10× the measured worst (2.017e-4),
/// rounded to `2e-3` — do not copy this band to another test, it is this
/// design's own measured floor, not a house default.
///
/// **Injection proof: a pin nobody has seen fail is not evidence.**
/// On a scratch copy, `SPARSE_FD_STEP_REL` was bumped from `1e-4`; at 100×
/// (`1e-2`) this fixture's Hessian barely moved (worst `stddev_se` 3.47e-4,
/// still under `BAND_HESSIAN`) — this design is markedly less step-sensitive
/// at moderate perturbations than the gamma/NB pins, consistent with the
/// self-noise measurement above. At 3000× (`3e-1`) the pin fails cleanly
/// (worst `stddev_se` movement 38%, `stddev_se[4]`). The real tree's
/// `SPARSE_FD_STEP_REL` is never touched by this test — only
/// the scratch copy used for this proof.
#[test]
fn fit_sparse_binomial_bigsd_hessian_is_pinned() {
    const BAND_HESSIAN: f64 = 2e-3;
    // Frozen 2026-08-06 on x86_64-unknown-linux-gnu.
    const REF_SE_HESSIAN: [f64; 3] = [0.4909865616608065, 0.06080857558362023, 0.12165156247329076];
    // Frozen 2026-08-06 on x86_64-unknown-linux-gnu. Eight q=1 blocks, glmm
    // order [g1 | c1..c7].
    const REF_STDDEV_SE: [f64; 8] = [
        0.2572890687064059,
        0.14371548959136732,
        0.12949644379802713,
        0.10753876761222778,
        0.0986178147917577,
        0.1264814679057893,
        0.14014902259598055,
        0.18715584406711308,
    ];

    let (x, y, n, p, model, ids) = sparse_binomial_bigsd_design();
    let opts = crate::FitOptions {
        target_indices: vec![0, 1, 2],
        ..crate::FitOptions::default() // default WaldSe::Hessian
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(
        f.converged(),
        "sparse binomial bigsd GLMM (Hessian) must converge"
    );
    assert_pinned(&f.se, &REF_SE_HESSIAN, BAND_HESSIAN, "se_hessian");
    assert_pinned(&f.stddev_se, &REF_STDDEV_SE, BAND_HESSIAN, "stddev_se");
}

/// `y ~ 1 + x + z + (1|g1) + (1|c1) + … + (1|c7)`, Bernoulli/logit:
/// `(x [n·3 row-major], y, n, p, model, ids)`. Seven intercept-only crossed
/// extras put it over `MAX_EXTRA_GROUPINGS`, so it routes to the sparse GLMM
/// solver — the callers assert that where it matters.
fn sparse_binomial_bigsd_design() -> (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    crate::ModelSpec,
    crate::GroupIds,
) {
    let csv = include_str!("../../validation/data/simulated/sim_sparse_binomial_bigsd.csv");
    // Columns: y, x, z, g1, c1..c7 (indices 0..10).
    let mut y = Vec::<f64>::new();
    let mut xvals = Vec::<(f64, f64)>::new();
    let mut fac: Vec<Vec<String>> = vec![Vec::new(); 8]; // g1, c1..c7
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        xvals.push((f[1].parse().unwrap(), f[2].parse().unwrap()));
        for k in 0..8 {
            fac[k].push(f[3 + k].to_string());
        }
    }
    let n = y.len();
    let p = 3;
    let mut x = vec![0.0f64; n * p];
    for (i, &(xv, zv)) in xvals.iter().enumerate() {
        x[i * p] = 1.0;
        x[i * p + 1] = xv;
        x[i * p + 2] = zv;
    }
    let model = crate::ModelSpec {
        family: Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // sized from ids
            slopes: vec![],
            extra_groupings: (0..7)
                .map(|_| Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                })
                .collect(),
        }),
    };
    let ids = crate::GroupIds {
        primary: dense_ids(&fac[0]),
        extra: fac[1..].iter().map(|f| dense_ids(f)).collect(),
    };
    (x, y, n, p, model, ids)
}

/// Sparse sibling of `fit::glmm_tests::fit_warm_glmm_partial_start_*`: on the
/// sparse GLMM route an EMPTY `beta` or `theta` cold-starts that component
/// alone. This route seeds the joint `[θ | β]` vector by ZIPPING the start into
/// it, so an empty field left unhandled would seed zeros rather than falling
/// back — a silent wrong start, not a panic. Both-empty must therefore be
/// BIT-identical to `fit_cold`; the one-sided arms must land on the cold optimum.
#[test]
fn fit_warm_sparse_glmm_partial_start_cold_starts_the_missing_component() {
    let (x, y, n, p, model, ids) = sparse_binomial_bigsd_design();
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse
    ));
    let opts = crate::FitOptions {
        target_indices: vec![0, 1, 2],
        ..crate::FitOptions::default()
    };
    let cold = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(cold.converged(), "cold sparse binomial GLMM must converge");

    let both_empty = crate::StartValues {
        beta: vec![],
        theta: vec![],
    };
    let empty = crate::fit_warm(&x, &y, n, p, &model, &ids, Some(&both_empty), &opts);
    // Bitwise (not PartialEq): non-target SE slots are NaN and NaN != NaN.
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&cold.beta), bits(&empty.beta));
    assert_eq!(bits(&cold.se), bits(&empty.se));

    // θ̂ (8 scalar blocks, σ² ≡ 1 on binomial) from the cold fit — a truth start
    // on one side, cold on the other.
    let theta_hat: Vec<f64> = cold.varcorr.iter().map(|b| b[0].sqrt()).collect();
    let starts = [
        (
            "theta-only",
            crate::StartValues {
                beta: vec![],
                theta: theta_hat,
            },
        ),
        (
            "beta-only",
            crate::StartValues {
                beta: cold.beta.clone(),
                theta: vec![],
            },
        ),
    ];
    for (label, start) in &starts {
        let warm = crate::fit_warm(&x, &y, n, p, &model, &ids, Some(start), &opts);
        assert!(warm.converged(), "{label}: must converge");
        for j in 0..p {
            let rel = (warm.beta[j] - cold.beta[j]).abs() / cold.beta[j].abs();
            assert!(
                rel < 1e-3,
                "{label}: β[{j}] warm {} vs cold {} (rel {rel})",
                warm.beta[j],
                cold.beta[j]
            );
        }
        for k in 0..8 {
            let (w, c) = (warm.varcorr[k][0].sqrt(), cold.varcorr[k][0].sqrt());
            let rel = (w - c).abs() / c;
            assert!(
                rel < 1e-3,
                "{label}: varcomp[{k}] stddev warm {w} vs cold {c} (rel {rel})"
            );
        }
    }
}

/// OVER-COUNT NB GLMM golden: `y ~ 1 + x + (1|g1) + (1|c1) + … + (1|c7)`,
/// negbin/log — 7 crossed extras > MAX_EXTRA_GROUPINGS route to the sparse NB
/// marginal-θ wrapper (`fit_glmm_nb_sparse`). Rx SE, as the gamma rung.
///
/// **The `Rx` arm's bit-exact Rust pin was dropped 2026-08-06.** This fixture
/// is the crate's worst-conditioned NB fit: a
/// single-ULP input nudge moved `beta[0]` 4.8e-4 median / 1.5e-3 worst
/// (~1e12–1e13 amplification), and its old bit-exact pin needed a 3e-3 band —
/// ~3.8x the drift, the thinnest margin any pin in the crate ever carried —
/// just to survive normal cross-machine rounding. The fix found and closed
/// the CAUSE (a golden-section stopping width three
/// decades tighter than the per-evaluation noise floor, `glm.rs:372`'s
/// provenance comment has the trace), and re-pinned the bit-exact NB gate on
/// `sim_nb`/`sim_nb_nested` instead (`fit_glmm_nb_sim_matches_lme4`,
/// `fit_glmm_nb_nested_unbalanced_matches_lme4`, `src/fit/glmm_tests.rs`) —
/// those fixtures are well-conditioned, so a pin there can actually tell a
/// regression from rounding, which a pin on THIS fixture never could even
/// with the width fix (its conditioning is a property of the design, not the
/// θ-search). Buying a second copy of that same gate here, on the crate's
/// worst-conditioned NB fit, was the trade the original `sim_sparse_nb`
/// rebuild plan tried and abandoned. What replaces the bit-exact
/// pin below is oracle agreement: the fixture converges and its `Rx` arm
/// agrees with frozen `lme4::glmer.nb` (`validation/goldens/sim_sparse_nb.json`)
/// at the same relative bands `validation/tol.R` uses for every other
/// cross-engine cell (`beta_rel`/`se_rel` = 1e-3, `stddev_rel` = 1e-3) — a
/// live reference the sparse route's own routing (`classify_design_pub`
/// below) still gets checked against, with no frozen-Rust value left in this
/// test to drift across machines. The sparse NB route's coverage does not
/// end here: `sparse_glmm_deviance_matches_dense` (fixed θ=5.0, NB among its
/// families) and `sparse_glmm_fit_matches_dense_in_envelope` (β/SE/τ² against
/// the dense NoZ fit) both gate the Λ-block application this route does
/// differently from dense, against a live reference each time — no
/// cross-machine drift accumulates in either.
///
/// **Hessian arm unchanged.** Everything below is about the `Rx`
/// arm this test already ran (`wald_se: WaldSe::Rx` below). Before this
/// addition the crate's default tier never exercised this fixture's
/// `WaldSe::Hessian` SEs at all — the rejected 0.1.4 FD-Hessian seeding
/// change moved this exact fixture's Hessian `se`/`stddev_se` by -61% with
/// nothing in `cargo test` able to notice. One extra `fit_cold` under
/// default options (glmm's default IS `WaldSe::Hessian`) plus pins on `se`
/// and `stddev_se` close that.
///
/// Values are the x86_64 anchor's (`x86_64-unknown-linux-gnu`, Intel Core
/// Ultra 7 265H — see `fit::common_tests::assert_pinned`, "which machine the
/// pins are frozen on"), re-anchored 2026-08-05. First frozen on
/// `aarch64-apple-darwin` with a "re-pin on anchor" marker on the
/// `REF_*_HESSIAN` constants below; both pin sets passed on the anchor
/// unchanged, so this swap is a re-freeze, not a regression fix.
///
/// **Band derivation, honest not copied.** The brief for this arm is
/// explicit: do not reuse `BAND` (3e-3) unexamined. Measured directly with
/// the committed lane-width harness (`validation/lanewidth/`, NEON vs a
/// scalar-forced pulp on this same host — the same mechanism that produced
/// this comment's own 7.91e-4/8.5e-4 cross-platform figures above), refit
/// under `WaldSe::Hessian` instead of the committed probe's `WaldSe::Rx`:
/// worst movement `se` 8.27e-5, `stddev_se` 9.09e-4 (component 2 of 8).
/// Both sit at or below this fixture's own documented beta[0] drift
/// (7.91e-4 cross-machine, 5.58e-4 NEON-vs-scalar on this Mac per
/// `validation/lanewidth/README.md`'s worked example) — expected, since
/// `se`/`stddev_se` ride on the same joint (θ,β) FD-Hessian machinery beta
/// does. `BAND_HESSIAN = 1e-2` clears the measured worst (9.09e-4) by ~11x
/// and the documented cross-machine figure (7.91e-4) by ~13x: normal
/// headroom, deliberately NOT this test's existing 3.8x-margin `BAND` (that
/// number was sized for the `Rx`-arm beta/varcorr quantities specifically,
/// on grounds stated above that do not transfer here unexamined). At 1e-2
/// the pin still catches the 0.1.4 regression class (-61% on Hessian SEs)
/// by 61x.
#[test]
fn fit_sparse_nb_glmm_is_pinned() {
    // Oracle-agreement bands, matching `validation/tol.R`'s corpus-wide
    // `beta_rel`/`se_rel`/`stddev_rel` (all 1e-3) — the same numbers the
    // cross-engine tier (`cargo test --features oracle-tests`) uses for this
    // exact golden via `m3_corpus()`. No frozen-Rust value here; the
    // reference is `validation/goldens/sim_sparse_nb.json`
    // (`lme4::glmer.nb`).
    const BAND: f64 = 1e-3;
    const REF_BETA: [f64; 2] = [0.508973335305305, 0.47617747616338];
    const REF_SE_RX: [f64; 2] = [0.369726927892902, 0.0610141749906039];
    // Eight q=1 blocks, glmm order [g1 | c1..c7]. Stddevs, matching the
    // golden's `varcomp[].stddev` — compared against `sqrt(varcorr[i][0])`
    // below, not the raw variance.
    const REF_SD: [f64; 8] = [
        0.618024381330367,
        0.284810042975813,
        0.302721286424016,
        0.465909143152014,
        0.399510526735986,
        0.174143226709312,
        0.382558062655764,
        0.283525863406369,
    ];
    // NB θ̂ rides in `dispersion`, from the marginal golden-section search.
    const REF_THETA: f64 = 1.39610186766246;

    let csv = include_str!("../../validation/data/simulated/sim_sparse_nb.csv");
    // Columns: y, x, g1, c1..c7 (indices 0..9).
    let mut y = Vec::<f64>::new();
    let mut xcol = Vec::<f64>::new();
    let mut fac: Vec<Vec<String>> = vec![Vec::new(); 8]; // g1, c1..c7
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        xcol.push(f[1].parse().unwrap());
        for k in 0..8 {
            fac[k].push(f[2 + k].to_string());
        }
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = xcol[i];
    }
    let model = crate::ModelSpec {
        family: Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // sized from ids
            slopes: vec![],
            extra_groupings: (0..7)
                .map(|_| Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                })
                .collect(),
        }),
    };
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse
    ));
    let ids = crate::GroupIds {
        primary: dense_ids(&fac[0]),
        extra: fac[1..].iter().map(|f| dense_ids(f)).collect(),
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        wald_se: crate::WaldSe::Rx,
        ..crate::FitOptions::default()
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(f.converged(), "sparse NB GLMM must converge");
    assert_pinned(&f.beta, &REF_BETA, BAND, "beta vs lme4");
    assert_pinned(&f.se, &REF_SE_RX, BAND, "se_rx vs lme4");
    assert_eq!(f.varcorr.len(), 8, "8 scalar varcomp blocks");
    let sds: Vec<f64> = f.varcorr.iter().map(|b| b[0].sqrt()).collect();
    assert_pinned(&sds, &REF_SD, BAND, "varcorr stddev vs lme4");
    assert_pinned(&[f.dispersion], &[REF_THETA], BAND, "theta vs lme4");

    // Hessian arm (see the doc comment above for the band
    // derivation). Same data, same model, one extra `fit_cold` under
    // default options (glmm's default IS `WaldSe::Hessian`).
    // Band unchanged on re-pin (2026-08-05) — its derivation above measured
    // headroom, not a machine-specific value.
    const BAND_HESSIAN: f64 = 1e-2;
    // Re-anchored 2026-08-05 on x86_64-unknown-linux-gnu; was aarch64-apple-darwin.
    const REF_SE_HESSIAN: [f64; 2] = [0.3704422044067882, 0.062359975459493726];
    // Re-anchored 2026-08-05 on x86_64-unknown-linux-gnu; was aarch64-apple-darwin.
    // θ-scale SE, 8 coordinates (one per scalar grouping: g1, c1..c7).
    const REF_STDDEV_SE: [f64; 8] = [
        0.14775362243032253,
        0.10397840749812272,
        0.10740948771648906,
        0.13998274984112666,
        0.1296563634917967,
        0.09305621725489749,
        0.12372555179327623,
        0.10387227423118507,
    ];
    let f_hessian = crate::fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &crate::FitOptions {
            target_indices: vec![0, 1],
            ..crate::FitOptions::default() // default WaldSe::Hessian
        },
    );
    assert!(
        f_hessian.converged(),
        "sparse NB GLMM (Hessian) must converge"
    );
    assert_pinned(&f_hessian.se, &REF_SE_HESSIAN, BAND_HESSIAN, "se_hessian");
    assert_pinned(
        &f_hessian.stddev_se,
        &REF_STDDEV_SE,
        BAND_HESSIAN,
        "stddev_se",
    );
}

/// Weighted twin of `fit_sparse_gamma_glmm_matches_lme4` (Task 7): same
/// over-width design and data. Uses `wᵢ = 1 + 0.2·((i mod 3) − 1)`
/// (0-based row index, cycling 0.8/1.0/1.2), NOT the integer `1 + (i mod
/// 3)` scheme the Gamma/NB replication tests use: on THIS wide design
/// (q_g = 5 slope-block extra, 21-dim joint BOBYQA), integer weights up
/// to 3× drove lme4's `vcov(use.hessian=TRUE)` to implausible SE ~250×
/// tighter than the unweighted golden's (0.0007 vs 0.19, same effect
/// sizes, only a mild dispersion shift) — a numerically unstable Hessian
/// on this over-parameterized shape, not a real precision gain (verified
/// interactively; `isSingular` false, no convergence messages, yet the
/// Hessian SE is not credible). The gentler weights keep glmer's Hessian
/// well-conditioned (SE lands back at the unweighted golden's scale) while
/// still exercising the same weighted code path. Closes the sparse Gamma
/// weighting gap — profiled dispersion (`gamma_aic`) and the post-fit
/// Pearson φ̂ both take `ws.prior_w`. Tier 2, gated behind `oracle-tests` like
/// its unweighted sibling — same cross-engine claim, and the same 21-dim joint
/// BOBYQA plus FD-Hessian SE cost.
/// Generated with (R 4.5.3, lme4 1.1-38):
/// ```r
///   d$w <- 1 + 0.2 * (((seq_len(nrow(d)) - 1) %% 3) - 1)
///   f <- glmer(y ~ 1 + x1 + x2 + x3 + x4 + (1|gp) + (1 + x1 + x2 + x3 + x4 | ge),
///              family = Gamma("log"), weights = d$w, data = d)
/// ```
/// se_hessian/dispersion at the unweighted golden's bands (3e-2). β at
/// 4e-2, not the unweighted golden's 2e-2: `x1..x4` land within 1% (the
/// weighting math is exact there), but `(Intercept)` — the design's
/// least-identified coefficient, t ≈ 1.2, SE ≈ 80% of the point estimate
/// — drifts ~3.4% between glmm's and lme4's independent 21-dim BOBYQA
/// paths to the same shallow optimum; se_hessian on that same coefficient
/// still lands within 0.1%, confirming the curvature (hence the
/// weighting) is correct and this is optimizer-path scatter on a
/// poorly-determined direction, not a weighting bug.
#[cfg(feature = "oracle-tests")]
#[test]
fn fit_sparse_gamma_glmm_weighted_matches_lme4() {
    const REF_BETA: [f64; 5] = [
        0.233369872688657,
        0.511759152360149,
        -0.345961162194708,
        0.236273530986550,
        -0.228445413694595,
    ];
    const REF_SE_HESSIAN: [f64; 5] = [
        0.1890907576028805,
        0.0921338121875385,
        0.0846377387177761,
        0.0585608862317671,
        0.0475892885636111,
    ];
    // Pearson moment Σwᵢrᵢ²/(n−p) (`residuals(f, type="pearson")`), NOT
    // `sigma(f)^2` (pwrss/n on the link scale) — the two are different
    // quantities (see `fit_glmm_sparse`'s `dispersion` arm doc) and only
    // the Pearson form matches `glmm`'s `Fit::dispersion` field.
    const REF_DISPERSION: f64 = 0.411217227312831;

    let csv = include_str!("../../validation/data/simulated/sim_sparse_gamma.csv");
    // Columns: y, x1..x4, gp, ge (indices 0..6).
    let mut y = Vec::<f64>::new();
    let mut xc: [Vec<f64>; 4] = [vec![], vec![], vec![], vec![]];
    let (mut gp_raw, mut ge_raw) = (Vec::<String>::new(), Vec::<String>::new());
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        for k in 0..4 {
            xc[k].push(f[1 + k].parse().unwrap());
        }
        gp_raw.push(f[5].to_string());
        ge_raw.push(f[6].to_string());
    }
    let n = y.len();
    let p = 5;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        for k in 0..4 {
            x[i * p + 1 + k] = xc[k][i];
        }
    }
    let weights: Vec<f64> = (0..n).map(|i| 1.0 + 0.2 * ((i % 3) as f64 - 1.0)).collect();
    let model = crate::ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // sized from ids
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![1, 2, 3, 4],
            }],
        }),
    };
    let ids = crate::GroupIds {
        primary: dense_ids(&gp_raw),
        extra: vec![dense_ids(&ge_raw)],
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1, 2, 3, 4],
        weights: Some(weights),
        ..crate::FitOptions::default() // default WaldSe::Hessian
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(f.converged(), "weighted sparse gamma GLMM must converge");
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs().max(1e-6) < 4e-2,
            "β[{j}] glmm={} lme4={}",
            f.beta[j],
            REF_BETA[j]
        );
        assert!(
            (f.se[j] - REF_SE_HESSIAN[j]).abs() / REF_SE_HESSIAN[j].max(1e-6) < 3e-2,
            "se[{j}] glmm={} lme4={}",
            f.se[j],
            REF_SE_HESSIAN[j]
        );
    }
    assert!(
        (f.dispersion - REF_DISPERSION).abs() / REF_DISPERSION < 3e-2,
        "φ̂ glmm={} lme4={REF_DISPERSION}",
        f.dispersion
    );
}

/// Weighted sparse NB has NO lme4 golden (Task 7): the over-count
/// `sim_sparse_nb` design (7 crossed extras, several small clusters) is
/// already fragile for `glmer.nb`'s marginal-θ profile unweighted (see
/// `fit_sparse_nb_glmm_matches_lme4`'s doc — interim-refit convergence
/// warnings during golden generation). Verified interactively at THREE
/// weight magnitudes and none give a trustworthy weighted oracle: integer
/// `wᵢ = 1 + (i mod 3)` (1/2/3) collapses `isSingular` (several variance
/// components hit exactly 0); `wᵢ = 1 + 0.2·((i mod 3) − 1)` (0.8/1.0/1.2)
/// converges (`isSingular` false) but θ̂ jumps 17% off the unweighted
/// value; `wᵢ = 1 + 0.05·((i mod 3) − 1)` (0.95/1.0/1.05, i.e. a mere ±5%
/// perturbation) STILL prints "Model failed to converge" and moves θ̂ 4%.
/// glmm's θ golden-section search, by contrast, stays within <1% of its
/// unweighted value under the SAME ±5%/±20% perturbations — evidence
/// glmm's path is the more numerically stable one here, not that it is
/// ignoring the weights. Rather than pick a weight scheme until lme4
/// happens to land somewhere assertable (`p`-hacking the oracle — the
/// oracle is sacred, so it is not tuned to pass), this design is
/// covered instead by the mathematically exact replication-equivalence
/// test below, which needs no external oracle.
///
/// Integer prior weights = row replication (NB): `w = 2` on `n` unique
/// rows fits identically to the same rows each duplicated once — Σwᵢ·devᵢ
/// over unique rows equals Σdevᵢ over duplicated rows, so the two
/// marginal-θ profiles (`fit_glmm_nb_sparse`'s golden-section search over
/// `−½D(θ) + nb_profile_loglik(y, y, θ, weights)`) share an argmax. Full
/// β/SE/θ equality (NB's dispersion IS θ̂ itself, driven by the SAME
/// weighted profile on both sides — unlike Gamma's Pearson φ̂, nothing
/// here depends on the raw row count). Tolerances mirror the
/// dense-vs-sparse cross-check (sparse.rs:5735-5752): β 2e-3 rel, SE 2e-2
/// rel, θ 2e-2 rel.
#[test]
fn sparse_weighted_nb_matches_replicated() {
    let family = Family::NegativeBinomial {
        link: crate::NegBinomialLink::Log,
    };
    let ((xw, yw, w, nw, idsw), (xd, yd, nd, idsd), p, model) =
        build_sparse_weighted_replication_case(family, 613);
    let opts_w = crate::FitOptions {
        target_indices: vec![0, 1],
        weights: Some(w),
        ..crate::FitOptions::default()
    };
    let opts_d = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };
    let fw = crate::fit_cold(&xw, &yw, nw, p, &model, &idsw, &opts_w);
    let fd = crate::fit_cold(&xd, &yd, nd, p, &model, &idsd, &opts_d);
    assert!(fw.converged() && fd.converged(), "both fits must converge");
    for j in 0..p {
        assert!(
            (fw.beta[j] - fd.beta[j]).abs() < 2e-3 * (1.0 + fd.beta[j].abs()),
            "β[{j}] weighted={} replicated={}",
            fw.beta[j],
            fd.beta[j]
        );
        assert!(
            (fw.se[j] - fd.se[j]).abs() < 2e-2 * (1.0 + fd.se[j].abs()),
            "se[{j}] weighted={} replicated={}",
            fw.se[j],
            fd.se[j]
        );
    }
    assert!(
        (fw.dispersion - fd.dispersion).abs() < 2e-2 * (1.0 + fd.dispersion.abs()),
        "θ̂: weighted={} replicated={}",
        fw.dispersion,
        fd.dispersion
    );
}

/// Rung-18 shape: binomial logit, primary (1 + x | g1) + crossed extra
/// (1 + x | g2), prior weights = size — the first non-Gaussian design with
/// a slope-carrying extra grouping. IN-envelope (q_g = 2 ≤ MAX_EXTRA_Q):
/// reaches Sparse purely through classify_design's slope-extras clause.
///
/// Values recorded from glmm. They are validated by
/// `sim_binomial_slope_crossed`, whose cross-engine cell checks the same fit
/// against frozen glmer at tolPwrss = 1e-13.
///
/// Relative-tolerance, not bit-equal. These values reproduce BIT-EXACTLY on the
/// anchor machine (see `fit::common_tests::assert_pinned`, "which machine the
/// pins are frozen on"); `BAND` is margin for aarch64-apple-darwin, which drifts
/// 7.68e-6 (`beta[0]`) from architecture-dependent SIMD/FMA contraction on this
/// kernel's long reductions. ~13x that: loose enough to absorb cross-arch
/// reassociation, tight enough that a real change in the fit still trips it.
#[test]
fn fit_sparse_binomial_slope_crossed_is_pinned() {
    const BAND: f64 = 1e-4;
    const REF_BETA: [f64; 2] = [0.08290096060631727, 0.6343172639914979];
    const REF_SE: [f64; 2] = [0.2199779193352823, 0.21697243593898305];
    // Two q=2 blocks [g1, g2], each the column-major lower-triangle vech of D̂
    // on the link scale (binomial: no σ̂ scaling).
    const REF_VC_G1: [f64; 3] = [0.3535134591196692, -0.03346456385338107, 0.299602458176971];
    const REF_VC_G2: [f64; 3] = [0.27688146870530117, 0.22408711356841296, 0.2993874546311821];

    let csv = include_str!("../../validation/data/simulated/sim_binomial_slope_crossed.csv");
    // Columns: incidence, size, x, g1, g2 (indices 0..4). Aggregated
    // binomial: y = incidence/size (proportion), prior weights = size —
    // mirrors validation/engines/glmm.rs's weighted rung-18 lowering.
    let mut y = Vec::<f64>::new();
    let mut size_col = Vec::<f64>::new();
    let mut xcol = Vec::<f64>::new();
    let (mut g1_raw, mut g2_raw) = (Vec::<String>::new(), Vec::<String>::new());
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let incidence: f64 = f[0].parse().unwrap();
        let size: f64 = f[1].parse().unwrap();
        y.push(incidence / size);
        size_col.push(size);
        xcol.push(f[2].parse().unwrap());
        g1_raw.push(f[3].to_string());
        g2_raw.push(f[4].to_string());
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = xcol[i];
    }
    let model = crate::ModelSpec {
        family: Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // sized from ids
            slopes: vec![1],                                 // (1 + x | g1)
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![1], // (1 + x | g2)
            }],
        }),
    };
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse
    ));
    let ids = crate::GroupIds {
        primary: dense_ids(&g1_raw),
        extra: vec![dense_ids(&g2_raw)],
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        weights: Some(size_col.clone()),
        ..crate::FitOptions::default() // default WaldSe::Hessian
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(
        f.converged(),
        "sparse binomial slope-crossed GLMM must converge"
    );
    assert_pinned(&f.beta, &REF_BETA, BAND, "beta");
    assert_pinned(&f.se, &REF_SE, BAND, "se");
    assert_eq!(f.varcorr.len(), 2, "two q=2 varcomp blocks (g1 + g2)");
    assert_pinned(&f.varcorr[0], &REF_VC_G1, BAND, "g1 varcorr");
    assert_pinned(&f.varcorr[1], &REF_VC_G2, BAND, "g2 varcorr");
}

// ── Sparse non-Gaussian GLMM cross-checks ────────────────────

/// Deterministic in-envelope GLMM design shared by the both-paths
/// cross-checks: 4 primary clusters + one crossed extra (3 levels), p = 2
/// (intercept + covariate), family-appropriate y generated from a linear
/// predictor with genuine per-level RE effects. In-envelope on every axis,
/// so `fit_cold` routes it to the dense NoZ GLMM kernel — the oracle.
fn build_glmm_case(
    family: Family,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, usize, usize, ModelSpec, crate::GroupIds) {
    let n = 96;
    let p = 2;
    let mut st = seed;
    let n_primary = 4usize;
    let n_extra_levels = 3usize;
    let u_c: Vec<f64> = (0..n_primary)
        .map(|_| 0.6 * super::test_lcg(&mut st))
        .collect();
    let v_e: Vec<f64> = (0..n_extra_levels)
        .map(|_| 0.4 * super::test_lcg(&mut st))
        .collect();
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut eid = vec![0u32; n];
    for i in 0..n {
        let cov = super::test_lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = cov;
        pid[i] = (i % n_primary) as u32;
        eid[i] = (i % n_extra_levels) as u32;
        let eta = 0.4 + 0.6 * cov + u_c[pid[i] as usize] + v_e[eid[i] as usize];
        y[i] = match family {
            Family::Binomial { .. } => {
                let pr = 1.0 / (1.0 + (-eta).exp());
                let uni = 0.5 * (super::test_lcg(&mut st) + 1.0); // (0, 1)
                if uni < pr {
                    1.0
                } else {
                    0.0
                }
            }
            Family::Poisson { .. } | Family::NegativeBinomial { .. } => {
                // Count-like data around exp(η) with one-sided jitter — the test
                // compares two fitters on the SAME data, so exact Poisson/NB
                // sampling is unnecessary.
                let jit = 1.0 + 0.4 * super::test_lcg(&mut st);
                (eta.exp() * jit).round().max(0.0)
            }
            Family::Gamma { .. } | Family::InverseGaussian { .. } => {
                let jit = 1.0 + 0.3 * super::test_lcg(&mut st);
                (eta.exp() * jit).max(0.05)
            }
            Family::Gaussian => unreachable!("non-Gaussian cases only"),
        };
    }
    let model = ModelSpec {
        family,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_primary as u32,
            },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed {
                    n_clusters: n_extra_levels as u32,
                },
                slopes: vec![],
            }],
        }),
    };
    let ids = crate::GroupIds {
        primary: pid,
        extra: vec![eid],
    };
    (x, y, n, p, model, ids)
}

/// The sparse parallel FD-Hessian grid must reproduce the serial one BITWISE.
/// Drives a full `fit_glmm_sparse` (WaldSe::Hessian) twice — `parallel_inner`
/// off then on — on a crossed binomial design routed through the sparse path,
/// and asserts the returned marginal deviance, `se`, and `stddev_se` are
/// bit-identical. `parallel_inner` gates ONLY `sparse_fd_hessian_cov` here (the
/// sparse fit has no AGQ), so the BOBYQA optimum is shared and any difference
/// isolates to the parallel grid. Every eval cold-seeds û = 0, so per-thread
/// worker workspaces (`clone_worker`) are exact — a mismatch is a field-liveness
/// bug, not noise.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
#[test]
fn sparse_fd_hessian_parallel_bit_identical_to_serial() {
    let (xflat, y, n, p, model, ids) = build_glmm_case(
        Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        202,
    );
    let (sized, ids, _perm) = crate::fit::spec_sized_from_ids_pub(&model, &ids);
    let run = |parallel_inner: bool| {
        let opts = crate::FitOptions {
            target_indices: vec![0, 1],
            wald_se: crate::WaldSe::Hessian,
            parallel_inner,
            ..crate::FitOptions::default()
        };
        super::fit_glmm_sparse(
            &xflat,
            &y,
            n,
            p,
            &sized,
            &ids.primary,
            &ids.extra,
            f64::NAN,
            None,
            &opts,
        )
    };
    let (fit_s, dev_s) = run(false);
    let (fit_p, dev_p) = run(true);
    assert!(
        fit_s.converged() && fit_p.converged(),
        "both fits must converge"
    );
    assert_eq!(
        dev_s.to_bits(),
        dev_p.to_bits(),
        "marginal deviance not bit-identical: {dev_s} vs {dev_p}"
    );
    assert_eq!(fit_s.se.len(), fit_p.se.len());
    for (j, (&a, &b)) in fit_s.se.iter().zip(fit_p.se.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "se[{j}] not bit-identical: {a} vs {b}"
        );
    }
    assert_eq!(fit_s.stddev_se.len(), fit_p.stddev_se.len());
    for (k, (&a, &b)) in fit_s
        .stddev_se
        .iter()
        .zip(fit_p.stddev_se.iter())
        .enumerate()
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "stddev_se[{k}] not bit-identical: {a} vs {b}"
        );
    }
}

/// Deviance-value cross-check (the internal correctness anchor):
/// `sparse_glmm_deviance` equals the dense `glmm_laplace_deviance`
/// at the same (θ, β) on in-envelope designs both can evaluate. The two
/// PIRLS drivers share the discipline but not the arithmetic order (and the
/// dense logit path is fused-SIMD where the sparse one takes the general
/// family branch), so the bound is relative, not bitwise. A disagreement is
/// a bug in exactly one path.
#[test]
fn sparse_glmm_deviance_matches_dense() {
    use faer::Mat;
    for (family, seed) in [
        (
            Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            101u64,
        ),
        (
            Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            103,
        ),
        (
            Family::Gamma {
                link: crate::GammaLink::Log,
            },
            107,
        ),
        (
            Family::Gamma {
                link: crate::GammaLink::Inverse,
            },
            113,
        ),
        (
            Family::NegativeBinomial {
                link: crate::NegBinomialLink::Log,
            },
            109,
        ),
    ] {
        let (xflat, y, n, p, model, ids) = build_glmm_case(family, seed);
        let x = Mat::<f64>::from_fn(n, p, |i, j| xflat[i * p + j]);

        // This is a fixed-(θ,β) deviance cross-check, not a fit, so NB's θ
        // (normally profiled by an outer search) is just a probe constant
        // fed identically to both sides.
        let nb_theta = if matches!(family, Family::NegativeBinomial { .. }) {
            5.0
        } else {
            f64::NAN
        };

        // Dense workspace, mirroring the fit.rs adapter's construction.
        let mut dws = crate::glmm::GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
        crate::glmm::build_z(&mut dws, x.as_ref(), &ids.primary, &ids.extra, n);
        dws.structured_schur = if dws.groupings.structured_extras_eligible() {
            crate::glmm::StructuredSchur::new(&dws.groupings, &ids.primary, &ids.extra, n)
        } else {
            None
        };
        dws.nb_theta = nb_theta;

        let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &[]);
        let mut sws = super::SparseGlmmWorkspace::new(&g, &ids.primary, &ids.extra, n, p);

        // (θ, β) probe points: [θ_primary, θ_extra, β0, β1]. The inverse link
        // gets its own β probes with β₀ well inside the η > 0 domain (μ = 1/η):
        // the shared probes' β₀ leave η_fixed ≤ 0 rows, where PIRLS now
        // correctly refuses a boundary answer (`family::eta_infeasible`) and
        // both paths return ∞ — true agreement, but a vacuous probe.
        let probes: [[f64; 4]; 3] = if matches!(
            family,
            Family::Gamma {
                link: crate::GammaLink::Inverse,
            }
        ) {
            [
                [0.5, 0.7, 1.5, 0.3],
                [1.0, 0.2, 1.3, 0.2],
                [0.15, 1.1, 1.8, -0.3],
            ]
        } else {
            [
                [0.5, 0.7, 0.3, 0.5],
                [1.0, 0.2, -0.2, 0.8],
                [0.15, 1.1, 0.6, -0.4],
            ]
        };
        for params in probes {
            let dense = crate::glmm::glmm_laplace_deviance(
                &params,
                &mut dws,
                x.as_ref(),
                &y,
                &ids.primary,
                &ids.extra,
                n,
            );
            let sparse = super::sparse_glmm_deviance(
                family,
                nb_theta,
                &params,
                &mut sws,
                x.as_ref(),
                &y,
                n,
                false,
            );
            assert!(
                (dense - sparse).abs() < 1e-6 * (1.0 + dense.abs()),
                "{family:?} params={params:?}: dense {dense} vs sparse {sparse}"
            );
        }
    }
}

/// Fit-level both-paths cross-check (the acceptance criterion for the sparse
/// non-Gaussian path): force the
/// sparse non-Gaussian solver on in-envelope designs and diff β/SE/τ²
/// against the dense NoZ GLMM fit reached through `fit_cold`. The two sides
/// are independent BOBYQA minimizations of (numerically) the same Laplace
/// deviance — dense two-stage vs sparse single joint stage — so the bound
/// is optimizer-scatter-sized, not machine precision (the deviance-level
/// test above is the tight anchor). Covers both `WaldSe` arms.
#[test]
fn sparse_glmm_fit_matches_dense_in_envelope() {
    for (family, seed) in [
        (
            Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            201u64,
        ),
        (
            Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            203,
        ),
        (
            Family::Gamma {
                link: crate::GammaLink::Log,
            },
            207,
        ),
        (
            Family::NegativeBinomial {
                link: crate::NegBinomialLink::Log,
            },
            211,
        ),
    ] {
        let (xflat, y, n, p, model, ids) = build_glmm_case(family, seed);
        for wald_se in [crate::WaldSe::Hessian, crate::WaldSe::Rx] {
            let opts = crate::FitOptions {
                target_indices: vec![0, 1],
                wald_se,
                ..crate::FitOptions::default()
            };
            let dense = crate::fit_cold(&xflat, &y, n, p, &model, &ids, &opts);
            let (sized, ids, _perm) = crate::fit::spec_sized_from_ids_pub(&model, &ids);
            let sp = if matches!(family, Family::NegativeBinomial { .. }) {
                super::fit_glmm_nb_sparse(
                    &xflat,
                    &y,
                    n,
                    p,
                    &sized,
                    &ids.primary,
                    &ids.extra,
                    None,
                    &opts,
                )
            } else {
                super::fit_glmm_sparse(
                    &xflat,
                    &y,
                    n,
                    p,
                    &sized,
                    &ids.primary,
                    &ids.extra,
                    f64::NAN,
                    None,
                    &opts,
                )
                .0
            };
            let tag = format!("{family:?}/{wald_se:?}");
            assert!(
                dense.converged() && sp.converged(),
                "{tag}: both paths must converge"
            );
            for j in 0..p {
                assert!(
                    (sp.beta[j] - dense.beta[j]).abs() < 2e-3 * (1.0 + dense.beta[j].abs()),
                    "{tag} β[{j}]: sparse={} dense={}",
                    sp.beta[j],
                    dense.beta[j]
                );
                assert!(
                    (sp.se[j] - dense.se[j]).abs() < 2e-2 * (1.0 + dense.se[j].abs()),
                    "{tag} se[{j}]: sparse={} dense={}",
                    sp.se[j],
                    dense.se[j]
                );
            }
            assert_eq!(sp.tau2.len(), dense.tau2.len(), "{tag}: tau2 length");
            for (a, b) in sp.tau2.iter().zip(dense.tau2.iter()) {
                assert!(
                    (a - b).abs() < 2e-2 * (1.0 + b.abs()),
                    "{tag} tau2: sparse={a} dense={b}"
                );
            }
            assert!(
                (sp.dispersion - dense.dispersion).abs() < 2e-2 * (1.0 + dense.dispersion.abs()),
                "{tag} dispersion: sparse={} dense={}",
                sp.dispersion,
                dense.dispersion
            );
        }
    }
}

/// Regression test for the Gamma-inverse boundary-convergence defect: forced
/// through the sparse solver, `y ~ 1 + x + grp + (1|cluster)` on `sim_gamma`
/// with the INVERSE link used to report `converged = true` under `WaldSe::Rx`
/// at an optimum ~937 deviance units above the dense one (β₀ off by 11%), and
/// `deviance = inf` under `WaldSe::Hessian` — PIRLS accepted iterates that
/// `clamp_eta` had projected onto the η > 0 domain boundary, and the projected
/// row's μ² ≈ 1e20 working weight kept the WLS solve pinned there (see
/// `family::eta_infeasible` for the fix: infeasible trial ⇒ step-halve). The
/// log-link twin of this model was green throughout, which is what pins the
/// defect to the link rather than the sparse forcing.
///
/// Asserts the sparse fit against the dense `fit_cold` on identical inputs
/// (the envelope-test bounds — two independent BOBYQA minimizations) and
/// against frozen `glmer(family=Gamma("inverse"))`
/// (`validation/goldens/sim_gamma_inv_glmm.json`), both `WaldSe` arms. The loglik
/// pin is the branch check: the spurious boundary optimum misses it by ~469.
#[test]
fn sparse_glmm_gamma_inverse_fit_matches_dense_and_lme4() {
    const REF_BETA: [f64; 3] = [0.75205795080653, -0.187572954875194, -0.140275024148733];
    const REF_LOGLIK: f64 = -468.38415378098;
    let (x, y, cluster_ids, n_clusters) = crate::fit::common_tests::sim_clustered(include_str!(
        "../../validation/data/simulated/sim_gamma.csv"
    ));
    let (n, p) = (y.len(), 3);
    let model = ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Inverse,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = crate::GroupIds {
        primary: cluster_ids,
        extra: vec![],
    };
    for wald_se in [crate::WaldSe::Hessian, crate::WaldSe::Rx] {
        let opts = crate::FitOptions {
            target_indices: vec![0, 1, 2],
            wald_se,
            ..crate::FitOptions::default()
        };
        let dense = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
        let (sized, ids, _perm) = crate::fit::spec_sized_from_ids_pub(&model, &ids);
        let (sp, _dev) = super::fit_glmm_sparse(
            &x,
            &y,
            n,
            p,
            &sized,
            &ids.primary,
            &ids.extra,
            f64::NAN,
            None,
            &opts,
        );
        let tag = format!("gamma-inverse/{wald_se:?}");
        assert!(
            dense.converged() && sp.converged(),
            "{tag}: both paths must converge"
        );
        #[allow(clippy::needless_range_loop)] // j indexes three parallel slices
        for j in 0..p {
            assert!(
                (sp.beta[j] - dense.beta[j]).abs() < 2e-3 * (1.0 + dense.beta[j].abs()),
                "{tag} β[{j}]: sparse={} dense={}",
                sp.beta[j],
                dense.beta[j]
            );
            assert!(
                (sp.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 2e-3,
                "{tag} β[{j}]: sparse={} lme4={}",
                sp.beta[j],
                REF_BETA[j]
            );
            assert!(
                (sp.se[j] - dense.se[j]).abs() < 2e-2 * (1.0 + dense.se[j].abs()),
                "{tag} se[{j}]: sparse={} dense={}",
                sp.se[j],
                dense.se[j]
            );
        }
        assert!(
            (sp.loglik - REF_LOGLIK).abs() < 1e-2,
            "{tag} loglik {} vs lme4 {REF_LOGLIK}",
            sp.loglik
        );
    }
}

/// Aggregated-binomial / expanded-Bernoulli twin datasets on an over-count
/// design (7 crossed extras ⇒ `Solver::Sparse`): each aggregated row i
/// carries mᵢ ∈ 2..=5 trials with sᵢ successes; its expanded twin holds mᵢ
/// one-trial 0/1 rows with the SAME covariate and level ids, so both
/// describe identical Bernoulli data. Returns
/// `(aggregated (x, y=s/m, weights=m, n, ids), expanded (x, y, n, ids), p,
/// model, sat)` where `sat = 2Σᵢ[sᵢ ln(sᵢ/mᵢ) + (mᵢ−sᵢ) ln((mᵢ−sᵢ)/mᵢ)]`
/// (0·ln0 = 0) is the data-only saturated term by which the aggregated
/// weighted deviance falls below the expanded one — same argmin.
#[allow(clippy::type_complexity)]
fn build_binomial_weighted_pair() -> (
    (Vec<f64>, Vec<f64>, Vec<f64>, usize, crate::GroupIds),
    (Vec<f64>, Vec<f64>, usize, crate::GroupIds),
    usize,
    ModelSpec,
    f64,
) {
    let n_agg = 72;
    let p = 2;
    let mut st = 401u64;
    let n_primary = 6usize;
    let extra_levels = [3usize, 4, 3, 5, 3, 4, 3];
    let u_c: Vec<f64> = (0..n_primary)
        .map(|_| 0.5 * super::test_lcg(&mut st))
        .collect();
    let v_e: Vec<Vec<f64>> = extra_levels
        .iter()
        .map(|&l| (0..l).map(|_| 0.3 * super::test_lcg(&mut st)).collect())
        .collect();
    let pid_a: Vec<u32> = (0..n_agg).map(|i| (i % n_primary) as u32).collect();
    let extra_a: Vec<Vec<u32>> = extra_levels
        .iter()
        .enumerate()
        .map(|(g, &l)| (0..n_agg).map(|i| ((i / (g + 1)) % l) as u32).collect())
        .collect();
    let mut xa = vec![0.0f64; n_agg * p];
    let mut ya = vec![0.0f64; n_agg];
    let mut wa = vec![0.0f64; n_agg];
    let (mut xe, mut ye, mut pid_e) = (Vec::new(), Vec::new(), Vec::new());
    let mut extra_e: Vec<Vec<u32>> = vec![Vec::new(); extra_levels.len()];
    let mut sat = 0.0f64;
    for i in 0..n_agg {
        let cov = super::test_lcg(&mut st);
        xa[i * p] = 1.0;
        xa[i * p + 1] = cov;
        let mut e = 0.3 + 0.5 * cov + u_c[pid_a[i] as usize];
        for (g, ids_g) in extra_a.iter().enumerate() {
            e += v_e[g][ids_g[i] as usize];
        }
        let pr = 1.0 / (1.0 + (-e).exp());
        let m = 2 + (i % 4);
        let mut s = 0usize;
        for _ in 0..m {
            let uni = 0.5 * (super::test_lcg(&mut st) + 1.0);
            let yk = if uni < pr { 1.0 } else { 0.0 };
            s += yk as usize;
            ye.push(yk);
            xe.push(1.0);
            xe.push(cov);
            pid_e.push(pid_a[i]);
            for (g, col) in extra_e.iter_mut().enumerate() {
                col.push(extra_a[g][i]);
            }
        }
        ya[i] = s as f64 / m as f64;
        wa[i] = m as f64;
        let (mf, sf) = (m as f64, s as f64);
        if s > 0 {
            sat += 2.0 * sf * (sf / mf).ln();
        }
        if s < m {
            sat += 2.0 * (mf - sf) * ((mf - sf) / mf).ln();
        }
    }
    let n_exp = ye.len();
    let model = ModelSpec {
        family: Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_primary as u32,
            },
            slopes: vec![],
            extra_groupings: extra_levels
                .iter()
                .map(|&l| Grouping {
                    relation: GroupingRelation::Crossed {
                        n_clusters: l as u32,
                    },
                    slopes: vec![],
                })
                .collect(),
        }),
    };
    let ids_a = crate::GroupIds {
        primary: pid_a,
        extra: extra_a,
    };
    let ids_e = crate::GroupIds {
        primary: pid_e,
        extra: extra_e,
    };
    (
        (xa, ya, wa, n_agg, ids_a),
        (xe, ye, n_exp, ids_e),
        p,
        model,
        sat,
    )
}

/// Non-Gaussian sparse-classified design (7 crossed extras — same RE
/// shape as `build_binomial_weighted_pair`, `> MAX_EXTRA_GROUPINGS` so
/// `classify_design` routes `Sparse` regardless of family) for the
/// weighted/replicated equivalence tests below: integer weight `w = 2` on
/// `n_unique` rows must fit identically to the same `n_unique` rows each
/// duplicated once (weights unset) — Σwᵢ·devᵢ over the unique rows equals
/// Σdevᵢ over the doubled rows, so the two objectives share an argmin.
/// Returns `((x, y, weights=2, n, ids), (x2, y2, n2=2n, ids2))`.
#[allow(clippy::type_complexity)]
fn build_sparse_weighted_replication_case(
    family: Family,
    seed: u64,
) -> (
    (Vec<f64>, Vec<f64>, Vec<f64>, usize, crate::GroupIds),
    (Vec<f64>, Vec<f64>, usize, crate::GroupIds),
    usize,
    ModelSpec,
) {
    let n = 60;
    let p = 2;
    let mut st = seed;
    let n_primary = 6usize;
    let extra_levels = [3usize, 4, 3, 5, 3, 4, 3];
    let u_c: Vec<f64> = (0..n_primary)
        .map(|_| 0.3 * super::test_lcg(&mut st))
        .collect();
    let v_e: Vec<Vec<f64>> = extra_levels
        .iter()
        .map(|&l| (0..l).map(|_| 0.2 * super::test_lcg(&mut st)).collect())
        .collect();
    let pid: Vec<u32> = (0..n).map(|i| (i % n_primary) as u32).collect();
    let extra: Vec<Vec<u32>> = extra_levels
        .iter()
        .enumerate()
        .map(|(g, &l)| (0..n).map(|i| ((i / (g + 1)) % l) as u32).collect())
        .collect();
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let cov = 0.3 * super::test_lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = cov;
        let mut eta = 0.3 + 0.4 * cov + u_c[pid[i] as usize];
        for (g, ids_g) in extra.iter().enumerate() {
            eta += v_e[g][ids_g[i] as usize];
        }
        y[i] = match family {
            Family::Poisson { .. } => {
                let jit = 1.0 + 0.4 * super::test_lcg(&mut st);
                (eta.exp() * jit).round().max(0.0)
            }
            Family::Gamma { .. } => {
                let jit = 1.0 + 0.3 * super::test_lcg(&mut st);
                (eta.exp() * jit).max(0.05)
            }
            Family::NegativeBinomial { .. } => {
                // Count-like data around exp(η), one-sided jitter — same
                // rationale as `build_glmm_case`'s NB arm: the test
                // compares two fitters on the SAME data, so exact NB
                // sampling is unnecessary.
                let jit = 1.0 + 0.4 * super::test_lcg(&mut st);
                (eta.exp() * jit).round().max(0.0)
            }
            _ => unreachable!("Poisson/Gamma/NB cases only"),
        };
    }
    let model = ModelSpec {
        family,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_primary as u32,
            },
            slopes: vec![],
            extra_groupings: extra_levels
                .iter()
                .map(|&l| Grouping {
                    relation: GroupingRelation::Crossed {
                        n_clusters: l as u32,
                    },
                    slopes: vec![],
                })
                .collect(),
        }),
    };
    let ids = crate::GroupIds {
        primary: pid.clone(),
        extra: extra.clone(),
    };
    // Row-doubled twin: literal concatenation of every per-row vector with
    // itself, weights unset (all-1) — same data, same likelihood as the
    // unique rows under w=2.
    let mut x2 = x.clone();
    x2.extend_from_slice(&x);
    let mut y2 = y.clone();
    y2.extend_from_slice(&y);
    let mut pid2 = pid.clone();
    pid2.extend_from_slice(&pid);
    let extra2: Vec<Vec<u32>> = extra
        .iter()
        .map(|col| {
            let mut c2 = col.clone();
            c2.extend_from_slice(col);
            c2
        })
        .collect();
    let ids2 = crate::GroupIds {
        primary: pid2,
        extra: extra2,
    };
    let weights = vec![2.0; n];
    ((x, y, weights, n, ids), (x2, y2, 2 * n, ids2), p, model)
}

/// Integer prior weights = row replication (Poisson): `w = 2` on `n`
/// unique rows fits identically to the same rows each duplicated once —
/// Σwᵢ·devᵢ over unique rows equals Σdevᵢ over duplicated rows, so the
/// two sparse PIRLS objectives share an argmin. Full β/SE/τ² equality
/// (Poisson has no estimated dispersion, so nothing else differs between
/// the two row counts). Tolerances mirror the dense-vs-sparse cross-check
/// (sparse_glmm_matches_dense_glmm, sparse.rs:5735-5752): β 2e-3 rel, SE
/// 2e-2 rel, τ² 2e-2 rel — two independent BOBYQA runs of same-argmin
/// objectives, so the bound is optimizer-scatter-sized.
#[test]
fn sparse_weighted_poisson_matches_replicated() {
    let family = Family::Poisson {
        link: crate::PoissonLink::Log,
    };
    let ((xw, yw, w, nw, idsw), (xd, yd, nd, idsd), p, model) =
        build_sparse_weighted_replication_case(family, 601);
    let opts_w = crate::FitOptions {
        target_indices: vec![0, 1],
        weights: Some(w),
        ..crate::FitOptions::default()
    };
    let opts_d = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };
    let fw = crate::fit_cold(&xw, &yw, nw, p, &model, &idsw, &opts_w);
    let fd = crate::fit_cold(&xd, &yd, nd, p, &model, &idsd, &opts_d);
    assert!(fw.converged() && fd.converged(), "both fits must converge");
    for j in 0..p {
        assert!(
            (fw.beta[j] - fd.beta[j]).abs() < 2e-3 * (1.0 + fd.beta[j].abs()),
            "β[{j}] weighted={} replicated={}",
            fw.beta[j],
            fd.beta[j]
        );
        assert!(
            (fw.se[j] - fd.se[j]).abs() < 2e-2 * (1.0 + fd.se[j].abs()),
            "se[{j}] weighted={} replicated={}",
            fw.se[j],
            fd.se[j]
        );
    }
    assert_eq!(fw.tau2.len(), fd.tau2.len());
    for (a, b) in fw.tau2.iter().zip(fd.tau2.iter()) {
        assert!(
            (a - b).abs() < 2e-2 * (1.0 + b.abs()),
            "τ²: weighted={a} replicated={b}"
        );
    }
}

/// Gamma twin of `sparse_weighted_poisson_matches_replicated`. Asserts
/// β/τ² only, NOT SE/dispersion: Gamma's Pearson φ̂ divides by raw `n−p`
/// df (mirroring `glm(weights=)`/`glmer(weights=)`), and `n` differs
/// between the weighted (n rows) and replicated (2n rows) encodings, so
/// φ̂ — and every SE that scales with it — is NOT expected to match
/// between the two, even though the likelihood/argmin is identical.
#[test]
fn sparse_weighted_gamma_matches_replicated() {
    let family = Family::Gamma {
        link: crate::GammaLink::Log,
    };
    let ((xw, yw, w, nw, idsw), (xd, yd, nd, idsd), p, model) =
        build_sparse_weighted_replication_case(family, 607);
    let opts_w = crate::FitOptions {
        target_indices: vec![0, 1],
        weights: Some(w),
        ..crate::FitOptions::default()
    };
    let opts_d = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };
    let fw = crate::fit_cold(&xw, &yw, nw, p, &model, &idsw, &opts_w);
    let fd = crate::fit_cold(&xd, &yd, nd, p, &model, &idsd, &opts_d);
    assert!(fw.converged() && fd.converged(), "both fits must converge");
    for j in 0..p {
        assert!(
            (fw.beta[j] - fd.beta[j]).abs() < 2e-3 * (1.0 + fd.beta[j].abs()),
            "β[{j}] weighted={} replicated={}",
            fw.beta[j],
            fd.beta[j]
        );
    }
    assert_eq!(fw.tau2.len(), fd.tau2.len());
    for (a, b) in fw.tau2.iter().zip(fd.tau2.iter()) {
        assert!(
            (a - b).abs() < 2e-2 * (1.0 + b.abs()),
            "τ²: weighted={a} replicated={b}"
        );
    }
}

/// Prior-weight deviance anchor: at the SAME (θ, β) probe points, the
/// aggregated fit with `prior_w = m` and the expanded Bernoulli fit produce
/// deviances differing by exactly the data-only saturated constant (the
/// penalty and log|A| terms coincide — aggregated W̃ᵢ = mᵢ·W̃ scatters the
/// identical M'W̃M). Tight bound: same arithmetic up to summation order.
#[test]
fn sparse_weighted_binomial_deviance_matches_expanded() {
    use faer::Mat;
    let ((xa, ya, wa, n_a, ids_a), (xe, ye, n_e, ids_e), p, model, sat) =
        build_binomial_weighted_pair();
    let family = Family::Binomial {
        link: crate::BinomialLink::Logit,
    };
    let g_a = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n_a, &[], &[]);
    let mut ws_a = super::SparseGlmmWorkspace::new(&g_a, &ids_a.primary, &ids_a.extra, n_a, p);
    ws_a.prior_w[..n_a].copy_from_slice(&wa);
    let g_e = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n_e, &[], &[]);
    let mut ws_e = super::SparseGlmmWorkspace::new(&g_e, &ids_e.primary, &ids_e.extra, n_e, p);
    let xam = Mat::<f64>::from_fn(n_a, p, |i, j| xa[i * p + j]);
    let xem = Mat::<f64>::from_fn(n_e, p, |i, j| xe[i * p + j]);
    // (θ×8, β×2) probe points spanning small/moderate RE scales.
    for params in [
        [0.4f64, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.5],
        [0.8, 0.2, 0.5, 0.3, 0.6, 0.2, 0.4, 0.7, -0.2, 0.8],
    ] {
        let da = super::sparse_glmm_deviance(
            family,
            f64::NAN,
            &params,
            &mut ws_a,
            xam.as_ref(),
            &ya,
            n_a,
            false,
        );
        let de = super::sparse_glmm_deviance(
            family,
            f64::NAN,
            &params,
            &mut ws_e,
            xem.as_ref(),
            &ye,
            n_e,
            false,
        );
        assert!(
            da.is_finite() && de.is_finite(),
            "params={params:?}: both finite"
        );
        assert!(
            ((da - de) - sat).abs() < 1e-8 * (1.0 + de.abs()),
            "params={params:?}: agg {da} vs exp {de}, sat {sat}"
        );
    }
}

/// Prior-weight fit-level check through the stable surface: `fit_cold` on
/// the aggregated rows with `FitOptions::weights = Some(m)` matches the
/// expanded Bernoulli fit on β/SE/τ² for both `WaldSe` arms. Two
/// independent BOBYQA runs of same-argmin objectives, so the bound is
/// optimizer-scatter-sized (the deviance test above is the tight anchor).
#[test]
fn sparse_weighted_binomial_fit_matches_expanded() {
    let ((xa, ya, wa, n_a, ids_a), (xe, ye, n_e, ids_e), p, model, _sat) =
        build_binomial_weighted_pair();
    for wald_se in [crate::WaldSe::Hessian, crate::WaldSe::Rx] {
        let opts_e = crate::FitOptions {
            target_indices: vec![0, 1],
            wald_se,
            ..crate::FitOptions::default()
        };
        let fe = crate::fit_cold(&xe, &ye, n_e, p, &model, &ids_e, &opts_e);
        let opts_a = crate::FitOptions {
            target_indices: vec![0, 1],
            wald_se,
            weights: Some(wa.clone()),
            ..crate::FitOptions::default()
        };
        let fa = crate::fit_cold(&xa, &ya, n_a, p, &model, &ids_a, &opts_a);
        let tag = format!("{wald_se:?}");
        assert!(
            fe.converged() && fa.converged(),
            "{tag}: both fits must converge"
        );
        for j in 0..p {
            assert!(
                (fa.beta[j] - fe.beta[j]).abs() < 2e-3 * (1.0 + fe.beta[j].abs()),
                "{tag} β[{j}]: agg={} exp={}",
                fa.beta[j],
                fe.beta[j]
            );
            assert!(
                (fa.se[j] - fe.se[j]).abs() < 2e-2 * (1.0 + fe.se[j].abs()),
                "{tag} se[{j}]: agg={} exp={}",
                fa.se[j],
                fe.se[j]
            );
        }
        assert_eq!(fa.tau2.len(), fe.tau2.len(), "{tag}: tau2 length");
        for (a, b) in fa.tau2.iter().zip(fe.tau2.iter()) {
            assert!(
                (a - b).abs() < 2e-2 * (1.0 + b.abs()),
                "{tag} tau2: agg={a} exp={b}"
            );
        }
    }
}

/// Over-envelope non-Gaussian smoke, binomial half: a genuinely over-cap
/// design (7 crossed extras, `y ~ 1 + x + (1|g1) + (1|c1) + … + (1|c7)`) with
/// real signal CONVERGES through the routed `fit_cold` path (an upgrade over
/// the prior anti-panic floor, which merely returned non-converged NaN).
/// External truth is the validation `sim_sparse_binomial` rung; this is the
/// in-crate convergence gate, tightened by a deviance self-consistency
/// check: every RE here is scalar (q_p = 1, all extras scalar) and the
/// family has σ² ≡ 1, so `θ_i = √tau2[i]` exactly reconstructs the
/// converged Cholesky factor — feeding `[θ.., β..]` back through
/// `sparse_glmm_deviance` at the SAME (θ, β) must reproduce `f.deviance`
/// (mirrors `sparse_glmm_deviance_matches_dense`'s workspace-build pattern).
#[test]
fn sparse_glmm_over_envelope_converges_binomial() {
    use faer::Mat;
    let n = 210;
    let p = 2;
    let mut st = 301u64;
    let n_primary = 6usize;
    let u_c: Vec<f64> = (0..n_primary)
        .map(|_| 0.5 * super::test_lcg(&mut st))
        .collect();
    let extra_levels = [3usize, 4, 3, 5, 3, 4, 3];
    let v_e: Vec<Vec<f64>> = extra_levels
        .iter()
        .map(|&l| (0..l).map(|_| 0.3 * super::test_lcg(&mut st)).collect())
        .collect();
    let mut x = vec![0.0f64; n * p];
    let mut eta = vec![0.0f64; n];
    let pid: Vec<u32> = (0..n).map(|i| (i % n_primary) as u32).collect();
    let extra: Vec<Vec<u32>> = extra_levels
        .iter()
        .enumerate()
        .map(|(g, &l)| (0..n).map(|i| ((i / (g + 1)) % l) as u32).collect())
        .collect();
    for i in 0..n {
        let cov = super::test_lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = cov;
        let mut e = 0.3 + 0.5 * cov + u_c[pid[i] as usize];
        for (g, ids_g) in extra.iter().enumerate() {
            e += v_e[g][ids_g[i] as usize];
        }
        eta[i] = e;
    }
    let model = ModelSpec {
        family: Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_primary as u32,
            },
            slopes: vec![],
            extra_groupings: extra_levels
                .iter()
                .map(|&l| Grouping {
                    relation: GroupingRelation::Crossed {
                        n_clusters: l as u32,
                    },
                    slopes: vec![],
                })
                .collect(),
        }),
    };
    let ids = crate::GroupIds {
        primary: pid,
        extra,
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };
    let yb: Vec<f64> = eta
        .iter()
        .map(|&e| {
            let pr = 1.0 / (1.0 + (-e).exp());
            let uni = 0.5 * (super::test_lcg(&mut st) + 1.0);
            if uni < pr {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse
    ));
    let f = crate::fit_cold(&x, &yb, n, p, &model, &ids, &opts);
    assert!(f.converged(), "over-count binomial converges");
    assert!(f.beta.iter().all(|b| b.is_finite()) && f.se.iter().all(|s| s.is_finite()));

    let n_theta = f.tau2.len();
    let mut params = Vec::with_capacity(n_theta + p);
    params.extend(f.tau2.iter().map(|t| t.sqrt()));
    params.extend(f.beta.iter().copied());
    let x_mat = Mat::<f64>::from_fn(n, p, |i, j| x[i * p + j]);
    let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &[]);
    let mut sws = super::SparseGlmmWorkspace::new(&g, &ids.primary, &ids.extra, n, p);
    let recomputed = super::sparse_glmm_deviance(
        model.family,
        f64::NAN,
        &params,
        &mut sws,
        x_mat.as_ref(),
        &yb,
        n,
        false,
    );
    assert!(
        (recomputed - f.deviance).abs() < 1e-8 * (1.0 + f.deviance.abs()),
        "deviance self-consistency: recomputed {recomputed} vs fit {}",
        f.deviance
    );
}

/// Over-envelope non-Gaussian smoke, Poisson half — same over-cap shape and
/// deviance self-consistency check as the binomial twin above, on the
/// validation `sim_sparse_poisson` rung's design instead.
#[test]
fn sparse_glmm_over_envelope_converges_poisson() {
    use faer::Mat;
    let n = 210;
    let p = 2;
    let mut st = 401u64;
    let n_primary = 6usize;
    let u_c: Vec<f64> = (0..n_primary)
        .map(|_| 0.5 * super::test_lcg(&mut st))
        .collect();
    let extra_levels = [3usize, 4, 3, 5, 3, 4, 3];
    let v_e: Vec<Vec<f64>> = extra_levels
        .iter()
        .map(|&l| (0..l).map(|_| 0.3 * super::test_lcg(&mut st)).collect())
        .collect();
    let mut x = vec![0.0f64; n * p];
    let mut eta = vec![0.0f64; n];
    let pid: Vec<u32> = (0..n).map(|i| (i % n_primary) as u32).collect();
    let extra: Vec<Vec<u32>> = extra_levels
        .iter()
        .enumerate()
        .map(|(g, &l)| (0..n).map(|i| ((i / (g + 1)) % l) as u32).collect())
        .collect();
    for i in 0..n {
        let cov = super::test_lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = cov;
        let mut e = 0.3 + 0.5 * cov + u_c[pid[i] as usize];
        for (g, ids_g) in extra.iter().enumerate() {
            e += v_e[g][ids_g[i] as usize];
        }
        eta[i] = e;
    }
    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_primary as u32,
            },
            slopes: vec![],
            extra_groupings: extra_levels
                .iter()
                .map(|&l| Grouping {
                    relation: GroupingRelation::Crossed {
                        n_clusters: l as u32,
                    },
                    slopes: vec![],
                })
                .collect(),
        }),
    };
    let ids = crate::GroupIds {
        primary: pid,
        extra,
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };
    let yp: Vec<f64> = eta
        .iter()
        .map(|&e| {
            let jit = 1.0 + 0.4 * super::test_lcg(&mut st);
            (e.exp() * jit).round().max(0.0)
        })
        .collect();
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse
    ));
    let f = crate::fit_cold(&x, &yp, n, p, &model, &ids, &opts);
    assert!(f.converged(), "over-count poisson converges");
    assert!(f.beta.iter().all(|b| b.is_finite()) && f.se.iter().all(|s| s.is_finite()));

    // fitted/ranef consistency through the log link — pins the sparse-path
    // ranef layout (primary block, then the 7 scalar crossed extras in
    // declaration order) and the Λ scale: η̂ = Xβ̂ + Zb̂ must reproduce μ̂.
    let mut expected_levels = vec![n_primary];
    expected_levels.extend(extra_levels.iter());
    assert_eq!(f.ranef_levels, expected_levels);
    assert_eq!(f.ranef.len(), expected_levels.iter().sum::<usize>());
    assert_eq!(f.fitted.len(), n);
    for i in 0..n {
        let mut eta: f64 = (0..p).map(|j| x[i * p + j] * f.beta[j]).sum();
        eta += f.ranef[ids.primary[i] as usize];
        let mut off = n_primary;
        for (g, ids_g) in ids.extra.iter().enumerate() {
            eta += f.ranef[off + ids_g[i] as usize];
            off += extra_levels[g];
        }
        let mu = eta.exp();
        assert!(
            (f.fitted[i] - mu).abs() < 1e-6 * mu.max(1.0),
            "fitted[{i}] = {} vs exp(Xβ̂+Zb̂) = {mu}",
            f.fitted[i]
        );
    }

    let n_theta = f.tau2.len();
    let mut params = Vec::with_capacity(n_theta + p);
    params.extend(f.tau2.iter().map(|t| t.sqrt()));
    params.extend(f.beta.iter().copied());
    let x_mat = Mat::<f64>::from_fn(n, p, |i, j| x[i * p + j]);
    let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &[]);
    let mut sws = super::SparseGlmmWorkspace::new(&g, &ids.primary, &ids.extra, n, p);
    let recomputed = super::sparse_glmm_deviance(
        model.family,
        f64::NAN,
        &params,
        &mut sws,
        x_mat.as_ref(),
        &yp,
        n,
        false,
    );
    assert!(
        (recomputed - f.deviance).abs() < 1e-8 * (1.0 + f.deviance.abs()),
        "deviance self-consistency: recomputed {recomputed} vs fit {}",
        f.deviance
    );
}

/// Over-envelope non-Gaussian smoke, gamma half: one over-width slope-block
/// extra (`(1 + x1..x4 | ge)`, `q_g = 5 > MAX_EXTRA_Q`) CONVERGES with a
/// finite, real-signal fit. Unlike the binomial/Poisson twins above, no
/// deviance self-consistency check here: with a q=5 covariance block AND a
/// free Gamma dispersion, `f.tau2` alone can't reconstruct the converged Λ
/// (off-diagonal vech entries and the dispersion aren't recoverable from
/// `tau2`'s per-diagonal values), so external validation rungs (`sim_gamma_glmm`
/// shape) are what validate this case's numbers; this stays a convergence
/// gate.
#[test]
fn sparse_glmm_over_envelope_converges_gamma() {
    // Over-width: y ~ 1 + x1..x4 + (1|gp) + (1 + x1..x4 | ge), gamma/log.
    let n = 240;
    let p = 5;
    let mut st = 507u64;
    let n_gp = 8usize;
    let n_ge = 6usize;
    let q_g = 5usize;
    let u_gp: Vec<f64> = (0..n_gp).map(|_| 0.4 * super::test_lcg(&mut st)).collect();
    let v_ge: Vec<f64> = (0..n_ge * q_g)
        .map(|_| 0.25 * super::test_lcg(&mut st))
        .collect();
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let gp: Vec<u32> = (0..n).map(|i| (i % n_gp) as u32).collect();
    let ge: Vec<u32> = (0..n).map(|i| ((i / 2) % n_ge) as u32).collect();
    for i in 0..n {
        x[i * p] = 1.0;
        for j in 1..p {
            x[i * p + j] = super::test_lcg(&mut st);
        }
        let l = ge[i] as usize;
        let mut e = 0.5 + u_gp[gp[i] as usize] + v_ge[l * q_g];
        for j in 1..p {
            e += (0.4 + v_ge[l * q_g + j]) * x[i * p + j];
        }
        let jit = 1.0 + 0.3 * super::test_lcg(&mut st);
        y[i] = (e.exp() * jit).max(0.05);
    }
    let model = ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_gp as u32,
            },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed {
                    n_clusters: n_ge as u32,
                },
                slopes: vec![1, 2, 3, 4], // q_g = 5 > MAX_EXTRA_Q ⇒ over-width
            }],
        }),
    };
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse
    ));
    let ids = crate::GroupIds {
        primary: gp,
        extra: vec![ge],
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1, 2, 3, 4],
        ..crate::FitOptions::default()
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(f.converged(), "over-width gamma converges");
    assert!(f.beta.iter().all(|b| b.is_finite()) && f.se.iter().all(|s| s.is_finite()));

    // fitted/ranef consistency through the log link — the q_g=5 slope-block
    // extra pins the sparse ranef Λ_g block application and the level-major
    // [intercept, slope₁..slope₄] within-level order.
    assert_eq!(f.ranef_levels, vec![n_gp, n_ge]);
    assert_eq!(f.ranef.len(), n_gp + n_ge * q_g);
    assert_eq!(f.fitted.len(), n);
    for i in 0..n {
        let mut eta: f64 = (0..p).map(|j| x[i * p + j] * f.beta[j]).sum();
        eta += f.ranef[ids.primary[i] as usize];
        let base = n_gp + ids.extra[0][i] as usize * q_g;
        eta += f.ranef[base]; // extra intercept
        for (c, &col) in [1usize, 2, 3, 4].iter().enumerate() {
            eta += f.ranef[base + 1 + c] * x[i * p + col];
        }
        let mu = eta.exp();
        assert!(
            (f.fitted[i] - mu).abs() < 1e-6 * mu.max(1.0),
            "fitted[{i}] = {} vs exp(Xβ̂+Zb̂) = {mu}",
            f.fitted[i]
        );
    }
}

/// Task 6: weighted sparse Gaussian LMM. `validation/manifest.json` has no
/// sparse-Gaussian rung (only sparse binomial/poisson/gamma/nb), so this
/// fixture is generated directly in R rather than pulled from a committed
/// validation dataset, and pinned against a frozen lme4 golden (not routed
/// through the validation harness). Design: primary `(1|g1)` (20
/// levels) crossed with an extra grouping that carries a random SLOPE
/// `(1+x2|g2)` (15 levels) — `classify_design`'s `slope_extras` clause
/// routes any slope-carrying extra grouping Sparse regardless of size,
/// so this over-envelope classification is a structural property of the
/// design, not a size accident. Weights `w_i = 1 + (i mod 3)` (0-based
/// CSV row order), same convention as the dense Task 5 golden. Generated
/// with (R 4.5.3, lme4 1.1-38):
/// ```r
/// library(lme4)
/// set.seed(2026)
/// n <- 400; n_g1 <- 20; n_g2 <- 15
/// g1 <- rep(0:(n_g1 - 1), each = n / n_g1)
/// g2 <- (seq_len(n) - 1) %% n_g2
/// x1 <- rnorm(n); x2 <- rnorm(n)
/// b1 <- rnorm(n_g1, sd = 1.5)
/// b2_int <- rnorm(n_g2, sd = 1.0); b2_slope <- rnorm(n_g2, sd = 0.8)
/// y <- 2 + 0.7 * x1 + 1.2 * x2 + b1[g1 + 1] + b2_int[g2 + 1] +
///      b2_slope[g2 + 1] * x2 + rnorm(n, sd = 1.0)
/// w <- 1 + (seq_len(n) - 1) %% 3
/// d <- data.frame(y, x1, x2, g1, g2)
/// f <- lmer(y ~ x1 + x2 + (1 | g1) + (1 + x2 | g2), data = d,
///           weights = w, REML = TRUE)
/// print(summary(f)$coefficients, digits = 15)
/// print(as.data.frame(VarCorr(f)), digits = 15)
/// print(sigma(f), digits = 15); print(REMLcrit(f), digits = 15)
/// ```
#[test]
fn fit_sparse_lmm_weighted_matches_lme4() {
    const REF_B0: f64 = 2.266552572193687;
    const REF_B1: f64 = 0.638481148260981;
    const REF_B2: f64 = 0.962298890765967;
    const REF_SE0: f64 = 0.5077453454635866;
    const REF_SE1: f64 = 0.0513758552443655;
    const REF_SE2: f64 = 0.2352975102494342;
    const REF_SD_G1: f64 = 1.91959909459237; // g1 (Intercept) sd
    const REF_SD_G2_INT: f64 = 1.02899506510311; // g2 (Intercept) sd
    const REF_SD_G2_SLOPE: f64 = 0.88391047396846; // g2 x2 sd
    const REF_CORR_G2: f64 = -0.28713485636041; // g2 (Intercept)~x2 corr
    const REF_REMLCRIT: f64 = 1331.89208648957;

    // y,x1,x2,g1,g2 — printed by the R generator above, row order preserved.
    const CSV: &str = "\
2.26569271686515,0.520589072918523,1.33689257455037,0,0
-1.5186057959394,-1.07969076235228,0.552710403069674,0,1
-2.9811587236883,0.139238115019273,-1.05652641739161,0,2
-0.0601500700332526,-0.0847487849485765,-0.190182994643526,0,3
-4.81432513086854,-0.666639615284596,-0.909224176086877,0,4
-4.43453898627614,-2.51608903200946,-2.00091550631878,0,5
0.589552441301372,-0.735146797456677,1.90897086539068,0,6
1.92594155455609,-1.02012226313509,1.56898020541778,0,7
-0.309667203228029,0.113554441297307,0.140301225311454,0,8
-4.018794602209,-0.473790981840095,-1.42055946460262,0,9
-0.85804375253163,-0.408214704337928,0.513066799512409,0,10
2.19246042100517,-0.730433278593614,-0.987461198891432,0,11
0.0396200192993124,-0.221436599406174,0.597223916449456,0,12
-0.102030816413376,-0.225816524428951,0.867860400604364,0,13
-2.10851820990764,-2.5468814461283,0.189257763959423,0,14
3.49866965048435,1.34700149929703,-0.0693064069135194,0,0
-1.83351937516147,0.616408145849969,1.07625933935365,0,1
0.389550433600885,0.217564338307888,1.07089132376137,0,2
1.06757985063778,-0.804718830400263,0.421633377427644,0,3
-0.691576583717045,0.68974677762582,-0.12454978063033,0,4
2.38099780779407,-0.32867201409024,0.937192065289507,1,5
1.35114821662376,-0.16468157692584,0.200619126393445,1,6
1.7463285772661,-1.3920288797713,1.44817884292342,1,7
0.774296241405956,1.46582476120139,-1.28218256537373,1,8
3.2331770597448,0.0482068082254438,-0.084554210547583,1,9
3.63842951064809,1.90808383464199,1.10161395316682,1,10
3.80992041934262,1.73094452723732,-0.690699575305395,1,11
-0.422150051089063,0.0581458372592962,-1.72541538917727,1,12
2.95748859539586,0.645328161775171,0.565885692360889,1,13
-0.298134815037274,1.7256289865063,-0.596158417224681,1,14
3.87325259548931,-0.528966917456246,-0.489345024258458,1,0
1.92778170805841,0.166392025612551,0.768893360822469,1,1
3.20449145774661,-0.254723574763946,2.11461189033211,1,2
0.694748017765141,0.332782359437909,0.0447809835460714,1,3
2.9048257296396,0.182432812827635,0.947241069815817,1,4
0.459408074564175,1.164593749518,-2.1245087083504,1,5
1.01977812148409,0.593492850727868,0.0904283264577256,1,6
-3.17149145518441,-0.891779729217369,-1.73914879260394,1,7
-0.507176140710402,0.577253310357403,-1.7022883303277,1,8
5.25987772555658,-0.824905581058043,1.40829639529108,1,9
1.59371483138849,-1.15725385762344,0.69816811272235,2,10
3.93299040680764,0.777998774006214,0.5760854391738,2,11
-0.882950614081925,-1.20422178597307,-0.534336060471033,2,12
-1.20422221508216,0.30671410214768,-1.28198635078153,2,13
1.57549041393497,-0.833658540642056,0.429614218440963,2,14
5.79938991056964,1.41814116350375,-1.12943767906222,2,0
-1.58622346009505,0.711786117850718,2.41976367943378,2,1
-1.9262839558748,-0.402497722862373,0.0782761044809724,2,2
1.99934979881041,0.799362218714509,0.316225234142966,2,3
-4.39468388516595,0.426484435556948,-1.5794002086685,2,4
0.627053096634319,-1.16989866993562,0.518905620032067,2,5
0.827790744616347,-0.206200160293019,1.13539238442515,2,6
-0.787038368825956,-0.930610309243612,0.312404463905503,2,7
1.14738778062059,0.449397565791737,0.05554413008899,2,8
4.42713095224106,-0.644806454562313,1.56545282333373,2,9
1.81780019603085,-0.231422689810958,0.718830766147177,2,10
2.27148869810909,-1.23636801144537,0.634325711079806,2,11
1.25020193047794,-0.960955298223015,1.06590668950288,2,12
-1.17150955810353,0.133956281569561,0.341888554248612,2,13
0.352688193209334,-0.999052722969284,3.14336714054499,2,14
3.40686437894807,-0.141470690813213,0.349950832519039,3,0
-0.471635823978737,0.167329899648813,1.07333545251137,3,1
1.00554687919145,-0.198788612671681,0.916437872494606,3,2
3.50810764792225,-0.291207188092097,0.71765690362959,3,3
0.658857424220627,-1.73439703424929,1.05548502974692,3,4
-0.0904474330216343,-0.272728331732604,0.350592199470931,3,5
-5.09523064290751,-1.79992948169682,-1.93603847989057,3,6
-1.02846795703,1.15274097680304,-0.800741941084278,3,7
0.586483899298131,-1.003319485592,2.34293958213089,3,8
5.68295765735492,0.148210044292539,1.92087838946788,3,9
-0.121531734602508,0.519496680749176,-0.109538639946886,3,10
-0.826802938504754,0.00543629447576128,-2.51282078633865,3,11
0.226562421442393,1.34702083465394,-0.759623633062425,3,12
-1.58569819123598,-0.847033417295996,-1.50546286281683,3,13
-0.779971461417941,0.443398017315772,-1.67824570835886,3,14
0.294002162100748,-0.977149468323314,0.950098626350105,3,0
1.9322911155916,2.12449113551361,0.124953593318981,3,1
2.27730963144657,0.687698960561541,1.43839197208062,3,2
0.472084772409255,-0.343368220180396,-1.32535121599528,3,3
-0.800181337311174,0.785169472673946,-1.00393459317338,3,4
1.71145531996263,-1.15557746077052,0.642208565811121,4,5
5.994760052792,1.5089163463927,1.15655108036238,4,6
2.76846211305985,-1.15549616025736,-0.173573480005486,4,7
-0.190503450643527,-1.55681181020668,-0.964569098629924,4,8
7.36586891209075,-0.0552479252235566,1.85379901629885,4,9
5.46590601134132,0.849193162702715,1.68436511079776,4,10
6.23279004273887,-0.0110967225036592,1.05731214066179,4,11
3.24623705278299,-0.760313831780719,0.814561626549095,4,12
2.35336230811426,1.17579925167624,-1.19459693410643,4,13
4.33623961351255,2.41444970903476,-2.02885658613829,4,14
1.62715060166342,-1.95851276822613,2.12408653976063,4,0
2.50896885107831,1.4735898934298,0.299219062813124,4,1
0.54061021870137,-0.47488396149873,-0.824643051149339,4,2
5.46747558802284,0.981170983785874,1.52434567788945,4,3
-0.80736289054826,-1.82435521953656,-0.614820405730783,4,4
3.1214776049804,-0.260610817352268,-0.45976066348376,4,5
2.39631445252684,-0.95884531257366,0.443640987583446,4,6
0.0746993657381853,-0.490295710066525,-1.39598147444982,4,7
1.35084625343769,-1.07729671360066,-0.502477861628173,4,8
2.4791449476834,0.369425910544446,-1.06059878223795,4,9
3.98134052679875,1.21626648113935,1.17460715763047,5,10
2.95118815646248,-0.493843235614103,0.330371459656535,5,11
0.702861116552784,-0.227784720977188,-0.288433220566114,5,12
3.24685893560117,-1.11164888704727,1.32156210264278,5,13
1.29530044172321,0.995833089409624,-0.0454740273538609,5,14
4.14453044490995,0.561828890946087,-0.214793901616877,5,0
-1.45398705653092,-1.56427855651588,0.13480451241815,5,1
1.31399034047831,1.48107701910531,-0.644997500085118,5,2
1.43520607046579,-1.16920515011223,0.0570719155291017,5,3
3.67137424233706,-1.05201353746636,0.931184671545328,5,4
-1.0977512841226,-1.47184552482902,0.203663452276264,5,5
1.2509943559665,-1.19343692076419,1.62579612344971,5,6
-1.05167159879544,-0.347979424691937,-0.593355958720706,5,7
1.21793715924739,0.532822543149418,-0.308569082055948,5,8
3.67785038705233,-0.710988575850843,1.33644832852931,5,9
-0.825819371934104,-0.198989137632404,-1.20646409452483,5,10
4.05892226705497,-0.281373323518464,-1.40888180770124,5,11
2.22897955694354,1.17092627520498,0.644424623754144,5,12
2.39599317340592,2.28873318592,-0.408867428984605,5,13
-1.6352668953481,-1.06885661449908,-0.979236369824986,5,14
9.38708524659219,1.90620418856242,-0.927231216419259,6,0
3.43041227648816,2.13179771847887,0.79270411853512,6,1
2.01283540814053,0.231456363015352,-0.68909903597529,6,2
5.53768423353945,0.896945737827196,-0.0416516404215562,6,3
0.639810178440599,-1.73879271235152,-0.785618777019648,6,4
5.07122818829766,0.468847940135532,0.4344554586641,6,5
3.62873601312113,-0.544147673197739,-0.665566901846607,6,6
2.59485468781172,-0.165414153808447,0.449936130069846,6,7
2.70892697929705,0.552166562572813,-0.548040463452155,6,8
8.39262686140589,1.0333302116035,1.85693691208123,6,9
5.16427669880862,-0.0461788047835456,0.40207812111575,6,10
5.32354880498233,2.63870731259579,-1.86460318556058,6,11
4.97392739403681,0.589005749225018,1.37493052047263,6,12
3.23165006697256,-0.202377334229212,-0.0746035055403732,6,13
2.43005548122974,0.441560138002382,0.22779389971164,6,14
2.91105803005144,-0.100257702082834,1.38865619323333,6,0
2.95785821542996,-1.09340399199111,0.405822127350406,6,1
5.37220266451377,0.50324733858265,2.56136759868205,6,2
5.30300377418895,0.949240091117635,-0.526334569793747,6,3
2.7779627219107,0.382802056022312,0.352407393770692,6,4
2.82445985644845,0.371578210898053,0.475161141291331,7,5
3.44655441614692,0.157216401651425,0.791905108509305,7,6
3.08578795988607,0.847939889975375,0.831271029700769,7,7
1.26774946058887,-1.28339265650518,0.548697929846243,7,8
3.41203124650436,1.1582786080896,-1.24227498092802,7,9
2.4643441629754,-0.909106213165988,-0.413370331843489,7,10
4.1998551273158,0.334990705168581,0.0710386853613732,7,11
2.65106770610979,0.775854081916407,-0.494271818071687,7,12
3.30351154166187,0.237140152276008,0.322143267993693,7,13
2.27839323409679,-1.5598701345872,-0.796429134375704,7,14
5.47131261424797,0.0243267511888821,0.907116281386659,7,0
3.2402748889524,0.334635189158287,0.207539150347981,7,1
4.43736779623431,0.989410999147139,0.547842405004734,7,2
3.28381655339623,0.500375113541597,-0.22169528429994,7,3
1.46555566233963,0.576907865455451,-0.681797182178005,7,4
4.92305361813861,1.14269094011911,1.45985706973908,7,5
2.1163686791436,-0.707447814422978,-0.862507887588664,7,6
0.00643484611127254,-0.618559640981128,-0.800383855157766,7,7
1.04755524045436,-0.0083141961424557,-0.68514196580317,7,8
7.1062184037943,0.338919733952689,1.82132263410678,7,9
3.57533325210901,1.40540629884683,-0.104744220596189,8,10
5.64779425600001,-0.865269852362635,1.11147452679476,8,11
-0.891668553283255,-0.873953956620092,-1.02752209427414,8,12
0.356549447167564,0.7743182008249,-0.424123583125695,8,13
0.82592984745751,-0.401890241517252,-0.159149418177784,8,14
2.3005213649307,-0.215137830535894,0.658170841972663,8,0
2.41398792663423,0.605256130514011,-0.303464556482093,8,1
1.84386315529868,-0.614567683647475,0.696562801882666,8,2
2.18969036284595,-0.724422931424412,0.112772261493095,8,3
-0.0380423847239506,-0.256942637181777,0.180256561398902,8,4
4.69730769943493,-0.392072736349559,2.1212044307174,8,5
0.903192622928767,-0.686701961744657,-0.67724515768195,8,6
2.58575362837787,0.875840340130818,0.975279636884236,8,7
2.64125275066828,-0.481057986042492,1.09733861872641,8,8
5.84795221822321,0.0683575908680962,1.66738580380783,8,9
2.62193240171985,0.137024892815848,0.324119853477316,8,10
3.8563345261667,-1.85876851887058,0.250784954668287,8,11
3.43108823810242,0.34627657136052,0.153309143680586,8,12
2.27348663149062,1.50163381994693,0.715229305996022,8,13
4.32934407502284,-0.0137381203294002,0.610541619161322,8,14
5.90186898868146,-0.77524519751103,-0.281816891217829,9,0
5.01253698699677,1.3111581356762,0.885998134265093,9,1
5.31144205413505,0.260524556554842,0.594126927122837,9,2
6.19481710543531,1.01401700762575,0.700118281593645,9,3
11.9559811275874,0.215594208474653,2.81491279503349,9,4
5.2303358565521,0.613404626091988,0.277078099153463,9,5
5.15424953941641,1.44598848106203,-0.513741244477688,9,6
7.79144654070019,0.658085900745181,0.422214152789072,9,7
4.24658462605188,0.375234496491491,-0.265564340886777,9,8
5.08617950518273,-0.674688275938179,-0.770354222150151,9,9
1.94542169274344,0.580480573296274,-0.376500645612233,9,10
6.64787716742674,-0.414229332226612,0.110222930784873,9,11
6.80681545552938,2.18050398539242,0.972561427059444,9,12
3.31209546770424,0.342857643650257,0.356994016994474,9,13
2.32671707034836,-0.798024661907156,-0.824167316178698,9,14
7.21935671354296,1.64103616625365,-0.155630678991287,9,0
3.54361245488635,-0.460005111651252,-1.52971660161298,9,1
2.50288083851835,-1.58647098860618,-0.202227510837209,9,2
4.09482283397433,-1.04491567407179,-1.31151722791159,9,3
0.647744748214764,0.0416536658857592,-1.39007893871245,9,4
3.89151529857333,-0.43202637747734,0.962138141645872,10,5
1.13509005446349,0.458591147686799,-0.17813477901026,10,6
0.631617869951369,-0.229351986059321,0.00328960117506923,10,7
0.0684576631481358,-1.85696220984083,0.511615165517015,10,8
2.84919245138515,-0.289674172981153,-0.456060688239042,10,9
0.777461810452513,1.76714123786919,-0.21090411916793,10,10
2.40670685517177,-0.442682897300816,-0.572554868573356,10,11
0.380815972203836,-0.588864402001567,-0.353336222838735,10,12
-1.45935706620346,-0.120588258352901,-1.0496560241288,10,13
1.12096438102176,1.65306128186181,0.7883855209797,10,14
3.48842714061664,-0.871354568629743,0.619231580144126,10,0
2.56978568807385,0.780668270857782,-0.785768492164587,10,1
1.58309604180605,-0.613877869318662,1.78160405193807,10,2
2.12500448831244,-0.327591311306049,-0.249928557425748,10,3
-0.625176013058818,-0.355216015901553,-0.301097396588297,10,4
-0.988401769236405,0.78093407436684,-1.49933210991998,10,5
2.76616843242921,0.608670171785426,0.0342053478098827,10,6
-0.981172655493283,1.07617901747348,-0.911047222176761,10,7
1.93077229615646,1.06555765231322,0.351385587315867,10,8
5.2258968883385,2.16164641282926,0.195414362215204,10,9
5.01422594382777,0.0704564820522911,0.859629926713806,11,10
1.39621745569964,-2.53522521959317,-0.674155138589608,11,11
1.55940430075499,-0.541334144723391,-1.66258565149713,11,12
4.18126886782737,-0.775939573541146,0.390045780841936,11,13
2.19201867024286,-0.295955418239763,0.441365682409681,11,14
3.35019324418225,1.15137290542338,1.28396592434211,11,0
1.17910922211284,-0.568887867531516,-0.163185801701405,11,1
1.37144283995466,-0.805786340025032,0.499569043121688,11,2
4.68883410330573,0.492593861841638,0.460144067064487,11,3
2.49861156696724,0.700758337332243,-0.476282040022087,11,4
0.297125233019732,-0.243125291550084,-1.55664239038557,11,5
1.65586952413087,0.412481176922476,-0.689022539113182,11,6
2.20939174396122,0.498165309358799,0.148407389086632,11,7
3.60768937977397,0.367679114623033,0.135429490904212,11,8
0.473503070559156,-0.5706028691897,-1.37234184047492,11,9
2.65739735938482,0.738816517714368,-0.0931036773678838,11,10
1.09549858311319,-2.26829132566495,-1.11820793439463,11,11
4.34079101292021,0.614041040052466,0.744827357549737,11,12
2.94154761353848,0.40363580599992,0.0776825859578852,11,13
2.5789199169556,2.05157932669931,-0.499928792474106,11,14
7.15562877125623,0.263244987215003,0.611743478942702,12,0
7.52942712893248,0.110421189446662,-0.869034775516044,12,1
7.53658870103454,1.1957636212874,1.16615799034312,12,2
7.69875663943353,0.733309513633103,-0.769731638856476,12,3
3.53349570030228,-0.00451169154164245,-1.80724436689568,12,4
7.75599098760304,-0.0212154995950339,-0.917465740651156,12,5
8.29492652392404,0.138501732981574,-0.356456664570957,12,6
5.77118392614591,-1.07012706995328,-0.176553706785467,12,7
6.51894024206579,1.23606206697686,-0.799311276346493,12,8
8.37182081239118,-0.539810094735419,-0.392796425823479,12,9
8.06156180082909,-0.296228914656068,0.273725082324397,12,10
10.9751035432975,0.688218803349637,0.258141982501913,12,11
5.49644579149136,-0.399007303727868,0.110412063687764,12,12
4.76059001011066,-0.103484852849487,-0.915161122522455,12,13
6.89678540395981,1.52953404059542,0.212936649900559,12,14
7.84927463172794,-0.634636458448834,1.40333534124712,12,0
5.64298472650673,-0.226854589097728,-0.808434808833702,12,1
10.5775268684518,1.5010840927328,1.91953356242643,12,2
6.97098329237163,-0.342463482378634,0.865668884307897,12,3
3.36156799508795,-0.553727556011483,-1.2639077211064,12,4
0.378320492608427,0.257626870698959,-0.956625877294363,13,5
2.02271739165599,0.613121758652457,-0.303370061193043,13,6
2.50087308274843,-0.652287366777574,-0.00220360490056386,13,7
0.331516429772245,-0.138281029086895,-0.913071354975655,13,8
2.1759599046682,-0.123638001869199,-0.649643176443167,13,9
0.935698838684268,-0.964705216383371,0.376311077588093,13,10
4.76027228500782,0.0190431840153162,0.2779963826487,13,11
3.08274517152035,1.69169235319906,-0.754720890372454,13,12
0.362265554013126,0.885809071474169,-0.473084717719469,13,13
-0.209892881545623,-0.729930145482526,-0.91395042710506,13,14
5.29010757986385,0.351884683098013,-2.60329970942661,13,0
2.86222384533804,-0.247330893260195,-0.593467498200933,13,1
0.541305855262702,-0.163591831723322,-0.638979894573173,13,2
4.58747754254356,0.872360560208246,-0.105916956315576,13,3
-1.68077623742475,0.605938213883317,-0.980291154529057,13,4
3.34057025830932,0.927240812612819,0.476235459637523,13,5
2.15594421679729,0.458044436651209,-1.41239128440147,13,6
4.35417214217403,-0.801114024214896,1.63100315238839,13,7
0.929647833888533,-0.497123152995539,-1.12798588227457,13,8
4.14507827036828,-1.21296781485439,0.438451089522611,13,9
7.91086286973373,0.619712599356425,1.90065023437915,14,10
7.07793429487854,1.3596632579562,-0.700760424199841,14,11
3.04880350657769,-1.6210647627138,-0.3717393835228,14,12
3.38479824697546,0.0795594993555375,-0.678503248030561,14,13
2.47498174919331,0.512593137306753,0.101451284461551,14,14
5.67923212120527,-0.21910193815088,0.553279834051488,14,0
5.90169535140175,-0.82593869467054,-1.52222099582696,14,1
4.56353918278271,-1.01357824411184,-1.02154930182434,14,2
4.78511905042227,1.28206624602831,-1.39068213766217,14,3
4.26360932642697,-0.212144499179618,0.433339795920382,14,4
4.20876733341715,1.18487723662641,0.369698592138803,14,5
4.21124382466685,-0.650612436772287,-0.709493645627637,14,6
2.58569634407,1.28736727005853,-1.01894940268142,14,7
2.44114444578842,0.538933215663317,-1.64074625333368,14,8
6.97860927472029,0.0255405538039062,0.682710512141664,14,9
3.07432238994928,1.1639475327246,-0.751876713541874,14,10
8.43803748147673,-0.367625732117171,0.716236911745028,14,11
6.67354287901515,-0.363369569501376,1.99500140843909,14,12
5.85937193154085,-0.00675336474544712,1.34613637848412,14,13
4.17435054827609,-0.100472498864641,-0.826907119029068,14,14
1.09005031614737,-0.14859207836112,1.13846516751087,15,0
-2.83865032099968,-0.182952755436934,-0.670743488970344,15,1
-0.938325171293181,-0.85814783765375,0.28830533745626,15,2
1.94979856599394,0.958159182178004,0.163840249645937,15,3
-0.00341580540936931,-0.839076941832303,0.543705814431928,15,4
-1.26253096019206,-0.759724279311683,0.0089607826965823,15,5
-0.341783234965533,-0.866514819536214,0.29143972911296,15,6
1.48725828702367,1.65902566868139,0.632100222998882,15,7
-2.85197544813362,0.556942277210519,-2.12229472459214,15,8
3.31048872245074,0.601386129662215,-0.273116168728087,15,9
-2.83253555344797,-1.96978682110424,0.379269650648216,15,10
1.11096383170432,-0.380107748973682,0.207411582705335,15,11
1.54743343131236,1.15777354406233,0.130075482647751,15,12
0.416094923017226,1.38963465416294,1.7290152298527,15,13
-1.83423671122836,-2.01150842477619,-0.131777383691539,15,14
-0.108456295708925,-2.33630850037639,0.0901800389952223,15,0
-1.84243945652219,-1.36845977011138,-0.122655944893569,15,1
-1.19521946775632,-1.37309426035361,2.34798597248153,15,2
0.105926440769032,0.737388550085491,-0.62236219518714,15,3
2.34386701060351,0.454709341750177,1.09608012991909,15,4
5.1602502143626,1.77668132830334,1.13795870017689,16,5
3.44693311443338,0.194985588228136,-0.740714885875294,16,6
4.32823108923858,-0.344612622139577,1.23768243980547,16,7
3.52061151400087,1.61267663514805,0.589619597581652,16,8
5.8794048234845,0.775709363612466,1.00741836369821,16,9
0.15566753860005,0.70427912561433,-1.75300650648146,16,10
5.67209827449965,1.07215421601982,-0.773431753710342,16,11
2.52040974711728,-0.214130769685907,-0.386735295761461,16,12
3.41652904947139,0.363631063912172,0.188347531556599,16,13
4.04574049825908,1.08356378657411,-0.0619437868449765,16,14
3.11120645180015,-1.95227615716347,-0.148388226122151,16,0
1.98863932173461,0.0155101786160552,0.599105401299345,16,1
4.12865903208638,-0.87098092003781,0.913847832792144,16,2
3.36060928899094,-0.900934214910906,1.6666000600483,16,3
3.38226224255814,-0.776112067634224,0.603637972396393,16,4
1.50304186062726,0.361430751154015,-1.16201140949401,16,5
3.62454742547555,-0.0173493122643544,-1.37223198751365,16,6
5.51416981373375,-1.11819535979872,1.33404139102141,16,7
2.44199126032025,0.58669137484252,-0.393542558040908,16,8
5.49169300907432,0.486195009924034,0.353585138470633,16,9
-0.951818442783229,0.473894455796676,0.400055481683396,17,10
4.36670800464802,0.361427452891757,0.666740074189319,17,11
2.08786949709306,1.21768364371144,-0.420241256035585,17,12
2.59778611712379,-0.673116960131551,-0.328174632770276,17,13
0.670310546207943,-0.598848241837351,0.437923476479789,17,14
5.51612719379865,0.735873146511805,-0.784044326886472,17,0
1.56464658091896,0.60173965147488,-0.226837232126531,17,1
-0.516310093164733,1.10889193968727,-0.582210315071111,17,2
1.64550816362994,-0.0113905983860746,-1.098110347489,17,3
1.93506142881849,0.605035960687119,0.162352442701946,17,4
3.15528261004126,1.12352815317424,0.450619475819499,17,5
1.77978930877787,-0.705362053710043,0.0982995234973026,17,6
2.6668810620111,1.76247207167883,0.334055561690702,17,7
-0.210390495250704,-0.730020695494859,-0.0903722507191762,17,8
2.48061769643047,-1.63415762678176,-0.175437620600107,17,9
-3.05062592569791,-0.123993779923598,-1.57072135945584,17,10
5.74588350008261,0.204139470306424,1.21159302733537,17,11
2.11431080840813,-0.68323921942861,1.45793829178585,17,12
-1.53936155366522,-1.01636227170084,0.865075477914802,17,13
0.122737096929622,-0.671434165633787,0.589875474734274,17,14
4.02754464819057,1.12186281570044,0.0828203415601401,18,0
-3.89155121488193,-2.10066370498434,-0.403684784749441,18,1
-0.641225919377077,-2.09087185958204,0.193777916274814,18,2
0.968259515344114,-1.29110001012018,-0.553142679203068,18,3
2.67946939293739,1.07199151952253,0.70346433981999,18,4
1.09273451810304,0.894870152083501,0.397668729642592,18,5
-1.14510877567253,-0.842632298901112,0.0100055504835981,18,6
-2.74291672627674,0.0462974922741544,-0.255537690151586,18,7
-0.114157474872505,0.623368148702539,0.301799212469126,18,8
2.46194292180836,0.485091951868738,0.723584081288875,18,9
4.2638499717217,0.531944473401682,1.4151701539861,18,10
0.597612760000028,0.19742268076406,-1.37337950574091,18,11
-1.39896865485694,0.326566784462888,-0.599548070268292,18,12
1.14275226967094,-1.44023199151019,1.71015618024855,18,13
1.26818139246462,2.11470907987023,-0.163937529099855,18,14
-0.357599222211188,-0.184672313220123,1.60460144055646,18,0
0.595219788318556,0.492104327529081,-0.798674754667866,18,1
-1.09841049200863,-1.36132698556342,1.18037424884885,18,2
-0.690708526314822,-0.365217557903885,-0.859589831327811,18,3
-3.46994601445431,0.7561265592095,-0.86974967242439,18,4
1.44537791179307,0.146502808516189,0.295584565295028,19,5
0.320847946955593,0.261947248506149,-0.274837917019254,19,6
-1.67925440798133,-1.30615904372434,-1.11238074519984,19,7
1.00912714915343,0.492164032443828,-1.10611190648605,19,8
2.67352955812131,-0.511104000679042,0.308028588983761,19,9
-0.479845890329478,-1.66057454138599,-1.08568792211512,19,10
5.44120705815668,0.622376370866293,1.01683942966976,19,11
2.88984062323628,0.0157310112822735,1.29398257664835,19,12
-1.27063462328962,-0.733887125964384,-1.20692916422951,19,13
0.502941640703045,0.27521113796498,-1.74997090694174,19,14
4.65159670267425,-1.8587385910095,-2.19522755244751,19,0
1.61625280047006,2.07889036683578,0.780089945806557,19,1
-1.47686362308648,-2.36088032720629,-1.08546476672529,19,2
2.16827566267286,0.327135397295895,1.01612857543453,19,3
-1.37175699513399,-0.227522726278505,-0.21040661077906,19,4
-0.156059773478137,-1.18059531411805,-1.33866666877921,19,5
1.88241314709922,0.330421531559684,-0.876139679397935,19,6
1.50851097286275,-0.406934780043714,0.0471728881584331,19,7
3.69461149161872,1.52522324018498,2.01695237231075,19,8
2.6661495091834,-1.10318386829999,0.372699001110427,19,9
";

    let mut y = Vec::<f64>::new();
    let mut x1 = Vec::<f64>::new();
    let mut x2 = Vec::<f64>::new();
    let mut g1 = Vec::<u32>::new();
    let mut g2 = Vec::<u32>::new();
    for line in CSV.lines().filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        y.push(f[0].parse().unwrap());
        x1.push(f[1].parse().unwrap());
        x2.push(f[2].parse().unwrap());
        g1.push(f[3].parse().unwrap());
        g2.push(f[4].parse().unwrap());
    }
    let n = y.len();
    assert_eq!(n, 400);
    let p = 3;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = x1[i];
        x[i * p + 2] = x2[i];
    }
    let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — data path derives it
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 }, // placeholder likewise
                slopes: vec![2], // random slope on x2 (col 2) ⇒ slope_extras ⇒ Sparse
            }],
        }),
    };
    let ids = crate::GroupIds {
        primary: g1,
        extra: vec![g2],
    };
    let (sized, ids, _perm) = crate::fit::spec_sized_from_ids_pub(&model, &ids);
    assert!(
        matches!(
            crate::fit::classify_design_pub(&sized, 1),
            crate::fit::Solver::Sparse
        ),
        "slope-carrying extra grouping must route Sparse"
    );

    let opts = crate::FitOptions {
        target_indices: vec![0, 1, 2],
        weights: Some(w.clone()),
        ..crate::FitOptions::default()
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);

    assert!(f.converged(), "weighted sparse LMM must converge");
    assert!(
        (f.beta[0] - REF_B0).abs() / REF_B0.abs() < 1e-5,
        "β0 {} vs {REF_B0}",
        f.beta[0]
    );
    assert!(
        (f.beta[1] - REF_B1).abs() / REF_B1.abs() < 1e-5,
        "β1 {} vs {REF_B1}",
        f.beta[1]
    );
    assert!(
        (f.beta[2] - REF_B2).abs() / REF_B2.abs() < 1e-5,
        "β2 {} vs {REF_B2}",
        f.beta[2]
    );
    assert!(
        (f.se[0] - REF_SE0).abs() / REF_SE0 < 1e-3,
        "se0 {} vs {REF_SE0}",
        f.se[0]
    );
    assert!(
        (f.se[1] - REF_SE1).abs() / REF_SE1 < 1e-3,
        "se1 {} vs {REF_SE1}",
        f.se[1]
    );
    assert!(
        (f.se[2] - REF_SE2).abs() / REF_SE2 < 1e-3,
        "se2 {} vs {REF_SE2}",
        f.se[2]
    );

    // varcorr[0] = primary g1 (scalar vech), varcorr[1] = extra g2 (2×2
    // vech) — `assemble_varcorr` orders primary-then-extras (fit.rs:528).
    assert_eq!(
        f.varcorr.len(),
        2,
        "two grouping blocks: g1 (scalar) + g2 (2×2)"
    );
    let vc = &f.varcorr[1];
    assert_eq!(vc.len(), 3, "q=2 vech has 3 entries");
    let sd_int = vc[0].sqrt();
    let sd_slope = vc[2].sqrt();
    let corr = vc[1] / (sd_int * sd_slope);
    assert!(
        (sd_int - REF_SD_G2_INT).abs() / REF_SD_G2_INT < 1e-3,
        "g2 intercept sd {sd_int} vs {REF_SD_G2_INT}"
    );
    assert!(
        (sd_slope - REF_SD_G2_SLOPE).abs() / REF_SD_G2_SLOPE < 1e-3,
        "g2 slope sd {sd_slope} vs {REF_SD_G2_SLOPE}"
    );
    // corr = vc[1]/(sd_int·sd_slope) is a ratio of the off-diagonal vech
    // component to two stddevs on this weighted, small-n (n=400, 15 g2
    // levels) fit — it's the noisiest of the three θ coordinates, hence the
    // 0.05 band next to the 1e-5-scale β checks above.
    assert!(
        (corr - REF_CORR_G2).abs() < 0.05,
        "g2 corr {corr} vs {REF_CORR_G2}"
    );

    // g1 is a scalar primary RE: tau2[0] = θ0²·σ̂² is its variance directly.
    let sd_g1 = f.tau2[0].sqrt();
    assert!(
        (sd_g1 - REF_SD_G1).abs() / REF_SD_G1 < 1e-3,
        "g1 sd {sd_g1} vs {REF_SD_G1}"
    );

    // Fit.deviance vs REMLcrit(f) − (n−p)·(1+ln 2π) — same −Σlog wᵢ
    // constant convention pinned by the dense Task 5 golden
    // (`fit_lmm_weighted_matches_lme4`, fit.rs).
    let df = (n - p) as f64;
    let expected = REF_REMLCRIT - df * (1.0 + (2.0 * std::f64::consts::PI).ln());
    assert!(
        (f.deviance - expected).abs() < 1e-3,
        "deviance {} vs lme4-derived {expected}",
        f.deviance
    );
    // loglik = −REMLcrit/2 (weighted REML criterion on the logLik scale,
    // mirrors the dense weighted golden's loglik gate).
    assert!(
        (f.loglik - (-REF_REMLCRIT / 2.0)).abs() < 1e-3,
        "loglik {} vs lme4 {}",
        f.loglik,
        -REF_REMLCRIT / 2.0
    );
    assert!(f.reml);
    assert_eq!(f.df, 3 + 4 + 1); // 3 β + (g1 scalar + g2 q=2 vech) θ + σ²

    // Natural design has 15 g2 levels (< TAIL_SPARSE_MIN) ⇒ this golden takes
    // the dense tail. Force the sparse (AMD-ordered) tail on the SAME fit to
    // exercise that route against a real weighted golden instead of only
    // synthetic shapes. NOT bit-identical: AMD reorders the elimination, a
    // sanctioned reassociation of the same Cholesky (src/sparse/mod.rs
    // `SparseTail` doc), so the two converged optima agree only to
    // reassociation-error size — matching the existing dense/sparse-tail
    // equality precedent in this file (`sparse_tail_clique_pattern_and_
    // deviance_match_dense`, ~line 752: 1e-6 abs on β/se). A run showed
    // ~1e-9 drift on β0 here; 1e-6 keeps headroom without being loose.
    let f_forced = with_forced_sparse_tail(|| crate::fit_cold(&x, &y, n, p, &model, &ids, &opts));
    assert!(
        f_forced.converged(),
        "forced-sparse-tail refit must converge"
    );
    for j in 0..p {
        assert!(
            (f_forced.beta[j] - f.beta[j]).abs() < 1e-6,
            "forced-sparse-tail beta[{j}] {} vs unforced {}",
            f_forced.beta[j],
            f.beta[j]
        );
        assert!(
            (f_forced.se[j] - f.se[j]).abs() < 1e-6,
            "forced-sparse-tail se[{j}] {} vs unforced {}",
            f_forced.se[j],
            f.se[j]
        );
    }
    for k in 0..f.tau2.len() {
        assert!(
            (f_forced.tau2[k] - f.tau2[k]).abs() < 1e-6,
            "forced-sparse-tail tau2[{k}] {} vs unforced {}",
            f_forced.tau2[k],
            f.tau2[k]
        );
    }
}

/// Task 6: constant weights (w ≡ 2) on a sparse-classified design (extra
/// grouping slope ⇒ `slope_extras` ⇒ `Solver::Sparse`) must reproduce the
/// unweighted fit's β, SE, and tau2 — same θ̃ = √c·θ argument as the dense
/// twin `fit_lmm_constant_weights_invariant` (fit.rs), now exercised
/// through the sparse blocked-Cholesky kernel instead of the dense
/// suff-stats accumulator.
#[test]
fn sparse_lmm_constant_weights_invariant() {
    let n_g1 = 8usize;
    let n_g2 = 6usize;
    let per = 10usize;
    let n = n_g1 * per;
    let mut st = 29u64;
    let mut x = vec![0.0f64; n * 2];
    let mut y = vec![0.0f64; n];
    let mut g1 = vec![0u32; n];
    let mut g2 = vec![0u32; n];
    for i in 0..n {
        g1[i] = (i % n_g1) as u32;
        g2[i] = (i % n_g2) as u32;
        let x1 = super::test_lcg(&mut st);
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1;
        let re1 = 0.4 * ((g1[i] as f64) - (n_g1 as f64) / 2.0);
        let re2 = 0.3 * ((g2[i] as f64) - (n_g2 as f64) / 2.0);
        y[i] = 0.5 + 0.4 * x1 + re1 + re2 + 0.2 * super::test_lcg(&mut st);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![1], // slope on x ⇒ slope_extras ⇒ Sparse
            }],
        }),
    };
    let ids = crate::GroupIds {
        primary: g1,
        extra: vec![g2],
    };
    let (sized, ids, _perm) = crate::fit::spec_sized_from_ids_pub(&model, &ids);
    assert!(matches!(
        crate::fit::classify_design_pub(&sized, 1),
        crate::fit::Solver::Sparse
    ));

    let base_opts = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };
    let unweighted = crate::fit_cold(&x, &y, n, 2, &model, &ids, &base_opts);
    let weighted = crate::fit_cold(
        &x,
        &y,
        n,
        2,
        &model,
        &ids,
        &crate::FitOptions {
            weights: Some(vec![2.0; n]),
            ..base_opts
        },
    );
    assert!(unweighted.converged() && weighted.converged());
    for j in 0..2 {
        assert!(
            (unweighted.beta[j] - weighted.beta[j]).abs() / unweighted.beta[j].abs() < 1e-6,
            "β[{j}] unweighted {} vs w≡2 {}",
            unweighted.beta[j],
            weighted.beta[j]
        );
        // NOT a pin: there is no frozen value here, only two live fits that
        // must agree, so `assert_pinned`'s anchor machinery does not apply and
        // neither machine is a reference. Measured at the shipped RHO_END:
        // se[0] disagrees by 4.4e-7 on the anchor (x86_64) and 3.2e-6 on
        // aarch64-apple-darwin. Widened from the original 1e-6, which the
        // anchor still passes, to cover aarch64 with ~3x headroom.
        //
        // MECHANISM, measured (2026-08-05), not guessed. The
        // unweighted and w≡2 fits are two INDEPENDENT BOBYQA searches over the
        // same REML surface, not one trajectory, so they agree to the accuracy
        // the optimizer delivers in θ̂ rather than to machine precision. Same
        // argument as the dense twin `fit_lmm_constant_weights_invariant`
        // (`fit/lmm_tests.rs`) and `balanced_collapse_weighted_fit_invariant`
        // (`src/lmm/tests.rs`). A scratch-tree RHO_END sweep
        // over this fixture (five sweep points spanning four decades, three
        // equivalent constant weights w ∈ {2, 4, 8} per sweep point so the trend is
        // visible through the per-search scatter) has the gap tracking
        // RHO_END where RHO_END binds, then flooring:
        //
        //   RHO_END   worst se gap   geometric mean over w ∈ {2, 4, 8}
        //   1e-5      6.4e-6         5.0e-6
        //   1e-6      3.2e-6         1.1e-6   <- shipped
        //   1e-7      1.1e-6         3.0e-7
        //   1e-8      9.5e-7         3.7e-7
        //   1e-9      9.5e-7         3.7e-7
        //
        // Slope over 1e-5 -> 1e-7, where RHO_END is the binding stop: 4.1x per
        // decade raw, against 10x for exact proportionality. The 6.4x
        // floor-subtracted figure is not a slope over that range — it is the
        // single 1e-5 -> 1e-6 decade (a two-point basis); floor-subtracting
        // the next decade goes negative (the geometric-mean column is already
        // below the estimated floor at RHO_END = 1e-7). The floor below 1e-7 is the
        // REML objective's own FP noise, not a second mechanism: the
        // deviance resolves to ~1e-12 absolute out of 244 (4e-15 relative),
        // and the residual disagreement in θ̂ moves it by less than that, so
        // no trust radius can separate the two points.
        //
        // What is NOT in it: a fixed-θ control (both fits forced to the same
        // θ̂, rescaled by √c, so only the downstream linear algebra runs
        // independently) closes to ≤1.4e-13 relative on se and ≤3.8e-15 on
        // tau2, four to five orders below the shipped-RHO_END gaps. The
        // weighted path's √w scaling through the sufficient statistics
        // contributes essentially nothing; the search contributes essentially
        // all of it. The earlier "FP reassociation on long reductions" guess is
        // superseded by that control.
        assert!(
            (unweighted.se[j] - weighted.se[j]).abs() / unweighted.se[j] < 1e-5,
            "se[{j}] unweighted {} vs w≡2 {}",
            unweighted.se[j],
            weighted.se[j]
        );
    }
    assert_eq!(unweighted.tau2.len(), weighted.tau2.len());
    for k in 0..unweighted.tau2.len() {
        // BOBYQA rho_end floor, same mechanism and same 2026-08-05 measurement
        // as the `se` comment above: two independently-converged sparse fits,
        // not a shared trajectory. tau2 is θ², so it carries roughly 2x the θ̂
        // scatter and its gaps run ~3x the se ones over the same sweep:
        //
        //   RHO_END   worst tau2 gap   geometric mean over w ∈ {2, 4, 8}
        //   1e-5      2.2e-5           1.7e-5
        //   1e-6      1.0e-5           3.5e-6   <- shipped
        //   1e-7      3.6e-6           9.1e-7
        //   1e-8      3.1e-6           1.1e-6
        //   1e-9      3.1e-6           1.1e-6
        //
        // Slope over 1e-5 -> 1e-7: 4.3x per decade raw, 6.5x floor-subtracted
        // (the single 1e-5 -> 1e-6 decade only — same two-point basis caveat
        // as the se comment above).
        //
        // A boundary-pinned component (θ collapsed to exactly 0 in both
        // fits) needs an absolute floor: relative error is undefined at 0.
        //
        // Widened from the original 1e-5, which the anchor (x86_64) still
        // passes, because aarch64-apple-darwin shows tau2[1] at 1.01e-5,
        // already past the old bound on its own. Optimizer scatter, not a new
        // regression; 3e-5 gives ~3x headroom there while still catching a real
        // divergence.
        let denom = unweighted.tau2[k].abs().max(1e-8);
        assert!(
            (unweighted.tau2[k] - weighted.tau2[k]).abs() / denom < 3e-5,
            "tau2[{k}] unweighted {} vs w≡2 {}",
            unweighted.tau2[k],
            weighted.tau2[k]
        );
    }
}

/// The sparse LMM route names the components it pinned.
///
/// Both sparse routes assemble a `Fit` directly rather than going through a
/// `FitView`, so their `Diagnostics::pinned` is filled at the pin loop instead
/// of by `fit::common::materialize_diagnostics`. This asserts the two halves
/// that can drift: that the grid is non-empty at all on this route, and that it
/// lines up block-for-block and slot-for-slot with `stddev_corr`, which is the
/// alignment both wrapper packages iterate to name the collapsed component.
///
/// The design is `y ~ x + (1 | gp) + (1 + x | ge)`: the slope-carrying extra
/// grouping is what routes it to Sparse, and `y` is built with no `gp` effect
/// at all, so `gp`'s single component collapses to the boundary.
#[test]
fn sparse_lmm_pinned_names_the_collapsed_component() {
    let n = 240;
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let gp: Vec<u32> = (0..n).map(|i| (i % 12) as u32).collect();
    let ge: Vec<u32> = (0..n).map(|i| (i / 12) as u32).collect();
    // Per-ge intercept and slope offsets give `ge` real variance in both of its
    // components; `gp` enters nowhere. The residual is centred WITHIN each `gp`
    // level so that even the sampling noise carries no `gp` signal — otherwise a
    // few 1e-4 of spurious between-level variance keeps the optimizer just off
    // the boundary and the test stops testing what it is here to test.
    let mut st = 7u64;
    let mut noise = vec![0.0f64; n];
    for e in noise.iter_mut() {
        *e = 0.1 * (super::test_lcg(&mut st) - 0.5);
    }
    for lvl in 0..12u32 {
        let rows: Vec<usize> = (0..n).filter(|&i| gp[i] == lvl).collect();
        let mean = rows.iter().map(|&i| noise[i]).sum::<f64>() / rows.len() as f64;
        for &i in &rows {
            noise[i] -= mean;
        }
    }
    for i in 0..n {
        let xi = super::test_lcg(&mut st) * 2.0 - 1.0;
        let g = ge[i] as f64;
        x[i * p] = 1.0;
        x[i * p + 1] = xi;
        y[i] = 1.0 + 0.75 * xi + (g * 0.37).sin() + (g * 0.91).cos() * xi + noise[i];
    }

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 12 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 20 },
                slopes: vec![1],
            }],
        }),
    };
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse,
    ));
    let ids = crate::GroupIds {
        primary: gp,
        extra: vec![ge],
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);

    assert!(f.converged(), "sparse fit must converge");
    assert!(f.singular(), "gp has no signal, so its component pins");
    assert_eq!(f.diagnostics.boundary, crate::Boundary::AtBoundary);
    assert_eq!(
        f.diagnostics.pinned,
        vec![vec![true], vec![false, false]],
        "gp's only component pinned; ge's two did not"
    );
    // The alignment the wrappers rely on: one flag per stddev, per block.
    assert_eq!(f.diagnostics.pinned.len(), f.varcorr.len());
    for g in 0..f.varcorr.len() {
        let (sd, _) = f.stddev_corr(g);
        assert_eq!(f.diagnostics.pinned[g].len(), sd.len(), "block {g} width");
    }
    assert_eq!(
        f.stddev_corr(0).0[0],
        0.0,
        "a pinned component reports sd 0"
    );
}

/// [`Diagnostics::pinned`]'s reshape walks the varcorr blocks in order and
/// consumes `vech_q(block.len())` bits per block from a single
/// `diagonal_theta`-ordered mask (`pinned_flags` in `fit/common.rs`) — a
/// cursor bug that swaps two bits inside the SAME block would still pass
/// [`sparse_lmm_pinned_names_the_collapsed_component`], because that fixture
/// only ever pins a whole 1-wide block. This one pins the second bit of the
/// SECOND block instead, so a swapped cursor (block 1's slope bit landing at
/// `pinned[1][0]` instead of `pinned[1][1]`, or the pin leaking into block 0)
/// fails the assertion below.
///
/// The design is `y ~ x + (1 | g) + (1 + x | h)`: primary `g` (12 levels) and
/// extra `h` (20 levels, which is what routes this to Sparse) both get a real
/// per-level intercept effect, but `h`'s slope carries no signal — there is no
/// `x·h` interaction term. The residual noise is assigned in `(even g, odd g)`
/// pairs sharing one draw per `h` block, the same decorrelation trick
/// `sparse_lmm_pinned_names_the_collapsed_component` uses (there: noise keyed
/// off `k/2` so it can't correlate with `x`'s `k`-parity): `x`'s sign flips
/// with `g`'s parity, so a shared draw across an (even, odd) pair contributes
/// `+v` and `−v` to the pair's `x`-covariance, exactly zero, in EVERY `h`
/// block — not just on average. Without that, the LCG noise alone would
/// occasionally correlate with `x` inside some `h` block and keep the slope
/// component just off the boundary (see `diagnostics_pinned_aligns_with_varcorr_blocks`'s
/// note on why this fixture only pins reliably once the residual is genuinely
/// slope-free).
#[test]
fn sparse_lmm_pinned_names_the_second_groupings_slope() {
    let n = 240;
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let g: Vec<u32> = (0..n).map(|i| (i % 12) as u32).collect();
    let h: Vec<u32> = (0..n).map(|i| (i / 12) as u32).collect();

    let mut st = 11u64;
    let u0: Vec<f64> = (0..12).map(|_| 0.6 * super::test_lcg(&mut st)).collect();
    let v0: Vec<f64> = (0..20).map(|_| 0.6 * super::test_lcg(&mut st)).collect();
    // One noise draw per (g-pair, h) cell, shared by the pair's even/odd `g`
    // — see the doc comment above for why this zeroes the x-noise covariance
    // exactly rather than only in expectation.
    let mut noise = vec![0.0f64; n];
    for hh in 0..20u32 {
        for pair in 0..6usize {
            let v = 0.4 * super::test_lcg(&mut st);
            for gi in [2 * pair, 2 * pair + 1] {
                let i = hh as usize * 12 + gi;
                noise[i] = v;
            }
        }
    }
    for i in 0..n {
        let gi = g[i] as usize;
        let hi = h[i] as usize;
        let x1 = if gi % 2 == 0 { 1.0 } else { -1.0 };
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        y[i] = 1.0 + 0.75 * x1 + u0[gi] + v0[hi] + noise[i];
    }

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 12 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 20 },
                slopes: vec![1],
            }],
        }),
    };
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse,
    ));
    let ids = crate::GroupIds {
        primary: g,
        extra: vec![h],
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);

    assert!(f.converged(), "sparse fit must converge");
    assert!(
        f.singular(),
        "h's slope has no signal, so its component pins"
    );
    assert_eq!(f.diagnostics.boundary, crate::Boundary::AtBoundary);
    assert_eq!(
        f.diagnostics.pinned,
        vec![vec![false], vec![false, true]],
        "g's intercept and h's intercept are real; h's slope pins"
    );
    assert_eq!(f.diagnostics.pinned.len(), f.varcorr.len());
    for blk in 0..f.varcorr.len() {
        let (sd, _) = f.stddev_corr(blk);
        assert_eq!(
            f.diagnostics.pinned[blk].len(),
            sd.len(),
            "block {blk} width"
        );
    }
    // NOT exactly zero: the pin fixes the diagonal θ (λ₁₁ = 0 exactly), but
    // `stddev_corr(1).0[1]` is `√(λ₁₀² + λ₁₁²)`, so it inherits whatever the
    // off-diagonal λ₁₀ settled on — the same q≥2-pinned-reads-as-negligible
    // shape `diagnostics_pinned_aligns_with_varcorr_blocks` documents.
    let (h_sd, _) = f.stddev_corr(1);
    assert!(
        h_sd[1] / h_sd[0] < 1e-6,
        "h's pinned slope component must be negligible next to its intercept: {h_sd:?}"
    );
    assert!(
        f.stddev_corr(0).0[0] > 0.0,
        "g's intercept variance is real"
    );
    assert!(h_sd[0] > 0.0, "h's intercept variance is real");
}

/// The sparse GLMM route names the components it pinned — the binomial twin of
/// [`sparse_lmm_pinned_names_the_collapsed_component`], covering the second of
/// the two sites that fill `Diagnostics::pinned` outside the view mappers. The
/// sparse NB route reaches the same site through `fit_glmm_sparse`.
#[test]
fn sparse_glmm_pinned_names_the_collapsed_component() {
    let n = 240;
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let gp: Vec<u32> = (0..n).map(|i| (i % 12) as u32).collect();
    let ge: Vec<u32> = (0..n).map(|i| (i / 12) as u32).collect();
    let mut st = 13u64;
    for i in 0..n {
        let xi = super::test_lcg(&mut st) * 2.0 - 1.0;
        let g = ge[i] as f64;
        x[i * p] = 1.0;
        x[i * p + 1] = xi;
        let eta = 0.4 + 1.2 * xi + 1.5 * (g * 0.37).sin();
        y[i] = f64::from(super::test_lcg(&mut st) < 1.0 / (1.0 + (-eta).exp()));
    }

    let model = ModelSpec {
        family: Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 12 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 20 },
                slopes: vec![1],
            }],
        }),
    };
    assert!(matches!(
        crate::fit::classify_design_pub(&model, 1),
        crate::fit::Solver::Sparse,
    ));
    let ids = crate::GroupIds {
        primary: gp,
        extra: vec![ge],
    };
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);

    assert!(f.converged(), "sparse binomial fit must converge");
    assert!(f.singular(), "gp carries no signal, so its component pins");
    assert!(
        !f.diagnostics.pinned.is_empty(),
        "the sparse GLMM route must name its pins, not report an empty grid"
    );
    // The WHOLE grid, not just `pinned[0][0]`. `ge`'s block is 2 wide, and
    // `pinned_flags` consumes its bits from the same `diagonal_theta`-ordered
    // mask with a running cursor — the swap this catches is the one
    // [`sparse_lmm_pinned_names_the_second_groupings_slope`] exists for, on the
    // GLMM route. Asserting only the 1-wide block leaves both of `ge`'s bits
    // unread and a cursor that mixed them would pass.
    // `ge`'s slope pins too, and that is the discriminating bit: the DGP has no
    // `x·ge` term, so only `ge`'s intercept carries signal. `pinned[1]` is
    // therefore `[false, true]` — a cursor that swapped the block's two bits, or
    // let one leak into block 0, changes this grid.
    assert_eq!(
        f.diagnostics.pinned,
        vec![vec![true], vec![false, true]],
        "gp's component and ge's signal-free slope pinned; ge's intercept did not"
    );
    assert_eq!(f.diagnostics.pinned.len(), f.varcorr.len());
    for g in 0..f.varcorr.len() {
        let (sd, _) = f.stddev_corr(g);
        assert_eq!(f.diagnostics.pinned[g].len(), sd.len(), "block {g} width");
    }
}

// ---------------------------------------------------------------------------
// Internal random-effect column scaling (`LmmGroupings::set_slope_scales`) —
// sparse LMM/GLMM rescale tests
//
// GOVERNING IDEA, shared with `crate::fit::lmm_tests`'s and `crate::fit::
// glmm_tests`'s rescale tests: multiply a random-slope design column by an
// exact power of two `C` and refit. A dropped back-map shows up unmistakably
// as a ratio of 1 instead of the predicted `1/C` or `1/C²` (see `lmm_tests.
// rs`'s rescale test for the full `Z~ = Z·diag(1/s)`, `Λ~ = diag(s)·Λ`
// derivation). Here the random slope sits on an EXTRA (crossed) grouping
// rather than the primary — `classify_design` sends any design with a slope
// on an extra grouping to `Solver::Sparse` regardless of level counts, which
// is what forces both tests below onto the sparse route.
// ---------------------------------------------------------------------------

/// Primary intercept-only grouping (12 levels) + one CROSSED extra grouping
/// (20 levels) carrying a genuine random slope on column 1 — the
/// `slope_extras` shape `classify_design` always routes `Solver::Sparse`.
/// Column 1 is both the fixed-effect covariate and the extra grouping's slope
/// covariate, mirroring the dense rescale tests' single-column double-duty
/// design. `x1`'s ±1-by-primary-level-parity shape is
/// `sparse_lmm_pinned_names_the_second_groupings_slope`'s proven-convergent
/// design; UNLIKE that fixture, `v1` here carries real per-level signal so the
/// extra block's slope variance converges away from the boundary — this test
/// needs a genuinely nonzero D10/D11 to exercise the rescale identity on.
fn sparse_lmm_slope_extra_design() -> (Vec<f64>, Vec<f64>, usize, usize, ModelSpec, crate::GroupIds)
{
    let n = 240;
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let g: Vec<u32> = (0..n).map(|i| (i % 12) as u32).collect();
    let h: Vec<u32> = (0..n).map(|i| (i / 12) as u32).collect();

    let mut st = 71u64;
    let u0: Vec<f64> = (0..12).map(|_| 0.6 * super::test_lcg(&mut st)).collect();
    let v0: Vec<f64> = (0..20).map(|_| 0.6 * super::test_lcg(&mut st)).collect();
    let v1: Vec<f64> = (0..20).map(|_| 0.3 * super::test_lcg(&mut st)).collect();
    for i in 0..n {
        let gi = g[i] as usize;
        let hi = h[i] as usize;
        let x1 = if gi % 2 == 0 { 1.0 } else { -1.0 };
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        y[i] = 1.0 + 0.75 * x1 + u0[gi] + v0[hi] + v1[hi] * x1 + 0.3 * super::test_lcg(&mut st);
    }

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 12 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 20 },
                slopes: vec![1],
            }],
        }),
    };
    let ids = crate::GroupIds {
        primary: g,
        extra: vec![h],
    };
    (x, y, n, p, model, ids)
}

/// Sparse LMM rescale identity, `C = 1024.0` — the sparse-route twin of
/// `crate::fit::lmm_tests`'s dense rescale test, same predicted moves (see
/// that test's doc comment for the derivation): the primary block here is
/// intercept-only (q_p=1, `block_row_scale` always 1.0 on it) so it is
/// untouched by construction; the extra grouping's q_g=2 block carries the
/// same `[untouched, /C, /C²]` vech pattern the dense test's primary block
/// does.
#[test]
fn sparse_lmm_rescaling_slope_column_moves_every_quantity_by_the_predicted_power_of_c() {
    const C: f64 = 1024.0;
    const BAND: f64 = 1e-5;
    const DEV_ABS: f64 = 1e-9;

    let (x, y, n, p, model, ids) = sparse_lmm_slope_extra_design();
    assert!(
        matches!(
            crate::fit::classify_design_pub(&model, 1),
            crate::fit::Solver::Sparse
        ),
        "extra-grouping slope must route Sparse"
    );
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        ..crate::FitOptions::default()
    };

    let base = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(base.converged(), "base sparse LMM must converge");

    let mut x_c = x.clone();
    for i in 0..n {
        x_c[i * p + 1] *= C;
    }
    let scaled = crate::fit_cold(&x_c, &y, n, p, &model, &ids, &opts);
    assert!(scaled.converged(), "column-scaled sparse LMM must converge");

    assert_pinned(&[scaled.beta[0]], &[base.beta[0]], BAND, "beta[0]");
    assert_pinned(&[scaled.beta[1]], &[base.beta[1] / C], BAND, "beta[1]");
    assert_pinned(&[scaled.se[0]], &[base.se[0]], BAND, "se[0]");
    assert_pinned(&[scaled.se[1]], &[base.se[1] / C], BAND, "se[1]");

    // varcorr: block 0 primary (q=1, untouched — no slope on it), block 1
    // extra vech [D00, D10, D11].
    assert_eq!(scaled.varcorr.len(), 2, "primary + one extra block");
    assert_pinned(
        &scaled.varcorr[0],
        &base.varcorr[0],
        BAND,
        "varcorr primary",
    );
    assert_pinned(
        &scaled.varcorr[1],
        &[
            base.varcorr[1][0],
            base.varcorr[1][1] / C,
            base.varcorr[1][2] / (C * C),
        ],
        BAND,
        "varcorr extra vech",
    );

    // tau2: primary contributes 1 entry (untouched), the extra's q=2 vech
    // contributes 3 more — indices 2, 3 are both the extra's slope ROW (row
    // 1), so both carry the squared /C² factor; index 1 (row 0) is untouched.
    assert_eq!(scaled.tau2.len(), 4);
    assert_pinned(
        &scaled.tau2,
        &[
            base.tau2[0],
            base.tau2[1],
            base.tau2[2] / (C * C),
            base.tau2[3] / (C * C),
        ],
        BAND,
        "tau2",
    );

    // ranef: primary block (12 levels × q=1, untouched) then extra block (20
    // levels × q=2: [b0, b1/C] per level).
    assert_eq!(scaled.ranef_levels, vec![12, 20]);
    let mut want_ranef = Vec::with_capacity(scaled.ranef.len());
    want_ranef.extend_from_slice(&base.ranef[..12]);
    for l in 0..20 {
        want_ranef.push(base.ranef[12 + l * 2]);
        want_ranef.push(base.ranef[12 + l * 2 + 1] / C);
    }
    assert_pinned(&scaled.ranef, &want_ranef, BAND, "ranef");

    // fitted — unchanged elementwise (same model, same conditional means).
    assert_pinned(&scaled.fitted, &base.fitted, BAND, "fitted");

    // deviance / loglik — same log|X'V⁻¹X| Jacobian argument as the dense LMM
    // test: one column of a 2-column X scaled by C moves the REML deviance by
    // +2·ln(C) and loglik by -ln(C).
    let dev_shift = scaled.deviance - base.deviance;
    let expected_dev_shift = 2.0 * C.ln();
    assert!(
        (dev_shift - expected_dev_shift).abs() < DEV_ABS,
        "deviance shift {dev_shift} vs predicted {expected_dev_shift}"
    );
    let loglik_shift = scaled.loglik - base.loglik;
    let expected_loglik_shift = -C.ln();
    assert!(
        (loglik_shift - expected_loglik_shift).abs() < DEV_ABS,
        "loglik shift {loglik_shift} vs predicted {expected_loglik_shift}"
    );
}

/// Parses `validation/data/simulated/sim_binomial_slope_crossed.csv` — the
/// same aggregated-binomial fixture `fit_sparse_binomial_slope_crossed_is_
/// pinned` gates against a frozen pin, reused here because it is already
/// known to converge with BOTH its q=2 blocks (primary g1, extra g2) away
/// from the boundary (that pin's `REF_VC_G1`/`REF_VC_G2` are real, nonzero
/// vechs). Column 2 (`x`) is the fixed-effect covariate and the random-slope
/// covariate on BOTH groupings; column 1 (`size`) is the prior weight, not a
/// design column. Returns `(x, y, n, p, model, ids, weights)`.
#[allow(clippy::type_complexity)]
fn sparse_glmm_slope_crossed_design() -> (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    ModelSpec,
    crate::GroupIds,
    Vec<f64>,
) {
    let csv = include_str!("../../validation/data/simulated/sim_binomial_slope_crossed.csv");
    let mut y = Vec::<f64>::new();
    let mut size_col = Vec::<f64>::new();
    let mut xcol = Vec::<f64>::new();
    let (mut g1_raw, mut g2_raw) = (Vec::<String>::new(), Vec::<String>::new());
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let incidence: f64 = f[0].parse().unwrap();
        let size: f64 = f[1].parse().unwrap();
        y.push(incidence / size);
        size_col.push(size);
        xcol.push(f[2].parse().unwrap());
        g1_raw.push(f[3].to_string());
        g2_raw.push(f[4].to_string());
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = xcol[i];
    }
    let model = ModelSpec {
        family: Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // sized from ids
            slopes: vec![1],                                 // (1 + x | g1)
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![1], // (1 + x | g2)
            }],
        }),
    };
    let ids = crate::GroupIds {
        primary: dense_ids(&g1_raw),
        extra: vec![dense_ids(&g2_raw)],
    };
    (x, y, n, p, model, ids, size_col)
}

/// Sparse GLMM rescale identity, `C = 4.0`, `WaldSe::Hessian` (the crate
/// default) — the sparse-route twin of `crate::fit::glmm_tests`'s dense GLMM
/// rescale test, same predicted moves and the same reason for the smaller `C`
/// and looser band (the joint `[θ | β]` BOBYQA search with a `BETA_BOX` on
/// raw β makes the two fits' internal paths genuinely different, not one
/// bit-identical search read twice — see that test's doc comment). Both
/// groupings here (primary g1, extra g2) carry a random slope on the SAME
/// column, so both blocks' vechs move by the identity at once.
#[test]
fn sparse_glmm_rescaling_slope_column_moves_stddev_se_by_the_predicted_power_of_c() {
    const C: f64 = 4.0;
    const BAND: f64 = 3e-4;
    const DEV_ABS: f64 = 1e-9;

    let (x, y, n, p, model, ids, weights) = sparse_glmm_slope_crossed_design();
    assert!(
        matches!(
            crate::fit::classify_design_pub(&model, 1),
            crate::fit::Solver::Sparse
        ),
        "extra-grouping slope must route Sparse"
    );
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        weights: Some(weights.clone()),
        ..crate::FitOptions::default() // WaldSe::Hessian
    };

    let base = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(base.converged(), "base sparse GLMM must converge");
    assert!(
        base.stddev_se.iter().all(|v| v.is_finite()),
        "base stddev_se must be finite on a converged Hessian fit: {:?}",
        base.stddev_se
    );

    let mut x_c = x.clone();
    for i in 0..n {
        x_c[i * p + 1] *= C;
    }
    let scaled = crate::fit_cold(&x_c, &y, n, p, &model, &ids, &opts);
    assert!(
        scaled.converged(),
        "column-scaled sparse GLMM must converge"
    );
    assert!(
        scaled.stddev_se.iter().all(|v| v.is_finite()),
        "scaled stddev_se must be finite on a converged Hessian fit: {:?}",
        scaled.stddev_se
    );

    assert_pinned(&[scaled.beta[0]], &[base.beta[0]], BAND, "beta[0]");
    assert_pinned(&[scaled.beta[1]], &[base.beta[1] / C], BAND, "beta[1]");
    assert_pinned(&[scaled.se[0]], &[base.se[0]], BAND, "se[0]");
    assert_pinned(&[scaled.se[1]], &[base.se[1] / C], BAND, "se[1]");

    // varcorr: both g1 (primary) and g2 (extra) are q=2 blocks with a real
    // slope, so both vechs carry the [untouched, /C, /C²] pattern.
    assert_eq!(scaled.varcorr.len(), 2, "g1 + g2, both q=2");
    for blk in 0..2 {
        let want = [
            base.varcorr[blk][0],
            base.varcorr[blk][1] / C,
            base.varcorr[blk][2] / (C * C),
        ];
        assert_pinned(
            &scaled.varcorr[blk],
            &want,
            BAND,
            &format!("varcorr[{blk}]"),
        );
    }

    // tau2: two q=2 vechs back to back — same [untouched, /C², /C²] pattern
    // in each 3-entry block.
    assert_eq!(scaled.tau2.len(), 6);
    let want_tau2: Vec<f64> = (0..2)
        .flat_map(|blk| {
            let b = blk * 3;
            [
                base.tau2[b],
                base.tau2[b + 1] / (C * C),
                base.tau2[b + 2] / (C * C),
            ]
        })
        .collect();
    assert_pinned(&scaled.tau2, &want_tau2, BAND, "tau2");

    // ranef: g1's block (n_g1 levels × q=2) then g2's block (n_g2 levels ×
    // q=2), each level [b0, b1/C].
    assert_eq!(scaled.ranef_levels.len(), 2);
    let mut want_ranef = Vec::with_capacity(scaled.ranef.len());
    let mut off = 0usize;
    for &levels in &scaled.ranef_levels {
        for l in 0..levels {
            want_ranef.push(base.ranef[off + l * 2]);
            want_ranef.push(base.ranef[off + l * 2 + 1] / C);
        }
        off += levels * 2;
    }
    assert_pinned(&scaled.ranef, &want_ranef, BAND, "ranef");

    // stddev_se — the item this test exists for: ONE power of the row scale
    // per block (θ-scale SE, not squared like tau2).
    assert_eq!(scaled.stddev_se.len(), 6);
    let want_stddev_se: Vec<f64> = (0..2)
        .flat_map(|blk| {
            let b = blk * 3;
            [
                base.stddev_se[b],
                base.stddev_se[b + 1] / C,
                base.stddev_se[b + 2] / C,
            ]
        })
        .collect();
    assert_pinned(&scaled.stddev_se, &want_stddev_se, BAND, "stddev_se");

    // deviance — no REML Jacobian on the GLMM route: a genuine
    // reparameterization, so the marginal criterion is invariant.
    assert!(
        (scaled.deviance - base.deviance).abs() < DEV_ABS,
        "deviance moved under a column reparameterization: {} vs {}",
        scaled.deviance,
        base.deviance
    );
}

/// The sparse GLMM route runs the same two-stage search as the dense one, so
/// it must report the same three quantities. The extra stage-1 warm-start
/// evaluation at theta-hat-1 (`sparse/glmm.rs`'s `d1`) is a fit-path eval and
/// is counted; `n_eval` does not include it, so the split is compared with
/// that one evaluation allowed for.
#[cfg(feature = "counters")]
#[test]
fn sparse_glmm_counters_split_stages_and_histogram() {
    let (x, y, n, p, model, ids, weights) = sparse_glmm_slope_crossed_design();
    let opts = crate::FitOptions {
        target_indices: vec![0, 1],
        weights: Some(weights),
        ..crate::FitOptions::default()
    };
    let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(f.converged(), "sparse GLMM fixture must converge");
    let c = f.counters;
    assert!(c.stage_evals[0] > 0, "sparse stage 1 must record evals");
    assert!(c.stage_evals[1] > 0, "sparse stage 2 must record evals");
    assert_eq!(
        (c.stage_evals[0] + c.stage_evals[1]) as usize,
        f.n_eval + 1,
        "stage split reconstructs n_eval plus the stage-1 warm-start eval"
    );
    assert_eq!(
        c.pirls_hist.iter().sum::<u32>(),
        c.stage_evals[0] + c.stage_evals[1],
        "one histogram entry per fit-path eval"
    );
    assert_eq!(
        c.pirls_hist[0], 0,
        "no eval solves PIRLS in zero iterations"
    );
}
