use super::agq::*;
use super::derivative::*;
use super::deviance::*;
use super::pirls::*;
use super::se::*;
use super::workspace::*;
use super::*;
use crate::counters::EvalCounters;
use crate::fit::SizedSpec;
use crate::test_support::{intercept_only_spec, TestWs};
use crate::{
    BinomialLink, Family, GammaLink, Grouping, GroupingRelation, ModelSpec, NegBinomialLink,
    PoissonLink, ReStructure, WaldSe,
};
use faer::linalg::solvers::Solve;

/// SE of each θ coordinate: the square roots of the θ-block covariance's
/// diagonal (on a scalar block, the `stddev_se` that
/// `stddev_se_kernel_to_caller` reports).
fn theta_se(ws: &GlmmWorkspace) -> Vec<f64> {
    let cov = &ws.inference.theta_cov;
    (0..cov.nrows())
        .map(|k| match cov[(k, k)] {
            v if v < 0.0 => 0.0,
            v => v.sqrt(),
        })
        .collect()
}

/// Fold the packed `M` values at `params` into a workspace already built on the
/// packed layout with its columns filled (`fill_packed_cols`) — the oracle fill
/// the blocked and structured kernels are checked against below. The packed
/// layout orders the primary RE block component-major (`c·n_primary + f`) where
/// the other two are level-major (`f·q_p + c`), so `u` and `M`'s columns come
/// out permuted; see [`packed_u_index`].
fn packed_oracle_fill(ws: &mut GlmmWorkspace, x: faer::MatRef<f64>, params: &[f64], n: usize) {
    let nt = ws.n_theta;
    crate::sparse::fill_lambda_small(&params[..nt], &ws.groupings, &mut ws.packed.lam_small);
    fill_z_f64(&ws.groupings, x, x.subcols(0, 0), &mut ws.z_buf, n);
    fill_m_vals(&mut ws.packed, &ws.groupings, &ws.z_buf, n);
}

/// Level-major RE index (`f·q_p + c`, the blocked and structured layouts) → the
/// packed layout's component-major index (`c·n_primary + f`). Outside the
/// primary block the two orders coincide.
fn packed_u_index(q: usize, s: usize, idx: usize) -> usize {
    if idx < q * s {
        (idx % q) * s + idx / q
    } else {
        idx
    }
}

/// Dense `M = ZΛ` scattered out of a packed workspace's rows — the dense Schur
/// recomputations below need a full `n×k` matrix. Columns are in the packed
/// layout's order, which permutes `M` relative to the blocked and structured
/// layouts; a Schur complement is invariant under that permutation, since `A`
/// and `X'W̃M` permute together.
fn packed_dense_m(ws: &GlmmWorkspace, n: usize, k: usize) -> Mat<f64> {
    let width = ws.packed.width;
    let mut m = Mat::<f64>::zeros(n, k);
    for i in 0..n {
        for t in i * width..(i + 1) * width {
            m[(i, ws.packed.m_cols[t] as usize)] = ws.packed.m_vals[t];
        }
    }
    m
}

/// The blocked and structured GLMM kernels reject a hand-built workspace whose
/// extra grouping carries a random slope (`extra_slopes_any`): they build
/// intercept-only extras, so such a design would fit a REDUCED model and report
/// it as a normal success. `classify_design` routes any extra-slope shape to
/// Sparse and `GlmmLayout::for_design` puts it on the packed layout, which does
/// carry the slope — the guard is the backstop for a caller that asks for a
/// blocked or structured workspace over that design anyway. The panic comes from
/// `from_groupings` (release too), not from a per-eval `debug_assert`.
#[test]
#[should_panic(expected = "slope-carrying extra grouping")]
fn structured_glmm_entry_rejects_extra_slopes() {
    let model = SizedSpec {
        spec: ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes: vec![1],
                }],
            }),
        },
        n_primary: 2,
        extra_counts: vec![2],
    };
    let n = 8;
    let p = 2;
    let groupings = crate::lmm::LmmGroupings::from_cluster_spec(&model, &[]);
    let _ = GlmmWorkspace::from_groupings(
        groupings,
        model.spec.family,
        p,
        n,
        1,
        GlmmLayout::Structured,
    );
}

/// The packed layout is Laplace-only (see `from_groupings`'s doc): whatever
/// `nagq` the caller asks for, a packed workspace pins it to 1 before
/// `outer_search` is computed, so the route decision sees the nAGQ the
/// workspace will actually run.
#[test]
fn packed_layout_pins_nagq_to_one() {
    let model = SizedSpec {
        spec: ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes: vec![1],
                }],
            }),
        },
        n_primary: 2,
        extra_counts: vec![2],
    };
    let n = 8;
    let p = 2;
    let groupings = crate::lmm::LmmGroupings::from_cluster_spec(&model, &[]);
    let ws =
        GlmmWorkspace::from_groupings(groupings, model.spec.family, p, n, 7, GlmmLayout::Packed);
    assert_eq!(ws.nagq, 1);
}

/// Intercept-only spec carrying the **binomial-logit** family. `ws.family` drives
/// the deviance and IRLS weights the GLMM kernel computes, so these binomial tests
/// must set it explicitly rather than inherit `intercept_only_spec`'s Gaussian default.
fn logit_intercept_spec() -> ModelSpec {
    let mut s = intercept_only_spec();
    s.family = Family::Binomial {
        link: BinomialLink::Logit,
    };
    s
}

// Tiny deterministic LCG → reproducible test data without RNG-determinism caveats.
fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
}

/// Clustered-binary dataset: n=80, p=2 (intercept + x1), 8 clusters, a
/// random intercept (q_p=1). Returns (X f64, y∈{0,1}, cluster_ids).
/// `contiguous` selects the id layout: false = round-robin `i % nc`
/// (the `FixedClusters` production layout, the historical fixture), true =
/// block layout `i / per_cluster` (the `FixedSize`/DGEN-FS production
/// layout). The blocked path must hold on both — layout-sensitive rewrites
/// of its row loops are a live optimization direction.
fn glmm_intercept_dataset_layout(
    contiguous: bool,
    n: usize,
    nc: usize,
) -> (Mat<f64>, Vec<f64>, Vec<u32>) {
    let mut st = 7u64;
    let u0: Vec<f64> = (0..nc).map(|_| 0.6 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = if contiguous { i / (n / nc) } else { i % nc };
        ids[i] = c as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        let eta = 0.2 + 0.8 * x1 + u0[c];
        let p = 1.0 / (1.0 + (-eta).exp());
        y[i] = if lcg(&mut st) + 0.5 < p { 1.0 } else { 0.0 };
    }
    (x, y, ids)
}

fn glmm_intercept_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>) {
    glmm_intercept_dataset_layout(false, 80, 8)
}

/// q_p=2 (intercept + slope on col 0) primary grouping plus ONE crossed extra
/// grouping. Returns (X f64 [1, x1], y, primary ids, crossed ids, spec). Every
/// crossed level gets ≥1 obs (round-robin), and every primary level too.
fn glmm_slope_crossed_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>, Vec<u32>, SizedSpec) {
    glmm_slope_crossed_dataset_sized(4, 1.0)
}

/// `glmm_slope_crossed_dataset` with a caller-chosen crossed level count and a
/// multiplier on all three RE draw scales. `(4, 1.0)` reproduces it
/// byte-for-byte.
fn glmm_slope_crossed_dataset_sized(
    n_crossed: usize,
    re_scale: f64,
) -> (Mat<f64>, Vec<f64>, Vec<u32>, Vec<u32>, SizedSpec) {
    let (n, n_prim) = (96usize, 8usize);
    let mut st = 13u64;
    let u0: Vec<f64> = (0..n_prim).map(|_| re_scale * 0.6 * lcg(&mut st)).collect();
    let u1: Vec<f64> = (0..n_prim).map(|_| re_scale * 0.4 * lcg(&mut st)).collect();
    let uc: Vec<f64> = (0..n_crossed)
        .map(|_| re_scale * 0.5 * lcg(&mut st))
        .collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    let mut crossed = vec![0u32; n];
    for i in 0..n {
        let c = i % n_prim;
        let cc = i % n_crossed;
        ids[i] = c as u32;
        crossed[i] = cc as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        let eta = 0.2 + 0.8 * x1 + u0[c] + u1[c] * x1 + uc[cc];
        let p = 1.0 / (1.0 + (-eta).exp());
        y[i] = if lcg(&mut st) + 0.5 < p { 1.0 } else { 0.0 };
    }
    // Slope on design col 0 of the [1, x1] X the kernels read (the slope_cols
    // arg is &[1] — the x1 column index in the full X); the spec carries
    // ONE slope + ONE crossed grouping so for_cluster_spec sizes q_p=2 + tail.
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![0],
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes: vec![],
                }],
            }),
        },
        n_primary: n_prim,
        extra_counts: vec![n_crossed],
    };
    (x, y, ids, crossed, cluster)
}

/// Independent Laplace deviance: dense Z (intercept indicators), M = Z·λ
/// (scalar λ = θ[0]), Newton-on-u PIRLS, deviance = d + ‖u‖² + logdet(M'WM+I).
fn brute_force_intercept_laplace(
    theta0: f64,
    beta: &[f64],
    x: &Mat<f64>,
    y: &[f64],
    ids: &[u32],
    nc: usize,
) -> f64 {
    let (n, p) = (x.nrows(), x.ncols());
    let mut m = Mat::<f64>::zeros(n, nc);
    for i in 0..n {
        m[(i, ids[i] as usize)] = theta0;
    }
    let mut u = vec![0.0f64; nc];
    let pen_dev = |u: &[f64], w_out: Option<&mut [f64]>| -> f64 {
        let mut d = 0.0;
        let mut pen = 0.0;
        let mut wbuf = vec![0.0; n];
        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x[(i, j)] * beta[j];
            }
            for c in 0..nc {
                eta += m[(i, c)] * u[c];
            }
            let pi = 1.0 / (1.0 + (-eta).exp());
            d += if eta > 0.0 {
                eta + (-eta).exp().ln_1p()
            } else {
                eta.exp().ln_1p()
            } - y[i] * eta;
            wbuf[i] = (pi * (1.0 - pi)).max(1e-6);
            let _ = pi;
        }
        for &uc in u {
            pen += uc * uc;
        }
        if let Some(w) = w_out {
            w.copy_from_slice(&wbuf);
        }
        2.0 * d + pen
    };
    let mut w = vec![0.0; n];
    for _ in 0..50 {
        let mut eta = vec![0.0; n];
        let mut pvec = vec![0.0; n];
        for i in 0..n {
            let mut e = 0.0;
            for j in 0..p {
                e += x[(i, j)] * beta[j];
            }
            for c in 0..nc {
                e += m[(i, c)] * u[c];
            }
            eta[i] = e;
            let pi = 1.0 / (1.0 + (-e).exp());
            pvec[i] = pi;
            w[i] = (pi * (1.0 - pi)).max(1e-6);
        }
        let mut g = vec![0.0; nc];
        for c in 0..nc {
            let mut s = 0.0;
            for i in 0..n {
                s += m[(i, c)] * (y[i] - pvec[i]);
            }
            g[c] = 2.0 * u[c] - 2.0 * s;
        }
        let mut h = Mat::<f64>::zeros(nc, nc);
        for a in 0..nc {
            for b in 0..nc {
                let mut s = 0.0;
                for i in 0..n {
                    s += m[(i, a)] * w[i] * m[(i, b)];
                }
                h[(a, b)] = 2.0 * (s + if a == b { 1.0 } else { 0.0 });
            }
        }
        let hc = h.as_ref().llt(faer::Side::Lower).unwrap();
        let mut step = Mat::<f64>::zeros(nc, 1);
        for c in 0..nc {
            step[(c, 0)] = g[c];
        }
        hc.solve_in_place(step.as_mut());
        let mut max = 0.0f64;
        for c in 0..nc {
            u[c] -= step[(c, 0)];
            max = max.max(step[(c, 0)].abs());
        }
        if max < 1e-10 {
            break;
        }
    }
    let _ = pen_dev(&u, Some(&mut w));
    let mut a = Mat::<f64>::zeros(nc, nc);
    for r in 0..nc {
        for c in 0..nc {
            let mut s = 0.0;
            for i in 0..n {
                s += m[(i, r)] * w[i] * m[(i, c)];
            }
            a[(r, c)] = s + if r == c { 1.0 } else { 0.0 };
        }
    }
    let ac = a.as_ref().llt(faer::Side::Lower).unwrap();
    let mut logdet = 0.0;
    for r in 0..nc {
        logdet += ac.L()[(r, r)].ln();
    }
    pen_dev(&u, None) + 2.0 * logdet
}

/// White-box, no external oracle: the aggregated-binomial AGQ objective and its
/// per-trial expanded twin, evaluated at a SHARED (β, θ), differ by exactly the
/// data-only saturated constant `2·Σᵢ wᵢ[yᵢ ln yᵢ + (1−yᵢ) ln(1−yᵢ)]` (yᵢ the
/// aggregated proportion, wᵢ the trial count). The RE prior, GH nodes/weights,
/// and log-sum-exp are identical between the encodings, and the constant is
/// u-independent, so it cancels in every `ℓ_c(u_cj) − ℓ_c(ũ_c)` and survives only
/// in the `−2·ℓ_c(ũ_c)` center term. This is the decisive check that `prior_w` is
/// folded into the per-row dev_resid sums exactly as the aggregated≡expanded
/// identity demands. Two independent PIRLS runs converge the shared mode, so the
/// residual is bounded by PIRLS tol, not machine ε (hence 1e-8, not bit-equal).
/// Held at nAGQ ∈ {1, 7}.
#[test]
fn agq_weighted_aggregated_equals_expanded_plus_saturated_const() {
    // 6 clusters, round-robin ids (FixedClusters layout). Each aggregated row is
    // intercept + covariate x1, m trials, s successes strictly in (0, m) so the
    // saturated constant is nonzero.
    let nc = 6usize;
    let mut st = 42u64;
    let mut xa = Vec::<f64>::new(); // aggregated [1, x1] row-major
    let mut ya = Vec::<f64>::new(); // aggregated proportions
    let mut wa = Vec::<f64>::new(); // trial counts (weights)
    let mut ids_a = Vec::<u32>::new();
    let mut xe = Vec::<f64>::new(); // expanded [1, x1] row-major
    let mut ye = Vec::<f64>::new(); // expanded 0/1
    let mut ids_e = Vec::<u32>::new();
    for i in 0..18usize {
        let c = (i % nc) as u32;
        let x1 = lcg(&mut st);
        let m = 4 + (i % 5); // 4..8 trials
        let succ = 1 + i % (m - 1); // 1..m-1 successes (strictly inside)
        xa.extend_from_slice(&[1.0, x1]);
        ya.push(succ as f64 / m as f64);
        wa.push(m as f64);
        ids_a.push(c);
        for k in 0..m {
            xe.extend_from_slice(&[1.0, x1]);
            ye.push(if k < succ { 1.0 } else { 0.0 });
            ids_e.push(c);
        }
    }
    let (n_a, n_e) = (ya.len(), ye.len());
    let mut xa_mat = Mat::<f64>::zeros(n_a, 2);
    for r in 0..n_a {
        xa_mat[(r, 0)] = xa[r * 2];
        xa_mat[(r, 1)] = xa[r * 2 + 1];
    }
    let mut xe_mat = Mat::<f64>::zeros(n_e, 2);
    for r in 0..n_e {
        xe_mat[(r, 0)] = xe[r * 2];
        xe_mat[(r, 1)] = xe[r * 2 + 1];
    }
    // 2·Σ wᵢ[yᵢ ln yᵢ + (1−yᵢ) ln(1−yᵢ)] — the analytic offset (parameter-free).
    let sat: f64 = (0..n_a)
        .map(|i| {
            let y = ya[i];
            2.0 * wa[i] * (y * y.ln() + (1.0 - y) * (1.0 - y).ln())
        })
        .sum();
    assert!(sat < -1e-6, "saturated const must be nonzero (got {sat})");

    let cluster = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: nc,
        extra_counts: vec![],
    };
    for nagq in [1u8, 7] {
        let mut ws_a = GlmmWorkspace::for_cluster_spec(2, &cluster, n_a, &[], nagq);
        ws_a.weighted = true;
        ws_a.prior_w[..n_a].copy_from_slice(&wa);
        ws_a.params[0] = 0.5; // θ (Λ scalar)
        ws_a.params[1] = 0.2; // β0
        ws_a.params[2] = 0.8; // β1
        let pa = ws_a.params.clone();
        let da = glmm_agq_deviance(&pa, &mut ws_a, xa_mat.as_ref(), &ya, &ids_a, n_a, nagq);

        let mut ws_e = GlmmWorkspace::for_cluster_spec(2, &cluster, n_e, &[], nagq);
        ws_e.params[0] = 0.5;
        ws_e.params[1] = 0.2;
        ws_e.params[2] = 0.8;
        let pe = ws_e.params.clone();
        let de = glmm_agq_deviance(&pe, &mut ws_e, xe_mat.as_ref(), &ye, &ids_e, n_e, nagq);

        let got = da - de;
        assert!(
            (got - sat).abs() <= 1e-8 * (1.0 + sat.abs()),
            "nagq={nagq}: dev_agg−dev_exp = {got}, saturated const = {sat} (Δ {})",
            (got - sat).abs()
        );
    }
}

/// Cluster-outer AGQ must be BIT-identical to the node-outer loop: same
/// operands, same per-accumulator summation order (ClusterRowIndex's
/// ascending-row guarantee). Exact equality — approximate closeness here
/// would hide a reordering that invalidates the no-golden-revalidation
/// safety argument. Under `--features parallel` the `Some(idx)` arm is the
/// rayon cluster loop, so this also proves parallel = serial.
#[test]
fn agq_cluster_outer_bit_identical_to_node_outer() {
    let (xf64, y, ids) = glmm_intercept_dataset();
    let cluster = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: 8,
        extra_counts: vec![],
    };
    for nagq in [3u8, 7, 11] {
        let mut ws_a = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], nagq);
        ws_a.params[0] = 0.5; // θ (Λ scalar) ≠ 1 — exercises the λ-scale
        ws_a.params[1] = 0.2;
        ws_a.params[2] = 0.8;
        let params = ws_a.params.clone();
        ws_a.cluster_rows = Some(ClusterRowIndex::build(&ids, ws_a.groupings.n_primary));

        let mut ws_b = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], nagq);
        ws_b.cluster_rows = None;

        let da = glmm_agq_deviance(&params, &mut ws_a, xf64.as_ref(), &y, &ids, 80, nagq);
        let db = glmm_agq_deviance(&params, &mut ws_b, xf64.as_ref(), &y, &ids, 80, nagq);
        assert_eq!(da.to_bits(), db.to_bits(), "nagq={nagq}");
    }
}

/// Compressed Poisson-log / Gamma-log scalar AGQ with some clusters on the row-kernel
/// fallback: cluster 0 carries a −205 offset (and responses scaled by e^−205 to match,
/// so Poisson counts round to 0), so its mode η sits past the compressed route's window
/// and it falls back; the rest compress. The shift is downward because a +205 shift
/// scales a Poisson cluster's deviance by e^205, and every row kernel then forms its
/// node sums from differences of ~1e88 terms: finite or not by round-off alone (it
/// came out `inf` on wasm32). The cluster-outer and node-outer arms must agree bitwise (the compressed
/// clusters go through the same function in both, the fallback ones keep the same
/// per-accumulator order), and the result must be finite (a skipped fallback cluster
/// would leave its sum at 0).
#[test]
fn agq_compressed_with_fallback_cluster_outer_matches_node_outer() {
    for family in [
        Family::Poisson {
            link: PoissonLink::Log,
        },
        Family::Gamma {
            link: GammaLink::Log,
        },
    ] {
        for shifted in [false, true] {
            for nagq in [3u8, 7, 25] {
                let run = |outer: bool| {
                    let (mut ws, x, mut y, ids, _p, n) = fixture_with_nagq(family, "int1", nagq);
                    if shifted {
                        let mut off = vec![0.0f64; n];
                        for i in 0..n {
                            if ids[i] == 0 {
                                off[i] = -205.0;
                                y[i] = if matches!(family, Family::Poisson { .. }) {
                                    (y[i] * (-205f64).exp()).round()
                                } else {
                                    y[i] * (-205f64).exp()
                                };
                            }
                        }
                        ws.offset = Some(off);
                    }
                    ws.params[0] = 0.5;
                    ws.params[1] = 0.2;
                    ws.params[2] = 0.8;
                    let params = ws.params.clone();
                    ws.cluster_rows = if outer {
                        Some(ClusterRowIndex::build(&ids, ws.groupings.n_primary))
                    } else {
                        None
                    };
                    glmm_agq_deviance(&params, &mut ws, x.as_ref(), &y, &ids, n, nagq)
                };
                let (da, db) = (run(true), run(false));
                assert!(
                    da.is_finite(),
                    "{family:?} shifted={shifted} nagq={nagq}: {da}"
                );
                assert_eq!(
                    da.to_bits(),
                    db.to_bits(),
                    "{family:?} shifted={shifted} nagq={nagq}: {da} vs {db}"
                );
            }
        }
    }
}

// --- Vector AGQ (agq_deviance_vec) white-box invariants --

/// q=2 vector-RE spec: intercept + slope on design col 1, single grouping
/// factor, no extras — the shape the widened gate routes to `agq_deviance_vec`.
fn slope1_spec() -> SizedSpec {
    SizedSpec {
        spec: ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        },
        n_primary: 8,
        extra_counts: vec![],
    }
}

/// q=3 vector-RE spec: intercept + slopes on design cols 1,2 (pins the q≤3 cap).
fn slope2_spec() -> SizedSpec {
    SizedSpec {
        spec: ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1, 2],
                extra_groupings: vec![],
            }),
        },
        n_primary: 8,
        extra_counts: vec![],
    }
}

/// No-extras q_p=3 (intercept + 2 slopes) clustered-binary dataset — the vector
/// AGQ q=3 fixture. p=3, 8 clusters, correlated intercept+2 slopes.
fn glmm_slope2_noextra_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>) {
    let (n, nc) = (120usize, 8usize);
    let mut st = 29u64;
    let u0: Vec<f64> = (0..nc).map(|_| 0.6 * lcg(&mut st)).collect();
    let u1: Vec<f64> = (0..nc).map(|_| 0.4 * lcg(&mut st)).collect();
    let u2: Vec<f64> = (0..nc).map(|_| 0.3 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 3);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = i % nc;
        ids[i] = c as u32;
        let x1 = lcg(&mut st);
        let x2 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x2;
        let eta = 0.2 + 0.8 * x1 - 0.5 * x2 + u0[c] + u1[c] * x1 + u2[c] * x2;
        let p = 1.0 / (1.0 + (-eta).exp());
        y[i] = if lcg(&mut st) + 0.5 < p { 1.0 } else { 0.0 };
    }
    (x, y, ids)
}

/// Vector AGQ k=1 ≡ Laplace at q=2 (rel<1e-12): the single z=0 node collapses the
/// multivariate Liu–Pierce bracket to `q·ln√π`, leaving `−2ℓ_c(ũ_c)+log|A_c|` —
/// the Laplace term exactly. Mirrors the scalar k = 1 check in
/// `agq_nb_ladder_converges_and_one_node_is_laplace` one dimension up;
/// run at a non-identity Λ_p (correlated intercept/slope) so a missing `Λ_p·u_cj`
/// or `L_c⁻ᵀ` transform in the integrand would surface.
#[test]
fn agq_vec_k1_reduces_to_laplace_q2() {
    let (xf64, y, ids) = glmm_slope_noextra_dataset();
    let n = y.len();
    let cluster = slope1_spec();
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    ws.params[0] = 0.5; // vech(Λ_p): [σ_int, cov, σ_slope] ≠ I
    ws.params[1] = 0.1;
    ws.params[2] = 0.4;
    ws.params[3] = 0.2; // β
    ws.params[4] = 0.8;
    let p = ws.params.clone();
    let lap = glmm_laplace_deviance(&p, &mut ws, xf64.as_ref(), &y, &ids, &[], n);
    let agq1 = glmm_agq_deviance(&p, &mut ws, xf64.as_ref(), &y, &ids, n, 1);
    assert!(
        (agq1 - lap).abs() <= 1e-12 * lap.abs().max(1.0),
        "agq_vec(k=1) q2 {agq1} vs laplace {lap}"
    );
}

/// Vector AGQ k=1 ≡ Laplace at q=3 (rel<1e-12) — the same reduction on the cap
/// surface (full 3×3 Λ_p, q²=9 transform), proving dimensional generality.
#[test]
fn agq_vec_k1_reduces_to_laplace_q3() {
    let (xf64, y, ids) = glmm_slope2_noextra_dataset();
    let n = y.len();
    let cluster = slope2_spec();
    let mut ws = GlmmWorkspace::for_cluster_spec(3, &cluster, n, &[1, 2], 1);
    // vech(Λ_p) column-major lower-tri of a 3×3, non-identity.
    let theta = [0.6, 0.1, 0.05, 0.4, 0.08, 0.3];
    ws.params[..6].copy_from_slice(&theta);
    ws.params[6] = 0.2; // β
    ws.params[7] = 0.8;
    ws.params[8] = -0.5;
    let p = ws.params.clone();
    let lap = glmm_laplace_deviance(&p, &mut ws, xf64.as_ref(), &y, &ids, &[], n);
    let agq1 = glmm_agq_deviance(&p, &mut ws, xf64.as_ref(), &y, &ids, n, 1);
    assert!(
        (agq1 - lap).abs() <= 1e-12 * lap.abs().max(1.0),
        "agq_vec(k=1) q3 {agq1} vs laplace {lap}"
    );
}

/// Vector AGQ parallel cluster loop ≡ serial bitwise (q=2): disjoint `sum[c]`
/// writes make the cluster-outer result schedule-independent, and the `None`
/// (serial) arm builds the same row-index ordering as the prebuilt `Some` arm.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
#[test]
fn agq_vec_parallel_bit_identical_to_serial_q2() {
    let (xf64, y, ids) = glmm_slope_noextra_dataset();
    let n = y.len();
    let cluster = slope1_spec();
    for nagq in [3u8, 7, 9] {
        let mut ws_a = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], nagq);
        ws_a.params[0] = 0.5;
        ws_a.params[1] = 0.1;
        ws_a.params[2] = 0.4;
        ws_a.params[3] = 0.2;
        ws_a.params[4] = 0.8;
        let params = ws_a.params.clone();
        // Some(idx) ⇒ rayon arm under --features parallel (this test is itself
        // feature-gated), so `da` is the parallel result.
        ws_a.cluster_rows = Some(ClusterRowIndex::build(&ids, ws_a.groupings.n_primary));

        let mut ws_b = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], nagq);
        ws_b.cluster_rows = None; // serial cluster-outer arm (builds a transient index)

        let da = glmm_agq_deviance(&params, &mut ws_a, xf64.as_ref(), &y, &ids, n, nagq);
        let db = glmm_agq_deviance(&params, &mut ws_b, xf64.as_ref(), &y, &ids, n, nagq);
        assert_eq!(da.to_bits(), db.to_bits(), "nagq={nagq}");
    }
}

/// Vector AGQ k-convergence self-consistency (q=2): the product Gauss–Hermite
/// grid converges, so a fixed dataset's deviance at k=9/15/21 agrees to
/// tightening tolerances — high-k is its own deviance-scale reference (no oracle).
#[test]
fn agq_vec_k_convergence_self_consistent_q2() {
    let (xf64, y, ids) = glmm_slope_noextra_dataset();
    let n = y.len();
    let cluster = slope1_spec();
    let theta_beta = [0.5_f64, 0.1, 0.4, 0.2, 0.8];
    let dev_at = |nagq: u8| {
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], nagq);
        ws.params[..5].copy_from_slice(&theta_beta);
        let p = ws.params.clone();
        glmm_agq_deviance(&p, &mut ws, xf64.as_ref(), &y, &ids, n, nagq)
    };
    let d9 = dev_at(9);
    let d15 = dev_at(15);
    let d21 = dev_at(21);
    let scale = d21.abs().max(1.0);
    let gap_9_21 = (d9 - d21).abs();
    let gap_15_21 = (d15 - d21).abs();
    // Already at k=9 the value is close; k=15 is essentially converged against k=21; and the gap
    // must not grow with refinement (slack for the noise floor when all three
    // sit within a few ULP of the converged value).
    assert!(
        gap_9_21 < 1e-4 * scale,
        "k=9 vs k=21 gap {gap_9_21} too large"
    );
    assert!(
        gap_15_21 < 1e-7 * scale,
        "k=15 vs k=21 gap {gap_15_21} not converged"
    );
    assert!(
        gap_15_21 <= gap_9_21 + 1e-12 * scale,
        "gap not tightening: |d15-d21|={gap_15_21} > |d9-d21|={gap_9_21}"
    );
}

#[test]
fn laplace_deviance_collapses_to_glm_at_theta_zero() {
    let (xf64, y, ids) = glmm_intercept_dataset();
    let beta = [0.2_f64, 0.8];
    let cluster = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: 8,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
    ws.params[0] = 0.0;
    ws.params[1] = beta[0];
    ws.params[2] = beta[1];
    let got = glmm_laplace_deviance(
        &ws.params.clone(),
        &mut ws,
        xf64.as_ref(),
        &y,
        &ids,
        &[],
        80,
    );
    let mut d = 0.0;
    for i in 0..80 {
        let eta = beta[0] + beta[1] * xf64[(i, 1)];
        d += if eta > 0.0 {
            eta + (-eta).exp().ln_1p()
        } else {
            eta.exp().ln_1p()
        } - y[i] * eta;
    }
    let want = 2.0 * d;
    assert!(
        (got - want).abs() < 1e-9,
        "collapse: got {got}, want {want}"
    );
}

#[test]
fn packed_cols_populate_every_re_column() {
    let (_xf64, _y, ids, crossed_ids, cluster) = glmm_slope_crossed_dataset();
    let n = ids.len();
    // This model (q_p=2 slope + one intercept-only crossed extra) is
    // structured-eligible, so its packed buffers are zero-length by
    // construction. The test pins `fill_packed_cols`'s column layout, not the
    // structured deviance path, so it builds the workspace on the packed layout.
    let mut ws = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[1], 1);
    fill_packed_cols(&mut ws, &ids, &[crossed_ids], n);
    let width = ws.packed.width;
    // rows[c] = the rows whose packed nonzeros touch RE column c.
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); ws.k];
    for i in 0..n {
        for t in i * width..(i + 1) * width {
            rows[ws.packed.m_cols[t] as usize].push(i);
        }
    }
    assert!(
        rows.iter().all(|r| !r.is_empty()),
        "every RE column must be populated — offset wiring"
    );

    // "some row" can't catch an offset swap between two same-width groupings
    // (e.g. crossed level 1 landing in level 2's slot) — pin the EXACT row set
    // per column instead. Packed layout: the primary block is component-major
    // (`c·n_prim + lvl`, n_prim=8 levels at q=2, so cols 0..7 are intercepts and
    // 8..15 slopes); the crossed block starts at the absolute offset
    // `n_prim*q = 16`, one column per of its 4 levels.
    // `glmm_slope_crossed_dataset` sets `ids[i] = i % 8`, `crossed[i] = i % 4`.
    let n_prim = 8usize;
    let n_crossed = 4usize;
    for lvl in 0..n_prim {
        let expect_rows: Vec<usize> = (0..n).filter(|&i| i % n_prim == lvl).collect();
        for (kind, col) in [("intercept", lvl), ("slope", n_prim + lvl)] {
            assert_eq!(
                rows[col], expect_rows,
                "primary col {col} (level {lvl}, {kind}) nonzero rows"
            );
        }
    }
    for cc in 0..n_crossed {
        let col = n_prim * 2 + cc;
        let expect_rows: Vec<usize> = (0..n).filter(|&i| i % n_crossed == cc).collect();
        assert_eq!(
            rows[col], expect_rows,
            "crossed col {col} (level {cc}) nonzero rows"
        );
    }
}

/// The packed `M` fill must scale each extra grouping's slot by ITS OWN θ, even
/// when `extra_offsets` is non-monotonic. A crossed-before-nested declaration
/// places the nested block at the low `prim_width` slot (small offset) AFTER the
/// crossed block (large offset), so a fill that walked offsets in address order
/// would pair each slot with the wrong θ. `lam_off_decl` is what keeps the two
/// in declaration order; this is its regression guard.
#[test]
fn packed_m_vals_scale_each_extra_by_its_own_theta() {
    let (n_prim, n_crossed, n_per_parent) = (4usize, 3usize, 2usize);
    // Declaration order [Crossed, Nested] ⇒ extra_offsets[0] (crossed, high) >
    // extra_offsets[1] (nested, low) — the non-monotonic precondition.
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![
                    Grouping {
                        terms: Vec::new(),
                        relation: GroupingRelation::Crossed,
                        slopes: vec![],
                    },
                    Grouping {
                        terms: Vec::new(),
                        relation: GroupingRelation::NestedWithin,
                        slopes: vec![],
                    },
                ],
            }),
        },
        n_primary: n_prim,
        extra_counts: vec![n_crossed, n_per_parent],
    };
    let n = 8usize;
    let ids: Vec<u32> = (0..n).map(|i| (i % n_prim) as u32).collect();
    let crossed: Vec<u32> = (0..n).map(|i| (i % n_crossed) as u32).collect();
    // Nested ids are GLOBAL and dense over all parents: child `i / n_prim` of
    // parent `ids[i]`.
    let nested: Vec<u32> = (0..n)
        .map(|i| ids[i] * n_per_parent as u32 + (i / n_prim) as u32)
        .collect();
    // q_p = 1 ⇒ base_theta = 1; θ_crossed at idx 1, θ_nested at idx 2.
    let base_theta = 1usize;
    let (theta_crossed, theta_nested) = (2.0_f64, 3.0_f64);
    let mut ws = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[], 1);
    let mut params = vec![0.0; ws.n_theta];
    params[0] = 1.0; // primary Λ (irrelevant to the extra blocks)
    params[base_theta] = theta_crossed;
    params[base_theta + 1] = theta_nested;
    assert!(
        ws.groupings.extra_offsets[0] > ws.groupings.extra_offsets[1],
        "fixture must produce non-monotonic offsets, got {:?}",
        ws.groupings.extra_offsets
    );
    fill_packed_cols(&mut ws, &ids, &[crossed.clone(), nested.clone()], n);
    // Every grouping here is intercept-only (q = 1), so `x` is never read.
    let x = Mat::<f64>::zeros(n, 2);
    packed_oracle_fill(&mut ws, x.as_ref(), &params, n);
    // Row slots, in declaration order: [primary | crossed | nested].
    let width = ws.packed.width;
    assert_eq!(width, 3);
    let (coff, noff) = (ws.groupings.extra_offsets[0], ws.groupings.extra_offsets[1]);
    for i in 0..n {
        let base = i * width;
        assert_eq!(
            ws.packed.m_cols[base + 1] as usize,
            coff + crossed[i] as usize,
            "row {i} crossed column"
        );
        assert!(
            (ws.packed.m_vals[base + 1] - theta_crossed).abs() < 1e-12,
            "row {i} crossed value = {}, want {theta_crossed}",
            ws.packed.m_vals[base + 1]
        );
        assert_eq!(
            ws.packed.m_cols[base + 2] as usize,
            noff + nested[i] as usize,
            "row {i} nested column"
        );
        assert!(
            (ws.packed.m_vals[base + 2] - theta_nested).abs() < 1e-12,
            "row {i} nested value = {}, want {theta_nested}",
            ws.packed.m_vals[base + 2]
        );
    }
}

#[derive(serde::Deserialize)]
struct HessianFixture {
    n: usize,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    cluster_ids: Vec<u32>,
    theta: f64,
    beta: Vec<f64>,
    vcov_hessian: Vec<Vec<f64>>,
    vcov_rx: Vec<Vec<f64>>,
}

/// The n=96 / 12-cluster binomial fixture and its glmer reference, both written by
/// `tests/fixtures/gen_glmm_hessian_vcov.R` (data from `set.seed(2137)`), registered
/// in `validation/manifest.json`, `in_crate_references`.
fn load_hessian_fixture() -> HessianFixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/glmm_hessian_vcov.json"
    );
    let s = std::fs::read_to_string(path).expect("read hessian fixture");
    serde_json::from_str(&s).expect("parse hessian fixture")
}

/// The joint-Hessian fixed-effect covariance (`joint_hessian_cov`, which takes
/// its exact arm on this shape) matches lme4 `vcov(use.hessian = TRUE)`
/// on the committed n=96 / 12-cluster `y ~ x1 + (1|grp)` glmer fit. Runs OUR
/// `fit_glmm` to convergence (the production code path) and takes the Hessian
/// at OUR own θ̂/β̂ — NOT at lme4's fixture params. Pins both the kernel
/// convention and the load-bearing factor of 2 (deviance = −2logL ⇒
/// cov = 2·inv(H_dev)).
///
/// Why fit first: the joint Hessian's β-block invariance
/// to the θ↔β cross-curvature only holds AT a θ-stationary point. Evaluating at
/// our own converged θ̂ both (a) reflects production and (b) lets us assert our
/// solver agrees with lme4's θ̂ — proving any residual vcov gap is genuine
/// curvature, not an off-stationarity artifact. Measured: our θ̂ matches lme4's
/// to ~0.07% and β̂ to ~0.1%; against the artifact-free fixture the vcov gap is
/// ~3.4e-7 (see the band comment at the assertion).
#[test]
fn joint_hessian_cov_matches_glmer_use_hessian_true() {
    let fx = load_hessian_fixture();
    let n = fx.n;
    let p = fx.beta.len();
    let n_clusters = fx.cluster_ids.iter().max().unwrap() + 1;
    let cluster = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: n_clusters as usize,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[], 1);
    let mut xf64 = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            xf64[(i, j)] = fx.x[i][j];
        }
    }
    let y = fx.y.clone();
    let ids = fx.cluster_ids.clone();

    // RX/Schur machinery-exactness check, evaluated at lme4's EXACT θ̂/β̂ so the
    // comparison is point-matched: rx_cov_into reproduces lme4 vcov(use.hessian
    // =FALSE) to ~3.6e-6 (this path never goes through numDeriv — it inverts the
    // closed-form β information). Pins that the PIRLS / W̃ / Schur machinery the
    // joint Hessian shares is exact, isolating any vcov_hessian gap to the
    // Hessian ↔ numDeriv comparison. Requires a converged central PIRLS at those params.
    ws.params[0] = fx.theta;
    for j in 0..p {
        ws.params[1 + j] = fx.beta[j];
    }
    let _ = laplace_deviance_at(
        &mut ws,
        xf64.as_ref(),
        &y,
        &ids,
        &[],
        n,
        &mut crate::counters::EvalCounters::new(),
    );
    let mut rx = Mat::<f64>::zeros(p, p);
    assert!(rx_cov_into(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &ids,
        p,
        n,
        &mut rx
    ));
    for i in 0..p {
        for j in 0..p {
            let (got, want) = (rx[(i, j)], fx.vcov_rx[i][j]);
            assert!(
                (got - want).abs() < 1e-5,
                "rx[{i}][{j}] got {got} want {want} (gap {})",
                (got - want).abs()
            );
        }
    }

    // Production path: converge OUR fit (params overwritten with our θ̂/β̂).
    let fit = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[1u32],
        None,
        &vec![0.0; p],
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fit.converged, "fit_glmm must converge on the fixture");
    // Our solver vs lme4: θ̂ to ~0.07%, β̂ to ~0.1% (measured). Proves the two
    // optimisers land on the same stationary point, so the joint Hessian below is
    // taken at a genuine θ-stationary point — the residual vcov gap is NOT an
    // off-stationarity artifact. Tol = a few % (well above the achieved band);
    // a MATERIAL divergence here would be a real engine finding, not noise.
    assert!(
        (ws.params[0] - fx.theta).abs() / fx.theta < 0.01,
        "our θ̂ {} vs lme4 θ̂ {} ({}% rel)",
        ws.params[0],
        fx.theta,
        100.0 * (ws.params[0] - fx.theta).abs() / fx.theta
    );
    for j in 0..p {
        assert!(
            (ws.params[1 + j] - fx.beta[j]).abs() < 5e-3,
            "our β̂[{j}] {} vs lme4 β̂[{j}] {} (gap {})",
            ws.params[1 + j],
            fx.beta[j],
            (ws.params[1 + j] - fx.beta[j]).abs()
        );
    }
    let our_theta = ws.params[0];

    let mut cov = Mat::<f64>::zeros(p, p);
    let status = joint_hessian_cov(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        p,
        n,
        &mut cov,
    );
    assert_eq!(status, FdHessianStatus::Ok);
    // ws.params restored to OUR converged snapshot on return.
    assert!((ws.params[0] - our_theta).abs() < 1e-15);
    // Achieved FD-vs-lme4(use.hessian=TRUE) band at OUR converged fit: worst
    // entry gap ~3.4e-7. The fixture is generated artifact-free
    // (gen_glmm_hessian_vcov.R, glmer tolPwrss = 1e-13 — at lme4's default
    // 1e-7 its ldL2 uses working weights one PIRLS iteration behind the mode
    // and vcov(use.hessian=TRUE) carried ~1.7e-3 of spurious θ/θβ curvature,
    // which this band absorbs; see joint_hessian_cov's doc comment) and
    // our FD runs PIRLS at `pirls_tol_fd` — never looser than the fit's own exit
    // tolerance and capped at PIRLS_TOL_REL_FD — so what remains is the
    // two solvers' θ̂ offset plus FD truncation. tol = achieved band + ~30×
    // margin (cross-platform FP headroom); it pins the convention + the
    // load-bearing factor of 2. Mirrors hessian_mode_t_sq_uses_joint_hessian_cov's
    // band — change together.
    let tol = 1e-5;
    for i in 0..p {
        for j in 0..p {
            let (got, want) = (cov[(i, j)], fx.vcov_hessian[i][j]);
            assert!(
                (got - want).abs() < tol,
                "vcov[{i}][{j}] got {got} want {want} (gap {})",
                (got - want).abs()
            );
        }
    }
}

/// The exact joint Hessian and the FD stencil agree on the committed
/// n=96 / 12-cluster `y ~ x1 + (1|grp)` fixture, at Laplace and at nAGQ = 7.
/// The two sides are the SAME function's second derivative computed two ways,
/// so the gap is the FD stencil's own truncation-plus-noise error and nothing
/// else — this is the test that says the exact arm did not change the meaning
/// of the quantity, only its accuracy. Bands: the `validation/grid/tol.R`
/// `se_hessian_rel` default (1e-3) at Laplace and `AGQ_SE_HESSIAN_REL` (2e-2,
/// `tests/oracle_support/mod.rs`) at nAGQ = 7, the order rungs 44/45 carry in
/// `validation/manifest.json`. The AGQ gate (`deviance.rs`: no extras,
/// `q_p ≤ 3`, binomial/Poisson) sits entirely inside the blocked branch, so at
/// nAGQ = 7 the exact Hessian differentiates the AGQ deviance rather than the
/// Laplace one.
#[test]
fn exact_hessian_matches_fd_on_fixture() {
    let fx = load_hessian_fixture();
    let n = fx.n;
    let p = fx.beta.len();
    let n_clusters = fx.cluster_ids.iter().max().unwrap() + 1;
    let cluster = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: n_clusters as usize,
        extra_counts: vec![],
    };
    let mut xf64 = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            xf64[(i, j)] = fx.x[i][j];
        }
    }
    let y = fx.y.clone();
    let ids = fx.cluster_ids.clone();

    for (nagq, band) in [(1u8, 1e-3), (7, 2e-2)] {
        let mut cov_exact = Mat::<f64>::zeros(p, p);
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[], nagq);
        // Rx: the fit itself runs no joint Hessian; the explicit
        // joint_hessian_cov calls below are the comparison.
        let fit = fit_glmm(
            &mut ws,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &y,
            &ids,
            &[],
            &[1u32],
            None,
            &vec![0.0; p],
            n,
            WaldSe::Rx,
            ColumnScaling::IDENTITY,
        );
        assert!(fit.converged, "nagq={nagq}");
        // Cleared first so the guard below has teeth: each exact arm sizes its
        // own scratch and the FD stencil touches none. With both calls on the
        // stencil the two sides agree bitwise and the band passes vacuously. On
        // an AGQ shape the assembled pass declines before allocating, so only
        // the gradient-difference rung (this logit fixture) or the hyper-dual
        // pass can take it.
        ws.hyper_scratch = None;
        ws.dual_scratch = None;
        ws.agq_gd = None;
        let st = joint_hessian_cov(
            &mut ws,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &y,
            &ids,
            &[],
            p,
            n,
            &mut cov_exact,
        );
        assert_eq!(st, FdHessianStatus::Ok, "nagq={nagq}");
        let exact_arm_ran = if nagq == 1 {
            ws.hyper_scratch.is_some() || ws.dual_scratch.is_some()
        } else {
            ws.agq_gd.is_some() || ws.hyper_scratch.is_some()
        };
        assert!(
            exact_arm_ran,
            "nagq={nagq}: joint_hessian_cov fell through to the FD stencil"
        );
        let tse_exact = theta_se(&ws);

        let mut cov_fd = Mat::<f64>::zeros(p, p);
        ws.fd.force_fd_hessian = true;
        let st = joint_hessian_cov(
            &mut ws,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &y,
            &ids,
            &[],
            p,
            n,
            &mut cov_fd,
        );
        assert_eq!(st, FdHessianStatus::Ok, "nagq={nagq}");
        let tse_fd = theta_se(&ws);

        // The band applies to the SEs the caller sees, not to the raw
        // covariance entries.
        for j in 0..p {
            let (a, b) = (cov_exact[(j, j)].sqrt(), cov_fd[(j, j)].sqrt());
            assert!(
                (a - b).abs() <= band * b.abs(),
                "nagq={nagq} se[{j}]: exact {a} vs fd {b}"
            );
        }
        for k in 0..ws.n_theta {
            assert!(
                (tse_exact[k] - tse_fd[k]).abs() <= band * tse_fd[k].abs(),
                "nagq={nagq} theta_se[{k}]: exact {} vs fd {}",
                tse_exact[k],
                tse_fd[k]
            );
        }
    }
}

/// The exact hyper-dual Hessian and the FD stencil agree on the structured
/// extras path, to `validation/grid/tol.R`'s `se_hessian_rel` default band (1e-3).
/// Three regimes, because they take three different tails: nested-only
/// (`e = 0`, no tail), crossed with a scalar rank-1 downdate (`qc == 1`), and
/// nested+crossed (`qc > 1`, whose f64 route is the panel downdate and whose
/// dual route is the scalar default). The two sides are the same function's
/// second derivative computed two ways; the gap is the FD stencil's own
/// truncation-plus-noise error.
///
/// Each regime carries its own `(n_primary, n)`, chosen so every θ̂ lands well
/// inside the box (≥0.25 here), for a reason the band cannot absorb: the
/// stencil's θ step is ABSOLUTE (`FD_STEP_BASE`), so its truncation error in
/// the θ block grows as θ̂ shrinks. Measured on this fixture family, the θ SE
/// gap runs 2e-5 at θ̂ = 0.41, 1.1e-3 at θ̂ = 0.13 and 3.3e-3 at θ̂ = 0.08 — the
/// stencil moving, not the exact side. The β SEs, which are what the
/// validation dump's `se_hessian` reports, stay under 1e-4 across all of them.
/// `pinned_crossed_theta_gets_the_exact_hessian` covers θ̂ = 0 exactly, with
/// its own measured θ band.
#[test]
fn exact_hessian_matches_fd_on_structured_fixture() {
    for (np, ncr, n_prim, n_rows, label) in [
        (2usize, 0usize, 16usize, 320usize, "nested2"),
        (0, 6, 20, 200, "crossed6"),
        (2, 6, 18, 180, "nested2_crossed6"),
    ] {
        let (xf64, y, ids, extra_ids, spec) = glmm_extras_q1_dataset_sized(np, ncr, n_prim, n_rows);
        let n = y.len();
        let p = 2usize;
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &spec, n, &[], 1);
        assert!(
            ws.groupings.structured_extras_eligible(),
            "{label}: fixture must route through the structured extras path"
        );
        // Built so the f64 FD side runs the production cached tail (`None` on the
        // nested-only cell, where `e == 0` skips the tail altogether).
        ws.pattern.structured_schur = StructuredSchur::new(&ws.groupings, &ids, &extra_ids, n);
        // Rx: mirrors `exact_hessian_matches_fd_on_fixture` — the fit itself
        // runs no joint Hessian; the explicit joint_hessian_cov calls below are
        // the comparison.
        let fit = fit_glmm(
            &mut ws,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &y,
            &ids,
            &extra_ids,
            &[1u32],
            None,
            &vec![0.0; p],
            n,
            WaldSe::Rx,
            ColumnScaling::IDENTITY,
        );
        assert!(fit.converged, "{label}: fixture must converge");
        assert!(
            ws.params[..ws.n_theta].iter().all(|v| *v > 0.0),
            "{label}: θ̂ = {:?} touches the pin — see this test's doc comment",
            &ws.params[..ws.n_theta]
        );

        // Cleared first so the assertion below has teeth: each exact arm sizes
        // its own dual scratch — the assembled pass `dual_scratch`, the
        // hyper-dual pass `hyper_scratch` — and the FD stencil touches
        // neither, which is what separates the two arms here. With both calls
        // on the stencil the two sides agree BITWISE and every band below
        // passes vacuously.
        ws.hyper_scratch = None;
        ws.dual_scratch = None;
        let mut cov_exact = Mat::<f64>::zeros(p, p);
        let st = joint_hessian_cov(
            &mut ws,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            &mut cov_exact,
        );
        assert_eq!(st, FdHessianStatus::Ok, "{label}");
        assert!(
            ws.hyper_scratch.is_some() || ws.dual_scratch.is_some(),
            "{label}: joint_hessian_cov fell through to the FD stencil"
        );
        let tse_exact = theta_se(&ws);

        let mut cov_fd = Mat::<f64>::zeros(p, p);
        ws.fd.force_fd_hessian = true;
        let st = joint_hessian_cov(
            &mut ws,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            &mut cov_fd,
        );
        assert_eq!(st, FdHessianStatus::Ok, "{label}");
        let tse_fd = theta_se(&ws);
        ws.fd.force_fd_hessian = false;

        // Band = grid/tol.R's `se_hessian_rel` default (1e-3), applied to the SEs the
        // caller sees, not to the raw covariance entries.
        for j in 0..p {
            let (a, b) = (cov_exact[(j, j)].sqrt(), cov_fd[(j, j)].sqrt());
            assert!(
                (a - b).abs() <= 1e-3 * b.abs(),
                "{label} se[{j}]: exact {a} vs fd {b}"
            );
        }
        for k in 0..ws.n_theta {
            assert!(
                (tse_exact[k] - tse_fd[k]).abs() <= 1e-3 * tse_fd[k].abs(),
                "{label} theta_se[{k}]: exact {} vs fd {}",
                tse_exact[k],
                tse_fd[k]
            );
        }
    }
}

/// A structured extras fit whose crossed θ̂ pins to 0 still gets the EXACT
/// hyper-dual Hessian, and both boundary diagnostics with it.
///
/// `build_packed_m`'s pin skip and the coupling-CSR pin mask drop a crossed
/// grouping at θ = 0 — but only at `T = f64`, where the column is a pure size
/// win. At a dual `T` the column is kept, so the lane seeded on that θ has
/// something to differentiate instead of returning an all-zero Hessian row.
/// This is the direct anti-regression for that: before the `T::IS_F64` guards
/// the crossed Hessian row was exactly `[0, 0, 0, 0]`, the joint Hessian was
/// singular, and both entry points refused the point.
///
/// `glmm_extras_q1_dataset(0, 6)` at its committed `n = 96` is the cell that
/// pins: θ̂ = [0.271, 0.0]. That is why
/// `exact_hessian_matches_fd_on_structured_fixture` had to enlarge it.
///
/// Fit at `WaldSe::Rx` so an in-fit SE arm cannot pre-empt the two explicit
/// `joint_hessian_cov` calls below, which are the comparison.
#[test]
fn pinned_crossed_theta_gets_the_exact_hessian() {
    let (xf64, y, ids, extra_ids, spec) = glmm_extras_q1_dataset(0, 6);
    let n = y.len();
    let p = 2usize;
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &spec, n, &[], 1);
    ws.pattern.structured_schur = StructuredSchur::new(&ws.groupings, &ids, &extra_ids, n);
    let fit = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &extra_ids,
        &[1u32],
        None,
        &vec![0.0; p],
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fit.converged, "pinned-crossed fixture must converge");
    let ti = ws.groupings.crossed[0].vech_start;
    assert_eq!(
        ws.params[ti],
        0.0,
        "fixture must pin its crossed θ — θ̂ = {:?}",
        &ws.params[..ws.n_theta]
    );
    assert!(
        !fit.hessian_fallback,
        "pinned-crossed fit must not fall back to Rx"
    );
    assert!(
        ws.inference.var_diag[1].is_finite() && ws.inference.var_diag[1] > 0.0,
        "var_diag[1] = {}",
        ws.inference.var_diag[1]
    );

    // The exact kernel accepts the point now. Read the row directly, because
    // that is the shape the bug destroyed.
    let m = ws.n_theta + p;
    let mut grad = vec![0.0f64; m];
    let mut hess = Mat::<f64>::zeros(m, m);
    let st = derivative::laplace_hessian(
        &mut ws,
        xf64.as_ref(),
        &y,
        &ids,
        &extra_ids,
        p,
        n,
        &mut grad,
        &mut hess,
    );
    assert!(
        matches!(st, DerivStatus::Ok(_)),
        "laplace_hessian must accept a pinned crossed θ̂"
    );
    assert!(
        hess[(ti, ti)].is_finite() && hess[(ti, ti)] > 0.0,
        "crossed Hessian diagonal = {} (was exactly 0)",
        hess[(ti, ti)]
    );
    assert!(
        (0..m).any(|j| hess[(ti, j)] != 0.0),
        "crossed Hessian row is identically zero"
    );
    // The deviance is even in θ, so the true gradient entry at θ = 0 is 0; what
    // is left is the PIRLS solve's own noise, measured at ~1.8e-6 on this
    // fixture.
    assert!(
        grad[ti].abs() <= 1e-6,
        "crossed gradient entry {} above the PIRLS noise floor",
        grad[ti]
    );

    // Each exact arm sizes its own dual scratch — the assembled pass
    // `dual_scratch`, the hyper-dual pass `hyper_scratch` — and the FD stencil
    // touches neither, so the pair is what separates the two arms below; the
    // bands alone cannot, since two stencil runs agree bitwise.
    ws.hyper_scratch = None;
    ws.dual_scratch = None;
    let mut cov_exact = Mat::<f64>::zeros(p, p);
    let st = joint_hessian_cov(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &extra_ids,
        p,
        n,
        &mut cov_exact,
    );
    assert_eq!(st, FdHessianStatus::Ok);
    assert!(
        ws.hyper_scratch.is_some() || ws.dual_scratch.is_some(),
        "joint_hessian_cov fell through to the FD stencil"
    );
    let hess_exact_tt = ws.inference.hess_scratch[(ti, ti)];

    let mut cov_fd = Mat::<f64>::zeros(p, p);
    ws.fd.force_fd_hessian = true;
    let st = joint_hessian_cov(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &extra_ids,
        p,
        n,
        &mut cov_fd,
    );
    assert_eq!(st, FdHessianStatus::Ok);
    let hess_fd_tt = ws.inference.hess_scratch[(ti, ti)];
    ws.fd.force_fd_hessian = false;

    // β band = grid/tol.R's `se_hessian_rel` default (1e-3), on the SEs the caller
    // sees rather than the raw covariance entries.
    for j in 0..p {
        let (a, b) = (cov_exact[(j, j)].sqrt(), cov_fd[(j, j)].sqrt());
        assert!(
            (a - b).abs() <= 1e-3 * b.abs(),
            "se[{j}]: exact {a} vs fd {b}"
        );
    }
    // The θ diagonal gets its OWN band, measured here rather than borrowed from
    // `se_hessian_rel`: the stencil's θ step is absolute (`FD_STEP_BASE`), so
    // its θ-block truncation error is at its worst at θ̂ = 0. Measured on this
    // fixture 2026-09-05: exact 33.342022283730216, FD 33.33806766931957, a
    // relative gap of 1.19e-4. Banded at 5e-4 — about four times the measured
    // gap, and the whole computation is deterministic on a fixed fixture, so
    // the margin is for a different microarchitecture, not for run-to-run
    // noise.
    assert!(
        (hess_exact_tt - hess_fd_tt).abs() <= 5e-4 * hess_fd_tt.abs(),
        "crossed hessian diagonal: exact {hess_exact_tt} vs fd {hess_fd_tt}"
    );
}

/// Run `joint_hessian_cov` twice on the same converged workspace — serial
/// (`parallel_inner = false`) then parallel (`= true`) — and assert every returned
/// covariance entry, θ-block SE, and `FdHessianStatus` is BITWISE equal. `.to_bits()`
/// comparison so a NaN (RX-fallback θ SE) matches a NaN. A mismatch means a
/// per-thread worker workspace (`fd_worker_ws`) missed a field the deviance reads.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
fn assert_fd_hessian_serial_eq_parallel(
    ws: &mut GlmmWorkspace,
    x: faer::MatRef<f64>,
    y: &[f64],
    ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
) {
    let mut cov_s = Mat::<f64>::zeros(p, p);
    ws.parallel_inner = false;
    let st_s = joint_hessian_cov(ws, x, x.subcols(0, 0), y, ids, extra_ids, p, n, &mut cov_s);
    let tse_s = theta_se(ws);

    let mut cov_p = Mat::<f64>::zeros(p, p);
    ws.parallel_inner = true;
    let st_p = joint_hessian_cov(ws, x, x.subcols(0, 0), y, ids, extra_ids, p, n, &mut cov_p);
    let tse_p = theta_se(ws);

    assert_eq!(st_s, st_p, "FdHessianStatus differs serial vs parallel");
    assert_eq!(
        st_s,
        FdHessianStatus::Ok,
        "the stencil must run, not fall back"
    );
    for i in 0..p {
        for j in 0..p {
            assert_eq!(
                cov_s[(i, j)].to_bits(),
                cov_p[(i, j)].to_bits(),
                "cov[{i}][{j}] not bit-identical: serial {} parallel {}",
                cov_s[(i, j)],
                cov_p[(i, j)]
            );
        }
    }
    assert_eq!(tse_s.len(), tse_p.len());
    for (k, (&a, &b)) in tse_s.iter().zip(tse_p.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "theta_se[{k}] not bit-identical: {a} vs {b}"
        );
    }
}

/// The parallel FD-Hessian grid must reproduce the serial one BITWISE on the
/// no-extras blocked path, the packed-row path and the structured crossed path. Every grid cell is a
/// pure function of the frozen (fd_saved, fd_steps, u_seed) seed, so per-thread
/// workspaces change nothing — a mismatch is an `fd_worker_ws` field-liveness bug,
/// not floating-point noise.
///
/// Runs the FD arm explicitly (`force_fd_hessian`): both fixtures here take an
/// exact Hessian otherwise, which has no rayon in it and is bit-identical by
/// construction. The grid's order-independence is still a live property on the
/// cells that keep a stencil — a packed-row fit the assembled pass declined,
/// and a cell both exact rungs declined — and that is what this test stands in
/// for.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
#[test]
fn fd_hessian_parallel_bit_identical_to_serial() {
    // Case A: no-extras (1|grp) fixture — the blocked FD path.
    {
        let fx = load_hessian_fixture();
        let n = fx.n;
        let p = fx.beta.len();
        let n_clusters = fx.cluster_ids.iter().max().unwrap() + 1;
        let cluster = SizedSpec {
            spec: logit_intercept_spec(),
            n_primary: n_clusters as usize,
            extra_counts: vec![],
        };
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[], 1);
        let mut xf64 = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xf64[(i, j)] = fx.x[i][j];
            }
        }
        let y = fx.y.clone();
        let ids = fx.cluster_ids.clone();
        let fit = fit_glmm(
            &mut ws,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &y,
            &ids,
            &[],
            &[1u32],
            None,
            &vec![0.0; p],
            n,
            WaldSe::Rx,
            ColumnScaling::IDENTITY,
        );
        assert!(fit.converged, "no-extras fixture must converge");
        ws.fd.force_fd_hessian = true;
        assert_fd_hessian_serial_eq_parallel(&mut ws, xf64.as_ref(), &y, &ids, &[], p, n);
    }
    // Case C: the same fixture on the forced packed-row layout — the packed
    // stencil is the one a declined assembled pass falls back to.
    {
        let fx = load_hessian_fixture();
        let n = fx.n;
        let p = fx.beta.len();
        let n_clusters = fx.cluster_ids.iter().max().unwrap() + 1;
        let cluster = SizedSpec {
            spec: logit_intercept_spec(),
            n_primary: n_clusters as usize,
            extra_counts: vec![],
        };
        let mut ws = GlmmWorkspace::for_cluster_spec_packed(p, &cluster, n, &[], 1);
        let mut xf64 = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xf64[(i, j)] = fx.x[i][j];
            }
        }
        let y = fx.y.clone();
        let ids = fx.cluster_ids.clone();
        fill_packed_cols(&mut ws, &ids, &[], n);
        let fit = fit_glmm(
            &mut ws,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &y,
            &ids,
            &[],
            &[1u32],
            None,
            &vec![0.0; p],
            n,
            WaldSe::Rx,
            ColumnScaling::IDENTITY,
        );
        assert!(fit.converged, "packed fixture must converge");
        assert_eq!(ws.layout, GlmmLayout::Packed);
        ws.fd.force_fd_hessian = true;
        assert_fd_hessian_serial_eq_parallel(&mut ws, xf64.as_ref(), &y, &ids, &[], p, n);
    }
    // Case B: structured crossed (grouseticks INDEX primary + BROOD, LOCATION) —
    // exercises the per-thread StructuredSchur rebuild in `fd_worker_ws`.
    {
        let (model, ids, x, y, n, p) = grouseticks_3crossed_fixture();
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
        assert!(
            ws.groupings.structured_extras_eligible(),
            "grouseticks 3-crossed must route through the structured extras path"
        );
        ws.pattern.structured_schur =
            StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n);
        let fit = fit_glmm(
            &mut ws,
            x.as_ref(),
            x.as_ref().subcols(0, 0),
            &y,
            &ids.primary,
            &ids.extra,
            &[0u32, 1, 2, 3],
            None,
            &vec![0.0; p],
            n,
            WaldSe::Rx,
            ColumnScaling::IDENTITY,
        );
        assert!(
            fit.converged,
            "grouseticks structured fixture must converge"
        );
        ws.fd.force_fd_hessian = true;
        assert_fd_hessian_serial_eq_parallel(
            &mut ws,
            x.as_ref(),
            &y,
            &ids.primary,
            &ids.extra,
            p,
            n,
        );
    }
}

/// `fit_glmm(.., WaldSe::Hessian)` sources the per-fit marginal SE from
/// `joint_hessian_cov` (glmer `use.hessian = TRUE`) instead of the Schur
/// forward-solve: on the committed fixture the x1 Hessian SE EXCEEDS the Rx
/// SE and matches the fixture's `vcov_hessian` diagonal. `WaldSe::Rx` keeps
/// the Schur path. Pins that the dispatch reads the joint-Hessian
/// covariance into `ws.inference.var_diag` end-to-end (not just the standalone kernel).
#[test]
fn hessian_mode_t_sq_uses_joint_hessian_cov() {
    let fx = load_hessian_fixture();
    let n = fx.n;
    let p = fx.beta.len();
    let n_clusters = fx.cluster_ids.iter().max().unwrap() + 1;
    let cluster = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: n_clusters as usize,
        extra_counts: vec![],
    };
    let mut xf64 = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            xf64[(i, j)] = fx.x[i][j];
        }
    }
    let y = fx.y.clone();
    let ids = fx.cluster_ids.clone();
    let t1 = 1usize; // x1 column

    let mut ws_h = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[], 1);
    let fit_h = fit_glmm(
        &mut ws_h,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[1u32],
        None,
        &vec![0.0; p],
        n,
        WaldSe::Hessian,
        ColumnScaling::IDENTITY,
    );
    assert!(fit_h.converged, "hessian-mode fit must converge");
    let se_h = ws_h.inference.var_diag[t1].sqrt();

    let mut ws_rx = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[], 1);
    let fit_rx = fit_glmm(
        &mut ws_rx,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[1u32],
        None,
        &vec![0.0; p],
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fit_rx.converged, "rx-mode fit must converge");
    let se_rx = ws_rx.inference.var_diag[t1].sqrt();

    assert!(se_h > se_rx, "hessian SE {se_h} must exceed rx SE {se_rx}");
    // Match the joint-Hessian kernel band (1e-5 ABSOLUTE on the covariance entries,
    // achieved ~3.4e-7 against the artifact-free fixture) on the variance the
    // dispatch wrote into ws.inference.var_diag — the band source is
    // `joint_hessian_cov_matches_glmer_use_hessian_true`: change together.
    let want_var = fx.vcov_hessian[t1][t1];
    assert!(
        (ws_h.inference.var_diag[t1] - want_var).abs() < 1e-5,
        "hessian var {} must match fixture vcov_hessian diag {want_var}",
        ws_h.inference.var_diag[t1]
    );
}

/// A non-PD joint (θ,β) Hessian makes `joint_hessian_cov` fall back to the
/// RX/Schur covariance (`NonPdFellBackToRx`) with the produced cov equal to
/// `rx_cov_into`'s, while the β-only Schur stays PD.
///
/// Where non-PD genuinely lives: NOT at the θ→0 floor (the intercept-only
/// Laplace deviance is EVEN in θ, so its θ-curvature there is structurally
/// POSITIVE), and NOT at a converged high-variance fit either — an interior
/// minimum has PSD curvature by definition. The deviance is genuinely concave
/// BEYOND the minimum along θ: at large θ the Laplace `log|A|` term grows
/// like `2s·log θ` (concave) while the data deviance saturates and the
/// penalty vanishes, so H_θθ < 0 there — tolerance-independent. The test
/// therefore evaluates the joint Hessian (its exact arm on this shape; no FD
/// stencil is forced) at θ pushed well past the converged
/// θ̂ of a high-ICC fit; production reaches this fallback only through such
/// degenerate geometry, kept as a defensive guard. The β fixed-effect
/// information (`rx_cov_into`'s Schur, computed from W̃ at the conditional
/// modes — a different matrix than the joint Hessian) stays PD throughout. The
/// β design is well-conditioned
/// (intercept + a within-cluster-varying continuous x1, NOT separable, NOT a
/// duplicate column) — only the θ/variance direction is degenerate.
///
/// Confirmed branch (asserted below): the FULLY-ASSEMBLED joint Hessian is
/// finite yet fails LLT — i.e. the non-PD-Hessian fallback, NOT the
/// non-finite-deviance fallback (this kernel's log1pexp deviance + `+I` ridge
/// keep every perturbed deviance finite, so the non-finite branch is effectively
/// unreachable with well-posed data).
#[test]
fn fd_hessian_non_pd_falls_back_to_rx() {
    let (n, nc) = (320usize, 8usize);
    let per = n / nc;
    // Strong between-cluster intercept offsets (SD ≈ 5.2) ⇒ high ICC ⇒ the fit
    // converges to a LARGE τ̂² (θ̂ ≳ 5), the concave region where the joint
    // Hessian goes non-PD. Noisy (logistic-sampled) labels — no separation.
    // Much larger offsets, or fewer rows a cluster, make the outer clusters all
    // 0 or all 1: the deviance then keeps falling as θ grows and has no minimum.
    let mut st = 4242u64;
    let mut xf64 = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = i / per; // block layout
        ids[i] = c as u32;
        let u_c = 8.0 * (2.0 * (c as f64) / ((nc - 1) as f64) - 1.0); // ∈ [-8, 8]
        let x1 = lcg(&mut st); // within-cluster-varying ⇒ β slope identified
        xf64[(i, 0)] = 1.0;
        xf64[(i, 1)] = x1;
        let eta = 0.0 + 0.8 * x1 + u_c;
        let pr = 1.0 / (1.0 + (-eta).exp());
        y[i] = if lcg(&mut st) + 0.5 < pr { 1.0 } else { 0.0 };
    }
    let p = 2usize;
    let cluster = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: nc,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[], 1);

    // Converge a fit (kernel precondition for joint_hessian_cov).
    let fit = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[1u32],
        None,
        &[0.0, 0.8],
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fit.converged, "fixture fit must converge");
    // High-ICC data converges to a large θ̂ (≈ 7.6 here) — the flank whose
    // far side is the concave region.
    assert!(
        ws.params[0] > 5.0,
        "fit must reach the high-variance regime (θ̂ = {})",
        ws.params[0]
    );
    // Step past the minimum into the genuinely concave stretch (H_θθ < 0 from
    // the `2s·log θ` term — see the doc comment): at 5·θ̂ the LLT failure is
    // deterministic, not tolerance noise. joint_hessian_cov evaluates AT ws.params
    // (its f0 re-solves PIRLS there), so overwriting θ is the supported way to
    // probe the fallback; β stays at the fitted values.
    ws.params[0] *= 5.0;

    let mut cov = Mat::<f64>::zeros(p, p);
    let status = joint_hessian_cov(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        p,
        n,
        &mut cov,
    );
    assert_eq!(
        status,
        FdHessianStatus::NonPdFellBackToRx,
        "high-variance joint Hessian must be non-PD ⇒ RX fallback"
    );
    // Confirm this is the non-PD-Hessian branch, not the non-finite-deviance
    // branch: the joint Hessian was FULLY assembled (every perturbed deviance
    // finite) AND is non-PD (LLT errors). The fallback macro re-evals only the
    // central deviance, leaving ws.inference.hess_scratch holding the complete Hessian.
    let m = ws.params.len();
    for i in 0..m {
        for j in 0..m {
            assert!(
                ws.inference.hess_scratch[(i, j)].is_finite(),
                "assembled Hessian must be finite (non-finite branch NOT taken): H[{i}][{j}]"
            );
        }
    }
    assert!(
        ws.inference
            .hess_scratch
            .as_ref()
            .llt(faer::Side::Lower)
            .is_err(),
        "assembled joint Hessian must be non-PD (LLT must fail)"
    );

    // Fallback cov must equal the RX/Schur cov (a real inverse, not NaN). The
    // central deviance was re-evaluated inside joint_hessian_cov's fallback, so
    // the β-information factors are valid for rx_cov_into here too.
    let mut rx = Mat::<f64>::zeros(p, p);
    assert!(
        rx_cov_into(
            &mut ws,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &ids,
            p,
            n,
            &mut rx
        ),
        "β-only Schur must stay PD (well-conditioned β design)"
    );
    for i in 0..p {
        for j in 0..p {
            assert!(
                (cov[(i, j)] - rx[(i, j)]).abs() < 1e-10,
                "cov[{i}][{j}] {} vs rx {}",
                cov[(i, j)],
                rx[(i, j)]
            );
        }
    }
}

/// τ²→0 collapse-to-glm.rs (standing gate, L1 form).
#[test]
fn fit_glmm_collapses_to_plain_irls_when_tau_negligible() {
    use crate::glm::{glm_irls_fit, GlmScratch};
    let (n, nc) = (200usize, 10usize);
    let mut st = 99u64;
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        ids[i] = (i % nc) as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        let eta = 0.1 + 0.7 * x1; // NO u term ⇒ τ²→0 truth
        let p = 1.0 / (1.0 + (-eta).exp());
        let v = if lcg(&mut st) + 0.5 < p {
            1.0f64
        } else {
            0.0f64
        };
        y[i] = v;
    }

    // Plain logistic on the same bytes (adapt GlmScratch fields to the REAL struct).
    let mut sw = TestWs::new(n, 2, 0);
    let irls = {
        let s = GlmScratch {
            irls_eta: &mut sw.irls_eta[..n],
            irls_p: &mut sw.irls_p[..n],
            irls_w: &mut sw.irls_w[..n],
            irls_z: &mut sw.irls_z[..n],
            irls_betas: &mut sw.irls_betas[..2],
            irls_betas_new: &mut sw.irls_betas_new[..2],
            irls_var_diag: &mut sw.irls_var_diag[..1],
            irls_t_sq: &mut sw.irls_t_sq[..1],
            irls_u_scratch: &mut sw.irls_u_scratch[..2],
            irls_xtwx: sw.irls_xtwx.as_mut().submatrix_mut(0, 0, 2, 2),
            irls_xtwz: &mut sw.irls_xtwz[..2],
            irls_l: sw.irls_l.as_mut().submatrix_mut(0, 0, 2, 2),
            irls_wx: &mut sw.irls_wx[..n * 2],
        };
        let f = glm_irls_fit(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &[1],
            None,
            None,
            None,
            s,
        );
        (f.betas.to_vec(), f.t_sq.to_vec(), f.converged)
    };
    assert!(irls.2, "plain IRLS must converge");

    let cluster = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: nc,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
    let fit = fit_glmm(
        &mut ws,
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[1],
        Some(&[0.05]),
        &[0.1, 0.7],
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fit.converged);
    assert!(
        (ws.betas[1] - irls.0[1]).abs() < 1e-2,
        "β̂₁ glmm {} vs irls {}",
        ws.betas[1],
        irls.0[1]
    );
    // GLMM z² is indexed by coefficient (ws.inference.t_sq[1] = slope = target 1); IRLS
    // z² is indexed by target position (irls.1[0] = first target = slope).
    assert!(
        (ws.inference.t_sq[1].sqrt() - irls.1[0].sqrt()).abs() < 5e-2,
        "z glmm {} vs irls {}",
        ws.inference.t_sq[1].sqrt(),
        irls.1[0].sqrt()
    );
}

/// fit_glmm on the q_p=2 slope + crossed fixture, then the structured and the
/// packed layout on a variant of it where every θ̂ is interior.
#[test]
fn fit_glmm_width_general_slope_and_crossed() {
    let (xf64, y, ids, crossed_ids, cluster) = glmm_slope_crossed_dataset();
    let n = y.len();
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    let fit = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        std::slice::from_ref(&crossed_ids),
        &[1],
        None,
        &[0.2, 0.8],
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fit.converged);
    // Planted slope 0.8; small balanced binary GLMM (n=96) inflates β̂₁ to ≈3.2 with
    // z²≈13 — direction + significance are the robust claims, not the magnitude. τ̂²
    // collapses to 0 on this draw (valid), so only bound it finite + non-blown. The
    // old `>= 0.0` clauses were tautological on squared quantities.
    assert!(
        ws.betas[1] > 0.5,
        "slope must be strongly positive, got {}",
        ws.betas[1]
    );
    assert!(
        ws.inference.t_sq[1].is_finite() && ws.inference.t_sq[1] > 3.84,
        "z² must clear the α=0.05 bar (3.84), got {}",
        ws.inference.t_sq[1]
    );
    // τ̂² = Λ̂₀₀², the design fitted as given.
    let tau2 = ws.params[0].powi(2);
    assert!(tau2.is_finite() && (0.0..5.0).contains(&tau2), "τ̂² {tau2}");

    // Cross-layout agreement. On the draw above every θ̂ pins at 0, where the
    // crossed coupling C_f vanishes and the Schur downdate S −= C_f'A_f⁻¹C_f
    // does nothing. Five crossed levels (coprime to the eight primary ones, so
    // the groupings cross instead of nesting) and RE scales ×5 keep all four θ̂
    // off the boundary. The layouts take different outer routes (ExactProfile
    // vs not), so agreement is carried through BOBYQA. Bands as in
    // `fit::glmm_tests::fit_glmm_nb_random_slope_seed_lands_in_convergent_basin`:
    // 1e-5 absolute on the deviance, 1e-3 relative on β̂ and the SEs.
    let (xf64, y, ids, crossed_ids, cluster) = glmm_slope_crossed_dataset_sized(5, 5.0);
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    let fit = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        std::slice::from_ref(&crossed_ids),
        &[0, 1],
        None,
        &[0.2, 0.8],
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fit.converged, "structured layout must converge");
    assert_eq!(
        fit.pinned_components, 0,
        "every θ̂ must be interior for the downdate to matter"
    );
    let mut wsp = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[1], 1);
    fill_packed_cols(&mut wsp, &ids, std::slice::from_ref(&crossed_ids), n);
    let fitp = fit_glmm(
        &mut wsp,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        std::slice::from_ref(&crossed_ids),
        &[0, 1],
        None,
        &[0.2, 0.8],
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fitp.converged, "packed layout must converge");
    assert!(
        (fit.deviance - fitp.deviance).abs() <= 1e-5,
        "deviance structured {} vs packed {}",
        fit.deviance,
        fitp.deviance
    );
    for j in 0..2 {
        let (a, b) = (ws.betas[j], wsp.betas[j]);
        assert!(
            (a - b).abs() <= 1e-3 * a.abs().max(1.0),
            "β[{j}] structured {a} vs packed {b}"
        );
        let (a, b) = (
            ws.inference.var_diag[j].sqrt(),
            wsp.inference.var_diag[j].sqrt(),
        );
        assert!(
            (a - b).abs() <= 1e-3 * a,
            "se[{j}] structured {a} vs packed {b}"
        );
    }
}

/// Warm-path zero-alloc lock — `#[ignore]` because `dhat::Profiler` measures
/// process-wide allocations; `alloc_test_guard` serializes test bodies, but
/// libtest's own per-test thread spawn still needs `--test-threads=1`. faer's
/// rayon parallelism jitters the count run-to-run on a multi-core box, so pin
/// it to one thread for a deterministic measurement:
/// Run: `RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests fit_glmm_warm_path_bounded_alloc -- --ignored --test-threads=1`
/// (`alloc-tests` installs the dhat global allocator the profiler requires.)
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn fit_glmm_warm_path_bounded_alloc() {
    let _serial = crate::test_support::alloc_test_guard();
    let (xf64, y, ids) = glmm_intercept_dataset();
    let cluster = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: 8,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
    let _ = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[1],
        Some(&[0.5]),
        &[0.2, 0.8],
        80,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    ); // warmup
    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..20 {
        let _ = fit_glmm(
            &mut ws,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &y,
            &ids,
            &[],
            &[1],
            Some(&[0.5]),
            &[0.2, 0.8],
            80,
            WaldSe::Rx,
            ColumnScaling::IDENTITY,
        );
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);
    // Measured 120 (this machine) — 6 blocks/fit on the no-extras intercept
    // fixture, down from 7780. The blocked PIRLS routes it, so
    // the dense per-iteration k×1 rhs Mat and the per-eval dense `llt` internals
    // are gone; the per-block Crout factor/solve work on pre-allocated a_blocks
    // and stack-sized m_row scratch. What remains is the joint [θ|β] BOBYQA's
    // own per-eval scratch. Block COUNT is deterministic for a fixed code path
    // under `RAYON_NUM_THREADS=1` — a change to that floor flags a new allocation
    // or a shifted eval/iteration trajectory. If faer changes its Cholesky
    // internals, update — do not relax. The within-fit warm-start allocates
    // nothing itself (u_seed copy/reset hit pre-allocated buffers) but its
    // within-exit-band objective shift nudges the BOBYQA trajectory by a few evals:
    // 120 → 124. The SIMD fit-path transcendentals keep the floor flat:
    // the `pulp` dispatch is alloc-free and the loop-split uses only the
    // pre-allocated eta/prob/w scratch + stack-sized m_row, so the bound holds.
    // Re-measured after the relative PIRLS exit: still exactly 124
    // on this fixture — fewer inner iterations, same eval-scratch count.
    const BOUND: u64 = 124;
    assert!(
        stats.total_blocks <= BOUND,
        "warm-path alloc regressed: {} blocks across 20 fits (BOUND = {})",
        stats.total_blocks,
        BOUND
    );
}

/// Structured-path warm zero-alloc lock, the crossed/nested twin of
/// `fit_glmm_warm_path_bounded_alloc`. The dense crossed/nested path allocates
/// inside faer's per-eval `llt`, so it carries no equivalent gate. The structured
/// path replaces that with `glmm_block_chol`/`glmm_block_solve` on the
/// pre-allocated `core_blocks`/`schur_blk`/`coupling` + stack-sized per-block
/// scratch, so the only per-eval blocks left are the joint [θ|β] BOBYQA's own
/// scratch (M is packed into `ws.structured.m_core_buf`/`cross_theta`).
/// `#[ignore]` + one-thread
/// for the same reasons as the no-extras gate; `alloc_test_guard` serializes
/// test bodies, but libtest's own per-test thread spawn still needs
/// `--test-threads=1`. Run:
/// `RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests fit_glmm_structured_warm_path_bounded_alloc -- --ignored --test-threads=1`
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn fit_glmm_structured_warm_path_bounded_alloc() {
    let _serial = crate::test_support::alloc_test_guard();
    // crossed_nested q_p=1: 8 core blocks of 3×3 + a 6×6 Schur (the richest
    // structured shape; exercises both the block factor and the Schur).
    let (xf64, y, ids, extra_ids, cluster) = glmm_extras_q1_dataset(2, 6);
    let n = y.len();
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
    assert!(ws.groupings.structured_extras_eligible());
    let theta = [0.5_f64, 0.4, 0.45];
    let _ = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &extra_ids,
        &[1],
        Some(&theta),
        &[0.2, 0.8],
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    ); // warmup
    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..20 {
        let _ = fit_glmm(
            &mut ws,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &y,
            &ids,
            &extra_ids,
            &[1],
            Some(&theta),
            &[0.2, 0.8],
            n,
            WaldSe::Rx,
            ColumnScaling::IDENTITY,
        );
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);
    // The structured path factors/solves on pre-allocated buffers
    // (core_blocks/schur_blk/coupling) with stack-sized per-block scratch, so it
    // adds NO per-fit faer `llt` allocation; what remains is purely the joint
    // [θ|β] BOBYQA's own per-eval scratch. A change here flags a new allocation
    // or a shifted eval trajectory; do not relax past the noise floor below —
    // find the alloc.
    //
    // Run inside the alloc tier
    // (`RAYON_NUM_THREADS=1 cargo test --features alloc-tests -- --ignored --test-threads=1`)
    // this fixture reads a flat 120 blocks over three runs; run solo (as this
    // gate is run, `--test-threads=1` on its own) it reads a flat 124 across
    // runs. The gap is rayon startup, not first-touch or an unexplained
    // scheduling race: with `RAYON_NUM_THREADS=1` the worker pool's own
    // startup allocation is removed, so the only spread left is a stable
    // 4-block first-touch gap paid by an earlier test in the tier. With rayon
    // unset the count spreads 123–148, because the worker pool spins up on
    // whichever thread trips it first — pinning rayon to one thread is what
    // makes 124 exactly reproducible solo.
    const BOUND: u64 = 124;
    assert!(
        stats.total_blocks <= BOUND,
        "structured warm-path alloc regressed: {} blocks across 20 fits (BOUND = {})",
        stats.total_blocks,
        BOUND
    );
}

/// The zero-alloc gate: unlike the two BOUND-based fits above,
/// `laplace_gradient` claims ZERO allocation on a repeat call at the same
/// shape (`dual_scratch` and its `GlmmModeBufs` mode-transfer pair are both
/// sized once, on the first call, by `for_shape`) — so this asserts equality,
/// not a bound; a bound would hide a regression back into a per-call `Vec`.
/// Before this gate's fix, `laplace_gradient` built a fresh `saved_u` and
/// `u_mode` `Vec` (`.to_vec()`) on EVERY call, which this test caught; both
/// are now buffers on the dual scratch's `GlmmModeBufs` (`derivative.rs`).
/// `#[ignore]` + `alloc_test_guard` for the same process-wide-profiler reason
/// as the fits above; `alloc_test_guard` serializes test bodies, but libtest's
/// own per-test thread spawn still needs `--test-threads=1`.
/// Run: `RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests dual_gradient_repeat_calls_allocate_nothing -- --ignored --test-threads=1`
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn dual_gradient_repeat_calls_allocate_nothing() {
    let _serial = crate::test_support::alloc_test_guard();
    let (mut ws, x, y, ids, p, n) = fixture(
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        "int1",
    );
    let m = ws.n_theta + p;
    let mut grad = vec![0.0; m];
    // First call builds `dual_scratch` (the `GlmmDualBufs<Dual<4>>` `Vec`s +
    // its `ClusterRowIndex` + its `GlmmModeBufs` pair) — measured 26 blocks
    // (this machine, RAYON_NUM_THREADS unset) for this int1/m=3 shape. Not
    // asserted: `for_shape`'s own allocation is a one-time, per-shape cost,
    // never claimed zero.
    let _ = laplace_gradient(&mut ws, x.as_ref(), &y, &ids, &[], p, n, &mut grad);
    let prof = dhat::Profiler::builder().testing().build();
    for _ in 0..3 {
        let _ = laplace_gradient(&mut ws, x.as_ref(), &y, &ids, &[], p, n, &mut grad);
    }
    let stats = dhat::HeapStats::get();
    drop(prof);
    assert_eq!(stats.total_blocks, 0, "repeat dual gradient allocated");
}

/// The same zero-alloc claim on the structured-extras route, where the dual
/// scratch also carries the packed-M / core / coupling / Schur twins: they are
/// sized once by `for_shape` alongside the blocked buffers, never lazily on the
/// first extras call, and the CSR pattern is borrowed from the workspace rather
/// than mirrored — so a repeat call at the same shape must allocate nothing at
/// all. Runs the `(0, 6)` crossed cell, the one whose tail actually factors.
/// `#[ignore]` + `alloc_test_guard` for the process-wide-profiler reason the
/// other alloc gates give; `alloc_test_guard` serializes test bodies, but
/// libtest's own per-test thread spawn still needs `--test-threads=1`.
/// Run: `RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests structured_dual_gradient_repeat_calls_allocate_nothing -- --ignored --test-threads=1`
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn structured_dual_gradient_repeat_calls_allocate_nothing() {
    let _serial = crate::test_support::alloc_test_guard();
    let (mut ws, x, y, ids, extra_ids, p, n) = extras_fixture(
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        0,
        6,
    );
    let m = ws.n_theta + p;
    let mut grad = vec![0.0; m];
    // First call builds `dual_scratch` (the `GlmmDualBufs<Dual<4>>` `Vec`s,
    // structured twins included, + its `ClusterRowIndex` + its `GlmmModeBufs`
    // pair) — measured 26 blocks (this machine, RAYON_NUM_THREADS unset) for
    // this crossed/m=4 shape. Not asserted: `for_shape`'s own allocation is a
    // one-time, per-shape cost, never claimed zero.
    let _ = laplace_gradient(&mut ws, x.as_ref(), &y, &ids, &extra_ids, p, n, &mut grad);
    let prof = dhat::Profiler::builder().testing().build();
    for _ in 0..3 {
        let _ = laplace_gradient(&mut ws, x.as_ref(), &y, &ids, &extra_ids, p, n, &mut grad);
    }
    let stats = dhat::HeapStats::get();
    drop(prof);
    assert_eq!(
        stats.total_blocks, 0,
        "repeat structured dual gradient allocated"
    );
}

/// `glmm_block_chol` factors a q×q SPD block in place to its lower Crout L,
/// and `glmm_block_solve` then solves L Lᵀ x = b — checked against a faer LLT
/// solve on the same matrix. q=3 with a deliberately non-trivial SPD block.
#[test]
fn block_chol_and_solve_match_faer() {
    let q = 3usize;
    // SPD A (row-major, lower triangle is what the helper reads).
    let a = [4.0, 0.0, 0.0, 2.0, 5.0, 0.0, 1.0, 3.0, 6.0];
    let b = [1.0_f64, -2.0, 0.5];

    // Reference: faer LLT solve.
    let mut af = Mat::<f64>::zeros(q, q);
    for r in 0..q {
        for c in 0..=r {
            af[(r, c)] = a[r * q + c];
            af[(c, r)] = a[r * q + c];
        }
    }
    let ac = af.as_ref().llt(faer::Side::Lower).unwrap();
    let mut rhs = Mat::<f64>::zeros(q, 1);
    for r in 0..q {
        rhs[(r, 0)] = b[r];
    }
    ac.solve_in_place(rhs.as_mut());

    // Under test.
    let mut blk = a;
    assert!(super::glmm_block_chol(&mut blk, q), "block should be PD");
    let mut x = b;
    super::glmm_block_solve(&blk, q, &mut x);
    for r in 0..q {
        assert!(
            (x[r] - rhs[(r, 0)]).abs() < 1e-12,
            "x[{r}] = {}, faer {}",
            x[r],
            rhs[(r, 0)]
        );
    }
    // log|A| from pivots = 2·Σ ln L[r,r] should match faer's.
    let logdet_helper: f64 = (0..q).map(|r| blk[r * q + r].ln()).sum::<f64>() * 2.0;
    let logdet_faer: f64 = (0..q).map(|r| ac.L()[(r, r)].ln()).sum::<f64>() * 2.0;
    assert!((logdet_helper - logdet_faer).abs() < 1e-12);
}

/// Non-PD block ⇒ `glmm_block_chol` returns false (the module's failure surface).
#[test]
fn block_chol_rejects_non_pd() {
    let q = 2usize;
    let mut blk = [1.0_f64, 0.0, 2.0, 1.0]; // [[1,·],[2,1]] ⇒ pivot 1 − 4 < 0
    assert!(!super::glmm_block_chol(&mut blk, q));
}

/// No-extras q_p=2 (intercept + slope) clustered-binary dataset — exercises the
/// blocked SLOPE path (no crossed/nested ⇒ extra_offsets empty ⇒ blocked dispatch).
/// `contiguous` selects the id layout: false = round-robin `i % nc`
/// (the `FixedClusters` production layout, the historical fixture), true =
/// block layout `i / per_cluster` (the `FixedSize`/DGEN-FS production
/// layout). The blocked path must hold on both — layout-sensitive rewrites
/// of its row loops are a live optimization direction.
fn glmm_slope_noextra_dataset_layout(
    contiguous: bool,
    n: usize,
    nc: usize,
) -> (Mat<f64>, Vec<f64>, Vec<u32>) {
    let mut st = 21u64;
    let u0: Vec<f64> = (0..nc).map(|_| 0.6 * lcg(&mut st)).collect();
    let u1: Vec<f64> = (0..nc).map(|_| 0.4 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = if contiguous { i / (n / nc) } else { i % nc };
        ids[i] = c as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        let eta = 0.2 + 0.8 * x1 + u0[c] + u1[c] * x1;
        let p = 1.0 / (1.0 + (-eta).exp());
        y[i] = if lcg(&mut st) + 0.5 < p { 1.0 } else { 0.0 };
    }
    (x, y, ids)
}

fn glmm_slope_noextra_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>) {
    glmm_slope_noextra_dataset_layout(false, 96, 8)
}

/// The blocked intercept path computes the same Laplace deviance as the
/// independent brute force (reorders accumulation ⇒ FP-close, not bit-equal).
/// After Phase 2 this routes through `pirls_solve_blocked` (extra_offsets empty).
/// Run on both id layouts: round-robin (`FixedClusters`-style interleaved ids)
/// and the `FixedSize`/DGEN-FS production layout (contiguous id runs); a
/// layout-sensitive rewrite of the blocked row loops behaves differently per
/// layout, so the brute-force oracle must hold on both.
#[test]
fn blocked_laplace_matches_brute_force_intercept() {
    for contiguous in [false, true] {
        let layout = if contiguous {
            "contiguous"
        } else {
            "round-robin"
        };
        let (xf64, y, ids) = glmm_intercept_dataset_layout(contiguous, 80, 8);
        let beta = [0.2_f64, 0.8];
        let want = brute_force_intercept_laplace(0.5, &beta, &xf64, &y, &ids, 8);
        let cluster = SizedSpec {
            spec: logit_intercept_spec(),
            n_primary: 8,
            extra_counts: vec![],
        };
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
        assert!(
            ws.groupings.extra_offsets.is_empty(),
            "fixture must route blocked"
        );
        ws.params[0] = 0.5;
        ws.params[1] = beta[0];
        ws.params[2] = beta[1];
        let got = glmm_laplace_deviance(
            &ws.params.clone(),
            &mut ws,
            xf64.as_ref(),
            &y,
            &ids,
            &[],
            80,
        );
        assert!(
            (got - want).abs() < 1e-6,
            "blocked laplace ({layout}): got {got}, want {want}"
        );
    }
}

/// Blocked deviance == packed deviance on the no-extras slope fixture, to FP
/// error. Drives `pirls_solve_blocked` and `pirls_solve_packed` directly on the
/// same design at the same θ / β (dev-time equivalence smoke test; a wild
/// divergence is a coding bug). The packed oracle is a workspace
/// built with `for_cluster_spec_packed`, which overrides the layout
/// `GlmmLayout::for_design` would pick for this shape.
///
/// Run on both id layouts: round-robin (`FixedClusters`) and cluster-contiguous
/// (`FixedSize`/DGEN-FS), since layout-sensitive rewrites of the blocked row
/// loops are a live optimization direction.
#[test]
fn blocked_pirls_matches_packed_slope_noextra() {
    for contiguous in [false, true] {
        let layout = if contiguous {
            "contiguous"
        } else {
            "round-robin"
        };
        let (xf64, y, ids) = glmm_slope_noextra_dataset_layout(contiguous, 96, 8);
        let n = y.len();
        // slope on design col 1 ⇒ slope_cols = &[1]; q_p = 2.
        let cluster = SizedSpec {
            spec: ModelSpec {
                family: Family::Binomial {
                    link: BinomialLink::Logit,
                },
                re: Some(ReStructure {
                    terms: Vec::new(),
                    slopes: vec![1],
                    extra_groupings: vec![],
                }),
            },
            n_primary: 8,
            extra_counts: vec![],
        };
        // Packed oracle: the same design on the packed-row layout, which the
        // constructor would otherwise route to the blocked one.
        let mut ws = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[1], 1);
        assert!(ws.groupings.extra_offsets.is_empty());
        fill_packed_cols(&mut ws, &ids, &[], n);
        let theta = [0.5_f64, 0.1, 0.4]; // vech(Λ_p): [σ_int, cov, σ_slope]
        let mut beta = [0.2_f64, 0.8];
        let (k, p, nt) = (ws.k, ws.p, ws.n_theta);

        let mut params = vec![0.0; nt + p];
        params[..nt].copy_from_slice(&theta);
        params[nt..].copy_from_slice(&beta);
        packed_oracle_fill(&mut ws, xf64.as_ref(), &params, n);
        let GlmmWorkspace {
            prior_w,
            wx,
            pirls,
            packed,
            ..
        } = &mut ws;
        let oracle = pirls_solve_packed(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            k,
            p,
            xf64.as_ref(),
            &y,
            &prior_w[..n],
            false,
            &mut params[nt..],
            BetaStep::Fixed,
            pirls,
            packed,
            wx,
            None, // offset
            None,
            n,
            &mut crate::counters::EvalCounters::new(),
        );

        // Blocked: primary_lambda → pirls_solve_blocked, fresh scratch.
        let mut ws2 = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
        crate::lmm::primary_lambda(&theta, ws2.groupings.primary_q, &mut ws2.pirls.lam);
        fill_z_f64(
            &ws2.groupings,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &mut ws2.z_buf,
            n,
        );
        let GlmmWorkspace {
            groupings,
            z_buf,
            prior_w,
            pirls,
            ..
        } = &mut ws2;
        let mut wx_scratch = faer::Mat::<f64>::zeros(n, beta.len());
        let blocked = pirls_solve_blocked(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            groupings,
            &ids,
            xf64.as_ref(),
            &y,
            &prior_w[..n],
            false,
            &mut beta,
            BetaStep::Fixed,
            pirls,
            z_buf,
            None,
            &mut wx_scratch,
            None, // offset
            None,
            n,
            &mut crate::counters::EvalCounters::new(),
        );

        // Dense and blocked share the same lme4 step-halving backtrack and mixed
        // stopping rule, so the paths agree to FP error: dev/pen/logdet 1e-9, u 1e-7.
        assert_eq!(oracle.3, blocked.3, "{layout}: convergence flag");
        assert!(
            (oracle.0 - blocked.0).abs() < 1e-9,
            "{layout} dev: packed {} blocked {}",
            oracle.0,
            blocked.0
        );
        assert!(
            (oracle.1 - blocked.1).abs() < 1e-9,
            "{layout} pen: packed {} blocked {}",
            oracle.1,
            blocked.1
        );
        assert!(
            (oracle.2 - blocked.2).abs() < 1e-9,
            "{layout} logdet: packed {} blocked {}",
            oracle.2,
            blocked.2
        );
        // `ws` is the packed layout, `ws2` the blocked one: index the packed `u`
        // through the primary block's component-major order.
        let (qq, ss) = (ws2.groupings.primary_q, ws2.groupings.n_primary);
        for c in 0..k {
            let pc = packed_u_index(qq, ss, c);
            assert!(
                (ws.pirls.u[pc] - ws2.pirls.u[c]).abs() < 1e-7,
                "{layout} u[{c}]: packed {} blocked {}",
                ws.pirls.u[pc],
                ws2.pirls.u[c]
            );
        }
    }
}

/// Blocked inference (β̂, Var(β̂)_jj, z²) matches an independent Schur
/// recomputation on the no-extras slope fixture, to tight FP tolerance — the
/// inference reorder is the same estimator. Runs the full `fit_glmm` (blocked),
/// then rebuilds `M` as a full `n×k` matrix out of a packed workspace's rows
/// (`packed_dense_m`) and recomputes the Schur Var by hand on the same converged
/// state.
#[test]
fn blocked_inference_matches_packed_slope_noextra() {
    let (xf64, y, ids) = glmm_slope_noextra_dataset();
    let n = y.len();
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        },
        n_primary: 8,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    let fit = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[1],
        Some(&[0.5, 0.1, 0.4]),
        &[0.2, 0.8],
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fit.converged);
    // Recompute Var(β̂) densely from the converged ws.pirls.w and ws.params via a
    // freshly-built M and its own A, and compare β̂ / var_diag / t_sq.
    let (k, p, nt) = (ws.k, ws.p, ws.n_theta);
    let beta_blocked: Vec<f64> = ws.betas[..p].to_vec();
    let var_blocked = ws.inference.var_diag[1];
    let tsq_blocked = ws.inference.t_sq[1];
    // Dense M = ZΛ̂ at the converged θ̂, scattered out of the packed rows.
    let params: Vec<f64> = ws.params[..nt].to_vec();
    let mut wsp = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[1], 1);
    fill_packed_cols(&mut wsp, &ids, &[], n);
    packed_oracle_fill(&mut wsp, xf64.as_ref(), &params, n);
    let m = packed_dense_m(&wsp, n, k);
    // X'W̃X, X'W̃M, A = M'W̃M + I (dense), Schur, Var(β̂)_11.
    let mut xtwx = Mat::<f64>::zeros(p, p);
    for r in 0..p {
        for c in 0..p {
            let mut sm = 0.0;
            for i in 0..n {
                sm += xf64[(i, r)] * ws.pirls.w[i] * xf64[(i, c)];
            }
            xtwx[(r, c)] = sm;
        }
    }
    let mut xtwm = Mat::<f64>::zeros(p, k);
    for r in 0..p {
        for c in 0..k {
            let mut sm = 0.0;
            for i in 0..n {
                sm += xf64[(i, r)] * ws.pirls.w[i] * m[(i, c)];
            }
            xtwm[(r, c)] = sm;
        }
    }
    let mut a = Mat::<f64>::zeros(k, k);
    for r in 0..k {
        for c in 0..k {
            let mut sm = if r == c { 1.0 } else { 0.0 };
            for i in 0..n {
                sm += m[(i, r)] * ws.pirls.w[i] * m[(i, c)];
            }
            a[(r, c)] = sm;
        }
    }
    let ac = a.as_ref().llt(faer::Side::Lower).unwrap();
    let mut ainv = Mat::<f64>::zeros(k, p);
    for r in 0..k {
        for c in 0..p {
            ainv[(r, c)] = xtwm[(c, r)];
        }
    }
    ac.solve_in_place(ainv.as_mut());
    let mut schur = Mat::<f64>::zeros(p, p);
    for r in 0..p {
        for c in 0..p {
            let mut sm = xtwx[(r, c)];
            for j in 0..k {
                sm -= xtwm[(r, j)] * ainv[(j, c)];
            }
            schur[(r, c)] = sm;
        }
    }
    let sc = schur.as_ref().llt(faer::Side::Lower).unwrap();
    let mut fwd = vec![0.0; p];
    for i in 0..p {
        let mut acc = if i == 1 { 1.0 } else { 0.0 };
        #[allow(clippy::needless_range_loop)]
        for kk in 0..i {
            acc -= sc.L()[(i, kk)] * fwd[kk];
        }
        fwd[i] = acc / sc.L()[(i, i)];
    }
    let var_packed: f64 = fwd.iter().map(|v| v * v).sum();
    assert!(
        (var_blocked - var_packed).abs() < 1e-8,
        "var: blocked {var_blocked} packed {var_packed}"
    );
    let tsq_packed = beta_blocked[1] * beta_blocked[1] / var_packed;
    assert!(
        (tsq_blocked - tsq_packed).abs() < 1e-6,
        "z²: blocked {tsq_blocked} packed {tsq_packed}"
    );
}

/// A packed-layout fit hands the pinned-γ̂ re-evaluation a `u_seed` that already
/// IS û at the canonical θ̂, so the re-eval's PIRLS starts on its own solution.
///
/// The seam this holds is `fit_glmm`'s sign canonicalization: `fix_mode_signs`
/// negates the `u` entries of every Λ column whose diagonal θ is negative, and
/// it has to be told how the primary block is ordered — the packed layout is
/// slope-major (`d·n_primary + level`), the blocked and structured ones
/// level-major. Reading a packed seed level-major negates the wrong entries, and
/// the re-eval then starts at a scrambled point instead of at û(γ̂).
///
/// The fixture converges on a negative primary intercept diagonal, so the flip
/// fires, and `k = 16` with `q_p = 2` makes the two orders disagree on 8 of the
/// 16 entries — a gap of order 1 in û. The band is 1e-4 because it also has to
/// absorb the honest displacement from the slope diagonal pinning to 0 between
/// the seed and the re-eval, measured at 4.7e-7 here. `WaldSe::Rx` matters: the
/// Hessian arm's `joint_hessian_cov` reseeds `u_seed` from `ws.pirls.u` itself,
/// which would make the comparison vacuous.
#[test]
fn packed_fit_mode_seed_survives_sign_canonicalization() {
    let (xf64, y, ids) = glmm_slope_noextra_dataset_layout(true, 96, 8);
    let n = y.len();
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        },
        n_primary: 8,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[1], 1);
    fill_packed_cols(&mut ws, &ids, &[], n);
    let fit = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[1],
        Some(&[0.5, 0.1, 0.4]),
        &[0.2, 0.8],
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fit.converged, "fixture must converge");
    let (worst, at) = (0..ws.k)
        .map(|i| ((ws.u_seed[i] - ws.pirls.u[i]).abs(), i))
        .fold(
            (0.0_f64, 0usize),
            |acc, d| if d.0 > acc.0 { d } else { acc },
        );
    assert!(
        worst < 1e-4,
        "u_seed vs û(γ̂): worst |diff| {worst:.6e} at RE column {at}"
    );
}

/// Within-fit û warm-start MUST reset per fit. A reused-workspace re-fit must
/// match a fresh-workspace (canonically cold) fit BIT-FOR-BIT. Carrying the
/// incumbent across fits would warm-start one simulation from another's state,
/// breaking merge / same-seed reproducibility. This is the per-fit-reset guard
/// against that.
#[test]
fn warm_start_is_per_fit_deterministic() {
    let (xf64, y, ids) = glmm_intercept_dataset();
    let cluster = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: 8,
        extra_counts: vec![],
    };
    let mk = || GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
    // Canonical: a fresh workspace's first (cold) fit.
    let mut ws_ref = mk();
    let _ = fit_glmm(
        &mut ws_ref,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[1],
        Some(&[0.5]),
        &[0.2, 0.8],
        80,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    let ref_beta = ws_ref.betas[1].to_bits();
    // Reused workspace: a throwaway fit pollutes u_seed, then the measured fit
    // must match the canonical cold result bit-for-bit (only if u_seed resets).
    let mut ws = mk();
    let _ = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[1],
        Some(&[0.5]),
        &[0.2, 0.8],
        80,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    let _ = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[1],
        Some(&[0.5]),
        &[0.2, 0.8],
        80,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert_eq!(
        ws.betas[1].to_bits(),
        ref_beta,
        "re-fit β̂ must match the fresh cold fit bit-for-bit"
    );
}

/// The Laplace objective is a (near-)pure function of (θ,β): the same point
/// solved from u=0 vs from a perturbed seed agrees within ≪ the estimate floor
/// — the conditional mode is unique, only the stopping iterate is seed-dependent.
#[test]
fn warm_start_objective_is_seed_independent() {
    let (xf64, y, ids) = glmm_intercept_dataset();
    let cluster = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: 8,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
    ws.params[0] = 0.5;
    ws.params[1] = 0.2;
    ws.params[2] = 0.8;
    for v in ws.pirls.u.iter_mut() {
        *v = 0.0;
    }
    let cold = glmm_laplace_deviance(
        &ws.params.clone(),
        &mut ws,
        xf64.as_ref(),
        &y,
        &ids,
        &[],
        80,
    );
    // `laplace_deviance_at` zero-fills `pirls.u` unless `warm_seed_active`, so
    // the perturbed seed goes into `u_seed`.
    for (c, v) in ws.u_seed[..8].iter_mut().enumerate() {
        *v = 0.05 * (c as f64 - 4.0);
    }
    ws.fd.warm_seed_active = true;
    let warm = glmm_laplace_deviance(
        &ws.params.clone(),
        &mut ws,
        xf64.as_ref(),
        &y,
        &ids,
        &[],
        80,
    );
    ws.fd.warm_seed_active = false;
    assert!(
        (cold - warm).abs() < 1e-6,
        "objective seed-dependent: cold {cold} warm {warm}"
    );
}

// -- Group G structured-solve armor (crossed/nested/crossed_nested, q_p=1) ----

/// q_p=1 clustered-binary dataset with intercept-only extra groupings, in the
/// three Group-G bench shapes. `np` = nested children per parent (0 = none),
/// `n_crossed` = crossed levels (0 = none). Returns (X f64 [1, x1], y∈{0,1},
/// primary ids, extra ids in DECLARATION order `[nested?, crossed?]`, spec).
/// Mirrors `glmm_slope_crossed_dataset`'s construction (round-robin ids so
/// every level is populated); the spec declares nested before crossed so the
/// θ vector is `[θ_p, θ_nested?, θ_crossed?]` and the engine's `extra_offsets`
/// land nested at `prim_width`, crossed in the trailing block.
// `pub(crate)`: reused by `src/scalar.rs`'s tail-methods test so that test's
// `S` comes from a real crossed-incidence pattern rather than a fabricated
// one.
#[allow(clippy::type_complexity)]
pub(crate) fn glmm_extras_q1_dataset(
    np: usize,
    n_crossed: usize,
) -> (Mat<f64>, Vec<f64>, Vec<u32>, Vec<Vec<u32>>, SizedSpec) {
    glmm_extras_q1_dataset_sized(np, n_crossed, 8, 96)
}

/// `glmm_extras_q1_dataset` generalized to a caller-chosen primary cluster
/// count and row count (the tail-boundary sweep needs both large enough to
/// keep a 500-level crossed tail populated). `glmm_extras_q1_dataset`'s own
/// `(n_prim, n) = (8, 96)` reproduces its committed callers byte-for-byte.
#[allow(clippy::type_complexity)]
pub(crate) fn glmm_extras_q1_dataset_sized(
    np: usize,
    n_crossed: usize,
    n_prim: usize,
    n: usize,
) -> (Mat<f64>, Vec<f64>, Vec<u32>, Vec<Vec<u32>>, SizedSpec) {
    let mut st = 29u64;
    let u0: Vec<f64> = (0..n_prim).map(|_| 0.6 * lcg(&mut st)).collect();
    // Nested children are globalized: parent·np + within ⇒ n_prim·np draws.
    let un: Vec<f64> = (0..n_prim * np.max(1))
        .map(|_| 0.4 * lcg(&mut st))
        .collect();
    let uc: Vec<f64> = (0..n_crossed.max(1)).map(|_| 0.5 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    let mut nested = vec![0u32; n];
    let mut crossed = vec![0u32; n];
    for i in 0..n {
        let c = i % n_prim;
        ids[i] = c as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        let mut eta = 0.2 + 0.8 * x1 + u0[c];
        if np > 0 {
            // within-parent child cycles through 0..np; globalized id = c·np + within.
            let within = (i / n_prim) % np;
            let gid = c * np + within;
            nested[i] = gid as u32;
            eta += un[gid];
        }
        if n_crossed > 0 {
            let cc = i % n_crossed;
            crossed[i] = cc as u32;
            eta += uc[cc];
        }
        let p = 1.0 / (1.0 + (-eta).exp());
        y[i] = if lcg(&mut st) + 0.5 < p { 1.0 } else { 0.0 };
    }
    let mut extra_groupings = Vec::new();
    let mut extra_counts = Vec::new();
    let mut extra_ids = Vec::new();
    if np > 0 {
        extra_groupings.push(Grouping {
            terms: Vec::new(),
            relation: GroupingRelation::NestedWithin,
            slopes: vec![],
        });
        extra_counts.push(np);
        extra_ids.push(nested);
    }
    if n_crossed > 0 {
        extra_groupings.push(Grouping {
            terms: Vec::new(),
            relation: GroupingRelation::Crossed,
            slopes: vec![],
        });
        extra_counts.push(n_crossed);
        extra_ids.push(crossed);
    }
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings,
            }),
        },
        n_primary: n_prim,
        extra_counts,
    };
    (x, y, ids, extra_ids, cluster)
}

/// Independent dense Laplace deviance for the intercept-only extras shapes:
/// builds a fully-dense Z `[primary | nested children | crossed]` in its OWN
/// column order (permutation-invariant to the engine's layout), `M = Z·θ`
/// (θ_p on primary cols, θ_nested on nested, θ_crossed on crossed), runs
/// Newton-on-u to the conditional mode, and returns `d(y,ũ) + ‖ũ‖² + log|A|`
/// (A = M'WM + I). The ground-truth oracle for `pirls_solve_blocked_extras`:
/// a parity test only proves "same as dense"; this proves "correct".
/// Mirrors `brute_force_intercept_laplace` generalized to the extra columns.
#[allow(clippy::too_many_arguments)]
fn brute_force_extras_laplace(
    theta_p: f64,
    theta_n: f64,
    theta_c: f64,
    beta: &[f64],
    x: &Mat<f64>,
    y: &[f64],
    ids: &[u32],
    nested: &[u32],
    crossed: &[u32],
    n_prim: usize,
    np: usize,
    n_crossed: usize,
) -> f64 {
    let n = x.nrows();
    let nc = n_prim + n_prim * np + n_crossed;
    let mut m = Mat::<f64>::zeros(n, nc);
    let nest_base = n_prim;
    let cross_base = n_prim + n_prim * np;
    for i in 0..n {
        m[(i, ids[i] as usize)] = theta_p;
        if np > 0 {
            m[(i, nest_base + nested[i] as usize)] = theta_n;
        }
        if n_crossed > 0 {
            m[(i, cross_base + crossed[i] as usize)] = theta_c;
        }
    }
    laplace_from_dense_m(&m, beta, x, y)
}

/// Laplace deviance `d(y,ũ)·2 + ‖ũ‖² + 2·ln|L|` (A = M'WM + I = LLᵀ) of a Bernoulli-logit
/// model with the dense RE design `M = ZΛ` given column by column, by Newton on u.
fn laplace_from_dense_m(m: &Mat<f64>, beta: &[f64], x: &Mat<f64>, y: &[f64]) -> f64 {
    let (n, p) = (x.nrows(), x.ncols());
    let nc = m.ncols();
    let eta_of = |u: &[f64], i: usize| -> f64 {
        let mut e = 0.0;
        for j in 0..p {
            e += x[(i, j)] * beta[j];
        }
        for c in 0..nc {
            e += m[(i, c)] * u[c];
        }
        e
    };
    let mut u = vec![0.0f64; nc];
    // Newton on the penalized binomial: H = 2(M'WM + I), g = 2(u − M'(y−p)).
    for _ in 0..80 {
        let mut pvec = vec![0.0; n];
        let mut w = vec![0.0; n];
        for i in 0..n {
            let e = eta_of(&u, i);
            let pi = 1.0 / (1.0 + (-e).exp());
            pvec[i] = pi;
            w[i] = (pi * (1.0 - pi)).max(1e-6);
        }
        let mut g = vec![0.0; nc];
        for c in 0..nc {
            let mut s = 0.0;
            for i in 0..n {
                s += m[(i, c)] * (y[i] - pvec[i]);
            }
            g[c] = 2.0 * u[c] - 2.0 * s;
        }
        let mut h = Mat::<f64>::zeros(nc, nc);
        for a in 0..nc {
            for b in 0..nc {
                let mut s = 0.0;
                for i in 0..n {
                    s += m[(i, a)] * w[i] * m[(i, b)];
                }
                h[(a, b)] = 2.0 * (s + if a == b { 1.0 } else { 0.0 });
            }
        }
        let hc = h.as_ref().llt(faer::Side::Lower).unwrap();
        let mut step = Mat::<f64>::zeros(nc, 1);
        for c in 0..nc {
            step[(c, 0)] = g[c];
        }
        hc.solve_in_place(step.as_mut());
        let mut max = 0.0f64;
        for c in 0..nc {
            u[c] -= step[(c, 0)];
            max = max.max(step[(c, 0)].abs());
        }
        if max < 1e-11 {
            break;
        }
    }
    // Laplace at the mode: d + ‖u‖² + log|A|.
    let mut d = 0.0;
    let mut w = vec![0.0; n];
    for i in 0..n {
        let e = eta_of(&u, i);
        let pi = 1.0 / (1.0 + (-e).exp());
        w[i] = (pi * (1.0 - pi)).max(1e-6);
        d += if e > 0.0 {
            e + (-e).exp().ln_1p()
        } else {
            e.exp().ln_1p()
        } - y[i] * e;
    }
    let pen: f64 = u.iter().map(|v| v * v).sum();
    let mut a = Mat::<f64>::zeros(nc, nc);
    for r in 0..nc {
        for c in 0..nc {
            let mut s = 0.0;
            for i in 0..n {
                s += m[(i, r)] * w[i] * m[(i, c)];
            }
            a[(r, c)] = s + if r == c { 1.0 } else { 0.0 };
        }
    }
    let ac = a.as_ref().llt(faer::Side::Lower).unwrap();
    let mut logdet = 0.0;
    for r in 0..nc {
        logdet += ac.L()[(r, r)].ln();
    }
    2.0 * d + pen + 2.0 * logdet
}

/// `laplace_deviance` matches the independent dense brute-force oracle on all
/// three Group-G shapes. With the structured kernel in place the dispatch
/// routes these eligible shapes through `pirls_solve_blocked_extras`, so this
/// is the structured-vs-oracle ground-truth test: a parity test only proves
/// "same as dense", this proves the Schur assembly is *correct*.
#[test]
fn structured_extras_laplace_matches_brute_force() {
    // (np, n_crossed, label)
    for (np, ncr, label) in [
        (0usize, 6usize, "crossed"),
        (2, 0, "nested"),
        (2, 6, "crossed_nested"),
    ] {
        let (xf64, y, ids, extra_ids, cluster) = glmm_extras_q1_dataset(np, ncr);
        let n = y.len();
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
        assert!(
            !ws.groupings.extra_offsets.is_empty(),
            "{label}: fixture must route through the extras path"
        );
        let extra_refs: Vec<&[u32]> = extra_ids.iter().map(|v| v.as_slice()).collect();
        // θ = [θ_p, θ_nested?, θ_crossed?] in declaration order [nested, crossed].
        let theta_p = 0.5;
        let theta_n = 0.4;
        let theta_c = 0.45;
        let beta = [0.2_f64, 0.8];
        let nt = ws.n_theta;
        ws.params[0] = theta_p;
        let mut ti = 1;
        if np > 0 {
            ws.params[ti] = theta_n;
            ti += 1;
        }
        if ncr > 0 {
            ws.params[ti] = theta_c;
        }
        ws.params[nt] = beta[0];
        ws.params[nt + 1] = beta[1];
        let got = glmm_laplace_deviance(
            &ws.params.clone(),
            &mut ws,
            xf64.as_ref(),
            &y,
            &ids,
            &extra_ids,
            n,
        );
        let nested = if np > 0 { extra_refs[0] } else { &[][..] };
        let crossed = if ncr > 0 {
            extra_refs[extra_refs.len() - 1]
        } else {
            &[][..]
        };
        let want = brute_force_extras_laplace(
            theta_p, theta_n, theta_c, &beta, &xf64, &y, &ids, nested, crossed, 8, np, ncr,
        );
        assert!(
            (got - want).abs() < 1e-6,
            "{label} dense laplace: got {got}, want {want}"
        );
    }
}

/// The q_p=2 structured Laplace against the dense brute force: intercept and
/// slope on the primary grouping plus one crossed intercept
/// (`glmm_slope_crossed_dataset`). Λ_p is the lower triangle filled from the
/// column-major vech θ = [λ00, λ10, λ11] (`primary_lambda`), so a row's two
/// primary columns of M = ZΛ are λ00 + x·λ10 and x·λ11 (the
/// `Σ_{r≥c} z_r·lam[r·q+c]` core rule). The structured-vs-packed parity cannot see a
/// move of both sides (both read `z_buf`); this can.
#[test]
fn structured_slope_crossed_laplace_matches_brute_force() {
    let (xf64, y, ids, crossed, cluster) = glmm_slope_crossed_dataset();
    let n = y.len();
    let (n_prim, n_crossed) = (8usize, 4usize);
    let theta = [0.5_f64, 0.1, 0.4, 0.45];
    let beta = [0.2_f64, 0.8];
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    assert_eq!(ws.layout, GlmmLayout::Structured);
    let nt = ws.n_theta;
    assert_eq!(nt, theta.len());
    let mut params = vec![0.0; nt + beta.len()];
    params[..nt].copy_from_slice(&theta);
    params[nt..].copy_from_slice(&beta);
    let extra_ids = vec![crossed.clone()];
    let got = glmm_laplace_deviance(&params, &mut ws, xf64.as_ref(), &y, &ids, &extra_ids, n);
    let mut m = Mat::<f64>::zeros(n, 2 * n_prim + n_crossed);
    for i in 0..n {
        let (c, x1) = (ids[i] as usize, xf64[(i, 1)]);
        m[(i, 2 * c)] = theta[0] + x1 * theta[1];
        m[(i, 2 * c + 1)] = x1 * theta[2];
        m[(i, 2 * n_prim + crossed[i] as usize)] = theta[3];
    }
    let want = laplace_from_dense_m(&m, &beta, &xf64, &y);
    assert!(
        (got - want).abs() < 1e-6,
        "slope_q2_crossed laplace: got {got}, want {want}"
    );
}

/// A pin is a regression lock on a return no independent oracle reaches. A band
/// against another kernel is blind to what moves both sides, and to anything
/// below the band. A correct numerical change that moves a pin is re-pinned with
/// a provenance line (date, what moved by how many ULP, `converged` unchanged),
/// as the two Profile-mode pins below record.
///
/// A structured PIRLS return `(dev, pen, logdet, converged)` against its pin:
/// the flag exactly, each float to 4ε relative. Not `==`: the same build gives
/// the same bits, but libm's `exp`/`ln` and the SIMD backend differ across
/// platforms, and on wasm32 (plain and `+simd128`) these returns land 1-2 ULP
/// from the x86_64 pins.
fn assert_pinned_return(got: (f64, f64, f64, bool), pin: (f64, f64, f64, bool), ctx: &str) {
    let close = |a: f64, b: f64| (a - b).abs() <= 4.0 * f64::EPSILON * b.abs();
    assert!(
        got.3 == pin.3 && close(got.0, pin.0) && close(got.1, pin.1) && close(got.2, pin.2),
        "{ctx}: structured return moved: got {got:?}, pin {pin:?}"
    );
}

/// Structured == packed on all three Group-G shapes, to FP error: drives
/// `pirls_solve_packed` and `pirls_solve_blocked_extras` on the same design at
/// the same θ / β from fresh scratch. The Schur reassociation is the same
/// estimator. Bands mirror the no-extras blocked parity
/// (`blocked_pirls_matches_packed_slope_noextra`): dev/pen/logdet 1e-9, u 1e-7.
///
/// Parity only: what moves both sides is caught by the brute-force oracles at
/// the same θ/β (`structured_extras_laplace_matches_brute_force` for the three
/// q_p=1 shapes, `structured_slope_crossed_laplace_matches_brute_force` for
/// `slope_q2_crossed`).
#[test]
fn structured_extras_matches_packed() {
    for (np, ncr, label) in [
        (0usize, 6usize, "crossed"),
        (2, 0, "nested"),
        (2, 6, "crossed_nested"),
    ] {
        let (xf64, y, ids, extra_ids, cluster) = glmm_extras_q1_dataset(np, ncr);
        let n = y.len();
        // θ = [θ_p, θ_nested?, θ_crossed?] in declaration order [nested, crossed].
        let mut theta = vec![0.5_f64];
        if np > 0 {
            theta.push(0.4);
        }
        if ncr > 0 {
            theta.push(0.45);
        }
        let mut beta = [0.2_f64, 0.8];

        // Packed oracle: the same design on the packed-row layout.
        let mut ws = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[], 1);
        assert!(
            ws.groupings.structured_extras_eligible(),
            "{label}: fixture must be structured-eligible"
        );
        fill_packed_cols(&mut ws, &ids, &extra_ids, n);
        let (k, p, nt) = (ws.k, ws.p, ws.n_theta);
        let mut params = vec![0.0; nt + p];
        params[..nt].copy_from_slice(&theta);
        params[nt..].copy_from_slice(&beta);
        packed_oracle_fill(&mut ws, xf64.as_ref(), &params, n);
        let GlmmWorkspace {
            prior_w,
            wx,
            pirls,
            packed,
            ..
        } = &mut ws;
        let oracle = pirls_solve_packed(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            k,
            p,
            xf64.as_ref(),
            &y,
            &prior_w[..n],
            false,
            &mut params[nt..],
            BetaStep::Fixed,
            pirls,
            packed,
            wx,
            None, // offset
            None,
            n,
            &mut crate::counters::EvalCounters::new(),
        );

        // Structured: build_packed_m → pirls_solve_blocked_extras, fresh scratch.
        let mut ws2 = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
        {
            let GlmmWorkspace {
                groupings,
                params: prm,
                z_buf,
                pirls: PirlsScratch { lam, .. },
                structured:
                    StructuredScratch {
                        m_core_buf,
                        cross_theta,
                        ..
                    },
                pattern:
                    StructuredPattern {
                        cross_col, n_cross, ..
                    },
                ..
            } = &mut ws2;
            prm[..nt].copy_from_slice(&theta);
            prm[nt..nt + p].copy_from_slice(&beta);
            build_packed_m(
                groupings,
                &prm[..],
                z_buf,
                &extra_ids,
                lam,
                &ids,
                m_core_buf,
                cross_theta,
                cross_col,
                n_cross,
                n,
            );
        }
        let structured = {
            let GlmmWorkspace {
                groupings,
                prior_w,
                pirls,
                structured,
                pattern,
                ..
            } = &mut ws2;
            build_coupling_csr(
                &ids,
                &pattern.cross_col,
                &pattern.n_cross,
                groupings.n_primary,
                n,
                &mut pattern.coup_cols,
                &mut pattern.coup_ptr,
                &mut pattern.coup_pos,
            );
            let mut wx_scratch = faer::Mat::<f64>::zeros(n, p);
            pirls_solve_blocked_extras(
                crate::Family::Binomial {
                    link: crate::BinomialLink::Logit,
                },
                f64::NAN,
                groupings,
                &ids,
                xf64.as_ref(),
                &y,
                &prior_w[..n],
                false,
                &mut beta,
                BetaStep::Fixed,
                pirls,
                structured,
                pattern,
                None, // dual
                &mut wx_scratch,
                None, // offset
                None,
                n,
                &mut crate::counters::EvalCounters::new(),
            )
        };

        // 1e-9 on the three objective terms and 1e-7 on `u`: the structured
        // path (`pirls_solve_blocked_extras`) shares the packed loop's
        // step-halving backtrack, and when no halving fires both walk the same
        // iterate path, so they agree to FP error.
        assert_eq!(oracle.3, structured.3, "{label}: convergence flag");
        assert!(
            (oracle.0 - structured.0).abs() < 1e-9,
            "{label} dev: packed {} structured {}",
            oracle.0,
            structured.0
        );
        assert!(
            (oracle.1 - structured.1).abs() < 1e-9,
            "{label} pen: packed {} structured {}",
            oracle.1,
            structured.1
        );
        assert!(
            (oracle.2 - structured.2).abs() < 1e-9,
            "{label} logdet: packed {} structured {}",
            oracle.2,
            structured.2
        );
        // `ws` is the packed layout, `ws2` the structured one: index the packed
        // `u` through the primary block's component-major order.
        let (qq, ss) = (ws2.groupings.primary_q, ws2.groupings.n_primary);
        for c in 0..k {
            let pc = packed_u_index(qq, ss, c);
            assert!(
                (ws.pirls.u[pc] - ws2.pirls.u[c]).abs() < 1e-7,
                "{label} u[{c}]: packed {} structured {}",
                ws.pirls.u[pc],
                ws2.pirls.u[c]
            );
        }
    }

    // q_p=2 case: every shape above is q_p=1 (intercept-only primary), so none
    // of them exercises build_packed_m's z_buf-sourced primary-core read (read
    // 1 of the dense-Z phase-2 rewrite) at q>1 -- an unfilled or mis-widened
    // z_buf would silently zero the slope column and still pass every
    // assertion above. Reuses `glmm_slope_crossed_dataset` (q_p=2 slope + one
    // crossed extra), the same fixture `fit_glmm_width_general_slope_and_crossed`
    // drives end-to-end; same dense-vs-structured comparison, same bands.
    {
        let label = "slope_q2_crossed";
        let (xf64, y, ids, crossed_ids, cluster) = glmm_slope_crossed_dataset();
        let extra_ids = vec![crossed_ids];
        let n = y.len();
        // θ = [vech(Λ_p): σ_int, cov, σ_slope | θ_crossed]; β = [intercept, slope]
        // (mirrors `agq_vec_k1_reduces_to_laplace_q2`'s q=2 vech convention).
        let theta = vec![0.5_f64, 0.1, 0.4, 0.45];
        let mut beta = [0.2_f64, 0.8];

        // Packed oracle: the same design on the packed-row layout.
        let mut ws = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[1], 1);
        assert!(
            ws.groupings.structured_extras_eligible(),
            "{label}: fixture must be structured-eligible"
        );
        fill_packed_cols(&mut ws, &ids, &extra_ids, n);
        let (k, p, nt) = (ws.k, ws.p, ws.n_theta);
        let mut params = vec![0.0; nt + p];
        params[..nt].copy_from_slice(&theta);
        params[nt..].copy_from_slice(&beta);
        packed_oracle_fill(&mut ws, xf64.as_ref(), &params, n);
        let GlmmWorkspace {
            prior_w,
            wx,
            pirls,
            packed,
            ..
        } = &mut ws;
        let oracle = pirls_solve_packed(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            k,
            p,
            xf64.as_ref(),
            &y,
            &prior_w[..n],
            false,
            &mut params[nt..],
            BetaStep::Fixed,
            pirls,
            packed,
            wx,
            None, // offset
            None,
            n,
            &mut crate::counters::EvalCounters::new(),
        );

        // Structured: build_packed_m → pirls_solve_blocked_extras, fresh scratch.
        // build_packed_m's primary-core read sources the slope value from z_buf,
        // which production hoists once per fit (`fit_glmm`/`joint_hessian_cov`,
        // see the widened gate in mod.rs/se.rs); this raw kernel-level test must
        // do the same thing by hand.
        let mut ws2 = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
        fill_z_f64(
            &ws2.groupings,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &mut ws2.z_buf,
            n,
        );
        {
            let GlmmWorkspace {
                groupings,
                params: prm,
                z_buf,
                pirls: PirlsScratch { lam, .. },
                structured:
                    StructuredScratch {
                        m_core_buf,
                        cross_theta,
                        ..
                    },
                pattern:
                    StructuredPattern {
                        cross_col, n_cross, ..
                    },
                ..
            } = &mut ws2;
            prm[..nt].copy_from_slice(&theta);
            prm[nt..nt + p].copy_from_slice(&beta);
            build_packed_m(
                groupings,
                &prm[..],
                z_buf,
                &extra_ids,
                lam,
                &ids,
                m_core_buf,
                cross_theta,
                cross_col,
                n_cross,
                n,
            );
        }
        let structured = {
            let GlmmWorkspace {
                groupings,
                prior_w,
                pirls,
                structured,
                pattern,
                ..
            } = &mut ws2;
            build_coupling_csr(
                &ids,
                &pattern.cross_col,
                &pattern.n_cross,
                groupings.n_primary,
                n,
                &mut pattern.coup_cols,
                &mut pattern.coup_ptr,
                &mut pattern.coup_pos,
            );
            let mut wx_scratch = faer::Mat::<f64>::zeros(n, p);
            pirls_solve_blocked_extras(
                crate::Family::Binomial {
                    link: crate::BinomialLink::Logit,
                },
                f64::NAN,
                groupings,
                &ids,
                xf64.as_ref(),
                &y,
                &prior_w[..n],
                false,
                &mut beta,
                BetaStep::Fixed,
                pirls,
                structured,
                pattern,
                None, // dual
                &mut wx_scratch,
                None, // offset
                None,
                n,
                &mut crate::counters::EvalCounters::new(),
            )
        };

        assert_eq!(oracle.3, structured.3, "{label}: convergence flag");
        assert!(
            (oracle.0 - structured.0).abs() < 1e-9,
            "{label} dev: packed {} structured {}",
            oracle.0,
            structured.0
        );
        assert!(
            (oracle.1 - structured.1).abs() < 1e-9,
            "{label} pen: packed {} structured {}",
            oracle.1,
            structured.1
        );
        assert!(
            (oracle.2 - structured.2).abs() < 1e-9,
            "{label} logdet: packed {} structured {}",
            oracle.2,
            structured.2
        );
        // `ws` is the packed layout, `ws2` the structured one: index the packed
        // `u` through the primary block's component-major order.
        let (qq, ss) = (ws2.groupings.primary_q, ws2.groupings.n_primary);
        for c in 0..k {
            let pc = packed_u_index(qq, ss, c);
            assert!(
                (ws.pirls.u[pc] - ws2.pirls.u[c]).abs() < 1e-7,
                "{label} u[{c}]: packed {} structured {}",
                ws.pirls.u[pc],
                ws2.pirls.u[c]
            );
        }
    }
}

/// Two legs, distinct roles (2026-07-14 qc=1-downdate-route spec).
/// `structured_factor`'s per-cluster `S −= C_f'A_f⁻¹C_f` runs panel-wise when
/// `StructuredSchur` is cached AND `qc > 1`, and column-at-a-time scalar
/// otherwise (the `ss = None` arm, and — since the qc=1 route change — the
/// production path whenever `qc == 1`). `force_dense = true` on BOTH runs pins
/// the factor to the same dense Crout, so any disagreement is the downdate alone.
///
/// - `crossed_nested` (np=2 ⇒ qc=3): the real panel-vs-scalar oracle. `use_panel`
///   actually exercises the panel path; bands mirror
///   `structured_extras_matches_packed` (dev/pen/logdet 1e-9, u 1e-7) to absorb the
///   panel's downdate reassociation.
/// - `crossed` (np=0 ⇒ qc=1): a routing smoke check. Both runs take the scalar
///   arm (qc==1 routes there regardless of `ss`), so they run identical code and
///   agree bit-exact — the band tightens to `== 0.0`.
#[test]
fn structured_panel_downdate_matches_scalar() {
    for (np, ncr, label) in [(0usize, 6usize, "crossed"), (2, 6, "crossed_nested")] {
        // qc == 1 (np == 0) ⇒ both runs are the same scalar arm ⇒ bit-exact.
        let tol = if np == 0 { 0.0 } else { 1e-9 };
        let u_tol = if np == 0 { 0.0 } else { 1e-7 };
        let (xf64, y, ids, extra_ids, cluster) = glmm_extras_q1_dataset(np, ncr);
        let n = y.len();
        let mut theta = vec![0.5_f64];
        if np > 0 {
            theta.push(0.4);
        }
        theta.push(0.45);
        let beta0 = [0.2_f64, 0.8];

        let run = |use_panel: bool| {
            let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
            let (p, nt) = (ws.p, ws.n_theta);
            let mut prm = vec![0.0; nt + p];
            prm[..nt].copy_from_slice(&theta);
            prm[nt..].copy_from_slice(&beta0);
            ws.pattern.structured_schur = if use_panel {
                Some(
                    StructuredSchur::new(&ws.groupings, &ids, &extra_ids, n)
                        .expect("ncr > 0 ⇒ e > 0 ⇒ Some"),
                )
            } else {
                None
            };
            // force_dense: same factor arm both runs — isolate the downdate.
            ws.pattern.force_dense_schur = true;
            let mut beta = beta0;
            let GlmmWorkspace {
                groupings,
                z_buf,
                prior_w,
                pirls,
                structured,
                pattern,
                ..
            } = &mut ws;
            build_packed_m(
                groupings,
                &prm[..],
                z_buf,
                &extra_ids,
                &mut pirls.lam,
                &ids,
                &mut structured.m_core_buf,
                &mut structured.cross_theta,
                &mut pattern.cross_col,
                &mut pattern.n_cross,
                n,
            );
            build_coupling_csr(
                &ids,
                &pattern.cross_col,
                &pattern.n_cross,
                groupings.n_primary,
                n,
                &mut pattern.coup_cols,
                &mut pattern.coup_ptr,
                &mut pattern.coup_pos,
            );
            let mut wx_scratch = faer::Mat::<f64>::zeros(n, 2);
            let out = pirls_solve_blocked_extras(
                crate::Family::Binomial {
                    link: crate::BinomialLink::Logit,
                },
                f64::NAN,
                groupings,
                &ids,
                xf64.as_ref(),
                &y,
                &prior_w[..n],
                false,
                &mut beta,
                BetaStep::Fixed,
                pirls,
                structured,
                pattern,
                None, // dual
                &mut wx_scratch,
                None, // offset
                None,
                n,
                &mut crate::counters::EvalCounters::new(),
            );
            (out, pirls.u.clone())
        };
        let (scalar, u_scalar) = run(false);
        let (panel, u_panel) = run(true);
        assert_eq!(scalar.3, panel.3, "{label}: convergence flag");
        assert!(
            (scalar.0 - panel.0).abs() <= tol,
            "{label} dev: scalar {} panel {}",
            scalar.0,
            panel.0
        );
        assert!(
            (scalar.1 - panel.1).abs() <= tol,
            "{label} pen: scalar {} panel {}",
            scalar.1,
            panel.1
        );
        assert!(
            (scalar.2 - panel.2).abs() <= tol,
            "{label} logdet: scalar {} panel {}",
            scalar.2,
            panel.2
        );
        for (c, (a, b)) in u_scalar.iter().zip(u_panel.iter()).enumerate() {
            assert!(
                (a - b).abs() <= u_tol,
                "{label} u[{c}]: scalar {a} panel {b}"
            );
        }
    }
}

/// Parses `validation/data/empirical/grouseticks.csv` into the 3-crossed `TICKS ~ YEAR +
/// cHEIGHT + (1|INDEX) + (1|BROOD) + (1|LOCATION)` design (observation-level
/// INDEX primary + crossed BROOD, LOCATION). Returns the `SizedSpec`
/// (crossed widths resolved from the actual level counts via
/// `spec_sized_from_ids`), the `GroupIds`, the `n×4` design `X` (`[1, YEAR==96, YEAR==97,
/// cHEIGHT]`), the `TICKS` response `y`, and `(n, p)`. Shared by the
/// structured tests below; mirrors `fit::glmm_tests::grouseticks_3crossed_inputs`.
fn grouseticks_3crossed_fixture() -> (SizedSpec, crate::GroupIds, Mat<f64>, Vec<f64>, usize, usize)
{
    fn dense_ids(raw: &[u32]) -> Vec<u32> {
        use std::collections::HashMap;
        let mut map: HashMap<u32, u32> = HashMap::new();
        let mut next = 0u32;
        raw.iter()
            .map(|&r| {
                *map.entry(r).or_insert_with(|| {
                    let v = next;
                    next += 1;
                    v
                })
            })
            .collect()
    }
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

    let csv = include_str!("../../validation/data/empirical/grouseticks.csv");
    // cols: INDEX,TICKS,BROOD,HEIGHT,YEAR,LOCATION,cHEIGHT
    let p = 4;
    let mut rows = Vec::<[f64; 4]>::new();
    let mut y = Vec::<f64>::new();
    let mut index_raw = Vec::<u32>::new();
    let mut brood_raw = Vec::<String>::new();
    let mut loc_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        index_raw.push(f[0].parse().unwrap());
        let year: u32 = f[4].parse().unwrap();
        rows.push([
            1.0,
            f64::from(u32::from(year == 96)),
            f64::from(u32::from(year == 97)),
            f[6].parse().unwrap(), // cHEIGHT
        ]);
        y.push(f[1].parse().unwrap()); // TICKS
        brood_raw.push(f[2].to_string());
        loc_raw.push(f[5].to_string());
    }
    let n = y.len();
    let mut x = Mat::<f64>::zeros(n, p);
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            x[(i, j)] = v;
        }
    }
    let index_ids = dense_ids(&index_raw);
    let brood_ids = dense_str(&brood_raw);
    let loc_ids = dense_str(&loc_raw);

    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: Some(ReStructure {
            terms: Vec::new(),
            slopes: vec![],
            extra_groupings: vec![
                Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes: vec![],
                },
                Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes: vec![],
                },
            ],
        }),
    };
    let ids = crate::GroupIds {
        primary: index_ids,
        extra: vec![brood_ids, loc_ids],
    };
    // Level counts from the ids, as the fit entry sizes them before building
    // `LmmGroupings`.
    let (sized, ids, _perm) = crate::fit::spec_sized_from_ids(&model, &ids);
    (sized, ids.into_owned(), x, y, n, p)
}

/// Structured cold-start overshoot gate (the documented grouseticks
/// degenerate-fit bug): drive `glmm_laplace_deviance` on the 3-crossed Poisson
/// shape at θ = 1.0 (all three components) from a β = 0 cold start — historically
/// the fi.specrst structured PIRLS step overshot into a ~1e30-weight regime, the
/// crossed Schur went non-PD, and the kernel returned a non-finite deviance.
/// Retrospective step-halving in `pirls_solve_blocked_extras` recovers it, so the
/// raw kernel (no `fit_glmm` degenerate-fit guard) must now be finite.
#[test]
fn structured_cold_start_overshoot_is_finite() {
    let (model, ids, x, y, n, p) = grouseticks_3crossed_fixture();
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
    assert!(
        ws.groupings.structured_extras_eligible(),
        "grouseticks 3-crossed must route through the structured extras path"
    );
    // θ = 1.0 for every component (INDEX, BROOD, LOCATION); β = 0 cold start.
    let nt = ws.n_theta;
    for t in 0..nt {
        ws.params[t] = 1.0;
    }
    for j in 0..p {
        ws.params[nt + j] = 0.0;
    }
    let dev = glmm_laplace_deviance(
        &ws.params.clone(),
        &mut ws,
        x.as_ref(),
        &y,
        &ids.primary,
        &ids.extra,
        n,
    );
    assert!(
            dev.is_finite(),
            "structured cold-start deviance must be finite (step-halving recovers the overshoot), got {dev}"
        );
    // Sanity ceiling, not a precision bound: the guarded bug returned
    // INFINITY, so `is_finite` alone would also pass a recovery that landed
    // somewhere absurd. A run of this fixture converges to dev≈997; 2000 is
    // a generous 2× headroom against that, still well below "absurd".
    assert!(dev < 2000.0, "recovered deviance {dev} implausibly large");
}

/// Runs one GLMM shape through `fit_glmm` both ways — single-stage
/// (`outer_search = OuterSearch::Joint`) and the shape's default route — and
/// asserts the two optimizer paths land on the same optimum within the ORACLE
/// tolerances (β̂: beta_rel 1e-3; θ̂: abs+rel 1e-3 band; τ̂²: stddev_rel 1e-3).
/// Returns `(n_eval_single, n_eval_two)` for the baseline-doc print (the
/// eval-count win is a separate, measured concern; it is NOT gated here). Both
/// workspaces are built fresh from the same spec/data; only `outer_search`
/// differs.
#[allow(clippy::too_many_arguments)]
fn assert_two_stage_matches_single(
    label: &str,
    spec: &SizedSpec,
    x: faer::MatRef<f64>,
    y: &[f64],
    primary: &[u32],
    extra: &[Vec<u32>],
    n: usize,
    p: usize,
) -> (usize, usize) {
    let targets = [if p >= 2 { 1u32 } else { 0u32 }];
    let beta_start = vec![0.0_f64; p];

    let mut ws1 = GlmmWorkspace::for_cluster_spec(p, spec, n, &[], 1);
    ws1.outer_search = OuterSearch::Joint; // pin the single-stage reference
    let fit1 = fit_glmm(
        &mut ws1,
        x,
        x.subcols(0, 0),
        y,
        primary,
        extra,
        &targets,
        None,
        &beta_start,
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fit1.converged, "{label}: single-stage fit must converge");

    // Second workspace keeps the constructor's default route — the A/B is
    // "default route vs `Joint`", not "PQL warm start vs `Joint`" specifically.
    let mut ws2 = GlmmWorkspace::for_cluster_spec(p, spec, n, &[], 1);
    let fit2 = fit_glmm(
        &mut ws2,
        x,
        x.subcols(0, 0),
        y,
        primary,
        extra,
        &targets,
        None,
        &beta_start,
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fit2.converged, "{label}: two-stage fit must converge");

    let nt = ws1.n_theta;
    for j in 0..p {
        let (a, b) = (ws1.betas[j], ws2.betas[j]);
        let rel = (a - b).abs() / a.abs().max(1e-6);
        assert!(
            rel < 1e-3,
            "{label}: β[{j}] single {a} vs two-stage {b} (rel {rel})"
        );
    }
    for t in 0..nt {
        let (a, b) = (ws1.params[t], ws2.params[t]);
        // abs+rel band: θ can pin at 0 or sit near the boundary where a pure
        // relative test is ill-conditioned.
        assert!(
            (a - b).abs() < 1e-3 * (1.0 + a.abs()),
            "{label}: θ[{t}] single {a} vs two-stage {b}"
        );
    }
    let (ta, tb) = (ws1.params[0].powi(2), ws2.params[0].powi(2));
    let trel = (ta - tb).abs() / ta.abs().max(1e-6);
    assert!(
        trel < 1e-3,
        "{label}: τ² single {ta} vs two-stage {tb} (rel {trel})"
    );
    (fit1.n_eval, fit2.n_eval)
}

/// A/B: the default outer route reaches the same optimum as the single joint
/// solve on the grouseticks 3-crossed Poisson fixture (the flagship structured
/// shape) — grouseticks (canonical log link, structured extras) now defaults
/// to `OuterSearch::ExactProfile`, not `PqlThenJoint`. θ̂/β̂/τ̂² agree within
/// oracle tolerances — NOT bit-identity (two optimizer paths). n_eval both
/// ways is printed as a baseline doc; there is deliberately NO n_eval
/// assertion (the eval-count win is a separate, measured concern).
#[test]
fn two_stage_matches_single_stage_on_grouseticks() {
    let (model, ids, x, y, n, p) = grouseticks_3crossed_fixture();
    let (ne_single, ne_two) = assert_two_stage_matches_single(
        "grouseticks",
        &model,
        x.as_ref(),
        &y,
        &ids.primary,
        &ids.extra,
        n,
        p,
    );
    println!("grouseticks n_eval: single-stage {ne_single} vs two-stage {ne_two}");
}

/// What the exact-profile helpers below take: `(spec, primary ids, extra-grouping
/// ids, X, y, n, p)`. An empty `extra` marks a no-extras (blocked-path) shape;
/// a non-empty one routes the same helper through the structured-extras solve.
type ExactProfileFixture = (
    SizedSpec,
    Vec<u32>,
    Vec<Vec<u32>>,
    Mat<f64>,
    Vec<f64>,
    usize,
    usize,
);

/// The blocked no-extras intercept fixture (canonical logit link) the corpus
/// sweep above builds inline — pulled out so exact-profile tests can reuse it
/// without duplicating the dataset/spec wiring.
fn logit_intercept_fixture() -> ExactProfileFixture {
    let (x, y, ids) = glmm_intercept_dataset();
    let spec = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: 8,
        extra_counts: vec![],
    };
    (spec, ids, Vec::new(), x, y, 80, 2)
}

/// Its probit twin (blocked, non-canonical link) — the shape exact-mode
/// profiling exists for, since the Laplace merit and the PQL merit only
/// diverge off the canonical link.
fn probit_intercept_fixture() -> ExactProfileFixture {
    let (x, y, ids) = glmm_intercept_dataset();
    let mut spec = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: 8,
        extra_counts: vec![],
    };
    spec.spec.family = Family::Binomial {
        link: BinomialLink::Probit,
    };
    (spec, ids, Vec::new(), x, y, 80, 2)
}

/// The `sim_nb` validation fixture as a blocked NB-log shape (24 clusters,
/// `p = 3`). Non-canonical link with a live `nb_theta`, which is the pairing the
/// merit's accept band is calibrated against — the other two fixtures leave
/// `nb_theta` NaN.
fn sim_nb_intercept_fixture() -> ExactProfileFixture {
    let (xv, y, ids, n_clusters) = crate::fit::common_tests::sim_clustered(include_str!(
        "../../validation/data/simulated/sim_nb.csv"
    ));
    let (n, p) = (y.len(), 3);
    let mut x = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            x[(i, j)] = xv[i * p + j];
        }
    }
    let spec = SizedSpec {
        spec: ModelSpec {
            family: Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![],
            }),
        },
        n_primary: n_clusters,
        extra_counts: vec![],
    };
    (spec, ids, Vec::new(), x, y, n, p)
}

/// The 6-level crossed logit extras shape the corpus sweep builds as
/// `extras_crossed` (structured path, canonical link, `q_core = 1`, `e = 6`).
fn extras_crossed_logit_fixture() -> ExactProfileFixture {
    let (x, y, ids, extra_ids, spec) = glmm_extras_q1_dataset(0, 6);
    let n = y.len();
    (spec, ids, extra_ids, x, y, n, 2)
}

/// Grouseticks, INDEX primary + BROOD/LOCATION crossed (structured path,
/// Poisson-log = canonical, `e = 181`) — the flagship structured rung.
fn grouseticks_exact_fixture() -> ExactProfileFixture {
    let (spec, ids, x, y, n, p) = grouseticks_3crossed_fixture();
    (spec, ids.primary, ids.extra, x, y, n, p)
}

/// An exact-Profile solve at θ returns (ũ, β̂) such that a Fixed-mode
/// solve at that β̂ reproduces the same Laplace objective — the profile value IS
/// the objective at the profiled point, not a PQL surrogate.
#[test]
fn exact_profile_value_equals_fixed_mode_value_at_profiled_beta() {
    let (model, ids, extra, x, y, n, p) = probit_intercept_fixture();
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
    let mut ctrs = EvalCounters::new();
    ws.params[..ws.n_theta].fill(0.7);
    ws.beta_prof[..p].fill(0.0);
    ws.pirls.u.fill(0.0);
    let prof = laplace_deviance_ws(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &extra,
        n,
        BetaMode::ProfileExact,
        &mut ctrs,
    );
    assert!(prof.is_finite(), "exact profile solve must converge");
    let beta_hat = ws.beta_prof[..p].to_vec();
    // Fixed mode reads β off `ws.params[n_theta..]` (`laplace_deviance`'s own
    // convention), not off the transient `beta_rhs` scratch buffer.
    let nt = ws.n_theta;
    ws.params[nt..nt + p].copy_from_slice(&beta_hat);
    ws.pirls.u.fill(0.0);
    let fixed = laplace_deviance_ws(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &extra,
        n,
        BetaMode::Fixed,
        &mut ctrs,
    );
    assert!(
        (prof - fixed).abs() < 1e-6 * (1.0 + fixed.abs()),
        "profile {prof} vs fixed {fixed}"
    );
}

/// Builds the fixture's workspace (blocked on the no-extras cells, structured
/// on the extras ones), then hands off to the shared
/// core assertion — see `assert_exact_profile_is_beta_minimum_ws`. The cells
/// are the `#[test]` fns below this pair (logit, probit, extras crossed logit,
/// grouseticks); the structured-extras cells further down use the sibling
/// `assert_structured_exact_profile_is_beta_minimum` instead, since their
/// workspace comes from `extras_fixture`, not an `ExactProfileFixture` tuple.
fn assert_exact_profile_is_beta_minimum(fixture: fn() -> ExactProfileFixture, label: &str) {
    let (model, ids, extra, x, y, n, p) = fixture();
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
    assert_exact_profile_is_beta_minimum_ws(&mut ws, x.as_ref(), &y, &ids, &extra, n, p, label);
}

/// The exact-profile gate in unit form, shared by both callers above and
/// below: at a fixed θ, minimizing the Fixed-mode Laplace deviance over β
/// with BOBYQA lands on (within BOBYQA's own tolerance) the value the exact
/// Profile solve returns in one PIRLS call.
#[allow(clippy::too_many_arguments)]
fn assert_exact_profile_is_beta_minimum_ws(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    ids: &[u32],
    extra: &[Vec<u32>],
    n: usize,
    p: usize,
    label: &str,
) {
    let mut ctrs = EvalCounters::new();
    let n_theta = ws.n_theta;
    ws.params[..n_theta].fill(0.6);
    ws.beta_prof[..p].fill(0.0);
    ws.pirls.u.fill(0.0);
    let prof = laplace_deviance_ws(ws, x, y, ids, extra, n, BetaMode::ProfileExact, &mut ctrs);
    let beta_hat = ws.beta_prof[..p].to_vec();
    // β-only BOBYQA on the Fixed objective from β = 0.
    let mut cfg = bobyqa::Config::new(p);
    cfg.rho_begin = 0.5;
    cfg.rho_end = 1e-8;
    let mut solver = bobyqa::Bobyqa::new(p, cfg).unwrap();
    let mut b0 = vec![0.0; p];
    let lo = vec![-20.0; p];
    let hi = vec![20.0; p];
    let out = solver.minimize(
        |b| {
            ws.params[n_theta..n_theta + p].copy_from_slice(b);
            ws.pirls.u.fill(0.0);
            laplace_deviance_ws(ws, x, y, ids, extra, n, BetaMode::Fixed, &mut ctrs)
        },
        &mut b0,
        &lo,
        &hi,
    );
    assert!(
        matches!(out.status, bobyqa::Status::Converged),
        "{label}: β-only BOBYQA"
    );
    let joint_min = out.f;
    assert!(
        prof - joint_min < 1e-7 * (1.0 + joint_min.abs()),
        "{label}: profile {prof} above β-minimum {joint_min}"
    );
    for j in 0..p {
        assert!(
            (beta_hat[j] - b0[j]).abs() < 1e-4,
            "{label}: β̂[{j}] profile {} vs bobyqa {}",
            beta_hat[j],
            b0[j]
        );
    }
}

/// The structured-extras twin of `assert_exact_profile_is_beta_minimum`:
/// same assertion, on `extras_fixture`'s `nested2_crossed6` shape (`q_core =
/// 3`, `e = 6`) instead of a blocked no-extras fixture. Cells are the
/// non-canonical links the observed-information twin exists for — probit,
/// cloglog, NB-log — forced through `BetaMode::ProfileExact` directly rather
/// than through `fit_glmm`, isolating the profile solve itself from route
/// selection.
fn assert_structured_exact_profile_is_beta_minimum(family: Family, label: &str) {
    let (mut ws, x, y, ids, extra, p, n) = extras_fixture(family, 2, 6);
    assert_exact_profile_is_beta_minimum_ws(&mut ws, x.as_ref(), &y, &ids, &extra, n, p, label);
}

/// The exact border's `S_β + ½·d²log|A|/dβ²` against half the Hessian of the
/// Laplace β-profile. At the profiled β̂ (θ = `theta`) the curvature is
/// computed by `logdet_beta_curvature` on the solve's final state, since the
/// trigger rule (`BorderTrust::wants_curvature`) may never compute it in the
/// solve. Its `W′`, `W″` are formed at the returned η and μ
/// (`evaluate_at_mode`'s), one converged step past where the last iteration
/// formed `hᵢ` (pass A), `mᵢ′v` (pass C) and `A⁻¹M′WX` (the border). It is
/// compared with a central difference, step 3e-4, of the
/// Fixed-mode objective, whose PIRLS solves run at a 1e-13 band; the band is
/// 1e-5 of the largest entry. Without a crossed tail the two must agree
/// within the band. With one, the curvature leaves out the tail's Gram terms
/// and is an upper bound, so bound minus difference must be positive
/// semidefinite within the band. Measured, relative to the largest entry:
/// without a tail the two agree to 1.3e-8; with one, the smallest eigenvalue
/// of bound minus difference is +1.6e-6 (grouseticks) to +1.7e-4, and the
/// bound exceeds the difference by up to 1.3e-2 (grouseticks). `S_β` alone
/// misses by 2.1e-4 (grouseticks) to 3.0e-2.
#[allow(clippy::too_many_arguments)]
fn assert_border_curvature_matches_fd(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    ids: &[u32],
    extra: &[Vec<u32>],
    n: usize,
    p: usize,
    theta: f64,
    label: &str,
) {
    let mut ctrs = EvalCounters::new();
    let nt = ws.n_theta;
    ws.params[..nt].fill(theta);
    ws.beta_prof[..p].fill(0.0);
    ws.pirls.u.fill(0.0);
    let prof = laplace_deviance_ws(ws, x, y, ids, extra, n, BetaMode::ProfileExact, &mut ctrs);
    assert!(
        prof.is_finite(),
        "{label}: exact profile solve must converge"
    );
    let beta_hat = ws.beta_prof[..p].to_vec();
    let structured = ws.layout == GlmmLayout::Structured;
    let e = ws.groupings.k_crossed();
    {
        let GlmmWorkspace {
            groupings,
            pirls,
            structured: st,
            pattern,
            border,
            exact_prof,
            family,
            nb_theta,
            prior_w,
            ..
        } = &mut *ws;
        let tail = CrossedTail {
            cross_col: &pattern.cross_col[..],
            cross_theta: &st.cross_theta[..],
            n_cross: &pattern.n_cross[..],
            e,
        };
        let (m, tail) = if structured {
            (&st.m_core_buf[..], Some(&tail))
        } else {
            (&pirls.m_buf[..], None)
        };
        logdet_beta_curvature(
            exact_prof,
            x,
            border.ainv_mtwx.as_ref(),
            m,
            ids,
            groupings,
            tail,
            *family,
            *nb_theta,
            crate::family::exact_curvature_differs(*family),
            y,
            prior_w,
            &pirls.eta[..],
            &pirls.prob[..],
            n,
            p,
        );
    }
    let curv = Mat::<f64>::from_fn(p, p, |j, k| {
        2.0 * (ws.exact_prof.schur_plain[(j, k)] + ws.exact_prof.logdet_hess[(j, k)])
    });
    ws.fd.pirls_tol_override = Some(1e-13);
    let h = 3e-4;
    let mut f = |dj: (usize, f64), dk: (usize, f64)| {
        ws.params[nt..nt + p].copy_from_slice(&beta_hat);
        ws.params[nt + dj.0] += dj.1;
        ws.params[nt + dk.0] += dk.1;
        ws.pirls.u.fill(0.0);
        laplace_deviance_ws(ws, x, y, ids, extra, n, BetaMode::Fixed, &mut ctrs)
    };
    let mut fd = Mat::<f64>::zeros(p, p);
    for j in 0..p {
        for k in 0..=j {
            let v = (f((j, h), (k, h)) - f((j, h), (k, -h)) - f((j, -h), (k, h))
                + f((j, -h), (k, -h)))
                / (4.0 * h * h);
            fd[(j, k)] = v;
            fd[(k, j)] = v;
        }
    }
    ws.fd.pirls_tol_override = None;
    let scale = (0..p)
        .flat_map(|j| (0..p).map(move |k| (j, k)))
        .map(|(j, k)| fd[(j, k)].abs())
        .fold(0.0_f64, f64::max);
    if structured && e > 0 {
        // Smallest eigenvalue of bound − FD at least −band ⇔ bound − FD +
        // band·I is positive definite.
        let d = Mat::<f64>::from_fn(p, p, |j, k| {
            curv[(j, k)] - fd[(j, k)] + if j == k { 1e-5 * scale } else { 0.0 }
        });
        assert!(
            d.as_ref().llt(faer::Side::Lower).is_ok(),
            "{label}: border curvature bound {curv:?} minus FD of the profile {fd:?} is not \
             positive semidefinite (scale {scale})"
        );
        return;
    }
    for j in 0..p {
        for k in 0..p {
            assert!(
                (curv[(j, k)] - fd[(j, k)]).abs() <= 1e-5 * scale,
                "{label}: border curvature [{j},{k}] {} vs FD of the profile {} (scale {scale})",
                curv[(j, k)],
                fd[(j, k)]
            );
        }
    }
}

#[test]
fn exact_border_curvature_matches_fd_of_the_profile() {
    let blocked = [
        (
            logit_intercept_fixture as fn() -> ExactProfileFixture,
            "logit",
            f64::NAN,
        ),
        (probit_intercept_fixture, "probit", f64::NAN),
        (sim_nb_intercept_fixture, "sim_nb", 3.0),
    ];
    for (fixture, label, nb_theta) in blocked {
        let (model, ids, extra, x, y, n, p) = fixture();
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
        ws.nb_theta = nb_theta;
        assert_border_curvature_matches_fd(&mut ws, x.as_ref(), &y, &ids, &extra, n, p, 0.8, label);
    }
    let (x, y, ids) = glmm_slope2_noextra_dataset();
    let mut ws = GlmmWorkspace::for_cluster_spec(3, &slope2_spec(), 120, &[1, 2], 1);
    assert_border_curvature_matches_fd(&mut ws, x.as_ref(), &y, &ids, &[], 120, 3, 0.5, "slope2");
    for (family, label) in [
        (
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            "structured nb",
        ),
        (
            Family::Binomial {
                link: BinomialLink::Probit,
            },
            "structured probit",
        ),
    ] {
        let (mut ws, x, y, ids, extra, p, n) = extras_fixture(family, 2, 6);
        assert_border_curvature_matches_fd(&mut ws, x.as_ref(), &y, &ids, &extra, n, p, 0.6, label);
    }
    let (model, ids, extra, x, y, n, p) = grouseticks_exact_fixture();
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
    ws.pattern.structured_schur = StructuredSchur::new(&ws.groupings, &ids, &extra, n);
    assert_border_curvature_matches_fd(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &extra,
        n,
        p,
        0.7,
        "grouseticks",
    );
}

/// The exact-Profile solve must converge when PIRLS is warm-started from the
/// converged (ũ, β̂) of a LOWER θ — the directional warm start the θ-only outer
/// search does on every uphill probe. A cold start at the same θ converges, so a
/// non-finite value here is a globalization defect in the exact border, not a
/// property of the objective.
#[test]
fn exact_profile_converges_from_lower_theta_warm_start() {
    const WIDE: [(f64, f64); 4] = [(0.7, 0.9), (1.0, 1.1), (0.5, 0.6), (0.45, 0.65)];
    assert_exact_profile_warm_start(logit_intercept_fixture, "logit", f64::NAN, &WIDE);
    assert_exact_profile_warm_start(probit_intercept_fixture, "probit", f64::NAN, &WIDE);
    // `sim_nb` at a FIXED θ_NB near the optimum, not the fixture's own θ̂_NB,
    // on the θ steps the θ-only search takes there (θ̂ ≈ 0.5742). Every one of
    // these returned a non-finite value while the accept band charged neither
    // endpoint's mode-consistency correction: the outer search read them as
    // `inf`, stopped at θ = 0.5567 and
    // put β₀ 15% off lme4. The steps are deliberately small — the failure needs a
    // warm start already close to the mode, so a coarse pair like (0.5, 0.6) can
    // miss it.
    const NEAR: [(f64, f64); 4] = [
        (0.5567410345061036, 0.5569180345061036),
        (0.5567410345061036, 0.5584201322395581),
        (0.5567410345061036, 0.575),
        (0.5567410345061036, 0.58),
    ];
    assert_exact_profile_warm_start(
        sim_nb_intercept_fixture,
        "sim_nb",
        1.7832168350542466,
        &NEAR,
    );
}

/// `nb_theta` is NaN on every family that does not estimate one (`fit_glmm_impl`
/// threads it the same way); `pairs` are `(θ_cold, θ_warm)` probes in the RE
/// scale.
fn assert_exact_profile_warm_start(
    fixture: fn() -> ExactProfileFixture,
    label: &str,
    nb_theta: f64,
    pairs: &[(f64, f64)],
) {
    let (model, ids, extra, x, y, n, p) = fixture();
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
    ws.nb_theta = nb_theta;
    let mut ctrs = EvalCounters::new();
    let nt = ws.n_theta;
    for &(theta_lo, theta_hi) in pairs {
        ws.params[..nt].fill(theta_lo);
        ws.beta_prof[..p].fill(0.0);
        ws.pirls.u.fill(0.0);
        let lo = laplace_deviance_ws(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra,
            n,
            BetaMode::ProfileExact,
            &mut ctrs,
        );
        assert!(
            lo.is_finite(),
            "{label}: cold solve at theta={theta_lo} must converge"
        );
        let u_warm = ws.pirls.u.clone();
        let beta_warm = ws.beta_prof[..p].to_vec();
        // Uphill probe, warm-started exactly as the outer search does.
        ws.params[..nt].fill(theta_hi);
        ws.pirls.u.copy_from_slice(&u_warm);
        ws.beta_prof[..p].copy_from_slice(&beta_warm);
        let hi_warm = laplace_deviance_ws(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra,
            n,
            BetaMode::ProfileExact,
            &mut ctrs,
        );
        // Cold reference at the same theta — the value the warm solve must match.
        ws.params[..nt].fill(theta_hi);
        ws.beta_prof[..p].fill(0.0);
        ws.pirls.u.fill(0.0);
        let hi_cold = laplace_deviance_ws(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra,
            n,
            BetaMode::ProfileExact,
            &mut ctrs,
        );
        assert!(
            hi_warm.is_finite(),
            "{label}: warm start {theta_lo} -> {theta_hi}: got {hi_warm}, cold gives {hi_cold}"
        );
        assert!(
            (hi_warm - hi_cold).abs() < 1e-6 * (1.0 + hi_cold.abs()),
            "{label}: warm start {theta_lo} -> {theta_hi}: warm {hi_warm} vs cold {hi_cold}"
        );
    }
}

#[test]
fn exact_profile_is_beta_minimum_logit() {
    assert_exact_profile_is_beta_minimum(logit_intercept_fixture, "logit");
}
#[test]
fn exact_profile_is_beta_minimum_probit() {
    assert_exact_profile_is_beta_minimum(probit_intercept_fixture, "probit");
}
#[test]
fn exact_profile_is_beta_minimum_extras_crossed_logit() {
    assert_exact_profile_is_beta_minimum(extras_crossed_logit_fixture, "extras crossed logit");
}
#[test]
fn exact_profile_is_beta_minimum_grouseticks() {
    assert_exact_profile_is_beta_minimum(grouseticks_exact_fixture, "grouseticks");
}
#[test]
fn exact_profile_is_beta_minimum_structured_probit() {
    assert_structured_exact_profile_is_beta_minimum(
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        "structured probit nested2_crossed6",
    );
}
#[test]
fn exact_profile_is_beta_minimum_structured_cloglog() {
    assert_structured_exact_profile_is_beta_minimum(
        Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        "structured cloglog nested2_crossed6",
    );
}
#[test]
fn exact_profile_is_beta_minimum_structured_nb_log() {
    assert_structured_exact_profile_is_beta_minimum(
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        "structured NB-log nested2_crossed6",
    );
}

/// The structured twin of `exact_profile_converges_from_lower_theta_warm_start`:
/// the crossed extras path must not jam on the directional warm start either.
/// Separate test so a failure names the structured solve, not the blocked one.
#[test]
fn exact_profile_converges_from_lower_theta_warm_start_extras() {
    assert_exact_profile_warm_start(
        extras_crossed_logit_fixture,
        "extras crossed logit",
        f64::NAN,
        &[(0.7, 0.9), (1.0, 1.1), (0.5, 0.6), (0.45, 0.65)],
    );
}

/// Structured inference (Var(β̂)_jj, z²) matches an independent Schur
/// recomputation on all three Group-G shapes — the structured `A⁻¹` apply in
/// `structured_schur_fill` is the same estimator as the plain `k×k` one. Runs
/// the full `fit_glmm` (structured), then rebuilds `M` as a full `n×k` matrix
/// out of a packed workspace's rows (`packed_dense_m`) and recomputes Var(β̂) by
/// hand from the converged `ws.pirls.w` and `ws.params`. Mirrors
/// `blocked_inference_matches_packed_slope_noextra`.
#[test]
fn structured_inference_matches_packed() {
    for (np, ncr, label) in [
        (0usize, 6usize, "crossed"),
        (2, 0, "nested"),
        (2, 6, "crossed_nested"),
    ] {
        let (xf64, y, ids, extra_ids, cluster) = glmm_extras_q1_dataset(np, ncr);
        let n = y.len();
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
        assert!(
            ws.groupings.structured_extras_eligible(),
            "{label}: eligible"
        );
        let fit = fit_glmm(
            &mut ws,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &y,
            &ids,
            &extra_ids,
            &[1],
            None,
            &[0.2, 0.8],
            n,
            WaldSe::Rx,
            ColumnScaling::IDENTITY,
        );
        assert!(fit.converged, "{label}: fit must converge");
        let (k, p, nt) = (ws.k, ws.p, ws.n_theta);
        let var_structured = ws.inference.var_diag[1];
        let tsq_structured = ws.inference.t_sq[1];
        let beta1 = ws.betas[1];
        // Dense recompute of Var(β̂)_11 from the converged state: M = ZΛ̂ at θ̂,
        // scattered out of the packed rows.
        let params: Vec<f64> = ws.params[..nt].to_vec();
        let mut wsp = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[], 1);
        fill_packed_cols(&mut wsp, &ids, &extra_ids, n);
        packed_oracle_fill(&mut wsp, xf64.as_ref(), &params, n);
        let m = packed_dense_m(&wsp, n, k);
        let mut xtwx = Mat::<f64>::zeros(p, p);
        for r in 0..p {
            for c in 0..p {
                let mut sm = 0.0;
                for i in 0..n {
                    sm += xf64[(i, r)] * ws.pirls.w[i] * xf64[(i, c)];
                }
                xtwx[(r, c)] = sm;
            }
        }
        let mut xtwm = Mat::<f64>::zeros(p, k);
        for r in 0..p {
            for c in 0..k {
                let mut sm = 0.0;
                for i in 0..n {
                    sm += xf64[(i, r)] * ws.pirls.w[i] * m[(i, c)];
                }
                xtwm[(r, c)] = sm;
            }
        }
        let mut a = Mat::<f64>::zeros(k, k);
        for r in 0..k {
            for c in 0..k {
                let mut sm = if r == c { 1.0 } else { 0.0 };
                for i in 0..n {
                    sm += m[(i, r)] * ws.pirls.w[i] * m[(i, c)];
                }
                a[(r, c)] = sm;
            }
        }
        let ac = a.as_ref().llt(faer::Side::Lower).unwrap();
        let mut ainv = Mat::<f64>::zeros(k, p);
        for r in 0..k {
            for c in 0..p {
                ainv[(r, c)] = xtwm[(c, r)];
            }
        }
        ac.solve_in_place(ainv.as_mut());
        let mut schur = Mat::<f64>::zeros(p, p);
        for r in 0..p {
            for c in 0..p {
                let mut sm = xtwx[(r, c)];
                for j in 0..k {
                    sm -= xtwm[(r, j)] * ainv[(j, c)];
                }
                schur[(r, c)] = sm;
            }
        }
        let sc = schur.as_ref().llt(faer::Side::Lower).unwrap();
        let mut fwd = vec![0.0; p];
        for i in 0..p {
            let mut acc = if i == 1 { 1.0 } else { 0.0 };
            #[allow(clippy::needless_range_loop)]
            for kk in 0..i {
                acc -= sc.L()[(i, kk)] * fwd[kk];
            }
            fwd[i] = acc / sc.L()[(i, i)];
        }
        let var_packed: f64 = fwd.iter().map(|v| v * v).sum();
        assert!(
            (var_structured - var_packed).abs() < 1e-8,
            "{label} var: structured {var_structured} packed {var_packed}"
        );
        let tsq_packed = beta1 * beta1 / var_packed;
        assert!(
            (tsq_structured - tsq_packed).abs() < 1e-6,
            "{label} z²: structured {tsq_structured} packed {tsq_packed}"
        );
    }
}

/// Packed-path (`pirls_solve_packed`) overshoot fixture. The primary grouping
/// carries 8 slopes (q_p = 9 > MAX_PRIMARY_Q = 8), so `classify_design` puts the
/// design over the NoZ envelope and `GlmmLayout::for_design` gives it the
/// packed-row layout (not the blocked / structured ones). Poisson-log with large
/// counts (~5000) at the β = 0, u = 0 cold start drives the first full Fisher
/// step into the exp() blow-up regime (the grouseticks non-convergence class).
/// The crossed ids enter the packed columns through `fill_packed_cols` here,
/// since `glmm_laplace_deviance` threads only the primary ids. Verified to
/// return INFINITY on the pre-step-halving packed loop (non-convergence).
fn packed_path_overshoot_fixture() -> (GlmmWorkspace, Mat<f64>, Vec<f64>, Vec<u32>, usize) {
    let (n, n_prim, n_crossed, n_slopes) = (48usize, 8usize, 4usize, 8usize);
    let mut st = 20240703u64;
    let ncol = 1 + n_slopes; // intercept (fixed) + slope predictor columns
    let mut x = Mat::<f64>::zeros(n, ncol);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    let mut crossed = vec![0u32; n];
    for i in 0..n {
        ids[i] = (i % n_prim) as u32;
        crossed[i] = (i % n_crossed) as u32;
        x[(i, 0)] = 1.0;
        for d in 0..n_slopes {
            x[(i, 1 + d)] = lcg(&mut st); // slope predictors in [-0.5, 0.5]
        }
        y[i] = (10.0 + 4990.0 * (lcg(&mut st) + 0.5)).round(); // counts up to ~5000
    }
    let slope_cols: Vec<usize> = (1..=n_slopes).collect(); // x cols 1..=8 as RE slopes
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: (1..=n_slopes as u32).collect(),
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes: vec![],
                }],
            }),
        },
        n_primary: n_prim,
        extra_counts: vec![n_crossed],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(1, &cluster, n, &slope_cols, 1);
    fill_packed_cols(&mut ws, &ids, &[crossed], n);
    (ws, x, y, ids, n)
}

#[test]
fn pirls_packed_step_halving_recovers_from_overshoot() {
    // Shape/data chosen so the first full Fisher step at the β=0 cold start
    // overshoots η into the exp() blow-up regime (the grouseticks failure
    // class, on the packed path). Before step-halving this returned INFINITY.
    let (mut ws, x, y, ids, n) = packed_path_overshoot_fixture();
    assert_eq!(
        ws.layout,
        GlmmLayout::Packed,
        "fixture must route to pirls_solve_packed"
    );
    let n_params = ws.n_theta + ws.p;
    let params: Vec<f64> = (0..n_params)
        .map(|i| if i < ws.n_theta { 1.0 } else { 0.0 })
        .collect();
    // Packed route (asserted above): `extra_ids` is unread here (the packed
    // columns are already filled by the fixture, and `build_packed_m` never
    // runs), so an empty slice is safe.
    let dev = glmm_laplace_deviance(&params, &mut ws, x.as_ref(), &y, &ids, &[], n);
    assert!(
        dev.is_finite(),
        "step-halving must rescue the cold-start overshoot, got {dev}"
    );
    // Sanity ceiling against a finite-but-absurd recovery (the guarded bug
    // returned INFINITY). A run of this fixture converges to dev≈1269; 2600
    // is a generous 2× headroom above that.
    assert!(dev < 2600.0, "recovered deviance {dev} implausibly large");
}

/// PQL stationarity of the Profile-mode packed β step: after a converged
/// `pirls_solve_packed` in `BetaStep::Profile` (β seeded at 0, θ held at a blind
/// value), the β-gradient X'ρ must vanish — that IS the definition of "β is
/// PQL-optimal for this θ", needing no external oracle. Recompute ρ = y − p̂
/// at the RETURNED (u, β) and bound ‖X'ρ‖∞ by 1e-6·n. Canonical logit
/// (Newton) drives the joint (u,β) step to the PIRLS exit band, so this holds
/// tightly; the bound is tied to `pirls_tol` — loosen it (never the PIRLS
/// tolerance) if a non-canonical family ever falls short.
#[test]
fn pirls_packed_profile_beta_reaches_pql_stationarity() {
    let (xf64, y, ids) = glmm_slope_noextra_dataset();
    let n = y.len();
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        },
        n_primary: 8,
        extra_counts: vec![],
    };
    // The constructor would route this design to the blocked layout; the packed
    // kernel is what this test drives.
    let mut ws = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[1], 1);
    fill_packed_cols(&mut ws, &ids, &[], n);
    let (k, p, nt) = (ws.k, ws.p, ws.n_theta);
    // Blind θ (vech Λ_p), β seed = 0. β is a buffer DISTINCT from ws.beta_rhs
    // (the Profile δβ scratch), per the BetaStep contract.
    let theta = [0.5_f64, 0.1, 0.4];
    let mut params = vec![0.0; nt + p];
    params[..nt].copy_from_slice(&theta);
    let mut beta = vec![0.0_f64; p];
    packed_oracle_fill(&mut ws, xf64.as_ref(), &params, n);
    let m = packed_dense_m(&ws, n, k);
    let GlmmWorkspace {
        prior_w,
        wx,
        beta_rhs,
        pirls,
        packed,
        border:
            BorderScratch {
                xtwx: _,
                xtwm,
                ainv_mtwx,
                schur,
                schur_llt_mem,
                beta_prev,
            },
        ..
    } = &mut ws;
    let out = pirls_solve_packed(
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        f64::NAN,
        k,
        p,
        xf64.as_ref(),
        &y,
        &prior_w[..n],
        false,
        &mut beta,
        BetaStep::Profile {
            exact: None,
            xtwm,
            ainv_mtwx,
            schur,
            beta_rhs,
            beta_prev,
            schur_llt_mem,
        },
        pirls,
        packed,
        wx,
        None, // offset
        None,
        n,
        &mut crate::counters::EvalCounters::new(),
    );
    assert!(out.3, "Profile-mode solve must converge");
    // Recompute ρ = y − p̂ at the RETURNED (u, β): η = Xβ + Mu. The eta/prob
    // left in the workspace are the pre-step trial values of the converging
    // iteration, NOT the returned joint iterate.
    let mref = m.as_ref();
    let mut xr = vec![0.0_f64; p];
    for i in 0..n {
        let mut e = 0.0;
        for j in 0..p {
            e += xf64[(i, j)] * beta[j];
        }
        for c in 0..k {
            e += mref[(i, c)] * pirls.u[c];
        }
        let pi = 1.0 / (1.0 + (-e).exp());
        let rho = y[i] - pi;
        for j in 0..p {
            xr[j] += xf64[(i, j)] * rho;
        }
    }
    #[allow(clippy::needless_range_loop)]
    for j in 0..p {
        assert!(
            xr[j].abs() < 1e-6 * n as f64,
            "β not stationary in coord {j}: X'ρ = {}",
            xr[j]
        );
    }
}

/// PQL stationarity of the Profile-mode blocked β step: blocked twin of
/// `pirls_packed_profile_beta_reaches_pql_stationarity`. After a converged
/// `pirls_solve_blocked` in `BetaStep::Profile` (β seeded at 0, θ held blind),
/// the β-gradient X'ρ must vanish — the PQL-optimality definition, no external
/// oracle. Recompute ρ = y − p̂ at the RETURNED (u, β) with η = Xβ + Mu (M
/// applied blockwise via `m_buf`/`cluster_ids`) and bound ‖X'ρ‖∞ by 1e-6·n.
/// The bound is tied to `pirls_tol` — loosen it (never the PIRLS tolerance)
/// if a non-canonical family ever falls short.
#[test]
fn pirls_blocked_profile_beta_reaches_pql_stationarity() {
    let (xf64, y, ids) = glmm_slope_noextra_dataset();
    let n = y.len();
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        },
        n_primary: 8,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    assert!(
        ws.groupings.extra_offsets.is_empty(),
        "fixture must route to the blocked pirls_solve_blocked (no extras)"
    );
    let p = ws.p;
    let q = ws.groupings.primary_q;
    // Blind θ (vech Λ_p), β seed = 0. β is a buffer DISTINCT from ws.beta_rhs
    // (the Profile δβ scratch), per the BetaStep contract.
    let theta = [0.5_f64, 0.1, 0.4];
    crate::lmm::primary_lambda(&theta, ws.groupings.primary_q, &mut ws.pirls.lam);
    fill_z_f64(
        &ws.groupings,
        xf64.as_ref(),
        xf64.as_ref().subcols(0, 0),
        &mut ws.z_buf,
        n,
    );
    let mut beta = vec![0.0_f64; p];
    let GlmmWorkspace {
        groupings,
        z_buf,
        prior_w,
        beta_rhs,
        wx,
        pirls,
        border:
            BorderScratch {
                xtwx: _,
                xtwm,
                ainv_mtwx,
                schur,
                schur_llt_mem,
                beta_prev,
            },
        ..
    } = &mut ws;
    let out = pirls_solve_blocked(
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        f64::NAN,
        groupings,
        &ids,
        xf64.as_ref(),
        &y,
        &prior_w[..n],
        false,
        &mut beta,
        BetaStep::Profile {
            exact: None,
            xtwm,
            ainv_mtwx,
            schur,
            beta_rhs,
            beta_prev,
            schur_llt_mem,
        },
        pirls,
        z_buf,
        None,
        wx,
        None, // offset
        None,
        n,
        &mut crate::counters::EvalCounters::new(),
    );
    assert!(out.3, "Profile-mode blocked solve must converge");
    // Recompute ρ = y − p̂ at the RETURNED (u, β): η = Xβ + Mu, with Mu applied
    // blockwise (row i loads cluster `ids[i]`'s q_p columns from `m_buf`). The
    // eta/prob left in the workspace are the pre-step trial values, NOT the
    // returned joint iterate.
    let mut xr = vec![0.0_f64; p];
    for i in 0..n {
        let mut e = 0.0;
        for j in 0..p {
            e += xf64[(i, j)] * beta[j];
        }
        let ubase = ids[i] as usize * q;
        for c in 0..q {
            e += pirls.m_buf[i * q + c] * pirls.u[ubase + c];
        }
        let pi = 1.0 / (1.0 + (-e).exp());
        let rho = y[i] - pi;
        for j in 0..p {
            xr[j] += xf64[(i, j)] * rho;
        }
    }
    #[allow(clippy::needless_range_loop)]
    for j in 0..p {
        assert!(
            xr[j].abs() < 1e-6 * n as f64,
            "β not stationary in coord {j}: X'ρ = {}",
            xr[j]
        );
    }
}

/// PQL stationarity of the Profile-mode structured β step: structured twin of
/// the dense/blocked stationarity gates, on the grouseticks 3-crossed Poisson
/// shape (`grouseticks_3crossed_fixture`, INDEX primary + crossed BROOD,
/// LOCATION). After a converged `pirls_solve_blocked_extras` in
/// `BetaStep::Profile` (β seeded at 0, θ held blind), the β-gradient X'ρ must
/// vanish — the PQL-optimality definition, no external oracle. Recompute
/// ρ = (dμ/dη)(y−μ)/V at the RETURNED (u, β) with η = Xβ + Mu (M applied from
/// the packed `m_core_buf`/`cross_*` nonzeros) and bound ‖X'ρ‖∞ by 1e-6·n.
/// The bound is deliberately looser than the tight logit gates' — tied to
/// `pirls_tol`, never the PIRLS tolerance.
/// Re-pinned 2026-10-02: `X'WX` is summed in row order in the β-Profile row
/// pass instead of by a GEMM, and log links read `dμ/dη = μ` from the stored
/// mean; the return moved by 1 to 3 ULP (`dev` by −5.7e-14, 1 ULP lower,
/// `pen` and `logdet` 3 ULP each), `converged` unchanged, the stationarity
/// bound above still met.
#[test]
fn pirls_structured_profile_beta_reaches_pql_stationarity() {
    let (model, ids, x, y, n, p) = grouseticks_3crossed_fixture();
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
    assert!(
        ws.groupings.structured_extras_eligible(),
        "grouseticks 3-crossed must route through the structured extras path"
    );
    let nt = ws.n_theta;
    // Blind θ (one scalar per component: INDEX, BROOD, LOCATION); β seed = 0.
    for t in 0..nt {
        ws.params[t] = 0.5;
    }
    let mut beta = vec![0.0_f64; p];
    {
        let GlmmWorkspace {
            groupings,
            params,
            z_buf,
            pirls: PirlsScratch { lam, .. },
            structured:
                StructuredScratch {
                    m_core_buf,
                    cross_theta,
                    ..
                },
            pattern: StructuredPattern {
                cross_col, n_cross, ..
            },
            ..
        } = &mut ws;
        build_packed_m(
            groupings,
            params,
            z_buf,
            &ids.extra,
            lam,
            &ids.primary,
            m_core_buf,
            cross_theta,
            cross_col,
            n_cross,
            n,
        );
    }
    let out = {
        let GlmmWorkspace {
            groupings,
            prior_w,
            beta_rhs,
            wx,
            pirls,
            structured,
            pattern,
            border:
                BorderScratch {
                    xtwx: _,
                    xtwm,
                    ainv_mtwx,
                    schur,
                    schur_llt_mem,
                    beta_prev,
                },
            ..
        } = &mut ws;
        build_coupling_csr(
            &ids.primary,
            &pattern.cross_col,
            &pattern.n_cross,
            groupings.n_primary,
            n,
            &mut pattern.coup_cols,
            &mut pattern.coup_ptr,
            &mut pattern.coup_pos,
        );
        pirls_solve_blocked_extras(
            Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            f64::NAN,
            groupings,
            &ids.primary,
            x.as_ref(),
            &y,
            &prior_w[..n],
            false,
            &mut beta,
            BetaStep::Profile {
                exact: None,
                xtwm,
                ainv_mtwx,
                schur,
                beta_rhs,
                beta_prev,
                schur_llt_mem,
            },
            pirls,
            structured,
            pattern,
            None, // dual
            wx,
            None, // offset
            None,
            n,
            &mut crate::counters::EvalCounters::new(),
        )
    };
    assert!(out.3, "Profile-mode structured solve must converge");
    // Pinned (`assert_pinned_return` states the contract); the
    // stationarity bound below is a 1e-6·n band and cannot see a small move.
    assert_pinned_return(
        out,
        (
            259.10917864477506,
            224.41367163180058,
            189.17876051534216,
            true,
        ),
        "Profile structured",
    );
    // Recompute ρ at the RETURNED (u, β): η = Xβ + Mu over the packed M nonzeros.
    // The eta/prob left in the workspace are the pre-step trial values, NOT the
    // returned joint iterate.
    let g = &ws.groupings;
    let (q, np, s) = (g.primary_q, g.nested_per_parent, g.n_primary);
    let qc = q + np;
    let prim_width = q * s;
    let k_family = qc * s;
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    let core_col = |f: usize, local: usize| -> usize {
        if local < q {
            f * q + local
        } else {
            prim_width + f * np + (local - q)
        }
    };
    let fam = Family::Poisson {
        link: crate::PoissonLink::Log,
    };
    let mut xr = vec![0.0_f64; p];
    for i in 0..n {
        let f = ids.primary[i] as usize;
        let mut e = 0.0;
        for j in 0..p {
            e += x[(i, j)] * beta[j];
        }
        for local in 0..qc {
            e += ws.structured.m_core_buf[i * qc + local] * ws.pirls.u[core_col(f, local)];
        }
        let cbase = i * g_cap;
        for z in 0..ws.pattern.n_cross[i] as usize {
            let b = ws.pattern.cross_col[cbase + z] as usize;
            e += ws.structured.cross_theta[z] * ws.pirls.u[k_family + b];
        }
        let e = crate::family::clamp_eta(fam, e);
        let mui = crate::family::link_inv(fam, e);
        let dmu = crate::family::mu_eta(fam, e);
        let v = crate::family::variance(fam, f64::NAN, mui);
        let rho = dmu * (y[i] - mui) / v;
        for j in 0..p {
            xr[j] += x[(i, j)] * rho;
        }
    }
    #[allow(clippy::needless_range_loop)]
    for j in 0..p {
        assert!(
            xr[j].abs() < 1e-6 * n as f64,
            "β not stationary in coord {j}: X'ρ = {}",
            xr[j]
        );
    }
}

/// Profile-mode cross-variant consistency: the structured β step must agree with
/// the packed β step at the same θ. On a crossed extras shape the packed
/// `pirls_solve_packed` (a dense `k×k` A) and the structured
/// `pirls_solve_blocked_extras` (core blocks + crossed Schur) are the same
/// estimator (as `structured_extras_matches_packed` proves in Fixed mode); run
/// BOTH in `BetaStep::Profile` from β = 0 and require the converged β to agree
/// to ≤1e-8 rel. Guards a sign/transpose slip in exactly one variant — the
/// structured Schur-border apply is the intricate one this catches.
/// Re-pinned 2026-10-02: `X'WX` is summed in row order in the structured
/// β-Profile row pass instead of by a GEMM; `dev` moved by 1 ULP (−1.4e-14),
/// `pen` by 2 ULP, `logdet` unchanged, `converged` unchanged, the β agreement
/// below still met.
#[test]
fn structured_profile_beta_matches_packed_profile() {
    let (xf64, y, ids, extra_ids, cluster) = glmm_extras_q1_dataset(0, 6);
    let n = y.len();
    let theta = [0.5_f64, 0.45];

    // Packed reference: the same design on the packed-row layout, Profile mode,
    // β seed 0.
    let mut wsd = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[], 1);
    fill_packed_cols(&mut wsd, &ids, &extra_ids, n);
    let (k, p, nt) = (wsd.k, wsd.p, wsd.n_theta);
    let mut params = vec![0.0; nt + p];
    params[..nt].copy_from_slice(&theta);
    let mut beta_packed = vec![0.0_f64; p];
    packed_oracle_fill(&mut wsd, xf64.as_ref(), &params, n);
    let oracle = {
        let GlmmWorkspace {
            prior_w,
            wx,
            beta_rhs,
            pirls,
            packed,
            border:
                BorderScratch {
                    xtwx: _,
                    xtwm,
                    ainv_mtwx,
                    schur,
                    schur_llt_mem,
                    beta_prev,
                },
            ..
        } = &mut wsd;
        pirls_solve_packed(
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            f64::NAN,
            k,
            p,
            xf64.as_ref(),
            &y,
            &prior_w[..n],
            false,
            &mut beta_packed,
            BetaStep::Profile {
                exact: None,
                xtwm,
                ainv_mtwx,
                schur,
                beta_rhs,
                beta_prev,
                schur_llt_mem,
            },
            pirls,
            packed,
            wx,
            None, // offset
            None,
            n,
            &mut crate::counters::EvalCounters::new(),
        )
    };
    assert!(oracle.3, "packed Profile solve must converge");

    // Structured: build_packed_m → pirls_solve_blocked_extras in Profile mode.
    let mut wss = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
    let mut beta_str = vec![0.0_f64; p];
    {
        let GlmmWorkspace {
            groupings,
            params: prm,
            z_buf,
            pirls: PirlsScratch { lam, .. },
            structured:
                StructuredScratch {
                    m_core_buf,
                    cross_theta,
                    ..
                },
            pattern: StructuredPattern {
                cross_col, n_cross, ..
            },
            ..
        } = &mut wss;
        prm[..nt].copy_from_slice(&theta);
        build_packed_m(
            groupings,
            &prm[..],
            z_buf,
            &extra_ids,
            lam,
            &ids,
            m_core_buf,
            cross_theta,
            cross_col,
            n_cross,
            n,
        );
    }
    let structured = {
        let GlmmWorkspace {
            groupings,
            prior_w,
            beta_rhs,
            wx,
            pirls,
            structured,
            pattern,
            border:
                BorderScratch {
                    xtwx: _,
                    xtwm,
                    ainv_mtwx,
                    schur,
                    schur_llt_mem,
                    beta_prev,
                },
            ..
        } = &mut wss;
        build_coupling_csr(
            &ids,
            &pattern.cross_col,
            &pattern.n_cross,
            groupings.n_primary,
            n,
            &mut pattern.coup_cols,
            &mut pattern.coup_ptr,
            &mut pattern.coup_pos,
        );
        pirls_solve_blocked_extras(
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            f64::NAN,
            groupings,
            &ids,
            xf64.as_ref(),
            &y,
            &prior_w[..n],
            false,
            &mut beta_str,
            BetaStep::Profile {
                exact: None,
                xtwm,
                ainv_mtwx,
                schur,
                beta_rhs,
                beta_prev,
                schur_llt_mem,
            },
            pirls,
            structured,
            pattern,
            None, // dual
            wx,
            None, // offset
            None,
            n,
            &mut crate::counters::EvalCounters::new(),
        )
    };
    assert!(structured.3, "structured Profile solve must converge");
    // Pinned (`assert_pinned_return` states the contract); the
    // β comparison below is a 1e-8 relative band and cannot see a small move.
    assert_pinned_return(
        structured,
        (124.6931434447742, 2.7692222052497986, 3.7225226795925, true),
        "Profile structured vs packed",
    );

    for j in 0..p {
        let rel = (beta_packed[j] - beta_str[j]).abs() / beta_packed[j].abs().max(1.0);
        assert!(
            rel < 1e-8,
            "Profile β[{j}] cross-variant: packed {} structured {} (rel {rel:.2e})",
            beta_packed[j],
            beta_str[j]
        );
    }
}

/// The structured layout keeps one crossed value per slot
/// (`StructuredScratch::cross_theta`), and at `f64` a θ-pinned grouping takes
/// no slot, so pinning the FIRST of two crossed groupings shifts the second
/// grouping's θ into slot 0. A value left at its grouping's old slot (or a
/// stale one from the previous evaluation) would pair the pinned grouping's
/// θ with the surviving grouping's columns. The packed-row layout builds its own
/// `M` with every column kept, so it is an independent oracle for the same
/// objective. The coupling CSR is also cached per (fit, θ-pinning mask), and a
/// pinned crossed θ drops that grouping from build_packed_m's cross_col/n_cross,
/// so a stale cache would silently corrupt the Schur build. On one workspace,
/// unpinned, first pinned, last pinned, then unpinned again: each structured
/// deviance must match the packed one to FP error and a fresh-workspace eval
/// bit for bit, and un-pinning must restore the unpinned value exactly.
#[test]
fn structured_first_pinned_crossed_theta_matches_packed() {
    let (model, ids, x, y, n, p) = grouseticks_3crossed_fixture();
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
    assert_eq!(ws.layout, GlmmLayout::Structured);
    assert!(
        ws.groupings.crossed.len() >= 2,
        "fixture must have two crossed groupings"
    );
    let nt = ws.n_theta;
    let mut unpinned = vec![0.0; nt + p];
    for (a, v) in unpinned[..nt].iter_mut().enumerate() {
        *v = 0.4 + 0.1 * a as f64;
    }
    let mut first_pinned = unpinned.clone();
    first_pinned[ws.groupings.crossed[0].vech_start] = 0.0;
    let mut last_pinned = unpinned.clone();
    last_pinned[ws.groupings.crossed.last().unwrap().vech_start] = 0.0;
    let packed = |params: &[f64]| {
        let mut wp = GlmmWorkspace::for_cluster_spec_packed(p, &model, n, &[], 1);
        fill_packed_cols(&mut wp, &ids.primary, &ids.extra, n);
        glmm_laplace_deviance(params, &mut wp, x.as_ref(), &y, &ids.primary, &ids.extra, n)
    };
    let fresh = |params: &[f64]| {
        let mut w2 = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
        glmm_laplace_deviance(params, &mut w2, x.as_ref(), &y, &ids.primary, &ids.extra, n)
    };
    let mut d_unpinned = f64::NAN;
    for (label, params) in [
        ("unpinned", &unpinned),
        ("first pinned", &first_pinned),
        ("last pinned", &last_pinned),
        ("unpinned again", &unpinned),
    ] {
        let d_str =
            glmm_laplace_deviance(params, &mut ws, x.as_ref(), &y, &ids.primary, &ids.extra, n);
        let d_pk = packed(params);
        assert!(
            d_str.is_finite() && d_pk.is_finite(),
            "{label}: {d_str} {d_pk}"
        );
        assert!(
            (d_str - d_pk).abs() <= 1e-9 * d_pk.abs().max(1.0),
            "{label}: structured {d_str} vs packed {d_pk}"
        );
        assert_eq!(
            d_str,
            fresh(params),
            "{label}: reused workspace read a stale CSR"
        );
        match label {
            "unpinned" => d_unpinned = d_str,
            "unpinned again" => assert_eq!(
                d_str, d_unpinned,
                "un-pinning must restore the original pattern"
            ),
            _ => {}
        }
    }
}

/// No-extras twin of `packed_path_overshoot_fixture`: primary-only grouping
/// (8 RE slopes + intercept ⇒ q_p = 9, `extra_offsets` empty) so
/// `laplace_deviance` routes to `pirls_solve_blocked`. Same Poisson-log large
/// counts (~5000) at the β = 0, u = 0 cold start ⇒ the first full Fisher step
/// overshoots η into the exp() blow-up regime (the grouseticks non-convergence
/// class), so the pre-step-halving blocked loop returns INFINITY.
fn blocked_path_overshoot_fixture() -> (GlmmWorkspace, Mat<f64>, Vec<f64>, Vec<u32>, usize) {
    let (n, n_prim, n_slopes) = (48usize, 8usize, 8usize);
    let mut st = 20240703u64;
    let ncol = 1 + n_slopes; // intercept (fixed) + slope predictor columns
    let mut x = Mat::<f64>::zeros(n, ncol);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        ids[i] = (i % n_prim) as u32;
        x[(i, 0)] = 1.0;
        for d in 0..n_slopes {
            x[(i, 1 + d)] = lcg(&mut st); // slope predictors in [-0.5, 0.5]
        }
        y[i] = (10.0 + 4990.0 * (lcg(&mut st) + 0.5)).round(); // counts up to ~5000
    }
    let slope_cols: Vec<usize> = (1..=n_slopes).collect(); // x cols 1..=8 as RE slopes
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: (1..=n_slopes as u32).collect(),
                extra_groupings: vec![],
            }),
        },
        n_primary: n_prim,
        extra_counts: vec![],
    };
    let groupings = crate::lmm::LmmGroupings::from_cluster_spec(&cluster, &slope_cols);
    let ws =
        GlmmWorkspace::from_groupings(groupings, cluster.spec.family, 1, n, 1, GlmmLayout::Blocked);
    (ws, x, y, ids, n)
}

#[test]
fn pirls_blocked_step_halving_recovers_from_overshoot() {
    // Shape/data chosen so the first full Fisher step at the spec.β=0 cold start
    // overshoots η into the exp() blow-up regime (the grouseticks failure
    // class, on the no-extras blocked path). Before step-halving this returned
    // INFINITY.
    let (mut ws, x, y, ids, n) = blocked_path_overshoot_fixture();
    assert!(
        ws.groupings.extra_offsets.is_empty(),
        "fixture must route to the blocked pirls_solve_blocked (no extras)"
    );
    let n_params = ws.n_theta + ws.p;
    let params: Vec<f64> = (0..n_params)
        .map(|i| if i < ws.n_theta { 1.0 } else { 0.0 })
        .collect();
    let dev = glmm_laplace_deviance(&params, &mut ws, x.as_ref(), &y, &ids, &[], n);
    assert!(
        dev.is_finite(),
        "step-halving must rescue the cold-start overshoot, got {dev}"
    );
    // Sanity ceiling against a finite-but-absurd recovery (the guarded bug
    // returns INFINITY). A run of this fixture converges to dev≈1837; 3700
    // is a generous 2× headroom above that.
    assert!(dev < 3700.0, "recovered deviance {dev} implausibly large");
}

/// The NB dispersion is one trailing coordinate of both outer searches — joint
/// `[θ_RE | β | ln θ_NB]`, stage 1 `[θ_RE | ln θ_NB]` — boxed on the GLM bracket's
/// `[ln NB_THETA_LO, ln NB_THETA_HI]`. Every other family has no such slot, so its
/// solver dimensions are what they were.
#[test]
fn nb_workspace_carries_the_theta_nb_coordinate() {
    let re = Some(ReStructure {
        terms: Vec::new(),
        slopes: vec![],
        extra_groupings: vec![],
    });
    let p = 3;
    let nb = GlmmWorkspace::for_cluster_spec(
        p,
        &SizedSpec {
            spec: ModelSpec {
                family: Family::NegativeBinomial {
                    link: NegBinomialLink::Log,
                },
                re: re.clone(),
            },
            n_primary: 4,
            extra_counts: vec![],
        },
        40,
        &[],
        1,
    );
    let m = nb.n_theta + p;
    assert_eq!(nb.params.len(), m + 1);
    assert_eq!(nb.lower.len(), m + 1);
    assert_eq!(nb.upper.len(), m + 1);
    assert_eq!(nb.lower[m], crate::fit::NB_THETA_LO.ln());
    assert_eq!(nb.upper[m], crate::fit::NB_THETA_HI.ln());
    assert_eq!(nb.params_stage1.len(), nb.n_theta + 1);
    assert_eq!(nb.lower_stage1.len(), nb.n_theta + 1);
    assert_eq!(nb.lower_stage1[nb.n_theta], crate::fit::NB_THETA_LO.ln());
    assert_eq!(nb.upper_stage1[nb.n_theta], crate::fit::NB_THETA_HI.ln());
    assert_eq!(&nb.lower_stage1[..nb.n_theta], &nb.lower[..nb.n_theta]);

    let pois = GlmmWorkspace::for_cluster_spec(
        p,
        &SizedSpec {
            spec: ModelSpec {
                family: Family::Poisson {
                    link: PoissonLink::Log,
                },
                re,
            },
            n_primary: 4,
            extra_counts: vec![],
        },
        40,
        &[],
        1,
    );
    let m = pois.n_theta + p;
    assert_eq!(pois.params.len(), m);
    assert_eq!(pois.lower.len(), m);
    assert_eq!(pois.params_stage1.len(), pois.n_theta);
    assert_eq!(&pois.lower_stage1[..], &pois.lower[..pois.n_theta]);
    assert_eq!(&pois.upper_stage1[..], &pois.upper[..pois.n_theta]);
}

/// A converged search whose fixed-effect covariance cannot be formed keeps its
/// estimates. An all-zero third column (`fit_glmm` runs no alias gate; the
/// loop tier's `fit_on` reaches it the same way) leaves the β information with
/// an exactly zero row, so the Rx Schur Cholesky fails, and under
/// `WaldSe::Hessian` the joint Hessian fails before it: a deterministic double
/// SE failure. The search on `Joint` (no β profile, which would need that
/// Schur) converges regardless, to the optimum of the same data without the
/// column. This is the note's only deterministic fixture: `fit_cold` drops an
/// all-zero column at its alias gate, and the public inputs that do reach the
/// note (Poisson counts near e³⁸, constant within each cluster, a
/// cluster-level covariate, `WaldSe::Rx`) reach it only through cancellation
/// in the Rx Schur, which comes out non-PD on some draws and PD on others.
#[test]
fn fit_glmm_se_failure_keeps_the_estimates() {
    let (x2, y, ids) = glmm_intercept_dataset();
    let n = x2.nrows();
    let mut x3 = Mat::<f64>::zeros(n, 3);
    for i in 0..n {
        x3[(i, 0)] = x2[(i, 0)];
        x3[(i, 1)] = x2[(i, 1)];
    }
    let spec = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: 8,
        extra_counts: vec![],
    };
    let fit_on = |x: faer::MatRef<f64>, p: usize, wald: WaldSe| {
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &spec, n, &[], 1);
        ws.outer_search = OuterSearch::Joint;
        let targets: Vec<u32> = (0..p as u32).collect();
        let fit = fit_glmm(
            &mut ws,
            x,
            x.subcols(0, 0),
            &y,
            &ids,
            &[],
            &targets,
            None,
            &vec![0.0; p],
            n,
            wald,
            ColumnScaling::IDENTITY,
        );
        (fit, ws)
    };
    let (reference, ref_ws) = fit_on(x2.as_ref(), 2, WaldSe::Rx);
    assert!(reference.converged && !reference.se_unavailable);
    for wald in [WaldSe::Rx, WaldSe::Hessian] {
        let (fit, ws) = fit_on(x3.as_ref(), 3, wald);
        assert!(fit.converged, "{wald:?}: the search converged");
        assert!(
            fit.se_unavailable,
            "{wald:?}: no covariance at the endpoint"
        );
        assert!(
            !fit.hessian_fallback,
            "{wald:?}: no fallback covariance either"
        );
        assert!(fit.joint_t_sq.is_nan());
        for j in 0..3 {
            assert!(
                ws.inference.var_diag[j].is_nan() && ws.inference.t_sq[j].is_nan(),
                "{wald:?}: var[{j}] {} t²[{j}] {}",
                ws.inference.var_diag[j],
                ws.inference.t_sq[j]
            );
        }
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    ws.inference.vcov[(a, b)].is_nan(),
                    "{wald:?}: vcov[{a},{b}]"
                );
            }
        }
        assert!(
            (fit.deviance - reference.deviance).abs() < 1e-8 * reference.deviance,
            "{wald:?}: deviance {} against {}",
            fit.deviance,
            reference.deviance
        );
        for j in 0..2 {
            let (a, b) = (ws.betas[j], ref_ws.betas[j]);
            assert!(
                (a - b).abs() < 1e-3 * b.abs(),
                "{wald:?}: β[{j}] {a} against {b}"
            );
        }
    }
}

/// The binomial random-intercept fit on `entangled_dyadic_designs` (20
/// clusters): `X = [1, a, a + δb]`, `[1, a, b]` the same model with
/// `β' = (β₀, β₁ + β₂, δβ₂)`.
fn entangled_logit_fixture() -> (Mat<f64>, Mat<f64>, Vec<f64>, Vec<u32>, f64, SizedSpec) {
    let n = 400;
    let nc = 20usize;
    let (xv, xrv, delta) = crate::fit::common_tests::entangled_dyadic_designs(n, 20);
    let mut st = 11u64;
    let u: Vec<f64> = (0..nc)
        .map(|_| crate::fit::common_tests::lcg(&mut st))
        .collect();
    let ids: Vec<u32> = (0..n as u32).map(|i| i % nc as u32).collect();
    let mut x = Mat::<f64>::zeros(n, 3);
    let mut xr = Mat::<f64>::zeros(n, 3);
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..3 {
            x[(i, j)] = xv[3 * i + j];
            xr[(i, j)] = xrv[3 * i + j];
        }
        let eta = 0.2 + 0.02 * xr[(i, 1)] + 0.02 * xr[(i, 2)] + u[ids[i] as usize];
        let pr = 1.0 / (1.0 + (-eta).exp());
        y[i] = if 0.5 * (crate::fit::common_tests::lcg(&mut st) + 1.0) < pr {
            1.0
        } else {
            0.0
        };
    }
    let spec = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: nc,
        extra_counts: vec![],
    };
    (x, xr, y, ids, delta, spec)
}

/// The joint Wald over all three coefficients, `β'·Cov⁻¹·β`, does not depend
/// on the parametrization, so on [`entangled_logit_fixture`] it is computed on
/// the well-conditioned design from `cov` there and `beta` mapped to it.
fn entangled_wald_reference(cov: &Mat<f64>, beta: [f64; 3], delta: f64) -> f64 {
    let b = [beta[0], beta[1] + beta[2], delta * beta[2]];
    let mut sol = Mat::<f64>::from_fn(3, 1, |i, _| b[i]);
    crate::linalg::SeqLlt::new(cov.as_ref())
        .expect("a positive definite covariance")
        .solve_in_place(sol.as_mut());
    (0..3).map(|i| b[i] * sol[(i, 0)]).sum()
}

/// The Rx Schur on a nearly collinear design keeps its digits. On
/// [`entangled_logit_fixture`] the fit's own Rx variance of the entangled
/// coefficient must equal the variance read off the well-conditioned design
/// `[1, a, b]` at the SAME fitted mode (`rx_cov_into` with only the
/// fixed-effect design swapped: `η`, `W` and the random-effect factors are
/// unchanged), mapped through the exact `1/δ²`; so must the joint Wald.
/// Comparing at one point keeps the optimizer out of it. A Schur formed from
/// the Gram of `X` misses by about `cond(X)²·ε`; `rx_schur_llt` rebuilds it on
/// `X·U⁻¹`, and `joint_wald_chi_sq` reads the factor's columns without
/// forming a Gram again.
#[test]
fn rx_schur_on_entangled_columns_keeps_its_digits() {
    let (x, xr, y, ids, delta, spec) = entangled_logit_fixture();
    let n = y.len();
    let mut ws = GlmmWorkspace::for_cluster_spec(3, &spec, n, &[], 1);
    let fit = fit_glmm(
        &mut ws,
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[0, 1, 2],
        None,
        &[0.0; 3],
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fit.converged && !fit.se_unavailable);
    assert!(
        ws.params[0] > 0.1,
        "an interior variance, θ = {}",
        ws.params[0]
    );
    let var = ws.inference.var_diag[2];
    let mut cov = Mat::<f64>::zeros(3, 3);
    assert!(rx_cov_into(
        &mut ws,
        xr.as_ref(),
        xr.as_ref().subcols(0, 0),
        &ids,
        3,
        n,
        &mut cov
    ));
    let reference = cov[(2, 2)] / (delta * delta);
    assert!(
        (var - reference).abs() <= 1e-9 * reference,
        "Var(β₂) {var} against {reference} (rel {:e})",
        (var - reference).abs() / reference
    );
    let beta = [ws.betas[0], ws.betas[1], ws.betas[2]];
    let w = entangled_wald_reference(&cov, beta, delta);
    assert!(
        (fit.joint_t_sq - w).abs() <= 1e-9 * w,
        "joint Wald {} against {w}",
        fit.joint_t_sq
    );
    // The Hessian arm's Rx fallback reads its joint Wald off the root
    // `rx_cov_into` leaves in `cov_root`; a Cholesky of the covariance itself
    // misses by about `cond(X)²·ε` (1e-5 here).
    let mut raw = Mat::<f64>::zeros(3, 3);
    assert!(rx_cov_into(
        &mut ws,
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &ids,
        3,
        n,
        &mut raw
    ));
    let mut qr = Mat::<f64>::zeros(3, 3);
    let mut z = [0.0; 3];
    let w_root = crate::lmm::joint_wald_chi_sq(
        ws.inference.cov_root.as_ref(),
        &beta,
        1.0,
        &[0, 1, 2],
        qr.as_mut(),
        &mut z,
    );
    assert!(
        (w_root - w).abs() <= 1e-9 * w,
        "Rx-fallback joint Wald {w_root} against {w}"
    );
}

/// The joint-Hessian standard errors on a nearly collinear design keep their
/// digits. On [`entangled_logit_fixture`] the fit's Hessian variances and
/// joint Wald must equal those of the well-conditioned design `[1, a, b]` at
/// the same point (`joint_hessian_cov` on that design with β mapped to it).
/// The Hessian's β block is a Gram in `X`, so on `X` itself it misses by
/// about `cond(X)²·ε` (5e-4 here); `joint_hessian_cov` takes it in the
/// search's own coordinates `β̃ = Lᵀβ` instead. Twice: on the blocked layout,
/// and on the packed-row one with a random slope on `a` (column 1 of both
/// designs), whose kernels must read that slope from `z_buf`, never from the
/// orthogonalized design the Hessian differentiates.
#[test]
fn hessian_se_on_entangled_columns_keeps_its_digits() {
    let (x, xr, y, ids, delta, spec) = entangled_logit_fixture();
    let n = y.len();
    for mut ws in [
        GlmmWorkspace::for_cluster_spec(3, &spec, n, &[], 1),
        GlmmWorkspace::for_cluster_spec_packed(3, &spec, n, &[1], 1),
    ] {
        fill_packed_cols(&mut ws, &ids, &[], n);
        let fit = fit_glmm(
            &mut ws,
            x.as_ref(),
            x.as_ref().subcols(0, 0),
            &y,
            &ids,
            &[],
            &[0, 1, 2],
            None,
            &[0.0; 3],
            n,
            WaldSe::Hessian,
            ColumnScaling::IDENTITY,
        );
        let layout = ws.layout;
        assert!(fit.converged && !fit.hessian_fallback, "{layout:?}");
        let nt = ws.n_theta;
        let beta = [ws.params[nt], ws.params[nt + 1], ws.params[nt + 2]];
        ws.params[nt + 1] = beta[1] + beta[2];
        ws.params[nt + 2] = delta * beta[2];
        let mut cov = Mat::<f64>::zeros(3, 3);
        let status = joint_hessian_cov(
            &mut ws,
            xr.as_ref(),
            xr.as_ref().subcols(0, 0),
            &y,
            &ids,
            &[],
            3,
            n,
            &mut cov,
        );
        assert_eq!(status, FdHessianStatus::Ok);
        let reference = [
            cov[(0, 0)],
            cov[(1, 1)] - 2.0 * cov[(1, 2)] / delta + cov[(2, 2)] / (delta * delta),
            cov[(2, 2)] / (delta * delta),
        ];
        for (j, &r) in reference.iter().enumerate() {
            let v = ws.inference.var_diag[j];
            assert!(
                (v - r).abs() <= 1e-9 * r,
                "{layout:?}: Var(β{j}) {v} against {r} (rel {:e})",
                (v - r).abs() / r
            );
        }
        let w = entangled_wald_reference(&cov, beta, delta);
        assert!(
            (fit.joint_t_sq - w).abs() <= 1e-9 * w,
            "{layout:?}: joint Wald {} against {w}",
            fit.joint_t_sq
        );
    }
}

/// Adversarial halving coverage. Runs a
/// deliberately hostile fit both ways — single-stage (`outer_search =
/// OuterSearch::Joint`, the trusted reference) and the shape's default route —
/// and asserts the
/// two-stage path lands on exactly one of two surfaces, never a third:
/// (a) it converges to genuinely finite estimates AND, when single-stage also
/// converged, agrees with it at ORACLE tolerances, or
/// (b) it fails cleanly — `converged == false` with the whole inference output
/// NaN (`nan_fit`), the terminal state an exhausted-halving PIRLS reaches.
/// Only a failed search lands on (b): a covariance that cannot be formed at a
/// converged endpoint NaNs the SEs and keeps the estimates, which is (a).
/// A finite-but-wrong two-stage fit (converged, disagreeing with single-stage) is
/// precisely the failure this gate exists to catch.
#[allow(clippy::too_many_arguments)]
fn assert_two_stage_adversarial(
    label: &str,
    spec: &SizedSpec,
    x: faer::MatRef<f64>,
    y: &[f64],
    primary: &[u32],
    extra: &[Vec<u32>],
    n: usize,
    p: usize,
    theta_start: Option<&[f64]>,
    offset: Option<&[f64]>,
    expect_converged: Option<bool>,
) {
    let targets = [if p >= 2 { 1u32 } else { 0u32 }];
    let beta_start = vec![0.0_f64; p];

    let mut ws1 = GlmmWorkspace::for_cluster_spec(p, spec, n, &[], 1);
    ws1.outer_search = OuterSearch::Joint; // pin the single-stage reference
    ws1.offset = offset.map(<[f64]>::to_vec);
    let fit1 = fit_glmm(
        &mut ws1,
        x,
        x.subcols(0, 0),
        y,
        primary,
        extra,
        &targets,
        theta_start,
        &beta_start,
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );

    // Second workspace keeps the constructor's default route — the A/B is
    // "default route vs `Joint`", not "PQL warm start vs `Joint`" specifically.
    let mut ws2 = GlmmWorkspace::for_cluster_spec(p, spec, n, &[], 1);
    ws2.offset = offset.map(<[f64]>::to_vec);
    let fit2 = fit_glmm(
        &mut ws2,
        x,
        x.subcols(0, 0),
        y,
        primary,
        extra,
        &targets,
        theta_start,
        &beta_start,
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );

    if let Some(expected) = expect_converged {
        assert_eq!(
            fit2.converged, expected,
            "{label}: two-stage convergence outcome changed — it is pinned so the \
             branch below that it takes stays exercised"
        );
    }
    if !fit2.converged {
        // Clean NaN failure surface (nan_fit): every inference output NaN — never
        // a finite value smuggled out under converged == false.
        assert!(
            ws2.betas[..p].iter().all(|b| b.is_nan()),
            "{label}: failed two-stage fit must NaN β̂, got {:?}",
            &ws2.betas[..p]
        );
        return;
    }
    // Two-stage converged: outputs must be genuinely finite (not NaN under a
    // converged flag), the sanity floor when single-stage supplies no reference.
    let tau2 = ws2.params[0].powi(2);
    assert!(
        tau2.is_finite(),
        "{label}: converged two-stage fit must have finite τ̂², got {tau2}"
    );
    assert!(
        ws2.betas[..p].iter().all(|b| b.is_finite()),
        "{label}: converged two-stage fit must have finite β̂, got {:?}",
        &ws2.betas[..p]
    );
    // Finite-but-wrong catch: when single-stage also converged, the two paths must
    // land on the same optimum at oracle tolerances.
    if fit1.converged {
        for j in 0..p {
            let (a, b) = (ws1.betas[j], ws2.betas[j]);
            let rel = (a - b).abs() / a.abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "{label}: β[{j}] single {a} vs two-stage {b} (rel {rel})"
            );
        }
        let nt = ws1.n_theta;
        for t in 0..nt {
            let (a, b) = (ws1.params[t], ws2.params[t]);
            assert!(
                (a - b).abs() < 1e-3 * (1.0 + a.abs()),
                "{label}: θ[{t}] single {a} vs two-stage {b}"
            );
        }
        let (ta, tb) = (ws1.params[0].powi(2), ws2.params[0].powi(2));
        let trel = (ta - tb).abs() / ta.abs().max(1e-6);
        assert!(
            trel < 1e-3,
            "{label}: τ² single {ta} vs two-stage {tb} (rel {trel})"
        );
    }
}

/// Adversarial: a wildly over-scaled θ start (θ₀ = 100, ~100× the fitted scale)
/// on the logit intercept fixture. The default route converges and lands on
/// the `Joint` optimum — never finite-but-wrong.
#[test]
fn two_stage_adversarial_bad_theta_start() {
    let (x, y, ids) = glmm_intercept_dataset();
    let spec = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: 8,
        extra_counts: vec![],
    };
    assert_two_stage_adversarial(
        "bad_theta",
        &spec,
        x.as_ref(),
        &y,
        &ids,
        &[],
        80,
        2,
        Some(&[100.0]),
        None,
        Some(true),
    );
}

/// Adversarial: a near-collinear design (col 2 = col 1 + 1e-8·noise, condition
/// number ~1e8) makes the β-Schur near-singular — the stress the β-profiling step's
/// dense solve must survive. `fit_glmm` runs no rank-deficiency salvage (that lives
/// in `fit_checked`), so this hits the ill-conditioned kernel directly. The
/// `ExactProfile` attempt fails and the `PqlThenJoint` fallback's search
/// converges to the `Joint` optimum (deviance equal to ~1e-11, β₁ ≈ −β₂ ≈
/// 1.45e7: the near-duplicate column's own coefficient). The β Schur
/// complement is singular to rounding there, so whether its Cholesky succeeds
/// is round-off between points a few 1e-6 apart in β; a failure NaNs the SEs
/// alone, so the fit converges on both routes either way.
#[test]
fn two_stage_adversarial_near_collinear_x() {
    let (n, nc) = (80usize, 8usize);
    let mut st = 91u64;
    let u0: Vec<f64> = (0..nc).map(|_| 0.5 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 3);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = i % nc;
        ids[i] = c as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x1 + 1e-8 * lcg(&mut st); // near-duplicate of col 1
        let eta = 0.2 + 0.7 * x1 + u0[c];
        let pr = 1.0 / (1.0 + (-eta).exp());
        y[i] = if lcg(&mut st) + 0.5 < pr { 1.0 } else { 0.0 };
    }
    let spec = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: nc,
        extra_counts: vec![],
    };
    assert_two_stage_adversarial(
        "near_collinear",
        &spec,
        x.as_ref(),
        &y,
        &ids,
        &[],
        n,
        3,
        None,
        None,
        Some(true),
    );
}

/// Adversarial: 30 clusters of 2 observations each — near-singleton groups leave
/// the random intercept barely identified and the θ landscape flat/ill-posed, the
/// regime where a mis-stepped optimizer most easily wanders finite-but-wrong.
/// The default route converges and lands on the `Joint` optimum.
#[test]
fn two_stage_adversarial_tiny_clusters() {
    let (nc, per) = (30usize, 2usize);
    let n = nc * per;
    let mut st = 55u64;
    let u0: Vec<f64> = (0..nc).map(|_| 0.8 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = i / per; // contiguous tiny blocks
        ids[i] = c as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        let eta = 0.1 + 0.5 * x1 + u0[c];
        let pr = 1.0 / (1.0 + (-eta).exp());
        y[i] = if lcg(&mut st) + 0.5 < pr { 1.0 } else { 0.0 };
    }
    let spec = SizedSpec {
        spec: logit_intercept_spec(),
        n_primary: nc,
        extra_counts: vec![],
    };
    assert_two_stage_adversarial(
        "tiny_clusters",
        &spec,
        x.as_ref(),
        &y,
        &ids,
        &[],
        n,
        2,
        None,
        None,
        Some(true),
    );
}

/// Adversarial: a Poisson random intercept with a one-row offset of 800, past
/// the log link's η bound, so every PIRLS trial on that row is infeasible and
/// each solve fails through the halving cap. The search fails on every route —
/// the default `ExactProfile`, its `PqlThenJoint` fallback and the `Joint`
/// reference alike — so this is the fixture that pins the helper's clean
/// failure surface (`nan_fit`), independent of round-off.
#[test]
fn two_stage_adversarial_overflow_offset() {
    let n = 30;
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = ((i * 7) % 11) as f64 / 10.0 - 0.5;
        ids[i] = (i / 5) as u32;
        y[i] = ((i * 3) % 5) as f64;
    }
    let mut offset = vec![0.0f64; n];
    offset[10] = 800.0;
    let mut spec = SizedSpec {
        spec: intercept_only_spec(),
        n_primary: 6,
        extra_counts: vec![],
    };
    spec.spec.family = Family::Poisson {
        link: PoissonLink::Log,
    };
    assert_two_stage_adversarial(
        "overflow_offset",
        &spec,
        x.as_ref(),
        &y,
        &ids,
        &[],
        n,
        2,
        None,
        Some(&offset),
        Some(false),
    );
}

#[test]
fn cluster_row_index_matches_naive_filter_and_preserves_row_order() {
    // The second input leaves cluster 1 with no rows (ptr[1] == ptr[2]): its
    // slice must be empty, not a panic.
    let inputs: [&[u32]; 2] = [&[2, 0, 1, 0, 2, 1, 1, 0, 2, 0], &[0, 2, 0, 2]];
    let s = 3;
    for cluster_ids in inputs {
        let idx = ClusterRowIndex::build(cluster_ids, s);
        for c in 0..s {
            let naive: Vec<u32> = cluster_ids
                .iter()
                .enumerate()
                .filter(|&(_, &cc)| cc as usize == c)
                .map(|(i, _)| i as u32)
                .collect();
            assert_eq!(
                idx.cluster_rows(c),
                naive.as_slice(),
                "{cluster_ids:?} cluster {c}"
            );
            assert!(
                idx.cluster_rows(c).windows(2).all(|w| w[0] < w[1]),
                "{cluster_ids:?} cluster {c} rows not ascending: {:?}",
                idx.cluster_rows(c)
            );
        }
        let mut all: Vec<u32> = (0..s).flat_map(|c| idx.cluster_rows(c).to_vec()).collect();
        all.sort_unstable();
        assert_eq!(all, (0..cluster_ids.len() as u32).collect::<Vec<_>>());
    }
}

// --- `laplace_gradient`'s FD gate ---

/// Family × RE-shape fixture for the gradient FD gate: reuses the blocked-path
/// round-robin datasets' X design (`glmm_intercept_dataset` for `"int1"`,
/// n_theta=1; `glmm_slope_noextra_dataset` for `"q2s"`, n_theta=3), but
/// regenerates y in the family's own domain — Bernoulli for Binomial (every
/// link shares the same 0/1 domain), positive counts for Poisson/
/// NegativeBinomial, positive reals for Gamma — from a shape-keyed LCG stream,
/// so every cell is reproducible and none of the y values sit outside the
/// family's support regardless of which θ draw the gate later perturbs around.
/// NB's dispersion (`ws.nb_theta`) is held at a fixed constant: the gradient
/// this gate checks is with respect to `ws.params = [θ | β]` only.
fn fixture(
    family: Family,
    shape: &str,
) -> (GlmmWorkspace, Mat<f64>, Vec<f64>, Vec<u32>, usize, usize) {
    fixture_with_nagq(family, shape, 1)
}

/// `(n_rows, n_clusters)` each gradient-gate shape's fixed-size dataset used
/// before the cost table needed a size axis — the only two call sites that
/// care are `fixture_with_nagq`'s default and `fixture_padded`'s default, so
/// this stays a plain match rather than a stored constant.
fn default_fixture_size(shape: &str) -> (usize, usize) {
    match shape {
        "int1" => (80, 8),
        "q2s" | "q1n" => (96, 8),
        other => panic!("unknown gradient-gate shape {other}"),
    }
}

/// `fixture`, generalized over `nagq` — the AGQ gate tests need
/// `ws.pirls.agq_scratch` (and `ws.dual_scratch`'s eventual AGQ node table) sized
/// for `nagq > 1` from construction, which only `for_cluster_spec`'s own
/// `nagq` argument controls (`workspace.rs:415-427`); setting `ws.nagq` after
/// the fact would leave `agq_scratch` sized for the wrong `k`. `fixture`
/// itself is the `nagq = 1` case, unchanged.
fn fixture_with_nagq(
    family: Family,
    shape: &str,
    nagq: u8,
) -> (GlmmWorkspace, Mat<f64>, Vec<f64>, Vec<u32>, usize, usize) {
    let (n, nc) = default_fixture_size(shape);
    fixture_with_nagq_sized(family, shape, nagq, n, nc)
}

/// `fixture_with_nagq`, generalized over dataset size — the dual-call cost
/// table needs the same shapes at 100/3,000/30,000 rows, not only the fixed
/// ~80/96-row fixture. `n`/`nc` replace the shape match arm's hardcoded
/// row/cluster counts; everything past dataset generation (y-generation,
/// nagq wiring) is `fixture_with_nagq`'s own body, unchanged.
fn fixture_with_nagq_sized(
    family: Family,
    shape: &str,
    nagq: u8,
    n: usize,
    nc: usize,
) -> (GlmmWorkspace, Mat<f64>, Vec<f64>, Vec<u32>, usize, usize) {
    let (xf64, ids, n_clusters, slope_cols, seed): (Mat<f64>, Vec<u32>, u32, Vec<usize>, u64) =
        match shape {
            "int1" => {
                let (x, _y, ids) = glmm_intercept_dataset_layout(false, n, nc);
                (x, ids, nc as u32, vec![], 4001)
            }
            "q2s" | "q1n" => {
                let (x, _y, ids) = glmm_slope_noextra_dataset_layout(false, n, nc);
                (x, ids, nc as u32, vec![1], 4002)
            }
            other => panic!("unknown gradient-gate shape {other}"),
        };
    let n = xf64.nrows();
    let p = 2usize;
    let mut y = vec![0.0f64; n];
    let mut st = seed;
    for i in 0..n {
        let x1 = xf64[(i, 1)];
        let eta = 0.2 + 0.8 * x1;
        let noise = 0.4 + 1.6 * (lcg(&mut st) + 0.5); // in [0.4, 2.0]
        y[i] = match family {
            Family::Binomial { .. } => {
                let mu = 1.0 / (1.0 + (-eta).exp());
                if lcg(&mut st) + 0.5 < mu {
                    1.0
                } else {
                    0.0
                }
            }
            Family::Poisson { .. } | Family::NegativeBinomial { .. } => {
                let mu = eta.exp().min(20.0);
                (mu * noise).round().max(0.0)
            }
            Family::Gamma { .. } | Family::InverseGaussian { .. } => eta.exp() * noise,
            other => panic!("gradient-gate fixture: family {other:?} not wired"),
        };
    }
    // `"q1n"` is `(0 + x | g)` on the `"q2s"` design: column 0 of the block
    // and the entry below it are structural.
    let terms = match shape {
        "q1n" => vec![crate::ReTerm::no_intercept().slopes([1])],
        _ => Vec::new(),
    };
    let model = SizedSpec {
        spec: ModelSpec {
            family,
            re: Some(ReStructure {
                terms,
                slopes: slope_cols.iter().map(|&c| c as u32).collect(),
                extra_groupings: vec![],
            }),
        },
        n_primary: n_clusters as usize,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &slope_cols, nagq);
    if matches!(family, Family::NegativeBinomial { .. }) {
        ws.nb_theta = 4.0;
    }
    (ws, xf64, y, ids, p, n)
}

/// `fixture`, widened by `extra_p` synthetic zero-truth predictor columns
/// (small noise, no signal) so `m` crosses into the `Dual<12>` band — one
/// gate cell needs to exercise `N = 12` numerically, not only via
/// the speed campaign's padded timing run.
fn fixture_padded(
    family: Family,
    shape: &str,
    extra_p: usize,
) -> (GlmmWorkspace, Mat<f64>, Vec<f64>, Vec<u32>, usize, usize) {
    let (n, nc) = default_fixture_size(shape);
    fixture_padded_sized(family, shape, extra_p, n, nc)
}

/// `fixture_padded`, generalized over dataset size — same column-padding
/// trick, but the base fixture comes from `fixture_with_nagq_sized` so `m`
/// (lane count) and dataset size (n_rows/n_clusters) vary independently: the
/// cost table's size axis needs the same `extra_p` lane-count cells at
/// 100/3,000/30,000 rows.
fn fixture_padded_sized(
    family: Family,
    shape: &str,
    extra_p: usize,
    n: usize,
    nc: usize,
) -> (GlmmWorkspace, Mat<f64>, Vec<f64>, Vec<u32>, usize, usize) {
    let (_ws0, x0, y, ids, p0, n) = fixture_with_nagq_sized(family, shape, 1, n, nc);
    let p = p0 + extra_p;
    let mut x = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        x[(i, 0)] = x0[(i, 0)];
        x[(i, 1)] = x0[(i, 1)];
    }
    let mut st = 7777u64;
    for i in 0..n {
        for c in p0..p {
            x[(i, c)] = 0.3 * lcg(&mut st);
        }
    }
    let slope_cols: Vec<usize> = match shape {
        "q2s" => vec![1],
        "int1" => vec![],
        other => panic!("unknown gradient-gate shape {other}"),
    };
    let model = SizedSpec {
        spec: ModelSpec {
            family,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: slope_cols.iter().map(|&c| c as u32).collect(),
                extra_groupings: vec![],
            }),
        },
        n_primary: nc,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &slope_cols, 1);
    if matches!(family, Family::NegativeBinomial { .. }) {
        ws.nb_theta = 4.0;
    }
    (ws, x, y, ids, p, n)
}

/// Ten fixed-seed θ draws per RE shape for the gradient FD gate, β pinned at
/// the design's own true coefficients — the `0.2 + 0.8·x1` linear predictor
/// every `fixture` y is generated from. Diagonal θ lanes (vech(Λ) diagonal)
/// are drawn positive; off-diagonal lanes are unconstrained — Λ need not be
/// PD itself, only D = ΛΛ' is, and that holds for any off-diagonal value.
struct FixedSeedTheta {
    state: u64,
    n_theta: usize,
    diag: &'static [usize],
    beta: Vec<f64>,
}

impl FixedSeedTheta {
    fn next_params(&mut self) -> Vec<f64> {
        let mut out = vec![0.0f64; self.n_theta + self.beta.len()];
        #[allow(clippy::needless_range_loop)]
        for j in 0..self.n_theta {
            let r = lcg(&mut self.state);
            out[j] = if self.diag.contains(&j) {
                0.3 + 0.4 * (r + 0.5) // in [0.3, 0.7]
            } else {
                0.5 * r // in [-0.25, 0.25]
            };
        }
        out[self.n_theta..].copy_from_slice(&self.beta);
        out
    }
}

fn fixed_seed_theta(shape: &str) -> FixedSeedTheta {
    match shape {
        "int1" => FixedSeedTheta {
            state: 5001,
            n_theta: 1,
            diag: &[0],
            beta: vec![0.2, 0.8],
        },
        "q2s" => FixedSeedTheta {
            state: 5002,
            n_theta: 3,
            diag: &[0, 2],
            beta: vec![0.2, 0.8],
        },
        // The gates zero its structural entries (0 and 1) after each draw.
        "q1n" => FixedSeedTheta {
            state: 5006,
            n_theta: 3,
            diag: &[0, 2],
            beta: vec![0.2, 0.8],
        },
        // Structured-extras shapes: every factor is intercept-only (q = 1), so
        // every θ lane is a diagonal vech entry and is drawn positive. θ order
        // is the declaration order `glmm_extras_q1_dataset` builds — primary,
        // then nested, then crossed.
        "nested2" => FixedSeedTheta {
            state: 5003,
            n_theta: 2,
            diag: &[0, 1],
            beta: vec![0.2, 0.8],
        },
        "crossed6" => FixedSeedTheta {
            state: 5004,
            n_theta: 2,
            diag: &[0, 1],
            beta: vec![0.2, 0.8],
        },
        "nested2_crossed6" => FixedSeedTheta {
            state: 5005,
            n_theta: 3,
            diag: &[0, 1, 2],
            beta: vec![0.2, 0.8],
        },
        other => panic!("unknown gradient-gate shape {other}"),
    }
}

/// `fixed_seed_theta`, β widened with `extra_p` small fixed synthetic
/// coefficients for `fixture_padded`'s extra zero-truth columns.
fn fixed_seed_theta_padded(shape: &str, extra_p: usize) -> FixedSeedTheta {
    let mut rng = fixed_seed_theta(shape);
    let mut st = 8888u64;
    for _ in 0..extra_p {
        rng.beta.push(0.1 * lcg(&mut st));
    }
    rng
}

/// `{Binomial-logit, Binomial-probit, Binomial-cloglog, Poisson-log,
/// Gamma-log, NegativeBinomial}` × `{int1, q2s}` — the family/shape product
/// the gradient FD gate runs. `int1` (n_θ=1) and `q2s` (n_θ=3) are the only
/// GLMM-reachable blocked shapes among the speed campaign catalogue's four:
/// `nest2` is nested (extras non-empty, routes to `pirls_solve_blocked_extras`,
/// whose cells are `STRUCTURED_GATE_CELLS`) and `q3s` carries `glmm = FALSE` in the same catalogue (never fit as a GLMM, no
/// GLMM fixture exists). Binomial-cloglog stands in their place: a
/// GLMM-validated link, non-canonical (exercises the refinement
/// loop), and the only caller of `Scalar::exp_m1`.
const GRADIENT_GATE_CELLS: &[(Family, &str)] = &[
    (
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        "int1",
    ),
    (
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        "q2s",
    ),
    (
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        "int1",
    ),
    (
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        "q2s",
    ),
    (
        Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        "int1",
    ),
    (
        Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        "q2s",
    ),
    (
        Family::Poisson {
            link: PoissonLink::Log,
        },
        "int1",
    ),
    (
        Family::Poisson {
            link: PoissonLink::Log,
        },
        "q2s",
    ),
    (
        Family::Gamma {
            link: GammaLink::Log,
        },
        "int1",
    ),
    (
        Family::Gamma {
            link: GammaLink::Log,
        },
        "q2s",
    ),
    (
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        "int1",
    ),
    (
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        "q2s",
    ),
];

/// The masked shape the two dual FD gates run beside [`GRADIENT_GATE_CELLS`]:
/// `"q1n"`, `(0 + x | g)`, a block whose column 0 is structural, on every
/// family of that list.
const MASKED_GATE_CELLS: &[(Family, &str)] = &[
    (
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        "q1n",
    ),
    (
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        "q1n",
    ),
    (
        Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        "q1n",
    ),
    (
        Family::Poisson {
            link: PoissonLink::Log,
        },
        "q1n",
    ),
    (
        Family::Gamma {
            link: GammaLink::Log,
        },
        "q1n",
    ),
    (
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        "q1n",
    ),
];

/// Shared per-draw body of the gradient FD gate: `n_draws` fixed-seed θ (+
/// fixed β) draws, each checked coordinate-by-coordinate against a central FD
/// stencil of the SAME blocked Laplace deviance `laplace_gradient`
/// differentiates. Step: `h = 1e-5` absolute on θ, `1e-5·max(1, |β_k|)` on β
/// (`FD_STEP_BASE`'s asymmetry is calibrated against the FD-Hessian noise
/// floor, not a first difference — this uses its own steps). Band: 1e-6
/// relative. Both run at the CALLER's `ws.fd.pirls_tol_override` (the gate sets
/// `Some(1e-12)` — see `dual_gradient_matches_central_fd_per_family_and_shape`'s
/// doc comment for why the tolerance is tightened this far). On a masked
/// workspace each draw has its structural θ entries zeroed, and only the free
/// coordinates are differenced. Returns the number of coordinate checks.
#[allow(clippy::too_many_arguments)]
fn assert_dual_gradient_matches_fd(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    ids: &[u32],
    // Per-row extra-grouping level ids: empty on the blocked cells, the
    // fixture's own `[nested?, crossed?]` on the structured-extras cells.
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    family: Family,
    shape: &str,
    n_draws: usize,
    mut rng: FixedSeedTheta,
) -> usize {
    let m = ws.n_theta + p;
    let kk = ws.k.max(1);
    let mut ctrs = EvalCounters::new();
    let structural = ws.groupings.structural_theta().to_vec();
    let mut checks = 0;
    for _ in 0..n_draws {
        ws.params[..m].copy_from_slice(&rng.next_params());
        ws.groupings.zero_structural(&mut ws.params[..m]);
        let saved: Vec<f64> = ws.params[..m].to_vec();
        let mut grad = vec![0.0; m];
        let st = laplace_gradient(ws, x, y, ids, extra_ids, p, n, &mut grad);
        assert!(
            matches!(st, DerivStatus::Ok(_)),
            "{family:?}/{shape} did not converge"
        );
        for k in (0..m).filter(|k| structural.binary_search(k).is_err()) {
            let h = if k < ws.n_theta {
                1e-5
            } else {
                1e-5 * saved[k].abs().max(1.0)
            };
            // Cold-start `ws.pirls.u` (mirrors `laplace_deviance_at`'s own default
            // seed) before EACH directional eval, rather than letting it carry
            // over from the previous eval's converged mode: `laplace_deviance_ws`
            // seeds nothing itself, so an un-reset `u` chains a growing
            // warm-start history across the whole draw × coordinate loop, and
            // PIRLS's OWN convergence band (`tol`, however tight) is a band on
            // the mixed deviance, not on `u` — two warm starts can converge to
            // the same `mixed` to 1e-12 while `u` itself, and so the deviance
            // AT A DIFFERENT NEARBY θ, differ by more than that. Measured: on
            // this cell (Binomial-cloglog), an un-reset `u` puts the FD/dual
            // gap as high as ~1.5e-6 relative (over the 1e-6 band); resetting
            // to zero here brings every draw back to ~1e-9.
            for v in ws.pirls.u[..kk].iter_mut() {
                *v = 0.0;
            }
            ws.params[k] = saved[k] + h;
            ws.beta_rhs[..p].copy_from_slice(&ws.params[ws.n_theta..m]);
            let fp = laplace_deviance_ws(ws, x, y, ids, extra_ids, n, BetaMode::Fixed, &mut ctrs);
            for v in ws.pirls.u[..kk].iter_mut() {
                *v = 0.0;
            }
            ws.params[k] = saved[k] - h;
            ws.beta_rhs[..p].copy_from_slice(&ws.params[ws.n_theta..m]);
            let fm = laplace_deviance_ws(ws, x, y, ids, extra_ids, n, BetaMode::Fixed, &mut ctrs);
            ws.params[k] = saved[k];
            let fd = (fp - fm) / (2.0 * h);
            assert!(
                (grad[k] - fd).abs() <= 1e-6 * fd.abs().max(1.0),
                "{family:?}/{shape} coord {k}: dual {} vs fd {fd}",
                grad[k]
            );
            checks += 1;
        }
    }
    checks
}

/// Central FD of the Laplace deviance against the dual gradient, per family and
/// per RE shape, at a TIGHTENED PIRLS tolerance.
///
/// Why the tolerance is tightened: at the production 1e-9 band the PIRLS mode
/// carries an O(1e-5) positional error, and a gradient inherits it at first
/// order — an FD/analytic mismatch there would be the mode's error, not the
/// chain rule's. Forcing 1e-12 tests the math. Same reasoning the parked
/// gradient spec's validation section gives; the switch already exists as
/// `ws.fd.pirls_tol_override`.
///
/// The masked cells ([`MASKED_GATE_CELLS`]) run beside the list, and the gate
/// prints the check count of both.
#[test]
fn dual_gradient_matches_central_fd_per_family_and_shape() {
    let mut checks = [0usize; 2];
    for (set, cells) in [GRADIENT_GATE_CELLS, MASKED_GATE_CELLS]
        .into_iter()
        .enumerate()
    {
        for &(family, shape) in cells {
            let (mut ws, x, y, ids, p, n) = fixture(family, shape);
            ws.fd.pirls_tol_override = Some(1e-12);
            let rng = fixed_seed_theta(shape);
            checks[set] += assert_dual_gradient_matches_fd(
                &mut ws,
                x.as_ref(),
                &y,
                &ids,
                &[],
                p,
                n,
                family,
                shape,
                10,
                rng,
            );
            ws.fd.pirls_tol_override = None;
        }
    }
    println!(
        "dual-gradient FD gate: {} checks on {} cells, {} on {} masked cells",
        checks[0],
        GRADIENT_GATE_CELLS.len(),
        checks[1],
        MASKED_GATE_CELLS.len()
    );
    // Ten draws × (one free θ + two β) per masked cell.
    assert_eq!(checks[1], MASKED_GATE_CELLS.len() * 10 * 3);
}

/// The assembled `f64` gradient against `laplace_gradient`, the exact in-tree
/// oracle, per coordinate on every blocked and every structured-extras gate
/// cell. Both differentiate the same Laplace objective at the same `ws.params`
/// and the same tightened PIRLS tolerance, one through dual lanes and one
/// through the explicit `F`/`G` adjoint, so they must agree to round-off — the
/// band is `1e-10` relative, not the FD gates' `1e-6`.
///
/// A gap at the `1e-7` scale rather than at round-off would be the assembly
/// reading some η-dependent quantity at an iterate the kernel did not leave it
/// at; the fix for that is the assembly's evaluation point, never a wider band
/// here.
#[test]
fn assembled_gradient_matches_laplace_gradient() {
    const BAND: f64 = 1e-10;
    let mut cells: Vec<(String, f64)> = Vec::new();
    for &(family, shape) in GRADIENT_GATE_CELLS {
        let (mut ws, x, y, ids, p, n) = fixture(family, shape);
        ws.fd.pirls_tol_override = Some(1e-12);
        let rng = fixed_seed_theta(shape);
        let worst = assert_assembled_gradient_matches_laplace(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &[],
            p,
            n,
            family,
            shape,
            10,
            rng,
            BAND,
        );
        cells.push((format!("{family:?}/{shape}"), worst));
        ws.fd.pirls_tol_override = None;
    }
    for &(family, shape, np, n_crossed) in STRUCTURED_GATE_CELLS {
        let (mut ws, x, y, ids, extra_ids, p, n) = extras_fixture(family, np, n_crossed);
        ws.fd.pirls_tol_override = Some(1e-12);
        let rng = fixed_seed_theta(shape);
        let worst = assert_assembled_gradient_matches_laplace(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            family,
            shape,
            10,
            rng,
            BAND,
        );
        cells.push((format!("{family:?}/{shape}"), worst));
        ws.fd.pirls_tol_override = None;
    }
    for (name, worst) in &cells {
        println!("assembled-gradient cell {name}: worst relative gap {worst:e}");
    }
}

/// The same comparison as [`assembled_gradient_matches_laplace_gradient`], run
/// at the SHIPPED PIRLS tolerance instead of a tightened one.
///
/// What this gate separates, and the tightened twin cannot: whether the
/// objective's three terms and the factor sit at one iterate. At
/// `pirls_tol_override = Some(1e-12)` the last PIRLS step is at round-off, so
/// `u` and the point before it coincide and a wrong evaluation point is
/// invisible — that twin stays green either way. At the shipped tolerance the
/// step is not at round-off on a non-canonical link, and the refreshed
/// objective agrees with `laplace_gradient` two to three orders more closely
/// than an objective whose `log|A|` and factor lag the returned mode by a
/// Newton step (probit, cloglog, Gamma-log and NB-log cells; canonical logit
/// and Poisson-log cells sit at 1e-14 either way).
///
/// Band provenance, measured 2026-09-15 over all 30 gate cells: ten times the
/// worst cell. The five worst are Gamma-Log/int1 7.24e-9,
/// Binomial-Cloglog/nested2_crossed6 5.43e-9, Gamma-Log/crossed6 3.28e-9,
/// Gamma-Log/nested2_crossed6 2.37e-9 and Binomial-Cloglog/nested2 2.11e-9;
/// the rest of the non-canonical set runs 3.25e-13…8.49e-10, and every
/// canonical cell sits at 1.25e-15…2.84e-14.
#[test]
fn assembled_gradient_matches_laplace_gradient_at_shipped_tolerance() {
    const BAND: f64 = 7.3e-8;
    let mut cells: Vec<(String, f64)> = Vec::new();
    for &(family, shape) in GRADIENT_GATE_CELLS {
        let (mut ws, x, y, ids, p, n) = fixture(family, shape);
        let rng = fixed_seed_theta(shape);
        let worst = assert_assembled_gradient_matches_laplace(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &[],
            p,
            n,
            family,
            shape,
            10,
            rng,
            BAND,
        );
        cells.push((format!("{family:?}/{shape}"), worst));
    }
    for &(family, shape, np, n_crossed) in STRUCTURED_GATE_CELLS {
        let (mut ws, x, y, ids, extra_ids, p, n) = extras_fixture(family, np, n_crossed);
        let rng = fixed_seed_theta(shape);
        let worst = assert_assembled_gradient_matches_laplace(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            family,
            shape,
            10,
            rng,
            BAND,
        );
        cells.push((format!("{family:?}/{shape}"), worst));
    }
    for (name, worst) in &cells {
        println!("shipped-tolerance assembled-gradient cell {name}: worst relative gap {worst:e}");
    }
}

/// Shared per-draw body of the assembled-gradient gate: `n_draws` fixed-seed θ
/// (+ fixed β) draws, each compared coordinate by coordinate against
/// `laplace_gradient` on the same parameters. Returns the worst relative gap
/// seen over the whole cell.
#[allow(clippy::too_many_arguments)]
fn assert_assembled_gradient_matches_laplace(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    family: Family,
    shape: &str,
    n_draws: usize,
    mut rng: FixedSeedTheta,
    band: f64,
) -> f64 {
    let m = ws.n_theta + p;
    let mut worst = 0.0f64;
    for _ in 0..n_draws {
        ws.params[..m].copy_from_slice(&rng.next_params());
        let mut want = vec![0.0; m];
        let st = laplace_gradient(ws, x, y, ids, extra_ids, p, n, &mut want);
        assert!(
            matches!(st, DerivStatus::Ok(_)),
            "{family:?}/{shape}: laplace_gradient did not converge"
        );
        let mut got = vec![0.0; m];
        super::assembled::gradient_f64(ws, x, y, ids, extra_ids, p, n, &mut got)
            .unwrap_or_else(|| panic!("{family:?}/{shape}: the assembled gradient declined"));
        for c in 0..m {
            let gap = (got[c] - want[c]).abs() / want[c].abs().max(1.0);
            worst = worst.max(gap);
            assert!(
                gap <= band,
                "{family:?}/{shape} coord {c}: assembled {} vs laplace_gradient {} (relative gap {gap:e})",
                got[c],
                want[c]
            );
        }
    }
    worst
}

/// `glmm_extras_q1_dataset(0, 6)` with a SECOND crossed grouping (4 levels)
/// bolted onto the spec and the id list, so the design carries two crossed
/// factors and pinning the first at θ = 0 leaves a surviving one. The
/// response is the base dataset's own — a derivative gate compares two
/// differentiations of one objective and does not need `y` to reflect the
/// extra factor.
#[allow(clippy::type_complexity)]
fn two_crossed_fixture(
    family: Family,
) -> (
    GlmmWorkspace,
    Mat<f64>,
    Vec<f64>,
    Vec<u32>,
    Vec<Vec<u32>>,
    usize,
    usize,
) {
    const N_C2: usize = 4;
    let (x, y, ids, mut extra_ids, mut spec) = glmm_extras_q1_dataset(0, 6);
    let (n, p) = (y.len(), 2usize);
    spec.spec.family = family;
    spec.spec
        .re
        .as_mut()
        .expect("the extras dataset always declares an RE structure")
        .extra_groupings
        .push(Grouping {
            terms: Vec::new(),
            relation: GroupingRelation::Crossed,
            slopes: vec![],
        });
    spec.extra_counts.push(N_C2);
    extra_ids.push((0..n).map(|i| ((i / 3) % N_C2) as u32).collect());
    let ws = GlmmWorkspace::for_cluster_spec(p, &spec, n, &[], 1);
    assert_eq!(ws.groupings.crossed.len(), 2, "two crossed groupings");
    (ws, x, y, ids, extra_ids, p, n)
}

/// Every column of the assembled joint Hessian against a central difference of
/// the assembled `f64` gradient in that column's coordinate, on every blocked
/// and every structured-extras gate cell and every coordinate of each.
///
/// This is the supply check for the second derivatives, not a second gradient
/// check: at `T = Dual<N>` the assembly's own first-order quantities have to
/// carry the third derivatives of the likelihood in their lanes —
/// `observed_weight`'s lane is `½·dev'''`, `weight_eta_deriv`'s is `w''`, and
/// `gamma_phi_prime`'s is `Φ''` — with no third-derivative table written
/// anywhere. A lane-plumbing mistake (a quantity lifted with `from_f64` that
/// should have been differentiated, an `f64` evaluation point where a `T` one
/// belongs) shows here as a whole-coordinate miss.
///
/// The last cell is padded past the lane ladder's top rung so the pass runs in
/// three chunks: it is the only gate that exercises the chunk driver's
/// `base`/`width` bookkeeping, since the per-entry comparison against the
/// hyper-dual Hessian cannot go above that pass's own cap.
///
/// Steps are the ones the gradient gates use — `1e-5` absolute on θ,
/// `1e-5·max(1,|β_k|)` on β — and the band is the `1e-6` relative band the
/// existing FD-of-gradient gate uses, at `pirls_tol_override = Some(1e-12)`.
#[test]
fn assembled_hessian_columns_match_fd_of_f64_gradient() {
    const BAND: f64 = 1e-6;
    const DRAWS: usize = 3;
    let mut cells: Vec<(String, f64, f64)> = Vec::new();
    for &(family, shape) in GRADIENT_GATE_CELLS {
        let (mut ws, x, y, ids, p, n) = fixture(family, shape);
        ws.fd.pirls_tol_override = Some(1e-12);
        let rng = fixed_seed_theta(shape);
        let (wt, wb) = assert_assembled_hessian_columns_match_fd(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &[],
            p,
            n,
            family,
            shape,
            DRAWS,
            rng,
            &[],
            BAND,
        );
        cells.push((format!("{family:?}/{shape}"), wt, wb));
        ws.fd.pirls_tol_override = None;
    }
    for &(family, shape, np, n_crossed) in STRUCTURED_GATE_CELLS {
        let (mut ws, x, y, ids, extra_ids, p, n) = extras_fixture(family, np, n_crossed);
        ws.fd.pirls_tol_override = Some(1e-12);
        let rng = fixed_seed_theta(shape);
        let (wt, wb) = assert_assembled_hessian_columns_match_fd(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            family,
            shape,
            DRAWS,
            rng,
            &[],
            BAND,
        );
        cells.push((format!("{family:?}/{shape}"), wt, wb));
        ws.fd.pirls_tol_override = None;
    }
    // One cell with the FIRST of two crossed θ pinned at 0. The `f64` packer
    // drops that grouping's column and the dual packer keeps it, so the two
    // arms disagree on both the packed width and the column order; the
    // surviving grouping's derivative has to land on its own arm's slot, or
    // the seeded lanes read a column belonging to the pinned grouping. Two
    // groupings are the minimum that shows it: a single pinned crossed factor
    // is an exact-zero column whose adjoint entry is zero, so a misplaced
    // derivative there cancels.
    {
        let family = Family::Binomial {
            link: BinomialLink::Logit,
        };
        let (mut ws, x, y, ids, extra_ids, p, n) = two_crossed_fixture(family);
        ws.fd.pirls_tol_override = Some(1e-12);
        let pinned = [ws.groupings.crossed[0].vech_start];
        let rng = FixedSeedTheta {
            state: 5006,
            n_theta: 3,
            diag: &[0, 1, 2],
            beta: vec![0.2, 0.8],
        };
        let (wt, wb) = assert_assembled_hessian_columns_match_fd(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            family,
            "crossed6x4_pinned",
            DRAWS,
            rng,
            &pinned,
            BAND,
        );
        cells.push((format!("{family:?}/crossed6x4_pinned"), wt, wb));
        ws.fd.pirls_tol_override = None;
    }
    // One cell above the lane ladder's top rung: `q2s` (n_θ = 3) padded by 12
    // zero-truth columns gives `m = 17`, so the pass runs in three `Dual<6>`
    // chunks (`NLanes::pick_assembled`) and the last chunk's five columns land
    // at `base = 12`. Nothing
    // else in this file compares a chunked Hessian column against anything.
    {
        let family = Family::Poisson {
            link: PoissonLink::Log,
        };
        let extra_p = 12;
        let (mut ws, x, y, ids, p, n) = fixture_padded(family, "q2s", extra_p);
        assert_eq!(ws.n_theta + p, 17, "the padding must give three chunks");
        ws.fd.pirls_tol_override = Some(1e-12);
        let rng = fixed_seed_theta_padded("q2s", extra_p);
        let (wt, wb) = assert_assembled_hessian_columns_match_fd(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &[],
            p,
            n,
            family,
            "q2s_p14",
            DRAWS,
            rng,
            &[],
            BAND,
        );
        cells.push((format!("{family:?}/q2s_p14"), wt, wb));
        ws.fd.pirls_tol_override = None;
    }
    for (name, wt, wb) in &cells {
        println!("hessian-column cell {name}: worst relative gap theta {wt:e}, beta {wb:e}");
    }
}

/// Shared per-draw body of the supply check. Every coordinate of every draw is
/// compared — none is skipped and none is filtered — and the worst relative
/// gap is returned split by coordinate class, `(θ, β)`.
#[allow(clippy::too_many_arguments)]
fn assert_assembled_hessian_columns_match_fd(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    family: Family,
    shape: &str,
    n_draws: usize,
    mut rng: FixedSeedTheta,
    // γ coordinates forced to 0.0 in every draw — a θ-pinned crossed
    // grouping. They are also the only coordinates NOT seeded: at a pinned
    // θ_e the `f64` packer drops that grouping's column, so the `f64`
    // gradient's own θ_e entry is identically zero and a central difference
    // in that direction has nothing to difference. Every other coordinate is
    // seeded, and the pinned grouping is what the seeded ones have to pair
    // around.
    pinned: &[usize],
    band: f64,
) -> (f64, f64) {
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let kk = ws.k.max(1);
    let mut worst_theta = 0.0f64;
    let mut worst_beta = 0.0f64;
    for _ in 0..n_draws {
        ws.params[..m].copy_from_slice(&rng.next_params());
        for &j in pinned {
            ws.params[j] = 0.0;
        }
        let saved: Vec<f64> = ws.params[..m].to_vec();
        // The whole Hessian once per draw, unsymmetrized so each column is the
        // chunk's own answer. Forward-mode lanes are independent, so column
        // `coord` of a pass seeding a whole chunk is the number a pass seeding
        // `coord` alone would return.
        for v in ws.pirls.u[..kk].iter_mut() {
            *v = 0.0;
        }
        let mut hess = Mat::<f64>::zeros(m, m);
        let mut hgrad = vec![0.0; m];
        let st = super::assembled::joint_hessian_columns(
            ws, x, y, ids, extra_ids, p, n, &mut hgrad, &mut hess,
        );
        let DerivStatus::Ok(v_hess) = st else {
            panic!("{family:?}/{shape}: the assembled Hessian declined")
        };
        // The objective value the pass reports is the objective, whichever
        // chunk it came out of. Asserted in release, not only in debug: on the
        // padded cell this pass runs three chunks, and a chunk that
        // differentiated a different point would show here.
        for v in ws.pirls.u[..kk].iter_mut() {
            *v = 0.0;
        }
        let mut gref = vec![0.0; m];
        let DerivStatus::Ok(v_grad) = laplace_gradient(ws, x, y, ids, extra_ids, p, n, &mut gref)
        else {
            panic!("{family:?}/{shape}: the reference gradient declined")
        };
        assert!(
            (v_hess - v_grad).abs() <= 1e-9 * (1.0 + v_grad.abs()),
            "{family:?}/{shape}: the chunked pass reports objective {v_hess}, \
             the dual gradient reports {v_grad}"
        );
        // And the gradient the pass writes on its FIRST chunk is the gradient,
        // entry by entry — the value alone would not catch a chunk that filled
        // `grad` from a different point.
        for a in 0..m {
            let gap = (hgrad[a] - gref[a]).abs() / gref[a].abs().max(1.0);
            assert!(
                gap <= 1e-9,
                "{family:?}/{shape} coord {a}: the chunked pass's gradient {} vs \
                 the dual gradient {} (relative gap {gap:e})",
                hgrad[a],
                gref[a]
            );
        }
        for coord in 0..m {
            if pinned.contains(&coord) {
                continue;
            }
            let h = if coord < n_theta {
                1e-5
            } else {
                1e-5 * saved[coord].abs().max(1.0)
            };
            // Cold-start `ws.pirls.u` before EVERY evaluation, as the FD gradient
            // gate does: a warm start chained across the coordinate loop
            // converges to the same objective at a slightly different `u`, and
            // the difference reaches a differenced gradient at a scale this
            // band would catch.
            let eval = |ws: &mut GlmmWorkspace, at: &[f64], out: &mut [f64]| {
                ws.params[..m].copy_from_slice(at);
                for v in ws.pirls.u[..kk].iter_mut() {
                    *v = 0.0;
                }
                super::assembled::gradient_f64(ws, x, y, ids, extra_ids, p, n, out).unwrap_or_else(
                    || panic!("{family:?}/{shape} coord {coord}: the assembled gradient declined"),
                );
            };
            let mut at = saved.clone();
            at[coord] = saved[coord] + h;
            let mut gp = vec![0.0; m];
            eval(ws, &at, &mut gp);
            at[coord] = saved[coord] - h;
            let mut gm = vec![0.0; m];
            eval(ws, &at, &mut gm);

            ws.params[..m].copy_from_slice(&saved);

            for a in 0..m {
                let fd = (gp[a] - gm[a]) / (2.0 * h);
                let gap = (hess[(a, coord)] - fd).abs() / fd.abs().max(1.0);
                if coord < n_theta {
                    worst_theta = worst_theta.max(gap);
                } else {
                    worst_beta = worst_beta.max(gap);
                }
                assert!(
                    gap <= band,
                    "{family:?}/{shape} column {coord} entry {a}: lane {} vs fd {fd} (relative gap {gap:e})",
                    hess[(a, coord)]
                );
            }
        }
        ws.params[..m].copy_from_slice(&saved);
    }
    (worst_theta, worst_beta)
}

/// One gate cell padded past `m = 8` so `Dual<12>` is exercised by a numerical
/// gradient check, not only by the speed campaign's padded timing run. `q2s`
/// (n_θ=3) padded by 7 zero-truth columns (`p = 2 + 7 = 9`) lands exactly on
/// `m = 12`.
#[test]
fn dual_gradient_matches_central_fd_padded_to_n12() {
    let family = Family::Poisson {
        link: PoissonLink::Log,
    };
    let shape = "q2s";
    let extra_p = 7;
    let (mut ws, x, y, ids, p, n) = fixture_padded(family, shape, extra_p);
    ws.fd.pirls_tol_override = Some(1e-12);
    let m = ws.n_theta + p;
    assert_eq!(m, 12, "padding must land exactly on the N=12 band");
    let rng = fixed_seed_theta_padded(shape, extra_p);
    assert_dual_gradient_matches_fd(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &[],
        p,
        n,
        family,
        shape,
        10,
        rng,
    );
    ws.fd.pirls_tol_override = None;
}

/// The GLMM gradient above the lane cap: `q2s` (n_θ=3) padded by 12 zero-truth
/// columns gives `m = 17`, two `Dual<12>` passes. Central FD is the reference,
/// the same instrument and band the unpadded gate uses.
#[test]
fn dual_gradient_matches_central_fd_chunked_above_n12() {
    let family = Family::Poisson {
        link: PoissonLink::Log,
    };
    let shape = "q2s";
    let extra_p = 12;
    let (mut ws, x, y, ids, p, n) = fixture_padded(family, shape, extra_p);
    ws.fd.pirls_tol_override = Some(1e-12);
    let m = ws.n_theta + p;
    assert_eq!(m, 17, "padding must land above the N=12 rung");
    assert_eq!(NLanes::pick(m, false), Some(NLanes::D12));
    let rng = fixed_seed_theta_padded(shape, extra_p);
    assert_dual_gradient_matches_fd(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &[],
        p,
        n,
        family,
        shape,
        10,
        rng,
    );
    ws.fd.pirls_tol_override = None;
}

/// Shared per-draw body of the Hessian FD gate: `n_draws` fixed-seed θ (+
/// fixed β) draws. `laplace_hessian`'s Hessian is checked against a central
/// FD stencil of `laplace_gradient`'s ANALYTIC gradient — not a second
/// difference of the deviance itself — matching how `laplace_hessian` is
/// built (both objectives come off the same kernel calls). Step and band as
/// the gradient gate above. Every gradient eval also cold-starts `ws.pirls.u` first
/// (same reason as `assert_dual_gradient_matches_fd`'s directional evals: the
/// `f64` mode solve underneath `laplace_gradient` warm-starts from whatever
/// `ws.pirls.u` currently holds, and the plus/minus evals need to land in the same
/// basin for their difference to be the smooth branch's curvature). Also
/// asserts `hess[(i, j)] == hess[(j, i)]` exactly (`==`, no band) — both
/// entries are copies of the same packed `h` slot. On a masked workspace each
/// draw has its structural θ entries zeroed, and the gradient is differenced
/// along the free coordinates only. Returns the number of entry checks.
#[allow(clippy::too_many_arguments)]
fn assert_dual_hessian_matches_fd_of_gradient(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    ids: &[u32],
    // As `assert_dual_gradient_matches_fd`'s own `extra_ids`.
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    family: Family,
    shape: &str,
    n_draws: usize,
    mut rng: FixedSeedTheta,
) -> usize {
    let m = ws.n_theta + p;
    let kk = ws.k.max(1);
    let structural = ws.groupings.structural_theta().to_vec();
    let mut checks = 0;
    for _ in 0..n_draws {
        ws.params[..m].copy_from_slice(&rng.next_params());
        ws.groupings.zero_structural(&mut ws.params[..m]);
        let saved: Vec<f64> = ws.params[..m].to_vec();
        let mut grad = vec![0.0; m];
        let mut hess = Mat::<f64>::zeros(m, m);
        let st = laplace_hessian(ws, x, y, ids, extra_ids, p, n, &mut grad, &mut hess);
        assert!(
            matches!(st, DerivStatus::Ok(_)),
            "{family:?}/{shape} did not converge"
        );

        for i in 0..m {
            for j in 0..m {
                assert_eq!(
                    hess[(i, j)],
                    hess[(j, i)],
                    "{family:?}/{shape} hess[{i}][{j}] vs hess[{j}][{i}]"
                );
            }
        }

        for k in (0..m).filter(|k| structural.binary_search(k).is_err()) {
            let h = if k < ws.n_theta {
                1e-5
            } else {
                1e-5 * saved[k].abs().max(1.0)
            };
            for v in ws.pirls.u[..kk].iter_mut() {
                *v = 0.0;
            }
            ws.params[k] = saved[k] + h;
            let mut grad_p = vec![0.0; m];
            let stp = laplace_gradient(ws, x, y, ids, extra_ids, p, n, &mut grad_p);
            assert!(
                matches!(stp, DerivStatus::Ok(_)),
                "{family:?}/{shape} coord {k} (+h) did not converge"
            );
            for v in ws.pirls.u[..kk].iter_mut() {
                *v = 0.0;
            }
            ws.params[k] = saved[k] - h;
            let mut grad_m = vec![0.0; m];
            let stm = laplace_gradient(ws, x, y, ids, extra_ids, p, n, &mut grad_m);
            assert!(
                matches!(stm, DerivStatus::Ok(_)),
                "{family:?}/{shape} coord {k} (-h) did not converge"
            );
            ws.params[k] = saved[k];

            for row in 0..m {
                let fd = (grad_p[row] - grad_m[row]) / (2.0 * h);
                assert!(
                    (hess[(row, k)] - fd).abs() <= 1e-5 * fd.abs().max(1.0),
                    "{family:?}/{shape} hess[{row}][{k}]: analytic {} vs fd-of-grad {fd}",
                    hess[(row, k)]
                );
                checks += 1;
            }
        }
    }
    checks
}

/// Central FD of `laplace_gradient` against `laplace_hessian`, per family and
/// per RE shape, at a TIGHTENED PIRLS tolerance — same reasoning and same
/// tolerance as `dual_gradient_matches_central_fd_per_family_and_shape`.
/// The masked cells ([`MASKED_GATE_CELLS`]) run beside the list, and the gate
/// prints the check count of both.
#[test]
fn dual_hessian_matches_central_fd_of_gradient_per_family_and_shape() {
    let mut checks = [0usize; 2];
    for (set, cells) in [GRADIENT_GATE_CELLS, MASKED_GATE_CELLS]
        .into_iter()
        .enumerate()
    {
        for &(family, shape) in cells {
            let (mut ws, x, y, ids, p, n) = fixture(family, shape);
            ws.fd.pirls_tol_override = Some(1e-12);
            let rng = fixed_seed_theta(shape);
            checks[set] += assert_dual_hessian_matches_fd_of_gradient(
                &mut ws,
                x.as_ref(),
                &y,
                &ids,
                &[],
                p,
                n,
                family,
                shape,
                10,
                rng,
            );
            ws.fd.pirls_tol_override = None;
        }
    }
    println!(
        "dual-Hessian FD gate: {} checks on {} cells, {} on {} masked cells",
        checks[0],
        GRADIENT_GATE_CELLS.len(),
        checks[1],
        MASKED_GATE_CELLS.len()
    );
    // Ten draws × three free columns × five rows per masked cell.
    assert_eq!(checks[1], MASKED_GATE_CELLS.len() * 10 * 3 * 5);
}

/// One gate cell padded past `m = 8` so `HyperDual<12, 78>` is exercised by a
/// numerical Hessian check — mirrors `dual_gradient_matches_central_fd_padded_to_n12`.
#[test]
fn dual_hessian_matches_central_fd_of_gradient_padded_to_n12() {
    let family = Family::Poisson {
        link: PoissonLink::Log,
    };
    let shape = "q2s";
    let extra_p = 7;
    let (mut ws, x, y, ids, p, n) = fixture_padded(family, shape, extra_p);
    ws.fd.pirls_tol_override = Some(1e-12);
    let m = ws.n_theta + p;
    assert_eq!(m, 12, "padding must land exactly on the N=12 band");
    let rng = fixed_seed_theta_padded(shape, extra_p);
    assert_dual_hessian_matches_fd_of_gradient(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &[],
        p,
        n,
        family,
        shape,
        10,
        rng,
    );
    ws.fd.pirls_tol_override = None;
}

// --- The same two FD gates on the structured-extras route ---

/// Family × extras-shape fixture for the structured FD gates and the
/// structured exact-profile unit gate. Reuses `glmm_extras_q1_dataset`'s
/// design wholesale — X, primary ids, extra ids and the RE structure — and
/// only swaps the family and, where the domain differs, regenerates `y`: the
/// generator draws Bernoulli, which every Binomial link already shares, so
/// Poisson/NegativeBinomial (counts) and Gamma (a continuous positive
/// response) are the cases that need their own draw. Same
/// regenerate-y-on-a-fixed-design construction `fixture` uses for the blocked
/// cells, with its own seed stream; the Gamma and NegativeBinomial arms copy
/// `fixture_with_nagq_sized`'s own. The Bernoulli draw is generated under
/// `p = sigmoid(eta)` regardless of which Binomial link is being fit, so
/// cloglog fits it too — not degenerate here: `eta` stays inside roughly
/// `[0.2, 2.1]` on this design, so `p` never approaches 0 or 1 closely enough
/// to flatten cloglog's `dμ/dη`, confirmed by the passing FD gate cells below
/// and by the structured-exact-profile unit gate's cloglog cell.
///
/// `ws.pattern.structured_schur` is built here, so on the crossed cells the FD
/// reference (`laplace_deviance_ws`) runs the production cached sparse tail
/// while the dual gradient runs the dense generic one. The gate therefore
/// covers the tail routing on top of the chain rule: the two are a
/// reassociation of the same Cholesky, orders of magnitude inside the 1e-6
/// band. On the nested-only cell `StructuredSchur::new` returns `None`
/// (`e == 0`) and both sides take the dense arm, which is the whole tail
/// story there.
#[allow(clippy::type_complexity)]
fn extras_fixture(
    family: Family,
    np: usize,
    n_crossed: usize,
) -> (
    GlmmWorkspace,
    Mat<f64>,
    Vec<f64>,
    Vec<u32>,
    Vec<Vec<u32>>,
    usize,
    usize,
) {
    let (x, y_bin, ids, extra_ids, mut spec) = glmm_extras_q1_dataset(np, n_crossed);
    let (n, p) = (y_bin.len(), 2usize);
    let mut st = 6101u64;
    let y = match family {
        Family::Binomial { .. } => y_bin,
        Family::Poisson { .. } | Family::NegativeBinomial { .. } => (0..n)
            .map(|i| {
                let eta = 0.2 + 0.8 * x[(i, 1)];
                let noise = 0.4 + 1.6 * (lcg(&mut st) + 0.5); // in [0.4, 2.0]
                (eta.exp().min(20.0) * noise).round().max(0.0)
            })
            .collect(),
        // Copies the blocked `fixture_with_nagq_sized`'s Gamma arm — same eta,
        // same noise term, no rounding (a Gamma response is continuous).
        Family::Gamma { .. } => (0..n)
            .map(|i| {
                let eta = 0.2 + 0.8 * x[(i, 1)];
                let noise = 0.4 + 1.6 * (lcg(&mut st) + 0.5); // in [0.4, 2.0]
                eta.exp() * noise
            })
            .collect(),
        other => panic!("structured gradient-gate fixture: family {other:?} not wired"),
    };
    spec.spec.family = family;
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &spec, n, &[], 1);
    if matches!(family, Family::NegativeBinomial { .. }) {
        ws.nb_theta = 4.0;
    }
    ws.pattern.structured_schur = StructuredSchur::new(&ws.groupings, &ids, &extra_ids, n);
    (ws, x, y, ids, extra_ids, p, n)
}

/// Newton PIRLS on the four links whose exact curvature differs from Fisher
/// (probit, cloglog, Gamma/log, NB/log): the step weight is the observed one,
/// so the solve converges quadratically and a solve that stops at a loose exit
/// band already sits on the mode. Fisher scoring converges only linearly and
/// stops short of the mode by about the square root of its band, which leaves
/// the objective noisy at that level. The Laplace deviance at a fixed
/// `(θ, β)`, solved from `û = 0` at a loose band (1e-6) and at a tight one
/// (1e-13), must agree to 1e-12 relative, on the blocked, structured and
/// packed layouts. Measured: Newton ≤ 2.1e-16 on every cell; Fisher scoring
/// 3e-11 to 1.4e-9 on these fixtures.
#[test]
fn newton_pirls_lands_on_the_mode_at_a_loose_band() {
    let families = [
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        Family::Gamma {
            link: crate::GammaLink::Log,
        },
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
    ];
    let loose_and_tight = |ws: &mut GlmmWorkspace,
                           params: &[f64],
                           x: faer::MatRef<f64>,
                           y: &[f64],
                           ids: &[u32],
                           extra_ids: &[Vec<u32>],
                           n: usize| {
        let mut out = [0.0; 2];
        for (slot, tol) in out.iter_mut().zip([1e-6, 1e-13]) {
            ws.fd.pirls_tol_override = Some(tol);
            *slot = glmm_laplace_deviance(params, ws, x, y, ids, extra_ids, n);
        }
        ws.fd.pirls_tol_override = None;
        out
    };
    for family in families {
        let mut cells: Vec<(&str, [f64; 2])> = Vec::new();
        // Blocked: one grouping, random intercept and slope.
        let (mut ws, x, y, ids, _p, n) = fixture(family, "q2s");
        assert_eq!(ws.layout, GlmmLayout::Blocked);
        let params = [0.8, 0.3, 0.6, 0.2, 0.8];
        cells.push((
            "blocked",
            loose_and_tight(&mut ws, &params, x.as_ref(), &y, &ids, &[], n),
        ));
        // Structured: nested children plus a crossed grouping.
        let (mut ws, x, y, ids, extra_ids, _p, n) = extras_fixture(family, 2, 6);
        assert_eq!(ws.layout, GlmmLayout::Structured);
        let params = [0.5, 0.4, 0.45, 0.2, 0.8];
        cells.push((
            "structured",
            loose_and_tight(&mut ws, &params, x.as_ref(), &y, &ids, &extra_ids, n),
        ));
        // Packed: the same design and response on the packed-row layout.
        let (_, _, _, _, mut spec) = glmm_extras_q1_dataset(2, 6);
        spec.spec.family = family;
        let mut ws = GlmmWorkspace::for_cluster_spec_packed(2, &spec, n, &[], 1);
        if matches!(family, Family::NegativeBinomial { .. }) {
            ws.nb_theta = 4.0;
        }
        fill_packed_cols(&mut ws, &ids, &extra_ids, n);
        cells.push((
            "packed",
            loose_and_tight(&mut ws, &params, x.as_ref(), &y, &ids, &extra_ids, n),
        ));
        for (layout, [loose, tight]) in cells {
            assert!(
                loose.is_finite() && tight.is_finite(),
                "{family:?} {layout}: loose {loose}, tight {tight}"
            );
            let rel = (loose - tight).abs() / (1.0 + tight.abs());
            assert!(
                rel < 1e-12,
                "{family:?} {layout}: the 1e-6 solve is {rel:.3e} relative off the 1e-13 one"
            );
        }
    }
}

/// `{Binomial-logit, Binomial-probit, Binomial-cloglog, Poisson-log,
/// Gamma-log, NegativeBinomial-log}` × the three structured extras shapes, as
/// `(family, shape label, nested children per parent, crossed levels)`.
///
/// The shapes are the speed campaign catalogue's `nest2` and `int2x` classes plus
/// their combination: nested-only (`q_core = 3`, `e = 0`, the tail is skipped
/// entirely), crossed-only (`q_core = 1`, `e = 6`, the rank-1 scalar downdate
/// that is the production route at `q_core == 1`), and both (`q_core = 3`,
/// `e = 6`, whose `f64` route is the panel downdate while the dual route takes
/// the scalar default). Logit and Poisson-log are canonical controls; probit,
/// cloglog, Gamma-log and NegativeBinomial-log are the four non-canonical
/// links the observed-information twin (`DualStep::observed`) actually serves
/// on this route, so between them this gate checks the twin's chain rule
/// family by family, not just once.
///
/// Remaining gap: `GammaLink::Inverse` and `InverseGaussian` (both links) are
/// also non-canonical and structured-extras-eligible, so they build and read
/// the same twin, but neither is gated here or on the blocked path's
/// `GRADIENT_GATE_CELLS` — no FD gate exercises the twin on them yet (the
/// `f64` assembly on Gamma/inverse is FD-gated by
/// `gradient_at_mode_matches_finite_differences_across_theta_regimes`). Each
/// family/link pair above stands in for its own W/μ/dμ-dη formula, which is
/// where a family or link actually enters the chain rule; the crossed-tail
/// factoring and scatter code the twin adds is link-agnostic, so the four
/// links here are believed to already exercise every branch that code takes.
const STRUCTURED_GATE_CELLS: &[(Family, &str, usize, usize)] = &[
    (
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        "nested2",
        2,
        0,
    ),
    (
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        "crossed6",
        0,
        6,
    ),
    (
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        "nested2_crossed6",
        2,
        6,
    ),
    (
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        "nested2",
        2,
        0,
    ),
    (
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        "crossed6",
        0,
        6,
    ),
    (
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        "nested2_crossed6",
        2,
        6,
    ),
    (
        Family::Poisson {
            link: PoissonLink::Log,
        },
        "nested2",
        2,
        0,
    ),
    (
        Family::Poisson {
            link: PoissonLink::Log,
        },
        "crossed6",
        0,
        6,
    ),
    (
        Family::Poisson {
            link: PoissonLink::Log,
        },
        "nested2_crossed6",
        2,
        6,
    ),
    (
        Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        "nested2",
        2,
        0,
    ),
    (
        Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        "crossed6",
        0,
        6,
    ),
    (
        Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        "nested2_crossed6",
        2,
        6,
    ),
    (
        Family::Gamma {
            link: GammaLink::Log,
        },
        "nested2",
        2,
        0,
    ),
    (
        Family::Gamma {
            link: GammaLink::Log,
        },
        "crossed6",
        0,
        6,
    ),
    (
        Family::Gamma {
            link: GammaLink::Log,
        },
        "nested2_crossed6",
        2,
        6,
    ),
    (
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        "nested2",
        2,
        0,
    ),
    (
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        "crossed6",
        0,
        6,
    ),
    (
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        "nested2_crossed6",
        2,
        6,
    ),
];

/// Central FD of the Laplace deviance against the dual gradient on the
/// structured-extras route — the same stencil, step, band and tightened PIRLS
/// tolerance as `dual_gradient_matches_central_fd_per_family_and_shape`, run
/// through the same shared helper, on the extras cells instead of the blocked
/// ones.
#[test]
fn structured_dual_gradient_matches_central_fd_per_family_and_shape() {
    for &(family, shape, np, n_crossed) in STRUCTURED_GATE_CELLS {
        let (mut ws, x, y, ids, extra_ids, p, n) = extras_fixture(family, np, n_crossed);
        ws.fd.pirls_tol_override = Some(1e-12);
        let rng = fixed_seed_theta(shape);
        assert_dual_gradient_matches_fd(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            family,
            shape,
            10,
            rng,
        );
        ws.fd.pirls_tol_override = None;
    }
}

/// Central FD of `laplace_gradient` against `laplace_hessian` on the
/// structured-extras route — same cells, same shared helper and same band as
/// `dual_hessian_matches_central_fd_of_gradient_per_family_and_shape`.
#[test]
fn structured_dual_hessian_matches_central_fd_of_the_gradient() {
    for &(family, shape, np, n_crossed) in STRUCTURED_GATE_CELLS {
        let (mut ws, x, y, ids, extra_ids, p, n) = extras_fixture(family, np, n_crossed);
        ws.fd.pirls_tol_override = Some(1e-12);
        let rng = fixed_seed_theta(shape);
        assert_dual_hessian_matches_fd_of_gradient(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            family,
            shape,
            10,
            rng,
        );
        ws.fd.pirls_tol_override = None;
    }
}

// --- The AGQ branch inside `laplace_gradient`/`laplace_hessian` ---

/// `{Binomial-logit, Poisson-log} × {int1, q2s}`, each paired with the `nagq`
/// the speed campaign's AGQ-eligible cells use (`validation/campaigns/speed-campaign/prep.R:297`,
/// `:309`) — `int1` at `k=7`, `q2s` at `k=5`. Both families/shapes satisfy the
/// AGQ gate mirrored in `derivative.rs` (`nagq > 1 && extra_offsets.is_empty()
/// && (1..=3).contains(&primary_q) && Binomial|Poisson`) and are canonical
/// links, so these cells exercise `agq_deviance` (`int1`, `q_p==1`) and
/// `agq_deviance_vec` (`q2s`, `q_p==2`) without touching the refinement loop —
/// that loop is `canonical`-gated, orthogonal to AGQ routing, and already
/// covered by the gradient/Hessian FD gates' own non-canonical cells.
const AGQ_GATE_CELLS: &[(Family, &str, u8)] = &[
    (
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        "int1",
        7,
    ),
    (
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        "q2s",
        5,
    ),
    (
        Family::Poisson {
            link: PoissonLink::Log,
        },
        "int1",
        7,
    ),
    (
        Family::Poisson {
            link: PoissonLink::Log,
        },
        "q2s",
        5,
    ),
    // Gamma-log scalar intercept: the compressed AGQ route's second family
    // (Poisson-log int1 above is its first), so the dual gradient and Hessian
    // go through `compressed_cluster_sum` on both.
    (
        Family::Gamma {
            link: GammaLink::Log,
        },
        "int1",
        7,
    ),
    // Non-canonical binomial link: `agq_eligible` admits `Binomial { .. }` at
    // any link, so this cell exercises the non-canonical (Newton-weight) PIRLS
    // path inside the AGQ node loop, not just the logit fast path the other
    // cells cover.
    (
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        "int1",
        7,
    ),
];

/// Central FD of the AGQ deviance against the dual gradient, per
/// `AGQ_GATE_CELLS` cell, at the same tightened PIRLS tolerance as the
/// gradient FD gate. Reuses `assert_dual_gradient_matches_fd` verbatim: its FD reference
/// (`laplace_deviance_ws`) already routes through the production AGQ gate at
/// `ws.nagq > 1` (`deviance.rs:245`, unmodified), so it evaluates the SAME
/// AGQ objective `laplace_gradient`'s own AGQ branch differentiates — no new
/// FD evaluator needed.
#[test]
fn agq_dual_gradient_matches_central_fd() {
    for &(family, shape, nagq) in AGQ_GATE_CELLS {
        let (mut ws, x, y, ids, p, n) = fixture_with_nagq(family, shape, nagq);
        ws.fd.pirls_tol_override = Some(1e-12);
        let rng = fixed_seed_theta(shape);
        assert_dual_gradient_matches_fd(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &[],
            p,
            n,
            family,
            shape,
            10,
            rng,
        );
        ws.fd.pirls_tol_override = None;
    }
}

/// Central FD of `agq_dual_gradient_matches_central_fd`'s own analytic
/// gradient against the dual Hessian, per `AGQ_GATE_CELLS` cell — mirrors
/// `agq_dual_gradient_matches_central_fd`, reusing
/// `assert_dual_hessian_matches_fd_of_gradient` verbatim for the same reason.
#[test]
fn agq_dual_hessian_matches_central_fd() {
    for &(family, shape, nagq) in AGQ_GATE_CELLS {
        let (mut ws, x, y, ids, p, n) = fixture_with_nagq(family, shape, nagq);
        ws.fd.pirls_tol_override = Some(1e-12);
        let rng = fixed_seed_theta(shape);
        assert_dual_hessian_matches_fd_of_gradient(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &[],
            p,
            n,
            family,
            shape,
            10,
            rng,
        );
        ws.fd.pirls_tol_override = None;
    }
}

/// The shipped `f64` AGQ node sums (`agq_fast::scalar_sums`, `vector_sums`,
/// `probit_sums`) against the generic node loop of the same production
/// `agq_deviance`/`agq_deviance_vec`: the fast tiles run only when every
/// buffer is `f64`, so the value `laplace_gradient` returns from its `Dual`
/// evaluation is the generic loop's (compressed route included). Per
/// `AGQ_GATE_CELLS` cell, at 10 draws and a 1e-12 PIRLS tolerance on both
/// sides, the two agree to 1e-9 relative; the FD gate above sees the tiles
/// only through 1e-6 bands. The cluster-outer arm of `scalar_sums` runs only in
/// a `parallel` build with `parallel_inner` set, so it is not reached here: its
/// owner is `agq_cluster_outer_bit_identical_to_node_outer`.
#[test]
fn agq_fast_route_value_matches_the_generic_dual_route() {
    for &(family, shape, nagq) in AGQ_GATE_CELLS {
        let (mut ws, x, y, ids, p, n) = fixture_with_nagq(family, shape, nagq);
        ws.fd.pirls_tol_override = Some(1e-12);
        let mut rng = fixed_seed_theta(shape);
        let m = ws.n_theta + p;
        let mut grad = vec![0.0; m];
        for draw in 0..10 {
            let params = rng.next_params();
            let fast = glmm_laplace_deviance(&params, &mut ws, x.as_ref(), &y, &ids, &[], n);
            ws.params[..m].copy_from_slice(&params);
            let st = laplace_gradient(&mut ws, x.as_ref(), &y, &ids, &[], p, n, &mut grad);
            let DerivStatus::Ok(generic) = st else {
                panic!(
                    "{family:?}/{shape} k={nagq} draw {draw}: dual route not Ok (NotConverged: {})",
                    matches!(st, DerivStatus::NotConverged)
                );
            };
            let rel = (fast - generic).abs() / fast.abs().max(1.0);
            assert!(
                rel <= 1e-9,
                "{family:?}/{shape} k={nagq} draw {draw}: fast {fast} generic {generic} (rel {rel:e})"
            );
        }
        ws.fd.pirls_tol_override = None;
    }
}

/// The packed layout's sparse Cholesky of `A` against its dense factor on the
/// same design from the same start: a full fit (stage-1 PQL border solves
/// included) and the Rx SE, which `packed_schur_fill` builds from the dense
/// `A` the sparse path expands after its final refresh. The two factor
/// orders differ, so the agreement is round-off carried through BOBYQA: band
/// 1e-6 relative on β̂ and the SEs, 1e-8 on the deviance.
#[test]
fn packed_sparse_factor_matches_the_dense_one() {
    let (xf64, y, ids, crossed_ids, cluster) = glmm_slope_crossed_dataset();
    let n = y.len();
    let extra = vec![crossed_ids];
    let run = |sparse: bool| {
        let mut ws = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[1], 1);
        fill_packed_cols_with(&mut ws, &ids, &extra, n, Some(sparse));
        assert_eq!(ws.packed.sparse.is_some(), sparse);
        let fit = fit_glmm(
            &mut ws,
            xf64.as_ref(),
            xf64.as_ref().subcols(0, 0),
            &y,
            &ids,
            &extra,
            &[0, 1],
            None,
            &[0.0, 0.0],
            n,
            WaldSe::Rx,
            ColumnScaling::IDENTITY,
        );
        assert!(fit.converged, "sparse = {sparse}: must converge");
        (
            fit.deviance,
            ws.betas.clone(),
            ws.inference.var_diag.clone(),
        )
    };
    let (dd, bd, vd) = run(false);
    let (ds, bs, vs) = run(true);
    assert!(
        (dd - ds).abs() <= 1e-8 * dd.abs(),
        "deviance dense {dd} vs sparse {ds}"
    );
    for j in 0..2 {
        assert!(
            (bd[j] - bs[j]).abs() <= 1e-6 * bd[j].abs().max(1.0),
            "β[{j}] {} vs {}",
            bd[j],
            bs[j]
        );
        let (a, b) = (vd[j].sqrt(), vs[j].sqrt());
        assert!((a - b).abs() <= 1e-6 * a, "se[{j}] dense {a} vs sparse {b}");
    }
}

/// `fill_packed_cols` owns the sparse pattern: a `loop_advanced` caller that
/// refills a reused workspace with new level ids at the same `n` gets the
/// pattern of the new ids, identical to a fresh workspace's.
#[test]
fn fill_packed_cols_rebuilds_the_sparse_pattern_for_new_ids() {
    let (_x, y, ids, crossed_a, cluster) = glmm_slope_crossed_dataset();
    let n = y.len();
    let crossed_b: Vec<u32> = (0..n as u32).map(|i| (i / 2) % 4).collect();
    let mut reused = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[1], 1);
    fill_packed_cols_with(&mut reused, &ids, &[crossed_a], n, Some(true));
    fill_packed_cols_with(
        &mut reused,
        &ids,
        std::slice::from_ref(&crossed_b),
        n,
        Some(true),
    );
    let mut fresh = GlmmWorkspace::for_cluster_spec_packed(2, &cluster, n, &[1], 1);
    fill_packed_cols_with(&mut fresh, &ids, &[crossed_b], n, Some(true));
    let (r, f) = (
        reused.packed.sparse.as_ref().unwrap(),
        fresh.packed.sparse.as_ref().unwrap(),
    );
    assert_eq!(r.axx.symbolic().col_ptr(), f.axx.symbolic().col_ptr());
    assert_eq!(r.axx.symbolic().row_idx(), f.axx.symbolic().row_idx());
    assert_eq!(r.row_slots, f.row_slots);
}

/// The Laplace objective's `log|A|` is the EXACT curvature of the conditional
/// log-density at the mode on every link, not the Fisher approximation: on a
/// scalar random intercept, `A` is diagonal with
/// `h_f = ½·∂²/∂u_f² [D(u) + ‖u‖²]` at û, so `laplace_deviance − D(û) − ‖û‖²`
/// must equal `Σ_f ln h_f`. `h_f` is taken here straight from the data by a
/// central second difference of the deviance in `u_f` (the clusters decouple),
/// with no kernel code on the reference side. Probit, cloglog, NB/log and
/// Gamma/log — the links where the two curvatures differ — plus logit as the
/// canonical control. Band 1e-6 relative on `log|A|`: the O(h²) truncation of
/// the stencil at h = 1e-4.
#[test]
fn laplace_log_det_is_the_exact_curvature_at_the_mode() {
    let (n, s) = (60usize, 6usize);
    let theta = 0.8_f64;
    let beta = [0.1_f64, 0.3];
    let mut x = Mat::<f64>::zeros(n, 2);
    let ids: Vec<u32> = (0..n).map(|i| (i % s) as u32).collect();
    for i in 0..n {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = ((i * 7) % 11) as f64 / 10.0 - 0.5;
    }
    let fams = [
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        },
        Family::Gamma {
            link: crate::GammaLink::Log,
        },
        Family::Binomial {
            link: BinomialLink::Logit,
        },
    ];
    for family in fams {
        let y: Vec<f64> = (0..n)
            .map(|i| match family {
                Family::Binomial { .. } => f64::from(u8::from((i * 7) % 3 == 0 || i % 5 == 1)),
                Family::NegativeBinomial { .. } => ((i * 3) % 5) as f64,
                _ => 0.5 + ((i * 13) % 7) as f64 / 3.0,
            })
            .collect();
        let model = SizedSpec {
            spec: ModelSpec {
                family,
                re: Some(ReStructure {
                    terms: Vec::new(),
                    slopes: vec![],
                    extra_groupings: vec![],
                }),
            },
            n_primary: s,
            extra_counts: vec![],
        };
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &model, n, &[], 1);
        let nb_theta = 2.0;
        ws.nb_theta = nb_theta;
        let obj = glmm_laplace_deviance(
            &[theta, beta[0], beta[1]],
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &[],
            n,
        );
        assert!(obj.is_finite(), "{family:?}: objective must be finite");
        let u: Vec<f64> = ws.pirls.u[..s].to_vec();
        let dev_rows = |f: usize, uf: f64| -> f64 {
            (0..n)
                .filter(|&i| ids[i] as usize == f)
                .map(|i| {
                    let eta = beta[0] + beta[1] * x[(i, 1)] + theta * uf;
                    let mu = crate::family::link_inv(family, eta);
                    crate::family::dev_resid(family, nb_theta, y[i], mu)
                })
                .sum()
        };
        let dev_hat: f64 = (0..s).map(|f| dev_rows(f, u[f])).sum();
        let pen: f64 = u.iter().map(|v| v * v).sum();
        let h = 1e-4;
        let log_det_ref: f64 = (0..s)
            .map(|f| {
                let d2 = (dev_rows(f, u[f] + h) - 2.0 * dev_rows(f, u[f]) + dev_rows(f, u[f] - h))
                    / (h * h);
                (0.5 * d2 + 1.0).ln()
            })
            .sum();
        let log_det = obj - dev_hat - pen;
        assert!(
            (log_det - log_det_ref).abs() <= 1e-6 * log_det_ref.abs().max(1.0),
            "{family:?}: log|A| = {log_det} vs exact curvature {log_det_ref}"
        );
    }
}

/// A fitted `ln φ̂` on either bound of the Gamma coordinate's box is refused;
/// one just inside the tolerance is not.
#[test]
fn gamma_phi_on_box_bound_flags_both_bounds_only() {
    let (lo, hi) = (GAMMA_PHI_LO.ln(), GAMMA_PHI_HI.ln());
    assert!(gamma_phi_on_box_bound(lo));
    assert!(gamma_phi_on_box_bound(hi));
    assert!(gamma_phi_on_box_bound(lo + 5e-7));
    assert!(!gamma_phi_on_box_bound(lo + 1e-3));
    assert!(!gamma_phi_on_box_bound(hi - 1e-3));
    assert!(!gamma_phi_on_box_bound(0.0));
}

/// Negative-binomial AGQ: θ_NB sits outside the integral, so the
/// binomial/Poisson kernels serve NB unchanged. At a fixed (θ, β) the
/// quadrature ladder converges (7, 15 and 25 nodes agree to quadrature
/// error), and one adaptive node is the Laplace objective to round-off. On NB
/// and probit — links where the exact curvature differs from Fisher — the node
/// scale reads the factor the exit refresh leaves, `A_obs`, so the k = 1
/// reduction holds with the exact-curvature `log|A|` too. The logit row holds
/// the reduction on the canonical link at a tighter band; the scalar k = 1 node
/// sums dev_resid and a log-sum-exp where Laplace sums dev/pen and 2·logdet, so
/// the two agree to a relative tolerance, not bitwise.
#[test]
fn agq_nb_ladder_converges_and_one_node_is_laplace() {
    let (n, s) = (60usize, 6usize);
    let mut x = Mat::<f64>::zeros(n, 2);
    let ids: Vec<u32> = (0..n).map(|i| (i % s) as u32).collect();
    for i in 0..n {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = ((i * 7) % 11) as f64 / 10.0 - 0.5;
    }
    let params = [0.8_f64, 0.1, 0.3];
    // (family, k = 1 band, run the node ladder)
    for (family, k1_tol, ladder) in [
        (
            Family::NegativeBinomial {
                link: crate::NegBinomialLink::Log,
            },
            1e-10,
            true,
        ),
        (
            Family::Binomial {
                link: BinomialLink::Probit,
            },
            1e-10,
            true,
        ),
        (
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            1e-12,
            false,
        ),
    ] {
        let y: Vec<f64> = (0..n)
            .map(|i| match family {
                Family::Binomial { .. } => f64::from(u8::from((i * 7) % 3 == 0 || i % 5 == 1)),
                _ => ((i * 3) % 5) as f64,
            })
            .collect();
        let model = SizedSpec {
            spec: ModelSpec {
                family,
                re: Some(ReStructure {
                    terms: Vec::new(),
                    slopes: vec![],
                    extra_groupings: vec![],
                }),
            },
            n_primary: s,
            extra_counts: vec![],
        };
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &model, n, &[], 1);
        ws.nb_theta = 2.0;
        let lap = glmm_laplace_deviance(&params, &mut ws, x.as_ref(), &y, &ids, &[], n);
        let agq = |ws: &mut GlmmWorkspace, k: u8| {
            glmm_agq_deviance(&params, ws, x.as_ref(), &y, &ids, n, k)
        };
        let agq1 = agq(&mut ws, 1);
        assert!(
            (agq1 - lap).abs() <= k1_tol * lap.abs().max(1.0),
            "{family:?}: agq(k=1) {agq1} vs laplace {lap}"
        );
        if !ladder {
            continue;
        }
        let (d7, d15, d25) = (agq(&mut ws, 7), agq(&mut ws, 15), agq(&mut ws, 25));
        assert!(d25.is_finite(), "{family:?}: AGQ must evaluate");
        assert!(
            (d15 - d25).abs() <= 1e-7 * d25.abs(),
            "{family:?}: 15 vs 25 nodes: {d15} vs {d25}"
        );
        assert!(
            (d7 - d25).abs() <= 1e-4 * d25.abs(),
            "{family:?}: 7 vs 25 nodes: {d7} vs {d25}"
        );
    }
}

/// On the AGQ route the fit keeps the search's own Λ at a pinned diagonal, and
/// the reported deviance is the AGQ objective at that Λ. The data carry a
/// random slope and no random intercept (x in (1, 3), 20 clusters), so the fit
/// pins the intercept diagonal and the search stops with a small live entry
/// below it. The vector AGQ deviance is not a function of Σ alone: the first
/// check rotates that column at fixed Σ and sees the deviance move. Folding
/// the entry into the slope diagonal would therefore report a Λ the search did
/// not choose, at a deviance it never reached. The pin flags still come from
/// the canonical form, which names the intercept alone.
#[test]
fn fit_glmm_vector_agq_keeps_search_lambda_at_pinned_diagonal() {
    let (n, nc, nagq) = (400usize, 20usize, 7u8);
    // This design has two boundary basins, the intercept diagonal pinned or the
    // slope diagonal pinned under a rank-one Λ, and which one the search reaches
    // depends on its path. Seed 278 lands on the intercept-pinned face on both
    // β parameterisations of the joint search, at the same deviance.
    let (x_rows, y, ids) = pinned_diagonal_fixture(278, 0.0);
    let x = Mat::from_fn(n, 2, |i, j| x_rows[i * 2 + j]);
    let spec = SizedSpec {
        spec: ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        },
        n_primary: nc,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &spec, n, &[1], nagq);
    let fit = fit_glmm(
        &mut ws,
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[1],
        None,
        &[0.0, 0.0],
        n,
        WaldSe::Hessian,
        ColumnScaling::IDENTITY,
    );
    assert!(fit.converged);
    assert_eq!(fit.pinned_components, 0b01, "intercept pinned, slope live");
    let params = ws.params[..5].to_vec();
    assert_eq!(params[0], 0.0, "intercept diagonal pinned");
    assert!(
        params[1] != 0.0,
        "the search's entry below the pinned diagonal is kept, got θ = {:?}",
        &params[..3]
    );

    let mut ws_eval = GlmmWorkspace::for_cluster_spec(2, &spec, n, &[1], nagq);
    let dev_at = |ws_eval: &mut GlmmWorkspace, lambda: [f64; 3]| {
        let mut pr = params.clone();
        pr[..3].copy_from_slice(&lambda);
        glmm_agq_deviance(&pr, ws_eval, x.as_ref(), &y, &ids, n, nagq)
    };
    // Same Σ, two Λ: the column below the pinned diagonal rotated by a visible
    // angle against the canonical Λ.
    let r = (params[1] * params[1] + params[2] * params[2]).sqrt();
    let canonical = dev_at(&mut ws_eval, [0.0, 0.0, r]);
    let rotated = dev_at(&mut ws_eval, [0.0, 0.6 * r, 0.8 * r]);
    assert!(
        (rotated - canonical).abs() > 1e-4,
        "AGQ deviance must depend on Λ at fixed Σ here: {rotated} vs {canonical}"
    );

    let at_fit = dev_at(&mut ws_eval, [params[0], params[1], params[2]]);
    assert!(
        (fit.deviance - at_fit).abs() <= 1e-9 * at_fit.abs(),
        "reported deviance {} vs AGQ deviance at the reported θ̂, β̂ {at_fit}",
        fit.deviance
    );
}

/// The predicate `fit_glmm` applies to its own BOBYQA outcome
/// (`!(status == Converged && any_within(non_finite_points, x, NON_FINITE_RADIUS))`,
/// `glmm/mod.rs`), exercised on a plain 1-D `Bobyqa` run with no GLMM
/// machinery, at this test's own radius `config.rho_begin` (1.0). `f` is a
/// smooth quadratic with its true minimum at 5.5, except a narrow `+INFINITY`
/// band around 5.0 that sits inside that radius of the minimum. The search
/// still reports the true minimum as `Converged`, and `any_within` flags that
/// the optimum sits next to a failed evaluation. A point outside the radius is
/// not flagged. Only the predicate is tested here:
/// the fit-level status downgrade that consumes it has no test.
#[test]
fn any_within_flags_an_optimum_next_to_a_non_finite_eval() {
    let mut config = bobyqa::Config::new(1);
    config.rho_begin = 1.0;
    config.rho_end = 1e-8;
    let mut solver = bobyqa::Bobyqa::new(1, config).expect("valid 1-D config");
    let mut non_finite_points = Vec::new();
    let mut x = [5.0];
    let out = solver.minimize(
        |p| {
            let v = p[0];
            if (v - 5.0).abs() < 0.02 {
                non_finite_points.push(v);
                f64::INFINITY
            } else {
                (v - 5.5) * (v - 5.5)
            }
        },
        &mut x,
        &[-10.0],
        &[10.0],
    );
    assert_eq!(out.status, bobyqa::Status::Converged);
    assert!(
        (x[0] - 5.5).abs() < 1e-3,
        "expected the search to reach the true minimum 5.5, got {}",
        x[0]
    );
    assert!(
        !non_finite_points.is_empty(),
        "the search must have crossed the +INFINITY band at least once"
    );
    // The exact condition `fit_glmm` checks before trusting a `Converged`
    // status: this is the case it must catch.
    assert!(
        any_within(&non_finite_points, &x, config.rho_begin),
        "the reported optimum {} must sit within rho_begin of a failed \
         evaluation for this fixture to test anything",
        x[0]
    );
    // Negative control: a failed evaluation farther than `rho_begin` away is not
    // a neighbor.
    assert!(!any_within(&[3.0], &[5.5], config.rho_begin));
}

/// The data of `fit_glmm_vector_agq_keeps_search_lambda_at_pinned_diagonal`
/// for any seed, row-major `[1, x1 + shift]`: binary logit, 400 rows in 20
/// clusters, a random slope on `x1` drawn per cluster.
pub(crate) fn pinned_diagonal_fixture(seed: u64, shift: f64) -> (Vec<f64>, Vec<f64>, Vec<u32>) {
    let (n, nc) = (400usize, 20usize);
    let mut st = seed;
    let u1: Vec<f64> = (0..nc).map(|_| 2.0 * lcg(&mut st)).collect();
    let mut x = vec![0.0f64; n * 2];
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = i % nc;
        ids[i] = c as u32;
        let x1 = 2.0 + 2.0 * lcg(&mut st);
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1 + shift;
        let eta = 0.2 + 0.5 * x1 + u1[c] * x1;
        let p = 1.0 / (1.0 + (-eta).exp());
        y[i] = if lcg(&mut st) + 0.5 < p { 1.0 } else { 0.0 };
    }
    (x, y, ids)
}

/// Poisson counts near `e^c` with a cluster-level covariate `z` (`X = [1, z]`),
/// 12 clusters `g` crossed with 5 levels `h`, and every layout that fits them
/// with its `(θ, β)`: blocked `(1|g)`, and `(1|g) + (1|h)` on the structured
/// and the packed layouts. Both columns of `X` lie almost in the span of `ZΛ`,
/// and the crossed intercepts are confounded with the `g` intercepts, so `A`
/// has an eigenvalue of exactly 1 next to ones of order `θ²·Σwᵢ` (1e11 at
/// `c = 24`, 1e15 at `c = 33`).
#[allow(clippy::type_complexity)]
fn huge_count_cells(
    c: f64,
) -> (
    Mat<f64>,
    Vec<f64>,
    Vec<u32>,
    Vec<Vec<u32>>,
    [(&'static str, GlmmWorkspace, GlmmLayout, bool, [f64; 4]); 3],
) {
    let (n, n_g, n_h, p) = (120usize, 12usize, 5usize, 2usize);
    let mut st = 9u64;
    let b: Vec<f64> = (0..n_g).map(|_| lcg(&mut st)).collect();
    let z: Vec<f64> = (0..n_g).map(|_| lcg(&mut st)).collect();
    let d: Vec<f64> = (0..n_h).map(|_| 0.3 * lcg(&mut st)).collect();
    let (mut g, mut h) = (vec![0u32; n], vec![0u32; n]);
    let mut x = Mat::<f64>::zeros(n, p);
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let k = i % n_g;
        g[i] = k as u32;
        h[i] = ((i / n_g) % n_h) as u32;
        x[(i, 0)] = 1.0;
        x[(i, 1)] = z[k];
        y[i] = (c + 0.5 * z[k] + b[k] + d[h[i] as usize]).exp().round();
    }
    let spec = |crossed: bool| SizedSpec {
        spec: ModelSpec {
            family: Family::Poisson {
                link: PoissonLink::Log,
            },
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: if crossed {
                    vec![Grouping {
                        terms: Vec::new(),
                        relation: GroupingRelation::Crossed,
                        slopes: vec![],
                    }]
                } else {
                    vec![]
                },
            }),
        },
        n_primary: n_g,
        extra_counts: if crossed { vec![n_h] } else { vec![] },
    };
    let extra = vec![h];
    let blocked = GlmmWorkspace::for_cluster_spec(p, &spec(false), n, &[], 1);
    let mut structured = GlmmWorkspace::for_cluster_spec(p, &spec(true), n, &[], 1);
    structured.pattern.structured_schur =
        StructuredSchur::new(&structured.groupings, &g, &extra, n);
    let mut packed = GlmmWorkspace::for_cluster_spec_packed(p, &spec(true), n, &[], 1);
    fill_packed_cols(&mut packed, &g, &extra, n);
    // Blocked params are `[θ_g, β₀, β₁]`, the fourth slot unread.
    let cells = [
        (
            "blocked",
            blocked,
            GlmmLayout::Blocked,
            false,
            [1.0, c, 0.5, 0.0],
        ),
        (
            "structured",
            structured,
            GlmmLayout::Structured,
            true,
            [1.0, 0.3, c, 0.5],
        ),
        (
            "packed",
            packed,
            GlmmLayout::Packed,
            true,
            [1.0, 0.3, c, 0.5],
        ),
    ];
    (x, y, g, extra, cells)
}

/// The Rx β information keeps its digits where it is a small remainder of
/// `X'W̃X`: [`huge_count_cells`] at `c = 24`, `θ²·Σwᵢ` of order 1e11, where
/// `S = X'W̃X − X'W̃M·A⁻¹M'W̃X` is a remainder of that order. Evaluated at a
/// fixed `(θ, β)`, so no search is involved, on every Schur fill.
/// Reference: the same covariance with the modes and `S` in 60-digit mpmath
/// from this fixture's `y`. Measured: 2e-16 blocked, 4e-11 structured, 3e-10
/// packed; forming `S` by the subtraction is 4.4e-5 to 4.7e-5 off on all
/// three. On the crossed layouts the remaining error is the inner solve's:
/// `A` itself has condition about `θ²·Σwᵢ` there, which `S` feels only to
/// second order.
#[test]
fn rx_cov_keeps_its_digits_for_a_cluster_level_covariate_at_huge_counts() {
    // mpmath 1.3.0, dps 60, at the f64 values of the (θ, β) below.
    const BLOCKED: [f64; 2] = [0.291_336_336_505_089_3, 0.903_514_010_119_972_8];
    const CROSSED: [f64; 2] = [0.320_744_229_828_393_9, 0.903_514_010_119_972_8];
    let p = 2usize;
    let (x, y, g, extra, cells) = huge_count_cells(24.0);
    let n = y.len();
    for (label, mut ws, layout, crossed, params) in cells {
        assert_eq!(ws.layout, layout, "{label}");
        let (extra_ids, params, want) = if crossed {
            (&extra[..], &params[..], CROSSED)
        } else {
            (&[][..], &params[..3], BLOCKED)
        };
        let dev = glmm_laplace_deviance(params, &mut ws, x.as_ref(), &y, &g, extra_ids, n);
        assert!(dev.is_finite(), "{label}: deviance {dev}");
        let mut cov = Mat::<f64>::zeros(p, p);
        let ok = rx_cov_into(
            &mut ws,
            x.as_ref(),
            x.as_ref().subcols(0, 0),
            &g,
            p,
            n,
            &mut cov,
        );
        assert!(ok, "{label}: the Rx Schur must stay positive definite");
        for j in 0..p {
            let se = cov[(j, j)].sqrt();
            let rel = (se / want[j] - 1.0).abs();
            assert!(
                rel < 1e-8,
                "{label}: SE[{j}] {se} vs {} (rel {rel:.2e})",
                want[j]
            );
        }
    }
}

/// PIRLS from `û = 0` converges on the crossed layouts at `θ²·Σwᵢ ≈ 1e15`
/// ([`huge_count_cells`] at `c = 33`): with the u-step taken as
/// `u_new = A⁻¹((A − I)·u + M'ρ)`, the rounding of `(A − I)·u` moved `‖u‖²` by
/// 3e-3 between iterates along `A`'s unit eigenvector, the solve ran out its
/// halvings and the deviance came back `inf`. What is left is `log|A|`'s own
/// rounding on that eigenvalue: the two layouts measured 560.7352 and 560.6791.
#[test]
fn crossed_pirls_converges_at_huge_counts() {
    let (x, y, g, extra, cells) = huge_count_cells(33.0);
    let n = y.len();
    let mut devs = Vec::new();
    for (label, mut ws, layout, crossed, params) in cells {
        if !crossed {
            continue;
        }
        assert_eq!(ws.layout, layout, "{label}");
        let dev = glmm_laplace_deviance(&params, &mut ws, x.as_ref(), &y, &g, &extra, n);
        assert!(dev.is_finite(), "{label}: deviance {dev}");
        devs.push(dev);
    }
    assert!(
        (devs[0] - devs[1]).abs() < 0.1,
        "structured {} vs packed {}",
        devs[0],
        devs[1]
    );
}

// --- row-blocked assembled Hessian (`assembled::run_assembled_hessian_blocked`) ---

/// One assembled Hessian at the workspace's current `params`, from a cold
/// mode, with the row-block size forced to `block`. Returns
/// `(objective, gradient, Hessian columns)`.
fn assembled_hessian_at_block(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    ids: &[u32],
    p: usize,
    n: usize,
    block: usize,
) -> (f64, Vec<f64>, Mat<f64>) {
    let m = ws.n_theta + p;
    let kk = ws.k.max(1);
    ws.fd.se_row_block = block;
    for v in ws.pirls.u[..kk].iter_mut() {
        *v = 0.0;
    }
    let mut hess = Mat::<f64>::zeros(m, m);
    let mut grad = vec![0.0; m];
    let st =
        super::assembled::joint_hessian_columns(ws, x, y, ids, &[], p, n, &mut grad, &mut hess);
    ws.fd.se_row_block = super::assembled::SE_ROW_BLOCK;
    let DerivStatus::Ok(v) = st else {
        panic!("the assembled Hessian declined at block {block}")
    };
    (v, grad, hess)
}

/// The row-blocked pass returns the whole-data pass's objective, gradient and
/// Hessian to round-off, on every family the assembled Hessian serves on the
/// blocked layout (canonical and observed-information links, scalar and
/// slope blocks), with prior weights and an offset, and at block sizes from
/// one that leaves a single cluster per block to one that leaves two blocks.
/// The difference is per-block PIRLS exit tests and per-block summation, so
/// the band is relative 1e-10, against movements of order 1e-14 measured.
#[test]
fn row_blocked_hessian_matches_whole_data_pass() {
    let families = [
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        Family::Poisson {
            link: PoissonLink::Log,
        },
        Family::Gamma {
            link: crate::GammaLink::Log,
        },
    ];
    for family in families {
        for shape in ["int1", "q2s"] {
            for weighted in [false, true] {
                let (mut ws, x, y, ids, p, n) = fixture_with_nagq_sized(family, shape, 1, 600, 40);
                assert_eq!(ws.layout, GlmmLayout::Blocked, "{family:?}/{shape}");
                assert!(ws.groupings.extra_offsets.is_empty());
                if weighted {
                    for (i, w) in ws.prior_w[..n].iter_mut().enumerate() {
                        *w = 0.5 + (i % 5) as f64 * 0.25;
                    }
                    ws.weighted = true;
                    ws.offset = Some((0..n).map(|i| 0.05 * ((i % 7) as f64 - 3.0)).collect());
                }
                let m = ws.n_theta + p;
                let mut rng = fixed_seed_theta(shape);
                ws.params[..m].copy_from_slice(&rng.next_params());
                let (v0, g0, h0) =
                    assembled_hessian_at_block(&mut ws, x.as_ref(), &y, &ids, p, n, usize::MAX);
                // `n` itself takes the whole-data pass: bit-identical.
                let (vn, gn, hn) =
                    assembled_hessian_at_block(&mut ws, x.as_ref(), &y, &ids, p, n, n);
                assert_eq!(v0.to_bits(), vn.to_bits(), "{family:?}/{shape}: block = n");
                for r in 0..m {
                    assert_eq!(g0[r].to_bits(), gn[r].to_bits());
                    for c in 0..m {
                        assert_eq!(h0[(r, c)].to_bits(), hn[(r, c)].to_bits());
                    }
                }
                for block in [1, 16, 64, n / 2, n - 1] {
                    let (v, g, h) =
                        assembled_hessian_at_block(&mut ws, x.as_ref(), &y, &ids, p, n, block);
                    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);
                    assert!(
                        rel(v, v0) <= 1e-10,
                        "{family:?}/{shape} w={weighted} block {block}: objective {v} vs {v0}"
                    );
                    for r in 0..m {
                        assert!(
                            rel(g[r], g0[r]) <= 1e-10,
                            "{family:?}/{shape} w={weighted} block {block}: grad[{r}] {} vs {}",
                            g[r],
                            g0[r]
                        );
                        for c in 0..m {
                            assert!(
                                rel(h[(r, c)], h0[(r, c)]) <= 1e-10,
                                "{family:?}/{shape} w={weighted} block {block}: \
                                 H[{r},{c}] {} vs {}",
                                h[(r, c)],
                                h0[(r, c)]
                            );
                        }
                    }
                }
            }
        }
    }
}

/// A repeat row-blocked Hessian call at the same shape and data allocates
/// exactly what a repeat whole-data call does: the block scratch, the block
/// list and the gathered rows are all built once with the dual scratch.
/// Measured against the whole-data pass rather than against zero because the
/// call around both passes (the `f64` mode solve) is not itself gated here.
/// Run: `RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests row_blocked_hessian_repeat_calls_allocate_no_more -- --ignored --test-threads=1`
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn row_blocked_hessian_repeat_calls_allocate_no_more() {
    let _serial = crate::test_support::alloc_test_guard();
    let mut blocks = [0u64; 2];
    for (arm, block) in [usize::MAX, 32].into_iter().enumerate() {
        let (mut ws, x, y, ids, p, n) = fixture_with_nagq_sized(
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            "q2s",
            1,
            600,
            40,
        );
        let m = ws.n_theta + p;
        let mut rng = fixed_seed_theta("q2s");
        ws.params[..m].copy_from_slice(&rng.next_params());
        ws.fd.se_row_block = block;
        let mut hess = Mat::<f64>::zeros(m, m);
        let mut grad = vec![0.0; m];
        let _ = super::assembled::joint_hessian_columns(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &[],
            p,
            n,
            &mut grad,
            &mut hess,
        );
        let prof = dhat::Profiler::builder().testing().build();
        for _ in 0..3 {
            let _ = super::assembled::joint_hessian_columns(
                &mut ws,
                x.as_ref(),
                &y,
                &ids,
                &[],
                p,
                n,
                &mut grad,
                &mut hess,
            );
        }
        blocks[arm] = dhat::HeapStats::get().total_blocks;
        drop(prof);
    }
    assert_eq!(
        blocks[1], blocks[0],
        "row-blocked repeat Hessian allocated {} blocks, whole-data {}",
        blocks[1], blocks[0]
    );
}

// --- streamed structured assembled Hessian (`pirls::structured_stream_deviance`) ---

/// [`assembled_hessian_at_block`] on the structured layout: the same call with
/// the extra-grouping ids passed through.
#[allow(clippy::too_many_arguments)]
fn structured_hessian_at_block(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    block: usize,
) -> (f64, Vec<f64>, Mat<f64>) {
    let m = ws.n_theta + p;
    let kk = ws.k.max(1);
    ws.fd.se_row_block = block;
    for v in ws.pirls.u[..kk].iter_mut() {
        *v = 0.0;
    }
    let mut hess = Mat::<f64>::zeros(m, m);
    let mut grad = vec![0.0; m];
    let st = super::assembled::joint_hessian_columns(
        ws, x, y, ids, extra_ids, p, n, &mut grad, &mut hess,
    );
    ws.fd.se_row_block = super::assembled::SE_ROW_BLOCK;
    let DerivStatus::Ok(v) = st else {
        panic!("the assembled Hessian declined at block {block}")
    };
    (v, grad, hess)
}

/// The streamed pass on the structured layout (crossed, nested and crossed +
/// nested; canonical and observed-information links; with prior weights and
/// an offset) returns the whole-data pass's gradient and Hessian bit for bit,
/// at block sizes from a single row to `n − 1`: every per-row value is the
/// same arithmetic and every accumulator still takes its rows in ascending
/// order. Only the PIRLS deviance is summed per block, and it enters the
/// objective, so the objective is held to relative 1e-12 rather than to bits.
/// The scratch the streamed pass builds keeps no `n`-length dual row buffer.
#[test]
fn streamed_structured_hessian_matches_whole_data_pass() {
    let families = [
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        Family::Poisson {
            link: PoissonLink::Log,
        },
        Family::Gamma {
            link: crate::GammaLink::Log,
        },
    ];
    for family in families {
        for (np, ncr, shape) in [
            (0usize, 6usize, "crossed6"),
            (2, 0, "nested2"),
            (2, 6, "nested2_crossed6"),
        ] {
            for weighted in [false, true] {
                let (mut ws, x, y, ids, extra_ids, p, n) = extras_fixture(family, np, ncr);
                assert_eq!(ws.layout, GlmmLayout::Structured, "{family:?}/{shape}");
                if weighted {
                    for (i, w) in ws.prior_w[..n].iter_mut().enumerate() {
                        *w = 0.5 + (i % 5) as f64 * 0.25;
                    }
                    ws.weighted = true;
                    ws.offset = Some((0..n).map(|i| 0.05 * ((i % 7) as f64 - 3.0)).collect());
                }
                let m = ws.n_theta + p;
                let mut rng = fixed_seed_theta(shape);
                ws.params[..m].copy_from_slice(&rng.next_params());
                let (v0, g0, h0) = structured_hessian_at_block(
                    &mut ws,
                    x.as_ref(),
                    &y,
                    &ids,
                    &extra_ids,
                    p,
                    n,
                    usize::MAX,
                );
                for block in [1, 7, 64, n - 1] {
                    let (v, g, h) = structured_hessian_at_block(
                        &mut ws,
                        x.as_ref(),
                        &y,
                        &ids,
                        &extra_ids,
                        p,
                        n,
                        block,
                    );
                    let label = format!("{family:?}/{shape} w={weighted} block {block}");
                    let rows_kept = match ws.dual_scratch.as_deref() {
                        Some(GlmmDualScratch::D4(b, ..)) => b.pirls.eta.len(),
                        Some(GlmmDualScratch::D5(b, ..)) => b.pirls.eta.len(),
                        Some(GlmmDualScratch::D6(b, ..)) => b.pirls.eta.len(),
                        Some(GlmmDualScratch::D8(b, ..)) => b.pirls.eta.len(),
                        Some(GlmmDualScratch::D12(b, ..)) => b.pirls.eta.len(),
                        _ => panic!("{label}: expected a Dual scratch"),
                    };
                    assert_eq!(rows_kept, 0, "{label}: dual row buffers not streamed");
                    assert!(
                        (v - v0).abs() <= 1e-12 * v0.abs().max(1.0),
                        "{label}: objective {v} vs {v0}"
                    );
                    for r in 0..m {
                        assert_eq!(g[r].to_bits(), g0[r].to_bits(), "{label}: grad[{r}]");
                        for c in 0..m {
                            assert_eq!(
                                h[(r, c)].to_bits(),
                                h0[(r, c)].to_bits(),
                                "{label}: H[{r},{c}] {} vs {}",
                                h[(r, c)],
                                h0[(r, c)]
                            );
                        }
                    }
                }
            }
        }
    }
}

/// A repeat streamed Hessian call at the same shape and data allocates exactly
/// what a repeat whole-data call does: the block buffers are built once with
/// the dual scratch. Run: `RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests streamed_structured_hessian_repeat_calls_allocate_no_more -- --ignored --test-threads=1`
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn streamed_structured_hessian_repeat_calls_allocate_no_more() {
    let _serial = crate::test_support::alloc_test_guard();
    let mut blocks = [0u64; 2];
    for (arm, block) in [usize::MAX, 32].into_iter().enumerate() {
        let (mut ws, x, y, ids, extra_ids, p, n) = extras_fixture(
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            0,
            6,
        );
        let m = ws.n_theta + p;
        let mut rng = fixed_seed_theta("crossed6");
        ws.params[..m].copy_from_slice(&rng.next_params());
        ws.fd.se_row_block = block;
        let mut hess = Mat::<f64>::zeros(m, m);
        let mut grad = vec![0.0; m];
        let _ = super::assembled::joint_hessian_columns(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            &mut grad,
            &mut hess,
        );
        let prof = dhat::Profiler::builder().testing().build();
        for _ in 0..3 {
            let _ = super::assembled::joint_hessian_columns(
                &mut ws,
                x.as_ref(),
                &y,
                &ids,
                &extra_ids,
                p,
                n,
                &mut grad,
                &mut hess,
            );
        }
        blocks[arm] = dhat::HeapStats::get().total_blocks;
        drop(prof);
    }
    assert_eq!(
        blocks[1], blocks[0],
        "streamed repeat Hessian allocated {} blocks, whole-data {}",
        blocks[1], blocks[0]
    );
}

/// The structured dual twins size their coupling-shaped buffers from the
/// design's own coupling pattern once one is known (`dual_coupling_pairs`),
/// not from the `coupling_len` bound, and nothing they compute moves: the
/// gradient and the assembled Hessian from a scratch sized on the pattern are
/// bit-identical to those from a scratch sized at the bound.
#[test]
fn structured_dual_twins_size_coupling_from_the_pattern() {
    for family in [
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        Family::Poisson {
            link: PoissonLink::Log,
        },
    ] {
        let (mut ws, x, y, ids, extra_ids, p, n) = extras_fixture(family, 0, 6);
        let m = ws.n_theta + p;
        let mut rng = fixed_seed_theta("crossed6");
        ws.params[..m].copy_from_slice(&rng.next_params());
        let s = ws.groupings.n_primary;
        let q_core = ws.groupings.primary_q + ws.groupings.nested_per_parent;
        let e = ws.groupings.k_crossed();
        let kk = ws.k.max(1);
        let coupling_of = |ws: &GlmmWorkspace| match ws.dual_scratch.as_deref() {
            Some(GlmmDualScratch::D4(b, ..)) => (b.structured.coupling.len(), b.asm.tail_h.len()),
            Some(GlmmDualScratch::D5(b, ..)) => (b.structured.coupling.len(), b.asm.tail_h.len()),
            Some(GlmmDualScratch::D6(b, ..)) => (b.structured.coupling.len(), b.asm.tail_h.len()),
            Some(GlmmDualScratch::D8(b, ..)) => (b.structured.coupling.len(), b.asm.tail_h.len()),
            Some(GlmmDualScratch::D12(b, ..)) => (b.structured.coupling.len(), b.asm.tail_h.len()),
            _ => panic!("expected a Dual scratch"),
        };
        let run = |ws: &mut GlmmWorkspace| {
            for v in ws.pirls.u[..kk].iter_mut() {
                *v = 0.0;
            }
            let mut grad = vec![0.0; m];
            let DerivStatus::Ok(_) =
                laplace_gradient(ws, x.as_ref(), &y, &ids, &extra_ids, p, n, &mut grad)
            else {
                panic!("{family:?}: dual gradient declined")
            };
            for v in ws.pirls.u[..kk].iter_mut() {
                *v = 0.0;
            }
            let mut hess = Mat::<f64>::zeros(m, m);
            let mut hgrad = vec![0.0; m];
            let st = super::assembled::joint_hessian_columns(
                ws,
                x.as_ref(),
                &y,
                &ids,
                &extra_ids,
                p,
                n,
                &mut hgrad,
                &mut hess,
            );
            let DerivStatus::Ok(_) = st else {
                panic!("{family:?}: assembled Hessian declined")
            };
            (grad, hess)
        };
        // A fresh fit's pattern is unknown (`coup_mask == None`): bound.
        ws.pattern.coup_mask = None;
        ws.dual_scratch = None;
        let (g_bound, h_bound) = run(&mut ws);
        // The dual calls just built the full pattern (mask 0); a rebuild now
        // sizes on it.
        assert_eq!(ws.pattern.coup_mask, Some(0));
        let pairs = ws.pattern.coup_ptr[s] as usize;
        let bound = crate::glmm::workspace::coupling_len(n, s, q_core, e);
        ws.dual_scratch = None;
        let (g_pat, h_pat) = run(&mut ws);
        let (coup, tail) = coupling_of(&ws);
        assert_eq!(coup, (q_core * pairs).max(1), "{family:?}: twin coupling");
        assert_eq!(tail, (q_core * pairs).max(1), "{family:?}: tail_h");
        assert!(
            coup < bound,
            "{family:?}: pattern {coup} not below bound {bound}"
        );
        for r in 0..m {
            assert_eq!(
                g_bound[r].to_bits(),
                g_pat[r].to_bits(),
                "{family:?} grad[{r}]"
            );
            for c in 0..m {
                assert_eq!(
                    h_bound[(r, c)].to_bits(),
                    h_pat[(r, c)].to_bits(),
                    "{family:?} H[{r},{c}]"
                );
            }
        }
    }
}

// --- AGQ f64 gradient and the gradient-difference Hessian (rung 2a) ---

/// The families `agq_gradient::covers` accepts, each at a scalar intercept
/// and `nagq = 7`.
const AGQ_GD_FAMILIES: &[Family] = &[
    Family::Binomial {
        link: BinomialLink::Logit,
    },
    Family::Binomial {
        link: BinomialLink::Probit,
    },
    Family::Poisson {
        link: PoissonLink::Log,
    },
];

/// The f64 AGQ gradient against `laplace_gradient`'s dual AGQ branch, the
/// in-tree oracle: both differentiate the same AGQ deviance once, exactly, at
/// the same `ws.params` and a tightened PIRLS tolerance, so they agree to
/// round-off (band 1e-9 relative). Clusters of 2, 3 and 5 rows (where AGQ
/// matters most and the node terms are least Gaussian) and the default
/// fixture; draws include `λ = 0`, where `dD/dλ` is 0 by symmetry.
#[test]
fn agq_f64_gradient_matches_dual_gradient() {
    for &family in AGQ_GD_FAMILIES {
        for (n, nc) in [
            (60usize, 30usize),
            (60, 20),
            (60, 12),
            default_fixture_size("int1"),
        ] {
            let (mut ws, x, y, ids, p, n) = fixture_with_nagq_sized(family, "int1", 7, n, nc);
            assert!(super::agq_gradient::covers(&ws), "{family:?}: not covered");
            ws.fd.pirls_tol_override = Some(1e-12);
            let m = ws.n_theta + p;
            let kk = ws.k.max(1);
            let mut gd = super::agq_gradient::AgqGd::for_shape(p, ws.nagq);
            gd.idx.rebuild(&ids[..n], ws.groupings.n_primary);
            let mut rng = fixed_seed_theta("int1");
            let mut ctrs = EvalCounters::new();
            for draw in 0..6 {
                let mut prm = rng.next_params();
                if draw == 0 {
                    prm[0] = 0.0;
                }
                ws.params[..m].copy_from_slice(&prm);
                let mut g_dual = vec![0.0; m];
                ws.pirls.u[..kk].fill(0.0);
                let st = laplace_gradient(&mut ws, x.as_ref(), &y, &ids, &[], p, n, &mut g_dual);
                assert!(
                    matches!(st, DerivStatus::Ok(_)),
                    "{family:?}: dual gradient"
                );
                ws.params[..m].copy_from_slice(&prm);
                ws.beta_rhs[..p].copy_from_slice(&prm[ws.n_theta..m]);
                ws.pirls.u[..kk].fill(0.0);
                let f = laplace_deviance_ws(
                    &mut ws,
                    x.as_ref(),
                    &y,
                    &ids,
                    &[],
                    n,
                    BetaMode::Fixed,
                    &mut ctrs,
                );
                assert!(f.is_finite());
                let mut g = vec![0.0; m];
                assert!(super::agq_gradient::assemble(
                    ws.family,
                    ws.nagq,
                    ws.groupings.n_primary,
                    &ws.pirls,
                    &ws.prior_w,
                    x.as_ref(),
                    &y,
                    &gd.idx,
                    &prm,
                    &mut gd.sc,
                    &mut g
                ));
                for k in 0..m {
                    assert!(
                        (g[k] - g_dual[k]).abs() <= 1e-9 * g_dual[k].abs().max(1.0),
                        "{family:?} n={n} nc={nc} draw {draw} coord {k}: f64 {} vs dual {}",
                        g[k],
                        g_dual[k]
                    );
                }
                if draw == 0 {
                    assert!(g[0].abs() <= 1e-9, "{family:?}: dD/dλ at λ = 0 is {}", g[0]);
                }
            }
        }
    }
}

/// Fits `fixture_with_nagq_sized`'s data at `nagq = 7` with `WaldSe::Rx`
/// (the explicit `joint_hessian_cov` calls are the comparison), padded by
/// `extra_p` zero-truth columns when nonzero.
fn agq_gd_fit(
    family: Family,
    n: usize,
    nc: usize,
    extra_p: usize,
) -> (GlmmWorkspace, Mat<f64>, Vec<f64>, Vec<u32>, usize, usize) {
    let (_ws0, x0, y, ids, p0, n) = fixture_with_nagq_sized(family, "int1", 7, n, nc);
    let p = p0 + extra_p;
    let mut x = Mat::<f64>::zeros(n, p);
    let mut st = 7777u64;
    for i in 0..n {
        for c in 0..p0 {
            x[(i, c)] = x0[(i, c)];
        }
        for c in p0..p {
            x[(i, c)] = 0.3 * lcg(&mut st);
        }
    }
    let model = SizedSpec {
        spec: ModelSpec {
            family,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![],
            }),
        },
        n_primary: nc,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 7);
    let fit = fit_glmm(
        &mut ws,
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        &[1u32],
        None,
        &vec![0.0; p],
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    );
    assert!(fit.converged, "{family:?}: fit did not converge");
    (ws, x, y, ids, p, n)
}

/// Rung 2a against rung 2 on the same fit: the gradient-difference Hessian
/// and the hyper-dual pass differentiate the same AGQ deviance at the same
/// point, one exactly and one by a central difference of an exact gradient at
/// a 1e-5 relative step, so the β and θ SEs agree to well inside 1e-7. Also
/// pins that rung 2a, not rung 2, takes a covered shape (`ws.agq_gd` built, no
/// hyper-dual scratch), and that a second call on the same workspace returns
/// the same covariance bit for bit.
#[test]
fn agq_gradient_difference_hessian_matches_hyper_dual() {
    for &family in AGQ_GD_FAMILIES {
        for (n, nc) in [(120usize, 40usize), (200, 20)] {
            let (mut ws, x, y, ids, p, n) = agq_gd_fit(family, n, nc, 0);
            ws.hyper_scratch = None;
            ws.agq_gd = None;
            let mut cov_gd = Mat::<f64>::zeros(p, p);
            let st = joint_hessian_cov(
                &mut ws,
                x.as_ref(),
                x.as_ref().subcols(0, 0),
                &y,
                &ids,
                &[],
                p,
                n,
                &mut cov_gd,
            );
            assert_eq!(st, FdHessianStatus::Ok, "{family:?}");
            assert!(ws.agq_gd.is_some(), "{family:?}: rung 2a did not run");
            assert!(ws.hyper_scratch.is_none(), "{family:?}: rung 2 ran");
            let tse_gd = theta_se(&ws);

            let mut cov_again = Mat::<f64>::zeros(p, p);
            joint_hessian_cov(
                &mut ws,
                x.as_ref(),
                x.as_ref().subcols(0, 0),
                &y,
                &ids,
                &[],
                p,
                n,
                &mut cov_again,
            );
            for a in 0..p {
                for b in 0..p {
                    assert_eq!(
                        cov_gd[(a, b)].to_bits(),
                        cov_again[(a, b)].to_bits(),
                        "{family:?}: second call moved cov[{a},{b}]"
                    );
                }
            }

            ws.fd.skip_agq_gradient_difference = true;
            let mut cov_hd = Mat::<f64>::zeros(p, p);
            let st = joint_hessian_cov(
                &mut ws,
                x.as_ref(),
                x.as_ref().subcols(0, 0),
                &y,
                &ids,
                &[],
                p,
                n,
                &mut cov_hd,
            );
            ws.fd.skip_agq_gradient_difference = false;
            assert_eq!(st, FdHessianStatus::Ok, "{family:?}");
            assert!(ws.hyper_scratch.is_some(), "{family:?}: rung 2 did not run");
            let tse_hd = theta_se(&ws);
            for j in 0..p {
                let (a, b) = (cov_gd[(j, j)].sqrt(), cov_hd[(j, j)].sqrt());
                assert!(
                    (a - b).abs() <= 1e-7 * b,
                    "{family:?} n={n}: se[{j}] gd {a} vs hyper-dual {b}"
                );
            }
            for k in 0..ws.n_theta {
                assert!(
                    (tse_gd[k] - tse_hd[k]).abs() <= 1e-7 * tse_hd[k],
                    "{family:?} n={n}: theta_se[{k}] gd {} vs hyper-dual {}",
                    tse_gd[k],
                    tse_hd[k]
                );
            }
        }
    }
}

/// Past the hyper-dual lane cap (`m = 15 > MAX_DUAL_N`), where rung 2 refuses
/// and the FD stencil used to take every AGQ fit: rung 2a takes the covered
/// shapes, and its SEs agree with the stencil's within the stencil's own band
/// (`AGQ_SE_HESSIAN_REL`, 2e-2; the stencil is the less accurate of the two).
#[test]
fn agq_gradient_difference_hessian_covers_past_the_lane_cap() {
    for &family in AGQ_GD_FAMILIES {
        let (mut ws, x, y, ids, p, n) = agq_gd_fit(family, 200, 40, 12);
        assert!(ws.n_theta + p > crate::glmm::derivative::MAX_DUAL_N);
        ws.agq_gd = None;
        let mut cov_gd = Mat::<f64>::zeros(p, p);
        let st = joint_hessian_cov(
            &mut ws,
            x.as_ref(),
            x.as_ref().subcols(0, 0),
            &y,
            &ids,
            &[],
            p,
            n,
            &mut cov_gd,
        );
        assert_eq!(st, FdHessianStatus::Ok, "{family:?}");
        assert!(ws.agq_gd.is_some(), "{family:?}: rung 2a did not run");
        let tse_gd = theta_se(&ws);
        ws.fd.force_fd_hessian = true;
        let mut cov_fd = Mat::<f64>::zeros(p, p);
        let st = joint_hessian_cov(
            &mut ws,
            x.as_ref(),
            x.as_ref().subcols(0, 0),
            &y,
            &ids,
            &[],
            p,
            n,
            &mut cov_fd,
        );
        ws.fd.force_fd_hessian = false;
        assert_eq!(st, FdHessianStatus::Ok, "{family:?}");
        let tse_fd = theta_se(&ws);
        for j in 0..p {
            let (a, b) = (cov_gd[(j, j)].sqrt(), cov_fd[(j, j)].sqrt());
            assert!(
                (a - b).abs() <= 2e-2 * b,
                "{family:?}: se[{j}] gd {a} vs fd {b}"
            );
        }
        for k in 0..ws.n_theta {
            assert!(
                (tse_gd[k] - tse_fd[k]).abs() <= 2e-2 * tse_fd[k],
                "{family:?}: theta_se[{k}] gd {} vs fd {}",
                tse_gd[k],
                tse_fd[k]
            );
        }
    }
}

// --- The at-mode gradient (one mode solve per point) ---

/// The routed families with no dispersion coordinate (binomial, any link;
/// Poisson). The routed Gamma cells run through `GAMMA_GATE_SHAPES` at their
/// own φ instead.
fn routed_without_dispersion(f: Family) -> bool {
    matches!(f, Family::Binomial { .. } | Family::Poisson { .. })
}

/// The Gamma cells of the at-mode gradient gates, `(shape, nested children
/// per parent, crossed levels)`: the blocked `q2s` shape and the three
/// structured ones, each on both links.
const GAMMA_GATE_SHAPES: [(&str, usize, usize); 4] = [
    ("q2s", 0, 0),
    ("nested2", 2, 0),
    ("crossed6", 0, 6),
    ("nested2_crossed6", 2, 6),
];
const GAMMA_GATE_LINKS: [GammaLink; 2] = [GammaLink::Log, GammaLink::Inverse];

/// A Gamma gate cell: the blocked `fixture` or the structured
/// `extras_fixture`, with prior weights and an offset when `weighted`, and its
/// θ draws. On the inverse link β is `[0.8, −0.3]`, so η stays positive on
/// the fixtures' `x₁ ∈ [−0.5, 0.5]`.
#[allow(clippy::type_complexity)]
fn gamma_gate_cell(
    link: GammaLink,
    shape: &str,
    np: usize,
    n_crossed: usize,
    weighted: bool,
) -> (
    GlmmWorkspace,
    Mat<f64>,
    Vec<f64>,
    Vec<u32>,
    Vec<Vec<u32>>,
    usize,
    usize,
    FixedSeedTheta,
) {
    let family = Family::Gamma { link };
    let (mut ws, x, y, ids, extra_ids, p, n) = if np == 0 && n_crossed == 0 {
        let (ws, x, y, ids, p, n) = fixture(family, shape);
        (ws, x, y, ids, Vec::new(), p, n)
    } else {
        extras_fixture(family, np, n_crossed)
    };
    if weighted {
        for (i, w) in ws.prior_w[..n].iter_mut().enumerate() {
            *w = 0.5 + (i % 5) as f64 * 0.25;
        }
        ws.weighted = true;
        ws.offset = Some((0..n).map(|i| 0.05 * ((i % 7) as f64 - 3.0)).collect());
    }
    let mut rng = fixed_seed_theta(shape);
    if link == GammaLink::Inverse {
        rng.beta = vec![0.8, -0.3];
    }
    (ws, x, y, ids, extra_ids, p, n, rng)
}

/// `gradient_at_mode` after a fixed-β value evaluation at the shipped
/// PIRLS tolerance against `gradient_f64` (which re-solves the mode at the
/// derivative tolerance) at the same point, on every routed blocked and
/// structured gate cell, crossed and nested tails, canonical and
/// non-canonical links, and on the Gamma cells (both links, with and without
/// weights and an offset) at the same fixed φ. The two differ only by where
/// the mode solve stopped, so the band is that tolerance's effect on the
/// gradient: measured at most 2.6e-10 relative on 94 grid cells
/// (2026-10-07); 1e-8 here.
#[test]
fn gradient_at_mode_matches_the_re_solve_gradient() {
    const BAND: f64 = 1e-8;
    let mut worst_all = 0.0f64;
    let mut sparse_cells = 0usize;
    let mut run = |ws: &mut GlmmWorkspace,
                   x: MatRef<f64>,
                   y: &[f64],
                   ids: &[u32],
                   extra_ids: &[Vec<u32>],
                   p: usize,
                   n: usize,
                   label: &str,
                   mut rng: FixedSeedTheta| {
        let m = ws.n_theta + p;
        let k = ws.k.max(1);
        for _ in 0..6 {
            ws.params[..m].copy_from_slice(&rng.next_params());
            ws.pirls.u[..k].fill(0.0);
            let mut c = crate::counters::EvalCounters::new();
            let dev = super::deviance::laplace_deviance_ws(
                ws,
                x,
                y,
                ids,
                extra_ids,
                n,
                BetaMode::Fixed,
                &mut c,
            );
            assert!(dev.is_finite(), "{label}: value not finite");
            let mut got = vec![0.0; m];
            super::assembled::gradient_at_mode_ws(ws, x, y, ids, extra_ids, p, n, &mut got)
                .unwrap_or_else(|| panic!("{label}: at-mode gradient declined"));
            let mut want = vec![0.0; m];
            super::assembled::gradient_f64(ws, x, y, ids, extra_ids, p, n, &mut want)
                .unwrap_or_else(|| panic!("{label}: re-solve gradient declined"));
            for j in 0..m {
                let gap = (got[j] - want[j]).abs() / want[j].abs().max(1.0);
                worst_all = worst_all.max(gap);
                assert!(
                    gap <= BAND,
                    "{label} coord {j}: at-mode {} vs re-solve {} (gap {gap:e})",
                    got[j],
                    want[j]
                );
            }
        }
    };
    for &(family, shape) in GRADIENT_GATE_CELLS {
        if !routed_without_dispersion(family) {
            continue;
        }
        let (mut ws, x, y, ids, p, n) = fixture(family, shape);
        let label = format!("{family:?}/{shape}");
        run(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &[],
            p,
            n,
            &label,
            fixed_seed_theta(shape),
        );
    }
    for &(family, shape, np, n_crossed) in STRUCTURED_GATE_CELLS {
        if !routed_without_dispersion(family) {
            continue;
        }
        let (mut ws, x, y, ids, extra_ids, p, n) = extras_fixture(family, np, n_crossed);
        let label = format!("{family:?}/{shape}");
        run(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            &label,
            fixed_seed_theta(shape),
        );
        // The production structured path: the value pass on the cached sparse
        // Schur factor (built per fit by `fit::glmm`), the gradient inverting
        // `S` off that factor rather than re-solving on the dense one.
        ws.pattern.structured_schur =
            super::workspace::StructuredSchur::new(&ws.groupings, &ids, &extra_ids, n);
        if ws.pattern.structured_schur.is_some() {
            sparse_cells += 1;
            run(
                &mut ws,
                x.as_ref(),
                &y,
                &ids,
                &extra_ids,
                p,
                n,
                &format!("{label}/sparse"),
                fixed_seed_theta(shape),
            );
        }
    }
    // Gamma at a dispersion φ ≠ 1, where the value pass runs PIRLS on
    // `prior_wᵢ/φ` and `gradient_f64` re-solves at the same φ, then at φ = 1,
    // where the value pass leaves `prior_w_phi` unwritten: it is NaN-filled
    // here in place of an earlier evaluation's weights, and must not be read.
    for link in GAMMA_GATE_LINKS {
        for (shape, np, n_crossed) in GAMMA_GATE_SHAPES {
            for weighted in [false, true] {
                for phi in [0.3, 1.0] {
                    let (mut ws, x, y, ids, extra_ids, p, n, rng) =
                        gamma_gate_cell(link, shape, np, n_crossed, weighted);
                    if phi == 1.0 {
                        ws.prior_w_phi[..n].fill(f64::NAN);
                    }
                    ws.gamma_phi = phi;
                    run(
                        &mut ws,
                        x.as_ref(),
                        &y,
                        &ids,
                        &extra_ids,
                        p,
                        n,
                        &format!("Gamma/{link:?}/{shape} w={weighted} phi={phi}"),
                        rng,
                    );
                }
            }
        }
    }
    assert!(
        sparse_cells > 0,
        "no structured cell on the sparse Schur factor"
    );
    println!("at-mode vs re-solve gradient: worst relative gap {worst_all:e}");
}

/// The at-mode gradient against central differences of the fixed-β
/// value, in the θ regimes the search crosses: a random draw, one diagonal θ
/// near a variance collapse (1e-3), and large θ (diagonals 8). PIRLS runs
/// from û = 0 at 1e-12 for every evaluation, so the differences see the
/// objective, not the solver's stopping point. Band 1e-5 relative, with an
/// absolute floor of 1e-6·(1 + |g|∞), as the existing FD gates use.
///
/// On Gamma (both links, blocked and structured, with and without weights
/// and an offset) the objective is the full `F`, dispersion term included,
/// over `[θ | β | ln φ]`, and the `ln φ` entry is
/// `assembled::gamma_ln_phi_gradient`'s. It is checked at four `ln φ`: the
/// plug-in seed `fit_glmm` starts from, the fitted optimum, and the optimum
/// ± 1.
#[test]
fn gradient_at_mode_matches_finite_differences_across_theta_regimes() {
    // One cell: every θ regime at every entry of `ln_phis`. `None` checks
    // `[θ | β]` at φ = 1; `Some(ψ)` adds the trailing `ln φ` coordinate.
    let check = |ws: &mut GlmmWorkspace,
                 x: MatRef<f64>,
                 y: &[f64],
                 ids: &[u32],
                 extra_ids: &[Vec<u32>],
                 p: usize,
                 n: usize,
                 label: &str,
                 mut rng: FixedSeedTheta,
                 ln_phis: &[Option<f64>]| {
        ws.fd.pirls_tol_override = Some(1e-12);
        let n_theta = ws.n_theta;
        let m = n_theta + p;
        let k = ws.k.max(1);
        let diag: Vec<usize> = ws.groupings.diagonal_theta().to_vec();
        let base = rng.next_params();
        let mut regimes = vec![base.clone()];
        let mut near = base.clone();
        near[diag[0]] = 1e-3;
        regimes.push(near);
        let mut large = base.clone();
        for &i in &diag {
            large[i] = 8.0;
        }
        regimes.push(large);
        let sum_ln_y: f64 = y[..n].iter().map(|v| v.ln()).sum();
        let value = |ws: &mut GlmmWorkspace, pt: &[f64]| -> f64 {
            ws.params[..m].copy_from_slice(&pt[..m]);
            ws.pirls.u[..k].fill(0.0);
            if let Some(&psi) = pt.get(m) {
                ws.gamma_phi = psi.exp();
            }
            let mut c = crate::counters::EvalCounters::new();
            let dev = super::deviance::laplace_deviance_ws(
                ws,
                x,
                y,
                ids,
                extra_ids,
                n,
                BetaMode::Fixed,
                &mut c,
            );
            match pt.get(m) {
                Some(&psi) => {
                    let w = ws.weighted.then_some(&ws.prior_w[..n]);
                    dev + crate::family::gamma_dispersion_term(psi, w, n, sum_ln_y)
                }
                None => dev,
            }
        };
        for ln_phi in ln_phis {
            for (r, regime) in regimes.iter().enumerate() {
                let mut point = regime.clone();
                point.extend(ln_phi);
                let at = format!("{label} ln_phi {ln_phi:?} regime {r}");
                let f0 = value(ws, &point);
                assert!(f0.is_finite(), "{at}: value not finite");
                let mut g = vec![0.0; point.len()];
                super::assembled::gradient_at_mode_ws(ws, x, y, ids, extra_ids, p, n, &mut g[..m])
                    .unwrap_or_else(|| panic!("{at}: gradient declined"));
                if let Some(psi) = *ln_phi {
                    g[m] = super::assembled::gamma_ln_phi_gradient(
                        psi,
                        ws.weighted.then_some(&ws.prior_w[..n]),
                        n,
                        ws.pirls.pen_dev,
                        &point[..n_theta],
                        &g[..n_theta],
                    );
                }
                let gmax = g.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                let mut pt = point.clone();
                for j in 0..point.len() {
                    let h = 1e-5 * point[j].abs().max(1.0);
                    pt[j] = point[j] + h;
                    let fp = value(ws, &pt);
                    pt[j] = point[j] - h;
                    let fm = value(ws, &pt);
                    pt[j] = point[j];
                    let fd = (fp - fm) / (2.0 * h);
                    let tol = 1e-5 * fd.abs() + 1e-6 * (1.0 + gmax);
                    assert!(
                        (g[j] - fd).abs() <= tol,
                        "{at} coord {j}: at-mode {} vs FD {fd} (tol {tol:e})",
                        g[j]
                    );
                }
            }
        }
    };
    let mut cells = 0usize;
    for &(family, shape) in GRADIENT_GATE_CELLS {
        if !routed_without_dispersion(family) {
            continue;
        }
        let (mut ws, x, y, ids, p, n) = fixture(family, shape);
        check(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &[],
            p,
            n,
            &format!("{family:?}/{shape}"),
            fixed_seed_theta(shape),
            &[None],
        );
        cells += 1;
    }
    assert!(cells > 0, "no routed gate cell");
    for link in GAMMA_GATE_LINKS {
        for (shape, np, n_crossed) in GAMMA_GATE_SHAPES {
            for weighted in [false, true] {
                let (mut ws, x, y, ids, extra_ids, p, n, rng) =
                    gamma_gate_cell(link, shape, np, n_crossed, weighted);
                let label = format!("Gamma/{link:?}/{shape} w={weighted}");
                // `fit_glmm`'s plug-in seed: `D/Σw` at the β start, no random
                // effect.
                let (mut d, mut sw) = (0.0, 0.0);
                for i in 0..n {
                    let mut eta = ws.offset.as_ref().map_or(0.0, |o| o[i]);
                    for j in 0..p {
                        eta += x[(i, j)] * rng.beta[j];
                    }
                    d += ws.prior_w[i]
                        * crate::family::dev_resid_eta(ws.family, f64::NAN, y[i], eta);
                    sw += ws.prior_w[i];
                }
                let seed = (d / sw).ln();
                let fit = gate_fit(
                    &mut ws,
                    x.as_ref(),
                    &y,
                    &ids,
                    &extra_ids,
                    None,
                    &rng.beta,
                    n,
                );
                assert!(fit.converged, "{label}: fit did not converge");
                let opt = ws.gamma_phi.ln();
                check(
                    &mut ws,
                    x.as_ref(),
                    &y,
                    &ids,
                    &extra_ids,
                    p,
                    n,
                    &label,
                    rng,
                    &[Some(seed), Some(opt), Some(opt + 1.0), Some(opt - 1.0)],
                );
            }
        }
    }
}

// --- The gradient route ---

/// `fit_glmm` cold (or from `theta_start`) on a gate fixture, target column 1.
#[allow(clippy::too_many_arguments)]
fn gate_fit(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    ids: &[u32],
    extra_ids: &[Vec<u32>],
    theta_start: Option<&[f64]>,
    beta0: &[f64],
    n: usize,
) -> GlmmFit {
    fit_glmm(
        ws,
        x,
        x.subcols(0, 0),
        y,
        ids,
        extra_ids,
        &[1],
        theta_start,
        beta0,
        n,
        WaldSe::Rx,
        ColumnScaling::IDENTITY,
    )
}

/// On every routed gate cell (binomial or Poisson, and the Gamma cells on
/// both links with and without weights and an offset, blocked and structured,
/// n_θ ≥ 2) the gradient route's fit is converged and its deviance is at or
/// below the BOBYQA route's within `1e-6`, cold and from a warm start at the
/// BOBYQA optimum perturbed by 10%; one-θ Laplace cells do not route. The
/// same on the covered AGQ shapes, where a fit routes with one θ, a warm one
/// only from a θ₀ of at least `GRAD_WARM_AGQ_THETA_MIN`: logit and probit
/// random intercepts at 7 nodes and logit at 11, whose warm fits route, and
/// a Poisson log random intercept at 7 nodes and the aggregated binomial
/// `sim_binomial_zerosd` at 7 nodes, whose optima are at θ = 0, so their
/// warm starts near zero keep BOBYQA. There the certificate refuses the cold
/// endpoint and the joint polish (`solver_polish_wide` on AGQ) finishes the
/// cold fit (rung 3 on `sim_binomial_zerosd`). A routed AGQ fit never hands
/// over to BOBYQA here, so a broken gradient or wiring fails.
#[test]
fn gradient_route_matches_or_beats_the_bobyqa_route() {
    let mut routed = 0usize;
    let mut check = |mut ws: GlmmWorkspace,
                     x: MatRef<f64>,
                     y: &[f64],
                     ids: &[u32],
                     extra_ids: &[Vec<u32>],
                     beta0: &[f64],
                     n: usize,
                     label: &str|
     -> Option<(u8, bool)> {
        let plan = ws.route_plan;
        ws.route_plan = crate::optim::RoutePlan::BOBYQA;
        let bob = gate_fit(&mut ws, x, y, ids, extra_ids, None, beta0, n);
        let theta_bob = ws.params[..ws.n_theta].to_vec();
        assert!(
            ws.grad_report.is_none(),
            "{label}: BOBYQA plan ran a gradient search"
        );
        ws.route_plan = plan;
        let cold = gate_fit(&mut ws, x, y, ids, extra_ids, None, beta0, n);
        let cold_report = (ws.grad_report, ws.gradient_search_fallback);
        let warm_start: Vec<f64> = theta_bob.iter().map(|t| 1.1 * t).collect();
        let warm = gate_fit(&mut ws, x, y, ids, extra_ids, Some(&warm_start), beta0, n);
        if ws.n_theta < 2 && ws.nagq == 1 {
            assert!(ws.grad_report.is_none(), "{label}: a one-θ fit routed");
            return None;
        }
        routed += 1;
        let warm_routed = ws.grad_report.is_some();
        if ws.nagq > 1 {
            let off_zero = ws
                .groupings
                .diagonal_theta()
                .iter()
                .all(|&i| warm_start[i] >= GRAD_WARM_AGQ_THETA_MIN);
            assert_eq!(
                warm_routed, off_zero,
                "{label}: warm AGQ routing does not follow its θ₀ {warm_start:?}"
            );
            let mut searches = vec![("cold", cold_report)];
            if off_zero {
                searches.push(("warm", (ws.grad_report, ws.gradient_search_fallback)));
            }
            for (kind, (report, fallback)) in searches {
                let (_, _, _, rung) =
                    report.unwrap_or_else(|| panic!("{label} {kind}: did not route"));
                assert!(
                    rung != 1 && !fallback,
                    "{label} {kind}: the gradient search handed over"
                );
            }
        } else {
            assert!(
                ws.grad_report.is_some(),
                "{label}: the warm fit did not route"
            );
        }
        for (kind, f) in [("cold", &cold), ("warm", &warm)] {
            assert!(f.converged, "{label} {kind}: not converged");
            assert!(
                f.deviance <= bob.deviance + 1e-6,
                "{label} {kind}: gradient route {} vs BOBYQA {}",
                f.deviance,
                bob.deviance
            );
        }
        cold_report.0.map(|r| (r.3, warm_routed))
    };
    for &(family, shape) in GRADIENT_GATE_CELLS {
        if !routed_without_dispersion(family) {
            continue;
        }
        let (ws, x, y, ids, p, n) = fixture(family, shape);
        check(
            ws,
            x.as_ref(),
            &y,
            &ids,
            &[],
            &vec![0.0; p],
            n,
            &format!("{family:?}/{shape}"),
        );
    }
    for &(family, shape, np, n_crossed) in STRUCTURED_GATE_CELLS {
        if !routed_without_dispersion(family) {
            continue;
        }
        let (mut ws, x, y, ids, extra_ids, p, n) = extras_fixture(family, np, n_crossed);
        ws.pattern.structured_schur =
            super::workspace::StructuredSchur::new(&ws.groupings, &ids, &extra_ids, n);
        check(
            ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            &vec![0.0; p],
            n,
            &format!("{family:?}/{shape}"),
        );
    }
    for link in GAMMA_GATE_LINKS {
        for (shape, np, n_crossed) in GAMMA_GATE_SHAPES {
            for weighted in [false, true] {
                let (mut ws, x, y, ids, extra_ids, _, n, rng) =
                    gamma_gate_cell(link, shape, np, n_crossed, weighted);
                ws.pattern.structured_schur =
                    super::workspace::StructuredSchur::new(&ws.groupings, &ids, &extra_ids, n);
                check(
                    ws,
                    x.as_ref(),
                    &y,
                    &ids,
                    &extra_ids,
                    &rng.beta,
                    n,
                    &format!("Gamma/{link:?}/{shape} w={weighted}"),
                );
            }
        }
    }
    let logit = Family::Binomial {
        link: BinomialLink::Logit,
    };
    // The third column: whether the warm fit routes. The Poisson cell's
    // optimum is at θ = 0, so its warm start is too.
    let agq_cells = [
        (logit, 7u8, true),
        (
            Family::Binomial {
                link: BinomialLink::Probit,
            },
            7,
            true,
        ),
        (
            Family::Poisson {
                link: PoissonLink::Log,
            },
            7,
            false,
        ),
        (logit, 11, true),
    ];
    for (family, nagq, warm_routes) in agq_cells {
        let (ws, x, y, ids, p, n) = fixture_with_nagq(family, "int1", nagq);
        let label = format!("{family:?}/int1 nAGQ={nagq}");
        let report = check(ws, x.as_ref(), &y, &ids, &[], &vec![0.0; p], n, &label);
        assert_eq!(
            report.map(|(_, warm_routed)| warm_routed),
            Some(warm_routes),
            "{label}: warm routing"
        );
    }
    // `sim_binomial_zerosd`, lowered the way
    // `glmm_zerosd_boundary_reports_converged_and_singular` (src/fit/glmm_tests.rs)
    // does: y = incidence/size, the trial counts as prior weights, X = [1, x, z],
    // 20 groups.
    let csv = include_str!("../../validation/data/simulated/sim_binomial_zerosd.csv");
    let (mut xv, mut y, mut w, mut ids) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let size: f64 = f[1].parse().unwrap();
        y.push(f[0].parse::<f64>().unwrap() / size);
        w.push(size);
        xv.extend_from_slice(&[1.0, f[2].parse().unwrap(), f[3].parse().unwrap()]);
        ids.push(f[4].parse::<u32>().unwrap() - 1);
    }
    let (n, p) = (y.len(), 3);
    let model = SizedSpec {
        spec: ModelSpec {
            family: logit,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![],
            }),
        },
        n_primary: 20,
        extra_counts: vec![],
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 7);
    ws.prior_w[..n].copy_from_slice(&w);
    ws.weighted = true;
    let x = Mat::<f64>::from_fn(n, p, |i, j| xv[i * p + j]);
    let report = check(
        ws,
        x.as_ref(),
        &y,
        &ids,
        &[],
        &vec![0.0; p],
        n,
        "sim_binomial_zerosd nAGQ=7",
    );
    assert_eq!(
        report,
        Some((3, false)),
        "sim_binomial_zerosd nAGQ=7: the cold fit did not end on the polish, or \
         the warm fit from near θ = 0 routed"
    );
    assert!(routed > 0, "no gate cell took the gradient route");
}

/// A held Gamma φ on the gradient route: the search runs over `[θ | β̃]` at
/// that φ, and its joint polish carries the inert `ln φ` slot. On every Gamma
/// gate cell, at twice the BOBYQA route's free φ̂, the routed fit is converged
/// and its deviance is at or below the BOBYQA route's at the same φ within
/// `dev_eps` (4e-5), φ stays where it was held, and the search does not hand
/// over to BOBYQA. The same at a held φ of exactly 1. At least one cell ends on
/// the joint polish (rung 3), so the inert slot is exercised.
#[test]
fn gradient_route_matches_the_bobyqa_route_at_a_held_gamma_dispersion() {
    let mut polished = 0usize;
    for link in GAMMA_GATE_LINKS {
        for (shape, np, n_crossed) in GAMMA_GATE_SHAPES {
            for weighted in [false, true] {
                let (mut ws, x, y, ids, extra_ids, _, n, rng) =
                    gamma_gate_cell(link, shape, np, n_crossed, weighted);
                ws.pattern.structured_schur =
                    super::workspace::StructuredSchur::new(&ws.groupings, &ids, &extra_ids, n);
                let label = format!("Gamma/{link:?}/{shape} w={weighted}");
                let plan = ws.route_plan;
                ws.route_plan = crate::optim::RoutePlan::BOBYQA;
                let free = gate_fit(
                    &mut ws,
                    x.as_ref(),
                    &y,
                    &ids,
                    &extra_ids,
                    None,
                    &rng.beta,
                    n,
                );
                assert!(free.converged, "{label}: free fit");
                // Twice the free φ̂, and exactly 1: at φ = 1 the value pass
                // leaves `prior_w_phi` unwritten, and the gradient must read
                // the prior weights instead.
                for held in [2.0 * ws.gamma_phi, 1.0] {
                    let label = format!("{label} φ={held}");
                    ws.route_plan = crate::optim::RoutePlan::BOBYQA;
                    ws.gamma_phi_held = Some(held);
                    let bob = gate_fit(
                        &mut ws,
                        x.as_ref(),
                        &y,
                        &ids,
                        &extra_ids,
                        None,
                        &rng.beta,
                        n,
                    );
                    ws.route_plan = plan;
                    let f = gate_fit(
                        &mut ws,
                        x.as_ref(),
                        &y,
                        &ids,
                        &extra_ids,
                        None,
                        &rng.beta,
                        n,
                    );
                    let (_, _, _, rung) = ws
                        .grad_report
                        .unwrap_or_else(|| panic!("{label}: not routed"));
                    polished += usize::from(rung == 3);
                    assert!(
                        rung != 1 && !f.gradient_search_fallback,
                        "{label}: the gradient search handed over"
                    );
                    assert!(bob.converged && f.converged, "{label}: not converged");
                    assert_eq!(ws.gamma_phi, held, "{label}: φ moved");
                    assert!(
                        f.deviance <= bob.deviance + 4e-5,
                        "{label}: gradient route {} vs BOBYQA {}",
                        f.deviance,
                        bob.deviance
                    );
                }
            }
        }
    }
    assert!(polished > 0, "no held-φ cell reached the joint polish");
}

/// The rule's table: in scope binomial, Poisson and Gamma (both links)
/// Laplace, with n_θ ≥ 2 on the blocked or structured layout, on both start
/// kinds, and AGQ on the shapes the exact AGQ gradient covers (binomial logit
/// and probit, Poisson log, one scalar random intercept), one θ included, on
/// both start kinds; vector, NB, Gamma and cloglog AGQ, Laplace NB and one-θ
/// Laplace shapes keep BOBYQA on both start kinds.
#[test]
fn gradient_route_plan_follows_the_rule() {
    use crate::optim::{OuterOptimizer, RoutePlan};
    let logit = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let (ws, ..) = fixture(logit, "q2s");
    assert_eq!(ws.route_plan.cold, OuterOptimizer::Bfgs, "q2s logit cold");
    assert_eq!(ws.route_plan.warm, OuterOptimizer::Bfgs, "q2s logit warm");
    let (ws, ..) = fixture(logit, "int1");
    assert_eq!(ws.route_plan, RoutePlan::BOBYQA, "one θ");
    let (ws, ..) = fixture_with_nagq(logit, "q2s", 7);
    assert_eq!(ws.route_plan, RoutePlan::BOBYQA, "vector AGQ");
    let probit = Family::Binomial {
        link: BinomialLink::Probit,
    };
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    for family in [logit, probit, poisson] {
        let (ws, ..) = fixture_with_nagq(family, "int1", 7);
        assert_eq!(
            ws.route_plan.cold,
            OuterOptimizer::Bfgs,
            "{family:?} AGQ cold"
        );
        assert_eq!(
            ws.route_plan.warm,
            OuterOptimizer::Bfgs,
            "{family:?} AGQ warm"
        );
    }
    let nb = Family::NegativeBinomial {
        link: crate::NegBinomialLink::Log,
    };
    let cloglog = Family::Binomial {
        link: BinomialLink::Cloglog,
    };
    let gamma_log = Family::Gamma {
        link: GammaLink::Log,
    };
    for family in [nb, gamma_log, cloglog] {
        let (ws, ..) = fixture_with_nagq(family, "int1", 7);
        assert_eq!(ws.route_plan, RoutePlan::BOBYQA, "{family:?} AGQ");
    }
    let (ws, ..) = fixture(nb, "q2s");
    assert_eq!(ws.route_plan, RoutePlan::BOBYQA, "NB");
    for link in GAMMA_GATE_LINKS {
        let gamma = Family::Gamma { link };
        let (ws, ..) = fixture(gamma, "q2s");
        assert_eq!(ws.route_plan.cold, OuterOptimizer::Bfgs, "{gamma:?} cold");
        assert_eq!(ws.route_plan.warm, OuterOptimizer::Bfgs, "{gamma:?} warm");
        let (ws, ..) = fixture(gamma, "int1");
        assert_eq!(ws.route_plan, RoutePlan::BOBYQA, "{gamma:?} one θ");
    }
}

/// A fit on a covered AGQ shape keeps BOBYQA from `GRAD_WARM_SCALAR_ROWS_MAX`
/// rows per primary level (49 route, 50 do not, cold and from θ₀ = 0.5) and
/// from a warm start below `GRAD_WARM_AGQ_THETA_MIN` (θ₀ = 0.02 does not
/// route, 0.06 does), on the logit `int1` fixture at 7 nodes with 8 clusters;
/// a fit that does not route is the BOBYQA plan's bit for bit.
#[test]
fn gradient_route_keeps_bobyqa_at_the_agq_gate_thresholds() {
    let logit = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let nc = 8usize;
    for (per, theta0, routes) in [
        (GRAD_WARM_SCALAR_ROWS_MAX - 1, None, true),
        (GRAD_WARM_SCALAR_ROWS_MAX - 1, Some(0.5), true),
        (GRAD_WARM_SCALAR_ROWS_MAX, None, false),
        (GRAD_WARM_SCALAR_ROWS_MAX, Some(0.5), false),
        // `GRAD_WARM_AGQ_THETA_MIN`: a warm start near zero keeps BOBYQA.
        (10, Some(0.02), false),
        (10, Some(0.06), true),
    ] {
        let start_buf = theta0.map(|t| [t]);
        let start = start_buf.as_ref().map(|s| &s[..]);
        let (mut ws, x, y, ids, p, n) = fixture_with_nagq_sized(logit, "int1", 7, per * nc, nc);
        let beta0 = vec![0.0; p];
        let f = gate_fit(&mut ws, x.as_ref(), &y, &ids, &[], start, &beta0, n);
        let label = format!("{per} rows per level, start {theta0:?}");
        assert_eq!(ws.grad_report.is_some(), routes, "{label}");
        if !routes {
            let (mut wb, ..) = fixture_with_nagq_sized(logit, "int1", 7, per * nc, nc);
            wb.route_plan = crate::optim::RoutePlan::BOBYQA;
            let b = gate_fit(&mut wb, x.as_ref(), &y, &ids, &[], start, &beta0, n);
            assert_eq!(
                (f.deviance.to_bits(), f.n_eval),
                (b.deviance.to_bits(), b.n_eval),
                "{label}: not the BOBYQA fit"
            );
        }
    }
}

/// A warm Laplace fit with no random-slope block takes the gradient route at
/// any rows per primary level: on a logit fit with 8 primary levels and a
/// 6-level crossed grouping (two scalar θ, the structured layout), from
/// θ₀ = 0.5 at 50 and 200 rows per level, each fit routes, converges, and
/// ends at or below the BOBYQA plan's deviance from the same start within
/// `1e-6`.
#[test]
fn gradient_route_takes_warm_laplace_fits_at_any_rows_per_level() {
    let nc = 8usize;
    let theta0 = [0.5, 0.5];
    for per in [50usize, 200] {
        let start = Some(&theta0[..]);
        let build = || {
            let (x, y, ids, extra_ids, spec) = glmm_extras_q1_dataset_sized(0, 6, nc, per * nc);
            let n = y.len();
            let mut ws = GlmmWorkspace::for_cluster_spec(2, &spec, n, &[], 1);
            ws.pattern.structured_schur =
                super::workspace::StructuredSchur::new(&ws.groupings, &ids, &extra_ids, n);
            (ws, x, y, ids, extra_ids, n)
        };
        let label = format!("{per} rows per level");
        let (mut ws, x, y, ids, extra_ids, n) = build();
        assert_eq!(ws.n_theta, 2, "{label}: two scalar θ");
        assert_eq!(ws.layout, GlmmLayout::Structured, "{label}: layout");
        let beta0 = vec![0.0; 2];
        let f = gate_fit(&mut ws, x.as_ref(), &y, &ids, &extra_ids, start, &beta0, n);
        assert!(ws.grad_report.is_some(), "{label}: did not route");
        assert!(f.converged, "{label}: not converged");
        let (mut wb, ..) = build();
        wb.route_plan = crate::optim::RoutePlan::BOBYQA;
        let b = gate_fit(&mut wb, x.as_ref(), &y, &ids, &extra_ids, start, &beta0, n);
        assert!(
            f.deviance <= b.deviance + 1e-6,
            "{label}: gradient route {} vs BOBYQA {}",
            f.deviance,
            b.deviance
        );
    }
}

/// The hand-over rung: an iteration cap of 2 ends the gradient search on
/// `MaxIter`, the BOBYQA route finishes from its incumbent, the fit converges
/// at the BOBYQA route's optimum within `1e-6`, and the fallback is recorded
/// for `Note::GradientSearchFallback`.
#[test]
fn gradient_route_hands_over_on_the_iteration_cap() {
    let logit = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let (mut ws, x, y, ids, p, n) = fixture(logit, "q2s");
    let beta0 = vec![0.0; p];
    let plan = ws.route_plan;
    ws.route_plan = crate::optim::RoutePlan::BOBYQA;
    let bob = gate_fit(&mut ws, x.as_ref(), &y, &ids, &[], None, &beta0, n);
    ws.route_plan = plan;
    ws.grad_cfg.max_iter = 2;
    let f = gate_fit(&mut ws, x.as_ref(), &y, &ids, &[], None, &beta0, n);
    let (stop, _, _, rung) = ws.grad_report.expect("routed");
    assert_eq!(stop, crate::optim::bfgs::BfgsStop::MaxIter);
    assert_eq!(rung, 1);
    assert!(f.gradient_search_fallback, "fallback not recorded");
    assert!(f.converged, "the BOBYQA finish did not converge");
    assert!(
        (f.deviance - bob.deviance).abs() <= 1e-6,
        "fallback {} vs BOBYQA {}",
        f.deviance,
        bob.deviance
    );
}

/// The hand-over rung on Gamma, where the BOBYQA finish runs the `ln φ` scale
/// probes and their gate from the gradient search's incumbent: the fit
/// converges at the BOBYQA route's optimum within `1e-6` on both links.
#[test]
fn gradient_route_hands_over_on_gamma() {
    for link in GAMMA_GATE_LINKS {
        let (mut ws, x, y, ids, extra_ids, _, n, rng) = gamma_gate_cell(link, "q2s", 0, 0, false);
        let plan = ws.route_plan;
        ws.route_plan = crate::optim::RoutePlan::BOBYQA;
        let bob = gate_fit(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            None,
            &rng.beta,
            n,
        );
        ws.route_plan = plan;
        ws.grad_cfg.max_iter = 2;
        let f = gate_fit(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            None,
            &rng.beta,
            n,
        );
        let (stop, _, _, rung) = ws.grad_report.expect("routed");
        assert_eq!(stop, crate::optim::bfgs::BfgsStop::MaxIter, "{link:?}");
        assert_eq!(rung, 1, "{link:?}");
        assert!(f.converged, "{link:?}: the BOBYQA finish did not converge");
        assert!(
            (f.deviance - bob.deviance).abs() <= 1e-6,
            "{link:?}: fallback {} vs BOBYQA {}",
            f.deviance,
            bob.deviance
        );
    }
}

/// `GradSearch::start_curvature` seeds the `ln φ` entry on a free Gamma
/// inverse-link dispersion only. With an iteration cap of 0 the search ends
/// at its start, so `ws.bfgs` holds the start inverse Hessian: on Gamma/log
/// and under a held φ it is the identity, on Gamma/inverse its last diagonal
/// entry is a positive `1/(Gₚ'' + P_φ)` below 1.
#[test]
fn gradient_route_seeds_ln_phi_on_the_inverse_link_only() {
    for link in GAMMA_GATE_LINKS {
        for held in [false, true] {
            let (mut ws, x, y, ids, extra_ids, _, n, rng) =
                gamma_gate_cell(link, "q2s", 0, 0, false);
            if held {
                ws.gamma_phi_held = Some(0.5);
            }
            ws.grad_cfg.max_iter = 0;
            let _ = gate_fit(
                &mut ws,
                x.as_ref(),
                &y,
                &ids,
                &extra_ids,
                None,
                &rng.beta,
                n,
            );
            let label = format!("{link:?} held={held}");
            let (stop, ..) = ws.grad_report.expect("routed");
            assert_eq!(stop, crate::optim::bfgs::BfgsStop::MaxIter, "{label}");
            let h = ws.bfgs.inv_hessian();
            let m = ws.n_theta + ws.p + usize::from(!held);
            assert_eq!(h.len(), m * m, "{label}");
            for i in 0..m {
                for j in 0..m {
                    let v = h[i * m + j];
                    if link == GammaLink::Inverse && !held && i == m - 1 && j == i {
                        assert!(v > 0.0 && v < 1.0, "{label}: ψ entry {v}");
                    } else {
                        assert_eq!(v, f64::from(u8::from(i == j)), "{label}: H[{i},{j}]");
                    }
                }
            }
        }
    }
}

/// The β information the gradient search whitens by on a Laplace fit on the
/// opt-in,
/// recomputed outside the fit: one `ProfilePql` solve at `[θ₀ | β₀]` from
/// `û = 0`, factored by `BetaScale::set_info`, beside the design whitening
/// `BetaScale::set`. Returns `(info, set)`. φ is 1 (no Gamma here).
#[allow(clippy::too_many_arguments)]
fn start_whitenings(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    ids: &[u32],
    extra_ids: &[Vec<u32>],
    theta0: &[f64],
    beta0: &[f64],
    n: usize,
) -> (BetaScale, BetaScale) {
    let (nt, p) = (ws.n_theta, ws.p);
    ws.params[..nt].copy_from_slice(theta0);
    ws.params[nt..nt + p].copy_from_slice(beta0);
    ws.beta_prof[..p].copy_from_slice(beta0);
    ws.pirls.u.fill(0.0);
    let mut c = EvalCounters::new();
    let dev = laplace_deviance_ws(ws, x, y, ids, extra_ids, n, BetaMode::ProfilePql, &mut c);
    assert!(dev.is_finite(), "the start's ProfilePql solve");
    let sw: f64 = ws.prior_w[..n].iter().sum();
    let mut info = BetaScale::new(p, ws.params.len());
    assert!(
        info.set_info(ws.border.schur.as_mut(), sw),
        "set_info refused"
    );
    let mut set = BetaScale::new(p, ws.params.len());
    set.set(x, &ws.prior_w[..n], n);
    (info, set)
}

/// On a routed Laplace fit on the opt-in (`beta_info_scaling`) without a
/// hint, the search runs on β̃ whitened by the β information at its start and seeds every β̃ entry with
/// `1/(2·Σ prior weight)`, cold and warm. With an iteration cap of 0 the
/// search ends at its start, so `ws.bfgs` holds the start inverse Hessian:
/// θ's and an unseeded ψ's entries are the identity's, the β̃ entries the
/// seed, every other entry 0. Cells: a blocked logit random slope, a
/// structured Poisson fit with a crossed grouping, and a weighted Gamma/log
/// random slope with an offset (free φ, ψ unseeded). On the logit and
/// Poisson cells the map the fit keeps for its curvature hint,
/// `T = L_setᵀ·L_info⁻ᵀ`, takes `L_infoᵀβ` to `L_setᵀβ` for a recomputed
/// information whitening at the start (`start_whitenings`): cold from the
/// blind θ₀ and β₀, warm from the BOBYQA optimum's θ (which the start guard
/// keeps) and the guard's profiled β.
#[test]
#[allow(clippy::type_complexity)]
fn gradient_route_whitens_beta_by_its_information() {
    let logit = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    let mut cells: Vec<(
        String,
        GlmmWorkspace,
        Mat<f64>,
        Vec<f64>,
        Vec<u32>,
        Vec<Vec<u32>>,
        Vec<f64>,
    )> = Vec::new();
    let (ws, x, y, ids, p, _) = fixture(logit, "q2s");
    cells.push(("logit q2s".into(), ws, x, y, ids, Vec::new(), vec![0.0; p]));
    let (mut ws, x, y, ids, extra_ids, p, n) = extras_fixture(poisson, 0, 6);
    ws.pattern.structured_schur =
        super::workspace::StructuredSchur::new(&ws.groupings, &ids, &extra_ids, n);
    cells.push((
        "Poisson crossed6".into(),
        ws,
        x,
        y,
        ids,
        extra_ids,
        vec![0.0; p],
    ));
    let (ws, x, y, ids, extra_ids, _, _, rng) = gamma_gate_cell(GammaLink::Log, "q2s", 0, 0, true);
    cells.push((
        "Gamma/log q2s weighted".into(),
        ws,
        x,
        y,
        ids,
        extra_ids,
        rng.beta,
    ));
    let mut checked = 0usize;
    for (name, mut ws, x, y, ids, extra_ids, beta0) in cells {
        ws.beta_info_scaling = true;
        let n = y.len();
        let (nt, p) = (ws.n_theta, ws.p);
        let gamma = matches!(ws.family, Family::Gamma { .. });
        let plan = ws.route_plan;
        ws.route_plan = crate::optim::RoutePlan::BOBYQA;
        let _ = gate_fit(&mut ws, x.as_ref(), &y, &ids, &extra_ids, None, &beta0, n);
        let theta_hat = ws.params[..nt].to_vec();
        ws.route_plan = plan;
        ws.grad_cfg.max_iter = 0;
        let mut blind = vec![0.0; nt];
        for &i in ws.groupings.diagonal_theta() {
            blind[i] = THETA0;
        }
        for warm in [false, true] {
            let label = format!("{name}, warm {warm}");
            let start = warm.then_some(&theta_hat[..]);
            let _ = gate_fit(&mut ws, x.as_ref(), &y, &ids, &extra_ids, start, &beta0, n);
            let (stop, ..) = ws.grad_report.expect("routed");
            assert_eq!(stop, crate::optim::bfgs::BfgsStop::MaxIter, "{label}");
            assert!(ws.grad_beta_info, "{label}: not information-whitened");
            let m = nt + p + usize::from(gamma);
            let h = ws.bfgs.inv_hessian().to_vec();
            assert_eq!(h.len(), m * m, "{label}");
            let seed = 1.0 / (2.0 * ws.prior_w[..n].iter().sum::<f64>());
            for i in 0..m {
                for j in 0..m {
                    let want = if i != j {
                        0.0
                    } else if (nt..nt + p).contains(&i) {
                        seed
                    } else {
                        1.0
                    };
                    assert_eq!(h[i * m + j], want, "{label}: H[{i},{j}]");
                }
            }
            if gamma {
                continue;
            }
            let t = ws.grad_beta_map[..p * p].to_vec();
            let (theta0, b0) = if warm {
                // The warm θ₀ as `fit_glmm` floors it.
                let mut t0 = theta_hat.clone();
                for &i in ws.groupings.diagonal_theta() {
                    t0[i] = t0[i].max(THETA_TRUTH_FLOOR);
                }
                (t0, ws.guard_beta[..p].to_vec())
            } else {
                (blind.clone(), beta0.clone())
            };
            let (info, set) =
                start_whitenings(&mut ws, x.as_ref(), &y, &ids, &extra_ids, &theta0, &b0, n);
            for k in 0..3 {
                let b: Vec<f64> = (0..p)
                    .map(|j| ((k * p + j) as f64 * 0.7).sin() + 0.2)
                    .collect();
                let mut bi = b.clone();
                info.to_search(&mut bi);
                let mut bs = b.clone();
                set.to_search(&mut bs);
                for r in 0..p {
                    let got: f64 = (0..p).map(|c| t[r * p + c] * bi[c]).sum();
                    assert!(
                        (got - bs[r]).abs() <= 1e-10 * bs[r].abs().max(1.0),
                        "{label}: T·L_infoᵀβ {got} vs L_setᵀβ {}",
                        bs[r]
                    );
                }
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 4, "map checks run");
}

/// A routed fit without the opt-in (`beta_info_scaling`) keeps the design
/// whitening and the identity start: a logit random slope. On the opt-in, a
/// fit whose start solve or `set_info` fails keeps them too: the same shape
/// with one more fixed-effect column of zeros, whose β information is
/// singular. So do AGQ and hinted fits on the opt-in: a 7-node logit AGQ
/// random intercept starts from the identity, and a hinted logit random
/// slope from the hint itself, bit for bit. Every fit here has an iteration
/// cap of 0, so `ws.bfgs` holds the start inverse Hessian.
#[test]
fn gradient_route_keeps_the_design_whitening_by_default_and_on_fallback_agq_and_hinted_fits() {
    let logit = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let identity = |ws: &GlmmWorkspace, label: &str| {
        assert!(!ws.grad_beta_info, "{label}: information-whitened");
        let h = ws.bfgs.inv_hessian();
        let m = ws.n_theta + ws.p;
        assert_eq!(h.len(), m * m, "{label}");
        for i in 0..m {
            for j in 0..m {
                assert_eq!(
                    h[i * m + j],
                    f64::from(u8::from(i == j)),
                    "{label}: H[{i},{j}]"
                );
            }
        }
    };
    let (mut ws, x, y, ids, p, n) = fixture(logit, "q2s");
    ws.grad_cfg.max_iter = 0;
    let _ = gate_fit(&mut ws, x.as_ref(), &y, &ids, &[], None, &vec![0.0; p], n);
    assert!(ws.grad_report.is_some(), "default: did not route");
    identity(&ws, "default");
    let (mut ws, mut x, y, ids, p, n) = fixture_padded(logit, "q2s", 1);
    for i in 0..n {
        x[(i, p - 1)] = 0.0;
    }
    ws.beta_info_scaling = true;
    ws.grad_cfg.max_iter = 0;
    let _ = gate_fit(&mut ws, x.as_ref(), &y, &ids, &[], None, &vec![0.0; p], n);
    assert!(ws.grad_report.is_some(), "zero column: did not route");
    identity(&ws, "zero column");
    let (mut ws, x, y, ids, p, n) = fixture_with_nagq(logit, "int1", 7);
    ws.beta_info_scaling = true;
    ws.grad_cfg.max_iter = 0;
    let _ = gate_fit(&mut ws, x.as_ref(), &y, &ids, &[], None, &vec![0.0; p], n);
    assert!(ws.grad_report.is_some(), "AGQ: did not route");
    identity(&ws, "AGQ");
    let (mut pilot, x, y, ids, p, n) = fixture(logit, "q2s");
    let beta0 = vec![0.0; p];
    let _ = gate_fit(&mut pilot, x.as_ref(), &y, &ids, &[], None, &beta0, n);
    let hint = super::glmm_curvature_hint(&pilot).expect("a routed pilot offers a hint");
    let (mut ws, ..) = fixture(logit, "q2s");
    ws.beta_info_scaling = true;
    ws.grad_cfg.max_iter = 0;
    ws.grad_hint_in = Some(hint.clone());
    let _ = gate_fit(&mut ws, x.as_ref(), &y, &ids, &[], None, &beta0, n);
    assert!(!ws.grad_beta_info, "hinted: information-whitened");
    let got: Vec<u64> = ws.bfgs.inv_hessian().iter().map(|v| v.to_bits()).collect();
    let want: Vec<u64> = hint.inv_hessian().iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, want, "hinted: the start is not the hint");
}

/// The curvature hint after an information-whitened search is that search's
/// estimate in the design whitening's coordinates: for gradients `g` in the
/// caller's units, `gᵀHg` with `g` mapped by `L_set⁻¹` and the hint equals
/// `gᵀHg` with `g` mapped by `L_info⁻¹` and the search's own estimate
/// (relative 1e-9), on θ-only, β-only and mixed `g`; and a hinted refit of
/// the same data converges at the pilot's deviance within `1e-6`. On the
/// cold logit random slope and the structured Poisson crossed fit.
#[test]
#[allow(clippy::type_complexity)]
fn gradient_route_curvature_hint_is_in_the_design_coordinates() {
    let logit = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    let mut cells: Vec<(
        &str,
        GlmmWorkspace,
        Mat<f64>,
        Vec<f64>,
        Vec<u32>,
        Vec<Vec<u32>>,
    )> = Vec::new();
    let (ws, x, y, ids, ..) = fixture(logit, "q2s");
    cells.push(("logit q2s", ws, x, y, ids, Vec::new()));
    let (mut ws, x, y, ids, extra_ids, _, n) = extras_fixture(poisson, 0, 6);
    ws.pattern.structured_schur =
        super::workspace::StructuredSchur::new(&ws.groupings, &ids, &extra_ids, n);
    cells.push(("Poisson crossed6", ws, x, y, ids, extra_ids));
    for (name, mut ws, x, y, ids, extra_ids) in cells {
        ws.beta_info_scaling = true;
        let n = y.len();
        let (nt, p) = (ws.n_theta, ws.p);
        let m = nt + p;
        let beta0 = vec![0.0; p];
        let pilot = gate_fit(&mut ws, x.as_ref(), &y, &ids, &extra_ids, None, &beta0, n);
        assert!(ws.grad_beta_info, "{name}: not information-whitened");
        let hint = super::glmm_curvature_hint(&ws).expect("a routed pilot offers a hint");
        let h_search = ws.bfgs.inv_hessian().to_vec();
        let mut blind = vec![0.0; nt];
        for &i in ws.groupings.diagonal_theta() {
            blind[i] = THETA0;
        }
        let (info, set) =
            start_whitenings(&mut ws, x.as_ref(), &y, &ids, &extra_ids, &blind, &beta0, n);
        let quad = |h: &[f64], g: &[f64]| -> f64 {
            (0..m)
                .map(|i| (0..m).map(|j| g[i] * h[i * m + j] * g[j]).sum::<f64>())
                .sum()
        };
        for (kind, th, be) in [("θ", 1.0, 0.0), ("β", 0.0, 1.0), ("mixed", 1.0, 1.0)] {
            let g: Vec<f64> = (0..m)
                .map(|i| (if i < nt { th } else { be }) * ((i as f64) * 1.3 + 0.4).cos())
                .collect();
            let (mut gs, mut gi) = (g.clone(), g.clone());
            set.to_search_row(&mut gs[nt..]);
            info.to_search_row(&mut gi[nt..]);
            let (a, b) = (quad(hint.inv_hessian(), &gs), quad(&h_search, &gi));
            assert!(
                (a - b).abs() <= 1e-9 * b.abs(),
                "{name} {kind}: hint {a} vs search {b}"
            );
        }
        ws.grad_hint_in = Some(hint);
        let refit = gate_fit(&mut ws, x.as_ref(), &y, &ids, &extra_ids, None, &beta0, n);
        assert!(refit.converged, "{name}: hinted refit not converged");
        assert!(
            (refit.deviance - pilot.deviance).abs() <= 1e-6,
            "{name}: hinted refit {} vs pilot {}",
            refit.deviance,
            pilot.deviance
        );
    }
}

/// The gradient route keeps per-fit determinism: a routed fit on a
/// workspace that has fitted other data first (another response, and a warm
/// start) matches a fresh workspace's fit of the same data bit for bit,
/// because the search's state (`BfgsState`, the incumbent mode, the assembly
/// scratch) is reset or rewritten per fit. On logit and on Gamma/log, whose
/// `params` carry the fitted `ln φ`, and on a 7-node logit AGQ random
/// intercept, whose search also keeps the AGQ gradient's index and buffers.
#[test]
fn gradient_route_refit_on_a_reused_workspace_is_bit_identical() {
    let cells: [(Family, &str, u8, &[f64]); 3] = [
        (
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            "q2s",
            1,
            &[0.4, 0.1, 0.3],
        ),
        (
            Family::Gamma {
                link: GammaLink::Log,
            },
            "q2s",
            1,
            &[0.4, 0.1, 0.3],
        ),
        (
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            "int1",
            7,
            &[0.4],
        ),
    ];
    for (family, shape, nagq, warm) in cells {
        let (mut fresh, x, y, ids, p, n) = fixture_with_nagq(family, shape, nagq);
        let beta0 = vec![0.0; p];
        let want = gate_fit(&mut fresh, x.as_ref(), &y, &ids, &[], None, &beta0, n);
        assert!(
            fresh.grad_report.is_some(),
            "{family:?} {shape} nAGQ={nagq}: did not route"
        );
        let (mut ws, ..) = fixture_with_nagq(family, shape, nagq);
        let y2: Vec<f64> = match family {
            Family::Gamma { .. } => y.iter().map(|v| 1.0 / v).collect(),
            _ => y.iter().map(|v| 1.0 - v).collect(),
        };
        let _ = gate_fit(&mut ws, x.as_ref(), &y2, &ids, &[], None, &beta0, n);
        let _ = gate_fit(&mut ws, x.as_ref(), &y, &ids, &[], Some(warm), &beta0, n);
        let got = gate_fit(&mut ws, x.as_ref(), &y, &ids, &[], None, &beta0, n);
        let at = format!("{family:?} {shape} nAGQ={nagq}");
        assert_eq!(
            got.deviance.to_bits(),
            want.deviance.to_bits(),
            "{at}: deviance"
        );
        assert_eq!(got.n_eval, want.n_eval, "{at}: n_eval");
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&ws.params), bits(&fresh.params), "{at}: params");
        assert_eq!(bits(&ws.betas), bits(&fresh.betas), "{at}: betas");
    }
}

/// Units of X: the routed fit with the slope column multiplied by 1e3 and by
/// 1e-3 (`fit_glmm` directly, so no standardization upstream) ends at or
/// below the BOBYQA route's deviance on the same scaled data within `1e-6`,
/// and the logit fit at the unscaled optimum within `1e-7`. This is what
/// decided the whitened β coordinates (`BetaScale`): in the caller's units
/// the logit fit at 1e-3 stopped 5.5e-4 above the optimum. On the `q2s`
/// fixture the column also carries the random slope, so its θ rescales by the
/// inverse factor and the Poisson fixture's boundary optimum (every variance
/// at 0) is hard for both routes there: at 1e-3 BOBYQA ends 1.8e-5 above it,
/// BFGS 1.8e-6 (2026-10-07). Gamma/log, with its free `ln φ`, runs on the
/// structured `crossed6` cell instead, whose random effects are intercepts
/// only, so the scaled column enters β alone: the kernel searches θ in the
/// units of the slope column it is given, and only `fit_on`'s column scaling
/// makes those unit-free.
#[test]
fn gradient_route_is_unit_invariant_in_beta() {
    for family in [
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        Family::Poisson {
            link: PoissonLink::Log,
        },
        Family::Gamma {
            link: GammaLink::Log,
        },
    ] {
        let cell = || match family {
            Family::Gamma { link } => {
                let (mut ws, x, y, ids, extra_ids, p, n, _) =
                    gamma_gate_cell(link, "crossed6", 0, 6, false);
                ws.pattern.structured_schur =
                    super::workspace::StructuredSchur::new(&ws.groupings, &ids, &extra_ids, n);
                (ws, x, y, ids, extra_ids, p, n)
            }
            _ => {
                let (ws, x, y, ids, p, n) = fixture(family, "q2s");
                (ws, x, y, ids, Vec::new(), p, n)
            }
        };
        let (mut ws, x, y, ids, extra_ids, p, n) = cell();
        let beta0 = vec![0.0; p];
        let base = gate_fit(&mut ws, x.as_ref(), &y, &ids, &extra_ids, None, &beta0, n);
        for scale in [1e3, 1e-3] {
            let mut xs = x.clone();
            for i in 0..n {
                xs[(i, 1)] *= scale;
            }
            let (mut w2, ..) = cell();
            let f = gate_fit(&mut w2, xs.as_ref(), &y, &ids, &extra_ids, None, &beta0, n);
            assert!(w2.grad_report.is_some(), "{family:?} x{scale}: not routed");
            let (mut wb, ..) = cell();
            wb.route_plan = crate::optim::RoutePlan::BOBYQA;
            let b = gate_fit(&mut wb, xs.as_ref(), &y, &ids, &extra_ids, None, &beta0, n);
            assert!(f.converged, "{family:?} x{scale}: not converged");
            assert!(
                f.deviance <= b.deviance + 1e-6,
                "{family:?} x{scale}: gradient route {} vs BOBYQA {}",
                f.deviance,
                b.deviance
            );
            if matches!(family, Family::Binomial { .. }) {
                assert!(
                    (f.deviance - base.deviance).abs() <= 1e-7,
                    "{family:?} x{scale}: {} vs unscaled {}",
                    f.deviance,
                    base.deviance
                );
            }
        }
    }
}

/// The GLMM curvature hint: a pilot fit's own final BFGS estimate is
/// offered (`glmm_curvature_hint`), a hinted fit consumes it within its call,
/// results do not depend on the order of hinted draws, and a hinted draw ends
/// at or below the unhinted one's deviance within `1e-6`. On the logit `q2s`
/// Laplace fixture and on a 7-node logit AGQ random intercept; every fit here
/// is cold.
#[test]
fn gradient_route_curvature_hint_is_order_free() {
    let logit = Family::Binomial {
        link: BinomialLink::Logit,
    };
    for (shape, nagq) in [("q2s", 1u8), ("int1", 7)] {
        let (mut pilot, x, y, ids, p, n) = fixture_with_nagq(logit, shape, nagq);
        let beta0 = vec![0.0; p];
        let _ = gate_fit(&mut pilot, x.as_ref(), &y, &ids, &[], None, &beta0, n);
        let hint = super::glmm_curvature_hint(&pilot).expect("a routed pilot offers a hint");
        assert_eq!(hint.dim(), pilot.n_theta + p);
        let draws: Vec<Vec<f64>> = (0..3)
            .map(|d| {
                y.iter()
                    .enumerate()
                    .map(|(i, &v)| if (i + d) % 11 == 0 { 1.0 - v } else { v })
                    .collect()
            })
            .collect();
        let run = |order: &[usize]| -> Vec<(u64, usize)> {
            let (mut ws, ..) = fixture_with_nagq(logit, shape, nagq);
            let mut out = vec![(0u64, 0usize); draws.len()];
            for &d in order {
                ws.grad_hint_in = Some(hint.clone());
                let f = gate_fit(&mut ws, x.as_ref(), &draws[d], &ids, &[], None, &beta0, n);
                assert!(ws.grad_hint_in.is_none(), "the hint outlived its call");
                out[d] = (f.deviance.to_bits(), f.n_eval);
            }
            out
        };
        let a = run(&[0, 1, 2]);
        let b = run(&[2, 0, 1]);
        assert_eq!(
            a, b,
            "{shape}: hinted results depend on the order of the draws"
        );
        let (mut plain, ..) = fixture_with_nagq(logit, shape, nagq);
        for (d, yd) in draws.iter().enumerate() {
            let f = gate_fit(&mut plain, x.as_ref(), yd, &ids, &[], None, &beta0, n);
            assert!(
                f64::from_bits(a[d].0) <= f.deviance + 1e-6,
                "{shape} draw {d}: hinted {} vs unhinted {}",
                f64::from_bits(a[d].0),
                f.deviance
            );
            println!(
                "hint {shape} nAGQ={nagq} draw {d}: n_eval hinted {} unhinted {}",
                a[d].1, f.n_eval
            );
        }
    }
}

/// Quasi-separation: the `q2s` logit fixture with its response replaced by
/// a near-step in the slope covariate (`y = 1` where `x₁ > 0`, every 13th row
/// flipped), so β₁ runs off along a flat direction. The gradient route must
/// either end converged at a deviance no worse than the BOBYQA route's by
/// more than `dev_eps` (4e-5), or hand over to BOBYQA; it must never accept an
/// endpoint that stopped on its iteration cap (rung 0 only on a convergence
/// stop).
#[test]
fn gradient_route_handles_quasi_separation() {
    use crate::optim::bfgs::BfgsStop;
    let logit = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let (mut ws, x, _y, ids, p, n) = fixture(logit, "q2s");
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let step = if x[(i, 1)] > 0.0 { 1.0 } else { 0.0 };
            if i % 13 == 0 {
                1.0 - step
            } else {
                step
            }
        })
        .collect();
    let beta0 = vec![0.0; p];
    let f = gate_fit(&mut ws, x.as_ref(), &y, &ids, &[], None, &beta0, n);
    let report = ws.grad_report;
    assert!(report.is_some(), "the quasi-separated fit did not route");
    let (mut wb, ..) = fixture(logit, "q2s");
    wb.route_plan = crate::optim::RoutePlan::BOBYQA;
    let b = gate_fit(&mut wb, x.as_ref(), &y, &ids, &[], None, &beta0, n);
    assert!(f.deviance.is_finite(), "routed deviance {}", f.deviance);
    if let Some((stop, _, _, rung)) = report {
        if rung == 0 {
            assert!(
                matches!(
                    stop,
                    BfgsStop::Fdelta | BfgsStop::Pgrad | BfgsStop::NoiseFloor
                ),
                "an endpoint stopped on {stop:?} was accepted"
            );
        }
    }
    if f.converged && !f.gradient_search_fallback {
        assert!(
            f.deviance <= b.deviance + 4e-5,
            "gradient route {} vs BOBYQA {}",
            f.deviance,
            b.deviance
        );
    }
    assert!(
        f.converged || f.gradient_search_fallback || !b.converged,
        "the routed fit failed where BOBYQA converged, with no Note"
    );
}

/// The second and third routed fits on a reused workspace allocate the same
/// and no more than the same fits on the BOBYQA plan: the gradient search
/// (its `BfgsState`, the assembly scratch kept under `GRAD_ASM_KEEP_BYTES`)
/// allocates nothing per fit once the workspace has fitted once. On logit and
/// on Gamma/log with its free `ln φ`, and on a 7-node logit AGQ random
/// intercept, whose search also keeps the AGQ gradient's index and buffers.
/// Run: `RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests gradient_route_repeat_fits_allocate_no_more -- --ignored --test-threads=1`
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn gradient_route_repeat_fits_allocate_no_more() {
    let _serial = crate::test_support::alloc_test_guard();
    let logit = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let gamma_log = Family::Gamma {
        link: GammaLink::Log,
    };
    for (family, shape, nagq) in [
        (logit, "q2s", 1u8),
        (gamma_log, "q2s", 1),
        (logit, "int1", 7),
    ] {
        let (mut wr, x, y, ids, p, n) = fixture_with_nagq(family, shape, nagq);
        let (mut wb, ..) = fixture_with_nagq(family, shape, nagq);
        wb.route_plan = crate::optim::RoutePlan::BOBYQA;
        let beta0 = vec![0.0; p];
        let blocks = |ws: &mut GlmmWorkspace| {
            let prof = dhat::Profiler::builder().testing().build();
            let _ = gate_fit(ws, x.as_ref(), &y, &ids, &[], None, &beta0, n);
            let stats = dhat::HeapStats::get();
            drop(prof);
            stats.total_blocks
        };
        let _ = gate_fit(&mut wr, x.as_ref(), &y, &ids, &[], None, &beta0, n);
        assert!(
            wr.grad_report.is_some(),
            "{family:?} {shape} nAGQ={nagq}: did not route"
        );
        let _ = gate_fit(&mut wb, x.as_ref(), &y, &ids, &[], None, &beta0, n);
        let r2 = blocks(&mut wr);
        let r3 = blocks(&mut wr);
        let b2 = blocks(&mut wb);
        assert_eq!(
            r2, r3,
            "{family:?} {shape} nAGQ={nagq}: routed refits allocate a growing amount"
        );
        assert!(
            r2 <= b2,
            "{family:?} {shape} nAGQ={nagq}: routed refit {r2} blocks vs BOBYQA refit {b2}"
        );
    }
}
