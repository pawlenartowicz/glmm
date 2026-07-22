use super::agq::*;
use super::deviance::*;
use super::pirls::*;
use super::se::*;
use super::workspace::*;
use super::*;
use crate::test_support::{intercept_only_spec, TestWs};
use crate::{
    BinomialLink, Family, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing, WaldSe,
};
use faer::linalg::solvers::Solve;

/// The dense GLMM kernel entry rejects a hand-built workspace whose extra
/// grouping carries a random slope (`extra_slopes_any`). In normal use
/// `classify_design` routes any extra-slope shape to Sparse, so the guard is the
/// backstop for a caller that constructs the dense `GlmmWorkspace` directly — in
/// release this path used to silently drop the slope. The panic now comes from
/// `from_groupings` (release too), not from the per-eval `debug_assert`s, so it
/// fires at `for_cluster_spec` above rather than inside `fit_glmm`.
#[test]
#[should_panic(expected = "route it to the sparse solver")]
fn dense_glmm_entry_rejects_extra_slopes() {
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 2 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 2 },
                slopes: vec![1],
            }],
        }),
    };
    let n = 8;
    let p = 2;
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
    let x = faer::Mat::<f64>::zeros(n, p);
    let y = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let cluster_ids = vec![0u32, 0, 1, 1, 0, 0, 1, 1];
    let _ = fit_glmm(
        &mut ws,
        x.as_ref(),
        &y,
        &cluster_ids,
        &[],
        None,
        &[0.0; 2],
        n,
        WaldSe::Hessian,
    );
}

/// Intercept-only spec carrying the **binomial-logit** family — the family the
/// GLMM kernel has always used here. (Pre-M3 the kernel ignored `spec.family`;
/// M3 made `ws.family` load-bearing, so these legacy binomial tests must set it
/// rather than inherit `intercept_only_spec`'s Gaussian default.)
fn logit_intercept_spec(sizing: Sizing) -> ModelSpec {
    let mut s = intercept_only_spec(sizing);
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
fn glmm_intercept_dataset_layout(contiguous: bool) -> (Mat<f64>, Vec<f64>, Vec<u32>) {
    let (n, nc) = (80usize, 8usize);
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
    glmm_intercept_dataset_layout(false)
}

/// q_p=2 (intercept + slope on col 0) primary grouping plus ONE crossed extra
/// grouping. Returns (X f64 [1, x1], y, primary ids, crossed ids, spec). Every
/// crossed level gets ≥1 obs (round-robin), and every primary level too.
fn glmm_slope_crossed_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>, Vec<u32>, ModelSpec) {
    let (n, n_prim, n_crossed) = (96usize, 8usize, 4usize);
    let mut st = 13u64;
    let u0: Vec<f64> = (0..n_prim).map(|_| 0.6 * lcg(&mut st)).collect();
    let u1: Vec<f64> = (0..n_prim).map(|_| 0.4 * lcg(&mut st)).collect();
    let uc: Vec<f64> = (0..n_crossed).map(|_| 0.5 * lcg(&mut st)).collect();
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
    // Slope on design col 0 of the [1, x1] X passed to build_z (the slope_cols
    // arg is &[1] there — the x1 column index in the full X); the spec carries
    // ONE slope + ONE crossed grouping so for_cluster_spec sizes q_p=2 + tail.
    let cluster = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_prim as u32,
            },
            slopes: vec![0],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed {
                    n_clusters: n_crossed as u32,
                },
                slopes: vec![],
            }],
        }),
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

#[test]
fn laplace_deviance_matches_brute_force_intercept() {
    let (xf64, y, ids) = glmm_intercept_dataset();
    let beta = [0.2_f64, 0.8];
    let want = brute_force_intercept_laplace(0.5, &beta, &xf64, &y, &ids, 8);
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
    build_z(&mut ws, xf64.as_ref(), &ids, &[], 80);
    ws.params[0] = 0.5;
    ws.params[1] = beta[0];
    ws.params[2] = beta[1];
    let got = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf64.as_ref(), &y, &ids, 80);
    assert!(
        (got - want).abs() < 1e-6,
        "laplace dev: got {got}, want {want}"
    );
}

#[test]
fn beta_fixed_mode_is_pure_plumbing() {
    // Fixed-mode β threading is a value-exact copy (params[n_theta..n_theta+p]
    // → the β buffer) that moves no math, so repeat evaluations of the same
    // fixture must return the bit-identical f64. This pins determinism across
    // calls (which Profile mode must also keep in later tasks) on top of the
    // full oracle suite's bit-identity witness.
    let (xf64, y, ids) = glmm_intercept_dataset();
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
    build_z(&mut ws, xf64.as_ref(), &ids, &[], 80);
    ws.params[0] = 0.5;
    ws.params[1] = 0.2;
    ws.params[2] = 0.8;
    let params = ws.params.clone();
    let d1 = glmm_laplace_deviance(&params, &mut ws, xf64.as_ref(), &y, &ids, 80);
    let d2 = glmm_laplace_deviance(&params, &mut ws, xf64.as_ref(), &y, &ids, 80);
    assert_eq!(d1.to_bits(), d2.to_bits());
}

#[test]
fn agq_k1_reduces_to_laplace() {
    // The k=1 adaptive node sits at the mode with GH weight √π, so agq_deviance
    // at nagq=1 equals laplace_deviance analytically. They assemble it via
    // different ops (scalar dev_resid + log-sum-exp vs the kernel's dev/pen +
    // 2·logdet), so compare with a tight relative tolerance, NOT bit-equality.
    // The production nagq=1 path is bit-identical regardless: laplace_deviance's
    // AGQ guard fires only for nagq>1. Run at a non-unit τ² (= 0.5²) so a missing
    // λ-scale in the integrand would surface (the k=1 reduction is τ-dependent).
    let (xf64, y, ids) = glmm_intercept_dataset();
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
    build_z(&mut ws, xf64.as_ref(), &ids, &[], 80);
    ws.params[0] = 0.5; // θ (Λ scalar) ≠ 1 — exercises the λ-scale
    ws.params[1] = 0.2;
    ws.params[2] = 0.8;
    let p = ws.params.clone();
    let lap = glmm_laplace_deviance(&p, &mut ws, xf64.as_ref(), &y, &ids, 80);
    let agq1 = glmm_agq_deviance(&p, &mut ws, xf64.as_ref(), &y, &ids, 80, 1);
    assert!(
        (agq1 - lap).abs() <= 1e-12 * lap.abs().max(1.0),
        "agq(k=1) {agq1} vs laplace {lap}"
    );
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

    let cluster = logit_intercept_spec(Sizing::FixedClusters {
        n_clusters: nc as u32,
    });
    for nagq in [1u8, 7] {
        let mut ws_a = GlmmWorkspace::for_cluster_spec(2, &cluster, n_a, &[], nagq);
        build_z(&mut ws_a, xa_mat.as_ref(), &ids_a, &[], n_a);
        ws_a.weighted = true;
        ws_a.prior_w[..n_a].copy_from_slice(&wa);
        ws_a.params[0] = 0.5; // θ (Λ scalar)
        ws_a.params[1] = 0.2; // β0
        ws_a.params[2] = 0.8; // β1
        let pa = ws_a.params.clone();
        let da = glmm_agq_deviance(&pa, &mut ws_a, xa_mat.as_ref(), &ya, &ids_a, n_a, nagq);

        let mut ws_e = GlmmWorkspace::for_cluster_spec(2, &cluster, n_e, &[], nagq);
        build_z(&mut ws_e, xe_mat.as_ref(), &ids_e, &[], n_e);
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
/// safety argument.
#[test]
fn agq_cluster_outer_bit_identical_to_node_outer() {
    let (xf64, y, ids) = glmm_intercept_dataset();
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
    for nagq in [3u8, 7, 11] {
        let mut ws_a = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], nagq);
        build_z(&mut ws_a, xf64.as_ref(), &ids, &[], 80);
        ws_a.params[0] = 0.5; // θ (Λ scalar) ≠ 1 — exercises the λ-scale, mirrors agq_k1_reduces_to_laplace
        ws_a.params[1] = 0.2;
        ws_a.params[2] = 0.8;
        let params = ws_a.params.clone();
        ws_a.cluster_rows = Some(ClusterRowIndex::build(&ids, ws_a.groupings.n_primary));

        let mut ws_b = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], nagq);
        build_z(&mut ws_b, xf64.as_ref(), &ids, &[], 80);
        ws_b.cluster_rows = None;

        let da = glmm_agq_deviance(&params, &mut ws_a, xf64.as_ref(), &y, &ids, 80, nagq);
        let db = glmm_agq_deviance(&params, &mut ws_b, xf64.as_ref(), &y, &ids, 80, nagq);
        assert_eq!(da.to_bits(), db.to_bits(), "nagq={nagq}");
    }
}

/// Parallel cluster loop must equal the serial cluster-outer loop bitwise —
/// disjoint sum-slot writes make the result schedule-independent.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
#[test]
fn agq_parallel_bit_identical_to_serial() {
    let (xf64, y, ids) = glmm_intercept_dataset();
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
    for nagq in [3u8, 7, 11] {
        let mut ws_a = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], nagq);
        build_z(&mut ws_a, xf64.as_ref(), &ids, &[], 80);
        ws_a.params[0] = 0.5; // θ (Λ scalar) ≠ 1 — exercises the λ-scale, mirrors agq_k1_reduces_to_laplace
        ws_a.params[1] = 0.2;
        ws_a.params[2] = 0.8;
        let params = ws_a.params.clone();
        // Some(idx) here runs the rayon arm under --features parallel (this test is
        // itself feature-gated), so `da` below is the parallel result.
        ws_a.cluster_rows = Some(ClusterRowIndex::build(&ids, ws_a.groupings.n_primary));

        let mut ws_b = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], nagq);
        build_z(&mut ws_b, xf64.as_ref(), &ids, &[], 80);
        ws_b.cluster_rows = None; // knob-off path: always the original serial node-outer loop

        let da = glmm_agq_deviance(&params, &mut ws_a, xf64.as_ref(), &y, &ids, 80, nagq);
        let db = glmm_agq_deviance(&params, &mut ws_b, xf64.as_ref(), &y, &ids, 80, nagq);
        assert_eq!(da.to_bits(), db.to_bits(), "nagq={nagq}");
    }
}

// --- Vector AGQ (agq_deviance_vec) white-box invariants (spec Part 4 layer 1) --

/// q=2 vector-RE spec: intercept + slope on design col 1, single grouping
/// factor, no extras — the shape the widened gate routes to `agq_deviance_vec`.
fn slope1_spec() -> ModelSpec {
    ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    }
}

/// q=3 vector-RE spec: intercept + slopes on design cols 1,2 (pins the q≤3 cap).
fn slope2_spec() -> ModelSpec {
    ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![1, 2],
            extra_groupings: vec![],
        }),
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
/// the Laplace term exactly. Mirrors `agq_k1_reduces_to_laplace` one dimension up;
/// run at a non-identity Λ_p (correlated intercept/slope) so a missing `Λ_p·u_cj`
/// or `L_c⁻ᵀ` transform in the integrand would surface.
#[test]
fn agq_vec_k1_reduces_to_laplace_q2() {
    let (xf64, y, ids) = glmm_slope_noextra_dataset();
    let n = y.len();
    let cluster = slope1_spec();
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    build_z(&mut ws, xf64.as_ref(), &ids, &[], n);
    ws.params[0] = 0.5; // vech(Λ_p): [σ_int, cov, σ_slope] ≠ I
    ws.params[1] = 0.1;
    ws.params[2] = 0.4;
    ws.params[3] = 0.2; // β
    ws.params[4] = 0.8;
    let p = ws.params.clone();
    let lap = glmm_laplace_deviance(&p, &mut ws, xf64.as_ref(), &y, &ids, n);
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
    build_z(&mut ws, xf64.as_ref(), &ids, &[], n);
    // vech(Λ_p) column-major lower-tri of a 3×3, non-identity.
    let theta = [0.6, 0.1, 0.05, 0.4, 0.08, 0.3];
    ws.params[..6].copy_from_slice(&theta);
    ws.params[6] = 0.2; // β
    ws.params[7] = 0.8;
    ws.params[8] = -0.5;
    let p = ws.params.clone();
    let lap = glmm_laplace_deviance(&p, &mut ws, xf64.as_ref(), &y, &ids, n);
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
        build_z(&mut ws_a, xf64.as_ref(), &ids, &[], n);
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
        build_z(&mut ws_b, xf64.as_ref(), &ids, &[], n);
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
        build_z(&mut ws, xf64.as_ref(), &ids, &[], n);
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
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
    build_z(&mut ws, xf64.as_ref(), &ids, &[], 80);
    ws.params[0] = 0.0;
    ws.params[1] = beta[0];
    ws.params[2] = beta[1];
    let got = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf64.as_ref(), &y, &ids, 80);
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
fn build_z_width_general_populates_all_columns() {
    let (xf64, _y, ids, crossed_ids, cluster) = glmm_slope_crossed_dataset();
    let n = ids.len();
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    build_z(
        &mut ws,
        xf64.as_ref(),
        &ids,
        std::slice::from_ref(&crossed_ids),
        n,
    );
    let mut touched = vec![false; ws.k];
    #[allow(clippy::needless_range_loop)]
    for c in 0..ws.k {
        for i in 0..n {
            if ws.z[(i, c)] != 0.0 {
                touched[c] = true;
            }
        }
    }
    assert!(
        touched.iter().all(|&t| t),
        "every RE column must be populated — offset wiring"
    );

    // "some nonzero" can't catch an offset swap between two same-width
    // groupings (e.g. crossed level 1 landing in level 2's slot) — pin the
    // EXACT nonzero row set per column instead. Layout from `build_z`:
    // primary is per-level (`base = lvl*q`, col `2*lvl` = intercept, col
    // `2*lvl+1` = slope), n_prim=8 levels at q=2; the crossed block starts
    // at the absolute offset `n_prim*q=16`, one column per of its 4 levels.
    // `glmm_slope_crossed_dataset` sets `ids[i] = i % 8`, `crossed[i] = i % 4`.
    let n_prim = 8usize;
    let n_crossed = 4usize;
    for lvl in 0..n_prim {
        let expect_rows: Vec<usize> = (0..n).filter(|&i| i % n_prim == lvl).collect();
        for (kind, col) in [("intercept", 2 * lvl), ("slope", 2 * lvl + 1)] {
            let got_rows: Vec<usize> = (0..n).filter(|&i| ws.z[(i, col)] != 0.0).collect();
            assert_eq!(
                got_rows, expect_rows,
                "primary col {col} (level {lvl}, {kind}) nonzero rows"
            );
        }
    }
    for cc in 0..n_crossed {
        let col = n_prim * 2 + cc;
        let expect_rows: Vec<usize> = (0..n).filter(|&i| i % n_crossed == cc).collect();
        let got_rows: Vec<usize> = (0..n).filter(|&i| ws.z[(i, col)] != 0.0).collect();
        assert_eq!(
            got_rows, expect_rows,
            "crossed col {col} (level {cc}) nonzero rows"
        );
    }
}

/// `apply_lambda` must scale each extra grouping's columns by ITS OWN θ over
/// ITS OWN width, even when `extra_offsets` is non-monotonic. A
/// crossed-before-nested declaration places the nested block at the low
/// `prim_width` slot (small offset) AFTER the crossed block (large offset),
/// so a "span to the next declaration's offset" loop would leave the crossed
/// block unscaled and over-scale the nested block with the crossed θ.
/// Regression guard for that extra-offset span bug.
#[test]
fn apply_lambda_handles_nonmonotonic_extra_offsets() {
    let (n_prim, n_crossed, n_per_parent) = (4usize, 3usize, 2usize);
    // Declaration order [Crossed, Nested] ⇒ extra_offsets[0] (crossed, high) >
    // extra_offsets[1] (nested, low) — the non-monotonic precondition.
    let cluster = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_prim as u32,
            },
            slopes: vec![],
            extra_groupings: vec![
                Grouping {
                    relation: GroupingRelation::Crossed {
                        n_clusters: n_crossed as u32,
                    },
                    slopes: vec![],
                },
                Grouping {
                    relation: GroupingRelation::NestedWithin {
                        n_per_parent: n_per_parent as u32,
                    },
                    slopes: vec![],
                },
            ],
        }),
    };
    let n = 8usize;
    let ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
    let g = &ws.groupings;
    assert!(
        g.extra_offsets[0] > g.extra_offsets[1],
        "fixture must produce non-monotonic offsets, got {:?}",
        g.extra_offsets
    );
    // q_p = 1 ⇒ base_theta = 1; θ_crossed at idx 1, θ_nested at idx 2.
    let base_theta = 1usize;
    let (theta_crossed, theta_nested) = (2.0_f64, 3.0_f64);
    let mut params = vec![0.0; g.n_theta()];
    params[0] = 1.0; // primary Λ (irrelevant to the extra blocks)
    params[base_theta] = theta_crossed;
    params[base_theta + 1] = theta_nested;
    // z = ones everywhere so M directly reveals the per-column scale.
    let mut z = Mat::<f64>::zeros(n, g.k_total);
    for i in 0..n {
        for c in 0..g.k_total {
            z[(i, c)] = 1.0;
        }
    }
    let mut m = Mat::<f64>::zeros(n, g.k_total);
    let mut lam = vec![0.0; g.primary_q * g.primary_q];
    apply_lambda(g, &params, z.as_ref(), &mut m, &mut lam, n);
    let coff = g.extra_offsets[0];
    for c in coff..coff + n_crossed {
        assert!(
            (m[(0, c)] - theta_crossed).abs() < 1e-12,
            "crossed col {c} = {}, want {theta_crossed}",
            m[(0, c)]
        );
    }
    let noff = g.extra_offsets[1];
    for c in noff..noff + n_prim * n_per_parent {
        assert!(
            (m[(0, c)] - theta_nested).abs() < 1e-12,
            "nested col {c} = {}, want {theta_nested}",
            m[(0, c)]
        );
    }
}

#[test]
fn fit_glmm_recovers_direction_and_finite_inference() {
    let (xf64, y, ids) = glmm_intercept_dataset();
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
    build_z(&mut ws, xf64.as_ref(), &ids, &[], 80);
    let targets = [1u32];
    let beta_truth = [0.2_f64, 0.8];
    let fit = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        &y,
        &ids,
        &targets,
        Some(&[0.5]),
        &beta_truth,
        80,
        WaldSe::Rx,
    );
    assert!(fit.converged);
    assert!(
        ws.betas[1] > 0.3,
        "β̂₁ should be positive (truth 0.8), got {}",
        ws.betas[1]
    );
    // Strictly positive, not merely >= 0.0 (a squared quantity is trivially
    // non-negative): a zero Wald statistic means the SE diverged or β̂
    // collapsed. Direction is guarded by the β̂₁ > 0.3 check above; this is
    // one fixed binary dataset (n=80), not a power sim, so the single draw
    // need not clear significance.
    assert!(
        ws.t_sq[1].is_finite() && ws.t_sq[1] > 0.0,
        "t²[1] = {} must be finite and strictly positive",
        ws.t_sq[1]
    );
    assert!(fit.tau_squared_hat.is_finite() && fit.tau_squared_hat >= 0.0);
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
    #[allow(dead_code)]
    vcov_rx: Vec<Vec<f64>>,
}

fn load_hessian_fixture() -> HessianFixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/glmm_hessian_vcov.json"
    );
    let s = std::fs::read_to_string(path).expect("read hessian fixture");
    serde_json::from_str(&s).expect("parse hessian fixture")
}

/// FD-Hessian fixed-effect covariance matches lme4 `vcov(use.hessian = TRUE)`
/// on the committed n=96 / 12-cluster `y ~ x1 + (1|grp)` glmer fit. Runs OUR
/// `fit_glmm` to convergence (the production code path) and takes the FD Hessian
/// at OUR own θ̂/β̂ — NOT at lme4's fixture params. Pins both the kernel
/// convention and the load-bearing factor of 2 (deviance = −2logL ⇒
/// cov = 2·inv(H_dev)).
///
/// Why fit first (test rigor — GATE-1 I1): the FD Hessian's β-block invariance
/// to the θ↔β cross-curvature only holds AT a θ-stationary point. Evaluating at
/// our own converged θ̂ both (a) reflects production and (b) lets us assert our
/// solver agrees with lme4's θ̂ — proving any residual vcov gap is genuine
/// curvature, not an off-stationarity artifact. Measured: our θ̂ matches lme4's
/// to ~0.07% and β̂ to ~0.1%; against the artifact-free fixture the vcov gap is
/// ~3.4e-7 (see the band comment at the assertion).
#[test]
fn fd_hessian_cov_matches_glmer_use_hessian_true() {
    let fx = load_hessian_fixture();
    let n = fx.n;
    let p = fx.beta.len();
    let n_clusters = fx.cluster_ids.iter().max().unwrap() + 1;
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters });
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[], 1);
    let mut xf64 = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            xf64[(i, j)] = fx.x[i][j];
        }
    }
    let y = fx.y.clone();
    let ids = fx.cluster_ids.clone();
    build_z(&mut ws, xf64.as_ref(), &ids, &[], n);

    // RX/Schur machinery-exactness check, evaluated at lme4's EXACT θ̂/β̂ so the
    // comparison is point-matched: rx_cov_into reproduces lme4 vcov(use.hessian
    // =FALSE) to ~3.6e-6 (this path never goes through numDeriv — it inverts the
    // closed-form β information). Pins that the PIRLS / W̃ / Schur machinery the
    // FD Hessian shares is exact, isolating any vcov_hessian gap to the FD ↔
    // numDeriv comparison. Requires a converged central PIRLS at those params.
    ws.params[0] = fx.theta;
    for j in 0..p {
        ws.params[1 + j] = fx.beta[j];
    }
    let _ = laplace_deviance_at(&mut ws, xf64.as_ref(), &y, &ids, n);
    let mut rx = Mat::<f64>::zeros(p, p);
    assert!(rx_cov_into(&mut ws, xf64.as_ref(), &ids, p, n, &mut rx));
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
        &y,
        &ids,
        &[1u32],
        None,
        &vec![0.0; p],
        n,
        WaldSe::Rx,
    );
    assert!(fit.converged, "fit_glmm must converge on the fixture");
    // Our solver vs lme4: θ̂ to ~0.07%, β̂ to ~0.1% (measured). Proves the two
    // optimisers land on the same stationary point, so the FD Hessian below is
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
    let status = fd_hessian_cov(&mut ws, xf64.as_ref(), &y, &ids, p, n, &mut cov);
    assert_eq!(status, FdHessianStatus::Ok);
    // ws.params restored to OUR converged snapshot on return.
    assert!((ws.params[0] - our_theta).abs() < 1e-15);
    // Achieved FD-vs-lme4(use.hessian=TRUE) band at OUR converged fit: worst
    // entry gap ~3.4e-7. The fixture is generated artifact-free
    // (gen_glmm_hessian_vcov.R, glmer tolPwrss = 1e-13 — at lme4's default
    // 1e-7 its ldL2 uses working weights one PIRLS iteration behind the mode
    // and vcov(use.hessian=TRUE) carried ~1.7e-3 of spurious θ/θβ curvature,
    // which this band used to absorb; see fd_hessian_cov's doc comment) and
    // our FD runs PIRLS at the tight PIRLS_TOL_REL_FD, so what remains is the
    // two solvers' θ̂ offset plus FD truncation. tol = achieved band + ~30×
    // margin (cross-platform FP headroom); it pins the convention + the
    // load-bearing factor of 2. Mirrors hessian_mode_t_sq_uses_fd_hessian_cov's
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

/// Run `fd_hessian_cov` twice on the same converged workspace — serial
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
    p: usize,
    n: usize,
) {
    let mut cov_s = Mat::<f64>::zeros(p, p);
    ws.parallel_inner = false;
    let st_s = fd_hessian_cov(ws, x, y, ids, p, n, &mut cov_s);
    let tse_s = ws.theta_se.clone();

    let mut cov_p = Mat::<f64>::zeros(p, p);
    ws.parallel_inner = true;
    let st_p = fd_hessian_cov(ws, x, y, ids, p, n, &mut cov_p);
    let tse_p = ws.theta_se.clone();

    assert_eq!(st_s, st_p, "FdHessianStatus differs serial vs parallel");
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

/// The parallel FD-Hessian grid must reproduce the serial one BITWISE on both the
/// no-extras blocked path and the structured crossed path. Every grid cell is a
/// pure function of the frozen (fd_saved, fd_steps, u_seed) seed, so per-thread
/// workspaces change nothing — a mismatch is an `fd_worker_ws` field-liveness bug,
/// not floating-point noise.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
#[test]
fn fd_hessian_parallel_bit_identical_to_serial() {
    // Case A: no-extras (1|grp) fixture — the blocked FD path.
    {
        let fx = load_hessian_fixture();
        let n = fx.n;
        let p = fx.beta.len();
        let n_clusters = fx.cluster_ids.iter().max().unwrap() + 1;
        let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters });
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[], 1);
        let mut xf64 = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xf64[(i, j)] = fx.x[i][j];
            }
        }
        let y = fx.y.clone();
        let ids = fx.cluster_ids.clone();
        build_z(&mut ws, xf64.as_ref(), &ids, &[], n);
        let fit = fit_glmm(
            &mut ws,
            xf64.as_ref(),
            &y,
            &ids,
            &[1u32],
            None,
            &vec![0.0; p],
            n,
            WaldSe::Rx,
        );
        assert!(fit.converged, "no-extras fixture must converge");
        assert_fd_hessian_serial_eq_parallel(&mut ws, xf64.as_ref(), &y, &ids, p, n);
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
        build_z(&mut ws, x.as_ref(), &ids.primary, &ids.extra, n);
        ws.structured_schur = StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n);
        let fit = fit_glmm(
            &mut ws,
            x.as_ref(),
            &y,
            &ids.primary,
            &[0u32, 1, 2, 3],
            None,
            &vec![0.0; p],
            n,
            WaldSe::Rx,
        );
        assert!(
            fit.converged,
            "grouseticks structured fixture must converge"
        );
        assert_fd_hessian_serial_eq_parallel(&mut ws, x.as_ref(), &y, &ids.primary, p, n);
    }
}

/// `fit_glmm(.., WaldSe::Hessian)` sources the per-fit marginal SE from
/// `fd_hessian_cov` (glmer `use.hessian = TRUE`) instead of the Schur
/// forward-solve: on the committed fixture the x1 Hessian SE EXCEEDS the Rx
/// SE and matches the fixture's `vcov_hessian` diagonal. `WaldSe::Rx` keeps
/// the unchanged Schur path. Pins that the dispatch reads the FD-Hessian
/// covariance into `ws.var_diag` end-to-end (not just the standalone kernel).
#[test]
fn hessian_mode_t_sq_uses_fd_hessian_cov() {
    let fx = load_hessian_fixture();
    let n = fx.n;
    let p = fx.beta.len();
    let n_clusters = fx.cluster_ids.iter().max().unwrap() + 1;
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters });
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
    build_z(&mut ws_h, xf64.as_ref(), &ids, &[], n);
    let fit_h = fit_glmm(
        &mut ws_h,
        xf64.as_ref(),
        &y,
        &ids,
        &[1u32],
        None,
        &vec![0.0; p],
        n,
        WaldSe::Hessian,
    );
    assert!(fit_h.converged, "hessian-mode fit must converge");
    let se_h = ws_h.var_diag[t1].sqrt();

    let mut ws_rx = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[], 1);
    build_z(&mut ws_rx, xf64.as_ref(), &ids, &[], n);
    let fit_rx = fit_glmm(
        &mut ws_rx,
        xf64.as_ref(),
        &y,
        &ids,
        &[1u32],
        None,
        &vec![0.0; p],
        n,
        WaldSe::Rx,
    );
    assert!(fit_rx.converged, "rx-mode fit must converge");
    let se_rx = ws_rx.var_diag[t1].sqrt();

    assert!(se_h > se_rx, "hessian SE {se_h} must exceed rx SE {se_rx}");
    // Match the FD-Hessian kernel band (1e-5 ABSOLUTE on the covariance entries,
    // achieved ~3.4e-7 against the artifact-free fixture) on the variance the
    // dispatch wrote into ws.var_diag — the band source is
    // `fd_hessian_cov_matches_glmer_use_hessian_true`: change together.
    let want_var = fx.vcov_hessian[t1][t1];
    assert!(
        (ws_h.var_diag[t1] - want_var).abs() < 1e-5,
        "hessian var {} must match fixture vcov_hessian diag {want_var}",
        ws_h.var_diag[t1]
    );
}

/// A non-PD joint (θ,β) Hessian makes `fd_hessian_cov` fall back to the
/// RX/Schur covariance (`NonPdFellBackToRx`) with the produced cov equal to
/// `rx_cov_into`'s, while the β-only Schur stays PD.
///
/// Where non-PD genuinely lives: NOT at the θ→0 floor (the intercept-only
/// Laplace deviance is EVEN in θ, so its θ-curvature there is structurally
/// POSITIVE), and — since the FD evals run at the tight `PIRLS_TOL_REL_FD` —
/// no longer at a converged high-variance fit either: an interior minimum has
/// PSD curvature by definition, and the non-PD this fixture used to produce
/// AT θ̂ was loose-tol FD noise on that near-flat θ direction (this very
/// fixture flipped Ok when the tight tol landed — the noise the tol removes
/// is exactly what made the LLT fail). The deviance is genuinely concave
/// BEYOND the minimum along θ: at large θ the Laplace `log|A|` term grows
/// like `2s·log θ` (concave) while the data deviance saturates and the
/// penalty vanishes, so H_θθ < 0 there — tolerance-independent. The test
/// therefore evaluates the FD Hessian at θ pushed well past the converged
/// θ̂ of a high-ICC fit; production reaches this fallback only through such
/// degenerate geometry, kept as a defensive guard. The β fixed-effect
/// information (`rx_cov_into`'s Schur, computed from W̃ at the conditional
/// modes — a different matrix than the FD Hessian) stays PD throughout. Per
/// `[[faer-llt-rank-deficiency-grey-zone]]` the β design is well-conditioned
/// (intercept + a within-cluster-varying continuous x1, NOT separable, NOT a
/// duplicate column) — only the θ/variance direction is degenerate.
///
/// Confirmed branch (asserted below): the FULLY-ASSEMBLED joint Hessian is
/// finite yet fails LLT — i.e. the non-PD-Hessian fallback, NOT the
/// non-finite-deviance fallback (this kernel's log1pexp deviance + `+I` ridge
/// keep every perturbed deviance finite, so the non-finite branch is effectively
/// unreachable with well-posed data).
#[test]
fn fd_hessian_non_pd_falls_back_to_rx_and_counts() {
    let (n, nc) = (80usize, 8usize);
    let per = n / nc;
    // Strong between-cluster intercept offsets (SD ≈ 2.5) ⇒ high ICC ⇒ the fit
    // converges to a LARGE τ̂² (θ̂ ≳ 2), the concave region where the joint
    // Hessian goes non-PD. Noisy (logistic-sampled) labels — no separation.
    let mut st = 4242u64;
    let mut xf64 = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = i / per; // block layout
        ids[i] = c as u32;
        let u_c = 10.0 * (2.0 * (c as f64) / ((nc - 1) as f64) - 1.0); // ∈ [-10, 10]
        let x1 = lcg(&mut st); // within-cluster-varying ⇒ β slope identified
        xf64[(i, 0)] = 1.0;
        xf64[(i, 1)] = x1;
        let eta = 0.0 + 0.8 * x1 + u_c;
        let pr = 1.0 / (1.0 + (-eta).exp());
        y[i] = if lcg(&mut st) + 0.5 < pr { 1.0 } else { 0.0 };
    }
    let p = 2usize;
    let cluster = logit_intercept_spec(Sizing::FixedClusters {
        n_clusters: nc as u32,
    });
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &cluster, n, &[], 1);
    build_z(&mut ws, xf64.as_ref(), &ids, &[], n);

    // Converge a fit (kernel precondition for fd_hessian_cov).
    let fit = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        &y,
        &ids,
        &[1u32],
        None,
        &[0.0, 0.8],
        n,
        WaldSe::Rx,
    );
    assert!(fit.converged, "fixture fit must converge");
    // High-ICC data converges to a large θ̂ (≈ 30 here) — the flank whose
    // far side is the concave region.
    assert!(
        ws.params[0] > 5.0,
        "fit must reach the high-variance regime (θ̂ = {})",
        ws.params[0]
    );
    // Step past the minimum into the genuinely concave stretch (H_θθ < 0 from
    // the `2s·log θ` term — see the doc comment): at 5·θ̂ the LLT failure is
    // deterministic, not tolerance noise. fd_hessian_cov evaluates AT ws.params
    // (its f0 re-solves PIRLS there), so overwriting θ is the supported way to
    // probe the fallback; β stays at the fitted values.
    ws.params[0] *= 5.0;

    let mut cov = Mat::<f64>::zeros(p, p);
    let status = fd_hessian_cov(&mut ws, xf64.as_ref(), &y, &ids, p, n, &mut cov);
    assert_eq!(
        status,
        FdHessianStatus::NonPdFellBackToRx,
        "high-variance joint Hessian must be non-PD ⇒ RX fallback"
    );
    // Confirm this is the non-PD-Hessian branch, not the non-finite-deviance
    // branch: the joint Hessian was FULLY assembled (every perturbed deviance
    // finite) AND is non-PD (LLT errors). The fallback macro re-evals only the
    // central deviance, leaving ws.hess_scratch holding the complete Hessian.
    let m = ws.params.len();
    for i in 0..m {
        for j in 0..m {
            assert!(
                ws.hess_scratch[(i, j)].is_finite(),
                "assembled Hessian must be finite (non-finite branch NOT taken): H[{i}][{j}]"
            );
        }
    }
    assert!(
        ws.hess_scratch.as_ref().llt(faer::Side::Lower).is_err(),
        "assembled joint Hessian must be non-PD (LLT must fail)"
    );

    // Fallback cov must equal the RX/Schur cov (a real inverse, not NaN). The
    // central deviance was re-evaluated inside fd_hessian_cov's fallback, so
    // the β-information factors are valid for rx_cov_into here too.
    let mut rx = Mat::<f64>::zeros(p, p);
    assert!(
        rx_cov_into(&mut ws, xf64.as_ref(), &ids, p, n, &mut rx),
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

    let cluster = logit_intercept_spec(Sizing::FixedClusters {
        n_clusters: nc as u32,
    });
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
    build_z(&mut ws, x.as_ref(), &ids, &[], n);
    let fit = fit_glmm(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &[1],
        Some(&[0.05]),
        &[0.1, 0.7],
        n,
        WaldSe::Rx,
    );
    assert!(fit.converged);
    assert!(
        (ws.betas[1] - irls.0[1]).abs() < 1e-2,
        "β̂₁ glmm {} vs irls {}",
        ws.betas[1],
        irls.0[1]
    );
    // GLMM z² is indexed by coefficient (ws.t_sq[1] = slope = target 1); IRLS
    // z² is indexed by target position (irls.1[0] = first target = slope).
    assert!(
        (ws.t_sq[1].sqrt() - irls.1[0].sqrt()).abs() < 5e-2,
        "z glmm {} vs irls {}",
        ws.t_sq[1].sqrt(),
        irls.1[0].sqrt()
    );
}

/// fit_glmm on the q_p=2 slope + crossed fixture.
#[test]
fn fit_glmm_width_general_slope_and_crossed() {
    let (xf64, y, ids, crossed_ids, cluster) = glmm_slope_crossed_dataset();
    let n = y.len();
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    build_z(
        &mut ws,
        xf64.as_ref(),
        &ids,
        std::slice::from_ref(&crossed_ids),
        n,
    );
    let fit = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        &y,
        &ids,
        &[1],
        None,
        &[0.2, 0.8],
        n,
        WaldSe::Rx,
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
        ws.t_sq[1].is_finite() && ws.t_sq[1] > 3.84,
        "z² must clear the α=0.05 bar (3.84), got {}",
        ws.t_sq[1]
    );
    assert!(
        fit.tau_squared_hat.is_finite() && (0.0..5.0).contains(&fit.tau_squared_hat),
        "τ̂² {}",
        fit.tau_squared_hat
    );
}

/// Warm-path zero-alloc lock — `#[ignore]` because `dhat::Profiler` measures
/// process-wide allocations and concurrent tests contaminate the count. faer's
/// rayon parallelism also jitters the count run-to-run on a multi-core box, so
/// pin it to one thread for a deterministic measurement:
/// Run: `RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests fit_glmm_warm_path_bounded_alloc -- --ignored --test-threads=1`
/// (`alloc-tests` installs the dhat global allocator the profiler requires.)
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn fit_glmm_warm_path_bounded_alloc() {
    let (xf64, y, ids) = glmm_intercept_dataset();
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
    build_z(&mut ws, xf64.as_ref(), &ids, &[], 80);
    let _ = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        &y,
        &ids,
        &[1],
        Some(&[0.5]),
        &[0.2, 0.8],
        80,
        WaldSe::Rx,
    ); // warmup
    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..20 {
        let _ = fit_glmm(
            &mut ws,
            xf64.as_ref(),
            &y,
            &ids,
            &[1],
            Some(&[0.5]),
            &[0.2, 0.8],
            80,
            WaldSe::Rx,
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
/// `fit_glmm_warm_path_bounded_alloc`. There was no such gate before — the dense
/// crossed/nested path allocated inside faer's per-eval `llt`. The structured
/// path replaces that with `glmm_block_chol`/`glmm_block_solve` on the
/// pre-allocated `core_blocks`/`schur_blk`/`coupling` + stack-sized per-block
/// scratch, so the only per-eval blocks left are the joint [θ|β] BOBYQA's own
/// scratch (M is built into the pre-allocated `ws.m`). `#[ignore]` + one-thread
/// for the same reasons as the no-extras gate. Run:
/// `RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests fit_glmm_structured_warm_path_bounded_alloc -- --ignored --test-threads=1`
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn fit_glmm_structured_warm_path_bounded_alloc() {
    // crossed_nested q_p=1: 8 core blocks of 3×3 + a 6×6 Schur (the richest
    // structured shape; exercises both the block factor and the Schur).
    let (xf64, y, ids, extra_ids, cluster) = glmm_extras_q1_dataset(2, 6);
    let n = y.len();
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
    assert!(ws.groupings.structured_extras_eligible());
    build_z(&mut ws, xf64.as_ref(), &ids, &extra_ids, n);
    let theta = [0.5_f64, 0.4, 0.45];
    let _ = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        &y,
        &ids,
        &[1],
        Some(&theta),
        &[0.2, 0.8],
        n,
        WaldSe::Rx,
    ); // warmup
    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..20 {
        let _ = fit_glmm(
            &mut ws,
            xf64.as_ref(),
            &y,
            &ids,
            &[1],
            Some(&theta),
            &[0.2, 0.8],
            n,
            WaldSe::Rx,
        );
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);
    // Measured floor on this machine (RAYON_NUM_THREADS=1): exactly 124 blocks
    // across 20 fits — the SAME count as the no-extras blocked gate. The
    // structured path factors/solves on pre-allocated buffers
    // (core_blocks/schur_blk/coupling) with stack-sized per-block scratch, so it
    // adds NO per-fit faer `llt` allocation; what remains is purely the joint
    // [θ|β] BOBYQA's own per-eval scratch. A change here flags a new allocation
    // or a shifted eval trajectory; do not relax — find the alloc.
    const BOUND: u64 = 124;
    assert!(
        stats.total_blocks <= BOUND,
        "structured warm-path alloc regressed: {} blocks across 20 fits (BOUND = {})",
        stats.total_blocks,
        BOUND
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
fn glmm_slope_noextra_dataset_layout(contiguous: bool) -> (Mat<f64>, Vec<f64>, Vec<u32>) {
    let (n, nc) = (96usize, 8usize);
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
    glmm_slope_noextra_dataset_layout(false)
}

/// The blocked intercept path computes the same Laplace deviance as the
/// independent brute force (reorders accumulation ⇒ FP-close, not bit-equal).
/// After Phase 2 this routes through `pirls_solve_blocked` (extra_offsets empty).
#[test]
fn blocked_laplace_matches_brute_force_intercept() {
    let (xf64, y, ids) = glmm_intercept_dataset();
    let beta = [0.2_f64, 0.8];
    let want = brute_force_intercept_laplace(0.5, &beta, &xf64, &y, &ids, 8);
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
    assert!(
        ws.groupings.extra_offsets.is_empty(),
        "fixture must route blocked"
    );
    build_z(&mut ws, xf64.as_ref(), &ids, &[], 80);
    ws.params[0] = 0.5;
    ws.params[1] = beta[0];
    ws.params[2] = beta[1];
    let got = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf64.as_ref(), &y, &ids, 80);
    assert!(
        (got - want).abs() < 1e-6,
        "blocked laplace: got {got}, want {want}"
    );
}

/// `FixedSize`/DGEN-FS production layout (contiguous id runs) through the
/// blocked intercept path. The round-robin fixtures only cover
/// `FixedClusters`-style interleaved ids; a layout-sensitive rewrite of
/// the blocked row loops behaves differently per layout, so the
/// brute-force oracle must hold here too.
#[test]
fn blocked_laplace_matches_brute_force_intercept_contiguous() {
    let (xf64, y, ids) = glmm_intercept_dataset_layout(true);
    let beta = [0.2_f64, 0.8];
    let want = brute_force_intercept_laplace(0.5, &beta, &xf64, &y, &ids, 8);
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
    assert!(
        ws.groupings.extra_offsets.is_empty(),
        "fixture must route blocked"
    );
    build_z(&mut ws, xf64.as_ref(), &ids, &[], 80);
    ws.params[0] = 0.5;
    ws.params[1] = beta[0];
    ws.params[2] = beta[1];
    let got = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf64.as_ref(), &y, &ids, 80);
    assert!(
        (got - want).abs() < 1e-6,
        "blocked laplace (contiguous): got {got}, want {want}"
    );
}

/// Blocked deviance == dense deviance on the no-extras slope fixture, to FP
/// error. Drives both kernels directly on the same M / Λ_p (dev-time
/// equivalence smoke test; a wild divergence is a coding bug, per the spec).
#[test]
fn blocked_pirls_matches_dense_slope_noextra() {
    let (xf64, y, ids) = glmm_slope_noextra_dataset();
    let n = y.len();
    // slope on design col 1 ⇒ slope_cols = &[1]; q_p = 2.
    let cluster = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    assert!(ws.groupings.extra_offsets.is_empty());
    build_z(&mut ws, xf64.as_ref(), &ids, &[], n);
    let theta = [0.5_f64, 0.1, 0.4]; // vech(Λ_p): [σ_int, cov, σ_slope]
    let mut beta = [0.2_f64, 0.8];
    let (k, p, nt) = (ws.k, ws.p, ws.n_theta);

    // Dense reference: apply_lambda → pirls_solve.
    let mut params = vec![0.0; nt + p];
    params[..nt].copy_from_slice(&theta);
    params[nt..].copy_from_slice(&beta);
    let GlmmWorkspace {
        groupings,
        z,
        m,
        lam,
        prior_w,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        mu,
        wm,
        wx,
        a,
        a_chol,
        a_llt_mem,
        a_rhs,
        ..
    } = &mut ws;
    apply_lambda(groupings, &params, z.as_ref(), m, lam, n);
    let dense = pirls_solve(
        crate::Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        f64::NAN,
        k,
        p,
        m.as_ref(),
        xf64.as_ref(),
        &y,
        &prior_w[..n],
        false,
        &mut params[nt..],
        BetaStep::Fixed,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        mu,
        wm,
        wx,
        a,
        a_chol,
        a_rhs,
        a_llt_mem,
        None, // offset
        None,
        n,
    );

    // Blocked: primary_lambda → pirls_solve_blocked, fresh scratch.
    let mut ws2 = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    build_z(&mut ws2, xf64.as_ref(), &ids, &[], n);
    crate::lmm::primary_lambda(&theta, ws2.groupings.primary_q, &mut ws2.lam);
    fill_z_f64(&ws2.groupings, xf64.as_ref(), &mut ws2.z_buf, n);
    let GlmmWorkspace {
        groupings,
        lam,
        z_buf,
        m_buf,
        prior_w,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        a_blocks,
        a_rhs,
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
        lam,
        z_buf,
        m_buf,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        a_blocks,
        a_rhs,
        &mut wx_scratch,
        None, // offset
        None,
        n,
    );

    // Dense and blocked now share the same lme4 step-halving backtrack and
    // today's mixed stopping rule, so the paths agree to FP error again —
    // re-tightened from an earlier interim relaxation back to the original
    // dev/pen/logdet 1e-9, u 1e-7.
    assert_eq!(dense.3, blocked.3, "convergence flag");
    assert!(
        (dense.0 - blocked.0).abs() < 1e-9,
        "dev: dense {} blocked {}",
        dense.0,
        blocked.0
    );
    assert!(
        (dense.1 - blocked.1).abs() < 1e-9,
        "pen: dense {} blocked {}",
        dense.1,
        blocked.1
    );
    assert!(
        (dense.2 - blocked.2).abs() < 1e-9,
        "logdet: dense {} blocked {}",
        dense.2,
        blocked.2
    );
    for c in 0..k {
        assert!(
            (ws.u[c] - ws2.u[c]).abs() < 1e-7,
            "u[{c}]: dense {} blocked {}",
            ws.u[c],
            ws2.u[c]
        );
    }
}

/// Blocked == dense on the cluster-contiguous (`FixedSize`/DGEN-FS) id
/// layout — the q_p=2 twin of `blocked_pirls_matches_dense_slope_noextra`,
/// which only exercises round-robin ids. Same FP bands; if a reordered
/// accumulation ever moves them, re-derive the band with documentation —
/// never widen silently.
#[test]
fn blocked_pirls_matches_dense_slope_contiguous() {
    let (xf64, y, ids) = glmm_slope_noextra_dataset_layout(true);
    let n = y.len();
    // slope on design col 1 ⇒ slope_cols = &[1]; q_p = 2.
    let cluster = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    assert!(ws.groupings.extra_offsets.is_empty());
    build_z(&mut ws, xf64.as_ref(), &ids, &[], n);
    let theta = [0.5_f64, 0.1, 0.4]; // vech(Λ_p): [σ_int, cov, σ_slope]
    let mut beta = [0.2_f64, 0.8];
    let (k, p, nt) = (ws.k, ws.p, ws.n_theta);

    // Dense reference: apply_lambda → pirls_solve.
    let mut params = vec![0.0; nt + p];
    params[..nt].copy_from_slice(&theta);
    params[nt..].copy_from_slice(&beta);
    let GlmmWorkspace {
        groupings,
        z,
        m,
        lam,
        prior_w,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        mu,
        wm,
        wx,
        a,
        a_chol,
        a_llt_mem,
        a_rhs,
        ..
    } = &mut ws;
    apply_lambda(groupings, &params, z.as_ref(), m, lam, n);
    let dense = pirls_solve(
        crate::Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        f64::NAN,
        k,
        p,
        m.as_ref(),
        xf64.as_ref(),
        &y,
        &prior_w[..n],
        false,
        &mut params[nt..],
        BetaStep::Fixed,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        mu,
        wm,
        wx,
        a,
        a_chol,
        a_rhs,
        a_llt_mem,
        None, // offset
        None,
        n,
    );

    // Blocked: primary_lambda → pirls_solve_blocked, fresh scratch.
    let mut ws2 = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    build_z(&mut ws2, xf64.as_ref(), &ids, &[], n);
    crate::lmm::primary_lambda(&theta, ws2.groupings.primary_q, &mut ws2.lam);
    fill_z_f64(&ws2.groupings, xf64.as_ref(), &mut ws2.z_buf, n);
    let GlmmWorkspace {
        groupings,
        lam,
        z_buf,
        m_buf,
        prior_w,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        a_blocks,
        a_rhs,
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
        lam,
        z_buf,
        m_buf,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        a_blocks,
        a_rhs,
        &mut wx_scratch,
        None, // offset
        None,
        n,
    );

    // Re-tightened to 1e-9 (u 1e-7) — dense and blocked share the same
    // step-halving stopping rule now; see
    // `blocked_pirls_matches_dense_slope_noextra` for the full rationale.
    assert_eq!(dense.3, blocked.3, "convergence flag");
    assert!(
        (dense.0 - blocked.0).abs() < 1e-9,
        "dev: dense {} blocked {}",
        dense.0,
        blocked.0
    );
    assert!(
        (dense.1 - blocked.1).abs() < 1e-9,
        "pen: dense {} blocked {}",
        dense.1,
        blocked.1
    );
    assert!(
        (dense.2 - blocked.2).abs() < 1e-9,
        "logdet: dense {} blocked {}",
        dense.2,
        blocked.2
    );
    for c in 0..k {
        assert!(
            (ws.u[c] - ws2.u[c]).abs() < 1e-7,
            "u[{c}]: dense {} blocked {}",
            ws.u[c],
            ws2.u[c]
        );
    }
}

/// Blocked inference (β̂, Var(β̂)_jj, z²) matches the dense inference on the
/// no-extras slope fixture, to tight FP tolerance — the inference reorder is
/// the same estimator. Runs the full `fit_glmm` (blocked) and an explicit
/// dense recomputation of the Schur Var on the same converged state.
#[test]
fn blocked_inference_matches_dense_slope_noextra() {
    let (xf64, y, ids) = glmm_slope_noextra_dataset();
    let n = y.len();
    let cluster = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    build_z(&mut ws, xf64.as_ref(), &ids, &[], n);
    let fit = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        &y,
        &ids,
        &[1],
        Some(&[0.5, 0.1, 0.4]),
        &[0.2, 0.8],
        n,
        WaldSe::Rx,
    );
    assert!(fit.converged);
    // Recompute Var(β̂) densely from the converged ws.{w, lam, params} via a
    // freshly-built M and ws.a, and compare β̂ / var_diag / t_sq.
    let (k, p, nt) = (ws.k, ws.p, ws.n_theta);
    let beta_blocked: Vec<f64> = ws.betas[..p].to_vec();
    let var_blocked = ws.var_diag[1];
    let tsq_blocked = ws.t_sq[1];
    // Dense M = ZΛ̂ at the converged θ̂.
    crate::lmm::primary_lambda(&ws.params[..nt], ws.groupings.primary_q, &mut ws.lam);
    {
        let GlmmWorkspace {
            groupings,
            params,
            z,
            m,
            lam,
            ..
        } = &mut ws;
        apply_lambda(groupings, &params[..], z.as_ref(), m, lam, n);
    }
    // X'W̃X, X'W̃M, A = M'W̃M + I (dense), Schur, Var(β̂)_11.
    let mut xtwx = Mat::<f64>::zeros(p, p);
    for r in 0..p {
        for c in 0..p {
            let mut sm = 0.0;
            for i in 0..n {
                sm += xf64[(i, r)] * ws.w[i] * xf64[(i, c)];
            }
            xtwx[(r, c)] = sm;
        }
    }
    let mut xtwm = Mat::<f64>::zeros(p, k);
    for r in 0..p {
        for c in 0..k {
            let mut sm = 0.0;
            for i in 0..n {
                sm += xf64[(i, r)] * ws.w[i] * ws.m[(i, c)];
            }
            xtwm[(r, c)] = sm;
        }
    }
    let mut a = Mat::<f64>::zeros(k, k);
    for r in 0..k {
        for c in 0..k {
            let mut sm = if r == c { 1.0 } else { 0.0 };
            for i in 0..n {
                sm += ws.m[(i, r)] * ws.w[i] * ws.m[(i, c)];
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
    let var_dense: f64 = fwd.iter().map(|v| v * v).sum();
    assert!(
        (var_blocked - var_dense).abs() < 1e-8,
        "var: blocked {var_blocked} dense {var_dense}"
    );
    let tsq_dense = beta_blocked[1] * beta_blocked[1] / var_dense;
    assert!(
        (tsq_blocked - tsq_dense).abs() < 1e-6,
        "z²: blocked {tsq_blocked} dense {tsq_dense}"
    );
}

/// Within-fit û warm-start MUST reset per fit. A reused-workspace re-fit must
/// match a fresh-workspace (canonically cold) fit BIT-FOR-BIT. Carrying the
/// incumbent across fits is the rejected cross-sim warm-start that breaks
/// merge / same-seed reproducibility. This is the per-fit-reset guard.
#[test]
fn warm_start_is_per_fit_deterministic() {
    let (xf64, y, ids) = glmm_intercept_dataset();
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
    let mk = || {
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
        build_z(&mut ws, xf64.as_ref(), &ids, &[], 80);
        ws
    };
    // Canonical: a fresh workspace's first (cold) fit.
    let mut ws_ref = mk();
    let _ = fit_glmm(
        &mut ws_ref,
        xf64.as_ref(),
        &y,
        &ids,
        &[1],
        Some(&[0.5]),
        &[0.2, 0.8],
        80,
        WaldSe::Rx,
    );
    let ref_beta = ws_ref.betas[1].to_bits();
    // Reused workspace: a throwaway fit pollutes u_seed, then the measured fit
    // must match the canonical cold result bit-for-bit (only if u_seed resets).
    let mut ws = mk();
    let _ = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        &y,
        &ids,
        &[1],
        Some(&[0.5]),
        &[0.2, 0.8],
        80,
        WaldSe::Rx,
    );
    let _ = fit_glmm(
        &mut ws,
        xf64.as_ref(),
        &y,
        &ids,
        &[1],
        Some(&[0.5]),
        &[0.2, 0.8],
        80,
        WaldSe::Rx,
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
    let cluster = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, 80, &[], 1);
    build_z(&mut ws, xf64.as_ref(), &ids, &[], 80);
    ws.params[0] = 0.5;
    ws.params[1] = 0.2;
    ws.params[2] = 0.8;
    for v in ws.u.iter_mut() {
        *v = 0.0;
    }
    let cold = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf64.as_ref(), &y, &ids, 80);
    for (c, v) in ws.u.iter_mut().enumerate() {
        *v = 0.05 * (c as f64 - 4.0);
    }
    let warm = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf64.as_ref(), &y, &ids, 80);
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
#[allow(clippy::type_complexity)]
fn glmm_extras_q1_dataset(
    np: usize,
    n_crossed: usize,
) -> (Mat<f64>, Vec<f64>, Vec<u32>, Vec<Vec<u32>>, ModelSpec) {
    let (n, n_prim) = (96usize, 8usize);
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
    let mut extra_ids = Vec::new();
    if np > 0 {
        extra_groupings.push(Grouping {
            relation: GroupingRelation::NestedWithin {
                n_per_parent: np as u32,
            },
            slopes: vec![],
        });
        extra_ids.push(nested);
    }
    if n_crossed > 0 {
        extra_groupings.push(Grouping {
            relation: GroupingRelation::Crossed {
                n_clusters: n_crossed as u32,
            },
            slopes: vec![],
        });
        extra_ids.push(crossed);
    }
    let cluster = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_prim as u32,
            },
            slopes: vec![],
            extra_groupings,
        }),
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
    let (n, p) = (x.nrows(), x.ncols());
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
        build_z(&mut ws, xf64.as_ref(), &ids, &extra_ids, n);
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
        let got = glmm_laplace_deviance(&ws.params.clone(), &mut ws, xf64.as_ref(), &y, &ids, n);
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

/// Structured == dense on all three Group-G shapes, to FP error: drives the
/// dense `pirls_solve` and `pirls_solve_blocked_extras` on the same M / θ / β
/// from fresh scratch. The Schur reassociation is the same estimator. Bands
/// mirror the no-extras blocked parity (`blocked_pirls_matches_dense_slope_noextra`):
/// dev/pen/logdet 1e-9, u 1e-7.
#[test]
fn structured_extras_matches_dense() {
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

        // Dense reference: apply_lambda → pirls_solve.
        let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
        assert!(
            ws.groupings.structured_extras_eligible(),
            "{label}: fixture must be structured-eligible"
        );
        build_z(&mut ws, xf64.as_ref(), &ids, &extra_ids, n);
        let (k, p, nt) = (ws.k, ws.p, ws.n_theta);
        let mut params = vec![0.0; nt + p];
        params[..nt].copy_from_slice(&theta);
        params[nt..].copy_from_slice(&beta);
        let GlmmWorkspace {
            groupings,
            z,
            m,
            lam,
            prior_w,
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            mu,
            wm,
            wx,
            a,
            a_chol,
            a_llt_mem,
            a_rhs,
            ..
        } = &mut ws;
        apply_lambda(groupings, &params, z.as_ref(), m, lam, n);
        let dense = pirls_solve(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            k,
            p,
            m.as_ref(),
            xf64.as_ref(),
            &y,
            &prior_w[..n],
            false,
            &mut params[nt..],
            BetaStep::Fixed,
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            mu,
            wm,
            wx,
            a,
            a_chol,
            a_rhs,
            a_llt_mem,
            None, // offset
            None,
            n,
        );

        // Structured: build_packed_m → pirls_solve_blocked_extras, fresh scratch.
        let mut ws2 = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
        build_z(&mut ws2, xf64.as_ref(), &ids, &extra_ids, n);
        {
            let GlmmWorkspace {
                groupings,
                params: prm,
                z,
                lam,
                m_core_buf,
                cross_val,
                cross_col,
                n_cross,
                ..
            } = &mut ws2;
            prm[..nt].copy_from_slice(&theta);
            prm[nt..nt + p].copy_from_slice(&beta);
            build_packed_m(
                groupings,
                &prm[..],
                z.as_ref(),
                lam,
                &ids,
                m_core_buf,
                cross_val,
                cross_col,
                n_cross,
                n,
            );
        }
        let structured = {
            let GlmmWorkspace {
                groupings,
                m_core_buf,
                cross_val,
                cross_col,
                n_cross,
                prior_w,
                eta,
                prob,
                w,
                u,
                u_prev,
                eta_fixed,
                mu,
                core_blocks,
                coupling,
                schur_blk,
                coup_cols,
                coup_ptr,
                a_rhs,
                ..
            } = &mut ws2;
            build_coupling_csr(
                &ids,
                cross_col,
                n_cross,
                groupings.n_primary,
                n,
                coup_cols,
                coup_ptr,
            );
            let mut wx_scratch = faer::Mat::<f64>::zeros(n, p);
            pirls_solve_blocked_extras(
                crate::Family::Binomial {
                    link: crate::BinomialLink::Logit,
                },
                f64::NAN,
                groupings,
                &ids,
                m_core_buf,
                cross_val,
                cross_col,
                n_cross,
                xf64.as_ref(),
                &y,
                &prior_w[..n],
                false,
                &mut beta,
                BetaStep::Fixed,
                eta,
                prob,
                w,
                u,
                u_prev,
                eta_fixed,
                mu,
                core_blocks,
                coupling,
                schur_blk,
                coup_cols,
                coup_ptr,
                None,
                false,
                a_rhs,
                &mut wx_scratch,
                None, // offset
                None,
                n,
            )
        };

        // Re-tightened to 1e-9 (u 1e-7): when no halving fires the dense loop's
        // iterate path is bit-identical to the pre-halving one, and the
        // structured path (`pirls_solve_blocked_extras`, which shares the same
        // step-halving backtrack) agrees to FP error again.
        assert_eq!(dense.3, structured.3, "{label}: convergence flag");
        assert!(
            (dense.0 - structured.0).abs() < 1e-9,
            "{label} dev: dense {} structured {}",
            dense.0,
            structured.0
        );
        assert!(
            (dense.1 - structured.1).abs() < 1e-9,
            "{label} pen: dense {} structured {}",
            dense.1,
            structured.1
        );
        assert!(
            (dense.2 - structured.2).abs() < 1e-9,
            "{label} logdet: dense {} structured {}",
            dense.2,
            structured.2
        );
        for c in 0..k {
            assert!(
                (ws.u[c] - ws2.u[c]).abs() < 1e-7,
                "{label} u[{c}]: dense {} structured {}",
                ws.u[c],
                ws2.u[c]
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
///   `structured_extras_matches_dense` (dev/pen/logdet 1e-9, u 1e-7) to absorb the
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
            build_z(&mut ws, xf64.as_ref(), &ids, &extra_ids, n);
            let (p, nt) = (ws.p, ws.n_theta);
            let mut prm = vec![0.0; nt + p];
            prm[..nt].copy_from_slice(&theta);
            prm[nt..].copy_from_slice(&beta0);
            let mut ss = if use_panel {
                Some(
                    StructuredSchur::new(&ws.groupings, &ids, &extra_ids, n)
                        .expect("ncr > 0 ⇒ e > 0 ⇒ Some"),
                )
            } else {
                None
            };
            let mut beta = beta0;
            let GlmmWorkspace {
                groupings,
                z,
                lam,
                m_core_buf,
                cross_val,
                cross_col,
                n_cross,
                prior_w,
                eta,
                prob,
                w,
                u,
                u_prev,
                eta_fixed,
                mu,
                core_blocks,
                coupling,
                schur_blk,
                coup_cols,
                coup_ptr,
                a_rhs,
                ..
            } = &mut ws;
            build_packed_m(
                groupings,
                &prm[..],
                z.as_ref(),
                lam,
                &ids,
                m_core_buf,
                cross_val,
                cross_col,
                n_cross,
                n,
            );
            build_coupling_csr(
                &ids,
                cross_col,
                n_cross,
                groupings.n_primary,
                n,
                coup_cols,
                coup_ptr,
            );
            let mut wx_scratch = faer::Mat::<f64>::zeros(n, 2);
            let out = pirls_solve_blocked_extras(
                crate::Family::Binomial {
                    link: crate::BinomialLink::Logit,
                },
                f64::NAN,
                groupings,
                &ids,
                m_core_buf,
                cross_val,
                cross_col,
                n_cross,
                xf64.as_ref(),
                &y,
                &prior_w[..n],
                false,
                &mut beta,
                BetaStep::Fixed,
                eta,
                prob,
                w,
                u,
                u_prev,
                eta_fixed,
                mu,
                core_blocks,
                coupling,
                schur_blk,
                coup_cols,
                coup_ptr,
                ss.as_mut(),
                true, // force_dense: same factor arm both runs — isolate the downdate
                a_rhs,
                &mut wx_scratch,
                None, // offset
                None,
                n,
            );
            (out, u.clone())
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
/// INDEX primary + crossed BROOD, LOCATION). Returns the **sized** `ModelSpec`
/// (crossed widths resolved from the actual level counts via
/// `spec_sized_from_ids_pub` — a placeholder count would give the wrong crossed
/// width `e`), the `GroupIds`, the `n×4` design `X` (`[1, YEAR==96, YEAR==97,
/// cHEIGHT]`), the `TICKS` response `y`, and `(n, p)`. Shared by
/// `StructuredSchur` construction tests and the structured cold-start overshoot
/// gate below; mirrors `fit.rs::grouseticks_3crossed_inputs`.
pub(crate) fn grouseticks_3crossed_fixture(
) -> (ModelSpec, crate::GroupIds, Mat<f64>, Vec<f64>, usize, usize) {
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
    let n_index = *index_ids.iter().max().unwrap() as usize + 1;

    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_index as u32,
            },
            slopes: vec![],
            extra_groupings: vec![
                Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                },
                Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                },
            ],
        }),
    };
    let ids = crate::GroupIds {
        primary: index_ids,
        extra: vec![brood_ids, loc_ids],
    };
    // Resize the placeholder crossed `n_clusters` from the actual level counts
    // (matches `fit_warm` before building `LmmGroupings`).
    let sized = crate::fit::spec_sized_from_ids_pub(&model, &ids);
    (sized, ids, x, y, n, p)
}

/// Structured cold-start overshoot gate (the documented grouseticks
/// degenerate-fit bug): drive `glmm_laplace_deviance` on the 3-crossed Poisson
/// shape at θ = 1.0 (all three components) from a β = 0 cold start — historically
/// the first structured PIRLS step overshot into a ~1e30-weight regime, the
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
    build_z(&mut ws, x.as_ref(), &ids.primary, &ids.extra, n);
    // θ = 1.0 for every component (INDEX, BROOD, LOCATION); β = 0 cold start.
    let nt = ws.n_theta;
    for t in 0..nt {
        ws.params[t] = 1.0;
    }
    for j in 0..p {
        ws.params[nt + j] = 0.0;
    }
    let dev = glmm_laplace_deviance(&ws.params.clone(), &mut ws, x.as_ref(), &y, &ids.primary, n);
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
/// (`two_stage = false`) and two-stage (`two_stage = true`) — and asserts the
/// two optimizer paths land on the same optimum within the ORACLE tolerances
/// (β̂: beta_rel 1e-3; θ̂: abs+rel 1e-3 band; τ̂²: stddev_rel 1e-3). Returns
/// `(n_eval_single, n_eval_two)` for the baseline-doc print (the eval-count win is
/// a separate, measured concern; it is NOT gated here). Both workspaces are built
/// fresh from the same spec/data; only `two_stage` differs.
#[allow(clippy::too_many_arguments)]
fn assert_two_stage_matches_single(
    label: &str,
    spec: &ModelSpec,
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
    ws1.two_stage = false; // pin the single-stage reference: two_stage now defaults true
    build_z(&mut ws1, x, primary, extra, n);
    let fit1 = fit_glmm(
        &mut ws1,
        x,
        y,
        primary,
        &targets,
        None,
        &beta_start,
        n,
        WaldSe::Rx,
    );
    assert!(fit1.converged, "{label}: single-stage fit must converge");

    let mut ws2 = GlmmWorkspace::for_cluster_spec(p, spec, n, &[], 1);
    ws2.two_stage = true;
    build_z(&mut ws2, x, primary, extra, n);
    let fit2 = fit_glmm(
        &mut ws2,
        x,
        y,
        primary,
        &targets,
        None,
        &beta_start,
        n,
        WaldSe::Rx,
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
    let (ta, tb) = (fit1.tau_squared_hat, fit2.tau_squared_hat);
    let trel = (ta - tb).abs() / ta.abs().max(1e-6);
    assert!(
        trel < 1e-3,
        "{label}: τ² single {ta} vs two-stage {tb} (rel {trel})"
    );
    (fit1.n_eval, fit2.n_eval)
}

/// A/B: the two-stage β-profiling optimizer reaches the same optimum as the
/// single joint solve on the grouseticks 3-crossed Poisson fixture (the flagship
/// structured shape). θ̂/β̂/τ̂² agree within oracle tolerances — NOT bit-identity
/// (two optimizer paths). n_eval both ways is printed as a baseline doc; there is
/// deliberately NO n_eval assertion (the eval-count win is a separate, measured
/// concern).
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

/// Corpus A/B sweep: every cheaply-reachable committed GLMM fixture shape run
/// both ways (single- vs two-stage) must agree at oracle tolerances. Covers all
/// three PIRLS variants and both link classes: grouseticks 3-crossed Poisson-log
/// (structured, non-canonical), the logit intercept fixture (blocked, canonical),
/// its probit twin (blocked, non-canonical), and the crossed / nested /
/// crossed+nested logit extras shapes (structured). The cbpp herd-intercept
/// binomial GLMM (logit + probit, lme4-validated) and the Gamma mixed golden DO
/// exist as genuine GLMM fixtures, but their data/model helpers are private to
/// `fit.rs`'s `#[cfg(test)]` module and not reachable from here — the probit-GLMM
/// twin substitutes in this sweep, and `fit.rs`'s own test module covers
/// cbpp-probit + Gamma two-stage A/B coverage separately (where `ws.two_stage` is
/// settable). `#[ignore]`: run explicitly as the corpus proof; it is out of the
/// default fast suite.
#[test]
#[ignore]
fn two_stage_matches_single_stage_corpus_sweep() {
    // grouseticks 3-crossed Poisson (structured, non-canonical log link).
    {
        let (model, ids, x, y, n, p) = grouseticks_3crossed_fixture();
        let (a, b) = assert_two_stage_matches_single(
            "grouseticks_poisson",
            &model,
            x.as_ref(),
            &y,
            &ids.primary,
            &ids.extra,
            n,
            p,
        );
        println!("grouseticks_poisson n_eval: single {a} vs two {b}");
    }
    // Logit intercept, no extras (blocked, canonical) + its probit twin.
    {
        let (x, y, ids) = glmm_intercept_dataset();
        for (label, link) in [
            ("logit", BinomialLink::Logit),
            ("probit", BinomialLink::Probit),
        ] {
            let mut spec = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
            spec.family = Family::Binomial { link };
            let (a, b) = assert_two_stage_matches_single(
                &format!("intercept_{label}"),
                &spec,
                x.as_ref(),
                &y,
                &ids,
                &[],
                80,
                2,
            );
            println!("intercept_{label} n_eval: single {a} vs two {b}");
        }
    }
    // Logit intercept extras: crossed / nested / crossed+nested (structured).
    for (np, ncr, label) in [
        (0usize, 6usize, "crossed"),
        (2, 0, "nested"),
        (2, 6, "crossed_nested"),
    ] {
        let (x, y, ids, extra_ids, spec) = glmm_extras_q1_dataset(np, ncr);
        let n = y.len();
        let (a, b) = assert_two_stage_matches_single(
            &format!("extras_{label}"),
            &spec,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            n,
            2,
        );
        println!("extras_{label} n_eval: single {a} vs two {b}");
    }
}

/// Structured inference (Var(β̂)_jj, z²) matches a dense Schur recomputation on
/// all three Group-G shapes — the structured `A⁻¹` apply in `structured_schur_fill`
/// is the same estimator as the dense `dense_schur_fill`. Runs the full
/// `fit_glmm` (structured) then recomputes Var(β̂) densely from the converged
/// ws.{w, lam, params}. Mirrors `blocked_inference_matches_dense_slope_noextra`.
#[test]
fn structured_inference_matches_dense() {
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
        build_z(&mut ws, xf64.as_ref(), &ids, &extra_ids, n);
        let fit = fit_glmm(
            &mut ws,
            xf64.as_ref(),
            &y,
            &ids,
            &[1],
            None,
            &[0.2, 0.8],
            n,
            WaldSe::Rx,
        );
        assert!(fit.converged, "{label}: fit must converge");
        let (k, p, nt) = (ws.k, ws.p, ws.n_theta);
        let var_structured = ws.var_diag[1];
        let tsq_structured = ws.t_sq[1];
        let beta1 = ws.betas[1];
        // Dense recompute of Var(β̂)_11 from the converged state.
        crate::lmm::primary_lambda(&ws.params[..nt], ws.groupings.primary_q, &mut ws.lam);
        {
            let GlmmWorkspace {
                groupings,
                params,
                z,
                m,
                lam,
                ..
            } = &mut ws;
            apply_lambda(groupings, &params[..], z.as_ref(), m, lam, n);
        }
        let mut xtwx = Mat::<f64>::zeros(p, p);
        for r in 0..p {
            for c in 0..p {
                let mut sm = 0.0;
                for i in 0..n {
                    sm += xf64[(i, r)] * ws.w[i] * xf64[(i, c)];
                }
                xtwx[(r, c)] = sm;
            }
        }
        let mut xtwm = Mat::<f64>::zeros(p, k);
        for r in 0..p {
            for c in 0..k {
                let mut sm = 0.0;
                for i in 0..n {
                    sm += xf64[(i, r)] * ws.w[i] * ws.m[(i, c)];
                }
                xtwm[(r, c)] = sm;
            }
        }
        let mut a = Mat::<f64>::zeros(k, k);
        for r in 0..k {
            for c in 0..k {
                let mut sm = if r == c { 1.0 } else { 0.0 };
                for i in 0..n {
                    sm += ws.m[(i, r)] * ws.w[i] * ws.m[(i, c)];
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
        let var_dense: f64 = fwd.iter().map(|v| v * v).sum();
        assert!(
            (var_structured - var_dense).abs() < 1e-8,
            "{label} var: structured {var_structured} dense {var_dense}"
        );
        let tsq_dense = beta1 * beta1 / var_dense;
        assert!(
            (tsq_structured - tsq_dense).abs() < 1e-6,
            "{label} z²: structured {tsq_structured} dense {tsq_dense}"
        );
    }
}

/// Dense-path (`pirls_solve`) overshoot fixture. The primary grouping carries
/// 8 slopes (q_p = 9 > MAX_PRIMARY_Q = 8), so with a crossed extra present the
/// core width q_core = 9 makes `structured_extras_eligible()` false and
/// `laplace_deviance` routes to the genuinely-dense `pirls_solve` (not the
/// blocked / structured paths). Poisson-log with large counts (~5000) at the
/// β = 0, u = 0 cold start drives the first full Fisher step into the exp()
/// blow-up regime (the grouseticks non-convergence class). The crossed ids are
/// baked into ws.z via `build_z` here, since `glmm_laplace_deviance` threads
/// only the primary ids. Verified to return INFINITY on the pre-step-halving
/// dense loop (non-convergence).
fn dense_path_overshoot_fixture() -> (GlmmWorkspace, Mat<f64>, Vec<f64>, Vec<u32>, usize) {
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
    let cluster = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_prim as u32,
            },
            slopes: (1..=n_slopes as u32).collect(),
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed {
                    n_clusters: n_crossed as u32,
                },
                slopes: vec![],
            }],
        }),
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(1, &cluster, n, &slope_cols, 1);
    build_z(&mut ws, x.as_ref(), &ids, &[crossed], n);
    (ws, x, y, ids, n)
}

#[test]
fn pirls_dense_step_halving_recovers_from_overshoot() {
    // Shape/data chosen so the first full Fisher step at the β=0 cold start
    // overshoots η into the exp() blow-up regime (the grouseticks failure
    // class, on the dense path). Before step-halving this returned INFINITY.
    let (mut ws, x, y, ids, n) = dense_path_overshoot_fixture();
    assert!(
        !ws.groupings.structured_extras_eligible() && !ws.groupings.extra_offsets.is_empty(),
        "fixture must route to the dense pirls_solve (extras present, core oversized)"
    );
    let n_params = ws.n_theta + ws.p;
    let params: Vec<f64> = (0..n_params)
        .map(|i| if i < ws.n_theta { 1.0 } else { 0.0 })
        .collect();
    let dev = glmm_laplace_deviance(&params, &mut ws, x.as_ref(), &y, &ids, n);
    assert!(
        dev.is_finite(),
        "step-halving must rescue the cold-start overshoot, got {dev}"
    );
    // Sanity ceiling against a finite-but-absurd recovery (the guarded bug
    // returned INFINITY). A run of this fixture converges to dev≈1269; 2600
    // is a generous 2× headroom above that.
    assert!(dev < 2600.0, "recovered deviance {dev} implausibly large");
}

/// PQL stationarity of the Profile-mode dense β step: after a converged
/// `pirls_solve` in `BetaStep::Profile` (β seeded at 0, θ held at a blind
/// value), the β-gradient X'ρ must vanish — that IS the definition of "β is
/// PQL-optimal for this θ", needing no external oracle. Recompute ρ = y − p̂
/// at the RETURNED (u, β) and bound ‖X'ρ‖∞ by 1e-6·n. Canonical logit
/// (Newton) drives the joint (u,β) step to the PIRLS exit band, so this holds
/// tightly; the bound is tied to `pirls_tol` — loosen it (never the PIRLS
/// tolerance) if a non-canonical family ever falls short.
#[test]
fn pirls_dense_profile_beta_reaches_pql_stationarity() {
    let (xf64, y, ids) = glmm_slope_noextra_dataset();
    let n = y.len();
    let cluster = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    build_z(&mut ws, xf64.as_ref(), &ids, &[], n);
    let (k, p, nt) = (ws.k, ws.p, ws.n_theta);
    // Blind θ (vech Λ_p), β seed = 0. β is a buffer DISTINCT from ws.beta_rhs
    // (the Profile δβ scratch), per the BetaStep contract.
    let theta = [0.5_f64, 0.1, 0.4];
    let mut params = vec![0.0; nt + p];
    params[..nt].copy_from_slice(&theta);
    let mut beta = vec![0.0_f64; p];
    let GlmmWorkspace {
        groupings,
        z,
        m,
        lam,
        prior_w,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        mu,
        wm,
        wx,
        a,
        a_chol,
        a_llt_mem,
        a_rhs,
        xtwx,
        xtwm,
        ainv_mtwx,
        schur,
        schur_llt_mem,
        beta_rhs,
        beta_prev,
        ..
    } = &mut ws;
    apply_lambda(groupings, &params, z.as_ref(), m, lam, n);
    let out = pirls_solve(
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        f64::NAN,
        k,
        p,
        m.as_ref(),
        xf64.as_ref(),
        &y,
        &prior_w[..n],
        false,
        &mut beta,
        BetaStep::Profile {
            xtwx,
            xtwm,
            ainv_mtwx,
            schur,
            beta_rhs,
            beta_prev,
            schur_llt_mem,
        },
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        mu,
        wm,
        wx,
        a,
        a_chol,
        a_rhs,
        a_llt_mem,
        None, // offset
        None,
        n,
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
            e += mref[(i, c)] * u[c];
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
/// `pirls_dense_profile_beta_reaches_pql_stationarity`. After a converged
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
    let cluster = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    build_z(&mut ws, xf64.as_ref(), &ids, &[], n);
    assert!(
        ws.groupings.extra_offsets.is_empty(),
        "fixture must route to the blocked pirls_solve_blocked (no extras)"
    );
    let p = ws.p;
    let q = ws.groupings.primary_q;
    // Blind θ (vech Λ_p), β seed = 0. β is a buffer DISTINCT from ws.beta_rhs
    // (the Profile δβ scratch), per the BetaStep contract.
    let theta = [0.5_f64, 0.1, 0.4];
    crate::lmm::primary_lambda(&theta, ws.groupings.primary_q, &mut ws.lam);
    fill_z_f64(&ws.groupings, xf64.as_ref(), &mut ws.z_buf, n);
    let mut beta = vec![0.0_f64; p];
    let GlmmWorkspace {
        groupings,
        lam,
        z_buf,
        m_buf,
        prior_w,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        a_blocks,
        a_rhs,
        xtwx,
        xtwm,
        ainv_mtwx,
        schur,
        schur_llt_mem,
        beta_rhs,
        beta_prev,
        wx,
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
            xtwx,
            xtwm,
            ainv_mtwx,
            schur,
            beta_rhs,
            beta_prev,
            schur_llt_mem,
        },
        lam,
        z_buf,
        m_buf,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        a_blocks,
        a_rhs,
        wx,
        None, // offset
        None,
        n,
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
            e += m_buf[i * q + c] * u[ubase + c];
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
/// Poisson-log is non-canonical, so the bound is deliberately loosened relative
/// to the tight logit gates — tied to `pirls_tol`, never the PIRLS tolerance.
#[test]
fn pirls_structured_profile_beta_reaches_pql_stationarity() {
    let (model, ids, x, y, n, p) = grouseticks_3crossed_fixture();
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
    assert!(
        ws.groupings.structured_extras_eligible(),
        "grouseticks 3-crossed must route through the structured extras path"
    );
    build_z(&mut ws, x.as_ref(), &ids.primary, &ids.extra, n);
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
            z,
            lam,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            ..
        } = &mut ws;
        build_packed_m(
            groupings,
            params,
            z.as_ref(),
            lam,
            &ids.primary,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            n,
        );
    }
    let out = {
        let GlmmWorkspace {
            groupings,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            prior_w,
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            mu,
            core_blocks,
            coupling,
            schur_blk,
            coup_cols,
            coup_ptr,
            a_rhs,
            xtwx,
            xtwm,
            ainv_mtwx,
            schur,
            schur_llt_mem,
            beta_rhs,
            beta_prev,
            wx,
            ..
        } = &mut ws;
        build_coupling_csr(
            &ids.primary,
            cross_col,
            n_cross,
            groupings.n_primary,
            n,
            coup_cols,
            coup_ptr,
        );
        pirls_solve_blocked_extras(
            Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            f64::NAN,
            groupings,
            &ids.primary,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            x.as_ref(),
            &y,
            &prior_w[..n],
            false,
            &mut beta,
            BetaStep::Profile {
                xtwx,
                xtwm,
                ainv_mtwx,
                schur,
                beta_rhs,
                beta_prev,
                schur_llt_mem,
            },
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            mu,
            core_blocks,
            coupling,
            schur_blk,
            coup_cols,
            coup_ptr,
            None,
            false,
            a_rhs,
            wx,
            None, // offset
            None,
            n,
        )
    };
    assert!(out.3, "Profile-mode structured solve must converge");
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
            e += ws.m_core_buf[i * qc + local] * ws.u[core_col(f, local)];
        }
        let cbase = i * g_cap;
        for z in 0..ws.n_cross[i] as usize {
            let b = ws.cross_col[cbase + z] as usize;
            e += ws.cross_val[cbase + z] * ws.u[k_family + b];
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
/// the dense β step at the same θ. On a crossed extras shape the dense
/// `pirls_solve` (full M) and the structured `pirls_solve_blocked_extras` (packed
/// M) are the same estimator (as `structured_extras_matches_dense` proves in
/// Fixed mode); run BOTH in `BetaStep::Profile` from β = 0 and require the
/// converged β to agree to ≤1e-8 rel. Guards a sign/transpose slip in exactly one
/// variant — the structured Schur-border apply is the intricate one this catches.
#[test]
fn structured_profile_beta_matches_dense_profile() {
    let (xf64, y, ids, extra_ids, cluster) = glmm_extras_q1_dataset(0, 6);
    let n = y.len();
    let theta = [0.5_f64, 0.45];

    // Dense reference: apply_lambda → pirls_solve in Profile mode, β seed 0.
    let mut wsd = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
    build_z(&mut wsd, xf64.as_ref(), &ids, &extra_ids, n);
    let (k, p, nt) = (wsd.k, wsd.p, wsd.n_theta);
    let mut params = vec![0.0; nt + p];
    params[..nt].copy_from_slice(&theta);
    let mut beta_dense = vec![0.0_f64; p];
    {
        let GlmmWorkspace {
            groupings,
            z,
            m,
            lam,
            ..
        } = &mut wsd;
        apply_lambda(groupings, &params, z.as_ref(), m, lam, n);
    }
    let dense = {
        let GlmmWorkspace {
            m,
            prior_w,
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            mu,
            wm,
            wx,
            a,
            a_chol,
            a_llt_mem,
            a_rhs,
            xtwx,
            xtwm,
            ainv_mtwx,
            schur,
            schur_llt_mem,
            beta_rhs,
            beta_prev,
            ..
        } = &mut wsd;
        pirls_solve(
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            f64::NAN,
            k,
            p,
            m.as_ref(),
            xf64.as_ref(),
            &y,
            &prior_w[..n],
            false,
            &mut beta_dense,
            BetaStep::Profile {
                xtwx,
                xtwm,
                ainv_mtwx,
                schur,
                beta_rhs,
                beta_prev,
                schur_llt_mem,
            },
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            mu,
            wm,
            wx,
            a,
            a_chol,
            a_rhs,
            a_llt_mem,
            None, // offset
            None,
            n,
        )
    };
    assert!(dense.3, "dense Profile solve must converge");

    // Structured: build_packed_m → pirls_solve_blocked_extras in Profile mode.
    let mut wss = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[], 1);
    build_z(&mut wss, xf64.as_ref(), &ids, &extra_ids, n);
    let mut beta_str = vec![0.0_f64; p];
    {
        let GlmmWorkspace {
            groupings,
            params: prm,
            z,
            lam,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            ..
        } = &mut wss;
        prm[..nt].copy_from_slice(&theta);
        build_packed_m(
            groupings,
            &prm[..],
            z.as_ref(),
            lam,
            &ids,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            n,
        );
    }
    let structured = {
        let GlmmWorkspace {
            groupings,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            prior_w,
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            mu,
            core_blocks,
            coupling,
            schur_blk,
            coup_cols,
            coup_ptr,
            a_rhs,
            xtwx,
            xtwm,
            ainv_mtwx,
            schur,
            schur_llt_mem,
            beta_rhs,
            beta_prev,
            wx,
            ..
        } = &mut wss;
        build_coupling_csr(
            &ids,
            cross_col,
            n_cross,
            groupings.n_primary,
            n,
            coup_cols,
            coup_ptr,
        );
        pirls_solve_blocked_extras(
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            f64::NAN,
            groupings,
            &ids,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            xf64.as_ref(),
            &y,
            &prior_w[..n],
            false,
            &mut beta_str,
            BetaStep::Profile {
                xtwx,
                xtwm,
                ainv_mtwx,
                schur,
                beta_rhs,
                beta_prev,
                schur_llt_mem,
            },
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            mu,
            core_blocks,
            coupling,
            schur_blk,
            coup_cols,
            coup_ptr,
            None,
            false,
            a_rhs,
            wx,
            None, // offset
            None,
            n,
        )
    };
    assert!(structured.3, "structured Profile solve must converge");

    for j in 0..p {
        let rel = (beta_dense[j] - beta_str[j]).abs() / beta_dense[j].abs().max(1.0);
        assert!(
            rel < 1e-8,
            "Profile β[{j}] cross-variant: dense {} structured {} (rel {rel:.2e})",
            beta_dense[j],
            beta_str[j]
        );
    }
}

/// The coupling CSR is cached per (fit, θ-pinning mask). A BOBYQA eval can
/// pin a crossed θ to exactly 0 (its lower bound), which drops that grouping
/// from build_packed_m's cross_col/n_cross and changes the CSR pattern — a
/// stale cache here would silently corrupt the Schur build. Drive the
/// workspace through unpinned → pinned → unpinned evals and assert each
/// deviance is bit-identical to a fresh-workspace eval at the same point.
#[test]
fn coupling_csr_rebuilds_on_theta_pinning_transition() {
    let (model, ids, x, y, n, p) = grouseticks_3crossed_fixture();
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
    let nt = ws.n_theta;
    // Unpinned point: every θ > 0, β = 0.
    let mut unpinned = vec![0.5; nt + p];
    for b in unpinned[nt..].iter_mut() {
        *b = 0.0;
    }
    // Pinned point: last crossed θ exactly 0 (BOBYQA bound), rest unchanged.
    let mut pinned = unpinned.clone();
    let last_crossed = ws.groupings.crossed.last().unwrap().vech_start;
    pinned[last_crossed] = 0.0;

    let fresh = |params: &[f64]| {
        let mut w2 = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
        glmm_laplace_deviance(params, &mut w2, x.as_ref(), &y, &ids.primary, n)
    };
    let d1 = glmm_laplace_deviance(&unpinned, &mut ws, x.as_ref(), &y, &ids.primary, n);
    let d2 = glmm_laplace_deviance(&pinned, &mut ws, x.as_ref(), &y, &ids.primary, n);
    let d3 = glmm_laplace_deviance(&unpinned, &mut ws, x.as_ref(), &y, &ids.primary, n);
    assert_eq!(
        d1,
        fresh(&unpinned),
        "unpinned eval must match fresh workspace"
    );
    assert_eq!(d2, fresh(&pinned), "pinned eval read a stale CSR");
    assert_eq!(d3, d1, "un-pinning must restore the original pattern");
}

/// No-extras twin of `dense_path_overshoot_fixture`: primary-only grouping
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
    let cluster = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_prim as u32,
            },
            slopes: (1..=n_slopes as u32).collect(),
            extra_groupings: vec![],
        }),
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(1, &cluster, n, &slope_cols, 1);
    build_z(&mut ws, x.as_ref(), &ids, &[], n);
    (ws, x, y, ids, n)
}

#[test]
fn pirls_blocked_step_halving_recovers_from_overshoot() {
    // Shape/data chosen so the first full Fisher step at the β=0 cold start
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
    let dev = glmm_laplace_deviance(&params, &mut ws, x.as_ref(), &y, &ids, n);
    assert!(
        dev.is_finite(),
        "step-halving must rescue the cold-start overshoot, got {dev}"
    );
    // Sanity ceiling against a finite-but-absurd recovery (the guarded bug
    // returned INFINITY). A run of this fixture converges to dev≈1837; 3700
    // is a generous 2× headroom above that.
    assert!(dev < 3700.0, "recovered deviance {dev} implausibly large");
}

/// Profile evaluation via the `laplace_deviance` production entry point
/// (`glmm_laplace_deviance_profile` = the test-only twin: `profile_beta = true`,
/// `beta = ws.beta_prof`). Asserts (a) finite deviance; (b) `beta_prof` moved off
/// its 0 seed — β was PROFILED, not held (this is what would fail if the copy
/// gating or the Profile δβ wiring were wrong); (c) two calls from the same seed
/// return bit-identical deviance AND β — the BOBYQA objective-consistency
/// (determinism) the two-stage optimizer requires ("Why two-stage"). Routes
/// through the blocked PIRLS variant (no extras).
#[test]
fn laplace_deviance_profile_evaluates_and_is_deterministic() {
    let (xf64, y, ids) = glmm_slope_noextra_dataset();
    let n = y.len();
    let cluster = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    build_z(&mut ws, xf64.as_ref(), &ids, &[], n);
    assert!(
        ws.groupings.extra_offsets.is_empty(),
        "fixture must route to the blocked pirls_solve_blocked (no extras)"
    );
    let (p, nt) = (ws.p, ws.n_theta);
    // Blind θ (vech Λ_p); the β slots in `params` are IGNORED in Profile mode
    // (β starts at the beta_prof 0 seed the twin resets and is profiled from there).
    let theta = [0.5_f64, 0.1, 0.4];
    let mut params = vec![0.0; nt + p];
    params[..nt].copy_from_slice(&theta);

    let d1 = glmm_laplace_deviance_profile(&params, &mut ws, xf64.as_ref(), &y, &ids, n);
    let beta1 = ws.beta_prof.clone();
    assert!(d1.is_finite(), "Profile deviance must be finite, got {d1}");
    assert!(
        beta1.iter().any(|b| b.abs() > 1e-9),
        "beta_prof must move off its 0 seed (β profiled, not held): {beta1:?}"
    );

    let d2 = glmm_laplace_deviance_profile(&params, &mut ws, xf64.as_ref(), &y, &ids, n);
    let beta2 = ws.beta_prof.clone();
    assert_eq!(
        d1.to_bits(),
        d2.to_bits(),
        "Profile deviance must be bit-identical across identical calls"
    );
    for j in 0..p {
        assert_eq!(
            beta1[j].to_bits(),
            beta2[j].to_bits(),
            "profiled β must be bit-identical across identical calls, coord {j}"
        );
    }
}

/// The θ-only stage-1 scratch (`solver_stage1`, `params_stage1`, `two_stage`)
/// exists on every `GlmmWorkspace` and is the shipped default fit path (see
/// `fit_glmm`'s STAGE 1 block), not an inert addition. Uses the 1-slope +
/// 1-crossed fixture so `n_theta = 4 >= 3`, exercising the `sparse_lmm_seed`
/// mid-model npt branch.
#[test]
fn workspace_carries_stage1_scratch_sized_for_n_theta() {
    let (xf64, _y, ids, crossed_ids, cluster) = glmm_slope_crossed_dataset();
    let n = ids.len();
    let mut ws = GlmmWorkspace::for_cluster_spec(2, &cluster, n, &[1], 1);
    build_z(
        &mut ws,
        xf64.as_ref(),
        &ids,
        std::slice::from_ref(&crossed_ids),
        n,
    );
    assert_eq!(
        ws.n_theta, 4,
        "fixture shape: q_p=2 vech(3) + 1 crossed scalar"
    );
    assert_eq!(
        ws.params_stage1.len(),
        ws.n_theta,
        "params_stage1 is a θ-only candidate/incumbent buffer"
    );
    assert!(
        ws.two_stage,
        "two_stage defaults to true (the two-stage optimizer is the shipped path)"
    );
    // solver_stage1 must be usable as an n_theta-dim BOBYQA solver over the
    // θ-only bound slices — smoke-test one minimize call on a trivial objective.
    let lower = ws.lower[..ws.n_theta].to_vec();
    let upper = ws.upper[..ws.n_theta].to_vec();
    let mut x0 = ws.params_stage1.clone();
    let out = ws.solver_stage1.minimize(
        |xs: &[f64]| xs.iter().map(|v| v * v).sum(),
        &mut x0,
        &lower,
        &upper,
    );
    assert_ne!(
        out.status,
        bobyqa::Status::InvalidArgs,
        "stage1 solver must be a valid n_theta-dim BOBYQA config, got {:?}",
        out.status
    );
}

/// Objective-identity gate. Both the stage-2 BOBYQA
/// objective and the single-stage joint objective are, by construction, the SAME
/// function call — `laplace_deviance(profile_beta = false)` — with `ws.two_stage`
/// only gating whether stage 1 runs BEFORE that objective, never the objective
/// itself. This test does not exercise those two call sites directly; instead it
/// evaluates `glmm_laplace_deviance` (production stage-2 / single-stage closure
/// shape) at five [θ|β] points — the blind start, the converged optimum, and
/// three perturbations — on two independently built, identically-seeded
/// grouseticks-3crossed workspaces (structured, non-canonical log link), varying
/// only `ws.two_stage` (false vs true), and asserts `to_bits` equality. The
/// warm-start seed û is pinned to the SAME constant on both (`warm_seed_active` +
/// a fixed `u_seed`) so the only remaining difference between the two evals is
/// the flag itself — proving the objective is flag-independent and deterministic,
/// which is the by-construction argument that single-stage and stage-2 evaluate
/// the same function.
#[test]
fn stage2_objective_is_bit_identical_to_single_stage_objective() {
    let (model, ids, x, y, n, p) = grouseticks_3crossed_fixture();
    let nt = {
        let ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
        ws.n_theta
    };

    // Converged optimum: one single-stage fit, snapshot [θ̂|β̂] from ws.params.
    let converged: Vec<f64> = {
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
        build_z(&mut ws, x.as_ref(), &ids.primary, &ids.extra, n);
        let beta0 = vec![0.0_f64; p];
        let fit = fit_glmm(
            &mut ws,
            x.as_ref(),
            &y,
            &ids.primary,
            &[0, 1, 2, 3],
            None,
            &beta0,
            n,
            WaldSe::Rx,
        );
        assert!(
            fit.converged,
            "single-stage grouseticks must converge to supply the optimum point"
        );
        ws.params[..nt + p].to_vec()
    };

    // Five [θ|β] sample points of width nt+p: blind start, the optimum, and three
    // perturbations off it (θ-scale up, a β shift, a θ-shrink + opposite β shift).
    let mut points: Vec<Vec<f64>> = Vec::new();
    {
        let mut blind = vec![0.0_f64; nt + p];
        #[allow(clippy::needless_range_loop)]
        for t in 0..nt {
            blind[t] = 1.0; // THETA0
        }
        points.push(blind);
    }
    points.push(converged.clone());
    {
        let mut q = converged.clone();
        q[0] *= 1.1;
        points.push(q);
    }
    {
        let mut q = converged.clone();
        q[nt] += 0.2;
        points.push(q);
    }
    {
        let mut q = converged.clone();
        #[allow(clippy::needless_range_loop)]
        for t in 0..nt {
            q[t] *= 0.5;
        }
        q[nt + 1] -= 0.1;
        points.push(q);
    }

    // Evaluate glmm_laplace_deviance (= laplace_deviance with profile_beta=false,
    // the production stage-2 / single-stage closure shape) on a freshly built
    // workspace with the warm seed pinned to a shared constant.
    let eval = |pt: &[f64], two_stage: bool| -> f64 {
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
        build_z(&mut ws, x.as_ref(), &ids.primary, &ids.extra, n);
        ws.two_stage = two_stage;
        let k = ws.k.max(1);
        ws.warm_seed_active = true;
        for v in ws.u_seed[..k].iter_mut() {
            *v = 0.05;
        }
        glmm_laplace_deviance(pt, &mut ws, x.as_ref(), &y, &ids.primary, n)
    };

    for (i, pt) in points.iter().enumerate() {
        let a = eval(pt, false);
        let b = eval(pt, true);
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "point {i}: single-stage objective {a} vs stage-2 objective {b} must be bit-identical"
        );
    }
}

/// Adversarial halving coverage. Runs a
/// deliberately hostile fit both ways — single-stage (`two_stage = false`, the
/// trusted reference) and two-stage (`two_stage = true`) — and asserts the
/// two-stage path lands on exactly one of two surfaces, never a third:
/// (a) it converges to genuinely finite estimates AND, when single-stage also
/// converged, agrees with it at ORACLE tolerances, or
/// (b) it fails cleanly — `converged == false` with the whole inference output
/// NaN (`nan_fit`), the terminal state an exhausted-halving PIRLS reaches.
/// A finite-but-wrong two-stage fit (converged, disagreeing with single-stage) is
/// precisely the failure this gate exists to catch.
#[allow(clippy::too_many_arguments)]
fn assert_two_stage_adversarial(
    label: &str,
    spec: &ModelSpec,
    x: faer::MatRef<f64>,
    y: &[f64],
    primary: &[u32],
    extra: &[Vec<u32>],
    n: usize,
    p: usize,
    theta_start: Option<&[f64]>,
) {
    let targets = [if p >= 2 { 1u32 } else { 0u32 }];
    let beta_start = vec![0.0_f64; p];

    let mut ws1 = GlmmWorkspace::for_cluster_spec(p, spec, n, &[], 1);
    ws1.two_stage = false; // pin the single-stage reference: two_stage now defaults true
    build_z(&mut ws1, x, primary, extra, n);
    let fit1 = fit_glmm(
        &mut ws1,
        x,
        y,
        primary,
        &targets,
        theta_start,
        &beta_start,
        n,
        WaldSe::Rx,
    );

    let mut ws2 = GlmmWorkspace::for_cluster_spec(p, spec, n, &[], 1);
    ws2.two_stage = true;
    build_z(&mut ws2, x, primary, extra, n);
    let fit2 = fit_glmm(
        &mut ws2,
        x,
        y,
        primary,
        &targets,
        theta_start,
        &beta_start,
        n,
        WaldSe::Rx,
    );

    if !fit2.converged {
        // Clean NaN failure surface (nan_fit): every inference output NaN — never
        // a finite value smuggled out under converged == false.
        assert!(
            fit2.tau_squared_hat.is_nan(),
            "{label}: failed two-stage fit must NaN τ̂²"
        );
        assert!(
            ws2.betas[..p].iter().all(|b| b.is_nan()),
            "{label}: failed two-stage fit must NaN β̂, got {:?}",
            &ws2.betas[..p]
        );
        return;
    }
    // Two-stage converged: outputs must be genuinely finite (not NaN under a
    // converged flag), the sanity floor when single-stage supplies no reference.
    assert!(
        fit2.tau_squared_hat.is_finite() && fit2.tau_squared_hat >= 0.0,
        "{label}: converged two-stage fit must have finite τ̂² ≥ 0, got {}",
        fit2.tau_squared_hat
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
        let (ta, tb) = (fit1.tau_squared_hat, fit2.tau_squared_hat);
        let trel = (ta - tb).abs() / ta.abs().max(1e-6);
        assert!(
            trel < 1e-3,
            "{label}: τ² single {ta} vs two-stage {tb} (rel {trel})"
        );
    }
}

/// Adversarial: a wildly over-scaled θ start (θ₀ = 100, ~100× the fitted scale)
/// on the logit intercept fixture. The joint solve must recover to the single-stage
/// optimum or fail cleanly — never settle finite-but-wrong.
#[test]
fn two_stage_adversarial_bad_theta_start() {
    let (x, y, ids) = glmm_intercept_dataset();
    let spec = logit_intercept_spec(Sizing::FixedClusters { n_clusters: 8 });
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
    );
}

/// Adversarial: a near-collinear design (col 2 = col 1 + 1e-8·noise, condition
/// number ~1e8) makes the β-Schur near-singular — the stress the β-profiling step's
/// dense solve must survive. `fit_glmm` runs no rank-deficiency salvage (that lives
/// in `fit_warm`), so this hits the ill-conditioned kernel directly.
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
    let spec = logit_intercept_spec(Sizing::FixedClusters {
        n_clusters: nc as u32,
    });
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
    );
}

/// Adversarial: 30 clusters of 2 observations each — near-singleton groups leave
/// the random intercept barely identified and the θ landscape flat/ill-posed, the
/// regime where a mis-stepped optimizer most easily wanders finite-but-wrong.
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
    let spec = logit_intercept_spec(Sizing::FixedClusters {
        n_clusters: nc as u32,
    });
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
    );
}

#[test]
fn cluster_row_index_matches_naive_filter_and_preserves_row_order() {
    let cluster_ids: [u32; 10] = [2, 0, 1, 0, 2, 1, 1, 0, 2, 0];
    let s = 3;
    let idx = ClusterRowIndex::build(&cluster_ids, s);
    for c in 0..s {
        let naive: Vec<u32> = cluster_ids
            .iter()
            .enumerate()
            .filter(|&(_, &cc)| cc as usize == c)
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(idx.cluster_rows(c), naive.as_slice(), "cluster {c}");
        assert!(
            idx.cluster_rows(c).windows(2).all(|w| w[0] < w[1]),
            "cluster {c} rows not ascending: {:?}",
            idx.cluster_rows(c)
        );
    }
    let mut all: Vec<u32> = (0..s).flat_map(|c| idx.cluster_rows(c).to_vec()).collect();
    all.sort_unstable();
    assert_eq!(all, (0..cluster_ids.len() as u32).collect::<Vec<_>>());
}

#[test]
fn cluster_row_index_handles_empty_cluster() {
    // cluster 1 has zero rows — ptr[1]==ptr[2], cluster_rows(1) must be empty, not panic.
    let cluster_ids: [u32; 4] = [0, 2, 0, 2];
    let idx = ClusterRowIndex::build(&cluster_ids, 3);
    assert_eq!(idx.cluster_rows(0), &[0, 2]);
    assert!(idx.cluster_rows(1).is_empty());
    assert_eq!(idx.cluster_rows(2), &[1, 3]);
}
