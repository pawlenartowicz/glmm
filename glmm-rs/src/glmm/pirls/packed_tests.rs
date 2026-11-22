use super::*;
use crate::fit::SizedSpec;
use crate::lmm::LmmGroupings;

/// [`packed_m_vals_theta_deriv`] must reproduce a central difference of
/// [`fill_m_vals`] in every θ coordinate and every slot of every row, on
/// a packed shape carrying a random slope on the PRIMARY grouping and a
/// random slope on an EXTRA one — the two blocks whose vech walks, `Λ`
/// offsets and `z` columns the selection has to keep apart.
#[test]
fn packed_m_vals_theta_deriv_matches_fill_m_vals_difference() {
    let n = 16;
    let mut x = Mat::<f64>::zeros(n, 3);
    for i in 0..n {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = 0.7 + 0.31 * i as f64;
        x[(i, 2)] = 2.5 - 0.17 * (i % 7) as f64;
    }
    let spec = SizedSpec {
        spec: crate::ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(crate::ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![crate::Grouping {
                    terms: Vec::new(),
                    relation: crate::GroupingRelation::Crossed,
                    slopes: vec![2],
                }],
            }),
        },
        n_primary: 4,
        extra_counts: vec![3],
        tail_fill: 0.0,
    };
    let g = LmmGroupings::from_cluster_spec_ext(&spec, &[1], &[vec![2]]);
    let mut packed = PackedScratch::for_shape(&g, n, g.k_total);
    let width = packed.width;
    let n_theta = g.n_theta();
    let mut theta: Vec<f64> = (0..n_theta).map(|k| 0.4 + 0.11 * k as f64).collect();
    let eps = 1e-6;
    let mut z = vec![0.0; n * 2];
    crate::glmm::workspace::fill_z_f64(&g, x.as_ref(), x.as_ref().subcols(0, 0), &mut z, n);

    for a in 0..n_theta {
        let base = theta[a];
        theta[a] = base + eps;
        crate::sparse::fill_lambda_small(&theta, &g, &mut packed.lam_small);
        fill_m_vals(&mut packed, &g, &z, n);
        let plus = packed.m_vals[..n * width].to_vec();
        theta[a] = base - eps;
        crate::sparse::fill_lambda_small(&theta, &g, &mut packed.lam_small);
        fill_m_vals(&mut packed, &g, &z, n);
        let minus = packed.m_vals[..n * width].to_vec();
        theta[a] = base;

        let mut d = vec![0.0; width];
        for i in 0..n {
            packed_m_vals_theta_deriv(&g, a, &z, i, &mut d);
            for t in 0..width {
                let want = (plus[i * width + t] - minus[i * width + t]) / (2.0 * eps);
                assert!(
                    (d[t] - want).abs() < 1e-8,
                    "a={a} i={i} t={t}: selection {} vs central diff {want}",
                    d[t]
                );
            }
        }
    }
}

/// The exit refresh evaluates the loop's post-step iterate, which no trial
/// evaluation has passed, so a raw η outside the link's open domain can
/// reach [`evaluate_at_mode`]. It must refuse such a point instead of
/// reporting the `clamp_eta`-projected boundary point's deviance and
/// `log|A|` as the converged answer. Gamma-inverse here: `η = 1/μ`, so
/// `η ≤ 0` is outside the domain (`family::eta_infeasible`).
#[test]
fn evaluate_at_mode_refuses_an_infeasible_eta() {
    let (n, s) = (6usize, 2usize);
    let spec = SizedSpec {
        spec: crate::ModelSpec {
            family: Family::Gamma {
                link: crate::GammaLink::Inverse,
            },
            re: Some(crate::ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![],
            }),
        },
        n_primary: s,
        extra_counts: vec![],
        tail_fill: 0.0,
    };
    let g = LmmGroupings::from_cluster_spec_ext(&spec, &[], &[]);
    let k = g.k_total;
    let mut packed = PackedScratch::for_shape(&g, n, k);
    // The test drives `PackedFactor` directly with `sp: None`, so it sizes
    // the dense pair itself, as `fill_packed_cols` would.
    packed.set_factor(None, k);
    crate::sparse::fill_lambda_small(&[0.5], &g, &mut packed.lam_small);
    fill_m_vals(&mut packed, &g, &[], n);
    // Intercept-only primary: one nonzero per row, at that row's cluster.
    for (i, m) in packed.m_cols[..n].iter_mut().enumerate() {
        *m = (i % s) as u32;
    }
    let width = packed.width;
    let y = vec![1.0; n];
    let prior_w = vec![1.0; n];
    let u = vec![0.0; k];
    // u = 0, so η is η_fixed. The refused case has one row past the
    // boundary, the shape a post-step iterate takes when the step overshoots.
    let mut eta_fixed = vec![1.0; n];
    let mut eta = vec![0.0; n];
    let mut prob = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut a_rhs = vec![0.0; k];
    let mut layout = PackedFactor {
        m_cols: &packed.m_cols[..n * width],
        m_vals: &packed.m_vals[..n * width],
        a: &mut packed.a,
        a_chol: &mut packed.a_chol,
        a_llt_mem: &mut packed.a_llt_mem,
        a_rhs: &mut a_rhs[..],
        family: spec.spec.family,
        nb_theta: f64::NAN,
        width,
        k,
        sp: None,
    };
    let mut evaluates = |eta_fixed: &[f64]| {
        evaluate_at_mode(
            &mut layout,
            spec.spec.family,
            f64::NAN,
            &y,
            &prior_w,
            false,
            eta_fixed,
            &u,
            &mut eta,
            &mut prob,
            &mut w,
            n,
        )
        .is_some()
    };
    // Positive control: the same call on an all-feasible η evaluates, so the
    // refusal below is the infeasible row and nothing else.
    assert!(
        evaluates(&eta_fixed),
        "an all-feasible iterate must be evaluated"
    );
    eta_fixed[2] = -0.25;
    assert!(
        !evaluates(&eta_fixed),
        "a refreshed iterate with η outside the link's domain must not be reported as an evaluation"
    );
}
