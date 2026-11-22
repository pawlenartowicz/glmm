use super::*;
use crate::dual::Dual;
use crate::fit::SizedSpec;

/// `packed_m_theta_deriv`'s primary-vech selection — including the
/// off-diagonal (intercept-slope covariance) slot, which a diagonal-only
/// check would miss — held equal to a central difference of
/// `build_packed_m` itself (exact to round-off: `Λ` is linear in θ). No
/// extras on this shape, so `qc == q` and the crossed output is empty —
/// the blocked-path packing this same selection rule serves.
#[test]
fn packed_m_theta_deriv_matches_central_diff_primary_vech() {
    let n = 16;
    let n_prim = 4;
    let spec = SizedSpec {
        spec: crate::ModelSpec {
            family: crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            re: Some(crate::ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        },
        n_primary: n_prim,
        extra_counts: vec![],
        tail_fill: 0.0,
    };
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut ids = vec![0u32; n];
    for i in 0..n {
        ids[i] = (i % n_prim) as u32;
        x[(i, 0)] = 1.0;
        x[(i, 1)] = 0.3 + 0.11 * i as f64;
    }
    let g = LmmGroupings::from_cluster_spec(&spec, &[1]);
    let n_theta = g.n_theta();
    let q = g.primary_q;
    let qc = q; // no extras on this shape
    let mut z_buf = vec![0.0; n * (q - 1)];
    fill_z_f64(&g, x.as_ref(), x.as_ref().subcols(0, 0), &mut z_buf, n);
    let extra_ids: Vec<Vec<u32>> = vec![];
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    let mut lam = vec![0.0; q * q];
    let mut m_core = vec![0.0; n * qc];
    let mut cross_theta = vec![0.0; g_cap];
    let mut cross_col = vec![0u32; n * g_cap];
    let mut n_cross = vec![0u8; n];
    let mut params: Vec<f64> = (0..n_theta).map(|k| 0.4 + 0.1 * k as f64).collect();
    let eps = 1e-6;

    for a in 0..n_theta {
        let base = params[a];
        params[a] = base + eps;
        build_packed_m(
            &g,
            &params,
            &z_buf,
            &extra_ids,
            &mut lam,
            &ids,
            &mut m_core,
            &mut cross_theta,
            &mut cross_col,
            &mut n_cross,
            n,
        );
        let m_plus = m_core.clone();
        params[a] = base - eps;
        build_packed_m(
            &g,
            &params,
            &z_buf,
            &extra_ids,
            &mut lam,
            &ids,
            &mut m_core,
            &mut cross_theta,
            &mut cross_col,
            &mut n_cross,
            n,
        );
        let m_minus = m_core.clone();
        params[a] = base;

        let mut m_core_d = vec![0.0; qc];
        let mut cross_val_d: Vec<f64> = vec![];
        for i in [0usize, 1, n - 1] {
            packed_m_theta_deriv(
                &g,
                a,
                &params,
                true,
                &z_buf,
                &extra_ids,
                &ids,
                i,
                &mut m_core_d,
                &mut cross_val_d,
            );
            for c in 0..qc {
                let want = (m_plus[i * qc + c] - m_minus[i * qc + c]) / (2.0 * eps);
                assert!(
                    (m_core_d[c] - want).abs() < 1e-8,
                    "a={a} i={i} c={c}: deriv {} vs central diff {want}",
                    m_core_d[c]
                );
            }
        }
    }
}

/// `packed_m_theta_deriv`'s nested and crossed selections, on
/// `glmm_extras_q1_dataset`'s nested2+crossed3 shape (`q_core = 3`, one
/// crossed grouping) — the twin of the primary-vech test above.
#[test]
fn packed_m_theta_deriv_matches_central_diff_extras() {
    let (x, y, ids, extra_ids, spec) = crate::glmm::tests::glmm_extras_q1_dataset(2, 3);
    let n = y.len();
    let g = LmmGroupings::from_cluster_spec(&spec, &[]);
    let n_theta = g.n_theta();
    let q = g.primary_q;
    let qc = q + g.nested_per_parent;
    let z_buf: Vec<f64> = vec![0.0; n * q.saturating_sub(1)];
    let _ = x; // only ids/extra_ids drive this shape's packing (q_p = 1)
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    let mut lam = vec![0.0; q * q];
    let mut m_core = vec![0.0; n * qc];
    let mut cross_theta = vec![0.0; g_cap];
    let mut cross_col = vec![0u32; n * g_cap];
    let mut n_cross = vec![0u8; n];
    let mut params: Vec<f64> = (0..n_theta).map(|k| 0.3 + 0.05 * k as f64).collect();
    let eps = 1e-6;

    for a in 0..n_theta {
        let base = params[a];
        params[a] = base + eps;
        build_packed_m(
            &g,
            &params,
            &z_buf,
            &extra_ids,
            &mut lam,
            &ids,
            &mut m_core,
            &mut cross_theta,
            &mut cross_col,
            &mut n_cross,
            n,
        );
        let m_plus = m_core.clone();
        let cv_plus = cross_theta.clone();
        params[a] = base - eps;
        build_packed_m(
            &g,
            &params,
            &z_buf,
            &extra_ids,
            &mut lam,
            &ids,
            &mut m_core,
            &mut cross_theta,
            &mut cross_col,
            &mut n_cross,
            n,
        );
        let m_minus = m_core.clone();
        let cv_minus = cross_theta.clone();
        params[a] = base;

        let mut m_core_d = vec![0.0; qc];
        let mut cross_val_d = vec![0.0; g.crossed.len()];
        for i in [0usize, 1, n - 1] {
            packed_m_theta_deriv(
                &g,
                a,
                &params,
                true,
                &z_buf,
                &extra_ids,
                &ids,
                i,
                &mut m_core_d,
                &mut cross_val_d,
            );
            for c in 0..qc {
                let want = (m_plus[i * qc + c] - m_minus[i * qc + c]) / (2.0 * eps);
                assert!(
                    (m_core_d[c] - want).abs() < 1e-8,
                    "a={a} i={i} c={c}: deriv {} vs central diff {want}",
                    m_core_d[c]
                );
            }
            assert_eq!(
                n_cross[i] as usize,
                g.crossed.len(),
                "no pinning on this fixture"
            );
            for z in 0..g.crossed.len() {
                let want = (cv_plus[z] - cv_minus[z]) / (2.0 * eps);
                assert!(
                    (cross_val_d[z] - want).abs() < 1e-8,
                    "a={a} i={i} z={z}: deriv {} vs central diff {want}",
                    cross_val_d[z]
                );
            }
        }
    }
}

/// `packed_m_theta_deriv`'s crossed pin rule matches `build_packed_m` on
/// BOTH arms, on a shape with two crossed groupings where the first is
/// pinned (θ=0) and the second is active. At `f64` (`skip_pinned = true`)
/// the packed row carries exactly one crossed entry — the active
/// grouping's — and `cross_val_d[0]` must pair with THAT entry, not a slot
/// reserved for the pinned one. At `Dual<1>` (`skip_pinned = false`) the
/// packed row carries both columns in declaration order, and each
/// grouping's derivative must land on its own slot.
#[test]
fn packed_m_theta_deriv_pairs_with_pinned_crossed_packing() {
    let n_prim = 4;
    let n = 24;
    let n_c1 = 3;
    let n_c2 = 4;
    let mut x = Mat::<f64>::zeros(n, 1);
    let mut ids = vec![0u32; n];
    let mut c1 = vec![0u32; n];
    let mut c2 = vec![0u32; n];
    for i in 0..n {
        ids[i] = (i % n_prim) as u32;
        x[(i, 0)] = 1.0;
        c1[i] = (i % n_c1) as u32;
        c2[i] = ((i / n_c1) % n_c2) as u32;
    }
    let spec = SizedSpec {
        spec: crate::ModelSpec {
            family: crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            re: Some(crate::ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![
                    crate::Grouping {
                        terms: Vec::new(),
                        relation: crate::GroupingRelation::Crossed,
                        slopes: vec![],
                    },
                    crate::Grouping {
                        terms: Vec::new(),
                        relation: crate::GroupingRelation::Crossed,
                        slopes: vec![],
                    },
                ],
            }),
        },
        n_primary: n_prim,
        extra_counts: vec![n_c1, n_c2],
        tail_fill: 0.0,
    };
    let extra_ids = vec![c1, c2];
    let _ = x;
    let g = LmmGroupings::from_cluster_spec(&spec, &[]);
    let n_theta = g.n_theta();
    assert_eq!(n_theta, 3, "primary + two crossed scalars");
    let q = g.primary_q;
    let qc = q; // no nested
    let z_buf: Vec<f64> = vec![];
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    let mut lam = vec![0.0; q * q];
    let mut m_core = vec![0.0; n * qc];
    let mut cross_theta = vec![0.0; g_cap];
    let mut cross_col = vec![0u32; n * g_cap];
    let mut n_cross = vec![0u8; n];
    let cross1_theta = g.crossed[0].vech_start;
    let cross2_theta = g.crossed[1].vech_start;
    let mut params = vec![0.0; n_theta];
    params[0] = 0.5; // primary
    params[cross1_theta] = 0.0; // pinned
    params[cross2_theta] = 0.7; // active
    let eps = 1e-6;

    build_packed_m(
        &g,
        &params,
        &z_buf,
        &extra_ids,
        &mut lam,
        &ids,
        &mut m_core,
        &mut cross_theta,
        &mut cross_col,
        &mut n_cross,
        n,
    );
    let n_cross_base = n_cross.clone();
    for &i in &[0usize, 1, n - 1] {
        assert_eq!(
            n_cross_base[i] as usize, 1,
            "cross1 pinned ⇒ only cross2's column is packed"
        );
    }

    // Perturb only the active grouping's θ — cross1 stays exactly pinned
    // at 0.0 across the step, so the packed width (n_cross) does not
    // change and a plain central difference is valid.
    params[cross2_theta] = 0.7 + eps;
    build_packed_m(
        &g,
        &params,
        &z_buf,
        &extra_ids,
        &mut lam,
        &ids,
        &mut m_core,
        &mut cross_theta,
        &mut cross_col,
        &mut n_cross,
        n,
    );
    let cv_plus = cross_theta.clone();
    params[cross2_theta] = 0.7 - eps;
    build_packed_m(
        &g,
        &params,
        &z_buf,
        &extra_ids,
        &mut lam,
        &ids,
        &mut m_core,
        &mut cross_theta,
        &mut cross_col,
        &mut n_cross,
        n,
    );
    let cv_minus = cross_theta.clone();
    params[cross2_theta] = 0.7;

    let mut m_core_d = vec![0.0; qc];
    for &i in &[0usize, 1, n - 1] {
        assert_eq!(n_cross[i] as usize, 1);
        let want = (cv_plus[0] - cv_minus[0]) / (2.0 * eps);
        let mut active_d = vec![0.0; 1];
        packed_m_theta_deriv(
            &g,
            cross2_theta,
            &params,
            true,
            &z_buf,
            &extra_ids,
            &ids,
            i,
            &mut m_core_d,
            &mut active_d,
        );
        assert!(
            (active_d[0] - want).abs() < 1e-8,
            "i={i}: deriv {} vs central diff {want}",
            active_d[0]
        );
        // The pinned grouping's θ never claims a slot: differentiating
        // wrt an unrelated (primary) θ must leave the one active
        // position at zero, not shift the pinned grouping into it.
        let mut zero_d = vec![0.0; 1];
        packed_m_theta_deriv(
            &g,
            0,
            &params,
            true,
            &z_buf,
            &extra_ids,
            &ids,
            i,
            &mut m_core_d,
            &mut zero_d,
        );
        assert_eq!(
            zero_d[0], 0.0,
            "i={i}: unrelated θ must not claim the active slot"
        );
    }

    // --- the dual arm of the same rule: `build_packed_m` at `Dual<1>`
    // KEEPS the pinned grouping's column (value exactly 0.0), so the
    // packed row is two wide and `skip_pinned = false` must walk both
    // groupings in declaration order. ---
    let params_d: Vec<Dual<1>> = params.iter().map(|&v| Dual::<1> { v, d: [0.0] }).collect();
    let mut lam_d = vec![Dual::<1> { v: 0.0, d: [0.0] }; q * q];
    let mut m_core_dual = vec![Dual::<1> { v: 0.0, d: [0.0] }; n * qc];
    let mut cross_val_dual = vec![Dual::<1> { v: 0.0, d: [0.0] }; g_cap];
    let mut cross_col_dual = vec![0u32; n * g_cap];
    let mut n_cross_dual = vec![0u8; n];
    build_packed_m(
        &g,
        &params_d,
        &z_buf,
        &extra_ids,
        &mut lam_d,
        &ids,
        &mut m_core_dual,
        &mut cross_val_dual,
        &mut cross_col_dual,
        &mut n_cross_dual,
        n,
    );
    let k_family = qc * g.n_primary;
    let mut cvd = vec![0.0; 2];
    for &i in &[0usize, 1, n - 1] {
        assert_eq!(
            n_cross_dual[i] as usize, 2,
            "i={i}: the dual packer keeps the pinned grouping's column"
        );
        for (decl, ids_of) in [(0usize, &extra_ids[0]), (1usize, &extra_ids[1])] {
            let want_col = (g.extra_offsets[decl] + ids_of[i] as usize - k_family) as u32;
            assert_eq!(
                cross_col_dual[i * g_cap + decl],
                want_col,
                "i={i} slot {decl}: declaration order"
            );
        }
        assert_eq!(cross_val_dual[0].v, 0.0, "i={i}: pinned θ is 0");
        assert_eq!(cross_val_dual[1].v, 0.7, "i={i}: active θ");

        packed_m_theta_deriv(
            &g,
            cross1_theta,
            &params,
            false,
            &z_buf,
            &extra_ids,
            &ids,
            i,
            &mut m_core_d,
            &mut cvd,
        );
        assert_eq!(
            (cvd[0], cvd[1]),
            (1.0, 0.0),
            "i={i}: the pinned grouping owns slot 0 on the dual arm"
        );
        packed_m_theta_deriv(
            &g,
            cross2_theta,
            &params,
            false,
            &z_buf,
            &extra_ids,
            &ids,
            i,
            &mut m_core_d,
            &mut cvd,
        );
        assert_eq!(
            (cvd[0], cvd[1]),
            (0.0, 1.0),
            "i={i}: the active grouping keeps slot 1 on the dual arm"
        );
    }
}
