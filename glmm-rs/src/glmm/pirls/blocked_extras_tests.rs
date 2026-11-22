use super::*;
use crate::glmm::workspace::{build_packed_m, GlmmWorkspace};
use crate::spec::{GammaLink, NegBinomialLink};

/// One dual kernel call settles the implicit-function lanes on a non-canonical
/// link: the kernel takes the observed-information step, so its first-order
/// lanes are the IFT answer after ONE call — not merely a passing `exact`
/// flag, which a deleted twin would still report (`d.exact` starts `true`
/// unconditionally and is only cleared on a non-PD twin factor, so
/// `assert!(dual.exact)` alone proves PD-ness and convergence, never that
/// the twin ran). This test proves the stronger claim: it runs the SAME
/// seeded start twice — once with `observed: true` (one call), once with
/// `observed: false` driven the way `run_gradient`'s own refinement loop
/// drives a Fisher-only fallback (repeated calls, `u` warm-started between
/// them, until the lanes stop moving, same `1e-10` relative settle band
/// `derivative.rs`'s `run_gradient`/`run_hessian` settle checks use) — and
/// asserts the one-call observed lanes (of `u` AND of the assembled Laplace
/// merit `dev + pen + 2·log|A|`) equal the many-call settled Fisher lanes.
/// Delete the twin (fall through to Fisher while still reporting
/// `observed: true`) and this fails: the one-call lanes are then only the
/// FIRST Fisher contraction step, nowhere near the settled answer.
#[test]
fn structured_extras_dual_step_is_exact_on_non_canonical_links() {
    use crate::dual::Dual;
    use crate::scalar::Scalar;
    type D4 = Dual<4>;
    // Lane budget of the chosen dual type, and the settle band
    // `run_gradient`'s/`run_hessian`'s own refinement loops use
    // (`derivative.rs`, the `(d[j] - prev_d[j]).abs() < 1e-10 * (1.0 +
    // d[j].abs())` checks — no named constant, so the literal is repeated
    // here with this pointer).
    const LANES: usize = 4;
    const SETTLE: f64 = 1e-10;

    // nested-2 + crossed-6: exercises both the core blocks and the Schur.
    let (x, y_bin, ids, extra_ids, spec0) = crate::glmm::tests::glmm_extras_q1_dataset(2, 6);
    let n = y_bin.len();
    let p = 2usize;

    for family in [
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        Family::Gamma {
            link: GammaLink::Log,
        },
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
    ] {
        // Per-family y, the way `tests.rs`'s `fixture_with_nagq_sized` does:
        // binomial reuses the dataset's own logit-generated y, Gamma/NB
        // regenerate off the same `0.2 + 0.8·x1` linear predictor.
        let y: Vec<f64> = match family {
            Family::Binomial { .. } => y_bin.clone(),
            Family::Gamma { .. } => (0..n)
                .map(|i| {
                    let eta: f64 = 0.2 + 0.8 * x[(i, 1)];
                    let noise = 0.4 + 1.6 * (((i as u64 * 2654435761) % 1000) as f64 / 1000.0);
                    eta.exp() * noise
                })
                .collect(),
            Family::NegativeBinomial { .. } => (0..n)
                .map(|i| {
                    let eta: f64 = 0.2 + 0.8 * x[(i, 1)];
                    let noise = 0.4 + 1.6 * (((i as u64 * 2654435761) % 1000) as f64 / 1000.0);
                    (eta.exp().min(20.0) * noise).round().max(0.0)
                })
                .collect(),
            other => panic!("exactness-gate fixture: family {other:?} not wired"),
        };

        let mut spec = spec0.clone();
        spec.spec.family = family;
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &spec, n, &[], 1);

        let q = ws.groupings.primary_q;
        let np = ws.groupings.nested_per_parent;
        let qc = q + np;
        let s = ws.groupings.n_primary;
        let e = ws.groupings.k_crossed();
        let k_family = qc * s;
        let k = k_family + e;
        let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
        let n_theta = ws.n_theta;
        let m = n_theta + p;
        // Same seeding window `run_gradient`'s first chunk uses (`base =
        // 0`, `width = LANES.min(m)`): unit first-derivative lanes on
        // coordinates `0..width` (theta first, then beta), plain values
        // past `width`. `m = 5` here (`n_theta = 3` + `p = 2`), so all 4
        // `Dual<4>` lanes land on the three theta coordinates plus
        // `beta[0]`; `beta[1]` carries no lane.
        let width = LANES.min(m);
        let raw: Vec<f64> = (0..n_theta).map(|_| 0.5).chain([0.2, 0.8]).collect();
        let seeded: Vec<D4> = (0..m)
            .map(|j| {
                let mut d = [0.0f64; LANES];
                if j < width {
                    d[j] = 1.0;
                }
                D4 { v: raw[j], d }
            })
            .collect();

        let mut lam = vec![D4::ZERO; q * q];
        let mut m_core_buf = vec![D4::ZERO; (n * qc).max(1)];
        let mut cross_theta = vec![D4::ZERO; g_cap];
        build_packed_m::<D4>(
            &ws.groupings,
            &seeded[..n_theta],
            &ws.z_buf,
            &extra_ids,
            &mut lam,
            &ids,
            &mut m_core_buf,
            &mut cross_theta,
            &mut ws.pattern.cross_col,
            &mut ws.pattern.n_cross,
            n,
        );
        build_coupling_csr(
            &ids,
            &ws.pattern.cross_col,
            &ws.pattern.n_cross,
            s,
            n,
            &mut ws.pattern.coup_cols,
            &mut ws.pattern.coup_ptr,
            &mut ws.pattern.coup_pos,
        );
        // `params`/`lam`/`m_core_buf`/`cross_theta` stay fixed for every
        // call below — only `u` moves between calls, the same invariant
        // `run_gradient`'s own refinement loop relies on.
        let beta: Vec<D4> = seeded[n_theta..].to_vec();

        let fresh_dual = |observed: bool| DualStep {
            observed,
            obs_blocks: vec![D4::ZERO; (s * q * q).max(1)],
            obs_core_blocks: vec![D4::ZERO; (qc * qc * s).max(1)],
            obs_coupling: vec![D4::ZERO; (qc * s * e).max(1)],
            obs_schur_blk: vec![D4::ZERO; (e * e).max(1)],
            obs_rhs: vec![D4::ZERO; k],
            min_iters: 0,
            exact: false,
        };
        // One `pirls_solve_blocked_extras` call at the given `u`/`dual`;
        // every other buffer is fresh per call (PIRLS zeroes what it needs
        // internally, so reuse would be an optimization, not a
        // correctness requirement — freshness here just keeps this
        // closure simple to read).
        let mut run_call = |u: &mut [D4], dual: &mut DualStep<D4>| -> (D4, D4, D4, bool) {
            let mut beta_local = beta.clone();
            let mut scratch = PirlsScratch {
                eta: vec![D4::ZERO; n],
                prob: vec![D4::ZERO; n],
                w: vec![D4::ZERO; n],
                eta_fixed: vec![D4::ZERO; n],
                mu: vec![D4::ZERO; n],
                m_buf: vec![D4::ZERO; 1], // unused on the extras path
                lam: vec![D4::ZERO; 1],   // unused on the extras path
                u: u.to_vec(),
                u_prev: vec![D4::ZERO; k.max(1)],
                a_rhs: vec![D4::ZERO; k],
                a_blocks: vec![D4::ZERO; 1], // unused on the extras path
                a_half_logdet: vec![D4::ZERO; 1], // unused on the extras path
                a_excess: vec![D4::ZERO; 1], // unused on the extras path
                agq_scratch: vec![D4::ZERO; 1], // unused on the extras path
                x_rows: Vec::new(),
                bt_rows: Vec::new(),
                agq_rows: Vec::new(),
                agq_slow: Vec::new(),
                pen_dev: f64::NAN,
            };
            let mut structured = StructuredScratch {
                core_blocks: vec![D4::ZERO; (qc * qc * s).max(1)],
                coupling: vec![D4::ZERO; (qc * s * e).max(1)],
                schur_blk: vec![D4::ZERO; (e * e).max(1)],
                m_core_buf: m_core_buf.clone(),
                cross_theta: cross_theta.clone(),
            };
            let mut wx = Mat::<f64>::zeros(n, p);
            let mut counters = crate::counters::EvalCounters::new();
            let result = pirls_solve_blocked_extras::<D4>(
                family,
                4.0,
                &ws.groupings,
                &ids,
                x.as_ref(),
                &y,
                &ws.prior_w[..n],
                false,
                &mut beta_local,
                BetaStep::Fixed,
                &mut scratch,
                &mut structured,
                &mut ws.pattern,
                Some(dual),
                &mut wx,
                None,
                None,
                n,
                &mut counters,
            );
            u.copy_from_slice(&scratch.u);
            result
        };
        // The merit this test compares, in addition to `u` itself: the
        // same Laplace assembly `structured_laplace_deviance` builds off
        // this function's three return values.
        let merit = |dev: D4, pen: D4, logdet: D4| dev + pen + D4::from_f64(2.0) * logdet;
        let lanes_of = |v: &[D4]| -> Vec<[f64; LANES]> { v.iter().map(|x| x.d).collect() };

        // (i) one call, observed information.
        let mut u_obs = vec![D4::ZERO; k];
        let mut dual_obs = fresh_dual(true);
        let (dev_obs, pen_obs, logdet_obs, conv_obs) = run_call(&mut u_obs, &mut dual_obs);
        assert!(
            conv_obs,
            "{family:?}: one-call observed PIRLS did not converge"
        );
        assert!(
            dual_obs.exact,
            "{family:?}: dual step not exact on first call"
        );
        let u_obs_lanes = lanes_of(&u_obs);
        let merit_obs_lanes = merit(dev_obs, pen_obs, logdet_obs).d;

        // (ii) repeated Fisher-only calls, `u` warm-started between them
        // (never reset), until both `u`'s lanes and the merit's lanes
        // stop moving — mirrors `run_gradient`'s settle check, without its
        // `dual.exact` early exit: `d.exact` reports `true` unconditionally
        // whenever `observed` is `false` (nothing downgrades it), so it is
        // not a valid stopping signal for this deliberately-Fisher-only
        // sweep the way it is for the routing `run_gradient` actually uses.
        let mut u_fisher = vec![D4::ZERO; k];
        let mut prev: Option<(Vec<[f64; LANES]>, [f64; LANES])> = None;
        let mut settled = false;
        for call_idx in 0..crate::glmm::derivative::MAX_DUAL_REFINEMENTS {
            let mut dual_fisher = fresh_dual(false);
            let (dev_f, pen_f, logdet_f, conv_f) = run_call(&mut u_fisher, &mut dual_fisher);
            assert!(
                conv_f,
                "{family:?}: Fisher-only PIRLS call {call_idx} did not converge"
            );
            let cur_u = lanes_of(&u_fisher);
            let cur_merit = merit(dev_f, pen_f, logdet_f).d;
            if let Some((prev_u, prev_merit)) = &prev {
                let u_settled = cur_u.iter().zip(prev_u.iter()).all(|(c, p)| {
                    (0..width).all(|j| (c[j] - p[j]).abs() < SETTLE * (1.0 + c[j].abs()))
                });
                let merit_settled = (0..width).all(|j| {
                    (cur_merit[j] - prev_merit[j]).abs() < SETTLE * (1.0 + cur_merit[j].abs())
                });
                if u_settled && merit_settled {
                    settled = true;
                    break;
                }
            }
            prev = Some((cur_u, cur_merit));
        }
        assert!(
            settled,
            "{family:?}: Fisher-only lanes never settled within MAX_DUAL_REFINEMENTS calls"
        );
        let (u_fisher_lanes, merit_fisher_lanes) =
            prev.expect("settled ⇒ at least one recorded call");

        // (iii) the one-call observed lanes must equal the settled
        // many-call Fisher lanes — the pin a deleted twin cannot pass.
        for (idx, (a, b)) in u_obs_lanes.iter().zip(u_fisher_lanes.iter()).enumerate() {
            for j in 0..width {
                assert!(
                    (a[j] - b[j]).abs() < SETTLE * (1.0 + a[j].abs()),
                    "{family:?}: u[{idx}] lane {j}: one-call observed {} vs settled Fisher {}",
                    a[j],
                    b[j]
                );
            }
        }
        for j in 0..width {
            assert!(
                (merit_obs_lanes[j] - merit_fisher_lanes[j]).abs()
                    < SETTLE * (1.0 + merit_obs_lanes[j].abs()),
                "{family:?}: merit lane {j}: one-call observed {} vs settled Fisher {}",
                merit_obs_lanes[j],
                merit_fisher_lanes[j]
            );
        }
    }
}

/// The `f64` override IS the extras path's tail code, so there is no
/// second copy left to `==` it against. What this test can
/// check is the other arm: the generic default body against the `f64`
/// override on the same `S`. They are NOT expected to match to bits: the
/// sparse arm factors under an AMD ordering and the dense default in the
/// natural one, which is a reassociation of the same Cholesky. The assert
/// bands them and records the measured gap; the two are the same matrix, not
/// the same rounding.
#[test]
fn tail_default_body_is_the_sparse_arms_oracle() {
    let (_x, y, ids, extra_ids, spec) = crate::glmm::tests::glmm_extras_q1_dataset(0, 6);
    let n = y.len();
    let ws = GlmmWorkspace::for_cluster_spec(2, &spec, n, &[], 1);
    let g = &ws.groupings;
    let e = g.k_crossed();
    let mut ss = StructuredSchur::new(g, &ids, &extra_ids, n)
        .unwrap()
        .expect("6 crossed levels ⇒ e > 0 ⇒ Some");

    // A deterministic, diagonally-dominant SPD matrix built ONLY on the real
    // crossed-incidence pattern `ss` carries for this fixture (the same AMD
    // symbolic factor the production path builds) — off-pattern entries stay
    // 0.0 so the dense arm (reads the full e×e block) and the sparse arm
    // (reads only the pattern) factor the identical matrix.
    let mut s_raw = vec![0.0f64; e * e];
    for a in 0..e {
        s_raw[a * e + a] = 10.0 + 4.0 * a as f64;
    }
    {
        let sym = ss.axx.symbolic();
        let col_ptr = sym.col_ptr();
        let row_idx = sym.row_idx();
        for b in 0..e {
            for &a in &row_idx[col_ptr[b]..col_ptr[b + 1]] {
                if a != b {
                    s_raw[a * e + b] = 0.1;
                }
            }
        }
    }

    let mut s_dense = s_raw.clone();
    let mut s_sparse = s_raw.clone();

    let ld_dense = tail_factor_generic::<f64>(&mut s_dense, e).expect("dense PD");
    let ld_sparse =
        <f64 as TailKernel>::tail_factor(&mut s_sparse, e, Some(&mut ss)).expect("sparse PD");
    // Measured on this fixture (e=6, 8 primary clusters, near-complete crossed
    // coupling): dense (natural order) vs sparse (AMD order) Cholesky of the
    // same S agree in the half-log-determinant to within 1e-9 — the two
    // orderings are a reassociation of the same sum, not independent
    // computations.
    assert!(
        (ld_dense - ld_sparse).abs() < 1e-9,
        "half-logdet drifted: dense {ld_dense} sparse {ld_sparse}"
    );

    let mut rhs_dense = vec![1.0f64; e];
    let mut rhs_sparse = vec![1.0f64; e];
    tail_solve_generic::<f64>(&s_dense, e, &mut rhs_dense);
    <f64 as TailKernel>::tail_solve(&s_sparse, e, Some(&mut ss), &mut rhs_sparse);
    for i in 0..e {
        assert!(
            (rhs_dense[i] - rhs_sparse[i]).abs() < 1e-9,
            "tail_solve[{i}] drifted: dense {} sparse {}",
            rhs_dense[i],
            rhs_sparse[i]
        );
    }
}
