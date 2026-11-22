use super::*;
use crate::fit::SizedSpec;

/// A `HyperDual` rung carries no assembly scratch: the assembled pass runs on
/// the `Dual` rungs alone, and `m·k` second-order numbers would be megabytes
/// on a wide `k`.
#[test]
fn for_shape_gives_a_hyperdual_rung_no_assembly_scratch() {
    let (m, p, k, rows, s, q_p, nagq) = (6usize, 2usize, 18usize, 50usize, 9usize, 2usize, 5u8);
    // No-extras shape: q_core == q_p, e == 0, so the structured twins sit
    // at their `.max(1)` minimum.
    let (q_core, e) = (q_p, 0usize);
    let cluster_ids: Vec<u32> = (0..rows as u32).map(|i| i % s as u32).collect();
    let scratch = GlmmDualScratch::for_shape(
        NLanes::H8,
        m,
        p,
        k,
        rows,
        s,
        q_p,
        q_core,
        e,
        false,
        nagq,
        true,
        0,
        &cluster_ids,
        None,
    );
    match scratch.lanes {
        DualLanes::H8(bufs) => {
            // A `HyperDual` rung carries NO assembly scratch: the assembled
            // pass runs on the `Dual` rungs alone, and `m·k` second-order
            // numbers would be megabytes on a wide `k`.
            assert!(bufs.asm.tail_inv.is_empty());
            assert!(bufs.asm.tail_col.is_empty());
            assert!(bufs.asm.d_gamma.is_empty());
            assert!(bufs.asm.l_gamma.is_empty());
            assert!(bufs.asm.d_u.is_empty());
            assert!(bufs.asm.l_u.is_empty());
            assert!(bufs.asm.g_gamma.is_empty());
            assert!(bufs.asm.adj.is_empty());
            assert!(bufs.asm.tail_h.is_empty());
            assert!(bufs.asm.tail_g.is_empty());
            assert!(bufs.asm.obs_core.is_empty());
            assert!(bufs.asm.obs_coup.is_empty());
            assert!(bufs.asm.obs_schur.is_empty());
        }
        _ => panic!("expected H8 variant"),
    }
}

/// A canonical link never reads an observed twin, so `for_shape` gives it
/// none — on a wide crossed shape `obs_schur_blk` (`e²`) alone is
/// megabytes. The reuse check must pin the same thing, or a
/// canonical-sized scratch could be handed to a call that indexes them.
#[test]
fn for_shape_leaves_observed_twins_empty_on_a_canonical_link() {
    let (m, p, k, rows, s, q_p, nagq) = (5usize, 3usize, 7usize, 40usize, 7usize, 1usize, 7u8);
    let (q_core, e) = (q_p + 2, 6usize);
    let cluster_ids: Vec<u32> = (0..rows as u32).map(|i| i % s as u32).collect();
    let scratch = GlmmDualScratch::for_shape(
        NLanes::D8,
        m,
        p,
        k,
        rows,
        s,
        q_p,
        q_core,
        e,
        true,
        nagq,
        false,
        0,
        &cluster_ids,
        None,
    );
    match &scratch.lanes {
        DualLanes::D8(bufs) => {
            assert!(bufs.dual.obs_blocks.is_empty());
            assert!(bufs.dual.obs_core_blocks.is_empty());
            assert!(bufs.dual.obs_coupling.is_empty());
            assert!(bufs.dual.obs_schur_blk.is_empty());
            assert!(bufs.dual.obs_rhs.is_empty());
            // The Fisher buffers are untouched by the twin sizing.
            assert_eq!(
                bufs.structured.coupling.len(),
                crate::glmm::workspace::coupling_len(rows, s, q_core, e)
            );
            assert_eq!(bufs.structured.schur_blk.len(), e * e);
            // The assembly's observed factor follows the same rule.
            assert!(bufs.asm.obs_core.is_empty());
            assert!(bufs.asm.obs_coup.is_empty());
            assert!(bufs.asm.obs_schur.is_empty());
        }
        _ => panic!("expected D8 variant"),
    }
    assert!(scratch.matches_shape(m, p, k, rows, s, q_p, q_core, e, true, nagq, false, 0, None));
    assert!(!scratch.matches_shape(m, p, k, rows, s, q_p, q_core, e, true, nagq, true, 0, None));
}

/// `for_shape` then `matches_shape` round-trips on every layout × link,
/// and a request that differs only in `extras` or only in `observed` is
/// refused. The nested-only structured shape (`e = 0`, `q_core = q_p + np`)
/// against an extras-free shape at the same `q_core` is the pair the
/// `m_core_buf` length check exists for: `build_packed_m` writes
/// `m_core_buf[i·q_core + …]` on every row, so handing it the extras-free
/// minimum would be an out-of-bounds write.
#[test]
fn for_shape_round_trips_and_refuses_extras_or_observed_mismatch() {
    let (m, p, k, rows, s, nagq) = (6usize, 2usize, 30usize, 40usize, 5usize, 1u8);
    let cluster_ids: Vec<u32> = (0..rows as u32).map(|i| i % s as u32).collect();
    // (q_p, q_core, e, extras, packed_width)
    let shapes = [
        (2usize, 2usize, 0usize, false, 0usize), // blocked, no extras
        (1, 3, 6, true, 0),                      // structured, nested + crossed
        (1, 3, 0, true, 0),                      // structured, nested only
        (3, 3, 0, false, 0),                     // blocked at the nested shape's q_core
        (1, 1, 0, false, 4),                     // packed
    ];
    for &(q_p, q_core, e, extras, width) in &shapes {
        for observed in [false, true] {
            let sc = GlmmDualScratch::for_shape(
                NLanes::D8,
                m,
                p,
                k,
                rows,
                s,
                q_p,
                q_core,
                e,
                extras,
                nagq,
                observed,
                width,
                &cluster_ids,
                None,
            );
            let same = |x: bool, o: bool| {
                sc.matches_shape(m, p, k, rows, s, q_p, q_core, e, x, nagq, o, width, None)
            };
            assert!(
                same(extras, observed),
                "{q_p} {q_core} {e} {extras} {width}"
            );
            assert!(
                !same(extras, !observed),
                "{q_p} {q_core} {e} {extras} {width}"
            );
            if width == 0 {
                // Packed sizes the structured twin at its minimum whatever
                // `extras` says, so only the dense layouts can disagree.
                assert!(!same(!extras, observed), "{q_p} {q_core} {e} {extras}");
            }
        }
    }
    // The pair the `m_core_buf` check exists for: nested-only structured
    // (q_p = 1, q_core = 3) vs a request at the same q_core, e, without
    // extras.
    let nested = GlmmDualScratch::for_shape(
        NLanes::D8,
        m,
        p,
        k,
        rows,
        s,
        1,
        3,
        0,
        true,
        nagq,
        false,
        0,
        &cluster_ids,
        None,
    );
    assert!(!nested.matches_shape(m, p, k, rows, s, 1, 3, 0, false, nagq, false, 0, None));
}

/// Every `m` in `0..=MAX_DUAL_N` resolves to the smallest instantiated rung
/// at or above it — the padding an oversized rung would cost is the whole
/// reason the intermediate rungs exist, so a bucket that rounds past a
/// rung is a silent 2× on that cell.
#[test]
fn nlanes_pick_rounds_up_to_the_smallest_covering_rung() {
    let want = |m: usize| match m {
        0..=4 => 4,
        5 => 5,
        6 => 6,
        7..=8 => 8,
        _ => 12,
    };
    for m in 0..=MAX_DUAL_N {
        let n = |l: NLanes| match l {
            NLanes::D4 | NLanes::H4 => 4,
            NLanes::D5 | NLanes::H5 => 5,
            NLanes::D6 | NLanes::H6 => 6,
            NLanes::D8 | NLanes::H8 => 8,
            NLanes::D12 | NLanes::H12 => 12,
        };
        assert_eq!(
            n(NLanes::pick(m, false).unwrap()),
            want(m),
            "gradient m={m}"
        );
        assert_eq!(n(NLanes::pick(m, true).unwrap()), want(m), "hessian m={m}");
    }
}

/// `pick_assembled` hands the assembled Hessian a first-order (`D*`) rung for
/// every m: an `H*` rung makes `assembled.rs` return `Unsupported` and the SE
/// silently falls back to the FD stencil. Above the measured range (24) it is
/// the gradient ladder's pick. The measured table itself is a tuning choice,
/// not pinned here.
#[test]
fn nlanes_pick_assembled_stays_on_first_order_rungs() {
    let lanes = |l: NLanes| match l {
        NLanes::D4 => 4,
        NLanes::D5 => 5,
        NLanes::D6 => 6,
        NLanes::D8 => 8,
        NLanes::D12 => 12,
        other => panic!("first-order rung expected, got {other:?}"),
    };
    for m in 1..=100 {
        let l = NLanes::pick_assembled(m);
        let _ = lanes(l); // panics on an H rung: assembled.rs refuses those and SEs would fall to FD
        if m > 24 {
            assert_eq!(
                Some(l),
                NLanes::pick(m, false),
                "m={m}: unmeasured m keeps the gradient ladder"
            );
        }
    }
}

/// Build the workspace, `z_buf` and the `f64` starting state for one
/// `glmm_extras_q1_dataset` shape. Shared by the routing tests below, which
/// only care about which branch the entry point takes — the numbers
/// themselves are the FD gates' job (`glmm/tests.rs`).
#[allow(clippy::type_complexity)]
fn extras_routing_fixture(
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
    let (x, y, ids, extra_ids, spec) = crate::glmm::tests::glmm_extras_q1_dataset(np, n_crossed);
    let (n, p) = (y.len(), 2usize);
    let ws = GlmmWorkspace::for_cluster_spec(p, &spec, n, &[], 1);
    (ws, x, y, ids, extra_ids, p, n)
}

/// An oversized core — `primary_q + nested_per_parent > MAX_PRIMARY_Q`, the
/// shape `laplace_deviance` sends to the packed-row layout — has
/// no structured kernel to differentiate, so `supports_shape` rejects it
/// and the entry point returns `Unsupported` without allocating scratch.
#[test]
fn laplace_gradient_oversized_core_is_unsupported() {
    // q_core = primary_q(1) + nested children per parent (MAX_PRIMARY_Q) —
    // one past the cap by construction, so a wider cap cannot silently
    // make this shape eligible again.
    let np = crate::lmm::MAX_PRIMARY_Q;
    let (mut ws, x, y, ids, extra_ids, p, n) = extras_routing_fixture(np, 0);
    assert!(!ws.groupings.structured_extras_eligible());
    assert!(!supports_shape(ws.layout, &ws.groupings));
    let mut grad = vec![0.0f64; ws.n_theta + p];
    let status = laplace_gradient(&mut ws, x.as_ref(), &y, &ids, &extra_ids, p, n, &mut grad);
    assert!(matches!(status, DerivStatus::Unsupported));
    assert!(ws.dual_scratch.is_none());
}

/// A crossed tail one level past `DUAL_TAIL_MAX` is `Unsupported`, and the
/// clause that refuses it is `supports_shape`'s `Structured` tail clause:
/// the design stays on the structured layout (its dense tail fits), so
/// the layout alone does not refuse it. Built RELATIVE to the constant, so
/// re-pinning it moves this fixture with it rather than turning the test
/// into a silent pass.
#[test]
fn laplace_gradient_tail_past_boundary_is_unsupported() {
    let (mut ws, x, y, ids, extra_ids, p, n) = extras_routing_fixture(0, DUAL_TAIL_MAX + 1);
    assert_eq!(ws.groupings.k_crossed(), DUAL_TAIL_MAX + 1);
    assert!(ws.groupings.structured_extras_eligible());
    assert_eq!(ws.layout, GlmmLayout::Structured);
    assert!(!supports_shape(ws.layout, &ws.groupings));
    let mut grad = vec![0.0f64; ws.n_theta + p];
    let status = laplace_gradient(&mut ws, x.as_ref(), &y, &ids, &extra_ids, p, n, &mut grad);
    assert!(matches!(status, DerivStatus::Unsupported));
    assert!(ws.dual_scratch.is_none());
}

/// `laplace_hessian`'s own cap guard. The gradient does not refuse above the
/// cap: it chunks instead (`glmm::tests::dual_gradient_matches_central_fd_chunked_above_n12`).
/// The cap is this pass's alone: `fit::glmm_tests`'s
/// `assembled_hessian_covers_an_m_above_the_dual_lane_cap` holds the other
/// half, that an exact Hessian still reaches a dense shape above it.
#[test]
fn laplace_hessian_m_above_cap_is_unsupported() {
    assert!(NLanes::pick(MAX_DUAL_N + 1, true).is_none());

    let mut model = SizedSpec {
        spec: crate::test_support::intercept_only_spec(),
        n_primary: 3,
        extra_counts: vec![],
        tail_fill: 0.0,
    };
    model.spec.family = crate::Family::Binomial {
        link: crate::BinomialLink::Logit,
    };
    let n = 3;
    let p = MAX_DUAL_N; // m = 1 + 12 = 13 > MAX_DUAL_N
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
    let cluster_ids: [u32; 3] = [0, 1, 2];
    let x = faer::Mat::<f64>::zeros(n, p);
    let y = vec![0.0f64; n];
    let mut grad = vec![0.0f64; p + 1];
    let mut hess = Mat::<f64>::zeros(p + 1, p + 1);
    let status = laplace_hessian(
        &mut ws,
        x.as_ref(),
        &y,
        &cluster_ids,
        &[],
        p,
        n,
        &mut grad,
        &mut hess,
    );
    assert!(matches!(status, DerivStatus::Unsupported));
    assert!(ws.hyper_scratch.is_none());
}

/// `laplace_hessian`'s own guards on the two shapes `supports_shape`
/// rejects — mirrors `laplace_gradient_oversized_core_is_unsupported` and
/// `laplace_gradient_tail_past_boundary_is_unsupported`.
#[test]
fn laplace_hessian_unsupported_shapes_are_unsupported() {
    for (np, n_crossed) in [(crate::lmm::MAX_PRIMARY_Q, 0), (0, DUAL_TAIL_MAX + 1)] {
        let (mut ws, x, y, ids, extra_ids, p, n) = extras_routing_fixture(np, n_crossed);
        assert!(!supports_shape(ws.layout, &ws.groupings));
        let m = ws.n_theta + p;
        let mut grad = vec![0.0f64; m];
        let mut hess = Mat::<f64>::zeros(m, m);
        let status = laplace_hessian(
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
        assert!(matches!(status, DerivStatus::Unsupported));
        assert!(ws.hyper_scratch.is_none());
    }
}
