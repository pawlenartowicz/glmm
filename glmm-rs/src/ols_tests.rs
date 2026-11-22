use super::*;
use crate::test_support::TestWs;
use faer::Mat;

fn suff_stats(ws: &mut TestWs) -> OlsSuffStats<'_> {
    OlsSuffStats {
        xtx: ws.suff_xtx.as_mut(),
        xty: &mut ws.suff_xty,
        yty: &mut ws.suff_yty,
        sum_y: &mut ws.suff_sum_y,
        n_rows: &mut ws.suff_n_rows,
        panel_x: &mut ws.panel_x,
        panel_y: &mut ws.panel_y,
    }
}

fn build_x(n: usize, p: usize, mut fill: impl FnMut(usize, usize) -> f64) -> Mat<f64> {
    let mut m = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            m[(i, j)] = fill(i, j);
        }
    }
    m
}

// ---------------------------------------------------------------------
// aliased_columns rank-reveal tests
// ---------------------------------------------------------------------

/// Build the lower-tri Gram G = XᵀX (row-major X, n×p) into a faer Mat.
fn gram_of(x: &[f64], n: usize, p: usize) -> faer::Mat<f64> {
    let mut g = faer::Mat::<f64>::zeros(p, p);
    for i in 0..n {
        for a in 0..p {
            let xa = x[i * p + a];
            for b in 0..=a {
                g[(a, b)] += xa * x[i * p + b];
            }
        }
    }
    g
}

/// A dependent column is dropped and the earlier ones kept: col2 = col0 +
/// col1 exactly → column 2 aliased; an exact duplicate (col1 == col0) → the
/// LATER column (1) aliased; a full-rank design flags none.
#[test]
fn aliased_columns_flags_only_the_later_dependent_column() {
    let cases: [(Vec<f64>, usize, usize, Vec<bool>); 3] = [
        // rows: [1, days, 1+days]
        (
            vec![
                1.0, 0.0, 1.0, //
                1.0, 1.0, 2.0, //
                1.0, 2.0, 3.0, //
                1.0, 3.0, 4.0, //
            ],
            4,
            3,
            vec![false, false, true],
        ),
        (vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 3, 2, vec![false, true]),
        (
            vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
            4,
            2,
            vec![false, false],
        ),
    ];
    for (x, n, p, want) in cases {
        let g = gram_of(&x, n, p);
        let a = super::aliased_columns(g.as_ref(), p, super::ALIAS_EPS);
        assert_eq!(a, want, "n={n} p={p}");
    }
}

// ---------------------------------------------------------------------
// Suff-stats path tests
// ---------------------------------------------------------------------

/// `add_rows`'s panel GEMM must match a per-row rank-1 triangle
/// (inlined below as the oracle). X'X / X'y reassociate under the GEMM —
/// band 1e-12 relative, measured max 2.0e-15; yty / sum_y stay scalar in
/// the original row order and must be bit-identical. n = 611 crosses two
/// PANEL_ROWS=256 boundaries plus a 99-row tail; and a two-call split
/// accumulates the same statistics.
#[test]
fn add_rows_panel_matches_scalar_reference() {
    let n = 611;
    let p = 7;
    let x = build_x(n, p, |i, j| {
        ((((i * 13 + j * 29 + 3) % 47) as f64) / 11.0 - 2.0).sin()
    });
    let y: Vec<f64> = (0..n)
        .map(|i| ((((i * 31 + 7) % 53) as f64) / 9.0 - 2.5).cos())
        .collect();

    // Oracle: per-row scalar accumulation.
    let mut ref_xtx = Mat::<f64>::zeros(p, p);
    let mut ref_xty = vec![0.0f64; p];
    let (mut ref_yty, mut ref_sum_y) = (0.0f64, 0.0f64);
    for row in 0..n {
        let y_row = y[row];
        for j in 0..p {
            let x_rj = x[(row, j)] as f64;
            for i in j..p {
                ref_xtx[(i, j)] += x[(row, i)] as f64 * x_rj;
            }
            ref_xty[j] += x_rj * y_row;
        }
        ref_yty += y_row * y_row;
        ref_sum_y += y_row;
    }

    let mut ws = TestWs::new(n, p, 0);
    ws.reset_suff_stats();
    {
        let mut s = suff_stats(&mut ws);
        s.add_rows(x.as_ref(), &y);
    }
    for j in 0..p {
        for i in j..p {
            let (got, want) = (ws.suff_xtx[(i, j)], ref_xtx[(i, j)]);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "xtx[{i},{j}] = {got}, scalar reference {want}"
            );
        }
    }
    #[allow(clippy::needless_range_loop)]
    for k in 0..p {
        assert!(
            (ws.suff_xty[k] - ref_xty[k]).abs() <= 1e-12 * ref_xty[k].abs().max(1.0),
            "xty[{k}] = {}, scalar reference {}",
            ws.suff_xty[k],
            ref_xty[k]
        );
    }
    assert_eq!(
        ws.suff_yty.to_bits(),
        ref_yty.to_bits(),
        "yty must stay bit-identical (scalar row-order pass)"
    );
    assert_eq!(
        ws.suff_sum_y.to_bits(),
        ref_sum_y.to_bits(),
        "sum_y must stay bit-identical"
    );
    // Cross-call accumulation: two `add_rows` calls (split mid-panel) accumulate
    // the same statistics as the per-row oracle; the split reassociates, so banded.
    let mut ws2 = TestWs::new(n, p, 0);
    ws2.reset_suff_stats();
    {
        let mut s = suff_stats(&mut ws2);
        s.add_rows(x.as_ref().subrows(0, 300), &y[..300]);
        s.add_rows(x.as_ref().subrows(300, n - 300), &y[300..]);
    }
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
    for j in 0..p {
        for i in j..p {
            assert!(
                close(ws2.suff_xtx[(i, j)], ref_xtx[(i, j)]),
                "split xtx[{i},{j}]"
            );
        }
        assert!(close(ws2.suff_xty[j], ref_xty[j]), "split xty[{j}]");
    }
    assert!(
        close(ws2.suff_yty, ref_yty) && close(ws2.suff_sum_y, ref_sum_y),
        "split yty/sum_y"
    );
    assert_eq!(ws2.suff_n_rows, n);
    assert_eq!(ws.suff_n_rows, n);
}

/// Golden values for the production OLS fit (numeric kernels pin golden
/// values). Oracle hand-computed exactly from the closed-form normal
/// equations on x = [0, 1, 2, 3], y = [1, 3, 4, 8] with intercept:
/// X'X = [[4, 6], [6, 14]], X'y = [16, 35], y'y = 90, Σy = 16 →
/// β̂ = (0.7, 2.2); residuals (0.3, 0.1, −1.1, 0.7) → RSS = 1.8;
/// SST = 90 − 16²/4 = 26; σ̂² = RSS/2 = 0.9;
/// (X'X)⁻¹ = [[0.7, −0.3], [−0.3, 0.2]] → var_diag = (0.63, 0.18);
/// t² = (0.49/0.63, 4.84/0.18) = (7/9, 242/9).
///
/// Contrast slope − intercept, c = (−1, 1): c'(X'X)⁻¹c = 1.5, so
/// var = 0.9·1.5 = 1.35 and β̂₁ − β̂₀ = 1.5 gives t² = 2.25/1.35 = 5/3. A
/// symmetry check alone is blind to a `−`→`+` swap in the β-difference and
/// to `*`↔`/` in `(β_p−β_n)²/var` (both survive because the p/n swap
/// cancels the operator change), so the value is pinned both ways round.
#[test]
fn fit_suff_stats_golden_values() {
    let n = 4;
    let p = 2;
    let x = build_x(n, p, |i, j| if j == 0 { 1.0 } else { i as f64 });
    let y = [1.0_f64, 3.0, 4.0, 8.0];
    let targets: Vec<u32> = vec![0, 1];

    let mut ws = TestWs::new(n, p, 0);
    ws.reset_suff_stats();
    {
        let mut s = suff_stats(&mut ws);
        s.add_rows(x.as_ref(), &y);
    }
    let scratch = OlsScratch {
        fit_betas: &mut ws.fit_betas,
        fit_var_diag: &mut ws.fit_var_diag,
        fit_t_sq: &mut ws.fit_t_sq,
        fit_u_scratch: &mut ws.fit_u_scratch,
        fit_factor: ws.fit_factor.as_mut(),
        fit_rhs: ws.fit_rhs.as_mut(),
    };
    let res = fit_suff_stats_t_sq(
        ws.suff_xtx.as_ref(),
        &ws.suff_xty,
        ws.suff_yty,
        ws.suff_sum_y,
        ws.suff_n_rows,
        &targets,
        ws.suff_xtx_work.as_mut(),
        scratch,
    );
    assert!(res.converged);
    assert_eq!(res.df_resid, 2);

    let golden_betas = [0.7, 2.2];
    let golden_var = [0.63, 0.18];
    let golden_t_sq = [7.0 / 9.0, 242.0 / 9.0];
    for (j, (&got, &want)) in res.betas.iter().zip(golden_betas.iter()).enumerate() {
        assert!((got - want).abs() < 1e-9, "β̂[{j}] = {got}, golden {want}");
    }
    for k in 0..targets.len() {
        assert!(
            (res.var_diag[k] - golden_var[k]).abs() < 1e-9,
            "var_diag[{k}] = {}, golden {}",
            res.var_diag[k],
            golden_var[k]
        );
        assert!(
            (res.t_sq[k] - golden_t_sq[k]).abs() < 1e-9,
            "t²[{k}] = {}, golden {}",
            res.t_sq[k],
            golden_t_sq[k]
        );
    }
    assert!(
        (res.rss - 1.8).abs() < 1e-9,
        "rss = {}, golden 1.8",
        res.rss
    );
    assert!(
        (res.sst - 26.0).abs() < 1e-9,
        "sst = {}, golden 26.0",
        res.sst
    );
    assert!(
        (res.sigma_sq - 0.9).abs() < 1e-9,
        "σ̂² = {}, golden 0.9",
        res.sigma_sq
    );

    let mut cscratch = vec![0.0_f64; p];
    let t_sq = ols_contrast_t_sq(&res, 1, 0, &mut cscratch);
    assert!(
        (t_sq - 5.0 / 3.0).abs() < 1e-9,
        "contrast t² = {t_sq}, golden 5/3"
    );
    // Symmetric under swap — same golden, different code path through the sign.
    let rev = ols_contrast_t_sq(&res, 0, 1, &mut cscratch);
    assert!(
        (rev - 5.0 / 3.0).abs() < 1e-9,
        "reversed t² = {rev}, golden 5/3"
    );
}

/// `sst` summed from the rows on y = 1e9 + {−1, 0, 1}: exactly 2, where the
/// one-pass `y′y − (Σy)²/n` loses it (`y′y` ≈ 3e18 has an ULP of 512).
#[test]
fn sst_from_rows_keeps_digits_far_from_zero() {
    let n = 3;
    let p = 1;
    let x = build_x(n, p, |_, _| 1.0);
    let y = [1e9 - 1.0, 1e9, 1e9 + 1.0];
    let targets: Vec<u32> = vec![0];

    let mut ws = TestWs::new(n, p, 0);
    ws.reset_suff_stats();
    {
        let mut s = suff_stats(&mut ws);
        s.add_rows(x.as_ref(), &y);
    }
    let scratch = OlsScratch {
        fit_betas: &mut ws.fit_betas,
        fit_var_diag: &mut ws.fit_var_diag,
        fit_t_sq: &mut ws.fit_t_sq,
        fit_u_scratch: &mut ws.fit_u_scratch,
        fit_factor: ws.fit_factor.as_mut(),
        fit_rhs: ws.fit_rhs.as_mut(),
    };
    let res = fit_suff_stats_t_sq_mapped(
        ws.suff_xtx.as_ref(),
        &ws.suff_xty,
        ws.suff_yty,
        ws.suff_sum_y,
        ws.suff_n_rows,
        &targets,
        ws.suff_xtx_work.as_mut(),
        scratch,
        ColumnScaling::IDENTITY,
        Units::IDENTITY,
        None,
        Some((x.as_ref(), &y)),
    );
    assert!(res.converged);
    assert_eq!(res.sst, 2.0);
}

/// A degenerate design gives the non-converged view with all-NaN outputs
/// (the crate's non-convergence signal), including `rss`/`sst`: `n ≤ p`
/// takes the early return; a zero column gives an exactly-zero X'X
/// diagonal, so a zero Cholesky pivot that faer's LLT rejects as non-PD.
/// That rejection is the only thing this route refuses on: near-singularity
/// short of it is measured and reported, not refused. An exact-duplicate
/// column instead leaves a ~1e-7 roundoff pivot that LLT accepts, and is
/// FITTED with a huge SE — a different test.
#[test]
fn suff_stats_degenerate_design_is_not_converged() {
    // (label, n, p, column (i, j), slope of y = slope·i, targets)
    type Case = (
        &'static str,
        usize,
        usize,
        fn(usize, usize) -> f64,
        f64,
        &'static [u32],
    );
    let cases: [Case; 2] = [
        (
            "n < p",
            3,
            4,
            |i, j| ((i + j) as f64).sin() + 1.0,
            1.0,
            &[1, 2, 3],
        ),
        (
            "zero column",
            50,
            3,
            |i, j| match j {
                0 => 1.0,
                1 => (i as f64) * 0.1,
                _ => 0.0,
            },
            0.3,
            &[1, 2],
        ),
    ];
    for (name, n, p, column, slope, targets) in cases {
        let x = build_x(n, p, column);
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * slope).collect();
        let mut ws = TestWs::new(n.max(p), p, 0);
        ws.reset_suff_stats();
        {
            let mut s = suff_stats(&mut ws);
            s.add_rows(x.as_ref(), &y);
        }
        let scratch = OlsScratch {
            fit_betas: &mut ws.fit_betas,
            fit_var_diag: &mut ws.fit_var_diag,
            fit_t_sq: &mut ws.fit_t_sq,
            fit_u_scratch: &mut ws.fit_u_scratch,
            fit_factor: ws.fit_factor.as_mut(),
            fit_rhs: ws.fit_rhs.as_mut(),
        };
        let res = fit_suff_stats_t_sq(
            ws.suff_xtx.as_ref(),
            &ws.suff_xty,
            ws.suff_yty,
            ws.suff_sum_y,
            ws.suff_n_rows,
            targets,
            ws.suff_xtx_work.as_mut(),
            scratch,
        );
        assert!(!res.converged, "{name}: must not converge");
        for v in res.t_sq.iter() {
            assert!(v.is_nan(), "{name}: t² must be NaN");
        }
        for v in res.betas.iter() {
            assert!(v.is_nan(), "{name}: β̂ must be NaN");
        }
        assert!(res.rss.is_nan(), "{name}: rss must be NaN");
        assert!(res.sst.is_nan(), "{name}: sst must be NaN");
    }
}

// -----------------------------------------------------------------------
// ols_contrast_t_sq unit tests
// -----------------------------------------------------------------------

/// `ols_contrast_t_sq`'s bounds guard — `p_col >= p`, `n_col >= p`,
/// `p_col == n_col`, or `scratch.len() < p` — on a fit that otherwise
/// converges and would return a real value, so the guard itself is what is
/// under test, not the earlier `!converged` short-circuit. A p=3 factor
/// confirms in-range indices still compute normally; the last assertion
/// flips `converged` on the same otherwise-valid fit and must get NaN.
#[test]
fn contrast_t_sq_bounds_checks_return_nan() {
    let p = 3;
    let mut factor = Mat::<f64>::zeros(p, p);
    factor[(0, 0)] = 2.0;
    factor[(1, 0)] = 1.0;
    factor[(1, 1)] = 3.0;
    factor[(2, 1)] = 1.0;
    factor[(2, 2)] = 4.0;
    let betas = [0.5_f64, 1.2, -0.7];
    let var_diag = [0.0_f64; 3];
    let t_sq_dummy = [0.0_f64; 3];
    let fit = OlsFitView {
        betas: &betas,
        var_diag: &var_diag,
        t_sq: &t_sq_dummy,
        factor: factor.as_ref(),
        sigma_sq: 0.4,
        df_resid: 10,
        converged: true,
        rss: 0.0,
        sst: 0.0,
        pivot: 1.0,
        pivot_col: 0,
    };
    let mut scratch = vec![0.0_f64; p];

    assert!(
        ols_contrast_t_sq(&fit, p as u32, 0, &mut scratch).is_nan(),
        "p_col == p is out of range"
    );
    assert!(
        ols_contrast_t_sq(&fit, 0, p as u32, &mut scratch).is_nan(),
        "n_col == p is out of range"
    );
    let mut short_scratch = vec![0.0_f64; p - 1];
    assert!(
        ols_contrast_t_sq(&fit, 1, 2, &mut short_scratch).is_nan(),
        "scratch shorter than p must be refused"
    );
    assert!(
        ols_contrast_t_sq(&fit, 1, 1, &mut scratch).is_nan(),
        "a zero contrast has no test"
    );
    assert!(
        ols_contrast_t_sq(&fit, 1, 2, &mut scratch).is_finite(),
        "the same indices, in range with full-length scratch, must compute normally"
    );
    let not_converged = OlsFitView {
        converged: false,
        ..fit
    };
    let got = ols_contrast_t_sq(&not_converged, 1, 2, &mut scratch);
    assert!(got.is_nan(), "non-converged fit must return NaN, got={got}");
}

/// `triangular_solve_norm_sq`'s zero-diagonal guard never runs in
/// production: every caller hands it a factor whose Cholesky was accepted
/// (diagonal strictly positive by construction). Drive it directly: an
/// exact zero pivot comes back NaN, and a pivot of 1e-35, a factor in small
/// units, solves like any other.
#[test]
fn triangular_solve_norm_sq_zero_diagonal_returns_nan() {
    let p = 2;
    let mut factor = Mat::<f64>::zeros(p, p);
    factor[(0, 0)] = 2.0;
    factor[(1, 0)] = 1.0;
    let mut scratch = vec![0.0_f64; p];
    let b = |i: usize| if i == 0 { 1.0 } else { 0.0 };

    factor[(1, 1)] = 0.0;
    let got = triangular_solve_norm_sq(factor.as_ref(), b, &mut scratch, p, false);
    assert!(got.is_nan(), "a zero diagonal must return NaN, got {got}");

    factor[(1, 1)] = 1e-35;
    let got = triangular_solve_norm_sq(factor.as_ref(), b, &mut scratch, p, false);
    // u = (1/2, −(1/2)/1e-35): the second entry dominates ‖u‖².
    assert!(
        (got - 0.25e70).abs() <= 1e-15 * 0.25e70,
        "a 1e-35 pivot must solve normally, got {got}"
    );
}

/// Warm-path allocation guard for the production suff-stats OLS fit
/// (`reset_suff_stats` → `add_rows` → `fit_suff_stats_t_sq`). The bound
/// below is the per-fit `SeqLlt` factor plus faer's `llt` + `L()`
/// internals, pinned so a faer upgrade that regresses the warm path fails
/// loudly here instead of surfacing as a benchmark mystery.
///
/// `#[ignore]` because `dhat::Profiler` measures process-wide allocations;
/// `alloc_test_guard` serializes test bodies, but libtest's own per-test
/// thread spawn still needs `--test-threads=1`:
///   `RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests fit_suff_stats_warm_path_bounded_alloc -- --ignored --test-threads=1`
/// (`alloc-tests` installs the dhat global allocator the profiler requires.)
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn fit_suff_stats_warm_path_bounded_alloc() {
    let _serial = crate::test_support::alloc_test_guard();
    // 2 blocks/fit measured on faer 0.24 (`llt` factor + `L()` internals);
    // no one-time block — the GEMM-backend lazy init lands in the warmup
    // call outside the profiler window.
    const FAER_LLT_BLOCKS_PER_FIT: usize = 2;
    const N_CALLS: usize = 100;
    const ONE_TIME: usize = 0;
    const BOUND: usize = FAER_LLT_BLOCKS_PER_FIT * N_CALLS + ONE_TIME;

    let n = 200;
    let p = 6;
    let x = build_x(n, p, |i, j| {
        if j == 0 {
            1.0
        } else {
            ((i * 7 + j * 13 + 5) % 23) as f64 / 5.0 - 2.0
        }
    });
    let y: Vec<f64> = (0..n).map(|i| ((i * 11) % 13) as f64 / 10.0).collect();
    let targets: Vec<u32> = vec![1, 2];

    let mut ws = TestWs::new(n, p, 0);

    let run_fit = |ws: &mut TestWs| {
        ws.reset_suff_stats();
        {
            let mut s = suff_stats(ws);
            s.add_rows(x.as_ref(), &y);
        }
        let scratch = OlsScratch {
            fit_betas: &mut ws.fit_betas,
            fit_var_diag: &mut ws.fit_var_diag,
            fit_t_sq: &mut ws.fit_t_sq,
            fit_u_scratch: &mut ws.fit_u_scratch,
            fit_factor: ws.fit_factor.as_mut(),
            fit_rhs: ws.fit_rhs.as_mut(),
        };
        let fit = fit_suff_stats_t_sq(
            ws.suff_xtx.as_ref(),
            &ws.suff_xty,
            ws.suff_yty,
            ws.suff_sum_y,
            ws.suff_n_rows,
            &targets,
            ws.suff_xtx_work.as_mut(),
            scratch,
        );
        assert!(fit.converged);
    };

    // Warm everything once outside the measured window, then wait out the
    // rayon worker startup that warmup kicks off on other threads.
    run_fit(&mut ws);
    crate::test_support::settle_background_allocs();

    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..N_CALLS {
        run_fit(&mut ws);
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);
    assert!(
        stats.total_blocks as usize <= BOUND,
        "fit_suff_stats_t_sq allocated {} blocks across {} warm-path calls (expected ≤ {})",
        stats.total_blocks,
        N_CALLS,
        BOUND
    );
}
