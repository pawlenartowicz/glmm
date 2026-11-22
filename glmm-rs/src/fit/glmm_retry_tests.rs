use super::*;
use crate::SeFor;

/// What [`gamma_inverse_retry`] reports when it runs more than the first
/// search. The routed `gamma_inverse_slope_design`, `(1 + x₁ | g)`
/// (n_θ = 3), runs its real searches in each unit; only the reported
/// deviances are set, so that the search in `glm` units (second of four,
/// put back) or the last one is kept. Every search leaves a hint; after
/// the retry there is none in either case, since the BFGS state left is
/// the last search's, which need not be the kept one. The gradient
/// search's stop and decrement are the
/// kept search's own, and `n_grad` is the sum over all four searches.
#[test]
fn gamma_inverse_retry_reports_the_kept_search_and_drops_the_hint() {
    let (x, y, ids, model) = super::super::glmm_tests::gamma_inverse_slope_design();
    let (n, p) = (y.len(), 2usize);
    let opts = FitOptions {
        se_for: SeFor::Only(vec![0, 1]),
        ..FitOptions::default()
    };
    let (sized, sids, _) = super::super::common::spec_sized_from_ids(&model, &ids, opts.condvar);
    let beta = [0.8, -0.3];
    let u = GammaInverseUnits {
        fast: 1.0,
        clusters: 2.0,
        glm: 0.5,
        glm_point: f64::INFINITY,
    };
    for kept in [u.glm, 2.0 * u.clusters] {
        let Ok((mut ws, x_mat, _)) =
            fit_glmm_build(&x, &y, None, None, n, p, &sized, &sids.primary, &[], &opts)
        else {
            panic!("the design did not build");
        };
        let mut reports = Vec::new();
        let mut fit_in = |ws: &mut GlmmWorkspace, c: f64| {
            let mut f = fit_glmm_in_units(
                ws,
                x_mat.as_ref(),
                x_mat.as_ref().subcols(0, 0),
                &y,
                None,
                n,
                p,
                &sids.primary,
                &[],
                &beta,
                None,
                None,
                &opts.se_for.columns(p),
                &opts,
                ColumnScaling::IDENTITY,
                Units::IDENTITY,
                c,
            );
            assert!(ws.grad_hint_ready, "c = {c}: the search left no hint");
            reports.push((c, ws.grad_report.expect("the search routes")));
            f.converged = true;
            // Far below any real objective here, so the kept search is
            // also the lowest point seen and no failed search is preferred.
            f.deviance = if c == kept { -1e6 } else { 0.0 };
            f
        };
        let mut fast = fit_in(&mut ws, u.fast);
        fast.converged = false;
        let _ = gamma_inverse_retry(&mut ws, fast, u.fast, &u, fit_in);
        assert_eq!(reports.len(), 4, "kept c = {kept}");
        assert_eq!(ws.search_units, kept, "kept c = {kept}: search units");
        assert!(
            crate::glmm::glmm_curvature_hint(&ws).is_none(),
            "kept c = {kept}: a hint is left"
        );
        let own = reports.iter().find(|r| r.0 == kept).expect("kept ran").1;
        let last = reports[3].1;
        if kept != 2.0 * u.clusters {
            assert_ne!(
                own.2.to_bits(),
                last.2.to_bits(),
                "the kept and the last search end alike, so the check is empty"
            );
        }
        let got = ws.grad_report.expect("a report");
        assert_eq!(got.0, own.0, "kept c = {kept}: stop");
        assert_eq!(
            got.2.to_bits(),
            own.2.to_bits(),
            "kept c = {kept}: decrement"
        );
        let sum: usize = reports.iter().map(|r| r.1 .1).sum();
        assert_eq!(got.1, sum, "kept c = {kept}: n_grad");
    }
}
