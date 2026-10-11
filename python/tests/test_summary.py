import math
import re
import warnings

import numpy as np
import pytest

import glmm
from glmm.summary import build_summary


def make_fit(**kw):
    base = {
        "beta": np.array([2.0, -0.5]),
        "se": np.array([1.0, 0.25]),
        "vcov": np.array([[1.0, 0.0], [0.0, 0.0625]]),
        "tau2": np.array([]),
        "varcorr": [],
        "stddev_se": np.array([]),
        "diagnostics": {
            "converged": True,
            "singular": False,
            "aliased": np.array([False, False]),
            "boundary": "interior",
            "pinned": [],
            "notes": [],
        },
        "dispersion": 1.0,
        "names": ["(Intercept)", "x"],
        "re_groups": [],
        "n_eval": 0,
        "n_grad": 0,
        "outer_stop": None,
        "outer_decrement": math.nan,
        "deviance": math.nan,
        "loglik": math.nan,
        "df": 0,
        "reml": False,
        "fitted": np.array([]),
        "ranef": np.array([]),
        "ranef_levels": np.array([]),
        "ranef_blocks": [],
        "formula": "y ~ x",
        "family": "gaussian",
        "link": "identity",
        "nagq": 1,
        "nobs": 4,
        "y": np.array([1.0, 2.0, 3.0, 4.0]),
        "weights": None,
        # A z table (the families with no df) unless a test says otherwise.
        "coef_df": np.array([math.nan, math.nan]),
        "coef_p": np.array([math.nan, math.nan]),
        "df_method": "none",
        "rss": math.nan,
        "r_squared": math.nan,
        "adj_r_squared": math.nan,
    }
    # converged / singular / aliased are diagnostics entries, not Fit fields —
    # accept them as flat kwargs anyway so the tests below read the way the
    # attributes do.
    for name in ("converged", "singular", "aliased"):
        if name in kw:
            base["diagnostics"][name] = kw.pop(name)
    base.update(kw)
    return glmm.Fit(**base)


def test_stddev_corr_q2_hand_math():
    # D=[[4,1],[1,1.25]] -> vech(col-major lower-tri)=[4,1,1.25]
    f = make_fit(varcorr=[np.array([4.0, 1.0, 1.25])])
    sd, corr = f.stddev_corr(0)
    sd1 = math.sqrt(1.25)
    assert abs(sd[0] - 2.0) < 1e-12
    assert abs(sd[1] - sd1) < 1e-12
    rho = 1.0 / (2.0 * sd1)
    assert corr[0][0] == 1.0 and corr[1][1] == 1.0
    assert abs(corr[0][1] - rho) < 1e-12
    assert abs(corr[1][0] - rho) < 1e-12


def test_stddev_corr_invalid_vech_raises():
    f = make_fit(varcorr=[np.array([4.0, 1.0])])  # len 2 is no q(q+1)/2
    with pytest.raises(ValueError, match="vech"):
        f.stddev_corr(0)


def _mixed_fit(**kw):
    # Same D as test_stddev_corr_q2_hand_math: sd=[2, 1.1180], rho=0.4472.
    base = {
        "varcorr": [np.array([4.0, 1.0, 1.25])],
        "stddev_se": np.array([0.5, float("nan"), 0.3]),
        "re_groups": [("Subject", ["(Intercept)", "Days"])],
        "ranef_levels": np.array([18]),
        "formula": "y ~ x + (1 + x | Subject)",
        "reml": True,
        "loglik": -871.8,
        "df": 6,
        "dispersion": 654.94,
        "fitted": np.array([1.5, 1.5, 3.5, 3.5]),
        "n_eval": 12,
        "coef_df": np.array([15.0, 16.25]),
        "coef_p": np.array([0.0632, 0.0526]),
        "df_method": "satterthwaite",
    }
    base.update(kw)
    return make_fit(**base)


def test_summary_object_blocks_lmm():
    s = _mixed_fit().summary_object()
    assert s.method == "Linear mixed model fit by REML [glmm]"
    assert s.criterion == {"REML criterion at convergence": pytest.approx(1743.6)}
    assert s.groups == [("Subject", 18)]
    assert s.residual_variance == pytest.approx(654.94)
    re = s.random_effects[0]
    assert re["group"] == "Subject"
    np.testing.assert_allclose(re["variance"], [4.0, 1.25])
    np.testing.assert_allclose(re["stddev"], [2.0, math.sqrt(1.25)])
    assert re["corr"][1, 0] == pytest.approx(1.0 / (2.0 * math.sqrt(1.25)))
    # Scaled residuals: (y - fitted) / sqrt(dispersion), quantiles type 7.
    r = (np.array([1.0, 2.0, 3.0, 4.0]) - np.array([1.5, 1.5, 3.5, 3.5])) / math.sqrt(654.94)
    np.testing.assert_allclose(s.scaled_residuals, np.quantile(r, [0, 0.25, 0.5, 0.75, 1]))


def test_groups_count_excludes_unused_grouping_levels():
    # A declared level with no rows still owns a slot in ranef_levels (a
    # block is sized by the highest observed code), but lme4's count is
    # after droplevels() — one fewer per unused level, and the number every
    # renderer prints must agree.
    f = _mixed_fit(
        diagnostics={
            "converged": True,
            "singular": False,
            "aliased": np.array([False, False]),
            "boundary": "interior",
            "pinned": [],
            "notes": [
                {
                    "kind": "unused_grouping_levels",
                    "columns": [],
                    "pivot": math.nan,
                    "evals": 0,
                    "final_eval": False,
                    "detail": "Subject: s9, s10",
                }
            ],
        },
        ranef_levels=np.array([20]),  # 18 used levels + 2 declared-unused
    )
    s = f.summary_object()
    assert s.groups == [("Subject", 18)]
    for rendered in (s.text(), s.html(), s.latex(), s.typst()):
        assert "groups:  Subject, 18" in rendered or "groups: Subject, 18" in rendered


def test_summary_object_criterion_row_ml():
    f = make_fit(family="poisson", link="log", loglik=-92.0, df=5, nobs=56, reml=False)
    c = f.summary_object().criterion
    assert list(c) == ["AIC", "BIC", "logLik", "deviance", "df.resid"]
    assert c["AIC"] == pytest.approx(194.0)
    assert c["BIC"] == pytest.approx(-2 * -92.0 + 5 * math.log(56))
    assert c["deviance"] == pytest.approx(184.0)
    assert c["df.resid"] == 51


def test_summary_text_has_lme4_blocks_in_order():
    text = _mixed_fit().summary()
    order = [
        "Linear mixed model fit by REML [glmm]",
        "Formula: y ~ x + (1 + x | Subject)",
        " Family: gaussian (identity)",
        "REML criterion at convergence:",
        "Scaled residuals:",
        "Random effects:",
        "Groups",
        "Residual",
        "Number of obs: 4, groups:  Subject, 18",
        "Fixed effects:",
        "Pr(>|t|)",
        "Correlation of Fixed Effects:",
        "Optimizer evaluations: 12, converged: True",
        "t tests use Satterthwaite degrees of freedom",
    ]
    pos = [text.index(s) for s in order]
    assert pos == sorted(pos), text


def test_summary_text_wald_z_p_and_stars():
    text = make_fit(family="poisson", link="log").summary()
    # z = 2.0/1.0 = 2, p = erfc(2/sqrt(2)) = 0.0455 -> one star.
    row = next(ln for ln in text.splitlines() if ln.startswith("(Intercept)"))
    assert "0.0455" in row and row.rstrip().endswith("*")
    assert "Signif. codes" in text


def test_summary_aliased_row_prints_na():
    text = make_fit(
        aliased=np.array([False, True]),
        beta=np.array([2.0, float("nan")]),
        se=np.array([1.0, float("nan")]),
    ).summary()
    x_row = next(ln for ln in text.splitlines() if ln.startswith("x"))
    assert " NA " in x_row + " "
    assert "1 coefficient(s) not defined because of singularities" in text


def test_summary_non_converged_still_prints_header():
    text = _mixed_fit(converged=False, fitted=np.array([])).summary()
    assert "Number of obs: 4, groups:  Subject, 18" in text
    assert "Scaled residuals" not in text
    assert "converged: False" in text


def test_summary_no_re_block_when_empty():
    text = make_fit().summary()
    assert "Random effects" not in text
    assert "Number of obs: 4\n" in text


def test_summary_dispersion_footer_by_family():
    assert (
        "Shape (theta): 2.5"
        in make_fit(family="negativebinomial", link="log", dispersion=2.5).summary()
    )
    assert (
        "Dispersion (phi, Pearson): 2.5"
        in make_fit(family="gamma", link="log", dispersion=2.5).summary()
    )
    assert (
        "Dispersion (phi, ML): 2.5"
        in _mixed_fit(family="gamma", link="log", dispersion=2.5).summary()
    )
    assert (
        "Dispersion (phi, Pearson): 2.5"
        in make_fit(family="inversegaussian", link="log", dispersion=2.5).summary()
    )
    assert "Dispersion" not in make_fit(family="poisson", link="log").summary()


def test_summary_dispersion_footer_held_label():
    # A held dispersion= reports "fixed" regardless of family or mixedness.
    assert (
        "Dispersion (phi, fixed): 2.5"
        in make_fit(family="gamma", link="log", dispersion=2.5, dispersion_held=2.5).summary()
    )
    assert (
        "Dispersion (phi, fixed): 2.5"
        in _mixed_fit(family="gamma", link="log", dispersion=2.5, dispersion_held=2.5).summary()
    )
    assert (
        "Dispersion (phi, fixed): 2.5"
        in make_fit(
            family="inversegaussian", link="log", dispersion=2.5, dispersion_held=2.5
        ).summary()
    )


def test_summary_prints_what_it_returns(capsys):
    text = make_fit().summary()
    assert capsys.readouterr().out.strip() == text.strip()


def test_summary_object_repr_is_text():
    s = make_fit().summary_object()
    assert repr(s) == s.text()


def test_residuals_response_and_pearson():
    f = make_fit(
        family="poisson",
        link="log",
        y=np.array([1.0, 2.0, 3.0, 4.0]),
        fitted=np.array([1.0, 1.0, 4.0, 4.0]),
    )
    np.testing.assert_allclose(f.residuals(), [0.0, 1.0, -1.0, 0.0])
    np.testing.assert_allclose(f.residuals(type="pearson"), [0.0, 1.0, -0.5, 0.0])
    with pytest.raises(ValueError, match="response.*pearson"):
        f.residuals(type="deviance")
    with pytest.raises(ValueError, match="did not converge"):
        make_fit(fitted=np.array([])).residuals()


def test_pearson_residuals_stay_finite_at_extreme_means():
    # V = mu^2 (Gamma) and mu^3 (inverse Gaussian) leave the double range here;
    # sqrt(V) does not. Expected: (y - mu) / sqrt(phi * V).
    gamma = make_fit(
        family="gamma",
        link="log",
        dispersion=0.5,
        y=np.array([1.5e200, 1e-200]),
        fitted=np.array([1e200, 2e-200]),
    )
    np.testing.assert_allclose(gamma.residuals(type="pearson"), [0.5**0.5, -(0.5**0.5)])
    ig = make_fit(
        family="inversegaussian",
        link="log",
        dispersion=0.25,
        y=np.array([2e150, 2e-150]),
        fitted=np.array([1e150, 1e-150]),
    )
    np.testing.assert_allclose(ig.residuals(type="pearson"), [2e-75, 2e75])


_NUM = re.compile(r"-?\d+\.\d+(?:e[-+]?\d+)?")


def _numbers(s):
    return sorted(_NUM.findall(s))


@pytest.mark.parametrize("render", ["html", "latex", "typst"])
def test_renderers_carry_the_same_numbers_as_text(render):
    # A formatting change then fails here, in one place, not in four snapshots.
    s = _mixed_fit().summary_object()
    text_nums = _numbers(s.text())
    other = _numbers(getattr(s, render)())
    missing = [n for n in text_nums if n not in other]
    assert not missing, f"{render} lost numbers present in text: {missing}"


def test_html_is_a_fragment_with_scoped_style():
    h = _mixed_fit().summary_object().html()
    assert h.lstrip().startswith("<style")
    assert "<html" not in h and "<body" not in h
    # criterion, scaled residuals, random effects, fixed effects, correlation (p = 2)
    assert h.count("<table") == 5


def test_latex_is_booktabs_fragments_only():
    t = _mixed_fit().summary_object().latex()
    assert "\\documentclass" not in t and "\\begin{document}" not in t
    assert t.count("\\begin{tabular}") >= 2
    assert "\\toprule" in t and "\\bottomrule" in t
    assert t.lstrip().startswith("% Linear mixed model fit by REML [glmm]")


def test_typst_is_table_fragments_only():
    t = _mixed_fit().summary_object().typst()
    assert "#set page" not in t
    assert t.count("#table(") >= 2
    assert t.lstrip().startswith("// Linear mixed model fit by REML [glmm]")


def test_html_escapes_names():
    s = make_fit(names=["(Intercept)", "a<b"]).summary_object()
    assert "a&lt;b" in s.html()


def test_latex_escapes_names():
    s = make_fit(names=["(Intercept)", "x_1"]).summary_object()
    assert "x\\_1" in s.latex()


def test_summary_ends_with_the_stored_warnings():
    w = [
        {
            "tier": "note",
            "kind": "argument_ignored",
            "title": "Argument ignored",
            "message": "dispersion= has no effect for family 'gaussian'.",
        },
        {
            "tier": "severe",
            "kind": "fit_failed",
            "title": "Fit failed",
            "message": "The fitting algorithm failed and returned no estimates.",
        },
    ]
    lines = build_summary(make_fit(warnings=w)).text().splitlines()
    assert lines[-3:] == [
        "Warnings:",
        "Note: Argument ignored. dispersion= has no effect for family 'gaussian'.",
        "Severe: Fit failed. The fitting algorithm failed and returned no estimates.",
    ]


def test_clean_summary_has_no_warnings_section():
    assert "Warnings:" not in build_summary(make_fit()).text()


def test_summary_works_on_a_fit_pickled_before_the_warnings_attribute():
    # A Fit pickled before `warnings` was added would unpickle without that
    # attribute (dataclasses don't backfill missing fields on old pickles).
    # build_summary must not blow up on it, and must render with no
    # Warnings section.
    fit = make_fit()
    del fit.__dict__["warnings"]
    text = build_summary(fit).text()
    assert "Warnings:" not in text


def _fixed_header(text):
    lines = text.splitlines()
    return lines[lines.index("Fixed effects:") + 1].split()


def _ols_fit(**kw):
    # n - p = 4 - 2 = 2 residual df.
    base = {
        "coef_df": np.array([2.0, 2.0]),
        "coef_p": np.array([0.18, 0.057]),
        "df_method": "residual",
        "rss": 0.5,
        "r_squared": 0.8125,
        "adj_r_squared": 0.71875,
        "fitted": np.array([1.0, 2.0, 3.0, 4.0]),
    }
    base.update(kw)
    return make_fit(**base)


def test_lmm_summary_has_lmertest_columns_with_the_crates_df_and_p():
    s = _mixed_fit().summary_object()
    text = s.text()
    assert _fixed_header(text) == ["Estimate", "Std.", "Error", "df", "t", "value", "Pr(>|t|)"]
    row = next(ln for ln in text.splitlines() if ln.startswith("(Intercept)"))
    # Estimate 2, se 1, df 15, t 2, and the crate's p (not a normal one).
    assert row.split()[1:6] == ["2", "1", "15", "2", "0.0632"]
    assert row.rstrip().endswith(".")
    x_row = next(ln for ln in text.splitlines() if ln.startswith("x "))
    assert "16.25" in x_row and "0.0526" in x_row
    assert text.splitlines()[-1] == glmm.summary.FOOTNOTE
    for render in (s.html(), s.latex(), s.typst()):
        assert "t value" in render
        assert "t tests use Satterthwaite degrees of freedom" in render


def test_coef_df_column_in_every_render():
    s = _mixed_fit().summary_object()
    assert ("<th>df</th>") in s.html()
    assert " & df & " in s.latex()
    assert 'strong("df")' in s.typst()
    assert re.search(r"Std\. Error\s+df\s+t value", s.text())


def test_ols_summary_prints_t_without_df_and_the_lm_footer():
    s = _ols_fit().summary_object()
    text = s.text()
    assert _fixed_header(text) == ["Estimate", "Std.", "Error", "t", "value", "Pr(>|t|)"]
    row = next(ln for ln in text.splitlines() if ln.startswith("(Intercept)"))
    assert "0.18" in row
    lines = text.splitlines()
    # sqrt(rss / (n - p)) = sqrt(0.5 / 2) = 0.5.
    i = lines.index("Residual standard error: 0.5 on 2 degrees of freedom")
    assert lines[i + 1] == "Multiple R-squared:  0.8125,\tAdjusted R-squared:  0.7188"
    assert "Satterthwaite" not in text and "Wald z" not in text and "df not computed" not in text
    for render in (s.html(), s.latex(), s.typst()):
        assert "Residual standard error: 0.5 on 2 degrees of freedom" in render
        assert "Adjusted R-squared:  0.7188" in render


def test_dispersion_glm_prints_t_and_its_residual_df():
    text = _ols_fit(family="gamma", link="log", dispersion=2.5, rss=math.nan).summary()
    assert _fixed_header(text)[-3:] == ["t", "value", "Pr(>|t|)"]
    assert "Dispersion (phi, Pearson): 2.5 on 2 degrees of freedom" in text
    assert "Residual standard error" not in text and "R-squared" not in text


def test_glmm_and_glm_keep_z_with_no_test_footer():
    for f in (
        _mixed_fit(
            family="binomial",
            link="logit",
            df_method="none",
            coef_df=np.full(2, np.nan),
            # A binomial fit's fitted values lie in (0, 1); the Pearson residuals need it.
            y=np.array([0.0, 1.0, 0.0, 1.0]),
            fitted=np.array([0.2, 0.4, 0.6, 0.8]),
        ),
        make_fit(family="poisson", link="log"),
    ):
        text = f.summary()
        assert _fixed_header(text)[-3:] == ["z", "value", "Pr(>|z|)"]
        assert "Satterthwaite" not in text and "Wald z" not in text
        assert "df not computed" not in text


def test_switched_off_lmm_prints_z_and_says_df_not_computed():
    nan2 = np.full(2, np.nan)
    s = _mixed_fit(df_method="disabled", coef_df=nan2, coef_p=nan2).summary_object()
    text = s.text()
    assert _fixed_header(text)[-3:] == ["z", "value", "Pr(>|z|)"]
    # z = 2 -> normal p 0.0455, not the (absent) crate p.
    row = next(ln for ln in text.splitlines() if ln.startswith("(Intercept)"))
    assert "0.0455" in row
    assert text.splitlines()[-1] == "df not computed (satterthwaite=False)"
    for render in (s.html(), s.latex(), s.typst()):
        assert "df not computed (satterthwaite=False)" in render


def test_failed_pass_prints_z():
    nan2 = np.full(2, np.nan)
    text = _mixed_fit(df_method="failed", coef_df=nan2, coef_p=nan2).summary()
    assert _fixed_header(text)[-3:] == ["z", "value", "Pr(>|z|)"]
    assert "Satterthwaite degrees of freedom" not in text


def test_random_effects_block_has_no_se_column():
    f = _mixed_fit()
    stddev_se = f.stddev_se.copy()
    s = f.summary_object()
    (header,) = [h for t, h, _ in s._tables() if t == "Random effects"]
    assert header == ["Groups", "Name", "Variance", "Std.Dev.", "Corr"]
    lines = s.text().splitlines()
    re_header = lines[lines.index("Random effects:") + 1].split()
    assert re_header == ["Groups", "Name", "Variance", "Std.Dev.", "Corr"]
    assert "<th>se</th>" not in s.html()
    assert " & se & " not in s.latex()
    assert 'strong("se")' not in s.typst()
    # The data stays on the fit.
    np.testing.assert_array_equal(f.stddev_se, stddev_se)


def test_confint_is_a_p_by_2_array_in_names_order():
    # df 2: qt(0.975, 2) = 4.3026527297494619 in R.
    ci = _ols_fit().confint()
    assert ci.shape == (2, 2)
    np.testing.assert_allclose(
        ci[:, 0], [2.0 - 4.3026527297494619, -0.5 - 0.25 * 4.3026527297494619]
    )
    np.testing.assert_allclose(
        ci[:, 1], [2.0 + 4.3026527297494619, -0.5 + 0.25 * 4.3026527297494619]
    )
    # z rows (no df) take qnorm: qnorm(0.95) = 1.6448536269514715.
    ci90 = make_fit(family="poisson", link="log").confint(level=0.9)
    np.testing.assert_allclose(
        ci90[:, 1], [2.0 + 1.6448536269514715, -0.5 + 0.25 * 1.6448536269514715]
    )


@pytest.mark.parametrize("df_method", ["disabled", "failed"])
def test_confint_is_normal_when_the_lmm_df_were_not_computed(df_method):
    nan2 = np.full(2, np.nan)
    ci = _mixed_fit(df_method=df_method, coef_df=nan2, coef_p=nan2).confint()
    # qnorm(0.975) = 1.9599639845400534; the SEs are finite, so no NaN.
    np.testing.assert_allclose(
        ci[:, 0], [2.0 - 1.9599639845400534, -0.5 - 0.25 * 1.9599639845400534]
    )


def test_confint_aliased_row_is_nan_and_bad_level_raises():
    f = _ols_fit(
        aliased=np.array([False, True]),
        beta=np.array([2.0, math.nan]),
        se=np.array([1.0, math.nan]),
        coef_df=np.array([3.0, math.nan]),
    )
    ci = f.confint()
    assert np.isfinite(ci[0]).all() and np.isnan(ci[1]).all()
    for bad in (0.0, 1.0, 1.5, math.nan):
        with pytest.raises(ValueError, match="level must be a number strictly between 0 and 1"):
            f.confint(level=bad)


# Fits through the port (the same MIXED-style fixture as test_warnings.py).
_EX = [float(v) for v in range(-10, 11) if v != 0]
_EG = [f"g{i % 5}" for i in range(20)]
_EOFF = {"g0": -3.0, "g1": -1.5, "g2": 0.0, "g3": 1.5, "g4": 3.0}
_EY = [0.3 * x + ((i * 7) % 5 - 2) * 0.4 + _EOFF[g] for i, (x, g) in enumerate(zip(_EX, _EG))]
_EMIXED = {"x": _EX, "g": _EG, "y": _EY}


def test_lmm_fit_carries_coef_df_and_the_switch_turns_it_off():
    m = glmm.fit(_EMIXED, "y ~ x + (1 | g)")
    assert m.df_method == "satterthwaite"
    assert np.all(np.isfinite(m.coef_df)) and np.all(m.coef_df > 0)
    assert np.all(np.isfinite(m.coef_p))
    s = m.summary_object()
    np.testing.assert_array_equal(s.coef_df, m.coef_df)
    np.testing.assert_array_equal(s.p_value, m.coef_p)
    # The interval is the t interval on each row's df: wider than the normal one.
    ci = m.confint()
    half = (ci[:, 1] - ci[:, 0]) / (2.0 * m.se)
    assert np.all(half > 1.9599639845400534)

    with warnings.catch_warnings():
        warnings.simplefilter("error", glmm.SatterthwaiteCurvatureWarning)
        off = glmm.fit(_EMIXED, "y ~ x + (1 | g)", satterthwaite=False)
    assert off.df_method == "disabled"
    assert np.all(np.isnan(off.coef_df)) and np.all(np.isnan(off.coef_p))
    np.testing.assert_array_equal(off.beta, m.beta)
    assert off.summary_object().text().splitlines()[-1] == "df not computed (satterthwaite=False)"
    np.testing.assert_allclose((off.confint()[:, 1] - off.beta) / off.se, 1.9599639845400534)


def test_ols_and_glmm_fits_label_their_test():
    ols = glmm.fit(_EMIXED, "y ~ x", satterthwaite=False)
    assert ols.df_method == "residual"
    np.testing.assert_array_equal(ols.coef_df, [18.0, 18.0])
    assert 0.0 < ols.r_squared < 1.0 and ols.adj_r_squared < ols.r_squared
    # qt(0.975, 18) = 2.1009220402410382 in R.
    np.testing.assert_allclose((ols.confint()[:, 1] - ols.beta) / ols.se, 2.1009220402410382)
    y = [float(v > 0) for v in _EY]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = glmm.fit(dict(_EMIXED, y=y), "y ~ x + (1 | g)", "binomial")
    assert m.df_method == "none"
    assert np.all(np.isnan(m.coef_df)) and np.all(np.isnan(m.coef_p))


def test_satterthwaite_must_be_a_bool():
    with pytest.raises(TypeError, match="satterthwaite must be True or False"):
        glmm.fit(_EMIXED, "y ~ x + (1 | g)", satterthwaite="no")


def test_satterthwaite_takes_a_numpy_bool():
    off = glmm.fit(_EMIXED, "y ~ x + (1 | g)", satterthwaite=np.False_)
    assert off.df_method == "disabled"
