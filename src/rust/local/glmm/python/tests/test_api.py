import dataclasses
import inspect
import math
import pickle
import re
import warnings

import numpy as np
import pytest

import glmm


def test_module_surface():
    # `fit`, `Fit`, `icc`, `model_matrix`, `term_labels`, the tests of
    # `glmm.anova` (`anova`, `compare` and their three result classes),
    # `pairwise` and its `PairwiseTable`, and the twenty-seven
    # warning categories the diagnostics channel raises; a user has to be able
    # to name them to filter on them.
    assert glmm.__all__ == [
        "AgqFallbackWarning",
        "ArgumentIgnoredWarning",
        "Comparison",
        "ComparisonTable",
        "ConstantResponseWarning",
        "DesignUnsolvableWarning",
        "DiagnosticWarning",
        "ExactProfileFallbackWarning",
        "FewRowsForRandomEffectsWarning",
        "Fit",
        "FitFailedWarning",
        "GlmDivergedWarning",
        "GradientSearchFallbackWarning",
        "HessianSeFallbackWarning",
        "IllConditionedWarning",
        "NbShapeUnsettledWarning",
        "NewGroupingLevelsWarning",
        "NoCoefficientsWarning",
        "NonIntegerResponseWarning",
        "PairwiseTable",
        "PirlsExhaustedWarning",
        "PrecisionLimitedWarning",
        "RowsDroppedWarning",
        "SatterthwaiteCurvatureWarning",
        "SeUnavailableWarning",
        "SearchLimitWarning",
        "SingleLevelGroupingDroppedWarning",
        "SingularFitWarning",
        "TermTable",
        "TooFewRowsWarning",
        "VarianceAtLimitWarning",
        "anova",
        "compare",
        "fit",
        "model_matrix",
        "pairwise",
        "term_labels",
    ]
    assert issubclass(glmm.IllConditionedWarning, glmm.DiagnosticWarning)
    assert issubclass(glmm.ExactProfileFallbackWarning, glmm.DiagnosticWarning)
    assert issubclass(glmm.DiagnosticWarning, UserWarning)
    assert all(hasattr(glmm, name) for name in glmm.__all__)


def test_term_labels_reads_the_formula_alone():
    assert glmm.term_labels("~ 0 + a*(b + c)") == (["a", "b", "c", "a:b", "a:c"], False)
    with pytest.raises(ValueError, match="formula syntax error"):
        glmm.term_labels("y ~ x + (1 | g")


def test_fit_signature_matches_spec():
    sig = inspect.signature(glmm.fit)
    assert list(sig.parameters) == [
        "data",
        "formula",
        "family",
        "link",
        "dispersion",
        "init_theta",
        "weights",
        "offset",
        "wald_se",
        "nagq",
        "warm_start",
        "satterthwaite",
        "condvar",
        "reml",
        "anova_test",
    ]
    p = sig.parameters
    assert p["family"].default == "gaussian"
    # Everything after family is keyword-only.
    for name in [
        "link",
        "dispersion",
        "init_theta",
        "weights",
        "offset",
        "wald_se",
        "nagq",
        "warm_start",
        "satterthwaite",
        "condvar",
        "reml",
        "anova_test",
    ]:
        assert p[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert p["link"].default is None
    assert p["wald_se"].default == "hessian"
    assert p["nagq"].default == 1
    assert p["satterthwaite"].default is True
    assert p["condvar"].default is True
    assert p["reml"].default is None
    assert p["anova_test"].default == "auto"


# `reml=`: twelve groups of eight rows with a random intercept, a Gaussian, a
# binary, a count and a positive response, and a single-level column `one`.
# Every default fit below is clean (no warning) and converges off the boundary.
_rng = np.random.default_rng(20261012)
_N_GROUPS, _PER_GROUP = 12, 8
_g = np.repeat([f"g{i:02d}" for i in range(_N_GROUPS)], _PER_GROUP)
_x = _rng.normal(size=_g.size)
_u = np.repeat(_rng.normal(size=_N_GROUPS), _PER_GROUP)
REML_DATA = {
    "x": _x.tolist(),
    "g": _g.tolist(),
    "one": ["only"] * _g.size,
    "y": (1.0 + 0.5 * _x + _u + 0.5 * _rng.normal(size=_g.size)).tolist(),
    "yb": _rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.2 + 0.8 * _x + _u)))).astype(float).tolist(),
    "yc": _rng.poisson(np.exp(0.2 + 0.4 * _x + _u)).astype(float).tolist(),
    "yp": _rng.gamma(5.0, np.exp(0.5 + 0.3 * _x) / 5.0).tolist(),
}


def _same(a, b):
    """Field-by-field equality of two fits' values, NaN equal to NaN (a
    GLMM's `coef_df` and `coef_p` are all NaN)."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(u, v) for u, v in zip(a, b))
    if isinstance(a, (np.ndarray, float)):
        return np.array_equal(a, b, equal_nan=True)
    return a == b


def _assert_same_fit(m, base):
    for f in dataclasses.fields(glmm.Fit):
        if f.name != "warnings":
            assert _same(getattr(m, f.name), getattr(base, f.name)), f.name


def _fit_clean(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return glmm.fit(REML_DATA, *args, **kwargs)


def test_reml_true_on_a_glmm_is_ignored_with_a_warning():
    base = _fit_clean("yb ~ x + (1 | g)", "binomial")
    with pytest.warns(glmm.ArgumentIgnoredWarning) as caught:
        m = glmm.fit(REML_DATA, "yb ~ x + (1 | g)", "binomial", reml=True)
    (entry,) = m.warnings
    assert entry["kind"] == "argument_ignored"
    assert entry["message"] == (
        "reml=True applies only to Gaussian models; it was ignored and the model was "
        "fitted by maximum likelihood."
    )
    assert [str(w.message) for w in caught] == [f"Note: Argument ignored. {entry['message']}"]
    assert not m.reml
    _assert_same_fit(m, base)


@pytest.mark.parametrize("value", [True, np.True_])
def test_reml_true_on_a_gaussian_mixed_model_is_the_default_fit(value):
    base = _fit_clean("y ~ x + (1 | g)")
    m = _fit_clean("y ~ x + (1 | g)", reml=value)
    assert m.reml is True
    _assert_same_fit(m, base)


def test_reml_false_on_a_gaussian_mixed_model_fits_by_ml():
    base = _fit_clean("y ~ x + (1 | g)")
    m = _fit_clean("y ~ x + (1 | g)", reml=False)
    assert base.reml is True
    assert m.reml is False
    # Equality with the Rust `Fit` is a port-parity check, not this test's.
    assert m.loglik != base.loglik
    assert m.df_method == "satterthwaite"
    assert np.all(np.isfinite(m.coef_df))
    assert np.all(m.coef_df <= m.nobs)


@pytest.mark.parametrize("value", [1, 0, 1.0, np.int64(1), "no"])
def test_reml_must_be_none_or_a_bool(value):
    msg = re.escape(f"reml must be None, True or False, got {value!r}")
    with pytest.raises(TypeError, match=msg):
        glmm.fit(REML_DATA, "y ~ x", reml=value)


# `Fit.aic` and `Fit.bic`: the value on every fit, with no warning. The
# `reml=False` mixed fit is ML, so its summary prints AIC and BIC.
@pytest.mark.parametrize(
    "args, kwargs",
    [
        (("y ~ x + (1 | g)",), {}),
        (("y ~ x + (1 | g)",), {"reml": False}),
        (("yb ~ x + (1 | g)", "binomial"), {}),
        (("y ~ x",), {}),
    ],
    ids=["reml_mixed", "ml_mixed", "binomial_glmm", "ols"],
)
def test_aic_and_bic_are_returned_on_every_fit(args, kwargs):
    m = _fit_clean(*args, **kwargs)
    if kwargs.get("reml") is False:
        assert m.reml is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aic, bic = m.aic, m.bic
        criterion = m.summary_object().criterion
    assert aic == -2 * m.loglik + 2 * m.df
    assert bic == -2 * m.loglik + m.df * math.log(m.nobs)
    assert math.isfinite(aic) and math.isfinite(bic)
    # Properties, not dataclass fields.
    assert {"aic", "bic"}.isdisjoint(dataclasses.asdict(m))
    if m.reml:
        assert list(criterion) == ["REML criterion at convergence"]
    else:
        assert (criterion["AIC"], criterion["BIC"]) == (aic, bic)


# R-squared and the ICC: a sleepstudy-shaped LMM, 18 subjects over 10 days
# with a random intercept and slope.
_rs = np.random.default_rng(20261026)
_subject = np.repeat(np.arange(18), 10)
_days = np.tile(np.arange(10.0), 18)
SLEEP = {
    "Reaction": (
        250
        + 10 * _days
        + _rs.normal(0, 25, 18)[_subject]
        + _rs.normal(0, 6, 18)[_subject] * _days
        + _rs.normal(0, 25, _days.size)
    ).tolist(),
    "Days": _days.tolist(),
    "Subject": [f"s{i:02d}" for i in _subject],
}


def _sleep_fit(formula):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return glmm.fit(SLEEP, formula)


def _r2_line(m):
    r2, icc = m.r2(), m.icc()
    return (
        f"Marginal R2: {r2['marginal']:.3f}, conditional R2: {r2['conditional']:.3f}, "
        f"adjusted ICC: {icc['adjusted']:.3f}"
    )


def test_summary_of_a_mixed_model_prints_r2_and_the_adjusted_icc():
    m = _sleep_fit("Reaction ~ Days + (Days | Subject)")
    r2, icc = m.r2(), m.icc()
    assert list(r2) == ["marginal", "conditional"]
    assert list(icc) == ["adjusted", "unadjusted"]
    assert 0 < r2["marginal"] < r2["conditional"] < 1
    lines = m.summary_object().text().splitlines()
    i = lines.index(_r2_line(m))
    assert lines[i - 1].startswith("Number of obs: 180, groups:")
    s = m.summary_object()
    for render in (s.html(), s.latex(), s.typst()):
        assert _r2_line(m) in render


def test_a_fit_without_random_effects_has_no_r2_line_and_nan_values():
    m = _sleep_fit("Reaction ~ Days")
    assert all(math.isnan(v) for v in (*m.r2().values(), *m.icc().values()))
    assert "Marginal R2" not in m.summary_object().text()


_REFIT_LINE = "R2 and ICC need a refit of the null model: call fit.r2() and fit.icc()"


def test_summary_of_a_poisson_glmm_points_to_r2_and_icc_instead_of_refitting():
    m = glmm.fit(REML_DATA, "yc ~ x + (1 | g)", "poisson")
    s = m.summary_object()
    lines = s.text().splitlines()
    i = lines.index(_REFIT_LINE)
    assert lines[i - 1].startswith("Number of obs: 96, groups:")
    assert "Marginal R2" not in s.text()
    for render in (s.html(), s.latex(), s.typst()):
        assert _REFIT_LINE in render
    r2 = m.r2()
    assert 0 < r2["marginal"] < r2["conditional"] < 1
    assert all(math.isfinite(v) for v in m.icc().values())


def test_a_fit_stored_by_0_6_x_prints_its_summary_and_refuses_r2_and_icc():
    m = _sleep_fit("Reaction ~ Days + (Days | Subject)")
    o = object.__new__(glmm.Fit)
    o.__dict__.update({k: v for k, v in m.__dict__.items() if k != "_kernel_parts"})
    old = pickle.loads(pickle.dumps(o))
    assert old.summary_object().text() == m.summary_object().text().replace(_r2_line(m) + "\n", "")
    for method in (old.r2, old.icc):
        with pytest.raises(ValueError, match=r"pairwise, r2 and icc read\. Refit"):
            method()
