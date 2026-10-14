import dataclasses
import inspect
import math
import re
import warnings

import numpy as np
import pytest

import glmm


def test_module_surface():
    # `fit`, `Fit`, the tests of `glmm.anova` (`anova`, `compare` and their
    # three result classes), and the twenty-four warning categories the
    # diagnostics channel raises; a user has to be able to name them to filter
    # on them.
    assert glmm.__all__ == [
        "AgqFallbackWarning",
        "ArgumentIgnoredWarning",
        "Comparison",
        "ComparisonTable",
        "ConstantResponseWarning",
        "DesignUnsolvableWarning",
        "DiagnosticWarning",
        "ExactProfileFallbackWarning",
        "Fit",
        "FitFailedWarning",
        "GlmDivergedWarning",
        "GradientSearchFallbackWarning",
        "HessianSeFallbackWarning",
        "IllConditionedWarning",
        "NbShapeUnsettledWarning",
        "NoCoefficientsWarning",
        "NonIntegerResponseWarning",
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
        "UnusedGroupingLevelsWarning",
        "anova",
        "compare",
        "fit",
    ]
    assert issubclass(glmm.IllConditionedWarning, glmm.DiagnosticWarning)
    assert issubclass(glmm.ExactProfileFallbackWarning, glmm.DiagnosticWarning)
    assert issubclass(glmm.DiagnosticWarning, UserWarning)


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
        "reml",
        "anova_test",
    ]:
        assert p[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert p["link"].default is None
    assert p["wald_se"].default == "hessian"
    assert p["nagq"].default == 1
    assert p["satterthwaite"].default is True
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


def test_reml_false_on_a_glmm_is_the_default_fit_without_a_warning():
    base = _fit_clean("yb ~ x + (1 | g)", "binomial")
    m = _fit_clean("yb ~ x + (1 | g)", "binomial", reml=False)
    assert m.warnings == []
    _assert_same_fit(m, base)


@pytest.mark.parametrize(
    ("formula", "family", "link"),
    [("yc ~ x + (1 | g)", "poisson", None), ("yp ~ x", "gamma", "log")],
)
def test_reml_false_on_a_poisson_glmm_and_a_gamma_glm_is_the_default_fit(formula, family, link):
    base = _fit_clean(formula, family, link=link)
    m = _fit_clean(formula, family, link=link, reml=False)
    assert m.warnings == []
    assert not m.reml
    _assert_same_fit(m, base)


def test_reml_true_on_ols_reports_the_reml_loglik():
    base = _fit_clean("y ~ x")
    m = _fit_clean("y ~ x", reml=True)
    assert not base.reml
    assert m.reml
    assert np.array_equal(m.beta, base.beta)
    assert np.array_equal(m.se, base.se)
    assert m.dispersion == base.dispersion
    # R's logLik(lm, REML = TRUE): -((n - p)(1 + ln 2 pi) + (n - p) ln(RSS / (n - p))
    # + ln|X'X|) / 2, here on an independent least-squares solve.
    X = np.column_stack([np.ones(len(_x)), _x])
    y = np.asarray(REML_DATA["y"])
    rss = float(np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2))
    nu = X.shape[0] - X.shape[1]
    logdet = np.linalg.slogdet(X.T @ X)[1]
    expected = -0.5 * (nu * (1 + math.log(2 * math.pi)) + nu * math.log(rss / nu) + logdet)
    assert math.isclose(m.loglik, expected, rel_tol=1e-12)


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


def test_reml_false_on_ols_is_the_default_fit():
    _assert_same_fit(_fit_clean("y ~ x", reml=False), _fit_clean("y ~ x"))


@pytest.mark.parametrize("value", [True, False])
def test_reml_on_a_dropped_single_level_grouping_keeps_the_note(value):
    with pytest.warns(glmm.SingleLevelGroupingDroppedWarning):
        m = glmm.fit(REML_DATA, "y ~ x + (1 | one)", reml=value)
    assert [w["kind"] for w in m.warnings] == ["single_level_grouping_dropped"]
    assert m.re_groups == []
    assert m.reml is value
    # The OLS row: the formula without the grouping, apart from the formula
    # and the dropped-grouping note.
    m = dataclasses.replace(m, formula="y ~ x", diagnostics={**m.diagnostics, "notes": []})
    _assert_same_fit(m, _fit_clean("y ~ x", reml=value))


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
