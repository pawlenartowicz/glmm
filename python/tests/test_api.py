import inspect

import pytest

import glmm

DATA = {"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]}


def test_module_surface():
    # `fit`, `Fit`, and the six warning categories the diagnostics channel
    # raises — a user has to be able to name them to filter on them.
    assert glmm.__all__ == [
        "DiagnosticWarning",
        "Fit",
        "HessianSeFallbackWarning",
        "IllConditionedWarning",
        "PirlsExhaustedWarning",
        "ReDesignScaleWarning",
        "UnusedGroupingLevelsWarning",
        "fit",
    ]
    assert issubclass(glmm.IllConditionedWarning, glmm.DiagnosticWarning)
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
    ]:
        assert p[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert p["link"].default is None
    assert p["wald_se"].default == "hessian"
    assert p["nagq"].default == 1


def test_typo_kwarg_is_typeerror():
    with pytest.raises(TypeError):
        glmm.fit(DATA, "y ~ x", nagk=3)  # misspelled nagq


def test_valid_call_returns_fit():
    result = glmm.fit(DATA, "y ~ x")
    assert isinstance(result, glmm.Fit)
    assert result.names == ["(Intercept)", "x"]
    assert result.converged


def test_fit_fields():
    assert list(glmm.Fit.__dataclass_fields__) == [
        "beta",
        "se",
        "vcov",
        "tau2",
        "varcorr",
        "stddev_se",
        # converged / singular / aliased live in here and are re-exposed as
        # properties, so they are deliberately absent from the field list.
        "diagnostics",
        "dispersion",
        "names",
        "re_groups",
        "n_eval",
        "deviance",
        "loglik",
        "df",
        "reml",
        "fitted",
        "ranef",
        "ranef_levels",
        "ranef_blocks",
    ]
