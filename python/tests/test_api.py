import inspect

import glmm


def test_module_surface():
    # `fit`, `Fit`, and the twenty-one warning categories the diagnostics
    # channel raises — a user has to be able to name them to filter on them.
    assert glmm.__all__ == [
        "AgqFallbackWarning",
        "ArgumentIgnoredWarning",
        "ConstantResponseWarning",
        "DesignUnsolvableWarning",
        "DiagnosticWarning",
        "ExactProfileFallbackWarning",
        "Fit",
        "FitFailedWarning",
        "GlmDivergedWarning",
        "HessianSeFallbackWarning",
        "IllConditionedWarning",
        "NbShapeUnsettledWarning",
        "NoCoefficientsWarning",
        "NonIntegerResponseWarning",
        "PirlsExhaustedWarning",
        "ReDesignScaleWarning",
        "RowsDroppedWarning",
        "SearchLimitWarning",
        "SingleLevelGroupingDroppedWarning",
        "SingularFitWarning",
        "TooFewRowsWarning",
        "UnusedGroupingLevelsWarning",
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
