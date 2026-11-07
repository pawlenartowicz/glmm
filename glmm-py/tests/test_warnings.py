import math
import warnings

import numpy as np
import pandas as pd
import pytest

import glmm

# Deterministic fixtures, mirrored in glmm-r/tests/testthat/test-warnings.R.
_X = [float(v) for v in range(-10, 11) if v != 0]  # 20 rows
_Y = [0.3 * x + ((i * 7) % 5 - 2) * 0.4 for i, x in enumerate(_X)]
FIXED = {"x": _X, "y": _Y}
_G = [f"g{i % 5}" for i in range(20)]
_OFF = {"g0": -3.0, "g1": -1.5, "g2": 0.0, "g3": 1.5, "g4": 3.0}
MIXED = {"x": _X, "g": _G, "y": [y + _OFF[g] for y, g in zip(_Y, _G)]}
# MIXED with groups of rows (i // 2) % 5, so y - 0.3x is not constant within a
# group (in MIXED it is, and the LMM fit rides the theta cap): theta-hat is
# interior and sigma-hat^2 is not tiny.
_G_PAIRS = [f"g{(i // 2) % 5}" for i in range(20)]
MIXED_PAIRS = {"x": _X, "g": _G_PAIRS, "y": [y + _OFF[g] for y, g in zip(_Y, _G_PAIRS)]}

# A single fixture that raises two argument_ignored notes from one call
# (dispersion=, nagq=, both inapplicable on a Gaussian model). Its declared but
# empty group level ("g_mid", between two used ones) is dropped and raises
# nothing. Five groups, eight rows each, with a random slope on z, whose values
# sit far from zero. The fit adds two cautions of its own: y has no random slope, so
# the slope's variance comes out at zero, and glmm compares standard
# deviations on its internally standardized design, where that is a singular
# fit (on the raw design z's distance from zero hid it); and y is exact in z
# and g, so the residual variance goes to zero and the intercept's theta ends
# on the upper limit of its search.
_PARITY_Z_SCALE = 1.0e4
_PARITY_N_GROUPS = 5
_PARITY_PER_GROUP = 8
_parity_z = []
_parity_y = []
_parity_g = []
for _gi in range(_PARITY_N_GROUPS):
    for _j in range(_PARITY_PER_GROUP):
        _jitter = _j - (_PARITY_PER_GROUP - 1) / 2.0
        _zv = _PARITY_Z_SCALE + _jitter
        _parity_z.append(_zv)
        _parity_y.append(1.0 + 0.1 * _zv / _PARITY_Z_SCALE + 0.05 * _gi)
        _parity_g.append(f"g{_gi}")
PARITY = {
    "z": _parity_z,
    "g": pd.Categorical(_parity_g, categories=["g0", "g1", "g2", "g_mid", "g3", "g4"]),
    "y": _parity_y,
}


def printed(entry):
    return f"{entry['tier'].capitalize()}: {entry['title']}. {entry['message']}"


def only(fit, kind):
    (entry,) = [w for w in fit.warnings if w["kind"] == kind]
    return entry


def test_clean_fit_stores_an_empty_list():
    # MIXED_PAIRS, not MIXED: MIXED rides the theta limit and warns.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = glmm.fit(MIXED_PAIRS, "y ~ x + (1 | g)")
    assert m.warnings == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"formula": "y ~ x", "dispersion": 2.0},
            "dispersion= has no effect for family 'gaussian'.",
        ),
        (
            {"formula": "y ~ x", "init_theta": 1.5},
            "init_theta= is not used for family 'gaussian'.",
        ),
        (
            {"formula": "y ~ x", "warm_start": {"beta": [0.0, 0.0], "phi": 1.0, "b": 2}},
            "warm_start accepts only 'beta', 'theta' and 'dispersion'; these entries were ignored: phi, b.",
        ),
    ],
)
def test_argument_ignored_is_stored_and_printed(kwargs, message):
    with pytest.warns(glmm.ArgumentIgnoredWarning) as caught:
        m = glmm.fit(FIXED, **kwargs)
    entry = only(m, "argument_ignored")
    assert entry["message"] == message
    assert printed(entry) in [str(w.message) for w in caught]
    assert printed(entry).startswith("Note: Argument ignored. ")


def test_quasi_dispersion_on_mixed_binomial_is_stored():
    data = dict(MIXED, y=[float(y > 0) for y in MIXED["y"]])
    with pytest.warns(glmm.ArgumentIgnoredWarning) as caught:
        m = glmm.fit(data, "y ~ x + (1 | g)", "binomial", dispersion="estimate")
    entry = only(m, "argument_ignored")
    assert entry["message"] == (
        "Quasi-likelihood dispersion= is not supported yet for binomial or Poisson models. "
        "The default dispersion of 1 was used."
    )
    assert printed(entry) in [str(w.message) for w in caught]


def test_agq_fallback_is_stored_and_printed():
    # A binomial model with two grouping factors: mixed, but outside what AGQ covers.
    data = dict(MIXED, y=[float(y > 0) for y in MIXED["y"]], h=[f"h{i % 4}" for i in range(20)])
    with pytest.warns(glmm.AgqFallbackWarning) as caught:
        m = glmm.fit(data, "y ~ x + (1 | g) + (1 | h)", "binomial", nagq=3)
    entry = only(m, "agq_fallback")
    assert entry["message"] == (
        "nagq=3 works only for binomial, Poisson, negative-binomial or Gamma models whose "
        "random effects are in one grouping factor, with at most 3 random effects in it. "
        "This model was fitted without adaptive quadrature."
    )
    assert printed(entry) in [str(w.message) for w in caught]
    assert [w["kind"] for w in m.warnings] == ["agq_fallback"]
    assert m.nagq == 1


@pytest.mark.parametrize(
    ("data", "formula", "family", "reason"),
    [
        (MIXED_PAIRS, "y ~ x + (1 | g)", "gaussian", "a Gaussian model"),
        (
            {"x": _X, "y": [float(v > 0) for v in _Y]},
            "y ~ x",
            "binomial",
            "a model without random effects",
        ),
    ],
)
def test_nagq_that_changes_nothing_is_an_ignored_argument(data, formula, family, reason):
    with pytest.warns(glmm.ArgumentIgnoredWarning) as caught:
        m = glmm.fit(data, formula, family, nagq=3)
    entry = only(m, "argument_ignored")
    assert (
        entry["message"] == f"nagq=3 has no effect for {reason}, because nothing is approximated."
    )
    assert printed(entry) in [str(w.message) for w in caught]
    assert not any(w["kind"] == "agq_fallback" for w in m.warnings)
    assert m.nagq == 1


def test_single_level_grouping_dropped_and_other_grouping_still_fits():
    # "onegroup" has one level; "h" has real groups. The random effect for
    # "onegroup" is dropped with a warning, and "h" is promoted to primary
    # and still fitted.
    n = 40
    h = [f"h{i % 4}" for i in range(n)]
    x = [float(i) for i in range(n)]
    y = [1.0 + 0.2 * xv + (0.5 if hv in ("h0", "h1") else -0.5) for xv, hv in zip(x, h)]
    data = {"y": y, "x": x, "onegroup": ["only"] * n, "h": h}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = glmm.fit(data, "y ~ x + (1 | onegroup) + (1 | h)")
    entry = only(m, "single_level_grouping_dropped")
    assert entry["message"].startswith("Grouping factor 'onegroup' has only one level")
    assert [group for group, _ in m.re_groups] == ["h"]
    assert len(m.varcorr) == 1


def test_every_grouping_single_level_fits_as_a_model_without_random_effects():
    # Both declared groupings are single-level: the fit routes exactly like
    # "y ~ x" (OLS here), keeping both warnings.
    n = 20
    x = [float(i) for i in range(n)]
    y = [1.0 + 0.2 * xv for xv in x]
    data = {"y": y, "x": x, "g1": ["only"] * n, "g2": ["one"] * n}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = glmm.fit(data, "y ~ x + (1 | g1) + (1 | g2)")
    kinds = [w["kind"] for w in m.warnings]
    assert kinds.count("single_level_grouping_dropped") == 2
    assert m.converged
    assert m.re_groups == []
    assert m.varcorr == []
    assert m.ranef.size == 0


def test_ignored_warnings_are_still_stored():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = glmm.fit(FIXED, "y ~ x", dispersion=2.0)
    assert [w["kind"] for w in m.warnings] == ["argument_ignored"]


def test_warning_points_at_the_caller():
    with pytest.warns(glmm.ArgumentIgnoredWarning) as caught:
        glmm.fit(FIXED, "y ~ x", dispersion=2.0)
    assert caught[0].filename == __file__


def test_store_is_per_fit():
    with pytest.warns(glmm.ArgumentIgnoredWarning):
        noisy = glmm.fit(FIXED, "y ~ x", dispersion=2.0)
    clean = glmm.fit(FIXED, "y ~ x")
    assert len(noisy.warnings) == 1
    assert clean.warnings == []


def test_argument_order_is_the_documented_one():
    # MIXED_PAIRS, not MIXED: MIXED rides the theta limit and warns.
    with pytest.warns(glmm.DiagnosticWarning):
        m = glmm.fit(
            MIXED_PAIRS,
            "y ~ x + (1 | g)",
            dispersion=2.0,
            init_theta=1.5,
            warm_start={"beta": [0.0, 0.0], "phi": 1.0},
            nagq=3,
        )
    assert [w["kind"] for w in m.warnings] == [
        "argument_ignored",
        "argument_ignored",
        "argument_ignored",
        "argument_ignored",
    ]
    assert [w["message"][:10] for w in m.warnings] == [
        "dispersion",
        "init_theta",
        "warm_start",
        "nagq=3 has",
    ]


def test_singular_is_stored_and_printed():
    rng = np.random.default_rng(1)
    x = rng.normal(size=120)
    p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x)))
    data = {
        "x": x.tolist(),
        "g": [f"g{i}" for i in np.repeat(np.arange(30), 4).tolist()],
        "y": rng.binomial(1, p).astype(float).tolist(),
    }
    with pytest.warns(glmm.SingularFitWarning) as caught:
        m = glmm.fit(data, "y ~ x + (1 | g)", "binomial")
    assert m.converged
    assert m.singular
    assert [w["kind"] for w in m.warnings] == ["singular"]
    entry = only(m, "singular")
    assert entry["message"] == (
        "The random effects are too complex for the data: a variance is estimated at or "
        "near zero, or a correlation at or near −1 or 1. Consider removing the "
        "affected random effect. Affected: (Intercept) in g."
    )
    assert printed(entry) in [str(w.message) for w in caught]


def test_separated_glm_stores_glm_diverged_last():
    data = {"x": _X, "y": [float(x > 0) for x in _X]}
    with pytest.warns(glmm.GlmDivergedWarning) as caught:
        m = glmm.fit(data, "y ~ x", "binomial")
    assert not m.converged
    assert m.warnings[-1]["kind"] == "glm_diverged"
    assert printed(m.warnings[-1]) in [str(w.message) for w in caught]
    assert sum(w["tier"] == "severe" for w in m.warnings) == 1


def test_separated_glmm_stores_glm_diverged_last():
    data = {"x": _X, "g": _G, "y": [float(x > 0) for x in _X]}
    with pytest.warns(glmm.GlmDivergedWarning) as caught:
        m = glmm.fit(data, "y ~ x + (1 | g)", "binomial")
    assert not m.converged
    assert m.warnings[-1]["kind"] == "glm_diverged"
    assert printed(m.warnings[-1]) in [str(w.message) for w in caught]
    assert sum(w["tier"] == "severe" for w in m.warnings) == 1


def test_glm_whose_first_solve_fails_returns_no_coefficients():
    # A 5e-324 weight on the only row that tells x1 from x2 makes X'WX singular to
    # rounding: no step is taken, so no coefficient is reported.
    data = {"x1": [1.0, 1.0, 1.0, 1.0], "x2": [0.0, 1.0, 1.0, 1.0], "y": [1.0, 2.0, 3.0, 2.5]}
    with pytest.warns(glmm.DesignUnsolvableWarning):
        m = glmm.fit(
            data, "y ~ 0 + x1 + x2", "gamma", link="log", weights=np.array([5e-324, 1, 1, 1])
        )
    assert not m.converged
    assert np.all(np.isnan(m.beta))
    assert m.warnings[-1]["kind"] == "design_unsolvable"


def test_reciprocal_link_response_out_of_range_names_the_cause():
    # 1/mu = 1/(1e80 * ...) is below the kernel's 1e-75 bound on the Gamma inverse link.
    y = [1e80 * (2.0 + 0.1 * x) for x in _X]
    with pytest.warns(glmm.GlmDivergedWarning):
        m = glmm.fit({"x": _X, "y": y}, "y ~ x", "gamma", link="inverse")
    assert not m.converged
    assert m.warnings[-1]["kind"] == "glm_diverged"
    assert "Rescale the response" in m.warnings[-1]["message"]
    # The GLMM searches on a rescaled response when the cluster means are far
    # out of scale (src/fit/glmm.rs, `gamma_inverse_units`), so it fits these
    # data as it fits y * 1e-80, the log-likelihood moved by -n * ln(1e80).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = glmm.fit({"x": _X, "g": _G, "y": y}, "y ~ x + (1 | g)", "gamma", link="inverse")
        small = glmm.fit(
            {"x": _X, "g": _G, "y": [v * 1e-80 for v in y]},
            "y ~ x + (1 | g)",
            "gamma",
            link="inverse",
        )
    assert m.converged and small.converged
    want = small.loglik - len(y) * math.log(1e80)
    assert abs(m.loglik - want) < 1e-6 * abs(want)


def test_all_zero_binomial_is_constant_response_not_separation():
    with pytest.warns(glmm.ConstantResponseWarning):
        m = glmm.fit({"x": _X, "y": [0.0] * 20}, "y ~ x", "binomial")
    assert not m.converged
    assert [w["kind"] for w in m.warnings] == ["constant_response"]


def test_too_few_rows_end_to_end():
    # 3 rows, 3 linearly independent columns: nothing is aliased, and n <= p.
    data = {"y": [1.0, 2.0, 4.0], "x1": [0.0, 1.0, 3.0], "x2": [1.0, 0.0, 2.0]}
    with pytest.warns(glmm.TooFewRowsWarning):
        m = glmm.fit(data, "y ~ x1 + x2")
    assert not m.converged
    assert m.warnings[-1]["message"].startswith(
        "The model has 3 coefficients to estimate but only 3 rows"
    )


def test_port_parity_sequence_adds_three_more_kinds():
    # Same PARITY fixture and call in glmm-r/tests/testthat/test-warnings.R —
    # change together. argument_ignored fires twice, from two different
    # arguments, plus the fit's own singular and variance_at_limit cautions.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = glmm.fit(PARITY, "y ~ z + (1 + z | g)", dispersion=2.0, nagq=3)
    assert m.converged
    assert [(w["tier"], w["kind"], w["title"]) for w in m.warnings] == [
        ("note", "argument_ignored", "Argument ignored"),
        ("note", "argument_ignored", "Argument ignored"),
        ("caution", "singular", "Singular fit"),
        ("caution", "variance_at_limit", "Variance at its search limit"),
    ]


def test_port_parity_non_integer_response():
    # Same fixture and call in glmm-r/tests/testthat/test-warnings.R — change
    # together. Intercept-only Poisson, mirrors the Rust unit test
    # poisson_non_integer_response_is_noted (src/fit/common_tests.rs).
    with pytest.warns(glmm.NonIntegerResponseWarning, match="not a whole number"):
        m = glmm.fit({"y": [1.0, 2.5, 3.0, 4.0]}, "y ~ 1", "poisson")
    assert m.converged
    assert [(w["tier"], w["kind"], w["title"]) for w in m.warnings] == [
        ("caution", "non_integer_response", "Non-integer response"),
    ]


# A random intercept and slope on 10 subjects, two of them with one row: 20
# random effects for 18 rows. Same fixture and call in
# glmm-r/tests/testthat/test-warnings.R - change together.
_FEW_ROWS_ID = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 9, 10, 10]
_FEW_ROWS_Y = [0.26, -0.09, 0.46, -1.73, -0.38, -0.91, -0.86, 0.91, -1.42]
_FEW_ROWS_Y += [-0.40, -1.23, -1.62, -1.46, -0.49, 1.31, 0.85, 0.06, 0.36]
FEW_ROWS = {
    "y": _FEW_ROWS_Y,
    "time": [float(t) for t in (0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1)],
    "id": [f"s{i}" for i in _FEW_ROWS_ID],
}


def test_port_parity_few_rows_for_random_effects():
    with pytest.warns(glmm.FewRowsForRandomEffectsWarning) as caught:
        m = glmm.fit(FEW_ROWS, "y ~ time + (1 + time | id)")
    entry = only(m, "few_rows_for_random_effects")
    assert (entry["tier"], entry["title"]) == ("caution", "Too few rows for the random effects")
    assert entry["message"] == (
        "The term (1 + time | id) has 20 random effects for 18 rows, so its variance "
        "parameters probably cannot be estimated from these data; lme4 stops on such a model "
        "by default. The estimates are shown; read the variance components with care, or "
        "simplify the random-effects structure."
    )
    assert printed(entry) in [str(w.message) for w in caught]


def test_port_parity_variance_at_limit():
    # Same fixture and call in glmm-r/tests/testthat/test-warnings.R - change together.
    # On MIXED, y - 0.3x is constant within each group, so the residual variance
    # goes to zero and theta ends on the upper limit of its search.
    with pytest.warns(glmm.VarianceAtLimitWarning) as caught:
        m = glmm.fit(MIXED, "y ~ x + (1 | g)")
    assert m.converged
    assert not m.singular
    assert m.diagnostics["boundary"] == "upper_limit"
    assert [w["kind"] for w in m.warnings] == ["variance_at_limit"]
    entry = only(m, "variance_at_limit")
    assert (entry["tier"], entry["title"]) == ("caution", "Variance at its search limit")
    assert entry["message"] == (
        "A random-effect parameter reached the upper limit of the search (1000 on glmm's "
        "internal scale). On a Gaussian model this usually means the residual variance is "
        "estimated at or near zero; on other families the random-effect variance has no "
        "finite estimate. The estimates are at that limit, not at an optimum. Check for a "
        "response that the predictors and groups explain exactly."
    )
    assert printed(entry) in [str(w.message) for w in caught]


def test_port_parity_rows_dropped_na():
    # Same fixture and call in glmm-r/tests/testthat/test-warnings.R — change
    # together. One NaN in x drops one of four rows.
    data = {"x": [1.0, 2.0, float("nan"), 4.0], "y": [2.1, 3.9, 8.2, 8.1]}
    with pytest.warns(glmm.RowsDroppedWarning):
        m = glmm.fit(data, "y ~ x")
    assert m.converged
    assert m.nobs == 3
    assert [(w["tier"], w["kind"], w["title"]) for w in m.warnings] == [
        ("caution", "rows_dropped_na", "Rows dropped for missing values"),
    ]
    (entry,) = m.warnings
    assert entry["message"] == (
        "Dropped 1 of 4 row(s): a column the formula uses had a missing value there."
    )


def test_degenerate_gaussian_lmm_carries_the_satterthwaite_note():
    # Same fixture and call in glmm-r/tests/testthat/test-warnings.R - change together.
    # A random slope on an aliased column (x2 = 2x) leaves a NaN-filled fit with no
    # finite end point, so the Satterthwaite pass is recorded as failed.
    data = dict(MIXED, x2=[2.0 * v for v in _X])
    with (
        pytest.warns(glmm.FitFailedWarning),
        pytest.warns(glmm.SatterthwaiteCurvatureWarning, match="could not be computed"),
    ):
        m = glmm.fit(data, "y ~ x + x2 + (1 + x2 | g)")
    assert not m.converged
    assert m.df_method == "failed"
    assert [w["kind"] for w in m.warnings] == ["satterthwaite_curvature", "fit_failed"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off = glmm.fit(data, "y ~ x + x2 + (1 + x2 | g)", satterthwaite=False)
    assert off.df_method == "disabled"
    assert [w["kind"] for w in off.warnings] == ["fit_failed"]
