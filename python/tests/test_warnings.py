import warnings
from pathlib import Path

import numpy as np
import pytest

import glmm

# Deterministic fixtures, mirrored in r/tests/testthat/test-warnings.R.
_X = [float(v) for v in range(-10, 11) if v != 0]  # 20 rows
_Y = [0.3 * x + ((i * 7) % 5 - 2) * 0.4 for i, x in enumerate(_X)]
FIXED = {"x": _X, "y": _Y}
_G = [f"g{i % 5}" for i in range(20)]
_OFF = {"g0": -3.0, "g1": -1.5, "g2": 0.0, "g3": 1.5, "g4": 3.0}
MIXED = {"x": _X, "g": _G, "y": [y + _OFF[g] for y, g in zip(_Y, _G)]}


def printed(entry):
    return f"{entry['tier'].capitalize()}: {entry['title']}. {entry['message']}"


def only(fit, kind):
    (entry,) = [w for w in fit.warnings if w["kind"] == kind]
    assert (entry["tier"], entry["title"]) == glmm._WARNING_KINDS[kind]
    return entry


def test_clean_fit_stores_an_empty_list():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = glmm.fit(MIXED, "y ~ x + (1 | g)")
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
            "warm_start accepts only 'beta' and 'theta'; these keys were ignored: phi, b.",
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
    assert m.nagq == 1


@pytest.mark.parametrize(
    ("data", "formula", "family", "reason"),
    [
        (MIXED, "y ~ x + (1 | g)", "gaussian", "a Gaussian model"),
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


def test_ill_conditioned_is_stored_and_printed():
    n, split = 60, 40
    a = [((i * 13) % 17) - 8.0 for i in range(n)]
    b = [a[i] + (0.0 if i < split else 1.0) for i in range(n)]
    y = [0.5 + 1.3 * a[i] + 0.477 * b[i] + ((i % 3) - 1.0) for i in range(n)]
    w = [1.0 if i < split else 1e-11 for i in range(n)]
    with pytest.warns(glmm.IllConditionedWarning) as caught:
        m = glmm.fit({"y": y, "a": a, "b": b}, "y ~ a + b", weights=w)
    entry = only(m, "ill_conditioned")
    assert entry["message"].startswith("b is almost a combination of other columns")
    assert printed(entry) in [str(w.message) for w in caught]


def test_constructed_notes_store_with_their_tier_and_title():
    # Kinds no fixture here reaches end-to-end: build the note, route it through
    # _note_warning and _warn, and check store and printed text together.
    base = {
        "columns": [],
        "pivot": float("nan"),
        "evals": 0,
        "final_eval": False,
        "detail": "",
        "ratio": float("nan"),
    }
    cases = [
        (
            dict(base, kind="unused_grouping_levels", detail="g: z, w"),
            glmm.UnusedGroupingLevelsWarning,
            (
                "Grouping factor 'g' has levels with no rows (z, w). They stay in the model with "
                "random effects of exactly zero and are counted in the number of groups. Remove "
                "unused categories before fitting."
            ),
        ),
        (
            dict(base, kind="re_design_scale_spread", detail="g", ratio=4200.0),
            glmm.ReDesignScaleWarning,
            (
                "The predictors with random slopes for 'g' are on very different scales (ratio "
                "4.2e+03). The fit is not affected, but the reported random-effect standard "
                "deviations are hard to compare. Rescaling these predictors makes them easier "
                "to read."
            ),
        ),
        (
            dict(base, kind="hessian_se_fallback"),
            glmm.HessianSeFallbackWarning,
            (
                "The usual standard errors could not be computed, so a simpler method was used. "
                "Its standard errors tend to be too small, so p-values and confidence intervals "
                "may look more precise than they are. Standard errors for the random-effect "
                "standard deviations are not available."
            ),
        ),
        (
            dict(base, kind="pirls_exhausted", final_eval=True),
            glmm.PirlsExhaustedWarning,
            (
                "The final step that computes the reported results ran out of iterations. The "
                "estimates and their standard errors may be less accurate than usual. "
                "Try simplifying the random effects or rescaling the predictors."
            ),
        ),
        (
            dict(base, kind="nb_shape_unsettled", evals=25),
            glmm.NbShapeUnsettledWarning,
            (
                "The search for the negative binomial shape parameter stopped at its limit of 25 "
                "rounds before it settled. The coefficients and standard errors are computed at the "
                "last value it reached, which may not be the best one."
            ),
        ),
        (
            dict(base, kind="from_the_future"),
            glmm.DiagnosticWarning,
            (
                "The solver reported something ('from_the_future') that this version of glmm "
                "does not recognize. Please report it at "
                "https://github.com/pawlenartowicz/glmm/issues."
            ),
        ),
    ]
    for note, category, message in cases:
        msg, cat = glmm._note_warning(note, [], True)
        assert (msg, cat) == (message, category), note["kind"]
        store = []
        with pytest.warns(category) as caught:
            glmm._warn(store, note["kind"], msg, cat)
        (entry,) = store
        tier, title = glmm._WARNING_KINDS.get(note["kind"], glmm._UNKNOWN_KIND)
        assert entry == {"tier": tier, "kind": note["kind"], "title": title, "message": message}
        assert str(caught[0].message) == printed(entry)


def test_pirls_exhausted_is_raised_only_on_a_converged_final_eval():
    note = {
        "kind": "pirls_exhausted",
        "columns": [],
        "pivot": float("nan"),
        "evals": 3,
        "final_eval": False,
        "detail": "",
    }
    assert glmm._note_warning(note, [], True) is None  # rejected trial point
    assert glmm._note_warning(note, [], False) is None  # folded into search_limit / fit_failed
    assert glmm._note_warning(dict(note, final_eval=True), [], False) is None
    assert glmm._note_warning(dict(note, final_eval=True), [], True)[1] is (
        glmm.PirlsExhaustedWarning
    )


def test_unused_levels_splits_on_the_first_separator():
    note = {
        "kind": "unused_grouping_levels",
        "columns": [],
        "pivot": float("nan"),
        "evals": 0,
        "final_eval": False,
        "detail": "g: a: b, c",
    }
    msg, _ = glmm._note_warning(note, [], True)
    assert msg.startswith("Grouping factor 'g' has levels with no rows (a: b, c).")


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
    with pytest.warns(glmm.DiagnosticWarning):
        m = glmm.fit(
            MIXED,
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
    # The fixture of test_fit_wiring.py::test_singular_fit_warning_names_component.
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
    entry = only(m, "singular")
    assert entry["message"] == (
        "The random effects are too complex for the data: a variance is estimated at or "
        "near zero, or a correlation at or near −1 or 1. Consider removing the "
        "affected random effect. Affected: (Intercept) in g."
    )
    assert printed(entry) in [str(w.message) for w in caught]


_INNER = "Some of its inner steps ran out of iterations."
_PIRLS = [
    {
        "kind": "pirls_exhausted",
        "columns": [],
        "pivot": float("nan"),
        "evals": 3,
        "final_eval": False,
        "detail": "",
    }
]
_FIN = np.array([1.5, -2.0])
_NAN = np.array([np.nan, np.nan])
_NONE = np.zeros(2, dtype=bool)
_Y = np.array(_Y)  # the varying 20-row response from the top of this file
_ADVICE = "Try a simpler random-effects structure or rescale the predictors."
_SEARCH = (
    "The search for the variance estimates reached its step limit before it settled. "
    "{inner}The estimates shown are the best point found; they are often close, but this is "
    "not checked. Do not use them until the fit converges. " + _ADVICE
)
_FAILED = "The fitting algorithm failed and returned no estimates. {inner}" + _ADVICE
_BINOMIAL = (
    "The fit did not converge. This usually means a predictor, or a combination of "
    "predictors, predicts the outcome perfectly (separation), so some fitted probabilities "
    "go to 0 or 1. The coefficients are from the last step; standard errors are not "
    "reported. Check the data for separation."
)
_COUNTS = (
    "The fit did not converge. This usually means that some category of a predictor, or "
    "some combination of predictors, has only zero counts, so some fitted counts go to 0. "
    "The coefficients are from the last step; standard errors are not reported. Check for "
    "categories whose counts are all zero."
)
_UNSETTLED = (
    "The fit did not settle on an answer: the fitting steps stopped before converging. The "
    "coefficients are from the last step; standard errors are not reported. Check predictors "
    "with extreme values; with a link other than log, the log link is usually more stable."
)
_UNSOLVABLE = (
    "The predictors could not be separated numerically, so no estimates were computed. "
    "Check for predictors that are copies or near-copies of each other."
)


@pytest.mark.parametrize(
    ("family", "mixed", "notes", "beta", "aliased", "deviance", "y", "kind", "category", "message"),
    [
        (
            "gaussian",
            False,
            [],
            _NAN,
            _NONE,
            np.nan,
            _Y,
            "design_unsolvable",
            glmm.DesignUnsolvableWarning,
            _UNSOLVABLE,
        ),
        (
            "binomial",
            False,
            [],
            _FIN,
            _NONE,
            np.nan,
            _Y,
            "glm_diverged",
            glmm.GlmDivergedWarning,
            _BINOMIAL,
        ),
        (
            "poisson",
            False,
            [],
            _FIN,
            _NONE,
            np.nan,
            _Y,
            "glm_diverged",
            glmm.GlmDivergedWarning,
            _COUNTS,
        ),
        (
            "negativebinomial",
            False,
            _PIRLS,
            _FIN,
            _NONE,
            np.nan,
            _Y,
            "glm_diverged",
            glmm.GlmDivergedWarning,
            _COUNTS,
        ),  # no inner-steps sentence without random effects
        (
            "gamma",
            False,
            [],
            _FIN,
            _NONE,
            np.nan,
            _Y,
            "glm_diverged",
            glmm.GlmDivergedWarning,
            _UNSETTLED,
        ),
        (
            "inversegaussian",
            False,
            [],
            _FIN,
            _NONE,
            np.nan,
            _Y,
            "glm_diverged",
            glmm.GlmDivergedWarning,
            _UNSETTLED,
        ),
        (
            "gaussian",
            True,
            [],
            _FIN,
            _NONE,
            12.5,
            _Y,
            "search_limit",
            glmm.SearchLimitWarning,
            _SEARCH.format(inner=""),
        ),
        (
            "binomial",
            True,
            _PIRLS,
            _FIN,
            _NONE,
            12.5,
            _Y,
            "search_limit",
            glmm.SearchLimitWarning,
            _SEARCH.format(inner=_INNER + " "),
        ),
        (
            "poisson",
            True,
            [],
            _NAN,
            _NONE,
            np.nan,
            _Y,
            "fit_failed",
            glmm.FitFailedWarning,
            _FAILED.format(inner=""),
        ),
        (
            "poisson",
            True,
            _PIRLS,
            _NAN,
            _NONE,
            np.nan,
            _Y,
            "fit_failed",
            glmm.FitFailedWarning,
            _FAILED.format(inner=_INNER + " "),
        ),
        # finite coefficients but no end point: failed, not a search limit
        (
            "gaussian",
            True,
            [],
            _FIN,
            _NONE,
            np.nan,
            _Y,
            "fit_failed",
            glmm.FitFailedWarning,
            _FAILED.format(inner=""),
        ),
        # no fixed effects at all (y ~ 0 + (1 | g)): the deviance decides
        (
            "gaussian",
            True,
            [],
            np.array([]),
            np.zeros(0, dtype=bool),
            np.nan,
            _Y,
            "fit_failed",
            glmm.FitFailedWarning,
            _FAILED.format(inner=""),
        ),
        (
            "gaussian",
            True,
            [],
            np.array([]),
            np.zeros(0, dtype=bool),
            12.5,
            _Y,
            "search_limit",
            glmm.SearchLimitWarning,
            _SEARCH.format(inner=""),
        ),
        # an aliased NaN slot is not a failed coefficient
        (
            "gaussian",
            True,
            [],
            np.array([1.0, np.nan]),
            np.array([False, True]),
            12.5,
            _Y,
            "search_limit",
            glmm.SearchLimitWarning,
            _SEARCH.format(inner=""),
        ),
        # constant response wins over every model-based kind, any family
        (
            "binomial",
            False,
            [],
            _NAN,
            _NONE,
            np.nan,
            np.zeros(20),
            "constant_response",
            glmm.ConstantResponseWarning,
            (
                "Every value of the response is 0, so there is nothing to estimate. Check the "
                "response column."
            ),
        ),
        (
            "poisson",
            True,
            _PIRLS,
            _NAN,
            _NONE,
            np.nan,
            np.full(20, 3.0),
            "constant_response",
            glmm.ConstantResponseWarning,
            (
                "Every value of the response is 3, so there is nothing to estimate. Check the "
                "response column."
            ),
        ),
        # nothing to estimate
        (
            "gaussian",
            False,
            [],
            np.array([]),
            np.zeros(0, dtype=bool),
            np.nan,
            _Y,
            "no_coefficients",
            glmm.NoCoefficientsWarning,
            (
                "The model has no coefficients and no random effects, so there is nothing to "
                "estimate. Add an intercept or a predictor."
            ),
        ),
        # rows <= estimated coefficients, with and without random effects
        (
            "gaussian",
            False,
            [],
            _NAN,
            _NONE,
            np.nan,
            np.array([1.0, 2.0]),
            "too_few_rows",
            glmm.TooFewRowsWarning,
            (
                "The model has 2 coefficients to estimate but only 2 rows, so no estimates were "
                "computed. Use more rows or fewer predictors."
            ),
        ),
        (
            "binomial",
            True,
            [],
            _NAN,
            _NONE,
            np.nan,
            np.array([1.0]),
            "too_few_rows",
            glmm.TooFewRowsWarning,
            (
                "The model has 2 coefficients to estimate but only 1 row, so no estimates were "
                "computed. Use more rows or fewer predictors."
            ),
        ),
        (
            "gaussian",
            False,
            [],
            np.array([np.nan, 1.0]),
            np.array([True, False]),
            np.nan,
            np.array([4.0]),
            "too_few_rows",
            glmm.TooFewRowsWarning,
            (
                "The model has 1 coefficient to estimate but only 1 row, so no estimates were "
                "computed. Use more rows or fewer predictors."
            ),
        ),
    ],
)
def test_nonconvergence_picks_one_kind(
    family, mixed, notes, beta, aliased, deviance, y, kind, category, message
):
    got = glmm._nonconvergence(family, mixed, notes, beta, aliased, deviance, y)
    assert got == (kind, message, category)
    store = []
    with pytest.warns(category) as caught:
        glmm._warn(store, *got)
    assert store[0]["tier"] == "severe"
    assert str(caught[0].message) == printed(store[0])


def test_separated_glm_stores_glm_diverged_last():
    data = {"x": _X, "y": [float(x > 0) for x in _X]}
    with pytest.warns(glmm.GlmDivergedWarning) as caught:
        m = glmm.fit(data, "y ~ x", "binomial")
    assert not m.converged
    assert m.warnings[-1]["kind"] == "glm_diverged"
    assert printed(m.warnings[-1]) in [str(w.message) for w in caught]
    assert sum(w["tier"] == "severe" for w in m.warnings) == 1


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


_WARNINGS_MD = Path(__file__).resolve().parents[2] / "documentation" / "warnings.md"


def test_kind_table_matches_warnings_md():
    # Fails rather than skips when the page is missing (good_practices RULE 6).
    rows = set()
    in_table = False
    for line in _WARNINGS_MD.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Warnings"):
            in_table = True
        elif in_table and line.startswith("| ") and not line.startswith(("| Kind", "|---")):
            kind, tier, title = (c.strip() for c in line.split("|")[1:4])
            rows.add((kind.strip("`"), tier.lower(), title))
    expected = {(k, t, ti) for k, (t, ti) in glmm._WARNING_KINDS.items()}
    expected.add(("any other kind", *glmm._UNKNOWN_KIND))
    assert rows == expected


def test_port_parity_sequence():
    # Same data and call in r/tests/testthat/test-warnings.R - change together.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = glmm.fit(MIXED, "y ~ x + (1 | g)", dispersion=2.0, nagq=3)
    assert [(w["tier"], w["kind"], w["title"]) for w in m.warnings] == [
        ("note", "argument_ignored", "Argument ignored"),
        ("note", "argument_ignored", "Argument ignored"),
    ]
