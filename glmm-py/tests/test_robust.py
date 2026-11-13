"""Robust fits (se_type): CR2 and HC2 against clubSandwich, the default,
the robust notes and refusals. Mirrors glmm-r/tests/testthat/test-robust.R -
change together."""

import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

import glmm

_VALIDATION = Path(__file__).resolve().parents[2] / "validation"


def _golden(name):
    return json.loads((_VALIDATION / "goldens" / "inference" / "cr2" / f"{name}.json").read_text())


def _data(name):
    """`validation/data/inference/<name>.csv` as a dict of columns, `g` as labels."""
    path = _VALIDATION / "data" / "inference" / f"{name}.csv"
    raw = np.genfromtxt(path, delimiter=",", names=True)
    data = {k: raw[k].tolist() for k in raw.dtype.names}
    if "g" in data:
        data["g"] = [f"g{int(v)}" for v in data["g"]]
    return data


def _fit(name, formula, **kw):
    data = _data(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return glmm.fit(data, formula, weights=np.asarray(data["w"]), **kw)


_RI = "y ~ x1 + x2 + (1 | g)"
_RS = "y ~ x1 + x2 + (1 + x1 | g)"
_WLS = "y ~ x1 + x2"


def _rel(got, want):
    return np.max(np.abs(np.asarray(got) / np.asarray(want) - 1))


def test_cr2_matches_club_sandwich_on_an_nlme_fit():
    # One cell, for the wiring: clubSandwich on nlme's lme at the
    # reference's power, to 1e-7. The kernel's tests cover every cell.
    g = _golden("cr2_lme_ri_estimated_s10")
    m = _fit("cr2_ri_s10", _RI, weight_power=g["delta"], se_type="robust", reml=True)
    assert m.se_type == "robust"
    assert _rel(m.se, g["se"]) < 1e-7
    assert _rel(m.coef_df, g["df"]) < 1e-7


def test_claimed_weights_with_robust_se_are_hc2():
    # One cell: HC2 and the Bell-McCaffrey df against clubSandwich on
    # lm, to 1e-10.
    g = _golden("hc2_n20")
    m = _fit("hc2_n20", _WLS, weight_power=1, se_type="robust")
    assert (m.se_type, m.df_method, m.weight_power) == ("robust", "bell_mccaffrey", 1.0)
    assert _rel(m.se, g["se"]) < 1e-10
    assert _rel(m.coef_df, g["df"]) < 1e-10
    np.testing.assert_allclose(m.vcov, g["vcov"], rtol=1e-10)


def test_the_default_se_type():
    # One robust and one model row, for the wiring; the kernel's tests hold
    # the whole decision table.
    rows = [
        ("cr2_ri_s10", _RI, {}, "robust", "pustejovsky_tipton"),
        ("cr2_ri_s10", _RI, {"weight_power": "estimate", "reml": False}, "model", "satterthwaite"),
    ]
    for data, formula, kw, se_type, df_method in rows:
        m = _fit(data, formula, **kw)
        assert (m.se_type, m.df_method) == (se_type, df_method), (data, kw)
        assert (m.robust is None) == (se_type == "model")
        assert (m.model_vcov is None) == (se_type == "model")
    m = _fit("hc2_n20", _WLS)
    assert m.weight_power != 1
    assert np.all(np.isfinite(m.coef_p))


def test_a_robust_fit_keeps_its_pieces_through_a_pickle():
    m = _fit("cr2_ri_s10", _RI, weight_power=1, se_type="robust")
    p = len(m.beta)
    assert m.model_vcov.shape == (p, p)
    assert m.robust["er"].shape == (p * 10,)
    again = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(again.robust["er"], m.robust["er"])
    np.testing.assert_array_equal(again.model_vcov, m.model_vcov)


def test_robust_notes_warn_with_their_class_and_text():
    data = _data("cr2_rs_s10")
    w = np.asarray(data["w"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", glmm.RobustDfSmallWarning)
        with pytest.warns(
            glmm.RobustWithRandomSlopeWarning, match="on a model with a random slope"
        ):
            glmm.fit(data, _RS, weights=w, weight_power=1, se_type="robust")
    data = _data("cr2_ri_s10")
    w = np.asarray(data["w"])
    with pytest.warns(
        glmm.RobustDfSmallWarning, match=r"robust degrees of freedom of x2 \(2\.\d+\) are below 4"
    ):
        glmm.fit(data, _RI, weights=w, weight_power=1, se_type="robust")
    data["one"] = [float(v == "g1") for v in data["g"]]
    with pytest.warns(glmm.RobustSeUnidentifiedWarning, match="one: one group alone carries"):
        m = glmm.fit(data, "y ~ x1 + one + (1 | g)", weights=w, weight_power=1, se_type="robust")
    assert np.isnan(m.se[2]) and np.isnan(m.coef_df[2])


def test_contrast_test_and_pairwise_read_the_robust_pieces():
    # A one-row test is the robust t squared on the robust df (on a unit
    # vector the fit's own se and coef_df), with the Satterthwaite pass off too;
    # a comparison is the robust contrast of its L; the same after a pickle. A
    # joint test is the model fit's F, with a warning.
    data = _data("cr2_ri_s10")
    w = np.asarray(data["w"])
    data["f"] = [f"f{i % 3}" for i in range(len(w))]
    formula = "y ~ x1 + x2 + f + (1 | g)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        robust, off, model = (
            glmm.fit(data, formula, weights=w, weight_power=1, se_type=se, satterthwaite=s)
            for se, s in (("robust", True), ("robust", False), ("model", True))
        )
    names = list(robust.names)
    e = np.eye(len(names))
    x1, x2, f1, f2 = (names.index(k) for k in ("x1", "x2", "ff1", "ff2"))
    for m in (robust, off, pickle.loads(pickle.dumps(robust))):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            t = m.contrast_test(e[x1])
        assert t["F value"][0] == pytest.approx((m.beta[x1] / m.se[x1]) ** 2, rel=1e-12)
        assert t["DenDF"][0] == pytest.approx(m.coef_df[x1], rel=1e-12)
        with pytest.warns(glmm.RobustDfSmallWarning, match="robust degrees of freedom are below 4"):
            t = m.contrast_test(e[x2])
        assert m.coef_df[x2] < 4 and t["DenDF"][0] == pytest.approx(m.coef_df[x2], rel=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", glmm.RobustDfSmallWarning)
            pw = glmm.pairwise(m, "f", adjust="none")
            for i, L in enumerate((-e[f1], -e[f2], e[f1] - e[f2])):
                t = m.contrast_test(L)
                assert pw["t.ratio"][i] ** 2 == pytest.approx(t["F value"][0], rel=1e-12)
                assert pw["df"][i] == pytest.approx(t["DenDF"][0], rel=1e-12)
    with pytest.warns(glmm.ModelBasedJointTestWarning, match="model-based standard errors"):
        joint = robust.contrast_test(e[[x1, x2]])
    want = model.contrast_test(e[[x1, x2]])
    assert joint["F value"][0] == pytest.approx(want["F value"][0], rel=1e-12)
    assert joint["DenDF"][0] == pytest.approx(want["DenDF"][0], rel=1e-12)
    # anova() heading: the one-df rows on the robust df, f's row model-based;
    # on WLS every row is one df.
    wls = _fit("hc2_n20", _WLS, weight_power=1, se_type="robust")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        headings = [glmm.anova(m).heading.split("\n")[0] for m in (robust, wls)]
    assert headings == [
        "Type III Analysis of Variance Table with Pustejovsky-Tipton's and Satterthwaite's methods",
        "Type III Analysis of Variance Table with Bell-McCaffrey's method",
    ]


def test_robust_comparisons_and_predictions_warn():
    # A comparison one group alone identifies is NaN, with the rows named; a
    # predict SE on a robust fit is the model fit's, with a warning when asked for.
    data = _data("cr2_ri_s10")
    w = np.asarray(data["w"])
    data["f"] = ["f1" if g == "g1" else f"f{2 * (i % 2)}" for i, g in enumerate(data["g"])]
    formula = "y ~ x1 + f + (1 | g)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = glmm.fit(data, formula, weights=w, weight_power=1, se_type="robust")
        model = glmm.fit(data, formula, weights=w, weight_power=1)
    with pytest.warns(
        glmm.RobustSeUnidentifiedWarning, match=r"^f0 - f1, f1 - f2: One group alone"
    ):
        pw = glmm.pairwise(m, "f", adjust="none")
    assert np.isnan(pw["SE"][[0, 2]]).all() and np.isfinite(pw["SE"][1])
    with pytest.warns(glmm.PredictSeModelBasedWarning, match="model-based, not robust"):
        got = m.predict(se_fit=True)
    np.testing.assert_allclose(got["se_fit"], model.predict(se_fit=True)["se_fit"], rtol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m.predict()


def test_weight_power_search_notes_warn_with_their_class_and_text():
    # Narrow weights whose noise follows w**-40: the profile peaks past the
    # search range [-30, 30].
    rng = np.random.default_rng(4)
    n = 200
    x = rng.uniform(-1, 1, n)
    w = 1 + 0.05 * rng.uniform(0, 1, n)
    data = {"x": x.tolist(), "y": (1 + 0.5 * x + rng.normal(size=n) * w**-40).tolist()}
    with pytest.warns(
        glmm.WeightPowerAtBoundWarning, match=r"The fitted weight power, .* is not a maximum"
    ):
        glmm.fit(data, "y ~ x", weights=w)
    # Weights near 1e100, the largest allowed: no fit in the weight-power search fails.
    n = 60
    x = rng.uniform(-1, 1, n)
    r = 1e4 ** rng.uniform(0, 1, n)
    r[0], r[1] = 1.0, 1e4
    data = {"x": x.tolist(), "y": (1 + 0.5 * x + rng.normal(size=n) * r**-0.5).tolist()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = glmm.fit(data, "y ~ x", weights=1e96 * r)
    assert m.converged and np.all(np.isfinite(m.se))
    assert not [w for w in caught if w.category is glmm.WeightPowerSearchFailuresWarning]


def test_an_estimated_weight_power_far_from_the_weights_warns_by_level():
    # Weights spread 20 exactly, noise sd r**(-power / 2): 5 lands outside [-1, 2]
    # with the variances 20**4 off the weights (Caution), 1.5 inside it (Note).
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(-1, 1, n)
    r = 20 ** rng.uniform(0, 1, n)
    r[0], r[1] = 1.0, 20.0
    e = rng.normal(size=n)
    for power, category in [
        (5.0, glmm.WeightPowerImplausibleWarning),
        (1.5, glmm.WeightPowerOutsideUnitWarning),
    ]:
        data = {"x": x.tolist(), "y": (1 + 0.5 * x + e * r ** (-power / 2)).tolist()}
        with pytest.warns(category) as caught:
            m = glmm.fit(data, "y ~ x", weights=r)
        (entry,) = [w for w in m.warnings if w["kind"].startswith("weight_power")]
        d = m.weight_power
        if category is glmm.WeightPowerImplausibleWarning:
            assert entry["tier"] == "caution"
            assert entry["message"] == (
                f"The fitted weight power, {d:.4g}, is outside [-1, 2], the range the "
                "simulations behind it covered, and the fitted variances differ from the "
                f"weights by up to {20 ** abs(d - 1):.4g} times across the data. Check the "
                "weights, or fix the power with weight_power = 1 (the weights as given)."
            )
        else:
            assert entry["tier"] == "note" and 1 < d <= 2
            assert entry["message"] == (
                f"The fitted weight power, {d:.4g}, is outside [0, 1] (0: the weights carry "
                "no information on the variance; 1: the weights as given). With few rows or "
                "a weak fit (low R²) this happens by chance and is not by itself a problem."
            )
        printed = f"{entry['tier'].capitalize()}: {entry['title']}. {entry['message']}"
        assert printed in [str(w.message) for w in caught if w.category is category]


def test_robust_refusals_are_value_errors():
    data = _data("cr2_ri_s10")
    w = np.asarray(data["w"])
    data["h"] = [f"h{i % 4}" for i in range(len(w))]
    data["count"] = [float(round(abs(v) * 3)) for v in data["y"]]
    instead = 'use se_type = "model"'
    cases = [
        ("y ~ x1 + (1 | g) + (1 | h)", "gaussian", {}),
        ("count ~ x1", "poisson", {}),
        (_RI, "gaussian", {"reml": False}),
        (_WLS, "gaussian", {"weight_power": "estimate", "reml": False}),
    ]
    for formula, family, kw in cases:
        with pytest.raises(ValueError, match=instead):
            glmm.fit(data, formula, family, weights=w, se_type="robust", **kw)


def test_a_fit_pickled_before_the_robust_fields_reads_power_1_and_the_model_se():
    # A 0.8.0 fit has no weight_power, se_type, model_vcov or
    # robust; it summarizes and tests as the model fit it is.
    m = _fit("cr2_ri_s10", _RI, weight_power=1)
    o = object.__new__(glmm.Fit)
    dropped = ("weight_power", "estimated_weight_power", "se_type", "model_vcov", "robust")
    o.__dict__.update({k: v for k, v in m.__dict__.items() if k not in dropped})
    old = pickle.loads(pickle.dumps(o))
    assert (old.weight_power, old.estimated_weight_power, old.se_type) == (1.0, False, "model")
    assert (old.model_vcov, old.robust) == (None, None)
    assert old.summary_object().text() == m.summary_object().text()
    L = np.eye(len(m.beta))[1:]
    assert old.contrast_test(L)["F value"][0] == m.contrast_test(L)["F value"][0]


def test_summary_prints_the_weight_power_and_the_se_footnote():
    # The power when estimated or other than 1; CR2 or HC2 with its df; on a
    # model SE of a fitted power, that the SE treats it as known (and on an
    # ML fit, that the simulations fitted it by REML). Same lines as the R
    # port's print.summary.fastglmm.
    def lines(name, formula, **kw):
        s = _fit(name, formula, **kw).summary_object()
        text = s.text()
        for render in (s.html(), s.latex(), s.typst()):
            for line in s.weight_power_lines:
                assert line in render
        return text

    known = "Standard errors treat the estimated weight power as known"
    cr2 = lines("cr2_ri_s10", _RI)
    assert re.search(r"\nWeight power: -?[\d.]+ \(estimated\)\n", cr2)
    assert "Standard errors: CR2, clustered by g; t tests use Pustejovsky-Tipton" in cr2
    assert known not in cr2
    hc2 = lines("hc2_n20", _WLS)
    assert "Standard errors: HC2; t tests use Bell-McCaffrey degrees of freedom." in hc2
    slope = lines("cr2_rs_s10", _RS)
    assert f"{known}.\n" in slope and "Satterthwaite" in slope
    ml = lines("cr2_ri_s10", _RI, weight_power="estimate", reml=False)
    assert f"{known}; the simulations behind it estimated it by REML." in ml
    assert "\nWeight power: 0.5\n" in lines("cr2_ri_s10", _RI, weight_power=0.5)
    assert "Weight power" not in lines("cr2_ri_s10", _RI, weight_power=1)


# The WLS pair and the random-intercept pair, smaller formula first.
_DELTA_PAIRS = [
    ("hc2_n100", "y ~ x1", "y ~ x1 + x2"),
    ("cr2_ri_s10", "y ~ x1 + (1 | g)", "y ~ x1 + x2 + (1 | g)"),
]


@pytest.mark.parametrize("name,f0,f1", _DELTA_PAIRS)
def test_anova_of_two_ml_delta_fits_counts_each_power(name, f0, f1):
    # The likelihood-ratio test on the fixed columns, each side's
    # fitted power counted in its npar.
    m0, m1 = (_fit(name, f, weight_power="estimate", reml=False) for f in (f0, f1))
    assert m0.estimated_weight_power and m1.estimated_weight_power
    t = glmm.anova(m0, m1)
    re = 1 if "|" in f0 else 0
    assert list(t["npar"]) == [len(m0.beta) + 2 + re, len(m1.beta) + 2 + re]
    assert list(t["npar"]) == [m0.df, m1.df]
    assert t["Df"][1] == 1


@pytest.mark.parametrize("name,f0,f1", _DELTA_PAIRS)
def test_anova_refuses_pairs_whose_weight_powers_differ(name, f0, f1):
    reml = [_fit(name, f, weight_power="estimate") for f in (f0, f1)]
    with pytest.raises(ValueError, match=r'Pass reml=False and weight_power="estimate" to both'):
        glmm.anova(*reml)
    delta = _fit(name, f0, weight_power="estimate", reml=False)
    claimed = _fit(name, f1, weight_power=1, reml=False)
    with pytest.raises(ValueError, match=r"estimated its weight power.*takes weight_power once"):
        glmm.anova(delta, claimed)
    half = _fit(name, f0, weight_power=0.5, reml=False)
    with pytest.raises(
        ValueError, match=r"weight_power = 0\.5 and the second with weight_power = 1"
    ):
        glmm.anova(half, claimed)


def test_anova_refuses_two_delta_fits_differing_in_their_random_part():
    m0, m1 = (
        _fit("cr2_ri_s10", f, weight_power="estimate") for f in ("y ~ x1", _RI.replace(" + x2", ""))
    )
    with pytest.raises(ValueError, match=r"weight_power = 1 and compare again.*15\.5%"):
        glmm.anova(m0, m1)


@pytest.mark.parametrize("name,f0,f1", _DELTA_PAIRS)
def test_compare_fits_the_weight_power_by_ml_on_both_sides(name, f0, f1):
    # None resolves to the fitted power under compare's own criterion.
    data = _data(name)
    w = np.asarray(data["w"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = glmm.compare(data, f0, f1, weights=w)
        claimed = glmm.compare(data, f0, f1, weights=w, weight_power=1)
    for f in (c.fit0, c.fit1):
        assert f.estimated_weight_power and not f.reml and f.se_type == "model"
    assert list(c.table["npar"]) == [c.fit0.df, c.fit1.df]
    assert not (claimed.fit0.estimated_weight_power or claimed.fit1.estimated_weight_power)
    if "|" in f0:
        with pytest.raises(ValueError, match=r"weight_power = 1 and compare again"):
            glmm.compare(data, f0, "y ~ x1", weights=w)


def test_profile_confint_refuses_a_weight_power_other_than_1():
    for power in ("estimate", 0.5):
        m = _fit("cr2_ri_s10", _RI, weight_power=power)
        with pytest.raises(ValueError, match="weight power|weight_power"):
            m.profile_confint()


def test_an_explicit_test_on_a_robust_wls_fit_names_the_residual_df():
    # HC2's Bell-McCaffrey df are not Satterthwaite's: the fit has no random effects.
    m = _fit("hc2_n100", _WLS)
    assert m.df_method == "bell_mccaffrey"
    with pytest.raises(ValueError, match="F tests on the residual df"):
        glmm.anova(m, test="Wald")
