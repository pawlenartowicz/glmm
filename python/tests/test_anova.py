"""glmm.anova and glmm.compare: the per-term table of one fit, the comparison
of two, and the renders of both tables."""

import dataclasses
import inspect
import math
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pytest

import glmm

_rng = np.random.default_rng(31)
_N, _K = 120, 12
_G = np.repeat(np.arange(_K), _N // _K)
_X = _rng.normal(size=_N)
_Z = _rng.normal(size=_N)
_F = np.array(["a", "b", "c"])[np.arange(_N) % 3]
_U = _rng.normal(scale=0.8, size=_K)
_Y = 1.0 + 0.6 * _X + 0.4 * (_F == "b") + _U[_G] + _rng.normal(size=_N)
_COUNTS = _rng.poisson(np.exp(0.2 + 0.3 * _X + 0.5 * _U[_G])).astype(float)


def _data(**columns):
    d = {
        "y": _Y.tolist(),
        "x": _X.tolist(),
        "z": _Z.tolist(),
        "f": _F.tolist(),
        "g": [f"g{i}" for i in _G],
        "n": _COUNTS.tolist(),
    }
    d.update(columns)
    return d


D = _data()

# Dyestuff2 (lme4): the batch variance is estimated at exactly zero.
DYESTUFF2 = {
    "Batch": [b for b in "ABCDEF" for _ in range(5)],
    "Yield": [
        7.298, 3.846, 2.434, 9.566, 7.99, 5.22, 6.556, 0.608, 11.788, -0.892,
        0.11, 10.386, 13.434, 5.51, 8.166, 2.212, 4.852, 7.092, 9.288, 4.98,
        0.282, 9.014, 4.458, 9.446, 7.198, 1.722, 4.782, 8.106, 0.758, 3.758,
    ],
}  # fmt: skip


def quiet_fit(formula, data=D, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return glmm.fit(data, formula, **kw)


def quiet_compare(f0, f1, data=D, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return glmm.compare(data, f0, f1, **kw)


@pytest.fixture(scope="module")
def lmm():
    return quiet_fit("y ~ x + f + (1 | g)")


@pytest.fixture(scope="module")
def exact_pair():
    """An OLS fit made with reml=True and the REML fit that adds `(1 | g)`:
    the exact restricted likelihood-ratio test's pair."""
    return quiet_fit("y ~ x", reml=True), quiet_fit("y ~ x + (1 | g)")


# -- the per-term table -------------------------------------------------------


def test_anova_of_one_fit_reads_the_stored_type3_table(lmm):
    tab = glmm.anova(lmm)
    assert isinstance(tab, glmm.TermTable)
    stored = lmm.term_table_type3
    assert tab.row_names == stored["label"] == ["x", "f"]
    assert tab.columns == [
        "Sum Sq", "Mean Sq", "NumDF", "DenDF", "F value", "Pr(>F)",
    ]  # fmt: skip
    np.testing.assert_array_equal(tab["F value"], stored["statistic"])
    np.testing.assert_array_equal(tab["Pr(>F)"], stored["p_value"])
    np.testing.assert_array_equal(tab["DenDF"], stored["den_df"])
    assert tab.heading == "Type III Analysis of Variance Table with Satterthwaite's method"
    # A one-df term that no other term contains: its F is the square of the
    # summary's t on the same df.
    i = lmm.names.index("x")
    assert tab["F value"][0] == pytest.approx((lmm.beta[i] / lmm.se[i]) ** 2, rel=1e-8)
    assert tab["DenDF"][0] == pytest.approx(lmm.coef_df[i], rel=1e-8)


def test_anova_type_ii_reads_the_stored_type2_table(lmm):
    tab = glmm.anova(lmm, type="II")
    assert tab.heading.startswith("Type II Analysis of Variance Table")
    np.testing.assert_array_equal(tab["F value"], lmm.term_table_type2["statistic"])
    with pytest.raises(ValueError, match="type must be 'III' or 'II'"):
        glmm.anova(lmm, type="I")


def test_a_table_iterates_and_tests_membership_over_its_column_names(lmm):
    tab = glmm.anova(lmm)
    assert list(tab) == tab.columns
    assert "F value" in tab
    assert "AIC" not in tab


def test_anova_of_a_glmm_is_a_wald_chisq_table():
    tab = glmm.anova(quiet_fit("n ~ x + f + (1 | g)", family="poisson"), test="Wald")
    assert tab.heading == "Type III Analysis of Deviance Table (Wald χ² tests)"
    assert tab.columns == ["Chisq", "Df", "Pr(>Chisq)"]
    assert tab["Df"].tolist() == [1.0, 2.0]
    # The renders add each row's reference, built from its Df.
    html = tab.html()
    assert "<td>χ²(1)</td>" in html and "<td>χ²(2)</td>" in html


def test_anova_raises_the_stored_marginal_term_refusal():
    m = quiet_fit("y ~ x + x:f")
    with pytest.raises(ValueError) as err:
        glmm.anova(m)
    assert str(err.value) == m.term_table_type3["error"]
    assert "f" in str(err.value)


def test_an_aliased_term_names_its_coefficient():
    # x2 copies x, so its column is aliased and its row is not tested; the
    # note names the coefficient as the fit does, never a column index.
    tab = glmm.anova(quiet_fit("y ~ f + x + x2", _data(x2=_X.tolist())))
    assert "x2: The coefficient x2 is aliased, so the hypothesis was not tested." in tab.notes


def test_df_prints_whole_at_any_size_and_satterthwaite_keeps_its_decimals(exact_pair):
    rng = np.random.default_rng(7)
    n = 30000
    x = rng.normal(size=n)
    big = {
        "y": (x + rng.normal(size=n)).tolist(),
        "x": x.tolist(),
        "f": np.array(["a", "b", "c"])[np.arange(n) % 3].tolist(),
    }
    small, large = quiet_fit("y ~ x", big), quiet_fit("y ~ x + f", big)
    terms = glmm.anova(large)
    html = terms.html()
    assert "<td>F(1, 29996)</td>" in html and "<td>F(2, 29996)</td>" in html
    assert terms.text().splitlines()[2].split()[4] == "29996"  # x's DenDF cell
    tab = glmm.anova(small, large)
    assert "Reference: F(2, 29996)." in tab.text()
    # The comparison's df cells (Res.Df, Df, npar) follow the same rule; the
    # m1 row is the fifth line after the heading.
    assert tab.text().splitlines()[5].split()[:2] == ["m1", "29996"]  # Res.Df
    wide = dataclasses.replace(tab, _values={**tab._values, "Df": np.array([_nan, 12000.0])})
    assert wide.text().splitlines()[5].split()[3] == "12000"
    lr = glmm.anova(*exact_pair)
    lr = dataclasses.replace(lr, _values={**lr._values, "npar": np.array([12000.0, 12001.0])})
    assert lr.text().splitlines()[5].split()[:2] == ["m1", "12001"]
    # A Satterthwaite DenDF keeps four significant digits, in fixed notation.
    m = quiet_fit("Yield ~ 1 + (1 | Batch)", DYESTUFF2)
    stored = {**_STORED, "den_df": np.array([107.39, 12345.6, _nan])}
    sat = glmm.anova(dataclasses.replace(m, term_table_type3=stored))
    html = sat.html()
    assert "<td>F(1, 107.4)</td>" in html and "<td>F(2, 12346)</td>" in html


def test_a_df_that_is_not_finite_prints_as_r_prints_it():
    from glmm.summary import fmt_df

    assert [fmt_df(v) for v in (math.inf, -math.inf, math.nan)] == ["Inf", "-Inf", "NA"]


def test_the_heading_claims_satterthwaite_only_when_a_row_has_its_df():
    m = quiet_fit("y ~ x + f + (1 | g)", satterthwaite=False)
    assert glmm.anova(m).heading == "Type III Analysis of Variance Table"
    # An intercept-only model has no rows: the fit's df method decides.
    m = quiet_fit("y ~ 1 + (1 | g)", satterthwaite=False)
    assert glmm.anova(m).heading == "Type III Analysis of Variance Table"


def test_anova_of_a_fit_that_did_not_converge_says_so(lmm):
    stalled = dataclasses.replace(lmm, diagnostics={**lmm.diagnostics, "converged": False})
    assert glmm.anova(stalled).notes[0] == (
        "The fit did not converge, so its estimates are not at a maximum and these tests "
        "are not reliable. Refit until the fit converges, then test again."
    )
    assert not any("converge" in n for n in glmm.anova(lmm).notes)


def test_a_fit_from_before_these_tests_is_refused_by_name(exact_pair):
    # A fit pickled by glmm 0.5.0 has none of the fields added since.
    new = ("rss", "df_method", "offset", "objective", "objective_nodes", "term_table_type3")
    old = []
    for m in exact_pair:
        state = {k: v for k, v in m.__dict__.items() if k not in new}
        o = object.__new__(glmm.Fit)
        o.__dict__.update(state)
        old.append(pickle.loads(pickle.dumps(o)))
    with pytest.raises(ValueError, match=r"term_table_type3.*Refit"):
        glmm.anova(old[1])
    with pytest.raises(ValueError, match=r"rss, objective.*Refit"):
        glmm.anova(*old)


def test_anova_raises_a_stored_internal_error_as_an_error_not_a_refusal(lmm):
    broken = dataclasses.replace(
        lmm, term_table_type3={"error": "internal error while building the per-term table: x"}
    )
    with pytest.raises(RuntimeError, match="internal error"):
        glmm.anova(broken)


# -- the test argument ----------------------------------------------------------

_POIS = "n ~ x + f + (1 | g)"
_LRT_LINE = "Likelihood-ratio chi-squared tests (each term's model refitted without it)"
_AUTO_REASON = 'test = "auto": Poisson model with random effects'


@pytest.fixture(scope="module")
def pois():
    """A Poisson GLMM fitted with the default `anova_test`, and the same model
    fitted with `anova_test="LRT"`."""
    return quiet_fit(_POIS, family="poisson"), quiet_fit(_POIS, family="poisson", anova_test="LRT")


def _f_refusal(den_df, name):
    return (
        f"per-term tests on this fit are F tests with {den_df} denominator df; {name} rows "
        "are not available, so leave the test at its default"
    )


def test_anova_signature_takes_test_after_type():
    p = inspect.signature(glmm.anova).parameters
    assert list(p) == ["fits", "type", "test", "nsim", "seed"]
    assert p["test"].default == "auto"
    assert p["test"].kind is inspect.Parameter.KEYWORD_ONLY


def test_wald_gives_the_stored_wald_table_whatever_the_fit_stored(pois):
    plain, asked = pois
    wald = quiet_fit(_POIS, family="poisson", anova_test="Wald")
    assert wald.term_table_type3_lrt is None and wald.term_table_type2_lrt is None
    for ss, field in (("III", "term_table_type3"), ("II", "term_table_type2")):
        base = glmm.anova(wald, type=ss, test="Wald")
        assert base.heading == f"Type {ss} Analysis of Deviance Table (Wald χ² tests)"
        for m in (plain, wald, asked):
            tab = glmm.anova(m, type=ss, test="Wald")
            assert tab.text() == base.text() and tab.html() == base.html()
            # Storing the likelihood-ratio tables moves no Wald number.
            np.testing.assert_array_equal(tab["Chisq"], getattr(wald, field)["statistic"])
            np.testing.assert_array_equal(tab["Pr(>Chisq)"], getattr(wald, field)["p_value"])
    np.testing.assert_array_equal(asked.beta, plain.beta)
    np.testing.assert_array_equal(wald.beta, plain.beta)


def test_auto_gives_a_poisson_glmm_the_likelihood_ratio_table(pois):
    plain, asked = pois
    assert plain.term_test_auto == {"test": "lrt", "reason": "poisson_mixed"}
    for ss, design in (("III", " on the reduced design"), ("II", "")):
        lrt_field = "term_table_type3_lrt" if ss == "III" else "term_table_type2_lrt"
        stored = getattr(plain, lrt_field)
        assert (stored["test"], stored["reason"]) == ("lrt", "poisson_mixed")
        heading = [f"Type {ss} Analysis of Deviance Table", f"{_LRT_LINE}{design}; {_AUTO_REASON}"]
        for m in (plain, asked):
            tab = glmm.anova(m, type=ss)
            assert glmm.anova(m, type=ss, test="auto").text() == tab.text()
            assert tab.heading.split("\n") == heading
            assert tab.text().splitlines()[:2] == heading
            # The rows of test="LRT", on the fit made with either value.
            lrt = glmm.anova(m, type=ss, test="LRT")
            assert tab.columns == lrt.columns and tab.row_names == lrt.row_names
            for c in tab.columns:
                np.testing.assert_array_equal(tab[c], lrt[c])
            np.testing.assert_array_equal(tab["Chisq"], stored["statistic"])
            assert lrt.heading.split("\n")[1] == f'{_LRT_LINE}{design}; test = "LRT"'
            assert tab.notes == lrt.notes
            # Both heading lines in every render.
            assert tab.latex().splitlines()[:2] == [f"% {h}" for h in heading]
            assert tab.typst().splitlines()[:2] == [f"// {h}" for h in heading]
            assert lrt.latex().splitlines()[1] == f'% {_LRT_LINE}{design}; test = "LRT"'
            assert lrt.typst().splitlines()[1] == f'// {_LRT_LINE}{design}; test = "LRT"'


def test_auto_on_a_poisson_glmm_fitted_with_wald_says_so():
    m = quiet_fit(_POIS, family="poisson", anova_test="Wald")
    assert m.term_test_auto == {"test": "lrt", "reason": "poisson_mixed"}
    message = (
        "test='auto' gives likelihood-ratio tests on a Poisson model with random effects, "
        "and this fit stored none: it was made with anova_test='Wald'. Use test='Wald', or "
        "refit with glmm.fit(..., anova_test='auto')."
    )
    for ss in ("III", "II"):
        with pytest.raises(ValueError) as err:
            glmm.anova(m, type=ss)
        assert str(err.value) == message


def test_auto_names_the_wald_test_and_its_reason_on_every_other_chi_squared_fit():
    d = _data(
        b=(_Y > np.median(_Y)).astype(float).tolist(),
        pos=np.exp(0.3 * _X + 0.3 * _Z).tolist(),
    )
    other = "model with random effects, not binomial, Poisson or negative binomial"
    cases = (
        (quiet_fit("b ~ x + f + (1 | g)", d, family="binomial"), "binomial_mixed",
         "binomial model with random effects"),
        (quiet_fit(_POIS, family="negativebinomial"), "negative_binomial_mixed",
         "negative binomial model with random effects"),
        (quiet_fit("pos ~ x + f + (1 | g)", d, family="gamma"), "other_mixed", other),
        (quiet_fit("n ~ x + f", family="poisson"), "no_random_effects",
         "model without random effects"),
    )  # fmt: skip
    for m, tag, reason in cases:
        assert m.term_test_auto == {"test": "chisq", "reason": tag}
        assert m.term_table_type3_lrt is None and m.term_table_type2_lrt is None
        for ss in ("III", "II"):
            heading = [
                f"Type {ss} Analysis of Deviance Table",
                f'Wald chi-squared tests; test = "auto": {reason}',
            ]
            tab = glmm.anova(m, type=ss)
            assert glmm.anova(m, type=ss, test="auto").text() == tab.text()
            assert tab.heading.split("\n") == heading
            assert tab.text().splitlines()[:2] == heading
            assert tab.latex().splitlines()[:2] == [f"% {h}" for h in heading]
            assert tab.typst().splitlines()[:2] == [f"// {h}" for h in heading]
            # Explicit Wald keeps today's heading; the numbers are the same.
            wald = glmm.anova(m, type=ss, test="Wald")
            assert wald.heading == f"Type {ss} Analysis of Deviance Table (Wald χ² tests)"
            assert (tab.columns, tab.row_names, tab.notes) == (
                wald.columns, wald.row_names, wald.notes,
            )  # fmt: skip
            for c in tab.columns:
                np.testing.assert_array_equal(tab[c], wald[c])


def test_a_fit_without_the_auto_choice_still_takes_an_explicit_test(pois):
    # A fit made before the auto choice was stored refuses only "auto".
    old = dataclasses.replace(pois[1], term_test_auto=None)
    assert glmm.anova(old, test="Wald").text() == glmm.anova(pois[1], test="Wald").text()
    assert glmm.anova(old, test="LRT").text() == glmm.anova(pois[1], test="LRT").text()
    with pytest.raises(ValueError, match="this fit has no term_test_auto"):
        glmm.anova(old)


def test_lrt_reads_the_stored_likelihood_ratio_tables(pois):
    _, m = pois
    for ss, design in (("III", " on the reduced design"), ("II", "")):
        stored = getattr(m, "term_table_type3_lrt" if ss == "III" else "term_table_type2_lrt")
        assert (stored["test"], stored["reason"], stored["kind"]) == (
            "lrt", "requested", ["lrt", "lrt"],
        )  # fmt: skip
        tab = glmm.anova(m, type=ss, test="LRT")
        assert tab.row_names == ["x", "f"]
        assert tab.columns == ["Chisq", "Df", "Pr(>Chisq)"]
        np.testing.assert_array_equal(tab["Chisq"], stored["statistic"])
        np.testing.assert_array_equal(tab["Df"], [1.0, 2.0])
        np.testing.assert_array_equal(tab["Pr(>Chisq)"], stored["p_value"])
        heading = [f"Type {ss} Analysis of Deviance Table", f'{_LRT_LINE}{design}; test = "LRT"']
        assert tab.heading.split("\n") == heading
        assert tab.text().splitlines()[:2] == heading
        assert tab.text().splitlines()[2].split() == ["Chisq", "Df", "Pr(>Chisq)"]
        html = tab.html()
        assert all(
            f"<p>{line}</p>" in html.replace("&#x27;", "'").replace("&quot;", '"')
            for line in heading
        )
        assert "<td>χ²(1)</td>" in html and "<td>χ²(2)</td>" in html
    again = pickle.loads(pickle.dumps(m))
    assert glmm.anova(again, test="LRT").text() == glmm.anova(m, test="LRT").text()


def test_lrt_on_a_fit_that_stored_none_names_the_fit_argument():
    message = (
        "this fit stored no likelihood-ratio tables, which are built only when the fit "
        "asks for them: refit with glmm.fit(..., anova_test='LRT')."
    )
    for m in (
        quiet_fit(_POIS, family="poisson", anova_test="Wald"),
        quiet_fit("n ~ x + f", family="poisson"),
    ):
        for ss in ("III", "II"):
            with pytest.raises(ValueError) as err:
                glmm.anova(m, type=ss, test="LRT")
            assert str(err.value) == message


def test_a_stored_refusal_comes_before_the_choice_of_table():
    # As the R port: a formula without its marginal term gets its refusal,
    # never advice to refit with other tables, whatever test is asked.
    m = quiet_fit("n ~ x + x:f + (1 | g)", family="poisson", anova_test="Wald")
    refusal = m.term_table_type3["error"]
    assert "x:f" in refusal
    for test in ("auto", "Wald", "LRT"):
        with pytest.raises(ValueError) as err:
            glmm.anova(m, test=test)
        assert str(err.value) == refusal


def test_wald_or_lrt_on_an_f_table_is_refused_naming_its_test(lmm):
    for test, name in (("Wald", "Wald chi-squared"), ("LRT", "likelihood-ratio")):
        for ss in ("III", "II"):
            with pytest.raises(ValueError) as err:
                glmm.anova(lmm, type=ss, test=test)
            assert str(err.value) == _f_refusal("Satterthwaite's", name)
    ols = quiet_fit("y ~ x + f")
    with pytest.raises(ValueError) as err:
        glmm.anova(ols, test="Wald")
    assert str(err.value) == _f_refusal("the residual", "Wald chi-squared")
    # A fit that asked for the likelihood-ratio tables stands, and stores the
    # same refusal in their place.
    asked = quiet_fit("y ~ x + f + (1 | g)", anova_test="LRT")
    assert asked.term_table_type3_lrt == {
        "error": _f_refusal("Satterthwaite's", "likelihood-ratio")
    }
    with pytest.raises(ValueError) as err:
        glmm.anova(asked, test="LRT")
    assert str(err.value) == _f_refusal("Satterthwaite's", "likelihood-ratio")
    assert glmm.anova(asked).text() == glmm.anova(lmm).text()


def test_a_term_whose_column_carries_a_random_slope_keeps_its_wald_row():
    m = quiet_fit("n ~ x + f + (1 + x | g)", family="poisson", anova_test="LRT")
    stored = m.term_table_type3_lrt
    assert stored["kind"] == ["chisq", "lrt"]
    (note,) = stored["row_notes"][0]
    assert "random slope" in note
    tab = glmm.anova(m, test="LRT")
    assert f"x: {note}" in tab.notes
    assert tab["Chisq"][0] == m.term_table_type3["statistic"][0]


def test_test_and_anova_test_are_checked(exact_pair):
    m0, m1 = exact_pair
    for t in ("auto", "Wald", "LRT"):
        with pytest.raises(TypeError, match=r"two fits takes no test"):
            glmm.anova(m0, m1, test=t)
    with pytest.raises(ValueError, match=r"test must be 'auto', 'Wald' or 'LRT', got 'lrt'"):
        glmm.anova(m1, test="lrt")
    with pytest.raises(ValueError, match=r"anova_test must be 'auto', 'Wald' or 'LRT', got 'lrt'"):
        glmm.fit(D, "y ~ x", anova_test="lrt")


# -- the comparison of two fits -------------------------------------------------


def test_anova_of_two_ml_fits_is_the_comparison_table_smaller_first():
    small, large = (
        quiet_fit("y ~ x + (1 | g)", reml=False),
        quiet_fit("y ~ x + f + (1 | g)", reml=False),
    )
    tab = glmm.anova(large, small)
    assert isinstance(tab, glmm.ComparisonTable)
    # The rows are named after the fits as given, the smaller model first.
    assert tab.row_names == ["m1", "m0"]
    # The heading lists the two models in the rows' order.
    assert tab.text().splitlines()[1:3] == ["m1: y ~ x + (1 | g)", "m0: y ~ x + f + (1 | g)"]
    assert tab.columns == ["npar", "AIC", "BIC", "logLik", "Chisq", "Df", "Pr(>Chisq)"]
    assert tab["npar"].tolist() == [small.df, large.df]
    assert tab["AIC"].tolist() == [small.aic, large.aic]
    assert tab["logLik"].tolist() == [small.loglik, large.loglik]
    assert math.isnan(tab["Chisq"][0]) and math.isnan(tab["Pr(>Chisq)"][0])
    assert tab["Chisq"][1] == pytest.approx(2 * (large.loglik - small.loglik), rel=1e-12)
    assert tab["Df"][1] == 2
    assert 0 < tab["Pr(>Chisq)"][1] < 1
    assert tab.heading == "Model comparison: likelihood ratio test (ML)"
    assert "Reference: χ²(2)." in tab.text()
    # The same test with the two fits swapped.
    swapped = glmm.anova(small, large)
    assert swapped.row_names == ["m0", "m1"]
    assert swapped["Pr(>Chisq)"][1] == tab["Pr(>Chisq)"][1]
    assert (tab.nsim, tab.seed) == (10000, 20261008)


def test_a_reml_comparison_table_has_no_aic_or_bic_column():
    tab = glmm.anova(quiet_fit("y ~ x + (1 | g)"), quiet_fit("y ~ x + (1 + x | g)"))
    assert tab.columns == ["npar", "logLik", "Chisq", "Df", "Pr(>Chisq)"]
    assert tab.heading == "Model comparison: likelihood ratio test on the REML criterion"
    assert "50:50 mixture of χ²(1) and χ²(2)" in tab.text()


def test_two_ols_fits_get_anova_lm_columns():
    small, large = quiet_fit("y ~ x"), quiet_fit("y ~ x + f", reml=True)
    tab = glmm.anova(small, large)
    assert tab.columns == ["Res.Df", "RSS", "Df", "Sum of Sq", "F", "Pr(>F)"]
    assert tab["Res.Df"].tolist() == [_N - 2, _N - 4]
    assert tab["RSS"].tolist() == [small.rss, large.rss]
    assert tab["Sum of Sq"][1] == pytest.approx(small.rss - large.rss, rel=1e-12)
    # A comparison that adds one term gives that term's F on the larger fit.
    f_row = glmm.anova(large)["F value"][1]
    assert tab["F"][1] == pytest.approx(f_row, rel=1e-8)


def test_two_gamma_glms_get_the_wald_f_columns_of_car_linear_hypothesis():
    gamma = _data(pos=np.exp(0.3 * _X + 0.3 * _Z).tolist())
    small = quiet_fit("pos ~ x", gamma, family="gamma")
    large = quiet_fit("pos ~ x + f", gamma, family="gamma")
    tab = glmm.anova(small, large)
    assert tab.columns == ["Res.Df", "Df", "F", "Pr(>F)"]
    assert tab["Res.Df"].tolist() == [_N - 2, _N - 4]
    assert tab["Df"][1] == 2 and 0 < tab["Pr(>F)"][1] < 1
    assert tab.heading == "Model comparison: F test"
    assert tab.reference == f"F(2, {_N - 4})"


def test_an_agq_comparison_names_its_node_count():
    tab = glmm.anova(
        quiet_fit("n ~ x + (1 | g)", family="poisson", nagq=5),
        quiet_fit("n ~ x + f + (1 | g)", family="poisson", nagq=5),
    )
    assert tab.heading == "Model comparison: likelihood ratio test (AGQ 5)"


def test_a_dropped_variance_at_zero_gives_p_one():
    # The larger fit's batch variance lands at exactly 0 while its
    # log-likelihood differs from the smaller fit's by rounding only: the test
    # reads p = 1 from the zero-component rule, which matches each fit's random
    # terms by name.
    rng = np.random.default_rng(3)
    x = rng.normal(size=60)
    d = {
        "y": (1 + 0.5 * x + rng.normal(size=60)).tolist(),
        "x": x.tolist(),
        "g": [f"g{i}" for i in np.repeat(np.arange(6), 10)],
    }
    large = quiet_fit("y ~ x + (1 | g)", d, reml=False)
    assert large.varcorr == [[0.0]]
    tab = glmm.anova(quiet_fit("y ~ x", d, reml=False), large)
    assert tab["Chisq"][1] == 0.0 and tab["Pr(>Chisq)"][1] == 1.0


def test_compare_returns_the_table_and_the_two_fits_as_given():
    res = quiet_compare("y ~ x + f + (1 | g)", "y ~ x + (1 | g)")
    assert isinstance(res, glmm.Comparison)
    assert isinstance(res.table, glmm.ComparisonTable)
    assert (res.fit0.formula, res.fit1.formula) == ("y ~ x + f + (1 | g)", "y ~ x + (1 | g)")
    assert not res.fit0.reml and not res.fit1.reml  # the fixed parts differ: ML
    assert res.table.row_names == ["f1", "f0"]
    assert "Both models were fitted by ML because their fixed parts differ." in res.table.notes
    assert repr(res) == res.table.text()
    # The same pair through glmm.anova on the fits compare made.
    again = glmm.anova(res.fit0, res.fit1)
    assert again["Pr(>Chisq)"][1] == res.table["Pr(>Chisq)"][1]


def test_compare_and_anova_give_the_same_exact_rlrt_p_on_every_run(exact_pair):
    ols, lmm1 = exact_pair
    a = glmm.anova(ols, lmm1)
    c = quiet_compare("y ~ x", "y ~ x + (1 | g)").table
    assert 0 < a["Pr(>Chisq)"][1] < 1
    assert a["Pr(>Chisq)"][1] == c["Pr(>Chisq)"][1]
    assert glmm.anova(ols, lmm1)["Pr(>Chisq)"][1] == a["Pr(>Chisq)"][1]
    assert math.isnan(a["Df"][1])  # the exact reference is not a chi-squared
    assert "exact RLRT null distribution (Crainiceanu & Ruppert 2004), 10000 draws." in a.text()
    other = glmm.anova(ols, lmm1, nsim=200, seed=5)
    assert (other.nsim, other.seed) == (200, 5)
    assert "200 draws" in other.text()


def test_an_exact_rlrt_row_at_zero_needs_no_draws():
    ols = quiet_fit("Yield ~ 1", DYESTUFF2, reml=True)
    full = quiet_fit("Yield ~ 1 + (1 | Batch)", DYESTUFF2)
    assert full.varcorr == [[0.0]]
    tab = glmm.anova(ols, full)
    assert tab["Chisq"][1] == 0.0 and tab["Pr(>Chisq)"][1] == 1.0
    text = tab.text()
    assert "no draws needed" in text
    assert "draws." not in text and "10000" not in text


def test_anova_on_unpickled_fits_gives_the_same_tables(lmm, exact_pair):
    back = pickle.loads(pickle.dumps(lmm))
    assert glmm.anova(back).text() == glmm.anova(lmm).text()
    ols, lmm1 = (pickle.loads(pickle.dumps(m)) for m in exact_pair)
    assert glmm.anova(ols, lmm1).text() == glmm.anova(*exact_pair).text()


def test_the_fit_stores_its_offset_and_no_data_column():
    off = 0.1 * _X
    m = quiet_fit("y ~ x + (1 | g)", offset=off)
    np.testing.assert_array_equal(m.offset, off)
    assert (m.objective, m.objective_nodes) == ("reml", 0)
    assert quiet_fit("y ~ x").offset is None
    for name in ("x", "z"):
        column = np.asarray(D[name])
        for f in dataclasses.fields(m):
            value = getattr(m, f.name)
            assert not (
                isinstance(value, np.ndarray)
                and value.shape == column.shape
                and np.array_equal(value, column)
            ), f.name


def test_compare_fits_both_models_on_the_rows_both_formulas_can_use():
    z = _Z.copy()
    z[3] = np.nan
    with pytest.warns(glmm.RowsDroppedWarning) as caught:
        res = glmm.compare(_data(z=z.tolist()), "y ~ x + (1 | g)", "y ~ x + z + (1 | g)")
    assert res.fit0.nobs == res.fit1.nobs == _N - 1
    assert [w["kind"] for w in res.fit0.warnings] == ["rows_dropped_na"]
    # The warning points at the caller's line, from compare and from fit.
    assert caught[0].filename == __file__
    with pytest.warns(glmm.RowsDroppedWarning) as caught:
        glmm.fit(_data(z=z.tolist()), "y ~ z + (1 | g)")
    assert caught[0].filename == __file__
    # A missing weight or offset drops its row from both fits too.
    w, off = np.ones(_N), np.zeros(_N)
    w[5], off[7] = np.nan, np.nan
    with pytest.warns(glmm.RowsDroppedWarning):
        res = glmm.compare(D, "y ~ x + (1 | g)", "y ~ x + z + (1 | g)", weights=w, offset=off)
    assert res.fit0.nobs == res.fit1.nobs == _N - 2


# -- refusals -------------------------------------------------------------------

_POINTER = "glmm.compare(data, f0, f1)"


def _refusal(m0, m1):
    with pytest.raises(ValueError) as err:
        glmm.anova(m0, m1)
    message = str(err.value)
    assert "REML = FALSE" not in message and "compare(data" not in message.replace(_POINTER, "")
    return message


def test_each_refusal_of_two_fits_raises_value_error_with_its_fix():
    small = quiet_fit("y ~ x")
    y2 = _Y + 1.0
    cases = {
        "rows": (small, quiet_fit("y ~ x + z", _data(z=[math.nan] + _Z[1:].tolist()))),
        "y": (small, quiet_fit("y ~ x + z", _data(y=y2.tolist()))),
        "weights": (small, quiet_fit("y ~ x + z", weights=np.full(_N, 2.0))),
        "offset": (small, quiet_fit("y ~ x + z", offset=np.full(_N, 0.5))),
        "family": (
            quiet_fit("n ~ x", family="poisson"),
            quiet_fit("n ~ x + z", family="negativebinomial"),
        ),
        "estimators": (quiet_fit("y ~ x + (1 | g)"), quiet_fit("y ~ x + z + (1 | g)", reml=False)),
        "reml_fixed": (quiet_fit("y ~ x + (1 | g)"), quiet_fit("y ~ x + z + (1 | g)")),
    }
    for name, (m0, m1) in cases.items():
        message = _refusal(m0, m1)
        assert _POINTER in message, name
    reml_fixed = _refusal(*cases["reml_fixed"])
    assert "reml=False" in reml_fixed and reml_fixed.count("Refit both") == 1
    assert _refusal(*cases["estimators"]).startswith("m0 was fitted by REML and m1 by ML")


def test_refusals_with_no_pointer():
    pos = np.exp(0.1 * _X)
    gamma = {"pos": pos.tolist()}
    held = quiet_fit("pos ~ x", _data(**gamma), family="gamma", dispersion=2.0)
    free = quiet_fit("pos ~ x + z", _data(**gamma), family="gamma")
    assert "dispersion" in _refusal(held, free)
    assert "neither formula is nested" in _refusal(quiet_fit("y ~ x"), quiet_fit("y ~ z"))
    w = quiet_fit("w ~ x + z", _data(w=_Y.tolist()))
    assert "different responses" in _refusal(quiet_fit("y ~ x"), w)
    same = "m0 and m1 have the same fixed and random parts, so there is nothing to test."
    assert same in _refusal(quiet_fit("y ~ x"), quiet_fit("y ~ x", reml=True))
    one = quiet_fit("y ~ x")
    assert _refusal(one, one) == f"{same} Compare two models that differ by at least one parameter."
    m = quiet_fit("y ~ x + z")
    stalled = dataclasses.replace(m, diagnostics={**m.diagnostics, "converged": False})
    assert "did not converge" in _refusal(quiet_fit("y ~ x"), stalled)


def test_compare_refuses_formulas_it_cannot_compare():
    with pytest.raises(ValueError, match="neither formula is nested"):
        quiet_compare("y ~ x", "y ~ z")
    with pytest.raises(ValueError, match="nothing"):
        quiet_compare("y ~ x", "y ~ x")


def test_fits_passed_to_compare_raise_type_error(exact_pair):
    m0, m1 = exact_pair
    with pytest.raises(TypeError, match=r"glmm\.anova\(m0, m1\)"):
        glmm.compare(m0, m1)
    with pytest.raises(TypeError, match=r"glmm\.anova\(m0, m1\)"):
        glmm.compare(D, m0, m1)


def test_compare_refuses_a_third_formula():
    # As R's compare() refuses one through `...`, not as an unknown family.
    with pytest.raises(TypeError, match=r"^glmm\.compare takes two formulas, not three or more$"):
        glmm.compare(D, "y ~ x", "y ~ x + z", "y ~ x + z + f")


def test_compare_refuses_an_infinite_weight_or_offset_in_rs_words():
    # The R port's compare() gives the same text; a missing entry drops its row.
    for name, fill in (("weights", 1.0), ("offset", 0.0)):
        v = np.full(_N, fill)
        v[4] = np.inf
        with pytest.raises(ValueError, match=f"^{name} must be finite or missing: no inf entries$"):
            glmm.compare(D, "y ~ x", "y ~ x + z", **{name: v})


def test_anova_takes_one_or_two_fits(exact_pair):
    m0, m1 = exact_pair
    with pytest.raises(TypeError, match="one fit or two"):
        glmm.anova(m0, m1, m1)
    with pytest.raises(TypeError, match="one fit or two"):
        glmm.anova()
    with pytest.raises(TypeError, match="glmm.Fit"):
        glmm.anova(m0, "y ~ x")


def test_arguments_that_do_not_apply_raise_type_error(exact_pair):
    m0, m1 = exact_pair
    for kw in ({"nsim": 0}, {"nsim": 100}, {"seed": 1}):
        with pytest.raises(TypeError, match=r"one fit takes no nsim or seed"):
            glmm.anova(m1, **kw)
    for t in ("I", "II", "III"):
        with pytest.raises(TypeError, match=r"two fits takes no type"):
            glmm.anova(m0, m1, type=t)


def test_nsim_and_seed_are_checked(exact_pair):
    with pytest.raises(ValueError, match="from 1 to 1000000"):
        glmm.anova(*exact_pair, nsim=0)
    with pytest.raises(ValueError, match="from 1 to 1000000"):
        glmm.anova(*exact_pair, nsim=1_000_001)
    with pytest.raises(TypeError, match="nsim"):
        glmm.anova(*exact_pair, nsim=True)
    with pytest.raises(ValueError, match="seed"):
        glmm.anova(*exact_pair, seed=-1)


# -- pandas ---------------------------------------------------------------------


def test_to_pandas_gives_the_same_columns(lmm, exact_pair):
    pd = pytest.importorskip("pandas")
    tab = glmm.anova(*exact_pair)
    frame = tab.to_pandas()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == tab.columns
    assert list(frame.index) == tab.row_names
    np.testing.assert_array_equal(frame["Pr(>Chisq)"].to_numpy(), tab["Pr(>Chisq)"])
    assert list(glmm.anova(lmm).to_pandas().columns) == glmm.anova(lmm).columns


def test_everything_but_to_pandas_works_without_pandas():
    script = """
import sys, warnings
sys.modules["pandas"] = None  # import pandas now raises ImportError
import glmm
warnings.simplefilter("ignore")
d = {"y": [1.0, 2.5, 2.9, 4.2, 5.1, 6.8, 6.9, 8.4], "x": [1, 2, 3, 4, 5, 6, 7, 8],
     "z": [0.3, -1.0, 0.8, 0.1, -0.4, 1.2, -0.9, 0.5]}
m0, m1 = glmm.fit(d, "y ~ x"), glmm.fit(d, "y ~ x + z")
glmm.anova(m1).text()
glmm.anova(m0, m1).html()
tab = glmm.compare(d, "y ~ x", "y ~ x + z").table
try:
    tab.to_pandas()
except ImportError:
    print("ok")
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_the_new_names_are_public():
    for name in ("anova", "compare", "TermTable", "ComparisonTable", "Comparison"):
        assert name in glmm.__all__ and hasattr(glmm, name)


# -- renders --------------------------------------------------------------------

_nan = math.nan
# A stored Type III result with made-up numbers, so the snapshot does not move
# with the optimizer: an aliased term's NaN row, a row note and a table note.
_STORED = {
    "error": None,
    "ss_type": "III",
    "label": ["x", "f", "x:f"],
    "kind": ["F", "F", "F"],
    "statistic": np.array([31.37, 3.514, _nan]),
    "num_df": np.array([1.0, 2.0, _nan]),
    "den_df": np.array([107.4, 103.8, _nan]),
    "den_df_kind": ["satterthwaite"] * 3,
    "p_value": np.array([1.66e-07, 0.0334, _nan]),
    "sum_sq": np.array([30.78, 6.896, _nan]),
    "mean_sq": np.array([30.78, 3.448, _nan]),
    "row_notes": [[], [], ["The coefficient x:fc is aliased, so the hypothesis was not tested."]],
    "notes": ["Missing cells for: x:f. Interpret type III hypotheses with care."],
}

_TERM_TEXT = """\
Type III Analysis of Variance Table with Satterthwaite's method
    Sum Sq Mean Sq NumDF DenDF F value   Pr(>F)
x    30.78   30.78     1 107.4   31.37 1.66e-07 ***
f    6.896   3.448     2 103.8   3.514   0.0334 *
x:f     NA      NA    NA    NA      NA       NA
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
Missing cells for: x:f. Interpret type III hypotheses with care.
x:f: The coefficient x:fc is aliased, so the hypothesis was not tested."""

_TERM_HTML = """\
<div class="glmm-summary">
<p>Type III Analysis of Variance Table with Satterthwaite&#x27;s method</p>
<table><caption>Terms</caption>
<tr><th></th><th>Sum Sq</th><th>Mean Sq</th><th>NumDF</th><th>DenDF</th><th>F value</th><th>Pr(&gt;F)</th><th></th><th>Reference</th></tr>
<tr><td>x</td><td>30.78</td><td>30.78</td><td>1</td><td>107.4</td><td>31.37</td><td>1.66e-07</td><td>***</td><td>F(1, 107.4)</td></tr>
<tr><td>f</td><td>6.896</td><td>3.448</td><td>2</td><td>103.8</td><td>3.514</td><td>0.0334</td><td>*</td><td>F(2, 103.8)</td></tr>
<tr><td>x:f</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td><td></td><td></td></tr>
</table>
<p>Signif. codes:  0 &#x27;***&#x27; 0.001 &#x27;**&#x27; 0.01 &#x27;*&#x27; 0.05 &#x27;.&#x27; 0.1 &#x27; &#x27; 1</p>
<p>Missing cells for: x:f. Interpret type III hypotheses with care.</p>
<p>x:f: The coefficient x:fc is aliased, so the hypothesis was not tested.</p>
</div>"""

_TERM_LATEX = r"""% Type III Analysis of Variance Table with Satterthwaite's method
% Terms
\begin{tabular}{lrrrrrrrr}
\toprule
 & Sum Sq & Mean Sq & NumDF & DenDF & F value & Pr(\textgreater{}F) &  & Reference \\
\midrule
x & 30.78 & 30.78 & 1 & 107.4 & 31.37 & 1.66e-07 & *** & F(1, 107.4) \\
f & 6.896 & 3.448 & 2 & 103.8 & 3.514 & 0.0334 & * & F(2, 103.8) \\
x:f & NA & NA & NA & NA & NA & NA &  &  \\
\bottomrule
\end{tabular}

% Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
% Missing cells for: x:f. Interpret type III hypotheses with care.
% x:f: The coefficient x:fc is aliased, so the hypothesis was not tested."""

_TERM_TYPST = """\
// Type III Analysis of Variance Table with Satterthwaite's method
// Terms
#table(columns: 9, align: (left,right,right,right,right,right,right,right,right),
  "", strong("Sum Sq"), strong("Mean Sq"), strong("NumDF"), strong("DenDF"), strong("F value"), strong("Pr(>F)"), "", strong("Reference"),
  "x", "30.78", "30.78", "1", "107.4", "31.37", "1.66e-07", "***", "F(1, 107.4)",
  "f", "6.896", "3.448", "2", "103.8", "3.514", "0.0334", "*", "F(2, 103.8)",
  "x:f", "NA", "NA", "NA", "NA", "NA", "NA", "", "",
)

// Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
// Missing cells for: x:f. Interpret type III hypotheses with care.
// x:f: The coefficient x:fc is aliased, so the hypothesis was not tested."""

_EXACT = (
    "exact RLRT null distribution (Crainiceanu & Ruppert 2004); no draws needed: "
    "the statistic is 0, so p = 1"
)

_COMPARISON_TEXT = f"""\
Model comparison: likelihood ratio test on the REML criterion
m0: Yield ~ 1
m1: Yield ~ 1 + (1 | Batch)
   npar  logLik Chisq Df Pr(>Chisq)
m0    2 -80.914
m1    3 -80.914     0             1
  Reference: {_EXACT}."""

_COMPARISON_HTML = f"""\
<div class="glmm-summary">
<p>Model comparison: likelihood ratio test on the REML criterion</p>
<p>m0: Yield ~ 1</p>
<p>m1: Yield ~ 1 + (1 | Batch)</p>
<table><caption>Models</caption>
<tr><th></th><th>npar</th><th>logLik</th><th>Chisq</th><th>Df</th><th>Pr(&gt;Chisq)</th><th></th><th>Reference</th></tr>
<tr><td>m0</td><td>2</td><td>-80.914</td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td>m1</td><td>3</td><td>-80.914</td><td>0</td><td></td><td>1</td><td></td><td>{_EXACT.replace("&", "&amp;")}</td></tr>
</table>
</div>"""

_EXACT_TEX = _EXACT.replace("&", r"\&")
_COMPARISON_LATEX = rf"""% Model comparison: likelihood ratio test on the REML criterion
% m0: Yield ~ 1
% m1: Yield ~ 1 + (1 | Batch)
% Models
\begin{{tabular}}{{lrrrrrrr}}
\toprule
 & npar & logLik & Chisq & Df & Pr(\textgreater{{}}Chisq) &  & Reference \\
\midrule
m0 & 2 & -80.914 &  &  &  &  &  \\
m1 & 3 & -80.914 & 0 &  & 1 &  & {_EXACT_TEX} \\
\bottomrule
\end{{tabular}}
"""

_COMPARISON_TYPST = f"""\
// Model comparison: likelihood ratio test on the REML criterion
// m0: Yield ~ 1
// m1: Yield ~ 1 + (1 | Batch)
// Models
#table(columns: 8, align: (left,right,right,right,right,right,right,right),
  "", strong("npar"), strong("logLik"), strong("Chisq"), strong("Df"), strong("Pr(>Chisq)"), "", strong("Reference"),
  "m0", "2", "-80.914", "", "", "", "", "",
  "m1", "3", "-80.914", "0", "", "1", "", "{_EXACT}",
)
"""


def _snapshot(table, text, html, latex, typst):
    assert table.text() == text
    assert repr(table) == text
    style, body = table.html().split("\n", 1)
    assert style.startswith("<style>") and style.endswith("</style>")
    assert body == html
    assert table.latex() == latex
    assert table.typst() == typst


def test_term_table_renders():
    m = quiet_fit("Yield ~ 1 + (1 | Batch)", DYESTUFF2)
    tab = glmm.anova(dataclasses.replace(m, term_table_type3=_STORED))
    _snapshot(tab, _TERM_TEXT, _TERM_HTML, _TERM_LATEX, _TERM_TYPST)


def test_comparison_table_renders():
    ols = quiet_fit("Yield ~ 1", DYESTUFF2, reml=True)
    tab = glmm.anova(ols, quiet_fit("Yield ~ 1 + (1 | Batch)", DYESTUFF2))
    _snapshot(tab, _COMPARISON_TEXT, _COMPARISON_HTML, _COMPARISON_LATEX, _COMPARISON_TYPST)


def test_test_none_is_the_default_as_type_none_is(pois, exact_pair):
    m, _ = pois
    assert glmm.anova(m, test=None).text() == glmm.anova(m).text()
    m0, m1 = exact_pair
    assert glmm.anova(m0, m1, test=None).text() == glmm.anova(m0, m1).text()
