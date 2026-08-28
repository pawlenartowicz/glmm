import math
import re

import numpy as np
import pytest

import glmm


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
    # (mirrors the Rust test stddev_corr_q2_hand_math in GLMM/src/fit.rs).
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
    np.testing.assert_allclose(re["se"], [0.5, 0.3])
    assert re["corr"][1, 0] == pytest.approx(1.0 / (2.0 * math.sqrt(1.25)))
    # Scaled residuals: (y - fitted) / sqrt(dispersion), quantiles type 7.
    r = (np.array([1.0, 2.0, 3.0, 4.0]) - np.array([1.5, 1.5, 3.5, 3.5])) / math.sqrt(654.94)
    np.testing.assert_allclose(s.scaled_residuals, np.quantile(r, [0, 0.25, 0.5, 0.75, 1]))


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
        "Pr(>|z|)",
        "Correlation of Fixed Effects:",
        "Optimizer evaluations: 12, converged: True",
        "Wald z",
    ]
    pos = [text.index(s) for s in order]
    assert pos == sorted(pos), text


def test_summary_text_wald_z_p_and_stars():
    text = make_fit().summary()
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
    assert "Dispersion" not in make_fit(family="poisson", link="log").summary()


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
