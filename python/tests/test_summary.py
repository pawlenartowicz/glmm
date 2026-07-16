import math

import numpy as np
import pytest

import glmm


def make_fit(**kw):
    base = dict(
        beta=np.array([2.0, -0.5]),
        se=np.array([1.0, 0.25]),
        vcov=np.array([[1.0, 0.0], [0.0, 0.0625]]),
        tau2=np.array([]),
        varcorr=[],
        stddev_se=np.array([]),
        aliased=np.array([False, False]),
        dispersion=1.0,
        converged=True,
        singular=False,
        names=["(Intercept)", "x"],
        re_groups=[],
        n_eval=0,
        deviance=math.nan,
    )
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


def test_summary_wald_z_p():
    text = make_fit().summary()
    # z = 2.0/1.0 = 2, p = erfc(2/sqrt(2)) = 0.04550 -> "%.4g" = 0.0455
    assert "(Intercept)" in text
    assert "0.0455" in text


def test_summary_aliased_row_is_nan():
    text = make_fit(
        aliased=np.array([False, True]),
        beta=np.array([2.0, float("nan")]),
        se=np.array([1.0, float("nan")]),
    ).summary()
    x_row = next(ln for ln in text.splitlines() if ln.startswith("x"))
    assert "nan" in x_row.lower()


def test_summary_footer():
    text = make_fit(dispersion=2.5, converged=False).summary()
    assert "dispersion: 2.5" in text
    assert "converged: False" in text


def test_summary_re_block():
    # Same D as test_stddev_corr_q2_hand_math: sd=[2, 1.1180], rho=0.4472.
    f = make_fit(
        varcorr=[np.array([4.0, 1.0, 1.25])],
        stddev_se=np.array([float("nan")] * 3),
        re_groups=[("Subject", ["(Intercept)", "Days"])],
    )
    text = f.summary()
    assert "Random effects" in text
    assert "1.118" in text
    assert "0.447" in text
    # The grouping and its terms are named, not "group 0" with bare sd/se rows.
    assert "Subject:" in text
    assert "group 0" not in text
    assert "(Intercept)" in text
    assert "Days" in text


def test_summary_no_re_block_when_empty():
    assert "Random effects" not in make_fit().summary()


def test_summary_prints_what_it_returns(capsys):
    text = make_fit().summary()
    assert capsys.readouterr().out.strip() == text.strip()
