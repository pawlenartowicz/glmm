import warnings

import numpy as _np
import pytest

import glmm

DATA = {"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0], "g": ["a", "b", "a"]}

# A larger, family-appropriate dataset for tests whose call now reaches the
# kernel (unlike DATA above, which is deliberately too small/malformed to fit
# — it only exercises the pure-Python validation that runs before any native
# call, so it must stay unchanged for those tests).
_rng = _np.random.default_rng(0)
_N = 200
_GROUPS = _np.repeat(_np.arange(20), _N // 20)
_X = _rng.normal(size=_N)
FIT_DATA = {
    "x": _X.tolist(),
    "g": [f"g{i}" for i in _GROUPS.tolist()],
    "y_gauss": (1.0 + 2.0 * _X + _rng.normal(scale=0.5, size=_N)).tolist(),
    # Binomial's kernel domain is {0, 1} (see GLMM/src/glm.rs) — a proper 0/1
    # draw, unlike the count-valued y_pois below, which must never be fit as
    # family="binomial".
    "y_bin": _rng.binomial(1, 1.0 / (1.0 + _np.exp(-(0.2 + 0.8 * _X)))).astype(float).tolist(),
    "y_pois": _rng.poisson(_np.exp(0.5 + 0.3 * _X)).astype(float).tolist(),
    "y_gamma": _rng.gamma(shape=2.0, scale=_np.exp(0.5 + 0.1 * _X) / 2.0).tolist(),
}


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown family"):
        glmm.fit(DATA, "y ~ x", "logistic")


def test_link_not_offered_raises():
    with pytest.raises(ValueError, match="does not support link"):
        glmm.fit(DATA, "y ~ x", "poisson", link="identity")


def test_valid_link_reaches_kernel():
    # cloglog is a legal binomial link (0.1.1 addition) — passes Python
    # validation but the kernel doesn't have the variant yet.
    with pytest.raises(NotImplementedError, match="cloglog"):
        glmm.fit(DATA, "y ~ x", "binomial", link="cloglog")


def test_inversegaussian_mixed_raises():
    # GLM-only family: a mixed formula must be a clean Python error,
    # never a kernel panic (spec §3.2).
    with pytest.raises(ValueError, match="GLM-only"):
        glmm.fit(DATA, "y ~ x + (1 | g)", "inversegaussian")


def test_inversegaussian_glm_is_a_kernel_gap():
    with pytest.raises(NotImplementedError, match="inversegaussian"):
        glmm.fit(DATA, "y ~ x", "inversegaussian")


def test_wald_se_invalid_raises():
    with pytest.raises(ValueError, match="wald_se"):
        glmm.fit(DATA, "y ~ x", wald_se="observed")


@pytest.mark.parametrize("nagq", [0, 2, 4, 26, 27, -1, 1.0])
def test_nagq_invalid_raises(nagq):
    with pytest.raises(ValueError, match="nagq"):
        glmm.fit(DATA, "y ~ x + (1 | g)", "binomial", nagq=nagq)


def test_nagq_max_odd_fits():
    result = glmm.fit(FIT_DATA, "y_bin ~ x + (1 | g)", "binomial", nagq=25)
    assert result.converged


# Ineligible-shape nagq>1 is valid-but-inapplicable (spec §3.5): warn and strip
# to nagq=1, never surface the kernel's shape panic as a ValueError. Eligibility
# mirrors src/fit/common.rs::assert_model_shape — single grouping factor,
# binomial/Poisson, q ≤ 3.


def test_nagq_on_gaussian_mixed_warns_and_strips_to_laplace():
    with pytest.warns(UserWarning, match="nagq"):
        result = glmm.fit(FIT_DATA, "y_gauss ~ x + (1 | g)", "gaussian", nagq=3)
    assert result.converged
    # The stripped fit IS the Laplace fit — same answer as an explicit nagq=1 call.
    base = glmm.fit(FIT_DATA, "y_gauss ~ x + (1 | g)", "gaussian")
    assert _np.allclose(result.beta, base.beta)


def test_nagq_on_crossed_re_warns_and_strips():
    data = {**FIT_DATA, "h": [f"h{i % 5}" for i in range(_N)]}
    with pytest.warns(UserWarning, match="nagq"):
        result = glmm.fit(data, "y_bin ~ x + (1 | g) + (1 | h)", "binomial", nagq=3)
    assert result.converged


def test_nagq_over_q_cap_warns_and_strips():
    # q_p = 4 (intercept + 3 slopes) exceeds the temporary q ≤ 3 AGQ cap.
    data = {
        **FIT_DATA,
        "x1": _rng.normal(size=_N).tolist(),
        "x2": _rng.normal(size=_N).tolist(),
        "x3": _rng.normal(size=_N).tolist(),
    }
    with pytest.warns(UserWarning, match="nagq"):
        result = glmm.fit(
            data,
            "y_bin ~ x + x1 + x2 + x3 + (1 + x1 + x2 + x3 | g)",
            "binomial",
            nagq=3,
        )
    assert result.converged


def test_nagq_on_fixed_only_warns_and_strips():
    with pytest.warns(UserWarning, match="nagq"):
        result = glmm.fit(FIT_DATA, "y_bin ~ x", "binomial", nagq=3)
    assert result.converged


def test_dispersion_on_gaussian_warns_and_strips_then_fits():
    with pytest.warns(UserWarning, match="dispersion"):
        result = glmm.fit(FIT_DATA, "y_gauss ~ x", "gaussian", dispersion="estimate")
    assert result.converged


def test_dispersion_on_negativebinomial_warns_then_fits():
    # negbin's distribution param is theta, not phi (spec §3.2 table).
    with pytest.warns(UserWarning, match="dispersion"):
        result = glmm.fit(FIT_DATA, "y_pois ~ x", "negativebinomial", dispersion=1.5)
    assert result.converged


def test_quasi_on_mixed_binomial_warns_then_fits():
    # "estimate" on binomial/poisson is quasi-likelihood, GLM only (spec
    # §3.2) — on a MIXED formula it is stripped (warn), not a kernel gap.
    with pytest.warns(UserWarning, match="GLM-only"):
        result = glmm.fit(FIT_DATA, "y_bin ~ x + (1 | g)", "binomial", dispersion="estimate")
    assert result.converged


def test_quasi_on_glm_poisson_is_a_kernel_gap():
    # Non-mixed: dispersion reaches the family check un-stripped, and
    # quasi-Poisson has no kernel implementation yet (0.1.1).
    with pytest.raises(NotImplementedError, match="quasi-likelihood"):
        glmm.fit(FIT_DATA, "y_pois ~ x", "poisson", dispersion="estimate")


def test_dispersion_bad_value_raises():
    with pytest.raises(ValueError, match="dispersion"):
        glmm.fit(DATA, "y ~ x", "gamma", dispersion="pearson")


def test_dispersion_bool_raises():
    # bool is an int subclass — must not pass as a numeric dispersion.
    with pytest.raises(ValueError, match="dispersion"):
        glmm.fit(DATA, "y ~ x", "gamma", dispersion=True)


def test_theta_off_negbin_warns_then_fits():
    with pytest.warns(UserWarning, match="theta"):
        result = glmm.fit(FIT_DATA, "y_gamma ~ x", "gamma", theta=1.5)
    assert result.converged


def test_theta_on_negbin_is_a_kernel_gap():
    # theta=<float> has no kernel hook (no public theta_seed on fit_warm);
    # only the default theta=None cold-start search is supported.
    with pytest.raises(NotImplementedError, match="theta"):
        glmm.fit(FIT_DATA, "y_pois ~ x", "negativebinomial", theta=1.5)


def test_warm_start_unknown_key_warns_then_fits():
    with pytest.warns(UserWarning, match="warm_start"):
        result = glmm.fit(
            FIT_DATA,
            "y_gauss ~ x",
            warm_start={"beta": [0.0, 0.0], "phi": 1.0},
        )
    assert result.converged


def test_warm_start_not_dict_raises():
    with pytest.raises(TypeError, match="warm_start"):
        glmm.fit(DATA, "y ~ x", warm_start=[0.0, 0.0])


def test_clean_call_emits_no_warnings_and_fits():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        result = glmm.fit(
            FIT_DATA,
            "y_pois ~ x + (1 | g)",
            "poisson",
            warm_start={"beta": [0.0, 0.0], "theta": [1.0]},
        )
    assert result.converged
