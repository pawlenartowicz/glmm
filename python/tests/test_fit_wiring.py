import numpy as np
import pytest

import glmm

_rng = np.random.default_rng(1)
_N = 300
_GROUPS = np.repeat(np.arange(30), _N // 30)
_X = _rng.normal(size=_N)


def _data(y):
    return {"x": _X.tolist(), "g": [f"g{i}" for i in _GROUPS.tolist()], "y": y}


def test_gaussian_lmm():
    y = 1.0 + 2.0 * _X + _rng.normal(scale=0.5, size=_N)
    result = glmm.fit(_data(y.tolist()), "y ~ x + (1 | g)")
    assert result.converged
    assert result.names == ["(Intercept)", "x"]
    assert abs(result.beta[1] - 2.0) < 0.3


def test_binomial_glmm():
    p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * _X)))
    y = _rng.binomial(1, p).astype(float)
    result = glmm.fit(_data(y.tolist()), "y ~ x + (1 | g)", "binomial")
    assert result.converged


def test_poisson_glm():
    y = _rng.poisson(np.exp(0.5 + 0.3 * _X)).astype(float)
    result = glmm.fit(_data(y.tolist()), "y ~ x", "poisson")
    assert result.converged
    assert result.dispersion == pytest.approx(1.0)


def test_gamma_glm_fixed_dispersion():
    y = _rng.gamma(shape=2.0, scale=np.exp(0.5 + 0.1 * _X) / 2.0)
    result = glmm.fit(_data(y.tolist()), "y ~ x", "gamma", dispersion=1.0)
    assert result.converged
    assert result.dispersion == pytest.approx(1.0)


def test_negativebinomial_glm_cold_start():
    y = _rng.negative_binomial(5, 0.5, size=_N).astype(float)
    result = glmm.fit(_data(y.tolist()), "y ~ x", "negativebinomial")
    assert result.converged
    assert result.dispersion > 0  # theta estimate


def test_agq_vector_q2_smoke():
    # q=2 (random intercept + slope) binomial with small clusters, where the
    # Laplace bias AGQ corrects is visible — proves nagq>1 actually routed to
    # the vector-AGQ kernel instead of silently refitting Laplace.
    rng = np.random.default_rng(42)
    n_groups, per = 75, 4
    n = n_groups * per
    g = np.repeat(np.arange(n_groups), per)
    x = rng.normal(size=n)
    b0 = rng.normal(scale=1.2, size=n_groups)
    b1 = rng.normal(scale=0.8, size=n_groups)
    eta = 0.3 + 0.8 * x + b0[g] + b1[g] * x
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    data = {"y": y.tolist(), "x": x.tolist(), "g": [f"g{i}" for i in g.tolist()]}

    laplace = glmm.fit(data, "y ~ x + (1 + x | g)", "binomial")  # nagq=1 default
    agq = glmm.fit(data, "y ~ x + (1 + x | g)", "binomial", nagq=7, wald_se="hessian")
    assert agq.converged
    assert np.all(np.isfinite(agq.se))
    # Gate routed: quadrature moves the answer off the Laplace fit (empirically
    # ~3e-2 in beta and ~0.3 in varcorr on this dataset — far above tolerance).
    assert np.max(np.abs(agq.beta - laplace.beta)) > 1e-6
    assert np.max(np.abs(np.array(agq.varcorr[0]) - np.array(laplace.varcorr[0]))) > 1e-4


def test_warm_start_reaches_same_answer_as_cold():
    y = 1.0 + 2.0 * _X + _rng.normal(scale=0.5, size=_N)
    cold = glmm.fit(_data(y.tolist()), "y ~ x + (1 | g)")
    warm = glmm.fit(
        _data(y.tolist()),
        "y ~ x + (1 | g)",
        warm_start={"beta": cold.beta.tolist(), "theta": [1.0]},
    )
    assert warm.converged
    assert np.allclose(warm.beta, cold.beta, atol=1e-6)
