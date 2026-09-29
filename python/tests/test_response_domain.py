"""Response-domain and dispersion-value validation (GLMM/src/fit/mod.rs).

These checks live in the Rust kernel; `fit()` does no pre-check of its own, so
every case here surfaces as a `ValueError` through the same `catch_unwind` path
as a bad `weights=`/`offset=` (see `test_validation.py`'s FFI-level tests).
Mirrored in r/tests/testthat/test-response-domain.R - change together.
"""

import pytest

import glmm

# Small, deliberately out-of-domain fixtures. Never meant to converge — they
# only exercise the kernel's entry-level checks.
DATA = {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 2.0, 3.0]}


def test_poisson_negative_response_raises():
    data = dict(DATA, y=[1.0, -1.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="Poisson response"):
        glmm.fit(data, "y ~ x", "poisson")


def test_negativebinomial_negative_response_raises():
    data = dict(DATA, y=[1.0, -1.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="negative-binomial response"):
        glmm.fit(data, "y ~ x", "negativebinomial")


def test_binomial_response_above_one_raises():
    data = dict(DATA, y=[0.0, 1.0, 2.0, 0.0])
    with pytest.raises(ValueError, match=r"binomial response.*\[0, 1\]"):
        glmm.fit(data, "y ~ x", "binomial")


def test_binomial_response_below_zero_raises():
    data = dict(DATA, y=[0.0, -0.1, 1.0, 0.0])
    with pytest.raises(ValueError, match=r"binomial response.*\[0, 1\]"):
        glmm.fit(data, "y ~ x", "binomial")


def test_gamma_nonpositive_response_raises():
    data = dict(DATA, y=[1.0, 0.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="Gamma response"):
        glmm.fit(data, "y ~ x", "gamma")


def test_inversegaussian_nonpositive_response_raises():
    data = dict(DATA, y=[1.0, -3.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="inverse-Gaussian response"):
        glmm.fit(data, "y ~ x", "inversegaussian")


def test_dispersion_negative_raises():
    with pytest.raises(ValueError, match="dispersion"):
        glmm.fit(DATA, "y ~ x", "gamma", dispersion=-1.0)


def test_dispersion_zero_raises():
    with pytest.raises(ValueError, match="dispersion"):
        glmm.fit(DATA, "y ~ x", "gamma", dispersion=0.0)


def test_poisson_non_integer_response_warns():
    # Intercept-only, mirrors the Rust unit test
    # `poisson_non_integer_response_is_noted` (src/fit/common_tests.rs) exactly.
    with pytest.warns(glmm.NonIntegerResponseWarning, match="not a whole number"):
        result = glmm.fit({"y": [1.0, 2.5, 3.0, 4.0]}, "y ~ 1", "poisson")
    assert result.converged


def test_binomial_non_integer_successes_warns():
    # Intercept-only, mirrors the Rust unit test
    # `binomial_non_integer_successes_is_noted` (src/fit/common_tests.rs) exactly.
    with pytest.warns(glmm.NonIntegerResponseWarning, match="not a whole number"):
        result = glmm.fit(
            {"y": [0.5, 0.5, 0.5, 0.6]},
            "y ~ 1",
            "binomial",
            weights=[2.0, 2.0, 2.0, 3.0],
        )
    assert result.converged
