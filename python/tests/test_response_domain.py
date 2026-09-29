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


def test_dispersion_zero_raises():
    with pytest.raises(ValueError, match="dispersion"):
        glmm.fit(DATA, "y ~ x", "gamma", dispersion=0.0)
