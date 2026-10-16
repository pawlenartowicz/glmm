"""Response-domain and dispersion-value validation (GLMM/src/fit/mod.rs, and
GLMM/src/orchestrate.rs for the dispersion value).

These checks live in Rust; `fit()` does no pre-check of its own, so every case
here surfaces as a `ValueError` from `_native.fit`.
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


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_dispersion_not_finite_and_positive_raises(bad):
    with pytest.raises(ValueError, match=r"^dispersion must be a finite number > 0, got "):
        glmm.fit(DATA, "y ~ x", "gamma", dispersion=bad)
