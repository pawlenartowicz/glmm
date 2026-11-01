"""Dispersion-value validation (src/orchestrate.rs).

The check lives in Rust; `fit()` does no pre-check of its own, so every case
here surfaces as a `ValueError` from `_native.fit`.
Mirrored in r/tests/testthat/test-errors.R - change together.
"""

import pytest

import glmm

# A small fixture that never reaches the kernel: the dispersion check refuses
# the call before the formula is lowered.
DATA = {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 2.0, 3.0]}


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_dispersion_not_finite_and_positive_raises(bad):
    with pytest.raises(ValueError, match=r"^dispersion must be a finite number > 0, got "):
        glmm.fit(DATA, "y ~ x", "gamma", dispersion=bad)
