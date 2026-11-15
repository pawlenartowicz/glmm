#!/usr/bin/env python3
"""50-digit lnGamma at the negative-binomial shapes near the lower search
limit, for src/simd_transcendental.rs::ln_gamma_is_accurate_at_tiny_shapes and
src/fit/glm_tests.rs::nb_saturated_loglik_is_accurate_at_any_count, which keep
the printed values inline. Nothing is written to disk.

Prints mpmath loggamma(a) at a = 1e-6, 1e-5, 1e-4, and the NB saturated
log-likelihood of one count,
lnG(y+t) - lnG(t) - lnG(y+1) + t ln(t/(t+y)) + y ln(y/(t+y)), at
(y, t) = (1, 1e-6) and (1e10, 1e-6). Inputs are the f64 values (mpf of a
Python float is exact); each result is rounded to the nearest double.

Needs mpmath at the version of validation/grid/versions.json's `mpmath`; any
other version stops the run. Run from the crate root:

    python3 validation/tools/prep/ln_gamma_tiny_reference.py
"""

import json
import os
import sys

import mpmath as mp
from mpmath.libmp.libmpf import round_nearest, to_float

SUITE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(SUITE_DIR, "grid", "versions.json"), encoding="utf-8") as f:
    pinned = json.load(f)["mpmath"]
if mp.__version__ != pinned:
    sys.exit(f"mpmath is {mp.__version__}, grid/versions.json pins {pinned}")

mp.mp.dps = 50


def to_f64(v):
    return to_float(v._mpf_, rnd=round_nearest)


for a in [1e-6, 1e-5, 1e-4]:
    print(f"lgamma({a!r}) = {to_f64(mp.loggamma(mp.mpf(a)))!r}")
for y, t in [(1.0, 1e-6), (1e10, 1e-6)]:
    ym, tm = mp.mpf(y), mp.mpf(t)
    sat = (
        mp.loggamma(ym + tm)
        - mp.loggamma(tm)
        - mp.loggamma(ym + 1)
        + tm * mp.log(tm / (tm + ym))
        + ym * mp.log(ym / (tm + ym))
    )
    print(f"saturated(y={y!r}, theta={t!r}) = {to_f64(sat)!r}")
