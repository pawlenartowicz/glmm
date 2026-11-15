#!/usr/bin/env python3
"""sim_nb_tinytheta: a negative-binomial GLM whose shape maximum lies far below
1e-3, for src/fit/glm_tests.rs::nb_glm_tiny_shape_matches_the_base_r_profile.

400 rows y,x,g: numpy.random.default_rng(8), x ~ N(0, 1), g = 20 groups 0..19
of 20 rows. Then five gamma-Poisson draws, in this order at shape theta =
0.01, 0.003, 0.001, 0.0005, 0.0002: lam ~ Gamma(theta, exp(3 + 0.3 x) / theta),
y ~ Poisson(lam). Only the last draw (theta = 2e-4) is written; the four before
it only advance the generator. It has one positive count in 400 rows. The
base-R profile maximum of y ~ x on it (nb_tiny_shape_profile_reference.R in
this directory) is at theta about 5e-4.

Needs numpy at the version of validation/grid/versions.json's `numpy`; any
other version stops the run. Run from the crate root:

    python3 validation/tools/prep/gen_nb_tinytheta_data.py [<output-dir>]

The output directory defaults to validation/data/simulated.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(os.path.dirname(HERE))

with open(os.path.join(SUITE_DIR, "grid", "versions.json"), encoding="utf-8") as f:
    pinned = json.load(f)["numpy"]
if np.__version__ != pinned:
    sys.exit(f"numpy is {np.__version__}, grid/versions.json pins {pinned}")

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SUITE_DIR, "data", "simulated")
os.makedirs(out_dir, exist_ok=True)

rng = np.random.default_rng(8)
n, n_groups = 400, 20
g = np.repeat(np.arange(n_groups), n // n_groups)
x = rng.normal(size=n)
for theta in [0.01, 0.003, 0.001, 0.0005, 0.0002]:
    lam = rng.gamma(theta, np.exp(3 + 0.3 * x) / theta)
    y = rng.poisson(lam).astype(float)
np.savetxt(
    os.path.join(out_dir, "sim_nb_tinytheta.csv"),
    np.column_stack([y, x, g]),
    delimiter=",",
    header="y,x,g",
    comments="",
    fmt="%.17g",
)
