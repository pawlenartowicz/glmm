#!/usr/bin/env python3
"""Input data of the CR2 references -> validation/data/inference/cr2_*.csv and hc2_*.csv.

cr2_<re>_s<spread>.csv: tools/cr2/lib.py's make_data(G = 10, nk = 20, spread, re, seed = 7),
the seed tools/cr2/hp_ref.py draws with, so goldens_cr2_hp.py checks that its data equal these
rows. Columns g (cluster 1..G), x1, x2 (the design is [1, x1, x2]), lam (the working model's
row variance), w = 1/lam (the prior weight a fit takes), u1[, u2] (U = Z L with L L' the
random-effect covariance, so Phi_k = diag(lam) + U U'), y. re is ri (random intercept,
variance 1) or rs (intercept and x1 slope, covariance [[1, .2], [.2, .25]]).

hc2_n<n>.csv: n independent rows for WLS, x1, x2 standard normal, w log-uniform with largest /
smallest = 100 exactly, y = 1 + .5 x1 + .3 x2 + e / sqrt(w), seed = n.

Every value is written as Python's shortest round-trip repr, so each reader sees these doubles.
Run: python validation/tools/prep/gen_cr2_data.py (numpy at the grid/versions.json pin)."""
import csv
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TOOLS, "cr2"))
from lib import make_data  # noqa: E402

OUT = os.path.join(os.path.dirname(TOOLS), "data", "inference")
SPREADS = {"s1": 1.0, "s10": 10.0, "s100": 100.0, "s1e4": 1e4, "s1e6": 1e6}


def write(name, header, rows):
    with open(os.path.join(OUT, name + ".csv"), "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(header)
        out.writerows([[v if isinstance(v, int) else repr(float(v)) for v in r] for r in rows])
    print("wrote", name)


for re_ in ("ri", "rs"):
    for label, spread in SPREADS.items():
        cl = make_data(10, 20, spread, re_, seed=7)
        q = cl[0]["U"].shape[1]
        rows = [[k + 1, c["X"][i, 1], c["X"][i, 2], c["lam"][i], 1.0 / c["lam"][i], *c["U"][i],
                 c["y"][i]] for k, c in enumerate(cl) for i in range(len(c["y"]))]
        write(f"cr2_{re_}_{label}", ["g", "x1", "x2", "lam", "w", *[f"u{j + 1}" for j in range(q)], "y"],
              rows)

for n in (20, 100):
    rng = np.random.default_rng(n)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    lw = rng.uniform(0, np.log(100.0), n)
    lw[0], lw[1] = 0.0, np.log(100.0)
    w = np.exp(lw - lw.mean())
    y = 1 + 0.5 * x1 + 0.3 * x2 + rng.normal(size=n) / np.sqrt(w)
    write(f"hc2_n{n}", ["x1", "x2", "w", "y"], zip(x1, x2, w, y))
