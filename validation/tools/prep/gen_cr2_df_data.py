#!/usr/bin/env python3
"""Input data of the cr2_df references -> validation/data/inference/cr2_df_collinear_e7.csv.

A nearly collinear design, n = 120: a, xi and e standard normal, drawn in that order from
numpy's default_rng(31); xi2 = xi + 1e-7 e, so the design [1, a, xi, xi2] has two columns
1e-7 apart. g = 1..12, ten consecutive rows each (the clusters of the CR2 entry). y = 1 + a +
xi + a fourth standard normal draw, drawn last. The df the references hold do not depend on y.

Every value is written as Python's shortest round-trip repr, so each reader sees these doubles.
Run: python validation/tools/prep/gen_cr2_df_data.py (numpy at the grid/versions.json pin)."""

import csv
import json
import os
import sys

import numpy as np

VALIDATION = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OUT = os.path.join(VALIDATION, "data", "inference", "cr2_df_collinear_e7.csv")

with open(os.path.join(VALIDATION, "grid", "versions.json"), encoding="utf-8") as f:
    pinned = json.load(f)["numpy"]
if np.__version__ != pinned:
    sys.exit(
        f"gen_cr2_df_data.py: numpy is {np.__version__}, grid/versions.json pins {pinned}"
    )

n = 120
r = np.random.default_rng(31)
a, xi, e = r.normal(size=n), r.normal(size=n), r.normal(size=n)
xi2 = xi + 1e-7 * e
g = np.repeat(np.arange(12), 10) + 1
y = 1 + a + xi + r.normal(size=n)

with open(OUT, "w", newline="", encoding="utf-8") as f:
    out = csv.writer(f)
    out.writerow(["y", "a", "xi", "xi2", "g"])
    out.writerows(
        [
            [repr(float(v)) for v in row[:4]] + [int(row[4])]
            for row in zip(y, a, xi, xi2, g)
        ]
    )
print("wrote", os.path.relpath(OUT, VALIDATION))
