#!/usr/bin/env python3
"""sim_poisson_precise_e30: the precise design of gen_boundary_pin_data.R at
counts about 1e13, for the in-crate test of a gradient search whose final
BOBYQA polish ends with no result (src/fit/core_tests.rs).

Poisson log, y ~ 1 + x1 + x2 + (1 + x1 + x2 | g), 20 levels g000..g019 x 10
rows, x1, x2 ~ N(0, 1); intercept 30, x1 0.1, x2 -0.1; intercept SD 0.5, both
slope SDs 0. At these means the count is drawn by its normal approximation,
round(mu + sqrt(mu) * N(0, 1)). Dataset 5 of a 40-dataset search at log-mean
30 (seed 1000 * 30 + 5), chosen on 2026-11-03 because, on x86_64 Linux, its
likelihood-ratio refit without x2 in the fixed part was the one fit of 160
(log-means 18, 22, 26, 30) whose polish ended within a non-finite point's
radius.

Python's random.Random is seeded by the integer alone and float repr is
exact, so a re-run writes the same bytes on any platform:

    python3 validation/tools/prep/gen_precise_e30_data.py [<output-dir>]

The output directory defaults to validation/data/simulated.
"""
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "data", "simulated")
os.makedirs(out_dir, exist_ok=True)

r = random.Random(30005)
rows = ["y,x1,x2,g"]
for g in range(20):
    b = r.gauss(0, 0.5)
    for _ in range(10):
        x1, x2 = r.gauss(0, 1), r.gauss(0, 1)
        mu = math.exp(30 + 0.1 * x1 - 0.1 * x2 + b)
        y = max(0, round(mu + math.sqrt(mu) * r.gauss(0, 1)))
        rows.append(f"{y},{x1!r},{x2!r},g{g:03d}")
path = os.path.join(out_dir, "sim_poisson_precise_e30.csv")
with open(path, "w") as f:
    f.write("\n".join(rows) + "\n")
print(f"wrote {path}: {len(rows) - 1} rows")
