#!/usr/bin/env python3
"""REML criterion of sim_entangled_pair_lmm's y ~ 1 + t + v + z + (1 | g), in
60-digit arithmetic, evaluated directly on the CSV's doubles.

t and v are entangled to 3e-6 (gen_illcond_data.R in this directory), so lme4,
glmmTMB and glmm each converge to a slightly different theta (the random
intercept's sd over the residual sd). lme4's and glmmTMB's own reported REML
criteria carry round-off of 1e-4 to 7e-4 against the true optimum -- more
than validation/grid/tol.R's dev_eps (4e-05), even though the three theta
agree with each other to 1e-10 on the true objective. No oracle deviance can
gate this cell, so validation/grid/dev_ref.json freezes the smallest of the
three 60-digit values here instead of an oracle's f64 report.

    python3 validation/tools/prep/entangled_reml_reference.py

Needs mpmath (not a project dependency -- a local-only tool, like the R
scripts elsewhere in this directory need R packages that are not pinned
anywhere either). Read-only: nothing here is written back to the CSV or to
dev_ref.json.
"""
import csv
import os

import mpmath as mp

mp.mp.dps = 60

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "..", "..", "data", "simulated", "sim_entangled_pair_lmm.csv")

# Each engine's fitted theta on this cell, from the 2026-09-26 accuracy-grid
# comparison (validation/grid/runs/{glmm,glmmtmb,lme4}/...): sd(g intercept)
# over the residual sd, read off each engine's own reported variance
# components.
THETA = {
    "glmm": 4.211901007053366 / 0.280306689231264,
    "glmmTMB": 4.211896179324582 / 0.28030660782521066,
    "lme4": 4.211895307851695 / 0.2803066654730708,
}


def reml_criterion(rows, theta):
    """-2 * restricted log-likelihood + REML constant, for one scalar random
    intercept y = X*beta + Z*u + e, u ~ N(0, theta^2*sigma^2*I), e ~ N(0,
    sigma^2*I). Closed form per group via the Sherman-Morrison update of
    V^-1 = sigma^-2 * (I - theta^2/(1 + theta^2*n_j) * 1_j*1_j'), so nothing
    here ever forms or inverts the n x n V: log|V/sigma^2| is
    sum_j log(1 + theta^2*n_j), and X'V^-1X, X'V^-1y, y'V^-1y follow from
    each group's column sums the same way.
    """
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["g"], []).append(i)
    n, p = len(rows), 4
    # float() first, THEN mpf: float() is correctly rounded (IEEE 754
    # round-to-nearest), so this promotes exactly the double every engine
    # reads. Parsing the CSV string straight into an mpf would instead keep
    # more decimal digits than the doubles actually carry, and the whole
    # point of this file is to reproduce the round-off every engine has, not
    # a more precise dataset than the one they fitted.
    y = [mp.mpf(float(r["y"])) for r in rows]
    x = [[mp.mpf(1), mp.mpf(float(r["t"])), mp.mpf(float(r["v"])), mp.mpf(float(r["z"]))]
         for r in rows]

    th2 = mp.mpf(theta) ** 2
    xtx = mp.matrix(p, p)
    xty = mp.matrix(p, 1)
    yty = mp.mpf(0)
    logdet_v = mp.mpf(0)
    for idx in groups.values():
        shrink = th2 / (1 + th2 * len(idx))
        logdet_v += mp.log(1 + th2 * len(idx))
        sx = [sum(x[i][c] for i in idx) for c in range(p)]
        sy = sum(y[i] for i in idx)
        for a in range(p):
            xty[a] += sum(x[i][a] * y[i] for i in idx) - shrink * sx[a] * sy
            for c in range(p):
                xtx[a, c] += sum(x[i][a] * x[i][c] for i in idx) - shrink * sx[a] * sx[c]
        yty += sum(y[i] ** 2 for i in idx) - shrink * sy * sy
    beta = mp.lu_solve(xtx, xty)
    rss = yty - sum(xty[a] * beta[a] for a in range(p))
    return (logdet_v + mp.log(mp.det(xtx)) +
            (n - p) * (1 + mp.log(2 * mp.pi * rss / (n - p))))


def main():
    # float() on a CSV field is correctly rounded (IEEE 754 round-to-nearest),
    # so this reads exactly the doubles every engine reads from the same file.
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    results = {name: reml_criterion(rows, theta) for name, theta in THETA.items()}
    for name, theta in THETA.items():
        print(f"{name} theta {theta!r} REML crit {mp.nstr(results[name], 15)}")
    frozen_name = min(results, key=results.get)
    print(f"frozen (smallest, {frozen_name}): {mp.nstr(results[frozen_name], 15)}")


if __name__ == "__main__":
    main()
