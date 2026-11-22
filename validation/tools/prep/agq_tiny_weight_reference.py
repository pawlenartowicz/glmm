#!/usr/bin/env python3
"""160-digit nAGQ 5 deviance of a binomial logit (1 | g) GLMM whose 40 prior
weights are all 1e-100, for
src/glmm/tests.rs::agq_deviance_keeps_its_digits_at_tiny_weights, which keeps
the printed values inline. Nothing is written to disk.

The data are the test's own: x_i = sin(1.7 i + 0.3), y_i = 1 when
0.2 + 0.8 x_i + U[g_i] + 0.8 sin(2.3 i + 0.7) > 0, eight groups of five rows.
At each (beta, lambda) the deviance is

    -2 sum_c [ l_c(u_c) - 1/2 ln A_c + ln( sum_j w_j e^{z_j^2 + l_c(u_cj) - l_c(u_c)} / sum_j w_j ) ],

    l_c(u) = -u^2/2 + sum_{i in c} w_i [y_i eta_i - ln(1 + e^{eta_i})],  eta_i = x_i'beta + lambda u,

with u_c the exact mode of l_c, A_c = 1 - l_c''(u_c), u_cj = u_c + sqrt(2) z_j / sqrt(A_c), and
(z_j, w_j) the f64 5-node Gauss-Hermite table of src/consts.rs taken as exact.
The bracket is normalised by the table's own weight sum, not sqrt(pi): the
crate drops the constant ln(sum_j w_j) - ln(sqrt(pi)), which is the same at every
(beta, lambda). Inputs are the f64 values (mpf of a Python float is exact); each
result is rounded to the nearest double.

Needs mpmath at the version of validation/grid/versions.json's `mpmath`; any
other version stops the run. Run from the crate root:

    python3 validation/tools/prep/agq_tiny_weight_reference.py
"""

import json
import math
import os
import sys

import mpmath as mp
from mpmath.libmp.libmpf import round_nearest, to_float

SUITE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(SUITE_DIR, "grid", "versions.json"), encoding="utf-8") as f:
    pinned = json.load(f)["mpmath"]
if mp.__version__ != pinned:
    sys.exit(f"mpmath is {mp.__version__}, grid/versions.json pins {pinned}")

mp.mp.dps = 160

N = 40
W_PRIOR = 1e-100
U_G = [-0.6, 0.4, 0.1, -0.3, 0.7, -0.2, 0.5, -0.4]
xs = [math.sin(1.7 * i + 0.3) for i in range(N)]
noise = [math.sin(2.3 * i + 0.7) for i in range(N)]
g = [i // 5 for i in range(N)]
y = [1.0 if 0.2 + 0.8 * xs[i] + U_G[g[i]] + 0.8 * noise[i] > 0 else 0.0 for i in range(N)]
# src/consts.rs GH_NODES / GH_WEIGHTS, order 5.
Z = [-2.02018287045608558e00, -9.58572464613818509e-01, 0.0, 9.58572464613818509e-01, 2.02018287045608558e00]
WJ = [1.99532420590459170e-02, 3.93619323152241074e-01, 9.45308720482941789e-01, 3.93619323152241074e-01, 1.99532420590459170e-02]


def to_f64(v):
    return to_float(v._mpf_, rnd=round_nearest)


def deviance(beta, lam):
    w, lam = mp.mpf(W_PRIOR), mp.mpf(lam)
    ef = [mp.mpf(beta[0]) + mp.mpf(beta[1]) * mp.mpf(xs[i]) for i in range(N)]
    wsum = sum(mp.mpf(v) for v in WJ)
    dev = mp.mpf(0)
    for c in range(8):
        rows = range(5 * c, 5 * c + 5)

        def l_c(uu):
            return -uu * uu / 2 + sum(
                w * (y[i] * (ef[i] + lam * uu) - mp.log(1 + mp.exp(ef[i] + lam * uu))) for i in rows
            )

        u = mp.mpf(0)
        for _ in range(60):
            p = [1 / (1 + mp.exp(-(ef[i] + lam * u))) for i in rows]
            grad = sum(w * lam * (y[i] - pi) for i, pi in zip(rows, p)) - u
            curv = sum(w * lam * lam * pi * (1 - pi) for pi in p) + 1
            u += grad / curv
        p = [1 / (1 + mp.exp(-(ef[i] + lam * u))) for i in rows]
        a = sum(w * lam * lam * pi * (1 - pi) for pi in p) + 1
        sig = 1 / mp.sqrt(a)
        l0 = l_c(u)
        s = sum(
            mp.mpf(wj) * mp.exp(mp.mpf(zj) ** 2 + l_c(u + mp.sqrt(2) * sig * mp.mpf(zj)) - l0)
            for zj, wj in zip(Z, WJ)
        )
        dev += -2 * (l0 - mp.log(a) / 2 + mp.log(s / wsum))
    return dev


B0 = [0.3669824736291462, 1.586548701230152]
for beta, lam in [(B0, 1.0), (B0, 0.5), ([B0[0] + 0.1, B0[1] - 0.2], 1.0), ([0.0, 0.0], 0.1)]:
    print(f"beta = [{beta[0]!r}, {beta[1]!r}], lambda = {lam!r}: deviance = {to_f64(deviance(beta, lam))!r}")
