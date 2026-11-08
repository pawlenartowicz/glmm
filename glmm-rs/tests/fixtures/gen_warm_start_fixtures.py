#!/usr/bin/env python3
"""The two fixtures of the GLMM warm-start tests in src/fit/glmm_tests.rs
(a_far_warm_start_that_fails_falls_back_to_the_cold_fit,
a_huge_warm_beta_is_not_reported_converged_at_the_start). Their asserted values
are glmm's own cold fits of the same data; nothing here is an external oracle.

warm_start_far.csv, 160 rows x,g,h,y_poisson,y_gamma: a 64-bit xorshift draw
(seed 99 for x, g, h; seeds 3 and 2 for the two responses). y_poisson is a
Poisson count with log-mean 0.3 + 0.5 x + u_g + u_h (u_g ~ N(0, 0.7^2),
u_h ~ N(0, 0.5^2)); y_gamma is 1e-12 * (1 + U(0, 1)), a response on the 1e-12
scale.

warm_start_huge_beta.csv, 60 rows y,x,g: numpy.random.default_rng(1),
x ~ N(0, 1), y Bernoulli(0.5), 6 groups g0..g5 of 10 rows.

Floats are written by repr (shortest round-trip). Written with Python 3.14.2
and numpy 2.5.4 on x86_64 Linux. Run from the crate root:

    python3 tests/fixtures/gen_warm_start_fixtures.py
"""

import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
M = (1 << 64) - 1


class Rng:
    def __init__(self, s):
        self.s = s & M

    def u(self):
        s = self.s
        s ^= (s << 13) & M
        s ^= s >> 7
        s ^= (s << 17) & M
        self.s = s
        return (s >> 11) / float(1 << 53)

    def n(self):
        a = max(self.u(), 1e-300)
        b = self.u()
        return math.sqrt(-2.0 * math.log(a)) * math.cos(6.283185307179586 * b)

    def pois(self, mu):
        lim = math.exp(-mu)
        k, p = 0, 1.0
        while True:
            p *= self.u()
            if p < lim:
                return k
            k += 1


def far():
    n = 160
    r = Rng(99)
    x = [r.n() for _ in range(n)]
    g = [i % 8 for i in range(n)]
    h = [int(r.u() * 20.0) for _ in range(n)]

    def draw(seed):
        r = Rng(seed * 2654435761 + 12345)
        ug = [r.n() * 0.7 for _ in range(64)]
        uh = [r.n() * 0.5 for _ in range(64)]
        return r, [0.3 + 0.5 * x[i] + ug[g[i]] + uh[h[i]] for i in range(n)]

    r, eta = draw(3)
    y_poisson = [r.pois(math.exp(e)) for e in eta]
    r, _ = draw(2)
    y_gamma = [1e-12 * (1.0 + r.u()) for _ in range(n)]
    rows = ["x,g,h,y_poisson,y_gamma"]
    for i in range(n):
        rows.append(f"{x[i]!r},g{g[i]},h{h[i]},{y_poisson[i]},{y_gamma[i]!r}")
    (OUT / "warm_start_far.csv").write_text("\n".join(rows) + "\n")


def huge_beta():
    rng = np.random.default_rng(1)
    n = 60
    g = np.repeat(np.arange(6), 10)
    x = rng.normal(size=n)
    y = (rng.random(n) < 0.5).astype(int)
    rows = ["y,x,g"]
    for i in range(n):
        rows.append(f"{y[i]},{float(x[i])!r},g{g[i]}")
    (OUT / "warm_start_huge_beta.csv").write_text("\n".join(rows) + "\n")


far()
huge_beta()
