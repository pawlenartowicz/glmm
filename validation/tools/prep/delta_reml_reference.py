#!/usr/bin/env python3
"""110-digit REML profile of the random-intercept model with
Var(e_i) = s2 * w_i^-delta, for the `HI_PREC_REF` table of
src/fit/delta_tests.rs, which keeps the printed values inline.

Data: validation/data/simulated/delta_weights_seed13.csv, the rows of
`dataset(13, 150, 10, 1.0, 1e16, 1.5)` in src/fit/delta_tests.rs (columns x1,
y, w, g; the test `the_reference_data_is_the_test_dataset` checks the file
against the generator). The criterion is lme4's weighted REML deviance of
y ~ 1 + x1 + (1 | g) on weights w^delta, from the Cholesky factor of the
Gram matrix of [Z*theta | 1 | x1 | y] plus I on the Z block, all in Python's
decimal module at 110 digits. `profile` maximizes over ln(theta) by a coarse
scan of [-20, 40] and a golden section to 1e-7, and prints
delta,loglik,ln(theta); `theta` prints delta's loglik and sigma^2 at given
ln(theta) values. Nothing is written to disk. Run from the crate root:

    python3 validation/tools/prep/delta_reml_reference.py \
        validation/data/simulated/delta_weights_seed13.csv profile -2,-1.5,0.7,1,1.5
"""
import csv, sys
from decimal import Decimal as D, getcontext
getcontext().prec = 110
rows = list(csv.DictReader(open(sys.argv[1])))
n = len(rows); G = 1 + max(int(r["g"]) for r in rows); p = 2
x1 = [D(r["x1"]) for r in rows]; y = [D(r["y"]) for r in rows]
lw = [D(r["w"]).ln() for r in rows]; g = [int(r["g"]) for r in rows]
PI2 = D("6.28318530717958647692528676655900576839433879875021164194988918461563281")

def gram(delta):
    w = [(delta * l).exp() for l in lw]
    k = G + p + 1
    M = [[D(0)] * k for _ in range(k)]
    for i in range(n):
        v = [D(0)] * k; v[g[i]] = D(1); v[G] = D(1); v[G + 1] = x1[i]; v[G + 2] = y[i]
        nz = [g[i], G, G + 1, G + 2]
        for a in nz:
            for b in nz:
                M[a][b] += w[i] * v[a] * v[b]
    return M, sum(delta * l for l in lw)

def dev(M, slw, theta, reml=True):
    k = len(M); th = D(theta)
    A = [[M[a][b] * (th if a < G else 1) * (th if b < G else 1) for b in range(k)] for a in range(k)]
    for a in range(G): A[a][a] += 1
    L = [[D(0)] * k for _ in range(k)]
    for j in range(k):
        s = A[j][j] - sum(L[j][m] ** 2 for m in range(j))
        if s <= 0: return None
        L[j][j] = s.sqrt()
        for i in range(j + 1, k):
            L[i][j] = (A[i][j] - sum(L[i][m] * L[j][m] for m in range(j))) / L[j][j]
    ldz = sum(2 * L[j][j].ln() for j in range(G))
    ldx = sum(2 * L[j][j].ln() for j in range(G, G + p))
    r2 = L[k - 1][k - 1] ** 2
    nu = n - p if reml else n
    s2 = r2 / nu
    d = ldz + (ldx if reml else 0) + nu * (1 + (PI2 * s2).ln()) - slw
    return d, s2

def best(M, slw, reml=True, lo=-20.0, hi=40.0):
    # golden on ln theta, after a coarse scan
    import math
    f = lambda t: (lambda r: D("Infinity") if r is None else r[0])(dev(M, slw, D(t).exp(), reml))
    ts = [lo + (hi - lo) * i / 120 for i in range(121)]
    vs = [f(t) for t in ts]
    i = min(range(len(vs)), key=lambda j: vs[j])
    a, b = ts[max(i - 1, 0)], ts[min(i + 1, 120)]
    gr = (math.sqrt(5) - 1) / 2
    c, d_ = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d_)
    while b - a > 1e-7:
        if fc < fd: b, d_, fd = d_, c, fc; c = b - gr * (b - a); fc = f(c)
        else: a, c, fc = c, d_, fd; d_ = a + gr * (b - a); fd = f(d_)
    t = (a + b) / 2
    return f(t), t

if __name__ == "__main__":
    mode = sys.argv[2]
    if mode == "profile":
        for s in sys.argv[3].split(","):
            M, slw = gram(D(s)); dv, t = best(M, slw)
            print(f"{s},{-float(dv)/2:.10f},{t:.4f}", flush=True)
    elif mode == "theta":
        M, slw = gram(D(sys.argv[3]))
        for t in sys.argv[4].split(","):
            r = dev(M, slw, D(t).exp())
            print(f"{t},{-float(r[0])/2:.10f},{float(r[1]):.4e}", flush=True)
