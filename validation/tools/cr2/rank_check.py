# Copy of docs/0.9.0/cr2-large-clusters/rank_check.py; computes no B^(-1/2) by eigendecomposition (lib.dense_cluster does).
# reduce_u's cut is eps * max(1, largest ev) here, not eps, so rounding on a large S drops out.
"""Pole sum against the dense formula when U has less than full column rank in a cluster:
zero random-effect variance, a zero slope variance, a slope column constant inside a cluster,
and clusters with fewer rows than random effects.

The split needs S = U' Lam^-1 U invertible. Reduce U once per cluster: with S = V ev V', keep the
directions with ev > eps * max(1, max ev) (S has no units; above 1 the cut is relative, so the
rounding of a large S is not kept). A dropped direction with ev = 0 has U v = 0, so U U' is unchanged; one with
0 < ev <= eps moves I + Lam^-1/2 U U' Lam^-1/2 by less than rounding.
Run: python rank_check.py"""
import numpy as np
from lib import summarize, dense_cluster
from centered import gls, quad_cluster

EPS = np.finfo(float).eps


def reduce_u(c):
    U, lam = c["U"], c["lam"]
    S = U.T @ (U / lam[:, None])
    ev, V = np.linalg.eigh(0.5 * (S + S.T))
    c["U"] = U @ V[:, ev > EPS * max(1.0, ev.max())]
    return c


def cases(seed):
    rng = np.random.default_rng(seed)

    def build(nks, re_cov, slope_col):
        cl = []
        L = np.linalg.cholesky(re_cov + 1e-300 * np.eye(len(re_cov))) if np.any(re_cov) else np.zeros_like(re_cov)
        x2 = rng.normal(size=len(nks))
        for k, n in enumerate(nks):
            x1 = rng.normal(size=n)
            X = np.column_stack([np.ones(n), x1, np.full(n, x2[k])])
            lw = rng.uniform(0, np.log(100.0), n)
            lam = 1.0 / np.exp(lw - lw.mean())
            z = {"none": None, "x1": x1, "x2": X[:, 2]}[slope_col]
            Z = np.ones((n, 1)) if z is None else np.column_stack([np.ones(n), z])
            U = Z @ L
            y = X @ np.array([1.0, 0.5, 0.3]) + U @ rng.normal(size=U.shape[1]) + rng.normal(size=n) * np.sqrt(lam)
            cl.append(dict(X=X, U=U, lam=lam, y=y))
        return cl

    slope = np.array([[1.0, 0.2], [0.2, 0.25]])
    yield "zero RE variance", build([40] * 10, np.zeros((1, 1)), "none")
    yield "RE variance 1e-20", build([40] * 10, np.array([[1e-20]]), "none")
    yield "zero slope variance", build([40] * 10, np.array([[1.0, 0.0], [0.0, 0.0]]), "x1")
    yield "slope constant in cluster", build([40] * 10, slope, "x2")
    yield "clusters of 1 and 2, slope", build([1, 2, 1, 2, 30, 30, 30, 30, 30, 30], slope, "x1")
    yield "many clusters of 2 to 5", build(list(rng.integers(2, 6, 200)), slope, "x1")


print(f"{'case':30s} {'max SE rel':>11s} {'max df rel':>11s} {'mean q':>7s} {'N':>4s}")
for seed in (1, 2, 3):
    for name, cl in cases(seed):
        for c in cl:
            reduce_u(c)
        M, Mi, b = gls(cl)
        quad = [quad_cluster(c, M, Mi) for c in cl]
        dense = [dense_cluster(c, M) for c in cl]
        se_q, df_q = summarize(quad, M)
        se_d, df_d = summarize(dense, M)
        print(f"{name:30s} {np.max(np.abs(se_q / se_d - 1)):11.2e} {np.max(np.abs(df_q / df_d - 1)):11.2e} "
              f"{np.mean([c['U'].shape[1] for c in cl]):7.2f} {max(q['N'] for q in quad):4d}")
