# Computes no B^(-1/2) by eigendecomposition (lib.dense_cluster does).
"""High-precision truth for small clusters. Independent of the quadrature: A = Phi (C Phi)^(-1/2) with the
inverse root from scaled Denman-Beavers at PREC bits (flint arb, midpoints only).
Leverage-1 case: uses W = C Phi + s0 Pi0, then subtracts s0^(-1/2) Pi0, with Pi0 the spectral projector on the
null space of C Phi (derived in lib.quad_cluster_deflated's docstring, here applied to the dense matrix).
Writes hp_<tag>.npz.   Run: venv/bin/python hp_ref.py TAG G NK SPREAD RE [leverage]"""
import sys, time, numpy as np
from flint import arb_mat, arb, ctx
from lib import make_data

tag, G, nk, spread, re_ = sys.argv[1], int(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")], float(sys.argv[4]), sys.argv[5]
nk = nk[0] if len(nk) == 1 else nk
lev = len(sys.argv) > 6 and sys.argv[6]
PREC = int(sys.argv[7]) if len(sys.argv) > 7 else 1024
ctx.prec = PREC

def amat(a):
    a = np.atleast_2d(a)
    return arb_mat(a.shape[0], a.shape[1], [arb(float(x)) for x in a.ravel()])

def eye(n):
    return arb_mat(n, n, [arb(1) if i == j else arb(0) for i in range(n) for j in range(n)])

def tonp(m):
    # 53-bit midpoint of each entry; good to double precision, which is all the comparison needs
    return np.array([[float(x.mid()) for x in row] for row in m.tolist()])

def db_inv_root(W):
    n = W.nrows()
    Y, Z = W, eye(n)
    for it in range(80):
        dy, dz = Y.det().mid(), Z.det().mid()
        mu = (dy * dz).abs_lower() ** (arb(-1) / (2 * n))
        mu = mu.mid()
        Yi, Zi = Y.inv().mid(), Z.inv().mid()
        Yn = ((Y * mu + Zi * (1 / mu)) * arb(0.5)).mid()
        Zn = ((Z * mu + Yi * (1 / mu)) * arb(0.5)).mid()
        dd = max(abs(float((x - y).mid())) for r1, r2 in zip(Zn.tolist(), (Z).tolist()) for x, y in zip(r1, r2))
        nz = max(abs(float(x.mid())) for r in Zn.tolist() for x in r)
        Y, Z = Yn, Zn
        if abs(mu - 1) < arb(1e-60) and dd < 1e-70 * nz:
            break
    return Z, it

cl = make_data(G, nk, spread, re_, seed=7)
if lev == "row":             # 4th column: a dummy for one row of cluster 0
    for k, c in enumerate(cl):
        d = np.zeros(len(c["lam"]))
        if k == 0: d[5] = 1.0
        c["X"] = np.column_stack([c["X"], d])
elif lev and lev.startswith("cluster"):     # 4th column: only in cluster 0; "cluster:EPS" leaves EPS-sized noise elsewhere (leverage near 1)
    eps = float(lev.split(":")[1]) if ":" in lev else 0.0
    rng = np.random.default_rng(5)
    for k, c in enumerate(cl):
        d = rng.normal(size=len(c["lam"])) * (1.0 if k == 0 else eps)
        c["X"] = np.column_stack([c["X"], d])
p = cl[0]["X"].shape[1]
t0 = time.time()
Phi, PhiI, Xs, ys, Hs = [], [], [], [], []
for c in cl:
    n = len(c["lam"])
    P = amat(np.diag(c["lam"])) + amat(c["U"]) * amat(c["U"]).transpose()
    Xm = amat(c["X"]); ym = amat(c["y"][:, None])
    Pi = P.inv().mid()
    Phi.append(P); PhiI.append(Pi); Xs.append(Xm); ys.append(ym)
    Hs.append(Xm.transpose() * Pi * Xm)
Mi = Hs[0]
for h in Hs[1:]: Mi = Mi + h
M = Mi.inv().mid()
bvec = None
for Xm, Pi, ym in zip(Xs, PhiI, ys):
    t = Xm.transpose() * Pi * ym
    bvec = t if bvec is None else bvec + t
beta = M * bvec
res = {}
for k in range(G):
    n = Phi[k].nrows()
    r = ys[k] - Xs[k] * beta
    C = Phi[k] - Xs[k] * M * Xs[k].transpose()
    W = C * Phi[k]
    Pi0 = None
    if lev and k == 0 and ":" not in lev:
        # null direction: v = last coordinate, because H_{-k} e_p = 0 by construction
        v = arb_mat(p, 1, [arb(0)] * (p - 1) + [arb(1)])
        cvec = PhiI[k] * Xs[k] * v
        s = (cvec.transpose() * PhiI[k] * cvec)
        Pi0 = PhiI[k] * cvec * s.inv() * cvec.transpose()
        s0 = arb(float(np.max(cl[k]["lam"]) ** 2))
        W = W + Pi0 * s0
    Z, it = db_inv_root(W)
    if Pi0 is not None:
        Z = Z - Pi0 * (arb(1) / s0.sqrt())
    A = Phi[k] * Z
    E = M * Xs[k].transpose() * PhiI[k] * A
    e = E * r
    res[f"e{k}"] = tonp(e).ravel()
    res[f"EPE{k}"] = tonp(E * Phi[k] * E.transpose())
    res[f"EX{k}"] = tonp(E * Xs[k])
    print(k, "DB iterations", it, "elapsed %.1f" % (time.time() - t0), flush=True)
res["M"] = tonp(M)
np.savez(f"hp_{tag}.npz", **res)
print("done %.1f" % (time.time() - t0))
