# Copy of docs/0.9.0/cr2-large-clusters/lib.py; dense_cluster() carries the cut of spec 2-dcr §3.2.
# Its scipy.linalg calls are numpy ones here, so the validation venv needs no scipy.
"""CR2 on large clusters: shared code. Dense reference, data generator, quadrature method.

Notation matches the spec: Phi_k = diag(lam) + U U' (U = Z L, L L' = sigma^2 D), M = (sum H_k)^-1,
H_k = X' Phi^-1 X, C = Phi - X M X', E_k = M X' Phi^-1 A_k.
Stored per cluster: e = E r, EPE = E Phi E', EX = E X.
"""
import numpy as np

CUT = 1e-12

def make_data(G, nk, spread, re="ri", seed=1, p3=True, weights="logunif", tau2=1.0, sigma2=1.0):
    """X = [1, x1 (row level), x2 (cluster level)]. Weights log-uniform inside each cluster with
    largest/smallest = spread exactly. nk may be an int or a list."""
    rng = np.random.default_rng(seed)
    nks = [nk] * G if np.isscalar(nk) else list(nk)
    cl = []
    x2 = rng.normal(size=G)
    Gm = np.array([[tau2]]) if re == "ri" else np.array([[tau2, 0.2], [0.2, 0.25]])
    L = np.linalg.cholesky(Gm)
    beta = np.array([1.0, 0.5, 0.3])
    for k, n in enumerate(nks):
        x1 = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x1, np.full(n, x2[k])])
        if not p3:
            X = X[:, :2]
        if weights == "logunif":
            lw = rng.uniform(0, np.log(spread), n)
            if n >= 2: lw[0], lw[1] = 0.0, np.log(spread)
        elif weights == "twopoint":
            lw = np.where(rng.uniform(size=n) < 0.5, 0.0, np.log(spread))
            if n >= 2: lw[0], lw[1] = 0.0, np.log(spread)
        elif callable(weights):
            lw = weights(rng, n)
        w = np.exp(lw - lw.mean())
        lam = sigma2 / w
        Z = np.ones((n, 1)) if re == "ri" else np.column_stack([np.ones(n), x1])
        U = Z @ L
        b = rng.normal(size=U.shape[1])
        y = X @ beta[: X.shape[1]] + U @ b + rng.normal(size=n) * np.sqrt(lam)
        cl.append(dict(X=X, U=U, lam=lam, y=y))
    return cl


def woodbury_solve(lam, U, B):
    """Phi^-1 B for Phi = diag(lam) + U U'."""
    Oi = B / lam[:, None] if B.ndim == 2 else B / lam
    if U.shape[1] == 0:
        return Oi
    UO = U / lam[:, None]
    K = np.eye(U.shape[1]) + U.T @ UO
    return Oi - UO @ np.linalg.solve(K, U.T @ Oi)


def gls(cl):
    """Fill H, g(=X'Phi^-1 r) per cluster, M, Mi, H_mk (leave-one-cluster-out info), residuals."""
    p = cl[0]["X"].shape[1]
    for c in cl:
        c["PX"] = woodbury_solve(c["lam"], c["U"], c["X"])
        c["H"] = c["X"].T @ c["PX"]
        c["H"] = 0.5 * (c["H"] + c["H"].T)
    Mi = sum(c["H"] for c in cl)
    M = np.linalg.inv(Mi); M = 0.5 * (M + M.T)
    b = M @ sum(c["PX"].T @ c["y"] for c in cl)
    # prefix/suffix sums, so H_mk has no subtraction in it
    G = len(cl); pre = [np.zeros((p, p))]; suf = [np.zeros((p, p))]
    for c in cl: pre.append(pre[-1] + c["H"])
    for c in reversed(cl): suf.append(suf[-1] + c["H"])
    for k, c in enumerate(cl):
        c["Hmk"] = pre[k] + suf[G - 1 - k]
        c["r"] = c["y"] - c["X"] @ b
        c["g"] = c["PX"].T @ c["r"]
    return M, Mi, b


def summarize(st, M, Mi=None, contrasts=None):
    """st: list of dicts (e, EPE, EX). Returns SE and df per coefficient (spec's formulas)."""
    p = M.shape[0]
    cs = np.eye(p) if contrasts is None else contrasts
    V = sum(np.outer(s["e"], s["e"]) for s in st)
    se, df = [], []
    for c in cs:
        q = np.array([c @ s["EPE"] @ c for s in st])
        Ug = np.array([c @ s["EX"] for s in st])          # G x p, rows u_k' X_k
        Qd = np.diag(q) - Ug @ M @ Ug.T
        df.append(np.trace(Qd) ** 2 / np.sum(Qd ** 2))
        se.append(np.sqrt(c @ V @ c))
    return np.array(se), np.array(df)


def dense_cluster(c, M, ret_A=False):
    """The spec's dense formula with the T-eigenvalue cut. Port of cr2_phi()."""
    X, U, lam, r = c["X"], c["U"], c["lam"], c["r"]
    n = len(lam)
    Phi = np.diag(lam) + U @ U.T
    D = np.linalg.cholesky(Phi).T                  # upper, Phi = D'D (R's chol)
    Cm = Phi - X @ M @ X.T
    Tm = np.linalg.solve(D.T, np.linalg.solve(D.T, Cm).T)
    tev = np.linalg.eigvalsh(0.5 * (Tm + Tm.T))
    rk = int(np.sum(tev > CUT))
    B = D @ Cm @ D.T
    ev, vec = np.linalg.eigh(0.5 * (B + B.T))
    keep = np.argsort(ev)[n - rk:]
    Bmh = (vec[:, keep] * ev[keep] ** -0.5) @ vec[:, keep].T
    A = D.T @ Bmh @ D
    E = M @ c["PX"].T @ A
    out = dict(e=E @ r, EPE=E @ Phi @ E.T, EX=E @ X, rank=rk, tmin=tev.min())
    if ret_A:
        out["A"] = A
    return out


# ---- rational approximation of z^(-1/2) on [m, M] (Zolotarev / Hale-Higham-Trefethen method 3) ----

def agm_K(kc):
    """K(k) from the complementary modulus kc."""
    a, b = 1.0, kc
    while abs(a - b) > 1e-15 * a:
        a, b = 0.5 * (a + b), np.sqrt(a * b)
    return np.pi / (2 * a)


def sc_dn(u, kc):
    """sc(u) = sn/cn and dn(u), 0 < u <= K/2 + tiny, from the complementary modulus kc (Bulirsch's
    rational recurrence; takes kc directly, so kc^2 = 1e-36 loses nothing)."""
    a, emc, em, en = 1.0, kc * kc, [], []
    for _ in range(40):
        em.append(a)
        emc = np.sqrt(emc)
        en.append(emc)
        c = 0.5 * (a + emc)
        if abs(a - emc) <= 1e-9 * a:
            break
        emc *= a
        a = c
    u = u * c
    sn, cn = np.sin(u), np.cos(u)
    a = cn / sn
    c *= a
    dn = 1.0
    for ii in range(len(em) - 1, -1, -1):
        b = em[ii]
        a *= c
        c *= dn
        dn = (en[ii] + a) / (b + a)
        a = c / b
    return 1.0 / c, dn                                # c = cn/sn at the end


def nodes(zmin, zmax, N):
    """Nodes s_j, weights w_j with z^(-1/2) ~= sum_j w_j / (z + s_j^2) on [zmin, zmax].
    From (2/pi) int_0^inf ds/(z+s^2), s = sqrt(zmin) sc(u | k'), midpoint rule in u."""
    kc = np.sqrt(zmin / zmax)
    K = agm_K(kc)
    s = np.zeros(N); w = np.zeros(N)
    h = K / N
    pref = 2 * h * np.sqrt(zmin) / np.pi
    for j in range((N + 1) // 2):
        sc, dn = sc_dn((j + 0.5) * h, kc)
        s[j] = np.sqrt(zmin) * sc
        w[j] = pref * dn * (1 + sc * sc)              # dn / cn^2, and 1/cn^2 = 1 + sc^2
    cc = np.sqrt(zmin * zmax)                       # mirror in log s: exact, and avoids cn -> 0
    for j in range(N // 2):
        i = N - 1 - j
        s[i] = cc / s[j]
        w[i] = w[j] * cc / s[j] ** 2
    return s, w


def N_for(ratio, tol):
    """Poles for relative error tol on a spectrum of width ratio (HHT eq. 4.7, constant from nodes_check.py)."""
    return int(np.ceil(np.log(4 / tol) * (np.log(ratio) + 2.78) / (2 * np.pi ** 2))) + 1


def spectrum(c, M):
    """Bounds on the spectrum of B, the leverage cut on T's p special eigenvalues, null directions.
    tau: eigenvalues of T that differ from 1 = eigenvalues of M^(1/2) H_mk M^(1/2)."""
    Lm = np.linalg.cholesky(M)
    S = Lm.T @ c["Hmk"] @ Lm
    tau, W = np.linalg.eigh(0.5 * (S + S.T))
    dead = tau <= CUT
    Vnull = Lm @ W[:, dead]                         # v with H_mk v = 0 (up to the cut)
    live = tau[~dead]
    tmin = min(live.min() if len(live) else 1.0, 1.0)
    lo = c["lam"].min()
    hi = c["lam"].max() + (np.linalg.eigvalsh(c["U"].T @ c["U"]).max() if c["U"].shape[1] else 0.0)
    return lo ** 2 * tmin, hi ** 2, Vnull, tau


def quad_cluster(c, M, Mi, tol=1e-13, N=None, series=None, fresh=None):
    """CR2 stored quantities by quadrature. Costs N weighted cross-products of [X, U, r] and N small
    solves. No n x n matrix and no n-vector beyond the data.
    A = I + sum_j w_j R_j^-1 X G_j^-1 X' R_j^-1, R_s = Phi + s^2 Phi^-1, G_s = M^-1 - X' R_s^-1 X."""
    X, U, lam, r = c["X"], c["U"], c["lam"], c["r"]
    p, q = X.shape[1], U.shape[1]
    zmin, zmax, Vnull, tau = spectrum(c, M)
    d = Vnull.shape[1]
    Y = np.column_stack([X, U, r])
    ip = np.arange(p); iu = p + np.arange(q); ir = p + q
    H, g = c["H"], c["g"]
    ratio = zmax / zmin
    if N is None:
        N = N_for(ratio, tol)
    s, w = nodes(zmin, zmax, N)
    LX = H.copy(); Lr = g.copy()
    if d:                                            # a leverage-1 direction: see quad_cluster_deflated
        return quad_cluster_deflated(c, M, Mi, Vnull, zmin, zmax, N, s, w)
    for sj, wj in zip(s, w):
        den = lam ** 2 + sj ** 2
        wre = lam / den; wim = sj / den
        Gr = Y.T @ (Y * wre[:, None]); Gi = Y.T @ (Y * wim[:, None])
        if q:
            Gam = Gr + 1j * Gi
            Kq = np.eye(q) + Gam[np.ix_(iu, iu)]
            sel = np.r_[ip, ir]
            T = Gam[np.ix_(sel, sel)] - Gam[np.ix_(sel, iu)] @ np.linalg.solve(Kq, Gam[np.ix_(iu, sel)])
            Tr = T.real; Ti = T.imag
            Q, Qr = Tr[:p, :p], Tr[:p, p]
            P = Ti[:p, :p] / sj
        else:
            Q, Qr = Gr[:p, :p], Gr[:p, ir]
            P = Gi[:p, :p] / sj
        Gs = c["Hmk"] + (H - Q)
        if series is None:
            solved = np.linalg.solve(Gs, np.column_stack([Q, Qr]))
        else:                                        # truncated expansion in leverage, for candidate 4
            RHS = np.column_stack([Q, Qr]); term = M @ RHS; solved = term.copy()
            for _ in range(series):
                term = M @ (Q @ term); solved += term
        cor = P @ solved
        LX += wj * cor[:, :p]; Lr += wj * cor[:, p]
    F = M @ LX
    return dict(e=M @ Lr, EPE=M @ H @ M + F @ M @ F.T, EX=F, N=N, ratio=ratio, d=0)


def quad_cluster_deflated(c, M, Mi, Vnull, zmin, zmax, N, s, w):
    """Leverage-1 case. B has null vectors; C c = 0 with c = Phi^-1 X V. Add s0 on those directions so
    the matrix is full rank, take the quadrature with one more low-rank column y = Phi^-1 c, and remove
    the known term. A = A~ - s0^(-1/2) c (c' Phi^-1 c)^-1 c'.  Phi~ = Phi - c (c' Phi^-1 c)^-1 c'."""
    X, U, lam, r = c["X"], c["U"], c["lam"], c["r"]
    p, q = X.shape[1], U.shape[1]
    d = Vnull.shape[1]
    cc = c["PX"] @ Vnull                                 # n x d
    y = woodbury_solve(lam, U, cc)                       # Phi^-1 c
    S = cc.T @ y; S = 0.5 * (S + S.T)
    s0 = zmax
    Lc = np.column_stack([X, y])                         # low-rank columns of C~
    Jinv = np.zeros((p + d, p + d))
    Jinv[:p, :p] = Mi
    Jinv[p:, p:] = -S / s0
    Y = np.column_stack([Lc, U, r])
    pl = p + d
    iu = pl + np.arange(q); ir = pl + q
    H = c["H"]; g = c["g"]
    # A~ = I + sum_j w_j R^-1 Lc K^-1 Lc' R^-1, K = J^-1 - Lc' R^-1 Lc
    # X' Phi^-1 A~ b = X' Phi^-1 b + sum_j w_j (X' (Phi^2+s^2)^-1 Lc) K^-1 (Lc' R^-1 b)
    LX = H.copy(); Lr = g.copy()
    sel = np.r_[np.arange(pl), ir]
    for sj, wj in zip(s, w):
        den = lam ** 2 + sj ** 2
        wre = lam / den; wim = sj / den
        Gr = Y.T @ (Y * wre[:, None]); Gi = Y.T @ (Y * wim[:, None])
        if q:
            Gam = Gr + 1j * Gi
            Kq = np.eye(q) + Gam[np.ix_(iu, iu)]
            T = Gam[np.ix_(sel, sel)] - Gam[np.ix_(sel, iu)] @ np.linalg.solve(Kq, Gam[np.ix_(iu, sel)])
            Tr, Ti = T.real, T.imag
        else:
            Tr, Ti = Gr[np.ix_(sel, sel)], Gi[np.ix_(sel, sel)]
        K = Jinv - Tr[:pl, :pl]
        K[:p, :p] = c["Hmk"] + (H - Tr[:p, :p])        # same as Mi - Q, with no large cancellation
        P = Ti[:p, :pl] / sj
        sol = np.linalg.solve(K, Tr[:pl, np.r_[np.arange(p), pl]])
        cor = P @ sol
        LX += wj * cor[:, :p]; Lr += wj * cor[:, p]
    Pc = c["PX"].T @ cc                                   # X' Phi^-1 c
    Si = np.linalg.inv(S)
    LX -= s0 ** -0.5 * Pc @ Si @ (cc.T @ X)
    Lr -= s0 ** -0.5 * Pc @ Si @ (cc.T @ r)
    F = M @ LX
    Htil = H - Pc @ Si @ Pc.T
    return dict(e=M @ Lr, EPE=M @ Htil @ M + F @ M @ F.T, EX=F, N=N, ratio=zmax / zmin, d=d)


def whitened_cluster(c, M):
    """Closed form: CR2 applied after whitening. NOT the spec's estimator unless Phi is a multiple of I.
    Uses (I-RR')^a R = R (I-R'R)^a (Niccodemi et al. 2020 eq. 6) on the whitened X."""
    Lm = np.linalg.cholesky(M)
    S = Lm.T @ c["H"] @ Lm
    ev, W = np.linalg.eigh(0.5 * (S + S.T))
    f = np.where(1 - ev > CUT, (1 - ev).clip(1e-300) ** -0.5, 0.0)
    Lk = (Lm @ W * f) @ (Lm @ W).T
    return dict(e=Lk @ c["g"], EPE=Lk @ c["H"] @ Lk, EX=Lk @ c["H"])


def compare(a, b):
    """Largest relative error of the stored quantities of a against b, cluster by cluster."""
    out = {}
    for key in ("e", "EPE", "EX"):
        out[key] = max(np.max(np.abs(x[key] - y[key])) / np.max(np.abs(y[key])) for x, y in zip(a, b))
    return out
