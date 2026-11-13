# Computes no B^(-1/2) by eigendecomposition (lib.dense_cluster does).
"""Rounding-safe versions of gls() and quad_cluster(). Same maths as lib.py; only the arrangement of the small matrices changes.
Split each column once: [X, r] = Yt + U B with Yt' Lam^-1 U = 0. Then for Omega = (Lam - i s)^-1, S_s = U' Omega U, K = I + S_s:
    Yt' (Phi-is)^-1 Yt = Yt' Omega Yt - (Yt' Omega U) K^-1 (U' Omega Yt)     (no large cancellation any more)
    Yt' (Phi-is)^-1 U  = (Yt' Omega U) K^-1
    U'  (Phi-is)^-1 U  = S_s K^-1
At s = 0: Phi^-1 Yt = Lam^-1 Yt exactly, and U' Phi^-1 U = S (I+S)^-1."""
import numpy as np
from lib import CUT, nodes, N_for, spectrum, summarize, compare, make_data, dense_cluster

def split(lam, U, Y):
    """Y = Yt + U B with Yt' Lam^-1 U = 0, two passes so the orthogonality is good to rounding."""
    Ul = U / lam[:, None]
    S = U.T @ Ul
    B = np.linalg.solve(S, Ul.T @ Y)
    Yt = Y - U @ B
    dB = np.linalg.solve(S, Ul.T @ Yt)
    return Yt - U @ dB, B + dB, S

def base_T(lam, U, Yt, B, S):
    """Y' Phi^-1 Y, real."""
    Sf = np.linalg.solve(np.eye(S.shape[0]) + S, S)
    return Yt.T @ (Yt / lam[:, None]) + B.T @ Sf @ B

def node_T(lam, s, U, Yt, B):
    """Y' (Phi - i s)^-1 Y, complex, from one weighted cross-product of [Yt, U]."""
    den = lam ** 2 + s ** 2
    A = np.column_stack([Yt, U]); m = Yt.shape[1]; q = U.shape[1]
    G = A.T @ (A * (lam / den)[:, None]) + 1j * (A.T @ (A * (s / den)[:, None]))
    Gtt, Gtu, Guu = G[:m, :m], G[:m, m:], G[m:, m:]
    K = np.eye(q) + Guu
    KiGut = np.linalg.solve(K, Gtu.T)                  # K^-1 U' Omega Yt
    KiGuu = np.linalg.solve(K, Guu)
    Ttt = Gtt - Gtu @ KiGut
    Tut = KiGut                                         # = U'(Phi-is)^-1 Yt, by symmetry of K
    Tuu = KiGuu
    return Ttt + Tut.T @ B + B.T @ Tut + B.T @ Tuu @ B

def gls(cl):
    p = cl[0]["X"].shape[1]
    for c in cl:
        Yt, B, S = split(c["lam"], c["U"], np.column_stack([c["X"], c["y"]]))
        T = base_T(c["lam"], c["U"], Yt, B, S)
        c["H"] = 0.5 * (T[:p, :p] + T[:p, :p].T); c["gy"] = T[:p, p]
        Xt, Bx = Yt[:, :p], B[:, :p]
        c["PX"] = Xt / c["lam"][:, None] + (c["U"] / c["lam"][:, None]) @ np.linalg.solve(np.eye(S.shape[0]) + S, Bx)
    Mi = sum(c["H"] for c in cl)
    M = np.linalg.inv(Mi); M = 0.5 * (M + M.T)
    b = M @ sum(c["gy"] for c in cl)
    G = len(cl); pre = [np.zeros((p, p))]; suf = [np.zeros((p, p))]
    for c in cl: pre.append(pre[-1] + c["H"])
    for c in reversed(cl): suf.append(suf[-1] + c["H"])
    for k, c in enumerate(cl):
        c["Hmk"] = pre[k] + suf[G - 1 - k]
        c["r"] = c["y"] - c["X"] @ b
        c["g"] = c["PX"].T @ c["r"]
    return M, Mi, b

def quad_cluster(c, M, Mi, tol=1e-13, N=None):
    X, U, lam, r = c["X"], c["U"], c["lam"], c["r"]
    p, q = X.shape[1], U.shape[1]
    zmin, zmax, Vnull, tau = spectrum(c, M)
    d = Vnull.shape[1]
    N = N or N_for(zmax / zmin, tol)
    s, w = nodes(zmin, zmax, N)
    H = c["H"]
    Hmk = c["Hmk"]
    if d == 0:
        Yt, B, S = split(lam, U, np.column_stack([X, r]))
        T0 = base_T(lam, U, Yt, B, S)
        g = T0[:p, p]
        LX, Lr = H.copy(), g.copy()
        for sj, wj in zip(s, w):
            T = node_T(lam, sj, U, Yt, B)
            Tr, Ti = T.real, T.imag
            Q, Qr, P = Tr[:p, :p], Tr[:p, p], Ti[:p, :p] / sj
            cor = P @ np.linalg.solve(Hmk + (H - Q), np.column_stack([Q, Qr]))
            LX += wj * cor[:, :p]; Lr += wj * cor[:, p]
        F = M @ LX
        return dict(e=M @ Lr, EPE=M @ H @ M + F @ M @ F.T, EX=F, N=N, ratio=zmax / zmin, d=0, g=g)
    # Leverage 1. c = Phi^-1 X V; y = Phi^-1 c; add s0 on the null directions, take the same sum with y as one more column.
    def phi_solve(Z):
        Zt, Bz, S = split(lam, U, Z)
        return Zt / lam[:, None] + (U / lam[:, None]) @ np.linalg.solve(np.eye(q) + S, Bz)
    cc = phi_solve(X @ Vnull)
    y = phi_solve(cc)
    Sc = cc.T @ y; Sc = 0.5 * (Sc + Sc.T)
    s0 = zmax
    pl = p + d
    Y = np.column_stack([X, y, r])
    Yt, B, S = split(lam, U, Y)
    T0 = base_T(lam, U, Yt, B, S)
    g = T0[:p, pl]
    LX, Lr = H.copy(), g.copy()
    sel = np.r_[np.arange(p), pl]
    for sj, wj in zip(s, w):
        T = node_T(lam, sj, U, Yt, B)
        Tr, Ti = T.real, T.imag
        K = -Tr[:pl, :pl]
        K[:p, :p] = Hmk + (H - Tr[:p, :p])
        K[p:, p:] += -Sc / s0
        cor = (Ti[:p, :pl] / sj) @ np.linalg.solve(K, Tr[:pl][:, sel])
        LX += wj * cor[:, :p]; Lr += wj * cor[:, p]
    Pc = phi_solve(X).T @ cc
    Si = np.linalg.inv(Sc)
    LX -= s0 ** -0.5 * Pc @ Si @ (cc.T @ X)
    Lr -= s0 ** -0.5 * Pc @ Si @ (cc.T @ r)
    F = M @ LX
    return dict(e=M @ Lr, EPE=M @ (H - Pc @ Si @ Pc.T) @ M + F @ M @ F.T, EX=F, N=N, ratio=zmax / zmin, d=d)
