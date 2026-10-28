#!/usr/bin/env python3
"""Extended-precision GLM references on ill-conditioned designs ->
validation/goldens/inference/conditioning/<name>.json, one per
`kind = "conditioning"` entry of the manifest's `inference_references` block.

THE ORACLE IS SACRED. These JSONs are the frozen reference the crate's GLM fits
on near-singular designs are measured against. On any glmm disagreement, glmm
is presumed wrong: never relax a tolerance or edit a reference.

A BARE RUN REFITS AND OVERWRITES EVERY REFERENCE IT OWNS (every `conditioning`
entry). Use VALIDATION_ONLY for anything short of a deliberate full re-freeze:
    VALIDATION_ONLY=<name>[,<name>...] python3 validation/tools/goldens_glm_band.py

Needs mpmath at the version of grid/versions.json's `mpmath`; any other version
stops the run. No R, no threads: the result depends on mpmath alone.

An entry names its `design`, `family`, `link` (binomial logit or probit,
poisson log, gamma log), `weights` (the prior-weight column `w`, or null) and
`fits`, the sweep: one `label` and `r_formula` per fit, each read on the entry's
`data` CSV. Only `y ~ 1 + c1 + c2`, optionally `+ offset(o)`, is read; anything
else stops the run. Every CSV value is read with float() and then made an mpf,
so the fit treats exactly the doubles every engine reads as exact data.

Per fit the JSON carries `label`, `r_formula` and `finite_mle`. On design c the
block outcomes separate along x2 - a (the data generator builds them so); the
script checks that the log-likelihood keeps rising along this direction, so no
finite beta maximizes it, and records `finite_mle: false` only. On every other
design `finite_mle` is true and the fit also carries, each rounded to the
nearest double last:
  beta_ref    the MLE, in formula column order;
  cov_ref     the exact (X'WX)^-1 at beta_ref, W the Fisher weights (prior
              weight times dmu/deta^2 / V(mu)), unscaled;
  se_ref      sqrt(phi * diag(cov_ref)); z_ref = beta_ref / se_ref;
  tau_ref     min_j R_jj^2 / G_jj, G = X'WX in column order on the design as
              written (not standardized), R its Cholesky factor;
  phi_ref     gamma only: the Pearson dispersion at beta_ref,
              sum w (y - mu)^2 / mu^2 / (n - p), the crate's Gamma GLM
              convention; phi is 1 on binomial and poisson;
  iterations  Newton iterations to the stop;
  score_norm  the largest absolute component of the score at beta_ref.

THE FIT. Newton on the score at mp.dps = 50: from beta = 0 (gamma: the
intercept at log(mean(y))), the observed information as the Jacobian, the step
halved while the log-likelihood does not rise. It stops when the largest
component of the step is below 1e-40; 200 iterations without that stop the
run. Near the MLE the rounding of the log-likelihood (worst on the
ill-conditioned fits, where eta is a difference of large terms) hides the gain
of the last steps, so the stop is often reached by halving; `score_norm`
records where it stopped. The
binomial log-likelihood is sum w [y log mu + (1 - y) log(1 - mu)] (y
the proportion, w the trials); poisson sum w (y eta - mu); gamma
sum w (-y / mu - log mu), the phi = 1 kernel, whose maximizer does not depend
on phi.
"""
import csv
import json
import os
import platform
import re
import sys

import mpmath as mp
from mpmath.libmp.libmpf import round_nearest, to_float

KIND = "conditioning"
MP_DPS = 50
STEP_STOP = mp.mpf("1e-40")
MAX_ITER = 200
GENERATOR = os.path.basename(__file__)
SUITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMULA = re.compile(r"^y ~ 1 \+ (\w+) \+ (\w+)(?: \+ offset\((\w+)\))?$")

mp.mp.dps = MP_DPS


def die(msg):
    sys.exit(f"{GENERATOR}: {msg}")


def assert_mpmath_version():
    with open(os.path.join(SUITE_DIR, "grid", "versions.json"), encoding="utf-8") as f:
        pinned = json.load(f)["mpmath"]
    if mp.__version__ != pinned:
        die(f"mpmath is {mp.__version__}, grid/versions.json pins {pinned}")


# Mirrors inference_common.R's select_entries; change together.
def select_entries(kind):
    with open(os.path.join(SUITE_DIR, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if "inference_references" not in manifest:
        die("manifest.json has no `inference_references` block")
    entries = [e for e in manifest["inference_references"] if e.get("kind") == kind]
    names = [e["name"] for e in entries]
    only = os.environ.get("VALIDATION_ONLY", "")
    if only:
        keep = only.split(",")
        unknown = [n for n in keep if n not in names]
        if unknown:
            die(f"VALIDATION_ONLY names no {kind} entry: {', '.join(unknown)}")
        return [e for e in entries if e["name"] in keep]
    for e in entries:
        if e.get("pending_reference") is not None:
            print(f"skip {e['name']} (pending_reference: {e['pending_reference']})")
    return [e for e in entries if e.get("pending_reference") is None]


# Mirrors inference_common.R's provenance; change together. No date and no host
# name, so a re-run is byte-identical.
def provenance():
    return {
        "python_version": platform.python_version(),
        "packages": {"mpmath": mp.__version__},
        "mp_dps": MP_DPS,
        "generator": GENERATOR,
    }


# jsonlite's pretty layout (two-space indent, scalar arrays on one line, one
# line per matrix row), so these files read like the R references. A double is
# written as Python's shortest round-trip repr.
def to_json(v, indent=0):
    pad, inner = "  " * indent, "  " * (indent + 1)
    if isinstance(v, dict):
        items = [f"{inner}{json.dumps(k)}: {to_json(x, indent + 1)}" for k, x in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(v, list):
        if all(not isinstance(x, (dict, list)) for x in v):
            return "[" + ", ".join(to_json(x) for x in v) + "]"
        return "[\n" + ",\n".join(inner + to_json(x, indent + 1) for x in v) + "\n" + pad + "]"
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            die("a non-finite value would be written")
        return repr(v)
    return json.dumps(v)


# Mirrors inference_common.R's write_reference; change together.
def write_reference(obj, path):
    obj["name"] = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_json(obj) + "\n")


def f64(x):
    return to_float(mp.mpf(x)._mpf_, rnd=round_nearest)


def read_csv(rel):
    with open(os.path.join(SUITE_DIR, rel), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def column(rows, name):
    if name not in rows[0]:
        die(f"column {name} is not in the CSV")
    return [mp.mpf(float(r[name])) for r in rows]


# Per row of eta, prior weight w and outcome y: (log-likelihood, d/deta,
# -d2/deta2, Fisher weight), the last three before the prior weight.
def row_terms(family, link):
    if (family, link) == ("binomial", "logit"):
        def terms(eta, y):
            mu = 1 / (1 + mp.exp(-eta))
            return y * eta - mp.log1p(mp.exp(eta)), y - mu, mu * (1 - mu), mu * (1 - mu)
    elif (family, link) == ("binomial", "probit"):
        def terms(eta, y):
            p, q, dens = mp.ncdf(eta), mp.ncdf(-eta), mp.npdf(eta)
            lam, lamc = dens / p, dens / q
            return (y * mp.log(p) + (1 - y) * mp.log(q), y * lam - (1 - y) * lamc,
                    y * lam * (eta + lam) + (1 - y) * lamc * (lamc - eta), dens ** 2 / (p * q))
    elif (family, link) == ("poisson", "log"):
        def terms(eta, y):
            mu = mp.exp(eta)
            return y * eta - mu, y - mu, mu, mu
    elif (family, link) == ("gamma", "log"):
        def terms(eta, y):
            mu = mp.exp(eta)
            return -y / mu - eta, y / mu - 1, y / mu, mp.mpf(1)
    else:
        die(f"unsupported family {family} with link {link}")
    return terms


def eval_at(terms, X, y, w, off, beta):
    """(log-likelihood, score, observed information, Fisher G = X'WX)."""
    p = len(beta)
    ll = mp.mpf(0)
    score = mp.matrix(p, 1)
    obs = mp.matrix(p, p)
    fisher = mp.matrix(p, p)
    for xi, yi, wi, oi in zip(X, y, w, off):
        eta = sum(xi[j] * beta[j] for j in range(p)) + oi
        li, d1, d2, fw = terms(eta, yi)
        ll += wi * li
        for a in range(p):
            score[a] += wi * d1 * xi[a]
            for b in range(p):
                obs[a, b] += wi * d2 * xi[a] * xi[b]
                fisher[a, b] += wi * fw * xi[a] * xi[b]
    return ll, score, obs, fisher


def loglik(terms, X, y, w, off, beta):
    p = len(beta)
    return sum(wi * terms(sum(xi[j] * beta[j] for j in range(p)) + oi, yi)[0]
               for xi, yi, wi, oi in zip(X, y, w, off))


def newton(terms, X, y, w, off, beta, label):
    ll, score, obs, _ = eval_at(terms, X, y, w, off, beta)
    for it in range(1, MAX_ITER + 1):
        delta = mp.lu_solve(obs, score)
        t = mp.mpf(1)
        while True:
            step = [t * delta[j] for j in range(len(beta))]
            if max(abs(s) for s in step) < STEP_STOP:
                return beta, it
            trial = [beta[j] + step[j] for j in range(len(beta))]
            ll_trial = loglik(terms, X, y, w, off, trial)
            if ll_trial > ll:
                break
            t /= 2
        beta = trial
        ll, score, obs, _ = eval_at(terms, X, y, w, off, beta)
    die(f"{label}: Newton did not stop within {MAX_ITER} iterations")


# Design c: the outcomes separate along v = c2 - c1, so moving beta along v
# (binomial: +v; poisson: -v) raises every row's term or leaves it unchanged,
# and raises at least one, all along the ray, so no finite beta maximizes the
# log-likelihood.
def check_separates(family, c1, c2, y, label):
    v = [b - a for a, b in zip(c1, c2)]
    if family == "binomial":
        ok = all((vi > 0 and yi == 1) or (vi < 0 and yi == 0) or vi == 0 for vi, yi in zip(v, y))
    elif family == "poisson":
        ok = all(vi == 0 or (vi > 0 and yi == 0) for vi, yi in zip(v, y))
    else:
        ok = False
    if not ok or all(vi == 0 for vi in v):
        die(f"{label}: the outcomes do not separate along c2 - c1")


def fit_one(entry, rows, fit):
    label = f"{entry['name']} {fit['label']}"
    m = FORMULA.match(fit["r_formula"])
    if not m:
        die(f"{label}: r_formula {fit['r_formula']!r} is not y ~ 1 + c1 + c2 [+ offset(o)]")
    c1, c2, oname = m.groups()
    y = column(rows, "y")
    x1, x2 = column(rows, c1), column(rows, c2)
    out = {"label": fit["label"], "r_formula": fit["r_formula"]}
    if entry["design"] == "c":
        check_separates(entry["family"], x1, x2, y, label)
        out["finite_mle"] = False
        return out
    n = len(rows)
    w = column(rows, entry["weights"]) if entry["weights"] is not None else [mp.mpf(1)] * n
    off = column(rows, oname) if oname else [mp.mpf(0)] * n
    X = [[mp.mpf(1), a, b] for a, b in zip(x1, x2)]
    terms = row_terms(entry["family"], entry["link"])
    beta = [mp.mpf(0)] * 3
    if entry["family"] == "gamma":
        beta[0] = mp.log(sum(y) / n)
    beta, iterations = newton(terms, X, y, w, off, beta, label)
    _, score, _, G = eval_at(terms, X, y, w, off, beta)
    cov = mp.inverse(G)
    L = mp.cholesky(G)
    tau = min(L[j, j] ** 2 / G[j, j] for j in range(3))
    phi = mp.mpf(1)
    if entry["family"] == "gamma":
        resid = mp.mpf(0)
        for xi, yi, wi, oi in zip(X, y, w, off):
            mu = mp.exp(sum(xi[j] * beta[j] for j in range(3)) + oi)
            resid += wi * (yi - mu) ** 2 / mu ** 2
        phi = resid / (n - 3)
    se = [mp.sqrt(phi * cov[j, j]) for j in range(3)]
    out.update({
        "finite_mle": True,
        "beta_ref": [f64(b) for b in beta],
        "cov_ref": [[f64(cov[i, j]) for j in range(3)] for i in range(3)],
        "se_ref": [f64(s) for s in se],
        "z_ref": [f64(b / s) for b, s in zip(beta, se)],
        "tau_ref": f64(tau),
    })
    if entry["family"] == "gamma":
        out["phi_ref"] = f64(phi)
    out["iterations"] = iterations
    out["score_norm"] = f64(max(abs(score[j]) for j in range(3)))
    return out


def main():
    assert_mpmath_version()
    for entry in select_entries(KIND):
        rows = read_csv(entry["data"])
        fits = [fit_one(entry, rows, fit) for fit in entry["fits"]]
        write_reference({"provenance": provenance(), "fits": fits},
                        os.path.join(SUITE_DIR, "goldens", "inference", KIND,
                                     entry["name"] + ".json"))
        print(f"wrote {entry['name']} ({len(fits)} fits)")


if __name__ == "__main__":
    main()
