"""Recipe 4 — aggregated binomial via `weights=` (cbpp).

lme4 writes an aggregated binomial as `cbind(incidence, size - incidence) ~
...`. The shared formula parser also accepts `cbind()` directly with
`family="binomial"`, but both arguments must be columns — compute the
failures column first (`failures = size - incidence`; arithmetic inside
`cbind()` itself is not accepted) and pass `cbind(incidence, failures) ~
...`. This recipe instead spells the equivalent model as the success
*proportion* as the response plus the trial count as `weights=` — exactly
lme4's own objective underneath `cbind()`, just spelled differently.

Data: validation/data/empirical/cbpp.csv (the lme4 `cbpp` dataset, frozen for
the validation harness).

Rung: manifest.json rung 5 / goldens/cbpp_agq_k1.json (nagq=1 is the Laplace
default this recipe fits — the same model the AGQ study calls its k=1 case).
"""

import csv
from pathlib import Path

import glmm
from _oracle import TOL_BETA_REL, TOL_LOGLIK_ABS_GLMM, TOL_SE_HESSIAN_REL, TOL_STDDEV_REL, check_abs, check_rel, load_golden

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "empirical" / "cbpp.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

incidence = [float(r["incidence"]) for r in rows]
size = [float(r["size"]) for r in rows]
data = {
    "prop": [i / s for i, s in zip(incidence, size)],
    "period": [r["period"] for r in rows],
    "herd": [r["herd"] for r in rows],
}

fit = glmm.fit(data, "prop ~ period + (1 | herd)", family="binomial", weights=size)

print("converged:", fit.converged, " singular:", fit.singular)
fit.summary()

sd, _corr = fit.stddev_corr(0)
print("herd stddev:", sd[0])
print("loglik:", fit.loglik)

print("\noracle cross-check vs goldens/cbpp_agq_k1.json (manifest rung 5, nagq=1):")
g = load_golden("cbpp_agq_k1")
est = g["estimates"]
for i, name in enumerate(g["coef_names"]):
    check_rel(f"beta[{name}]", fit.beta[i], est["beta"][i], TOL_BETA_REL)
for i, name in enumerate(g["coef_names"]):
    check_rel(f"se_hessian[{name}]", fit.se[i], est["se_hessian"][i], TOL_SE_HESSIAN_REL)
check_rel("herd stddev", sd[0], est["varcomp"][0]["stddev"][0], TOL_STDDEV_REL)
check_abs("loglik", fit.loglik, est["loglik"], TOL_LOGLIK_ABS_GLMM)
