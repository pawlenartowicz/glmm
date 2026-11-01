"""Recipe 1 — correlated random slope (sleepstudy).

Data: validation/data/empirical/sleepstudy.csv (the lme4 `sleepstudy` dataset,
frozen in this repo for the validation harness; read here by relative path
rather than bundling a second copy).

Rung: manifest.json rung 2 / goldens/sleepstudy_lmm.json.
"""

import csv
from pathlib import Path

import glmm
from _oracle import TOL_BETA_REL, TOL_LOGLIK_ABS_LMM, TOL_SE_REL, TOL_STDDEV_REL, check_abs, check_rel, load_golden

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "empirical" / "sleepstudy.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

data = {
    "Reaction": [float(r["Reaction"]) for r in rows],
    "Days": [float(r["Days"]) for r in rows],
    "Subject": [r["Subject"] for r in rows],
}

fit = glmm.fit(data, "Reaction ~ Days + (1 + Days | Subject)")

print("converged:", fit.converged, " singular:", fit.singular)
fit.summary()

sd, corr = fit.stddev_corr(0)
print("Subject stddevs:", sd)
print("Subject corr:", corr.tolist())
print("loglik (REML crit):", fit.loglik, " reml:", fit.reml)
print("dispersion (residual sigma^2):", fit.dispersion)

print("\noracle cross-check vs goldens/sleepstudy_lmm.json (manifest rung 2):")
g = load_golden("sleepstudy_lmm")
est = g["estimates"]
check_rel("beta[Intercept]", fit.beta[0], est["beta"][0], TOL_BETA_REL)
check_rel("beta[Days]", fit.beta[1], est["beta"][1], TOL_BETA_REL)
check_rel("se[Intercept]", fit.se[0], est["se"][0], TOL_SE_REL)
check_rel("se[Days]", fit.se[1], est["se"][1], TOL_SE_REL)
check_rel("Subject sd[Intercept]", sd[0], est["varcomp"][0]["stddev"][0], TOL_STDDEV_REL)
check_rel("Subject sd[Days]", sd[1], est["varcomp"][0]["stddev"][1], TOL_STDDEV_REL)
check_abs("loglik (REML crit)", fit.loglik, est["loglik"], TOL_LOGLIK_ABS_LMM)
