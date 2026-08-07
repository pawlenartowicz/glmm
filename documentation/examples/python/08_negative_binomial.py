"""Recipe 8 — negative binomial (sim_nb).

A negative-binomial GLM: `dispersion` on the returned `Fit` is theta-hat, the
NB shape parameter (`MASS::glm.nb`'s `theta`), not phi -- unlike gamma or
inverse-Gaussian, where `dispersion` is the Pearson phi. Overdispersion
relative to Poisson is 1/theta; a large theta means "close to Poisson", not
"a lot of extra variance".

Data: validation/data/simulated/sim_nb.csv (a fixture generated for the
validation harness, since no lme4-bundled dataset exercises negative
binomial).

Rung: manifest.json's m3_goldens entry `sim_nb_glm` / goldens/sim_nb_glm.json
-- a GLM (no random effects), which is the model this recipe fits.
"""

import csv
from pathlib import Path

import glmm
from _oracle import TOL_BETA_REL, TOL_LOGLIK_ABS_GLMM, TOL_SE_REL, check_abs, check_rel, load_golden

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "simulated" / "sim_nb.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

data = {
    "y": [float(r["y"]) for r in rows],
    "x": [float(r["x"]) for r in rows],
    "grp": [r["grp"] for r in rows],
}

fit = glmm.fit(data, "y ~ x + grp", family="negativebinomial")

print("converged:", fit.converged, " singular:", fit.singular)
fit.summary()
print("theta (NB shape):", fit.dispersion)
print("loglik:", fit.loglik)

print("\noracle cross-check vs goldens/sim_nb_glm.json (GLM, no random effects):")
g = load_golden("sim_nb_glm")
est = g["estimates"]
for i, name in enumerate(g["coef_names"]):
    check_rel(f"beta[{name}]", fit.beta[i], est["beta"][i], TOL_BETA_REL)
for i, name in enumerate(g["coef_names"]):
    check_rel(f"se[{name}]", fit.se[i], est["se"][i], TOL_SE_REL)
check_rel("theta", fit.dispersion, est["theta"], TOL_BETA_REL)
check_abs("loglik", fit.loglik, est["loglik"], TOL_LOGLIK_ABS_GLMM)
