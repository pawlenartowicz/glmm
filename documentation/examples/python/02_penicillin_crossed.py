"""Recipe 2 — crossed grouping factors (Penicillin).

`plate` and `sample` are crossed, not nested: every sample was tested on every
plate, so neither grouping factor is a subdivision of the other. Two
independent `(1 | g)` terms, not a `/` nesting operator.

Data: validation/data/empirical/Penicillin.csv (the lme4 `Penicillin`
dataset, frozen for the validation harness).

Rung: manifest.json rung 3 / goldens/penicillin_lmm.json.
"""

import csv
from pathlib import Path

import glmm
from _oracle import TOL_BETA_REL, TOL_LOGLIK_ABS_LMM, TOL_SE_REL, TOL_STDDEV_REL, check_abs, check_rel, load_golden

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "empirical" / "Penicillin.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

data = {
    "diameter": [float(r["diameter"]) for r in rows],
    "plate": [r["plate"] for r in rows],
    "sample": [r["sample"] for r in rows],
}

# NOTE: the shared formula parser always carries an intercept implicitly and
# has no bare "1" fixed-effect term (see documentation/formula.md) -- the
# manifest's lme4-style "diameter ~ 1 + (1 | plate) + (1 | sample)" drops the
# leading "1 +" here.
fit = glmm.fit(data, "diameter ~ (1 | plate) + (1 | sample)")

print("converged:", fit.converged, " singular:", fit.singular)
fit.summary()

for i, (group_name, _terms) in enumerate(fit.re_groups):
    sd, _corr = fit.stddev_corr(i)
    print(f"{group_name} stddev: {sd[0]:.10g}")

print("loglik (REML crit):", fit.loglik, " reml:", fit.reml)

print("\noracle cross-check vs goldens/penicillin_lmm.json (manifest rung 3):")
g = load_golden("penicillin_lmm")
est = g["estimates"]
check_rel("beta[Intercept]", fit.beta[0], est["beta"][0], TOL_BETA_REL)
check_rel("se[Intercept]", fit.se[0], est["se"][0], TOL_SE_REL)
# re_groups order need not match the golden's varcomp order, so match by name.
golden_by_group = {v["group"]: v["stddev"][0] for v in est["varcomp"]}
for i, (group_name, _terms) in enumerate(fit.re_groups):
    sd, _corr = fit.stddev_corr(i)
    check_rel(f"{group_name} stddev", sd[0], golden_by_group[group_name], TOL_STDDEV_REL)
check_abs("loglik (REML crit)", fit.loglik, est["loglik"], TOL_LOGLIK_ABS_LMM)
