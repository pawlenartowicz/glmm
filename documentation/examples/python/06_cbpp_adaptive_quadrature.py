"""Recipe 6 — adaptive quadrature (cbpp at nagq=7).

Recipe 4's model again, but integrated over the random effect with 7-point
adaptive Gauss-Hermite quadrature instead of the nagq=1 Laplace default.
Eligible here because the model has a single grouping factor (`herd`) with
one random effect per level (q=1) — AGQ's current cap is q<=3 on a single
binomial/Poisson grouping factor.

Data: validation/data/empirical/cbpp.csv (see recipe 4 for the `weights=`
spelling of lme4's `cbind()`).

Rung: manifest.json rung 5 at nagq=7 / goldens/cbpp_agq_k7.json (the same
study's k=1 case, goldens/cbpp_agq_k1.json, is recipe 4's fit). The
cross-check below compares beta, se_hessian and the herd standard deviation
only -- the crate's own oracle test for this exact golden
(`fit_glmm_binomial_agq_matches_lme4`, src/fit/glmm_tests.rs) gates the same
three quantities and deliberately not log-likelihood. That is not a gap: the
Laplace-vs-AGQ log-likelihood does not even agree with itself across `nAGQ`
in lme4's own output (lme4's `glmer` reports -92.0 at nAGQ=1 and -50.0 at
nAGQ=7 for this exact fit -- a ~42-unit jump despite beta moving by <0.001),
so log-likelihood is not the quantity nAGQ eligibility is about here; beta,
se and the variance component are.
"""

import csv
from pathlib import Path

import glmm
from _oracle import TOL_BETA_REL, TOL_SE_HESSIAN_REL, TOL_STDDEV_REL, check_rel, load_golden

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

fit_laplace = glmm.fit(data, "prop ~ period + (1 | herd)", family="binomial", weights=size, nagq=1)
fit_agq = glmm.fit(data, "prop ~ period + (1 | herd)", family="binomial", weights=size, nagq=7)

print("=== nagq=1 (Laplace) ===")
fit_laplace.summary()
print("loglik:", fit_laplace.loglik)

print("\n=== nagq=7 (adaptive Gauss-Hermite) ===")
fit_agq.summary()
print("loglik:", fit_agq.loglik)

print("\nLaplace -> AGQ(7) movement on this data:")
for i, name in enumerate(fit_agq.names):
    d = fit_agq.beta[i] - fit_laplace.beta[i]
    rel = abs(d) / abs(fit_laplace.beta[i])
    print(f"  beta[{name}]: laplace={fit_laplace.beta[i]:.10g} agq7={fit_agq.beta[i]:.10g} "
          f"delta={d:.3g} rel={rel:.3g}")
loglik_delta = fit_agq.loglik - fit_laplace.loglik
print(f"  loglik: laplace={fit_laplace.loglik:.10g} agq7={fit_agq.loglik:.10g} delta={loglik_delta:.3g}")

print("\noracle cross-check vs goldens/cbpp_agq_k7.json (manifest rung 5 at nagq=7):")
print("(beta, se_hessian, herd stddev only -- see module docstring on why loglik is excluded)")
g = load_golden("cbpp_agq_k7")
est = g["estimates"]
for i, name in enumerate(g["coef_names"]):
    check_rel(f"beta[{name}]", fit_agq.beta[i], est["beta"][i], TOL_BETA_REL)
for i, name in enumerate(g["coef_names"]):
    check_rel(f"se_hessian[{name}]", fit_agq.se[i], est["se_hessian"][i], TOL_SE_HESSIAN_REL)
sd, _corr = fit_agq.stddev_corr(0)
check_rel("herd stddev", sd[0], est["varcomp"][0]["stddev"][0], TOL_STDDEV_REL)
