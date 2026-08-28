"""Recipe 3 — nested grouping factors (Pastes).

`cask` only means something inside its `batch` — cask "a" of batch "A" is
unrelated to cask "a" of batch "B" — so the model is `(1 | batch/cask)`, R's
nesting shorthand for "a random intercept per batch, plus a random intercept
per batch:cask combination". Two variance components come out: one for the
coarse `batch` grouping (10 levels), one for the finer `batch:cask` grouping
(30 levels, one per cask actually observed within its batch).

Data: validation/data/empirical/Pastes.csv (the lme4 `Pastes` dataset, frozen
for the validation harness). Pastes also carries a `sample` column that is
just the `batch:cask` label spelled out (`"A:a"`); `batch/cask` and the
`(1|batch)+(1|sample)` form used elsewhere in the manifest fit the same model.

Rung: manifest.json rung 4 / goldens/pastes_lmm.json.
"""

import csv
from pathlib import Path

import glmm
from _oracle import TOL_BETA_REL, TOL_LOGLIK_ABS_LMM, TOL_SE_REL, TOL_STDDEV_REL, check_abs, check_rel, load_golden

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "empirical" / "Pastes.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

data = {
    "strength": [float(r["strength"]) for r in rows],
    "batch": [r["batch"] for r in rows],
    "cask": [r["cask"] for r in rows],
}

# NOTE: no bare "1" fixed-effect term (the intercept is always implicit) --
# the manifest's "strength ~ 1 + (1 | batch/cask)" drops the leading "1 +".
fit = glmm.fit(data, "strength ~ (1 | batch/cask)")

print("converged:", fit.converged, " singular:", fit.singular)
fit.summary()

print("loglik (REML crit):", fit.loglik)

print("\noracle cross-check vs goldens/pastes_lmm.json (manifest rung 4):")
g = load_golden("pastes_lmm")
est = g["estimates"]
check_rel("beta[Intercept]", fit.beta[0], est["beta"][0], TOL_BETA_REL)
check_rel("se[Intercept]", fit.se[0], est["se"][0], TOL_SE_REL)
# The golden names the nested grouping "cask:batch"; this port names the same
# grouping "batch:cask" -- same set of factors, order differs. Match on the
# sorted factor set rather than the literal string.
def _key(name):
    return frozenset(name.split(":"))


golden_by_group = {_key(v["group"]): v["stddev"][0] for v in est["varcomp"]}
for i, (group_name, _terms) in enumerate(fit.re_groups):
    sd, _corr = fit.stddev_corr(i)
    check_rel(f"{group_name} stddev", sd[0], golden_by_group[_key(group_name)], TOL_STDDEV_REL)
check_abs("loglik (REML crit)", fit.loglik, est["loglik"], TOL_LOGLIK_ABS_LMM)
