"""Recipe 5 — Poisson GLMM (grouseticks).

Tick counts per chick, with a per-brood random intercept: chicks in the same
brood share unmeasured causes of infestation (nest site, parent condition)
that a fixed effect alone cannot capture.

Data: validation/data/empirical/grouseticks.csv (the lme4 `grouseticks`
dataset, frozen for the validation harness).

No manifest rung behind this exact formula: rung 6 (grouseticks) fits the
centered `cHEIGHT` against all three crossed grouping factors (`BROOD`,
`INDEX`, `LOCATION`) together, a different model from this recipe's single
`(1 | BROOD)` on raw `HEIGHT` — dropping two grouping factors and swapping
the height variable changes what each variance component absorbs, so the
golden numbers for rung 6 are not a valid comparison target for this fit.
This recipe's output is a run, not an oracle-pinned result.
"""

import csv
from pathlib import Path

import glmm

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "empirical" / "grouseticks.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

data = {
    "TICKS": [float(r["TICKS"]) for r in rows],
    "YEAR": [r["YEAR"] for r in rows],
    "HEIGHT": [float(r["HEIGHT"]) for r in rows],
    "BROOD": [r["BROOD"] for r in rows],
}

fit = glmm.fit(data, "TICKS ~ YEAR + HEIGHT + (1 | BROOD)", family="poisson")

print("converged:", fit.converged, " singular:", fit.singular)
fit.summary()

sd, _corr = fit.stddev_corr(0)
print("BROOD stddev:", sd[0])
print("loglik:", fit.loglik)
print("\n(no manifest rung matches this formula -- a run, not an oracle-pinned result)")
