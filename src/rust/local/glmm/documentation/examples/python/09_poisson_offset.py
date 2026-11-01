"""Recipe 9 — offset (sim_poisson_offset).

A Poisson rate model against a known exposure: each row's expected count
scales with `exposure`, so `offset = log(exposure)` is added to the linear
predictor with a fixed coefficient of 1 rather than an estimated one. Drop
the offset and the model instead treats every row as equally exposed, which
folds the (here, substantial) exposure variation into the fixed effect and
the random-effect variance instead of explaining it away.

Data: validation/data/simulated/sim_poisson_offset.csv (a fixture generated
for the validation harness with a `log_exposure` column already computed —
`offset=` takes the log-exposure directly, not the raw exposure).

No manifest rung's goldens/ entry covers this dataset (it is registered at
manifest rung 28 with a real `offset` field, but no lme4 reference JSON was
frozen for it under validation/goldens/) — this recipe's output is a run,
not an oracle-pinned result.
"""

import csv
from pathlib import Path

import glmm

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "simulated" / "sim_poisson_offset.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

data = {
    "y": [float(r["y"]) for r in rows],
    "x": [float(r["x"]) for r in rows],
    "cluster": [r["cluster"] for r in rows],
}
log_exposure = [float(r["log_exposure"]) for r in rows]

fit_with = glmm.fit(data, "y ~ x + (1 | cluster)", family="poisson", offset=log_exposure)
fit_without = glmm.fit(data, "y ~ x + (1 | cluster)", family="poisson")

print("=== with offset = log(exposure) ===")
fit_with.summary()
sd_with, _ = fit_with.stddev_corr(0)
print("cluster stddev:", sd_with[0])

print("\n=== without the offset (exposure variation folded into the fit) ===")
fit_without.summary()
sd_without, _ = fit_without.stddev_corr(0)
print("cluster stddev:", sd_without[0])

print(
    "\nDropping a real offset does not just bias the intercept: here the slope "
    f"on x moves from {fit_with.beta[1]:.4g} (with offset) to {fit_without.beta[1]:.4g} "
    f"(without), and the cluster standard deviation moves from {sd_with[0]:.4g} to "
    f"{sd_without[0]:.4g} -- the unexplained exposure variation is absorbed by "
    "both the fixed and the random-effect side, not cleanly by either alone."
)
print("\n(no goldens/ entry for sim_poisson_offset -- a run, not an oracle-pinned result)")
