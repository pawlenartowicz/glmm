"""Recipe 7 — factors and interactions (cake), and changing the base level.

`recipe*temp` desugars to `recipe + temp + recipe:temp`: a main effect per
recipe, a slope on `temp`, and a per-recipe deviation from that slope.
Treatment contrasts code `recipe` against its first level (alphabetically
"A", since a plain string column has no declared order) — every coefficient
named `recipeB`/`recipeC` is a *contrast against A*, not an independent
effect. `pandas.Categorical`'s declared category order lets you pick a
different base without relabeling the data.

Data: validation/data/empirical/cake.csv (the lme4 `cake` dataset, frozen for
the validation harness).

No manifest rung behind this formula — `cake` does not appear in
validation/goldens/ at all (manifest rung 13 names it, but no lme4 reference
JSON was frozen there). This recipe's output is a run, not an oracle-pinned
result.
"""

import csv
from pathlib import Path

import glmm
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "empirical" / "cake.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

angle = [float(r["angle"]) for r in rows]
temp = [float(r["temp"]) for r in rows]
recipe_labels = [r["recipe"] for r in rows]
replicate = [r["replicate"] for r in rows]

# Default: no declared order -> lexicographic -> base = "A".
data = {
    "angle": angle,
    "recipe": recipe_labels,
    "temp": temp,
    "replicate": replicate,
}
fit_a = glmm.fit(data, "angle ~ recipe*temp + (1 | recipe:replicate)")

print("=== base = A (default, no declared order) ===")
fit_a.summary()

# Same model, base = "B": a pandas.Categorical with "B" listed first.
data_b = dict(data)
data_b["recipe"] = pd.Categorical(recipe_labels, categories=["B", "A", "C"])
fit_b = glmm.fit(data_b, "angle ~ recipe*temp + (1 | recipe:replicate)")

print("\n=== base = B (pandas.Categorical(categories=['B', 'A', 'C'])) ===")
fit_b.summary()

print(
    "\nSame fit, different parameterization: fitted values and loglik agree "
    "(loglik A={:.10g}, loglik B={:.10g}, delta={:.3g}); only which contrasts "
    "are directly readable off beta changes.".format(
        fit_a.loglik, fit_b.loglik, fit_b.loglik - fit_a.loglik
    )
)
print("\n(cake carries no goldens/ entry -- a run, not an oracle-pinned result)")
