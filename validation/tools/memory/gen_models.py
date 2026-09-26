#!/usr/bin/env python3
"""Generate one of the 13 large synthetic memory-measurement models (models.json)
as a CSV. These are NOT validation rungs -- no oracle output is checked against
this data, only peak RSS while an engine fits it (see models.json's header).

    python gen_models.py <id> <out.csv> [--n N]

<id> selects a row from models.json; --n overrides that row's "n" (used by the
harness's smoke test to run a large shape at a fraction of its real row count --
the group/level counts, and so k_total, are unchanged by the override, only the
row count and therefore memory shrink).

Fixed seed per model (rng seeded on the model id, mirroring
toy28/ram_scaling.py's `np.random.default_rng(0)` precedent) -- deterministic,
reproducible, and irrelevant to what is being measured (peak RSS depends on
shape, not on the values).
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(model_id):
    with open(os.path.join(HERE, "models.json")) as fh:
        spec = json.load(fh)
    for m in spec["models"]:
        if m["id"] == model_id:
            return m
    raise SystemExit(f"no model id {model_id} in models.json")


def assign_crossed(rng, n, levels):
    """Round-robin group assignment: every level is hit regardless of n, so an
    --n override shrinks rows without shrinking the level count (the axis this
    set measures)."""
    return np.arange(n) % levels


def gen_rows(model, n):
    rng = np.random.default_rng(model["id"])
    cols = {}

    for p in model["predictors"]:
        cols[p] = rng.normal(size=n)

    eta = -0.2 + 0.3 * sum(cols[p] for p in model["predictors"])

    for g in model["groups"]:
        kind = g["kind"]
        if kind == "crossed":
            ids = assign_crossed(rng, n, g["levels"])
            cols[g["name"]] = np.array([f"{g['name']}{i:06d}" for i in ids])
            b0 = rng.normal(scale=0.5, size=g["levels"])
            eta = eta + b0[ids]
            for s in g["slopes"]:
                bs = rng.normal(scale=0.3, size=g["levels"])
                eta = eta + bs[ids] * cols[s]
        elif kind == "nested_parent":
            ids = assign_crossed(rng, n, g["levels"])
            cols[g["name"]] = np.array([f"{g['name']}{i:06d}" for i in ids])
            b0 = rng.normal(scale=0.5, size=g["levels"])
            eta = eta + b0[ids]
        elif kind == "nested_child":
            parent_levels = next(
                pg["levels"] for pg in model["groups"] if pg["name"] == g["parent"]
            )
            parent_ids = assign_crossed(rng, n, parent_levels)
            per_parent = g["per_parent"]
            # Child label is local to its parent (small alphabet, e.g. "0".."4")
            # -- the formula frontend composes it with the parent id into the
            # nested "parent:child" grouping internally (src/formula/materialize.rs),
            # the same convention lme4/glmm already use for a nested factor whose
            # raw labels repeat across parents (this repo's own Pastes rung nests
            # "cask" in "batch" the identical way).
            row_in_parent = np.arange(n) // parent_levels
            child_local = row_in_parent % per_parent
            cols[g["name"]] = np.array([f"c{i:03d}" for i in child_local])
            n_child_total = parent_levels * per_parent
            child_ids = parent_ids * per_parent + child_local
            b0 = rng.normal(scale=0.5, size=n_child_total)
            eta = eta + b0[child_ids]
        else:
            raise SystemExit(f"unknown group kind {kind}")

    if model["family"] == "gaussian":
        y = eta + rng.normal(scale=1.0, size=n)
    elif model["family"] == "binomial":
        p1 = 1.0 / (1.0 + np.exp(-eta))
        y = (rng.random(n) < p1).astype(float)
    else:
        raise SystemExit(f"unsupported family {model['family']}")
    cols["y"] = y
    return cols


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    model_id = int(sys.argv[1])
    out_path = sys.argv[2]
    n_override = None
    if "--n" in sys.argv:
        n_override = int(sys.argv[sys.argv.index("--n") + 1])

    model = load_model(model_id)
    n = n_override if n_override is not None else model["n"]
    cols = gen_rows(model, n)

    header = list(cols.keys())
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for i in range(n):
            fh.write(",".join(str(cols[c][i]) for c in header) + "\n")

    factor_names = [g["name"] for g in model["groups"]]
    print(f"model {model_id}: n={n} factors={factor_names} -> {out_path}")


if __name__ == "__main__":
    main()
