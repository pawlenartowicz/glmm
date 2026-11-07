#!/usr/bin/env python3
"""Generate one dataset of the weight-power and CR2 benchmark (spec 2-dcr §12)
as a CSV. Not a validation rung: nothing is checked against an oracle, only
wall time and peak RSS are measured (dcr_bench.sh).

    python gen_dcr_bench.py <id> <out.csv>     write dataset <id>
    python gen_dcr_bench.py --list            print "<id> <row> <formula>" per dataset, in run order

Weights are log-uniform at spread 100 inside each cluster, and the noise has
variance w^(-0.5), so the fitted power is away from 0 and 1. Fixed seed per
dataset (its index in DATASETS).
"""

import sys

import numpy as np

SMALL_G = (1_000, 10_000, 100_000)
LARGE_NK = (200, 1_000, 2_000, 3_000, 5_000, 20_000, 100_000)
HC2_N = (10_000, 100_000, 1_000_000)


def _datasets():
    """(id, row, design) in run order: the small-cluster rows first (§12).
    A row is the unit a capped run skips the rest of, its sizes ascending."""
    out = []
    for g in SMALL_G:
        out.append(
            (f"small-G{g}", "small", {"g": g, "nk": None, "p": 2, "slope": False})
        )
    for p, slope in ((10, False), (2, True), (10, True)):
        tag = f"p{p}-{'slope' if slope else 'int'}"
        out.append(
            (
                f"wide-small-{tag}",
                f"wide-small-{tag}",
                {"g": 100_000, "nk": None, "p": p, "slope": slope},
            )
        )
    for nk in LARGE_NK:
        out.append(
            (f"large-nk{nk}", "large", {"g": 10, "nk": nk, "p": 2, "slope": False})
        )
    for p, slope in ((10, False), (2, True), (10, True)):
        tag = f"p{p}-{'slope' if slope else 'int'}"
        out.append(
            (
                f"wide-large-{tag}",
                f"wide-large-{tag}",
                {"g": 10, "nk": 20_000, "p": p, "slope": slope},
            )
        )
    for n in HC2_N:
        out.append((f"hc2-n{n}", "hc2", {"g": 0, "nk": n, "p": 2, "slope": False}))
    return out


DATASETS = _datasets()


def formula(d):
    xs = " + ".join(f"x{j}" for j in range(1, d["p"]))
    if d["g"] == 0:
        return f"y ~ {xs}"
    return f"y ~ {xs} + ({'1 + x1' if d['slope'] else '1'} | g)"


def generate(index, d):
    rng = np.random.default_rng(index)
    if d["g"] == 0:
        ids = np.zeros(d["nk"], dtype=np.int64)
    elif d["nk"] is None:
        sizes = rng.integers(2, 21, size=d["g"])  # n_k from 2 to 20
        ids = np.repeat(np.arange(d["g"]), sizes)
    else:
        ids = np.repeat(np.arange(d["g"]), d["nk"])
    n = ids.size
    x = rng.normal(size=(n, d["p"] - 1))
    w = np.exp(rng.uniform(0.0, np.log(100.0), size=n))
    eta = 0.5 + x @ np.full(d["p"] - 1, 0.3)
    if d["g"] > 0:
        eta += rng.normal(scale=0.5, size=d["g"])[ids]
        if d["slope"]:
            eta += rng.normal(scale=0.3, size=d["g"])[ids] * x[:, 0]
    y = eta + rng.normal(size=n) * w**-0.25
    cols = {"y": y, **{f"x{j + 1}": x[:, j] for j in range(d["p"] - 1)}, "w": w}
    if d["g"] > 0:
        cols["g"] = ids
    return cols


def main():
    if sys.argv[1:] == ["--list"]:
        for id_, row, d in DATASETS:
            print(f"{id_}\t{row}\t{formula(d)}")
        return
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    index, (id_, _, d) = next(
        (i, e) for i, e in enumerate(DATASETS) if e[0] == sys.argv[1]
    )
    cols = generate(index, d)
    names = list(cols)
    data = np.column_stack([cols[c] for c in names])
    fmt = ["%d" if c == "g" else "%.17g" for c in names]
    np.savetxt(
        sys.argv[2], data, fmt=fmt, delimiter=",", header=",".join(names), comments=""
    )
    print(f"{id_}: n={data.shape[0]} -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
