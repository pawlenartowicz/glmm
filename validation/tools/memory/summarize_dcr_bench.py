#!/usr/bin/env python3
"""The tables of the weight-power and CR2 benchmark report from dcr_bench.sh's
runs.tsv, as Markdown on stdout.

    python summarize_dcr_bench.py <out_dir>

A cell is the minimum over runs 1 to 3, or run 0 when it is the only run (a
run over 60 s). Fit cost is wall time and peak RSS minus arm 0 (packages
loaded, data read, no fit). The delta search is arm 2 minus arm 1, CR2 arm 3
minus arm 2, the per-call copy arm 3c minus arm 3; arm 3c also prints the
time of its one robust contrast_test, of which the last table takes the minimum.
"""

import csv
import glob
import re
import sys


def cells(path):
    runs = {}
    order = []
    status = {}
    with open(path) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        key = (r["dataset"], r["arm"])
        if r["dataset"] not in order:
            order.append(r["dataset"])
        if r["status"] != "ok":
            status[key] = r["status"]
            continue
        runs.setdefault(key, []).append(
            (int(r["rep"]), float(r["wall_s"]), int(r["peak_kb"]))
        )
    best = {}
    for key, rs in runs.items():
        timed = [x for x in rs if x[0] > 0] or rs
        best[key] = (min(x[1] for x in timed), min(x[2] for x in timed))
    return order, best, status


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    order, best, status = cells(f"{sys.argv[1]}/runs.tsv")

    def get(d, arm):
        if (d, arm) in best:
            return best[(d, arm)]
        return status.get((d, arm))

    def fit(d, arm, i):
        v, base = get(d, arm), get(d, "0")
        if v is None:
            return ""
        if not isinstance(v, tuple) or not isinstance(base, tuple):
            return v if isinstance(v, str) else ""
        return f"{v[i] - base[i]:.2f}" if i == 0 else f"{(v[i] - base[i]) / 1024:.0f}"

    def diff(d, a, b, i):
        va, vb = get(d, a), get(d, b)
        if not isinstance(va, tuple) or not isinstance(vb, tuple):
            return ""
        return f"{va[i] - vb[i]:.2f}" if i == 0 else f"{(va[i] - vb[i]) / 1024:.0f}"

    for i, title in ((0, "Wall time, seconds"), (1, "Peak RSS, MB")):
        print(f"### {title}, fit cost (arm 0 subtracted)\n")
        print(
            "| Dataset | arm 0 (load, read) | 1 claimed | 2 delta | 3 delta + robust | 4 lme + clubSandwich "
            "| delta search (2 - 1) | CR2 (3 - 2) | per-call copy (3c - 3) |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
        for d in order:
            base = get(d, "0")
            b = (
                (f"{base[0]:.2f}" if i == 0 else f"{base[1] / 1024:.0f}")
                if isinstance(base, tuple)
                else (base or "")
            )
            row = [
                d,
                b,
                *(fit(d, a, i) for a in ("1", "2", "3", "4")),
                diff(d, "2", "1", i),
                diff(d, "3", "2", i),
                diff(d, "3c", "3", i),
            ]
            print("| " + " | ".join(row) + " |")
        print()
    print("### One robust contrast_test, seconds\n")
    print("| Dataset | contrast_test |")
    print("|---|---|")
    for d in order:
        times = []
        for f in glob.glob(f"{sys.argv[1]}/{d}.3c.*.log"):
            with open(f) as fh:
                times += [
                    float(m)
                    for m in re.findall(r"contrast_test ([0-9.]+) s", fh.read())
                ]
        if times:
            print(f"| {d} | {min(times):.3f} |")


if __name__ == "__main__":
    main()
