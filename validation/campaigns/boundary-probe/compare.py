#!/usr/bin/env python3
"""Join two arms of the boundary-probe measurement and report what moved.

    compare.py BEFORE.jsonl AFTER.jsonl       Markdown report on stdout
    compare.py --same A.jsonl B.jsonl         determinism check of two runs of one arm

Both inputs are `grid_fit` record files (one JSON object per fitted cell). The
two arms are joined on `case_id`; the report refuses (exit 1) when the two
files do not hold the same set of ids, or when either holds an id twice.

Fits are grouped by (corpus, route), by corpus over all routes, by route over
all corpora, and overall. The route is the site of
`config_tag` (`lmm-dense`, `lmm-sparse`, `glmm`, `glmm-sparse`), with `glmm`
split by the record's `nagq` into `glmm-laplace` and `glmm-agq`.

Per group: fits; failed fits per arm (no deviance); fits pinned in the before
arm (any `pinned` entry true); fits moved (any bit of `deviance`, or of a
`varcomp` stddev or correlation, differs); the deviance change after - before
(count below -1e-6 and -1e-3, min, max); fits worse by more than
64*eps*max(1, |dev|), listed by id; extra evaluations per before-pinned fit
(median, p90, max, and total after/before); flips of `converged`, `singular`
and of each `pinned` entry, by direction. When the after arm carries the
`post_pin_searches` counter (optional, absent from builds without it), also
its histogram.

`--same` checks that two runs of the same build wrote the same records in the
same order, ignoring only the timing fields (`wall_seconds`, `lower_seconds`);
exit 0 when identical, 1 otherwise.

Standard library only.
"""

import json
import math
import statistics
import struct
import sys
from collections import Counter, defaultdict

EPS = 2.0**-52
NOISE_ULPS = 64.0
TIMING_KEYS = ("wall_seconds", "lower_seconds")


def load(path):
    records = {}
    order = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec["case_id"]
            if cid in records:
                sys.exit(f"{path}:{n}: case_id {cid} appears twice")
            records[cid] = rec
            order.append(cid)
    return records, order


def route(rec):
    site = rec.get("config_tag", "lower-fail").split(":")[0]
    if site == "glmm":
        return "glmm-agq" if (rec.get("nagq") or 1) > 1 else "glmm-laplace"
    return site


def bits(x):
    """A value with every float replaced by its bit pattern, for exact comparison
    (so -0.0 and 0.0 differ)."""
    if isinstance(x, float):
        return struct.pack("<d", x)
    if isinstance(x, (list, tuple)):
        return [bits(v) for v in x]
    return x


def theta_view(rec):
    return [[g.get("stddev"), g.get("corr")] for g in rec.get("varcomp") or []]


def moved(b, a):
    return bits(b.get("deviance")) != bits(a.get("deviance")) or bits(
        theta_view(b)
    ) != bits(theta_view(a))


def failed(rec):
    return rec.get("deviance") is None


def any_pinned(rec):
    return any(any(row) for row in rec.get("pinned") or [])


def pinned_flips(b, a):
    """Counts of pinned entries going false->true and true->false. An empty
    `pinned` means nothing pinned and takes the other arm's shape."""
    pb, pa = b.get("pinned") or [], a.get("pinned") or []
    if not pb and not pa:
        return 0, 0, False
    if not pb:
        pb = [[False] * len(row) for row in pa]
    if not pa:
        pa = [[False] * len(row) for row in pb]
    if [len(r) for r in pb] != [len(r) for r in pa]:
        return 0, 0, True
    up = down = 0
    for rb, ra in zip(pb, pa):
        for xb, xa in zip(rb, ra):
            up += (not xb) and xa
            down += xb and (not xa)
    return up, down, False


def tol_of(dev):
    return NOISE_ULPS * EPS * max(1.0, abs(dev))


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, int):
        return str(x)
    if x == 0:
        return "0"
    return f"{x:.3g}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def same(path_a, path_b):
    a, order_a = load(path_a)
    b, order_b = load(path_b)
    if order_a != order_b:
        print(
            f"record order or id set differs: {len(order_a)} vs {len(order_b)} records"
        )
        return 1

    def strip(rec):
        return {k: v for k, v in rec.items() if k not in TIMING_KEYS}

    differ = [
        cid
        for cid in order_a
        if json.dumps(strip(a[cid]), sort_keys=True)
        != json.dumps(strip(b[cid]), sort_keys=True)
    ]
    print(
        f"{len(order_a)} records, {len(differ)} differ outside {', '.join(TIMING_KEYS)}"
    )
    for cid in differ[:20]:
        print(f"  {cid}")
    return 1 if differ else 0


def compare(path_before, path_after):
    before, _ = load(path_before)
    after, _ = load(path_after)
    only_b = sorted(set(before) - set(after))
    only_a = sorted(set(after) - set(before))
    if only_b or only_a:
        print(
            f"id sets differ: {len(only_b)} only in {path_before}, {len(only_a)} only in {path_after}"
        )
        for cid in (only_b + only_a)[:20]:
            print(f"  {cid}")
        return 1

    # Leaf groups are (corpus, route); "all" in either place aggregates over it.
    # Only leaf groups feed the id lists below, so no fit is listed twice.
    groups = defaultdict(list)
    for cid, b in before.items():
        corpus, rt = b.get("corpus", "-"), route(b)
        for key in ((corpus, rt), (corpus, "all"), ("all", rt), ("all", "all")):
            groups[key].append(cid)
    leaves = sorted(k for k in groups if "all" not in k)
    keys = (
        leaves
        + sorted(k for k in groups if k[0] != "all" and k[1] == "all")
        + sorted(k for k in groups if k[0] == "all" and k[1] != "all")
        + [("all", "all")]
    )

    has_counters = any("post_pin_searches" in r for r in after.values())
    main_rows, eval_rows, flip_rows, ctr_rows = [], [], [], []
    worse, improved, route_changes = [], [], []

    for key in keys:
        ids = groups[key]
        leaf = "all" not in key
        n_pinned = n_moved = n_worse = shape_mismatch = 0
        fail_b = sum(failed(before[c]) for c in ids)
        fail_a = sum(failed(after[c]) for c in ids)
        ddev = []
        extra, ev_b, ev_a = [], 0, 0
        flips = Counter()
        pps = Counter()
        for cid in ids:
            b, a = before[cid], after[cid]
            if leaf and route(a) != route(b):
                route_changes.append((cid, route(b), route(a)))
            pinned_b = any_pinned(b)
            n_pinned += pinned_b
            n_moved += moved(b, a)
            db, da = b.get("deviance"), a.get("deviance")
            if db is not None and da is not None:
                d = da - db
                ddev.append(d)
                n_worse += d > tol_of(db)
                if leaf:
                    if d > tol_of(db):
                        worse.append((cid, key, d))
                    elif d < 0:
                        improved.append((cid, key, d))
            if pinned_b and b.get("n_eval") is not None and a.get("n_eval") is not None:
                extra.append(a["n_eval"] - b["n_eval"])
                ev_b += b["n_eval"]
                ev_a += a["n_eval"]
            for flag in ("converged", "singular"):
                xb, xa = bool(b.get(flag)), bool(a.get(flag))
                if xb != xa:
                    flips[f"{flag} {'T>F' if xb else 'F>T'}"] += 1
            up, down, mismatch = pinned_flips(b, a)
            flips["pinned F>T"] += up
            flips["pinned T>F"] += down
            shape_mismatch += mismatch
            if has_counters and "post_pin_searches" in a:
                pps[a["post_pin_searches"]] += 1
        corpus, rt = key
        main_rows.append(
            [
                corpus,
                rt,
                len(ids),
                fail_b,
                fail_a,
                n_pinned,
                n_moved,
                sum(d < -1e-6 for d in ddev),
                sum(d < -1e-3 for d in ddev),
                fmt(min(ddev)) if ddev else "-",
                fmt(max(ddev)) if ddev else "-",
                n_worse,
            ]
        )
        eval_rows.append(
            [
                corpus,
                rt,
                len(extra),
                fmt(statistics.median(extra)) if extra else "-",
                fmt(quantile(extra, 0.9)) if extra else "-",
                fmt(max(extra)) if extra else "-",
                fmt(ev_a / ev_b) if ev_b else "-",
            ]
        )
        flip_rows.append(
            [
                corpus,
                rt,
                flips["converged T>F"],
                flips["converged F>T"],
                flips["singular T>F"],
                flips["singular F>T"],
                flips["pinned F>T"],
                flips["pinned T>F"],
                shape_mismatch,
            ]
        )
        if has_counters:
            hist = ", ".join(f"{k}: {v}" for k, v in sorted(pps.items())) or "-"
            ctr_rows.append([corpus, rt, hist])

    print(f"# Boundary probe: {path_before} -> {path_after}\n")
    print("## Fits, pins, moves and deviance change (after - before)\n")
    print(
        table(
            [
                "corpus",
                "route",
                "fits",
                "failed before",
                "failed after",
                "pinned before",
                "moved",
                "Δdev < -1e-6",
                "Δdev < -1e-3",
                "Δdev min",
                "Δdev max",
                "worse",
            ],
            main_rows,
        )
    )
    print("\n## Extra evaluations per fit pinned in the before arm\n")
    print(
        table(
            [
                "corpus",
                "route",
                "pinned fits",
                "median",
                "p90",
                "max",
                "total after/before",
            ],
            eval_rows,
        )
    )
    print("\n## Flag flips (before > after)\n")
    print(
        table(
            [
                "corpus",
                "route",
                "converged T>F",
                "converged F>T",
                "singular T>F",
                "singular F>T",
                "pinned entries F>T",
                "pinned entries T>F",
                "pinned shape differs",
            ],
            flip_rows,
        )
    )
    if has_counters:
        print("\n## Post-pin searches (after arm)\n")
        print(table(["corpus", "route", "post_pin_searches histogram"], ctr_rows))
    print(f"\n## Worse by more than {NOISE_ULPS:g}·eps·max(1, |dev|): {len(worse)}\n")
    for cid, (corpus, rt), d in sorted(worse, key=lambda w: -w[2]):
        print(f"- {cid} ({corpus}, {rt}): Δdev {d:+.3g}")
    print("\n## Five largest improvements\n")
    for cid, (corpus, rt), d in sorted(improved, key=lambda w: w[2])[:5]:
        print(f"- {cid} ({corpus}, {rt}): Δdev {d:+.3g}")
    if route_changes:
        print(f"\n## Route differs between arms: {len(route_changes)}\n")
        for cid, rb, ra in route_changes[:20]:
            print(f"- {cid}: {rb} -> {ra}")
    return 0


def main(argv):
    if len(argv) == 4 and argv[1] == "--same":
        return same(argv[2], argv[3])
    if len(argv) == 3:
        return compare(argv[1], argv[2])
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
