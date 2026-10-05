#!/usr/bin/env python3
"""Summary and answer check for loop_bench JSONL output (Python 3 stdlib only).

Usage:
    loop_summary.py [options] PATH [PATH ...]

PATH is a loop_bench JSONL file or a directory searched recursively for
*.jsonl. Records are grouped by the arm named in each file's header record.
Warm-up draws (warmup: true) are left out of everything.

Sections:
  - per arm and block (shape, size, variant, start): fits, median [IQR] and
    minimum of wall and CPU time per fit, CPU against wall, instructions per
    fit, evals per fit, PIRLS iterations per eval, convergence and singular
    rates, the observed outer-search route;
  - process totals appended by run_loop.sh (`/usr/bin/time -l`), if present;
  - --base ARM: the answer check of every other arm against ARM on the same
    seed and draw: |objective - base| <= dev_eps (dev_eps_agq_vec on
    vector-AGQ blocks) and identical converged and singular flags;
  - --starts: the same check inside each arm, truth and carry starts against
    the cold start of the same draw;
  - --determinism: repeated (block, seed, draw) records inside one arm must
    carry bit-identical objectives and equal eval counts;
  - --sanity: cold-start mean beta-hat and theta-hat against the generating
    values.

Counter fields (cpu_ns, instr, cycles) are null when loop_bench was built
without its os_counters feature; their columns then print "-".

The exit status is 1 when any requested check fails, unless --no-fail.
"""
import argparse
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

# The accuracy grid's deviance bands (validation/grid/tol.R): dev_eps, and
# dev_eps_agq_vec for a vector-AGQ cell compared with an earlier glmm run whose
# design was standardized differently (the quadrature difference at the same
# optimum).
DEV_EPS = 4e-05
DEV_EPS_AGQ_VEC = 5e-04

BLOCK = ("shape", "size", "variant", "start")


def iter_files(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    if f.endswith(".jsonl"):
                        yield os.path.join(root, f)
        else:
            yield p


def load(paths):
    """Draw records per arm, block metadata, process records."""
    draws = defaultdict(list)  # arm -> [record]
    blocks = {}  # (arm, shape, size, variant) -> block record
    procs = defaultdict(list)  # (arm, block key) -> [process record]
    headers = defaultdict(list)
    for path in iter_files(paths):
        arm = None
        with open(path) as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    print(f"warning: {path}:{ln}: not JSON, skipped", file=sys.stderr)
                    continue
                t = r.get("type")
                if t == "header":
                    arm = r.get("arm", "unnamed")
                    headers[arm].append(r)
                elif t == "block":
                    blocks[(arm,) + tuple(r[k] for k in BLOCK[:3])] = r
                elif t == "draw":
                    if not r.get("warmup"):
                        r["_arm"] = arm
                        r["_file"] = path
                        draws[arm].append(r)
                elif t == "process":
                    procs[(r.get("arm", arm), tuple(r.get(k) for k in BLOCK))].append(r)
    return draws, blocks, procs, headers


def q3(v):
    v = sorted(v)
    if not v:
        return (math.nan,) * 3
    if len(v) < 2:
        return v[0], v[0], v[0]
    qs = st.quantiles(v, n=4, method="inclusive")
    return qs[1], qs[0], qs[2]


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def block_key(r):
    return tuple(r[k] for k in BLOCK)


def eps_for(blocks, arm, r):
    b = blocks.get((arm, r["shape"], r["size"], r["variant"]), {})
    return DEV_EPS_AGQ_VEC if b.get("agq_vec") else DEV_EPS


def summary(draws, blocks, procs, headers):
    for arm in sorted(draws):
        hs = headers.get(arm, [])
        h = hs[0] if hs else {}
        print(f"=== arm {arm}  source={h.get('source', '')!r}  features={h.get('features')}  "
              f"prestd={h.get('prestandardized')}  rustc={h.get('rustc', '')}")
        states = sorted({(x.get("power_source"), x.get("low_power_mode"), x.get("loadavg")) for x in hs})
        if states:
            print(f"    machine state over {len(hs)} process(es): power/LPM/load first={states[0]} last={states[-1]}")
        g = defaultdict(list)
        for r in draws[arm]:
            g[block_key(r)].append(r)
        print(f"{'shape':<17}{'size':<6}{'variant':<9}{'start':<6}{'fits':>5}"
              f"{'wall us med [IQR]':>32}{'min':>11}{'cpu us med':>11}{'cpu/wall':>9}"
              f"{'Minstr med':>11}{'evals med':>10}{'(mean)':>8}{'it/eval':>8}"
              f"{'conv':>6}{'sing':>6}  route")
        for k in sorted(g):
            rs = g[k]
            wall = [r["wall_ns"] / 1e3 for r in rs]
            cpu = [r["cpu_ns"] / 1e3 for r in rs if r.get("cpu_ns") is not None]
            ins = [r["instr"] / 1e6 for r in rs if r.get("instr") is not None]
            ev = [r["n_eval"] for r in rs]
            m, lo, hi = q3(wall)
            cw = (sum(cpu) / sum(wall)) if cpu and sum(wall) > 0 else math.nan
            hist_it = sum(r.get("pirls_iters", 0) for r in rs)
            hist_ev = sum(sum(r.get("pirls_hist", [])) for r in rs)
            ipe = hist_it / hist_ev if hist_ev else math.nan
            conv = sum(r["converged"] for r in rs) / len(rs)
            sing_v = [r["singular"] for r in rs if "singular" in r]
            sing = sum(sing_v) / len(sing_v) if sing_v else math.nan
            routes = sorted({r.get("route", "?") for r in rs})
            print(f"{k[0]:<17}{k[1]:<6}{k[2]:<9}{k[3]:<6}{len(rs):>5}"
                  f"{fmt(m, 1) + ' [' + fmt(lo, 1) + ',' + fmt(hi, 1) + ']':>32}{fmt(min(wall), 1):>11}"
                  f"{fmt(q3(cpu)[0], 1):>11}{fmt(cw, 3):>9}{fmt(q3(ins)[0], 3):>11}"
                  f"{fmt(q3(ev)[0], 1):>10}{fmt(st.mean(ev), 1):>8}{fmt(ipe, 2):>8}"
                  f"{fmt(conv, 2):>6}{fmt(sing, 2):>6}  {','.join(routes)}")
        pk = [k for k in procs if k[0] == arm]
        if pk:
            print(f"  process totals (run_loop.sh, /usr/bin/time -l; include data generation, "
                  f"warm-up, readout and process start):")
            print("  Minstr/fit = process instructions / fits; net = (process - its --dry twin) / fits,")
            print("  i.e. without data generation and process start (needs DRY=1 runs).")
            print(f"  {'shape':<17}{'size':<6}{'variant':<9}{'start':<6}{'procs':>6}{'Minstr/fit':>12}"
                  f"{'net':>9}{'Mcyc/fit':>10}{'peakRSS MB':>11}{'user s':>8}{'sys s':>7}")
            for key in sorted(pk, key=lambda k: tuple(str(x) for x in k)):
                allp = procs[key]
                ps = [p for p in allp if not p.get("dry")]
                dry = [p for p in allp if p.get("dry")]
                if not ps:
                    continue
                fits = [p.get("fits") or 0 for p in ps]
                ipf = [p["instructions"] / f / 1e6 for p, f in zip(ps, fits) if f and p.get("instructions")]
                cpf = [p["cycles"] / f / 1e6 for p, f in zip(ps, fits) if f and p.get("cycles")]
                net = math.nan
                if dry and ipf:
                    di = q3([p["instructions"] for p in dry if p.get("instructions")])[0]
                    fi = q3([p["instructions"] for p in ps if p.get("instructions")])[0]
                    net = (fi - di) / max(fits) / 1e6
                rss = [p["maxrss_bytes"] / 2**20 for p in ps if p.get("maxrss_bytes") is not None]
                s = key[1]
                print(f"  {str(s[0]):<17}{str(s[1]):<6}{str(s[2]):<9}{str(s[3]):<6}{len(ps):>6}"
                      f"{fmt(q3(ipf)[0], 3):>12}{fmt(net, 3):>9}{fmt(q3(cpf)[0], 3):>10}"
                      f"{fmt(max(rss) if rss else math.nan, 1):>11}"
                      f"{fmt(st.mean([p.get('user_s', math.nan) for p in ps]), 3):>8}"
                      f"{fmt(st.mean([p.get('sys_s', math.nan) for p in ps]), 3):>7}")
        print()


def index(rs):
    """(block, seed, draw) -> first record; repeats are the determinism check's."""
    out = {}
    for r in rs:
        out.setdefault(block_key(r) + (r["seed"], r["draw"]), r)
    return out


def compare(label, pairs, blocks, arm_of_eps, symmetric=True):
    """pairs: [(record, reference)] with the same key. Returns failures."""
    by = defaultdict(lambda: {"n": 0, "ok": 0, "maxd": 0.0, "worse": 0, "better": 0,
                              "conv": 0, "sing": 0, "nodev": 0})
    fails = 0
    for r, b in pairs:
        s = by[block_key(r)]
        s["n"] += 1
        eps = eps_for(blocks, arm_of_eps, r)
        o, ob = r.get("objective"), b.get("objective")
        flags_ok = True
        if r["converged"] != b["converged"]:
            s["conv"] += 1
            flags_ok = False
        if r.get("singular") != b.get("singular"):
            s["sing"] += 1
            flags_ok = False
        if o is None or ob is None:
            # Both non-converged with no endpoint: nothing to compare but the flags.
            if o is None and ob is None:
                s["ok"] += int(flags_ok)
            else:
                s["nodev"] += 1
            continue
        d = o - ob
        s["maxd"] = max(s["maxd"], abs(d))
        dev_ok = abs(d) <= eps if symmetric else d <= eps
        if d > eps:
            s["worse"] += 1
        elif d < -eps:
            s["better"] += 1
        if dev_ok and flags_ok:
            s["ok"] += 1
    print(f"--- {label}")
    print(f"{'shape':<17}{'size':<6}{'variant':<9}{'start':<6}{'n':>5}{'pass':>6}{'max|d obj|':>12}"
          f"{'worse':>6}{'better':>7}{'conv!=':>7}{'sing!=':>7}{'no-dev':>7}")
    for k in sorted(by):
        s = by[k]
        bad = s["n"] - s["ok"]
        fails += bad
        print(f"{k[0]:<17}{k[1]:<6}{k[2]:<9}{k[3]:<6}{s['n']:>5}{s['ok']:>6}{s['maxd']:>12.3g}"
              f"{s['worse']:>6}{s['better']:>7}{s['conv']:>7}{s['sing']:>7}{s['nodev']:>7}"
              + ("" if bad == 0 else "   FAIL"))
    print(f"    {'PASS' if fails == 0 else 'FAIL'}: {sum(s['ok'] for s in by.values())}/"
          f"{sum(s['n'] for s in by.values())} draws (dev_eps {DEV_EPS:g}, "
          f"vector-AGQ {DEV_EPS_AGQ_VEC:g}, from validation/grid/tol.R)\n")
    return fails


def answer_check(draws, blocks, base):
    if base not in draws:
        print(f"answer check: base arm {base!r} not found (arms: {sorted(draws)})")
        return 1
    ref = index(draws[base])
    fails = 0
    for arm in sorted(draws):
        if arm == base:
            continue
        mine = index(draws[arm])
        pairs = [(r, ref[k]) for k, r in mine.items() if k in ref]
        missing = len(mine) - len(pairs)
        fails += compare(f"answer check: {arm} against {base} (same seed and draw; "
                         f"{missing} draw(s) without a base twin)", pairs, blocks, base)
    return fails


def start_check(draws, blocks):
    fails = 0
    for arm in sorted(draws):
        idx = index(draws[arm])
        pairs = []
        for k, r in idx.items():
            if k[3] == "cold":
                continue
            ck = k[:3] + ("cold",) + k[4:]
            if ck in idx:
                pairs.append((r, idx[ck]))
        if pairs:
            fails += compare(f"start check in arm {arm}: truth and carry against cold, same draw "
                             f"(pass = |d| <= eps; 'better' = a lower optimum than cold)",
                             pairs, blocks, arm)
    return fails


def determinism(draws):
    fails = 0
    print("--- determinism (repeated records inside one arm)")
    for arm in sorted(draws):
        seen = defaultdict(list)
        for r in draws[arm]:
            seen[block_key(r) + (r["seed"], r["draw"])].append(r)
        rep = {k: v for k, v in seen.items() if len(v) > 1}
        bad = 0
        for k, v in rep.items():
            o0, e0 = v[0].get("objective"), v[0]["n_eval"]
            for r in v[1:]:
                same_o = (o0 == r.get("objective"))
                if not same_o or e0 != r["n_eval"] or v[0].get("pirls_hist") != r.get("pirls_hist"):
                    bad += 1
                    if bad <= 5:
                        print(f"  {arm} {k}: objective {o0!r} vs {r.get('objective')!r}, "
                              f"evals {e0} vs {r['n_eval']}")
        print(f"  {arm}: {len(rep)} repeated keys, {bad} mismatches "
              f"({'PASS' if bad == 0 else 'FAIL'}; bit-identical objective, equal evals and PIRLS histogram)")
        fails += bad
    print()
    return fails


def sanity(draws, blocks):
    print("--- sanity (cold starts): mean estimate vs generating value; "
          "z = (mean - truth) / (sd / sqrt(fits))")
    for arm in sorted(draws):
        g = defaultdict(list)
        for r in draws[arm]:
            if r["start"] == "cold":
                g[block_key(r)].append(r)
        for k in sorted(g):
            rs = g[k]
            b = blocks.get((arm,) + k[:3], {})
            bt, tt = b.get("beta_true", []), b.get("theta_true", [])
            conv = [r for r in rs if r["converged"]]
            parts = []
            for name, truth, field in (("b", bt, "beta"), ("t", tt, "theta")):
                for j, tv in enumerate(truth):
                    v = [r[field][j] for r in conv if len(r.get(field) or []) > j and r[field][j] is not None]
                    if len(v) < 2:
                        continue
                    m, sd = st.mean(v), st.stdev(v)
                    z = (m - tv) / (sd / math.sqrt(len(v))) if sd > 0 else math.nan
                    parts.append(f"{name}{j} {m:.4g}/{tv:.4g} z{z:+.1f}")
            sing = sum(1 for r in rs if r.get("singular")) / len(rs)
            print(f"  {arm:<10}{k[0]:<17}{k[1]:<6}{k[2]:<9} conv {len(conv)}/{len(rs)} sing {sing:.2f}  "
                  + "  ".join(parts))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--base", help="arm the answer check compares every other arm against")
    ap.add_argument("--starts", action="store_true", help="truth/carry vs cold inside each arm")
    ap.add_argument("--determinism", action="store_true", help="repeated records must be identical")
    ap.add_argument("--sanity", action="store_true", help="estimates vs generating values")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--no-fail", action="store_true", help="exit 0 even when a check fails")
    a = ap.parse_args()
    draws, blocks, procs, headers = load(a.paths)
    if not draws:
        print("no draw records found")
        return 1
    if not a.no_summary:
        summary(draws, blocks, procs, headers)
    fails = 0
    if a.base:
        fails += answer_check(draws, blocks, a.base)
    if a.starts:
        fails += start_check(draws, blocks)
    if a.determinism:
        fails += determinism(draws)
    if a.sanity:
        sanity(draws, blocks)
    return 0 if (fails == 0 or a.no_fail) else 1


if __name__ == "__main__":
    sys.exit(main())
