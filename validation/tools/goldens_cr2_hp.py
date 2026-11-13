#!/usr/bin/env python3
"""High-precision CR2 references -> validation/goldens/inference/cr2_hp/<name>.json, one per
`kind = "cr2_hp"` entry of the manifest's `inference_references` block.

THE ORACLE IS SACRED. These JSONs are the frozen reference the crate's CR2 kernel is measured
against at wide weights, where the dense double-precision code is the
less accurate side. On any glmm disagreement, glmm is presumed wrong: never relax a tolerance or
edit a reference.

A BARE RUN OVERWRITES EVERY REFERENCE IT OWNS. Use VALIDATION_ONLY for anything short of a
deliberate full re-freeze:
    VALIDATION_ONLY=<name>[,<name>...] python3 validation/tools/goldens_cr2_hp.py

Needs python-flint and numpy at their grid/versions.json pins; any other version stops the run.

Per entry, tools/cr2/hp_ref.py runs as is (in a scratch directory) on make_data(G, nk, spread,
re, seed = 7) at `prec_bits`: A = Phi (C Phi)^(-1/2) by Denman-Beavers in arb arithmetic, and
per cluster the stored pieces e = E r, EPE = E Phi E', EX = E X, each rounded to the nearest
double. The run first checks that the entry's data CSV (tools/prep/gen_cr2_data.py) holds
exactly those rows. SE and df of each coefficient come from the pieces by tools/cr2/lib.py's
summarize. The JSON carries coefficients, se, df, M and the pieces per cluster (`clusters`).
"""
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile

import flint
import numpy as np

KIND = "cr2_hp"
GENERATOR = os.path.basename(__file__)
TOOLS = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(TOOLS)
sys.path.insert(0, os.path.join(TOOLS, "cr2"))
from lib import make_data, summarize  # noqa: E402


def die(msg):
    sys.exit(f"{GENERATOR}: {msg}")


def assert_versions():
    with open(os.path.join(SUITE_DIR, "grid", "versions.json"), encoding="utf-8") as f:
        pins = json.load(f)
    for name, have in (("python-flint", flint.__version__), ("numpy", np.__version__)):
        if have != pins[name]:
            die(f"{name} is {have}, grid/versions.json pins {pins[name]}")


# Mirrors goldens_glm_band.py's select_entries; change together.
def select_entries(kind):
    with open(os.path.join(SUITE_DIR, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    entries = [e for e in manifest["inference_references"] if e.get("kind") == kind]
    names = [e["name"] for e in entries]
    only = os.environ.get("VALIDATION_ONLY", "")
    if only:
        keep = only.split(",")
        unknown = [n for n in keep if n not in names]
        if unknown:
            die(f"VALIDATION_ONLY names no {kind} entry: {', '.join(unknown)}")
        return [e for e in entries if e["name"] in keep]
    for e in entries:
        if e.get("pending_reference") is not None:
            print(f"skip {e['name']} (pending_reference: {e['pending_reference']})")
    return [e for e in entries if e.get("pending_reference") is None]


# Mirrors goldens_glm_band.py's provenance, with flint's precision for mpmath's digits.
def provenance(prec_bits):
    return {
        "python_version": platform.python_version(),
        "packages": {"python-flint": flint.__version__, "numpy": np.__version__},
        "prec_bits": prec_bits,
        "generator": GENERATOR,
    }


# Mirrors goldens_glm_band.py's to_json and write_reference; change together.
def to_json(v, indent=0):
    pad, inner = "  " * indent, "  " * (indent + 1)
    if isinstance(v, dict):
        items = [f"{inner}{json.dumps(k)}: {to_json(x, indent + 1)}" for k, x in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(v, list):
        if all(not isinstance(x, (dict, list)) for x in v):
            return "[" + ", ".join(to_json(x) for x in v) + "]"
        return "[\n" + ",\n".join(inner + to_json(x, indent + 1) for x in v) + "\n" + pad + "]"
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            die("a non-finite value would be written")
        return repr(v)
    return json.dumps(v)


def write_reference(obj, path):
    obj["name"] = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_json(obj) + "\n")


def check_data(entry, cl):
    with open(os.path.join(SUITE_DIR, entry["data"]), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    flat = [(k, c, i) for k, c in enumerate(cl) for i in range(len(c["y"]))]
    ok = len(rows) == len(flat) and all(
        int(r["g"]) == k + 1 and float(r["x1"]) == c["X"][i, 1] and float(r["x2"]) == c["X"][i, 2]
        and float(r["lam"]) == c["lam"][i] and float(r["y"]) == c["y"][i]
        and all(float(r[f"u{j + 1}"]) == u for j, u in enumerate(c["U"][i]))
        for r, (k, c, i) in zip(rows, flat))
    if not ok:
        die(f"{entry['name']}: {entry['data']} does not hold make_data's rows")


def hp_reference(entry):
    G, nk, spread, re_, prec = entry["G"], entry["nk"], entry["spread"], entry["re"], entry["prec_bits"]
    cl = make_data(G, nk, spread, re_, seed=7)
    check_data(entry, cl)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, os.path.join(TOOLS, "cr2", "hp_ref.py"), "ref", str(G), str(nk),
                        repr(float(spread)), re_, "", str(prec)], cwd=tmp, check=True,
                       stdout=subprocess.DEVNULL)
        hp = dict(np.load(os.path.join(tmp, "hp_ref.npz")))
    st = [dict(e=hp[f"e{k}"], EPE=hp[f"EPE{k}"], EX=hp[f"EX{k}"]) for k in range(G)]
    se, df = summarize(st, hp["M"])
    tolist = lambda a: [float(x) for x in a] if a.ndim == 1 else [[float(x) for x in r] for r in a]
    return {
        "provenance": provenance(prec),
        "coefficients": ["(Intercept)", "x1", "x2"],
        "se": tolist(se),
        "df": tolist(df),
        "M": tolist(hp["M"]),
        "clusters": [{"e": tolist(s["e"]), "EPE": tolist(s["EPE"]), "EX": tolist(s["EX"])} for s in st],
    }


def main():
    assert_versions()
    for entry in select_entries(KIND):
        write_reference(hp_reference(entry),
                        os.path.join(SUITE_DIR, "goldens", "inference", KIND, entry["name"] + ".json"))
        print(f"wrote {entry['name']}")


if __name__ == "__main__":
    main()
