#!/usr/bin/env python3
"""50-digit CR2 and HC2 degrees of freedom on nearly collinear designs ->
validation/goldens/inference/cr2_df/<name>.json, one per `kind = "cr2_df"` entry of the
manifest's `inference_references` block.

THE ORACLE IS SACRED. These JSONs are the frozen reference the crate's robust df are measured
against where the design is nearly singular and double precision loses the df. On any glmm
disagreement, glmm is presumed wrong: never relax a tolerance or edit a reference.

A BARE RUN OVERWRITES EVERY REFERENCE IT OWNS (every `cr2_df` entry). Use VALIDATION_ONLY for
anything short of a deliberate full re-freeze:
    VALIDATION_ONLY=<name>[,<name>...] python3 validation/tools/goldens_cr2_df.py

Needs mpmath at the version of grid/versions.json's `mpmath`; any other version stops the run.
No R, no threads: the result depends on mpmath alone.

An entry reads its `data` CSV (tools/prep/gen_cr2_df_data.py) with float(), each value then
made an mpf, so the doubles every engine reads are the exact data. The design is X = [1, a, xi,
xi2]; an `r_formula` other than null or "y ~ a + xi + xi2" stops the run. `cluster` names the
CSV column of the clusters (absent: one row per cluster), and the working model is
Phi_k = I + `tau2` 11' on each cluster.

At mp.dps = 50, with M = (sum_k X_k' Phi_k^-1 X_k)^-1 and per cluster the CR2 adjustment
A_k = L_k B_k^-1/2 L_k', B_k = L_k' (Phi_k - X_k M X_k') L_k, L_k the Cholesky factor of Phi_k,
E_k = M X_k' Phi_k^-1 A_k. For coefficient j, g_k = E_k' e_j and the G x G matrix
Q_kl = g_k' (delta_kl Phi_k - X_k M X_l') g_l give the Pustejovsky-Tipton df
(tr Q)^2 / tr(Q^2), rounded to the nearest double. The JSON carries `df`, one per coefficient
in the order of X.
"""

import csv
import json
import math
import os
import platform
import sys

import mpmath as mp
from mpmath.libmp.libmpf import round_nearest, to_float

KIND = "cr2_df"
MP_DPS = 50
GENERATOR = os.path.basename(__file__)
SUITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLUMNS = ["a", "xi", "xi2"]
FORMULAS = (None, "y ~ a + xi + xi2")

mp.mp.dps = MP_DPS


def die(msg):
    sys.exit(f"{GENERATOR}: {msg}")


def assert_mpmath_version():
    with open(os.path.join(SUITE_DIR, "grid", "versions.json"), encoding="utf-8") as f:
        pinned = json.load(f)["mpmath"]
    if mp.__version__ != pinned:
        die(f"mpmath is {mp.__version__}, grid/versions.json pins {pinned}")


# Mirrors goldens_glm_band.py's select_entries; change together.
def select_entries(kind):
    with open(os.path.join(SUITE_DIR, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if "inference_references" not in manifest:
        die("manifest.json has no `inference_references` block")
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


# Mirrors goldens_glm_band.py's provenance; change together. No date and no host name, so a
# re-run is byte-identical.
def provenance():
    return {
        "python_version": platform.python_version(),
        "packages": {"mpmath": mp.__version__},
        "mp_dps": MP_DPS,
        "generator": GENERATOR,
    }


# Mirrors goldens_glm_band.py's to_json; change together.
def to_json(v, indent=0):
    pad, inner = "  " * indent, "  " * (indent + 1)
    if isinstance(v, dict):
        items = [
            f"{inner}{json.dumps(k)}: {to_json(x, indent + 1)}" for k, x in v.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(v, list):
        if all(not isinstance(x, (dict, list)) for x in v):
            return "[" + ", ".join(to_json(x) for x in v) + "]"
        return (
            "[\n"
            + ",\n".join(inner + to_json(x, indent + 1) for x in v)
            + "\n"
            + pad
            + "]"
        )
    if isinstance(v, float):
        if not math.isfinite(v):
            die("a non-finite value would be written")
        return repr(v)
    return json.dumps(v)


# Mirrors goldens_glm_band.py's write_reference; change together.
def write_reference(obj, path):
    obj["name"] = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_json(obj) + "\n")


def f64(x):
    return to_float(mp.mpf(x)._mpf_, rnd=round_nearest)


def clusters(entry):
    """(X_k, Phi_k) per cluster, in the order of the cluster values (or of the rows)."""
    if entry["r_formula"] not in FORMULAS:
        die(
            f"{entry['name']}: r_formula {entry['r_formula']!r} is not y ~ a + xi + xi2"
        )
    with open(
        os.path.join(SUITE_DIR, entry["data"]), newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))
    key = entry.get("cluster")
    if key is not None and key not in rows[0]:
        die(f"{entry['name']}: cluster column {key} is not in the CSV")
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(float(row[key]) if key is not None else i, []).append(row)
    tau2 = mp.mpf(float(entry["tau2"]))
    out = []
    for _, members in sorted(groups.items()):
        x = mp.matrix(
            [[mp.mpf(1)] + [mp.mpf(float(r[c])) for c in COLUMNS] for r in members]
        )
        nk = len(members)
        out.append((x, mp.eye(nk) + tau2 * mp.ones(nk, nk)))
    return out


def df_all(cl):
    """The df of each coefficient, from the full G x G Q."""
    p = cl[0][0].cols
    h = mp.zeros(p, p)
    for x, phi in cl:
        h += x.T * mp.inverse(phi) * x
    m = mp.inverse(h)
    es = []
    for x, phi in cl:
        lk = mp.cholesky(phi)
        w, v = mp.eigsy(lk.T * (phi - x * m * x.T) * lk)
        a = lk * v * mp.diag([1 / mp.sqrt(e) for e in w]) * v.T * lk.T
        es.append(m * x.T * mp.inverse(phi) * a)
    big_g = len(cl)
    out = []
    for j in range(p):
        g = [e[j, :].T for e in es]
        u = [cl[k][0].T * g[k] for k in range(big_g)]
        mu = [m * uk for uk in u]
        q = mp.zeros(big_g, big_g)
        for k in range(big_g):
            for l in range(k, big_g):
                val = -sum(u[k][i] * mu[l][i] for i in range(p))
                if k == l:
                    val += (g[k].T * cl[k][1] * g[k])[0]
                q[k, l] = q[l, k] = val
        tr = sum(q[k, k] for k in range(big_g))
        tr2 = sum(q[k, l] * q[l, k] for k in range(big_g) for l in range(big_g))
        out.append(f64(tr * tr / tr2))
    return out


def main():
    assert_mpmath_version()
    for entry in select_entries(KIND):
        df = df_all(clusters(entry))
        write_reference(
            {"name": entry["name"], "df": df, "provenance": provenance()},
            os.path.join(
                SUITE_DIR, "goldens", "inference", KIND, entry["name"] + ".json"
            ),
        )
        print(f"wrote {entry['name']} ({len(df)} df)")


if __name__ == "__main__":
    main()
