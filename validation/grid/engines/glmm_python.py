#!/usr/bin/env python3
# The grid's Python port engine: the SAME kernel as grid/engines/glmm.rs reached
# through PyO3, so its numbers must match the Rust engine to ROUND-OFF, not
# merely sit inside the cross-engine bands. It gates the BINDING, not the math.
# Mirrors grid/engines/common.rs step for step -- change together.
#
# Environment, the same five variables every engine of this grid reads (the R
# and Julia engines cannot share this code -- mirrored there, change
# together): GRID_DIR (the grid/ directory; every cell's `data` path is
# relative to it), GRID_MANIFEST, GRID_OUT, GRID_CELLS (comma-separated ids
# this worker must fit) and GRID_TIMED (""/"0" = untimed, else the sample
# count, an integer >= 2). Resume-safe: a cell already in GRID_OUT is skipped,
# so a killed worker can simply be relaunched. Per-cell exceptions are caught
# and recorded as engine-fail -- a grid corner breaking the port is a data
# point, not a reason to lose the rest of the run. Caught as BaseException,
# not Exception: a Rust panic inside the kernel crosses the PyO3 FFI boundary
# as `pyo3_runtime.PanicException`, which derives from BaseException (pyo3's
# panic.rs) so that it cannot be accidentally swallowed by ordinary
# `except Exception` code -- but that means THIS handler must reach for it
# deliberately, or an in-kernel panic kills the whole worker with no record,
# and a relaunch would walk straight back into the same cell forever.
# KeyboardInterrupt/SystemExit are re-raised unchanged: those are the
# operator stopping the run, not a cell failing.

import json
import math
import os
import statistics
import sys
import time
from importlib.metadata import version

import glmm

GRID_DIR = os.environ.get("GRID_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.environ.get("GRID_MANIFEST", f"{GRID_DIR}/manifest.json")
OUT = os.environ.get("GRID_OUT", f"{GRID_DIR}/results.jsonl")
ONLY = [c for c in os.environ.get("GRID_CELLS", "").split(",") if c]
# The package version exactly as it reports it -- the same spelling glmm.rs
# writes ("0.4.0", no suffix); a dirty tree's provenance lives in
# run_meta.json's glmm_git_rev, not here.
VERSION = version("glmm")

# Manifest family string -> the Python port's family string. The grid
# manifest already spells every family the crate's own way (unlike the old
# per-rung validation manifest, which used "negbin"), so this is an identity
# table kept for the same shape as glmm.rs::family_of / the old port.
_FAMILY = {
    "gaussian": "gaussian",
    "binomial": "binomial",
    "poisson": "poisson",
    "gamma": "gamma",
    "negativebinomial": "negativebinomial",
    "inversegaussian": "inversegaussian",
}


def timed_samples():
    """Sample count for this run, or None when timing is off. Mirrors
    grid/engines/glmm.rs::timed_samples -- change together. Raises rather than
    silently not timing when the engine is run by hand with a malformed
    value."""
    raw = os.environ.get("GRID_TIMED", "").strip()
    if raw in ("", "0"):
        return None
    try:
        n = int(raw)
    except ValueError:
        n = -1
    if n < 2:
        raise SystemExit(
            f"GRID_TIMED must be 0 or an integer >= 2 (got {raw!r}); "
            "N=2 keeps 1 sample after the warm-up discard"
        )
    return n


TIMED = timed_samples()


def read_csv_path(path):
    """Read a grid CSV (unquoted header + rows, `,`-split) -- mirrors
    common.rs::read_csv_path, deliberately including its naivety: the grid
    corpus carries no embedded commas."""
    with open(path) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    header = [unquote(s) for s in lines[0].split(",")]
    rows = [[unquote(s) for s in ln.split(",")] for ln in lines[1:]]
    return header, rows


def unquote(s):
    return s.strip().strip('"')


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_data(header, rows, factors):
    """Columns keyed by name, typed the way common.rs::build_table types them:
    manifest `factors` are categorical, as is any column that fails to parse
    as f64 anywhere (Pastes' `sample` -- carried in the CSV, referenced by no
    formula).

    A str column reaches `glmm.fit` as a factor with lexicographic levels,
    which is exactly `Column::factor_from_labels` (R's `factor()` default,
    and what the reference side did when the goldens were frozen) -- so the
    two engines agree on the contrast base without either declaring an order.

    Dots in R-origin headers (Arabidopsis' `total.fruits`) become underscores
    to match `glmm_formula`'s sanitized names -- mirrors build_table's
    rename.
    """
    data = {}
    for j, name in enumerate(header):
        values = [r[j] for r in rows]
        is_factor = name in factors or any(not _is_float(v) for v in values)
        data[name.replace(".", "_")] = values if is_factor else [float(v) for v in values]
    return data


def num(x):
    """NaN/Inf -> JSON null (mirrors common.rs::num): a non-converged fit
    leaves NaN-filled estimates, and json.dumps(allow_nan=False) would raise
    rather than write the invalid `NaN` literal the comparator cannot read."""
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def nums(xs):
    return [num(float(x)) for x in xs]


def base_record(cell_id, engine, engine_version):
    """The comparison record, pre-seeded with every key at its "missing"
    value -- the Python twin of common.rs::base_record. `se_rx`/`se_hessian`
    are NOT seeded: they are the two keys legitimately ABSENT where an engine
    has none, so the caller inserts them only when it has them."""
    return {
        "cell": cell_id,
        "engine": engine,
        "engine_version": engine_version,
        "converged": False,
        "singular": False,
        "status": "engine-fail",
        "message": None,
        "coef_names": [],
        "beta": [],
        "varcomp": [],
        "sigma": None,
        "nb_theta": None,
        "loglik": None,
        "deviance": None,
        "n_eval": None,
        "wall_seconds": None,
        "fits_per_sample": 1,
    }


def set_loglik(rec, loglik):
    """Set `loglik` AND the `deviance` that must agree with it, in one call --
    mirrors common.rs::set_loglik. deviance = -2*loglik, null when loglik is
    non-finite."""
    rec["loglik"] = num(loglik)
    rec["deviance"] = num(-2.0 * loglik)


def varcomp(fh):
    """Variance components in the grid schema, one entry per grouping factor
    in declaration order, from `Fit.stddev_corr` -- mirrors
    common.rs::varcomp. Empty for a fixed-only cell (no re_groups) and empty
    when the fit did not fill varcorr: a non-converged mixed fit leaves it
    empty, and `stddev_corr` indexes it directly, so calling it per declared
    grouping would raise and cost the record every other field."""
    if len(fh.varcorr) != len(fh.re_groups):
        return []
    out = []
    for i, (name, terms) in enumerate(fh.re_groups):
        stddev, corr = fh.stddev_corr(i)
        out.append(
            {
                "group": name,
                "terms": list(terms),
                "stddev": nums(stddev),
                "corr": [nums(row) for row in corr],
            }
        )
    return out


def not_converged_text(fh):
    """What the fit says about itself when it did not converge: the boundary
    state, which variance components were pinned there, and any solver
    notes. Mirrors common's common.rs::not_converged_text -- keeps a
    non-convergence diagnosable in a run of ~800 cells, where the
    alternative is a record that only says "engine-fail"."""
    d = fh.diagnostics
    return f"not converged: boundary={d['boundary']!r}, pinned={d['pinned']!r}, notes={d['notes']!r}"


def done_cells(path):
    """Cell ids already present in `path` -- mirrors common.rs::done_cells: a
    kill -9 can truncate the final line, so a line that does not parse is
    skipped, never fatal."""
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return set()
    done = set()
    for line in lines:
        try:
            done.add(json.loads(line)["cell"])
        except (json.JSONDecodeError, KeyError):
            pass
    return done


def fit_one(cell):
    rec = base_record(cell["cell"], "glmm_python", VERSION)
    try:
        # `data` is relative to grid/, so a generated cell and a crate
        # fixture are read by the same line. Everything from here on is
        # inside this try, mirroring common.rs's catch_unwind scope -- a grid
        # corner that breaks the port anywhere in setup (a bad formula, an
        # unsupported family/link pair) is one failed cell, not a
        # run-ending crash.
        header, rows = read_csv_path(os.path.join(GRID_DIR, cell["data"]))
        data = build_data(header, rows, cell.get("factors", []))

        def column(name):
            j = header.index(name)  # hoisted: .index() per row is O(rows x cols)
            return [float(r[j]) for r in rows]

        # Trial counts vs prior weights -- the two manifest fields, mutually
        # exclusive by construction (gen_manifest.R asserts it).
        weights = None
        if cell.get("weights") is not None:
            sizes = column(cell["weights"])
            data["prop"] = [y / s for y, s in zip(column(cell["response"]), sizes)]
            weights = sizes
        elif cell.get("weights_col") is not None:
            weights = column(cell["weights_col"])

        # THE OFFSET, applied exactly once: glmm_formula carries no
        # offset(...) term, so it arrives here and only here.
        offset = column(cell["offset_col"]) if cell.get("offset_col") is not None else None

        # glmm_formula, NOT jl_formula: gen_manifest.R already produced the
        # crate's dialect, so this file does no formula rewriting at all.
        formula = cell["glmm_formula"]
        kw = {"link": cell["link"], "weights": weights, "offset": offset}
        if cell.get("nagq") is not None:
            kw["nagq"] = int(cell["nagq"])

        gaussian = cell["family"] == "gaussian"

        def timed_fit(call):
            """Run `call` once when GRID_TIMED is off; otherwise GRID_TIMED
            times, first sample discarded, `rec["wall_seconds"]` set to the
            median of the rest. Returns the last (or only) sample's Fit --
            `call` is deterministic (BOBYQA on a fixed input), so every
            sample agrees. Mirrors common.rs's separate untimed report fit
            plus timed wall-clock loop, folded into one helper because the
            port has one call to re-run rather than two Wald-SE arms."""
            if TIMED is None:
                return call()
            fh = None
            samples = []
            for _ in range(TIMED):
                t0 = time.perf_counter()
                fh = call()
                samples.append(time.perf_counter() - t0)
            rec["wall_seconds"] = statistics.median(samples[1:])
            return fh

        fh = timed_fit(lambda: glmm.fit(data, formula, _FAMILY[cell["family"]], wald_se="hessian", **kw))
        if gaussian:
            rec["se_rx"] = nums(fh.se)  # one profiled SE, no method split
        else:
            fr = glmm.fit(data, formula, _FAMILY[cell["family"]], wald_se="rx", **kw)
            rec["se_hessian"] = nums(fh.se)
            rec["se_rx"] = nums(fr.se)
        rec["coef_names"] = list(fh.names)
        rec["beta"] = nums(fh.beta)
        rec["varcomp"] = varcomp(fh)  # by group NAME; no ref_order
        set_loglik(rec, fh.loglik)
        rec["sigma"] = num(math.sqrt(fh.dispersion)) if cell["family"] in ("gaussian", "gamma") else None
        rec["nb_theta"] = num(fh.dispersion) if cell["family"] == "negativebinomial" else None
        rec["n_eval"] = fh.n_eval
        rec["converged"] = bool(fh.converged)
        rec["singular"] = bool(fh.singular)
        rec["status"] = "ok" if fh.converged else "engine-fail"
        if not fh.converged:
            rec["message"] = not_converged_text(fh)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # a grid corner (including a kernel panic) is a data point
        rec["message"] = f"{type(exc).__name__}: {exc}"
    return rec


def main():
    out_dir = os.path.dirname(OUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(MANIFEST) as fh:
        manifest = json.load(fh)
    done = done_cells(OUT)
    # Cell lookup by id, and the fit order: GRID_CELLS's own order (manifest
    # order when it is empty), never re-derived from the manifest. The
    # runner's watchdog blames a timeout on the first cell of GRID_CELLS still
    # missing from the output, so fitting in any other order fits the wrong
    # cell first and the blame lands on the wrong one (mirrors run.sh's
    # `next_missing` -- change together).
    cells_by_id = {cell["cell"]: cell for cell in manifest["cells"]}
    ids = ONLY if ONLY else [cell["cell"] for cell in manifest["cells"]]
    with open(OUT, "a") as out:
        for cell_id in ids:
            if cell_id in done:
                continue
            if cell_id not in cells_by_id:
                raise KeyError(f"unknown cell id not in manifest: {cell_id}")
            rec = fit_one(cells_by_id[cell_id])
            # json.dumps uses float.__repr__ for floats (the CPython encoder's
            # default), which is already the shortest round-trip
            # representation -- full precision, not a display rounding, so
            # the 1e-12 port band is meaningful.
            out.write(json.dumps(rec) + "\n")
            out.flush()  # line-per-fit flush: the watchdog watches mtime


if __name__ == "__main__":
    sys.exit(main())
