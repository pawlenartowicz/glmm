#!/usr/bin/env python3
"""Python-port side of the cross-language parity harness — fits the parity datasets
with the `glmm` Python package (the PyO3 wheel over the same Rust kernel) and writes
`results/glmm_python_{empirical,simulated}/<ds>.json` in the common schema
(parity/README.md).

Why a fourth engine. The port calls the SAME kernel as `fit.rs`: `glmm.fit` lowers
the formula through `glmm::formula::lower` and fits through `fit_warm(start=None)`,
which is `fit_cold` byte-for-byte (src/fit/mod.rs). So the estimates must match the
Rust `glmm` row to round-off, and `compare.R` gates them there (TOL$port_rel) rather
than against lme4 — a port bug (a swapped column, a mis-ordered factor level, a
dropped weights vector) shows up as a Rust-vs-Python disagreement, which the Python
pytest suite cannot see: it fits fresh random data and asserts only convergence and
`beta[1]` within 0.3 of truth, never a reference number.

What the timing means. This harness times the whole `glmm.fit(data, formula, ...)`
call — every call re-does the dict scan, the per-row `float()` conversion, the FFI
copy, and the lowering — because that is the only thing a Python caller can invoke.
That is the SAME construct-and-fit span lme4 (`lmer`), MixedModels (`fit(MixedModel,
...)`) and now the Rust glmm harness's `_full` timing all measure, so `py_gap` in
summarize_timing.R compares same-to-same: with lowering timed on both the port and the
Rust side it cancels, leaving `py_gap` as the TRUE port tax — dict scan + `float()` +
FFI copy. (Before the Rust `_full` timing existed, `py_gap` was measured against
`fit_cold` alone and so charged Rust's own lowering to the port, inflating the tiny-fit
rungs — e.g. cake read 4.5x then, ~1.7x now.) The tax is ~1.0x on any rung whose fit
takes real time and rises only where the whole call is sub-millisecond. The wheel's
kernel carries the same codegen as `parity_fit`: `glmm-python` is a member of the root
workspace (Cargo.toml `members`), so the root `[profile.release]` (lto="thin",
codegen-units=1) applies to both sides — the gap is the port, not LTO. Build the wheel
`--release` before running, or the port "overhead" is a debug build.

Manifest-driven, mirroring `fit.rs`'s `fit_one` field for field (jl_formula lowering,
the r_formula/cbind fallback, `weights` aggregation, `weights_col`, `timing_batch`,
the lme4-reference varcomp reindex). It is generic over BOTH suites for the same
reason `fit.rs` is: `run.sh` is shared with `parity/weights/`, which execs it.

Run via `run.sh` (ENGINES has "py") or `python3 parity/oracle/fit.py`. Paths are
anchored at this file, so the cwd does not matter.
"""

import json
import math
import os
import statistics
import sys
import time
from importlib.metadata import version

import glmm

# The installed port's version — the wheel is pinned lockstep with the crate
# (python/pyproject.toml), so this is fit.rs's CARGO_PKG_VERSION by construction.
VERSION = version("glmm")

# Timing loop: first (cold) pass discarded, MEDIAN of the rest reported — the same
# convention and count as fit.rs's N_RUNS (change together), so the two engines'
# medians are comparable without renormalizing.
N_RUNS = 10

# Suite directory (manifest + data + results root). PARITY_SUITE_DIR overrides at RUN
# time (a sub-suite's run.sh, e.g. parity/weights/, sets it); unset = the main parity/
# dir. Mirrors fit.rs's suite_dir().
_HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.environ.get("PARITY_SUITE_DIR") or os.path.dirname(_HERE)

# Manifest family name -> the Python port's family string. Only negbin differs: the
# crate's Family::NegativeBinomial is spelled "negativebinomial" in the port's family
# table (python/glmm/__init__.py::_FAMILIES).
_FAMILY = {
    "gaussian": "gaussian",
    "binomial": "binomial",
    "poisson": "poisson",
    "gamma": "gamma",
    "negbin": "negativebinomial",
}


def read_csv_path(path):
    """Read a parity CSV (unquoted header + rows, `,`-split) — mirrors
    harness_common.rs::read_csv_path, deliberately including its naivety: the
    corpus carries no embedded commas, and a csv-module reader here could disagree
    with the Rust side on some future dataset without anyone noticing."""
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
    """Columns keyed by name, typed the way harness_common.rs::build_table types
    them: manifest `factors` are categorical, as is any column that fails to parse
    as f64 anywhere (Pastes' `cask` — carried in the CSV, referenced by no formula).

    A str column reaches `glmm.fit` as a factor with lexicographic levels, which is
    exactly `Column::factor_from_labels` (R's `factor()` default, and what the
    reference side did when the golden was frozen) — so the two engines agree on the
    contrast base without either declaring an order.

    Dots in R-origin headers (Arabidopsis' `total.fruits`) become underscores to
    match jl_formula's sanitized names — mirrors build_table's rename.
    """
    data = {}
    for j, name in enumerate(header):
        values = [r[j] for r in rows]
        is_factor = name in factors or any(not _is_float(v) for v in values)
        data[name.replace(".", "_")] = values if is_factor else [float(v) for v in values]
    return data


def formula_of(spec):
    """The manifest entry's formula, lowered to what the crate's parser takes —
    mirrors fit.rs step for step (see its comments for why each rewrite is safe):
    jl_formula is the source (guaranteed cbind-free), `@formula(...)` unwrapped;
    Julia's `&` grouping operator becomes the parser's `:`; the explicit `1`
    intercept is stripped (the parser treats it as implicit). Rungs without a
    jl_formula (the weights suite's R-only rungs) fall back to r_formula, whose
    aggregated-binomial `cbind(...)` response is rewritten to the `prop` column
    build_fit_data synthesizes.
    """
    jl = spec.get("jl_formula")
    if jl is not None:
        if not (jl.startswith("@formula(") and jl.endswith(")")):
            raise ValueError(f"jl_formula not in @formula(...) shape: {jl}")
        f = jl[len("@formula(") : -1]
    else:
        r = spec.get("r_formula")
        if r is None:
            raise ValueError("manifest entry missing both jl_formula and r_formula")
        resp, sep, rhs = r.partition("~")
        f = f"prop ~{rhs}" if sep and resp.lstrip().startswith("cbind(") else r
    return f.replace(" & ", ":").replace(" ~ 1 + ", " ~ ", 1)


def build_fit_data(spec, header, rows, factors):
    """`(data, weights)` for one manifest entry — the port-side twin of
    harness_common.rs::lower_dataset_generic plus fit.rs's `weights_col` branch.

    An aggregated-binomial rung (manifest `weights`) synthesizes
    `prop = incidence/<weights_col>` so jl_formula's `prop ~ ...` response resolves,
    and passes the cluster sizes as prior weights, one per aggregate row.
    `weights_col` (the weights suite) is plain per-row weights off a named column.
    The two are mutually exclusive per rung by design — asserted, as in fit.rs.
    """
    data = build_data(header, rows, factors)

    def column(name):
        j = header.index(name)  # hoisted: .index() per row is O(rows x cols)
        return [float(r[j]) for r in rows]

    w_name = spec.get("weights")
    if w_name is not None:
        if spec.get("weights_col") is not None:
            raise ValueError("weights_col and weights are mutually exclusive")
        sizes = column(w_name)
        data["prop"] = [i / s for i, s in zip(column("incidence"), sizes)]
        return data, sizes
    wc = spec.get("weights_col")
    if wc is not None:
        if wc not in header:
            raise ValueError(f"weights_col {wc!r} not in CSV header")
        return data, column(wc)
    return data, None


def offset_of(spec, header, rows):
    """Per-row known additive linear-predictor offset (R's `offset=`) -- a
    named CSV column, the plain-lookup counterpart of `weights_col` above
    (no synthesis, unlike the aggregated-binomial `weights` field). Mirrors
    fit.rs's offset handling."""
    oc = spec.get("offset")
    if oc is None:
        return None
    if oc not in header:
        raise ValueError(f"offset {oc!r} not in CSV header")
    j = header.index(oc)
    return [float(r[j]) for r in rows]


def num(x):
    """NaN/Inf -> JSON null (mirrors harness_common.rs::num): a non-converged fit
    leaves NaN-filled estimates, and json.dumps(allow_nan=False) would raise rather
    than write the invalid `NaN` literal the comparators cannot read."""
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def nums(xs):
    return [num(float(x)) for x in xs]


def median_secs(batch, call):
    """Median seconds over N_RUNS samples, warm-up (first) discarded. Each sample
    times `batch` fits so sub-resolution fits stay above the timer floor (the
    manifest `timing_batch` every engine reads) — the median is for `batch` fits;
    summarize_timing.R divides by `fits_per_sample`. Mirrors fit.rs::median_secs,
    including perf_counter as the Instant::now() analogue.
    """
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        for _ in range(batch):
            call()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples[1:])


def group_names_match(a, b):
    """Order-invariant on `:`-joined components — mirrors fit.rs::group_names_match:
    lme4 names a nested inner group `child:parent`, the formula frontend names it
    `parent:child`. A display convention, not a different grouping."""
    return sorted(a.split(":")) == sorted(b.split(":"))


def varcomp(f, ref_order, include_se):
    """Variance components in the common schema, one entry per grouping factor, from
    `Fit.stddev_corr` (arbitrary q). `stddev_se` (GLMM only) comes from the Hessian
    fit's theta block, laid out per theta coordinate: cumulative vech length across
    groupings in DECLARATION order (the layout `Fit.stddev_se` itself uses), gated to
    scalar (q=1) groupings — the only shape the theta==stddev identity holds for,
    same as lme4's own gating. Reindexed to `ref_order` because compare.R aligns
    varcomp POSITIONALLY, not by name. Mirrors fit.rs::varcomp.
    """
    natural = []
    theta_offset = 0
    for i, (name, terms) in enumerate(f.re_groups):
        stddev, corr = f.stddev_corr(i)
        q = len(terms)
        entry = {
            "group": name,
            "terms": list(terms),
            "stddev": nums(stddev),
            "corr": [nums(row) for row in corr],
        }
        if include_se and q == 1:
            entry["stddev_se"] = nums([f.stddev_se[theta_offset]])
        theta_offset += q * (q + 1) // 2
        natural.append(entry)

    out = []
    for name in ref_order:
        idx = next(
            (i for i, (g, _) in enumerate(f.re_groups) if group_names_match(g, name)),
            None,
        )
        if idx is None:
            raise ValueError(f"reference group {name!r} not found in fit's re_groups")
        out.append(natural[idx])
    return out


def fit_one(spec):
    """Fit one manifest entry end-to-end (load -> fit(+SE split) -> time -> reindex
    varcomp to the reference's grouping order -> write). Mirrors fit.rs::fit_one."""
    ds = spec["name"]
    family_str = spec["family"]
    gaussian = family_str == "gaussian"
    factors = spec.get("factors", [])
    source = "simulated" if spec.get("source") == "sim" else "empirical"

    # `data` field: CSV to read when it differs from the rung name — a re-linked rung
    # (cbpp_probit) reuses the committed dataset byte-for-byte.
    data_name = spec.get("data", ds)
    header, rows = read_csv_path(f"{SUITE}/data_{source}/{data_name}.csv")
    data, weights = build_fit_data(spec, header, rows, factors)
    offset = offset_of(spec, header, rows)
    formula = formula_of(spec)
    family = _FAMILY.get(family_str)
    if family is None:
        raise ValueError(f"unsupported family: {family_str}")
    link = spec.get("link")
    timing_batch = spec.get("timing_batch", 1)

    kw = {"link": link, "weights": weights, "offset": offset}

    # Reference grouping order (compare.R aligns varcomp positionally, not by name) —
    # read off the already-frozen lme4 result rather than re-deriving lme4's convention.
    with open(f"{SUITE}/results/lme4_{source}/{ds}.json") as fh:
        reference = json.load(fh)
    ref_order = [e["group"] for e in reference["estimates"]["varcomp"]]

    fh_fit = glmm.fit(data, formula, family, wald_se="hessian", **kw)
    fixed_only = not fh_fit.re_groups

    if gaussian or fixed_only:
        # One SE, no method choice: a gaussian rung has a single profiled `se`, and a
        # fixed-only GLM (weights suite) has no theta, so the Rx-vs-Hessian split is
        # moot. Emitted in the slot fit.rs uses for each — `se` for gaussian, `se_rx`
        # for fixed-only — so compare.R lines them up with the other engines.
        timing = {
            "fit_seconds_median": median_secs(
                timing_batch, lambda: glmm.fit(data, formula, family, wald_se="hessian", **kw)
            ),
            "n_runs": N_RUNS,
            "warmup_discarded": 1,
            "fits_per_sample": timing_batch,
        }
        estimates = {
            "beta": nums(fh_fit.beta),
            ("se" if gaussian else "se_rx"): nums(fh_fit.se),
            "loglik": num(fh_fit.loglik),
            "df": fh_fit.df,
            "varcomp": varcomp(fh_fit, ref_order, False),
        }
        converged, n_eval, deviance = fh_fit.converged, fh_fit.n_eval, fh_fit.deviance
    else:
        # GLMM SE has two genuinely different Laplace variants — emit both so compare.R
        # checks like to like: se_hessian keeps the theta-beta coupling (glmm default),
        # se_rx is conditional on theta-hat. beta/tau is wald_se-independent.
        fr_fit = glmm.fit(data, formula, family, wald_se="rx", **kw)
        # Split timing by SE method — the FD-Hessian is the main time consumer, Rx is
        # one closed-form Schur solve. Same PIRLS fit underlies both.
        timing = {
            "fit_seconds_median_rx": median_secs(
                timing_batch, lambda: glmm.fit(data, formula, family, wald_se="rx", **kw)
            ),
            "fit_seconds_median_hessian": median_secs(
                timing_batch, lambda: glmm.fit(data, formula, family, wald_se="hessian", **kw)
            ),
            "n_runs": N_RUNS,
            "warmup_discarded": 1,
            "fits_per_sample": timing_batch,
        }
        estimates = {
            "beta": nums(fh_fit.beta),
            "se_hessian": nums(fh_fit.se),
            "se_rx": nums(fr_fit.se),
            "loglik": num(fh_fit.loglik),
            "df": fh_fit.df,
            # stddev_se from the Hessian fit's theta block.
            "varcomp": varcomp(fh_fit, ref_order, True),
        }
        converged = fh_fit.converged and fr_fit.converged
        n_eval = fh_fit.n_eval + fr_fit.n_eval
        deviance = fh_fit.deviance

    res = {
        "dataset": ds,
        "engine": "glmm_python",
        "engine_version": f"{VERSION}-local",
        "family": family_str,
        "reml": spec.get("reml", False) if gaussian else None,
        "rung": spec["rung"],
        "converged": converged,
        # The real flag, unlike fit.rs's hardcoded `false` (which predates
        # Fit::singular being surfaced). Ungated by compare.R, so the two engines
        # differing here is a stale hardcode on the Rust side, not a port bug.
        "singular": fh_fit.singular,
        "optimizer": "bobyqa",
        "n_eval": n_eval,
        "deviance": num(deviance),
        "coef_names": fh_fit.names,
        "estimates": estimates,
        "timing": timing,
    }
    write_result(ds, source, res)


def write_result(ds, source, res):
    out = f"{SUITE}/results/glmm_python_{source}/{ds}.json"
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2, allow_nan=False)
        fh.write("\n")
    t = res["timing"].get("fit_seconds_median", res["timing"].get("fit_seconds_median_rx"))
    print(
        f"glmm_py  {ds:<12}  rung {res['rung']}  converged={res['converged']}  fit_median={t:.4f}s",
        flush=True,
    )


def main():
    for source in ("empirical", "simulated"):
        os.makedirs(f"{SUITE}/results/glmm_python_{source}", exist_ok=True)
    with open(f"{SUITE}/manifest.json") as fh:
        manifest = json.load(fh)
    # PARITY_ONLY=<name>[,<name>...]: fit only the named datasets (mirrors the other
    # engines) — reruns a single rung without repaying the full-corpus timing cost.
    only = os.environ.get("PARITY_ONLY", "")
    want = (lambda ds: True) if not only else (lambda ds: ds in only.split(","))
    for spec in manifest["datasets"]:
        if want(spec["name"]):
            fit_one(spec)


if __name__ == "__main__":
    sys.exit(main())
