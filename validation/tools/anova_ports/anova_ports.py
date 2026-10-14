"""The Python front end of the anova port gate: the call list of
`anova_ports.rs`, made through the Python port (`glmm.anova`, `glmm.compare`),
each table written to stdout as one JSONL row per cell. Mirrors
`anova_ports.rs` and `anova_ports.R` call for call: change together. The call
list and the cell format are documented in `anova_ports.rs`.

Run from the repository root:
    python/venv/bin/python validation/tools/anova_ports/anova_ports.py > anova_py.jsonl
"""

import json
import math
import os
import sys

import glmm

VALIDATION_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

PAIR_CASES = [
    "random_pair",
    "ml_pair",
    "ols_pair",
    "glm_pair",
    "glmm_pair",
    "glmm_vs_glm",
    "gamma_glmm_vs_glm",
]

# (label, data, factors, formula, family, link)
TERM_MODELS = [
    (
        "lmer_sim_factorial",
        "data/simulated/sim_factorial.csv",
        ["subj", "g", "h"],
        "y ~ g * h + x + (1 + x | subj)",
        "gaussian",
        "identity",
    ),
    (
        "lm_sim_factorial",
        "data/simulated/sim_factorial.csv",
        ["subj", "g", "h"],
        "y ~ g * h + x",
        "gaussian",
        "identity",
    ),
    (
        "glmer_sim_factorial_bern",
        "data/simulated/sim_factorial_bern.csv",
        ["subj", "g", "h"],
        "y ~ g * h + x + (1 | subj)",
        "binomial",
        "logit",
    ),
    (
        "glm_sim_factorial_bern",
        "data/simulated/sim_factorial_bern.csv",
        ["subj", "g", "h"],
        "y ~ g * h + x",
        "binomial",
        "logit",
    ),
    (
        "glmer_sim_binomial_slope1",
        "data/simulated/sim_binomial_slope1.csv",
        ["g"],
        "y ~ x + (1 + x | g)",
        "binomial",
        "logit",
    ),
    (
        "glmer_cbpp",
        "data/empirical/cbpp.csv",
        ["herd", "period"],
        "cbind(incidence, size - incidence) ~ period + (1 | herd)",
        "binomial",
        "logit",
    ),
    (
        "glmer_poisson_grouseticks",
        "data/empirical/grouseticks.csv",
        ["YEAR", "BROOD"],
        "TICKS ~ YEAR + cHEIGHT + (1 | BROOD)",
        "poisson",
        "log",
    ),
    (
        "glmer_nb_sim_nb",
        "data/simulated/sim_nb.csv",
        ["cluster", "grp"],
        "y ~ 1 + x + grp + (1 | cluster)",
        "negativebinomial",
        "log",
    ),
    (
        "glmer_gamma_sim_gamma",
        "data/simulated/sim_gamma.csv",
        ["cluster", "grp"],
        "y ~ 1 + x + grp + (1 | cluster)",
        "gamma",
        "log",
    ),
]

# One fit's per-term calls, as anova_ports.rs's TERM_CALLS: (model label,
# types, test, named). The fit is made with anova_test="LRT" for test="LRT"
# and at the default "auto" otherwise; a call is named terms:<label>:<type>,
# with :<test> appended when named.
TERM_CALLS = [
    ("lmer_sim_factorial", ["III", "II"], "auto", False),
    ("lm_sim_factorial", ["III", "II"], "auto", False),
    ("glmer_sim_factorial_bern", ["III", "II"], "auto", False),
    ("glm_sim_factorial_bern", ["III", "II"], "auto", False),
    ("glmer_sim_factorial_bern", ["III", "II"], "LRT", True),
    ("glm_sim_factorial_bern", ["III", "II"], "LRT", True),
    ("glmer_cbpp", ["III", "II"], "LRT", True),
    ("glmer_sim_binomial_slope1", ["III"], "LRT", True),
    ("glmer_poisson_grouseticks", ["III"], "auto", True),
    ("glmer_poisson_grouseticks", ["III"], "Wald", True),
    ("glmer_nb_sim_nb", ["III"], "auto", True),
    ("glmer_gamma_sim_gamma", ["III"], "auto", True),
]

MISSING_ROW = 4  # 0-based row of wls_basic.csv whose x2 is made missing


def read_data(path, factors):
    """The columns of one CSV, typed as the grid's engines type them: a column
    named in `factors`, or one with a cell that is not a number, is a list of
    labels (the port makes it a factor with its labels in byte order)."""
    with open(os.path.join(VALIDATION_DIR, path)) as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]

    def split(line):
        return [s.strip().strip('"') for s in line.split(",")]

    header = split(lines[0])
    rows = [split(ln) for ln in lines[1:]]
    data = {}
    for j, name in enumerate(header):
        cells = [r[j] for r in rows]
        try:
            values = [float(c) for c in cells]
        except ValueError:
            values = None
        data[name] = cells if values is None or name in factors else values
    return data


def aggregated(formula, data):
    """`cbind(s, t - s) ~ ...` as `cbind(s, failures) ~ ...` with the column
    `failures = t - s` added to `data`; any other formula as it is."""
    if not formula.startswith("cbind("):
        return formula
    inner, tail = formula[len("cbind(") :].split(")", 1)
    succ, fail = inner.split(",", 1)
    if "-" not in fail:
        return formula
    total, minus = (s.strip() for s in fail.split("-", 1))
    data["failures"] = [t - s for t, s in zip(data[total], data[minus])]
    return f"cbind({succ.strip()}, failures){tail}"


def member(m, data):
    return {
        "formula": aggregated(m["formula"], data),
        "family": m["family"],
        "link": m["link"],
        "reml": m["REML"],
        "nagq": m["nAGQ"] or 1,
    }


def cell_value(v):
    """A number as it is; a non-finite one as "NaN", "Inf" or "-Inf"."""
    if math.isfinite(v):
        return v
    return "NaN" if math.isnan(v) else ("Inf" if v > 0 else "-Inf")


def table(call, rows, cols):
    """One JSONL row per cell: `rows` in order, `cols` as (name, values)."""
    for i, row in enumerate(rows):
        for j, (col, values) in enumerate(cols):
            v = float(values[i])
            print(
                json.dumps(
                    {
                        "call": call,
                        "row": row,
                        "col": col,
                        "row_index": i,
                        "col_index": j,
                        "value": cell_value(v),
                    }
                )
            )


def text(call, row, col, at, s):
    """One text cell (a heading, a tag) at `at`, (row_index, col_index)."""
    print(
        json.dumps(
            {
                "call": call,
                "row": row,
                "col": col,
                "row_index": at[0],
                "col_index": at[1],
                "text": s,
            }
        )
    )


def refused(call, why):
    print(f"{call}: refused: {why}", file=sys.stderr)
    table(call, [""], [("error", [math.nan])])


def port_table(call, tab):
    """A port table (`ComparisonTable` or `TermTable`) as it is: its row
    names and its columns, in order."""
    table(call, list(tab.row_names), [(c, list(tab[c])) for c in tab.columns])


def fit(data, m, weights, anova_test="auto"):
    return glmm.fit(
        data,
        m["formula"],
        m["family"],
        link=m["link"],
        weights=weights,
        nagq=m["nagq"],
        reml=m["reml"],
        anova_test=anova_test,
    )


def anova_call(call, data, m, weights):
    try:
        port_table(call, glmm.anova(fit(data, m[0], weights), fit(data, m[1], weights)))
    except ValueError as e:
        refused(call, e)


def compare_call(call, data, m, weights, nobs=False):
    try:
        c = glmm.compare(
            data,
            m[0]["formula"],
            m[1]["formula"],
            m[0]["family"],
            link=m[0]["link"],
            weights=weights,
            nagq=max(m[0]["nagq"], m[1]["nagq"]),
        )
    except ValueError as e:
        refused(call, e)
        return
    port_table(call, c.table)
    if nobs:
        table(f"{call}:nobs", ["fit0", "fit1"], [("nobs", [c.fit0.nobs, c.fit1.nobs])])


def pairs():
    with open(os.path.join(VALIDATION_DIR, "manifest.json")) as fh:
        manifest = json.load(fh)
    for e in manifest["inference_references"]:
        if e.get("kind") != "anova" or e.get("case") not in PAIR_CASES:
            continue
        name = e["name"]
        data = read_data(e["data"], e["factors"])
        members = e["members"]
        m = [member(members[0], data), member(members[1], data)]
        w = members[0]["weights"]
        weights = None if w is None else data[w]
        compare_call(f"compare:{name}", data, m, weights)
        anova_call(f"anova:{name}", data, m, weights)


def term_calls():
    """`glmm.anova(m, type=, test=)` per TERM_CALLS, each table followed by
    its text cells: the heading and notes, and the fit's `term_test_auto`."""
    models = {t[0]: t[1:] for t in TERM_MODELS}
    for label, types, test, named in TERM_CALLS:
        path, factors, formula, family, link = models[label]
        data = read_data(path, factors)
        m = {
            "formula": aggregated(formula, data),
            "family": family,
            "link": link,
            "reml": None,
            "nagq": 1,
        }
        calls = [f"terms:{label}:{ss}" + (f":{test}" if named else "") for ss in types]
        try:
            f = fit(data, m, None, anova_test="LRT" if test == "LRT" else "auto")
        except ValueError as e:
            for call in calls:
                refused(call, e)
            continue
        for ss, call in zip(types, calls):
            try:
                tab = glmm.anova(f, type=ss, test=test)
            except ValueError as e:
                refused(call, e)
                continue
            port_table(call, tab)
            n, k = len(tab.row_names), len(tab.columns)
            text(call, "heading", "text", (n, k), "\n".join([tab.heading, *tab.notes]))
            text(call, "term_test_auto", "test", (n + 1, k + 1), f.term_test_auto["test"])
            text(call, "term_test_auto", "reason", (n + 1, k + 2), f.term_test_auto["reason"])


def missing_value_call():
    data = read_data("data/simulated/wls_basic.csv", [])
    data["x2"][MISSING_ROW] = math.nan
    m = [
        {"formula": f, "family": "gaussian", "link": "identity", "reml": None, "nagq": 1}
        for f in ("y ~ x1", "y ~ x1 + x2")
    ]
    compare_call("compare:missing_x2_wls_basic", data, m, data["w"], nobs=True)


def rlrt_calls():
    data = read_data("data/empirical/Dyestuff.csv", ["Batch"])
    m = [
        {"formula": f, "family": "gaussian", "link": "identity", "reml": True, "nagq": 1}
        for f in ("Yield ~ 1", "Yield ~ 1 + (1 | Batch)")
    ]
    compare_call("compare:rlrt_dyestuff", data, m, None)
    anova_call("anova:rlrt_dyestuff", data, m, None)


if __name__ == "__main__":
    pairs()
    term_calls()
    missing_value_call()
    rlrt_calls()
