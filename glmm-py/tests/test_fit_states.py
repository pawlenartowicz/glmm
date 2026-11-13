"""Every fit state the kernel reaches (converged, singular, failed, aliased),
on every model shape where it exists, through every public member of
`glmm.Fit` and the module calls that read a fit. A call must return or raise
`ValueError`; on a fit with no estimates (every coefficient NaN) every
estimate it returns is NaN; per-grouping outputs come in the order the formula
declares the groupings; and the summary and the pairwise notes match the fit.

The data table is deterministic (no RNG) and is built the same way in
`glmm-rs/tests/fit_states.rs` and `glmm-r/tests/testthat/test-fit-states.R`:
change the three together."""

import dataclasses
import functools
import math
import re
import warnings

import numpy as np
import pytest

import glmm
from glmm.anova import _NOT_CONVERGED

N = 96

# Shape A has no random effects; B, C and D add these terms.
_RANDOM = {"A": "", "B": " + (1 | g)", "C": " + (1 | a) + (1 | b)", "D": " + (1 + x | g)"}

# The severe warning each failed state raises.
_FAILURE_KIND = {
    "constant_response": "constant_response",
    "separated": "glm_diverged",
    "too_few_rows": "too_few_rows",
}

_INTERVALS_NOT_CONVERGED = (
    "The fit did not converge, so its estimates are not at a maximum and these intervals "
    "are not reliable. Change the model until it converges (a simpler random part often "
    "helps), then compute them again."
)


def _rows():
    """The 39 fits: every (state, family, shape) that exists."""
    rows = []
    for family in ("gaussian", "binomial"):
        rows += [("converged", family, s) for s in "ABCD"]
        rows += [("singular", family, s) for s in "BCD"]
        rows += [("too_few_rows", family, s) for s in "ABCD"]
        rows += [("aliased_fixed", family, s) for s in "ABCD"]
    rows += [("constant_response", "gaussian", s) for s in "ABCD"]
    rows += [("separated", "binomial", s) for s in "ABCD"]
    rows.append(("aliased_slope", "gaussian", "D"))
    return rows


def _formula(state, shape):
    if state == "aliased_fixed":
        return f"y ~ f + x + x2{_RANDOM[shape]}"
    if state == "aliased_slope":
        return "y ~ f + x + x2 + (1 + x2 | g)"
    return f"y ~ f + x{_RANDOM[shape]}"


def _groupings(state, shape):
    """The declared groupings and the random effects each one carries."""
    if state == "aliased_slope" or shape == "D":
        return [("g", 2)]
    return {"A": [], "B": [("g", 1)], "C": [("a", 1), ("b", 1)]}[shape]


def _mean_by(values, key):
    return (np.bincount(key, weights=values) / np.bincount(key))[key]


def _data(state, family, shape):
    i = np.arange(N)
    x = np.sin(1.7 * i + 0.3)
    z = 0.8 * np.sin(2.3 * i + 0.7)
    g, a, b = (i // 3) % 6, i % 4, (i // 4) % 12
    if state == "constant_response":
        y = np.full(N, 3.0)
    elif state == "separated":
        y = (x > 0).astype(float)
    elif state == "singular" and family == "binomial":
        y = ((i // 16) % 2 == 0).astype(float)
    elif state == "singular" and shape == "C":
        y = z - _mean_by(z, a) - _mean_by(z, b) + z.mean()
    elif state == "singular":
        y = z - _mean_by(z, g)
    else:
        u = np.array([-1.0, 0.6, 0.2, -0.4, 1.1, -0.5])
        s = np.array([0.5, -0.4, 0.3, -0.6, 0.2, 0.4])
        ua = np.array([-0.8, 0.5, 0.9, -0.6])
        random = {
            "A": 0.0,
            "B": u[g],
            "C": ua[a] + 0.7 * np.sin(b + 1.0),
            "D": u[g] + s[g] * x,
        }["D" if state == "aliased_slope" else shape]
        eta = 0.5 * x + np.array([0.0, 0.3, -0.4])[i % 3] + random
        if family == "binomial":
            t = 0.618034 * i + 0.1
            y = (t - np.floor(t) < 1.0 / (1.0 + np.exp(-2.0 * eta))).astype(float)
        else:
            y = 1.0 + eta + z
    # A too-few-rows fit keeps 4 rows, as many as the fixed part has columns.
    keep = [0, 13, 26, 39] if state == "too_few_rows" else slice(None)
    return {
        "y": y[keep],
        "f": np.array(["p", "q", "r"])[i % 3][keep],
        "g": np.array([f"g{v}" for v in g])[keep],
        "a": np.array([f"a{v}" for v in a])[keep],
        "b": np.array([f"b{v}" for v in b])[keep],
        "x": x[keep],
        "x2": 2.0 * x[keep],
    }


@dataclasses.dataclass
class Ctx:
    state: str
    family: str
    shape: str
    data: dict
    formula: str
    m: glmm.Fit
    # The row's formula without `x` in the fixed part, for the two-fit calls;
    # the ValueError when it did not fit.
    reduced_formula: str
    reduced: object

    @property
    def label(self):
        return f"{self.state}/{self.family}/{self.shape}"

    @property
    def no_estimates(self):
        return bool(np.all(np.isnan(self.m.beta)))


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@functools.cache
def _fits():
    fits = []
    for state, family, shape in _rows():
        data = _data(state, family, shape)
        formula = _formula(state, shape)
        reduced_formula = formula.replace(" + x", "", 1)
        m = _quiet(glmm.fit, data, formula, family)
        try:
            reduced = _quiet(glmm.fit, data, reduced_formula, family)
        except ValueError as e:
            reduced = e
        fits.append(Ctx(state, family, shape, data, formula, m, reduced_formula, reduced))
    return fits


def _flat(*parts):
    """Every number of `parts` (arrays, lists, scalars; None skipped) as one
    float array."""
    out = []
    for p in parts:
        if p is None:
            continue
        if isinstance(p, (list, tuple)):
            out.extend(_flat(*p))
        else:
            out.extend(np.asarray(p, dtype=float).ravel())
    return np.asarray(out, dtype=float)


# Count and label columns of the tables: not estimates.
_NOT_ESTIMATE_COLUMNS = {"contrast", "NumDF", "Df", "npar", "Res.Df"}


def _table(t, row=None):
    """The estimate columns of a table, all rows or one; an infinite `df`
    marks a z row, a label of the test, not an estimate."""
    out = []
    for c in t.columns:
        if c in _NOT_ESTIMATE_COLUMNS or any(isinstance(v, str) for v in t[c]):
            continue
        v = np.asarray(t[c], dtype=float)
        if row is not None:
            v = v[row : row + 1]
        out.extend(v[~np.isposinf(v)] if c == "df" else v)
    return np.asarray(out, dtype=float)


def _term_dict(d):
    """The estimate columns of a stored per-term table (`term_table_type3` and
    its siblings)."""
    if d is None or d["error"] is not None:
        return _flat()
    keys = ("statistic", "p_value", "sum_sq", "mean_sq", "den_df", "eta2_partial")
    return _flat(*(d[k] for k in keys if k in d))


def _stddev_corr(m):
    """Each grouping's SDs and its correlations below the diagonal: the
    diagonal is 1 by definition, not an estimate."""
    out = []
    for g in range(len(m.re_groups)):
        sd, corr = m.stddev_corr(g)
        out += [*sd, *corr[np.tril_indices(len(sd), -1)]]
    return _flat(out)


def _comparison(c, t):
    """This fit's row of a two-model table: the other model can have
    estimates where this one has none."""
    row = [f for _, f in t.formulas].index(c.formula)
    return _table(t, row)


def _summary_numbers(m):
    s = m.summary_object()
    re_parts = [[r["variance"], r["stddev"], r["corr"]] for r in s.random_effects]
    return _flat(
        list(s.criterion.values()),
        s.scaled_residuals,
        re_parts,
        s.residual_variance,
        s.estimate,
        s.std_error,
        s.coef_df,
        s.statistic,
        s.p_value,
        s.residual_df,
        s.rss,
        s.r_squared,
        s.adj_r_squared,
        s.r2_marginal,
        s.r2_conditional,
        s.icc_adjusted,
        s.dispersion,
    )


def _predict(c, type):
    p = c.m.predict(newdata=c.data, type=type, se_fit=True)
    return _flat(p["fit"], p["se_fit"], p["lower"], p["upper"])


def _ranef_blocks(m):
    return _flat([[b["values"], b["postvar"], b["condsd"]] for b in m.ranef_blocks])


def _satterthwaite_record(m):
    rec = m.satterthwaite_record
    if rec is None:
        return _flat()
    return _flat(rec["a_eff"], rec["jac_theta"], rec["c"], rec["tau"], rec["nu"])


def _robust(m):
    rec = m.robust
    if rec is None:
        return _flat()
    return _flat(*(rec[k] for k in ("er", "f", "epe", "row_e", "row_x", "row_phi", "null")))


def _none(value):
    """A member with no numbers (a string, a list of names): read, nothing returned."""
    return _flat()


# Every public member of `glmm.Fit`, read or called with fixed arguments, and
# the module calls that read a fit: name -> (its numbers are estimates, NaN on
# a fit with no estimates; the call). A name starting `glmm.` is a module call;
# for the others the part before any `(` is the member, the list
# `test_every_public_fit_member_is_called_or_skipped` checks.
CALLS = {
    "adj_r_squared": (True, lambda c: _flat(c.m.adj_r_squared)),
    "aic": (True, lambda c: _flat(c.m.aic)),
    "aliased": (False, lambda c: _flat(c.m.aliased)),
    "beta": (True, lambda c: _flat(c.m.beta)),
    "bic": (True, lambda c: _flat(c.m.bic)),
    "coef_df": (True, lambda c: _flat(c.m.coef_df)),
    "coef_p": (True, lambda c: _flat(c.m.coef_p)),
    "confint": (True, lambda c: _flat(c.m.confint())),
    "contrast_test": (
        True,
        lambda c: _table(c.m.contrast_test(np.eye(len(c.m.beta))[:1])),
    ),
    "converged": (False, lambda c: _flat(c.m.converged)),
    "design_summary": (False, lambda c: _flat(*(c.m.design_summary or {}).values())),
    "deviance": (True, lambda c: _flat(c.m.deviance)),
    "df": (False, lambda c: _flat(c.m.df)),
    "df_method": (False, lambda c: _none(c.m.df_method)),
    "diagnostics": (False, lambda c: _flat(len(c.m.diagnostics["notes"]))),
    "dispersion": (True, lambda c: _flat(c.m.dispersion)),
    "dispersion_held": (False, lambda c: _flat(c.m.dispersion_held)),
    "estimated_weight_power": (False, lambda c: _flat(c.m.estimated_weight_power)),
    "family": (False, lambda c: _none(c.m.family)),
    "fitted": (True, lambda c: _flat(c.m.fitted)),
    "formula": (False, lambda c: _none(c.m.formula)),
    "icc": (True, lambda c: _flat(list(c.m.icc().values()))),
    "link": (False, lambda c: _none(c.m.link)),
    "loglik": (True, lambda c: _flat(c.m.loglik)),
    "model_vcov": (True, lambda c: _flat(c.m.model_vcov)),
    "n_eval": (False, lambda c: _flat(c.m.n_eval)),
    "n_grad": (False, lambda c: _flat(c.m.n_grad)),
    "nagq": (False, lambda c: _flat(c.m.nagq)),
    "names": (False, lambda c: _none(c.m.names)),
    "nobs": (False, lambda c: _flat(c.m.nobs)),
    "objective": (False, lambda c: _none(c.m.objective)),
    "objective_nodes": (False, lambda c: _flat(c.m.objective_nodes)),
    "offset": (False, lambda c: _flat(c.m.offset)),
    # A measure of the gradient search, NaN on every fit that ran none.
    "outer_decrement": (False, lambda c: _flat(c.m.outer_decrement)),
    "outer_stop": (False, lambda c: _none(c.m.outer_stop)),
    "predict(type='link')": (True, lambda c: _predict(c, "link")),
    "predict(type='response')": (True, lambda c: _predict(c, "response")),
    "profile_confint": (True, lambda c: _table(c.m.profile_confint())),
    "r2": (True, lambda c: _flat(list(c.m.r2().values()))),
    "r_squared": (True, lambda c: _flat(c.m.r_squared)),
    "ranef": (True, lambda c: _flat(c.m.ranef)),
    "ranef_blocks": (True, lambda c: _ranef_blocks(c.m)),
    "ranef_levels": (False, lambda c: _flat(c.m.ranef_levels)),
    "re_groups": (False, lambda c: _none(c.m.re_groups)),
    "re_term_index": (False, lambda c: _flat(c.m.re_term_index)),
    "reml": (False, lambda c: _flat(c.m.reml)),
    "residuals('response')": (True, lambda c: _flat(c.m.residuals("response"))),
    "residuals('pearson')": (True, lambda c: _flat(c.m.residuals("pearson"))),
    "robust": (True, lambda c: _robust(c.m)),
    "rss": (True, lambda c: _flat(c.m.rss)),
    "satterthwaite_record": (True, lambda c: _satterthwaite_record(c.m)),
    "se": (True, lambda c: _flat(c.m.se)),
    "se_type": (False, lambda c: _none(c.m.se_type)),
    "simulate": (True, lambda c: _flat(c.m.simulate(nsim=2, seed=1))),
    "singular": (False, lambda c: _flat(c.m.singular)),
    "stddev_corr": (True, lambda c: _stddev_corr(c.m)),
    "stddev_se": (True, lambda c: _flat(c.m.stddev_se)),
    "summary": (False, lambda c: _none(c.m.summary())),
    "summary_object": (True, lambda c: _summary_numbers(c.m)),
    "tau2": (True, lambda c: _flat(c.m.tau2)),
    "term_table_type2": (True, lambda c: _term_dict(c.m.term_table_type2)),
    "term_table_type2_lrt": (True, lambda c: _term_dict(c.m.term_table_type2_lrt)),
    "term_table_type3": (True, lambda c: _term_dict(c.m.term_table_type3)),
    "term_table_type3_lrt": (True, lambda c: _term_dict(c.m.term_table_type3_lrt)),
    "term_test_auto": (False, lambda c: _none(c.m.term_test_auto)),
    "varcorr": (True, lambda c: _flat(c.m.varcorr)),
    "vcov": (True, lambda c: _flat(c.m.vcov)),
    "warnings": (False, lambda c: _none(c.m.warnings)),
    # NaN on a fit without weights, which every row is.
    "weight_power": (True, lambda c: _flat(c.m.weight_power)),
    "weights": (False, lambda c: _flat(c.m.weights)),
    "y": (False, lambda c: _flat(c.m.y)),
    "glmm.anova(m)": (True, lambda c: _table(glmm.anova(c.m))),
    "glmm.anova(m0, m)": (True, lambda c: _comparison(c, _two_fit_anova(c))),
    "glmm.compare": (
        True,
        lambda c: _comparison(
            c, glmm.compare(c.data, c.reduced_formula, c.formula, c.family).table
        ),
    ),
    "glmm.pairwise": (True, lambda c: _table(glmm.pairwise(c.m, "f"))),
    "glmm.pairwise(...).confint": (True, lambda c: _table(glmm.pairwise(c.m, "f").confint())),
}

# Public members of `glmm.Fit` no row reads, with the reason.
SKIP = {}


def _two_fit_anova(c):
    if isinstance(c.reduced, ValueError):
        raise c.reduced
    return glmm.anova(c.reduced, c.m)


def _summary_problems(c):
    """The summary's title and groups line against `re_groups`, and the one
    severe warning of a failed fit against its kind."""
    bad = []
    text = c.m.summary_object().text()
    lines = text.splitlines()
    mixed = len(c.m.re_groups) > 0
    if ("mixed model" in lines[0]) != mixed:
        bad.append(f"title {lines[0]!r} with re_groups {c.m.re_groups}")
    n = len(c.data["y"])
    declared = _groupings(c.state, c.shape)
    groups = "; ".join(f"{name}, {len(set(c.data[name]))}" for name, _ in declared)
    want = f"Number of obs: {n}, groups:  {groups}" if declared else f"Number of obs: {n}"
    if want not in lines:
        got = [ln for ln in lines if ln.startswith("Number of obs")]
        bad.append(f"groups line {got} for {want!r}")
    kind = _FAILURE_KIND.get(c.state)
    if kind is not None and not (c.state == "constant_response" and c.shape == "A"):
        severe = [w["kind"] for w in c.m.warnings if w["tier"] == "severe"]
        if severe != [kind]:
            bad.append(f"severe warnings {severe} for {kind!r}")
    return bad


def _pairwise_problems(c, returned):
    """On a fit that did not converge, the first notes of `pairwise` and of
    its `confint` say so. A row where `pairwise` returned is added to `returned`."""
    if c.m.converged:
        return []
    try:
        table = _quiet(glmm.pairwise, c.m, "f")
    except ValueError:
        return []
    returned.append(c.label)
    bad = []
    if table.notes[:1] != [_NOT_CONVERGED]:
        bad.append(f"pairwise notes {table.notes}")
    ci = _quiet(table.confint)
    if ci.notes[:1] != [_INTERVALS_NOT_CONVERGED]:
        bad.append(f"pairwise confint notes {ci.notes}")
    return bad


def test_every_fit_state_reaches_its_state():
    failures = []
    fits = _fits()
    for c in fits:
        m = c.m
        separated = any(n["kind"] == "separated" for n in m.diagnostics["notes"])
        x2 = m.names.index("x2") if "x2" in m.names else None
        x2_aliased = x2 is not None and bool(m.aliased[x2]) and math.isnan(m.beta[x2])
        if c.state == "converged":
            ok = m.converged and not m.singular
        elif c.state == "singular":
            ok = m.converged and m.singular
        elif c.state == "constant_response" and c.shape == "A":
            ok = m.converged
        elif c.state in ("constant_response", "too_few_rows"):
            ok = not m.converged and c.no_estimates
        elif c.state == "separated":
            ok = not m.converged and separated
        else:
            ok = m.converged and x2_aliased
        if not ok:
            failures.append(
                f"{c.label}: converged {m.converged}, singular {m.singular}, beta {m.beta}, "
                f"aliased {m.aliased}, notes {[n['kind'] for n in m.diagnostics['notes']]}"
            )
    assert len(fits) == 39
    assert not failures, "\n".join(failures)


def test_every_call_on_every_fit_returns_or_refuses():
    failures = []
    ran = 0
    pairwise_returned = []
    fits = _fits()
    for c in fits:
        for name, (estimate, call) in CALLS.items():
            ran += 1
            at = f"{c.label}: {name}"
            try:
                values = _quiet(call, c)
            except ValueError:
                continue
            except Exception as e:  # noqa: BLE001 - any other exception fails the row
                failures.append(f"{at}: raised {type(e).__name__}: {e}")
                continue
            if estimate and c.no_estimates and not np.all(np.isnan(values)):
                failures.append(f"{at}: a number is not NaN on a fit with no estimates: {values}")
            if name == "simulate" and c.m.converged and not np.all(np.isfinite(values)):
                failures.append(f"{at}: a draw is not finite on a converged fit: {values}")
        failures += [f"{c.label}: summary: {b}" for b in _summary_problems(c)]
        if c.label == "too_few_rows/gaussian/A":
            text = c.m.summary_object().text()
            if "on NA degrees of freedom" not in text:
                failures.append(f"{c.label}: summary footer has no 'on NA degrees of freedom'")
        failures += [f"{c.label}: pairwise: {b}" for b in _pairwise_problems(c, pairwise_returned)]
    assert (len(fits), ran) == (39, 39 * len(CALLS))
    assert len(pairwise_returned) == 15, pairwise_returned
    assert "separated/binomial/A" in pairwise_returned
    assert not failures, "\n".join(failures)


def test_per_grouping_outputs_follow_the_declaration_order():
    failures = []
    checked = 0
    for c in _fits():
        declared = _groupings(c.state, c.shape)
        if not declared:
            continue
        checked += 1
        m = c.m
        names = [name for name, _ in declared]
        qs = [q for _, q in declared]
        # Level counts from the table itself: a too-few-rows table keeps 4 of each.
        levels = [len(set(c.data[name])) for name in names]
        bad = []
        got = [(name, len(terms)) for name, terms in m.re_groups]
        if got != declared:
            bad.append(f"re_groups {got}")
        if list(m.ranef_levels) != levels:
            bad.append(f"ranef_levels {list(m.ranef_levels)} for {levels}")
        blocks = [b["group"] for b in m.ranef_blocks]
        if blocks not in ([], names):
            bad.append(f"ranef_blocks groups {blocks}")
        if np.asarray(m.ranef).size != sum(lv * q for lv, q in zip(levels, qs)):
            bad.append(f"ranef size {np.asarray(m.ranef).size}")
        vc = [len(v) for v in m.varcorr]
        if vc != [q * (q + 1) // 2 for q in qs]:
            bad.append(f"varcorr lengths {vc}")
        try:
            rows = _quiet(m.profile_confint).row_names
        except ValueError:
            rows = None
        if rows is not None:
            suffixes = [r.rsplit("|", 1)[1] for r in rows if re.match(r"(sd|cor)_", r)]
            order = [names.index(s) if s in names else -1 for s in suffixes]
            if -1 in order or order != sorted(order):
                bad.append(f"profile_confint rows {rows}")
        failures += [f"{c.label}: {b}" for b in bad]
    assert checked == 31
    assert not failures, "\n".join(failures)


def _public_members():
    return {n for n in dir(glmm.Fit) if not n.startswith("_")} | {
        f.name for f in dataclasses.fields(glmm.Fit) if not f.name.startswith("_")
    }


def test_every_public_fit_member_is_called_or_skipped():
    found = _public_members()
    listed = {n.split("(")[0] for n in CALLS if not n.startswith("glmm.")} | set(SKIP)
    assert len(found) >= 69, sorted(found)
    missing, stale = found - listed, listed - found
    assert not missing and not stale, (
        f"not called and not skipped: {sorted(missing)}; "
        f"listed but not a public member of glmm.Fit: {sorted(stale)}"
    )


def test_a_fit_refused_before_fitting_raises_value_error():
    empty = {"y": np.array([]), "x": np.array([]), "g": np.array([])}
    with pytest.raises(
        ValueError, match=r"^Data has no rows \(n = 0\); a fit needs at least one row$"
    ):
        glmm.fit(empty, "y ~ x + (1 | g)")
    with pytest.raises(
        ValueError,
        match=r"^formula syntax error: the formula has no response; write it as y ~ x$",
    ):
        glmm.fit(_data("converged", "gaussian", "B"), "~ x")


def test_predicting_the_training_rows_finds_every_row_estimable():
    failures = []
    fits = _fits()
    predicted = aliased_predicted = 0
    for c in fits:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                c.m.predict(newdata=c.data, re_form="NA")
            except ValueError:
                continue
        predicted += 1
        aliased_predicted += c.state.startswith("aliased")
        if any(issubclass(w.category, glmm.RowsNotEstimableWarning) for w in caught):
            failures.append(f"{c.label}: a training row is not estimable")
    assert len(fits) == 39
    assert predicted == 39
    assert aliased_predicted == 9
    assert not failures, "\n".join(failures)


def test_no_fit_is_at_the_negative_binomial_shape_limit():
    fits = _fits()
    failures = [
        c.label
        for c in fits
        if any(n["kind"] == "nb_shape_at_limit" for n in c.m.diagnostics["notes"])
    ]
    assert len(fits) == 39
    assert not failures, "\n".join(failures)


def test_one_trial_per_level_marks_the_one_row_per_level_fits():
    # The too-few-rows rows 0, 13, 26 and 39 fall in four different levels of
    # g, of a and of b, so every level of a binomial too-few-rows fit holds one
    # 0/1 row. No other fit carries the note.
    fits = _fits()
    failures = []
    for c in fits:
        got = [n["detail"] for n in c.m.diagnostics["notes"] if n["kind"] == "one_trial_per_level"]
        want = []
        if c.state == "too_few_rows" and c.family == "binomial":
            want = {"A": [], "B": ["g"], "C": ["a", "b"], "D": ["g"]}[c.shape]
        if got != want:
            failures.append(f"{c.label}: {got} != {want}")
    assert len(fits) == 39
    assert not failures, "\n".join(failures)


def test_no_fit_is_at_the_gamma_dispersion_limit():
    # No row is Gamma, so the note on any of them is raised on the wrong family.
    fits = _fits()
    failures = [
        c.label
        for c in fits
        if any(n["kind"] == "dispersion_at_limit" for n in c.m.diagnostics["notes"])
    ]
    assert len(fits) == 39
    assert not failures, "\n".join(failures)


def test_summary_counts_the_observed_levels_of_an_unbalanced_nested_grouping():
    # A has 4 levels holding 2, 4, 1 and 3 levels of B: 10 A:B combinations,
    # which the kernel pads to 4 x 4 slots. lme4 prints 10.
    a, b = [], []
    for level, count in enumerate([2, 4, 1, 3]):
        for k in range(count):
            a += [f"a{level}"] * 3
            b += [f"b{k}"] * 3
    n = len(a)
    data = {
        "y": np.array([math.sin(1.7 * i) + 0.1 * i for i in range(n)]),
        "x": np.array([math.cos(2.3 * i) for i in range(n)]),
        "A": np.array(a),
        "B": np.array(b),
    }
    m = _quiet(glmm.fit, data, "y ~ x + (1 | A/B)")
    assert sorted(m.ranef_levels) == [4, 16]
    assert "A:B, 10" in m.summary_object().text()
