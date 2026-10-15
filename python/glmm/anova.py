"""`glmm.anova` and `glmm.compare`: the per-term tests of one fit and the
comparison of two models, with their tables.

The crate computes every statistic and every p-value; this module only reads
them and lays them out, with R's column names, so both ports give the same
tables. `glmm.anova(m)` reads the per-term table the fit stored when it was
made. `glmm.anova(m0, m1)` compares two stored fits as they are, with no
refit and no data. `glmm.compare(data, f0, f1)` fits both formulas on the same
rows and compares them.

The module's name is also the function's: `glmm.anova` is the function, and
this module stays reachable as `sys.modules["glmm.anova"]`.
"""

import math
from dataclasses import dataclass

import numpy as np

from glmm import (
    Fit,
    _check_call,
    _check_nagq,
    _fit_from_native,
    _native,
    _prepare_columns,
)
from glmm.summary import Rendered, fmt, fmt_df, fmt_p, stars

_LEGEND = "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
_RLRT = "exact RLRT null distribution (Crainiceanu & Ruppert 2004)"
# The per-term heading and note texts (_NOT_CONVERGED, _LRT_LINE,
# _AUTO_REASONS, _WALD_LINE, and `_term_table`'s heading) are copied in
# validation/tools/anova_ports/anova_ports.rs, which the anova port gate
# compares with both ports - change together.
_NOT_CONVERGED = (
    "The fit did not converge, so its estimates are not at a maximum and these tests "
    "are not reliable. Refit until the fit converges, then test again."
)
_LRT_LINE = "Likelihood-ratio chi-squared tests (each term's model refitted without it)"
_TESTS = ("auto", "Wald", "LRT")
# Why "auto" gave a chi-squared table its test, by the fit's `term_test_auto`
# reason. Mirrors the R port's .AUTO_REASONS and
# validation/tools/anova_ports/anova_ports.rs - change together.
_AUTO_REASONS = {
    "poisson_mixed": "Poisson model with random effects",
    "binomial_mixed": "binomial model with random effects",
    "negative_binomial_mixed": "negative binomial model with random effects",
    "other_mixed": "model with random effects, not binomial, Poisson or negative binomial",
    "no_random_effects": "model without random effects",
}
_WALD_LINE = "Wald chi-squared tests"


class _Default(str):
    """`glmm.anova`'s default `test`, "auto", told apart from a `test` the
    caller passed, which two fits refuse."""


_AUTO = _Default("auto")


class _Table(Rendered):
    """Rows named `row_names`, columns indexed by R's column names
    (`tab["Pr(>Chisq)"]` is a float array, one entry per row), a `heading`
    and the `notes` printed under the table."""

    _values: dict
    row_names: list
    heading: str
    notes: list

    @property
    def columns(self):
        """The column names, in order."""
        return list(self._values)

    def __getitem__(self, name):
        return self._values[name]

    def __contains__(self, name):
        return name in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self.row_names)

    def __repr__(self):
        return self.text()

    def to_pandas(self):
        """The table as a `pandas.DataFrame` with the same columns, indexed by
        `row_names`. pandas is imported here, and only here: everything else in
        glmm works without it."""
        import pandas as pd

        return pd.DataFrame(dict(self._values), index=list(self.row_names))

    def _grid(self, header, cells, marks):
        """`text()`'s table: the row names left-aligned, every column
        right-aligned to its widest cell, the significance stars last."""
        name_w = max((len(n) for n in self.row_names), default=0)
        widths = [max([len(h), *(len(r[i]) for r in cells)]) for i, h in enumerate(header)]
        lines = [" " * name_w + "".join(f" {h:>{w}}" for h, w in zip(header, widths))]
        for name, row, mark in zip(self.row_names, cells, marks):
            line = f"{name:<{name_w}}" + "".join(f" {c:>{w}}" for c, w in zip(row, widths))
            lines.append(f"{line} {mark}".rstrip())
        return lines

    def _notes(self):
        return []


@dataclass(repr=False, eq=False)
class TermTable(_Table):
    """The per-term tests of one fit, `glmm.anova(m)`: one row per fixed term,
    in formula order. F tables have lmerTest's columns (`Sum Sq`, `Mean Sq`,
    `NumDF`, `DenDF`, `F value`, `Pr(>F)`), Wald chi-squared tables car's
    (`Chisq`, `Df`, `Pr(>Chisq)`), and likelihood-ratio tables the same
    columns under a two-line `heading` that names the test. The html, latex
    and typst renders add a `Reference` column naming each row's
    distribution, `F(NumDF, DenDF)` or `χ²(Df)`, built from the row's df.
    Name is provisional until 1.0.0."""

    _values: dict
    row_names: list
    heading: str
    notes: list
    ss_type: str  # "III" or "II"

    def _cells(self):
        p_col = "Pr(>F)" if "Pr(>F)" in self._values else "Pr(>Chisq)"

        def cell(i, c):
            v = self._values[c][i]
            if c == p_col:
                return fmt_p(v)
            return fmt_df(v) if c in ("NumDF", "DenDF", "Df") else fmt(v)

        cells = [[cell(i, c) for c in self._values] for i in range(len(self))]
        return list(self._values), cells, [stars(p).strip() for p in self._values[p_col]]

    def text(self):
        header, cells, marks = self._cells()
        out = [self.heading, *self._grid(header, cells, marks), "---", _LEGEND]
        return "\n".join(out + self.notes)

    def _header_lines(self):
        return self.heading.split("\n")

    def _tables(self):
        header, cells, marks = self._cells()
        # Each row's reference, from its own df columns.
        if "DenDF" in self._values:
            refs = [_f_reference(a, b) for a, b in zip(self["NumDF"], self["DenDF"])]
        else:
            refs = [_chisq_reference(a) for a in self["Df"]]
        rows = [
            [name, *row, mark, ref]
            for name, row, mark, ref in zip(self.row_names, cells, marks, refs)
        ]
        return [("Terms", ["", *header, "", "Reference"], rows)]

    def _footer_lines(self):
        return [_LEGEND, *self.notes]


@dataclass(repr=False, eq=False)
class ComparisonTable(_Table):
    """Two models compared, `glmm.anova(m0, m1)` or `glmm.compare(...).table`:
    one row per model, named after the models as given (`m0` and `m1`, or `f0`
    and `f1`), the smaller model first. A likelihood-ratio table has lme4's
    columns (`npar`, `AIC`, `BIC`, `logLik`, `Chisq`, `Df`, `Pr(>Chisq)`; no
    `AIC` or `BIC` on REML fits), two OLS fits `anova.lm`'s (`Res.Df`, `RSS`,
    `Df`, `Sum of Sq`, `F`, `Pr(>F)`), and two Gamma or inverse Gaussian GLMs
    `Res.Df`, `Df`, `F`, `Pr(>F)`. The test sits on the larger model's row;
    the smaller model's test cells are NaN. `reference` names the distribution
    the p-value is read from; `nsim` and `seed` are the exact test's draws
    requested and seed. Name is provisional until 1.0.0."""

    _values: dict
    row_names: list
    heading: str
    notes: list
    formulas: list  # the two formulas in row order, as (label, formula)
    reference: str
    nsim: int
    seed: int
    _blank: set  # (row, column) cells printed empty rather than NA

    def _cells(self):
        p_col = "Pr(>F)" if "Pr(>F)" in self._values else "Pr(>Chisq)"
        five = {"AIC", "BIC", "logLik", "RSS"}

        def cell(i, c):
            if (i, c) in self._blank:
                return ""
            v = self._values[c][i]
            if c == p_col:
                return fmt_p(v)
            if c in ("npar", "Res.Df", "Df"):
                return fmt_df(v)
            return fmt(v, 5 if c in five else 4)

        cells = [[cell(i, c) for c in self._values] for i in range(len(self))]
        marks = [
            "" if (i, p_col) in self._blank else stars(p).strip()
            for i, p in enumerate(self._values[p_col])
        ]
        return list(self._values), cells, marks

    def text(self):
        header, cells, marks = self._cells()
        out = [*self._header_lines(), *self._grid(header, cells, marks)]
        out.append(f"  Reference: {self.reference}.")
        return "\n".join(out + [f"  {n}" for n in self.notes])

    def _header_lines(self):
        return [self.heading, *(f"{label}: {formula}" for label, formula in self.formulas)]

    def _tables(self):
        header, cells, marks = self._cells()
        rows = [
            [name, *row, mark, self.reference if i == 1 else ""]
            for i, (name, row, mark) in enumerate(zip(self.row_names, cells, marks))
        ]
        return [("Models", ["", *header, "", "Reference"], rows)]

    def _footer_lines(self):
        return list(self.notes)


@dataclass(repr=False, eq=False)
class Comparison:
    """What `glmm.compare` returns: the comparison `table` and the two fits it
    made, `fit0` of the first formula and `fit1` of the second, as given.
    Prints as its table. Name is provisional until 1.0.0."""

    table: ComparisonTable
    fit0: Fit
    fit1: Fit

    def __repr__(self):
        return self.table.text()


def anova(*fits, type=None, test=_AUTO, nsim=None, seed=None):
    """One fit: its per-term tests, a `TermTable`. Two fits: the two models
    compared as they are, with no refit, a `ComparisonTable`.

    One fit, `glmm.anova(m)`: one row per fixed term, read from the table the
    fit stored when it was made, so nothing is computed here. `type="III"`
    (the default when `type` is not given) builds each term's hypothesis the way lmerTest does, so it
    does not depend on the contrast coding; `type="II"` is the one
    alternative. OLS, a Gaussian mixed model and the Gamma and inverse
    Gaussian GLMs with phi estimated get F tests (the residual df, or
    Satterthwaite's on a mixed model); every other fit Wald chi-squared tests,
    but a Poisson model with random effects likelihood-ratio tests (`test`).
    There are no random-effect rows: random effects are tested by comparing
    two models. A formula whose interaction lacks one of its marginal terms
    raises `ValueError`. A fit that did not converge gets its table with a
    note saying so.

    `test` chooses the rows' test on one fit (None, as with `type`, is the
    default). "auto" (the default) gives the F
    rows on a fit that has them, the likelihood-ratio rows on a Poisson model
    with random effects whatever its size (in simulation the Wald test of a
    Poisson GLMM was never closer to the nominal level than the
    likelihood-ratio test, and rejected too often at few clusters), and the
    Wald chi-squared rows on every other fit. "Wald" gives the Wald chi-squared
    rows; "LRT" the likelihood-ratio rows, each term's hypothesis tested by
    refitting the model under it (Type II by the rule of car's
    `Anova(test.statistic = "LR")` on a GLM, applied to GLMMs too; Type III on
    the reduced design of the same hypothesis as the Wald row, as afex's
    `mixed(method = "LRT")` on a GLMM), in the Wald table's columns. The refits
    are made when the fit is: a fit made with `glmm.fit(..., anova_test="LRT")`
    has these rows, and so does a Poisson model with random effects made with
    the default `anova_test="auto"`; on a fit without them "LRT", and "auto" on
    a Poisson GLMM, raise `ValueError` naming the fit argument. A term whose
    column carries a random slope keeps its Wald row, with a note. "Wald" and
    "LRT" apply to every fit with chi-squared rows: any GLMM, and every GLM but
    a Gamma or inverse Gaussian one with phi estimated; on a fit with F rows
    they raise `ValueError` naming the F test. The heading names the test;
    under "auto" a chi-squared table's heading also names why "auto" chose it
    (the family, or no random effects).

    Two fits, `glmm.anova(m0, m1)`, in either order: the fits must be on the
    same rows (the same row count, response, weights and offset), with the
    same family, link and held dispersion, one formula nested in the other,
    and differ in their parameter count. Two OLS fits get `anova.lm`'s F test
    and two Gamma or inverse Gaussian GLMs the Wald F; every other pair a
    likelihood-ratio test, whose reference follows the random parts: chi-squared
    when they are the same, the 50:50 mixture when one random dimension is
    dropped (the exact restricted likelihood-ratio test when the larger model's
    only random term is one unweighted `(1 | g)`, with at least one
    fixed-effect column and a grouping with at most 100 levels with rows, and
    the smaller is an OLS fit made with `reml=True`), chi-squared ignoring the
    boundary otherwise. Two REML fits with different fixed parts, or one REML
    and one ML fit, are refused (lme4 refits such fits by ML; this does not):
    `glmm.compare(data, f0, f1)` fits both the right way. Every refusal raises
    `ValueError` naming the fix.

    nsim: draws of the exact test's simulated null, a whole number from 1 to
        1,000,000 (default 10,000).
    seed: the seed of those draws, a whole number from 0 to 2**64 - 1. The
        default is fixed, so the same call gives the same p-value on every run.

    Three or more fits raise `TypeError`: a comparison takes two models. So
    do the arguments of the other form: `nsim` or `seed` with one fit, `type`
    or `test` other than None with two. A fit made by a glmm version without these tests
    raises `ValueError` naming the fields it lacks: refit it. The name and
    the table layouts are provisional until 1.0.0."""
    if len(fits) not in (1, 2):
        raise TypeError(f"glmm.anova takes one fit or two, got {len(fits)}")
    for m in fits:
        if not isinstance(m, Fit):
            raise TypeError(f"glmm.anova takes glmm.Fit objects, got {_type_name(m)}")
    if test is None:
        test = _AUTO
    if len(fits) == 1:
        if nsim is not None or seed is not None:
            raise TypeError(
                "glmm.anova of one fit takes no nsim or seed: they are the exact test's "
                "draws, used only when comparing two fits, glmm.anova(m0, m1)"
            )
        if type is None:
            type = "III"
        if type not in ("III", "II"):
            raise ValueError(f"type must be 'III' or 'II', got {type!r}")
        if test not in _TESTS:
            raise ValueError(f"test must be 'auto', 'Wald' or 'LRT', got {test!r}")
        (m,) = fits
        field = "term_table_type3" if type == "III" else "term_table_type2"
        # Only "auto" reads the fit's auto choice: a fit stored before it
        # still takes an explicit test.
        _check_fields(m, [field, "term_test_auto"] if test == "auto" else [field])
        stored = getattr(m, field)
        if stored["error"] is not None:
            # The formula's refusal comes first, whatever `test` asks, as in
            # the R port.
            return _term_table(m, stored, test)
        if test != "auto" and stored.get("reason") == "f_test":
            den_df = "residual" if m.df_method == "residual" else "satterthwaite"
            raise ValueError(_native.f_table_refusal(test, den_df))
        if test == "LRT" or (test == "auto" and m.term_test_auto["test"] == "lrt"):
            stored = getattr(m, field + "_lrt", None)
            if stored is None and test == "LRT":
                raise ValueError(
                    "this fit stored no likelihood-ratio tables, which are built only when "
                    "the fit asks for them: refit with glmm.fit(..., anova_test='LRT')."
                )
            if stored is None:
                raise ValueError(
                    "test='auto' gives likelihood-ratio tests on a Poisson model with "
                    "random effects, and this fit stored none: it was made with "
                    "anova_test='Wald'. Use test='Wald', or refit with "
                    "glmm.fit(..., anova_test='auto')."
                )
        return _term_table(m, stored, test)
    if type is not None:
        raise TypeError(
            "glmm.anova of two fits takes no type: a comparison of two models has no "
            "sum-of-squares type; type applies to one fit, glmm.anova(m)"
        )
    if test is not _AUTO:
        raise TypeError(
            "glmm.anova of two fits takes no test: a comparison's test follows from "
            "the two models; test applies to one fit, glmm.anova(m)"
        )
    _check_draws(nsim, seed)
    r = _native.compare_fits(_side(fits[0]), _side(fits[1]), nsim, seed)
    return _comparison_table(r, ("m0", "m1"), None)


def compare(
    data,
    f0,
    f1=None,
    family="gaussian",
    *,
    link=None,
    weights=None,
    offset=None,
    nagq=1,
    nsim=None,
    seed=None,
):
    """Fit two nested formulas on the same rows of `data` and compare them; a
    `Comparison` with the `table` and the two fits (`fit0`, `fit1`).

    The estimator follows from what differs, so there is no `reml` argument:
    on a Gaussian model both fits are REML fits when only the random part
    differs (the model with no random effects fitted with `reml=True`) and ML
    fits when the fixed part differs, alone or with the random part. A GLM or
    GLMM is fitted by ML. `family`, `link`, `weights`, `offset` and `nagq` are
    given once and used for both fits, as `glmm.fit` takes them. A row with a
    missing value in any column either formula uses, in the weights or in the
    offset is dropped from both fits. The two formulas can be given in either order; the table lists the
    smaller model first. `nsim` and `seed` are as in `glmm.anova`. Both fits
    are made with `anova_test="Wald"`: they store no likelihood-ratio tables,
    so `glmm.anova` of a Poisson model with random effects among them needs
    `test="Wald"`. For the likelihood-ratio rows, fit the formula with
    `glmm.fit(..., anova_test="LRT")`.

    Formulas that are not nested, or that give the same model, raise
    `ValueError`. A third formula raises `TypeError`, and so do fits instead
    of formulas: two fits are compared with `glmm.anova(m0, m1)`. The name is
    provisional until 1.0.0."""
    if f1 is None or any(isinstance(a, Fit) for a in (data, f0, f1)):
        raise TypeError(
            "glmm.compare takes data and two formulas, glmm.compare(data, f0, f1); "
            "to compare two fits you already have, use glmm.anova(m0, m1)"
        )
    # A third positional argument is `family`, as in `glmm.fit`; a formula
    # there is a third formula, refused as the R port refuses one.
    if isinstance(family, str) and "~" in family:
        raise TypeError("glmm.compare takes two formulas, not three or more")
    for f in (f0, f1):
        if not isinstance(f, str):
            raise TypeError(f"glmm.compare takes two formula strings, got {_type_name(f)}")
    resolved = _check_call(f0, family, link)
    _check_call(f1, family, link)
    link = resolved
    _check_nagq(nagq)
    _check_draws(nsim, seed)
    store = []
    numeric_columns, factor_columns, weights, offset = _prepare_columns(
        data, [f0, f1], weights, offset, store, drop_missing_arguments=True
    )
    r = _native.compare(
        f0,
        f1,
        numeric_columns,
        factor_columns,
        family,
        link,
        nagq,
        None if weights is None else [float(w) for w in weights],
        None if offset is None else [float(v) for v in offset],
        nsim,
        seed,
    )
    fit0, fit1 = (
        _fit_from_native(r[key], f, family, link, nagq, None, list(store))
        for key, f in (("fit0", f0), ("fit1", f1))
    )
    return Comparison(_comparison_table(r, ("f0", "f1"), family), fit0, fit1)


def _type_name(x):
    return type(x).__name__


def _check_draws(nsim, seed):
    """`nsim` and `seed` as `glmm.anova` and `glmm.compare` take them. The
    crate bounds `nsim` above; a value below 1 cannot cross as an unsigned
    count, so it is refused here with the crate's message."""
    if nsim is not None:
        if isinstance(nsim, bool) or not isinstance(nsim, (int, np.integer)):
            raise TypeError(f"nsim must be a whole number, got {nsim!r}")
        if nsim < 1:
            raise ValueError(
                f"nsim must be a whole number from 1 to {_native.MAX_NSIM}, got {nsim}"
            )
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be a whole number, got {seed!r}")
        if not 0 <= seed < 2**64:
            raise ValueError(f"seed must be a whole number from 0 to 2**64 - 1, got {seed}")


def _check_fields(m, names):
    """Refuse a fit that lacks fields `glmm.anova` reads: one made by a glmm
    version before them (an old pickle), which has them missing or None."""
    missing = [n for n in names if getattr(m, n, None) is None]
    if missing:
        raise ValueError(
            f"this fit has no {', '.join(missing)}: it was made by a glmm version "
            "without these tests. Refit the model with this version."
        )


def _side(m):
    """What `glmm.anova(m0, m1)` reads of one stored fit, as plain values."""
    _check_fields(m, ["rss", "objective"])

    def rows(a):
        return None if a is None else np.asarray(a, dtype=float).tolist()

    ds = m.design_summary
    return {
        "formula": m.formula,
        "family": m.family,
        "link": m.link,
        "loglik": float(m.loglik),
        "df": int(m.df),
        "nobs": int(m.nobs),
        "reml": bool(m.reml),
        "objective": m.objective,
        "objective_nodes": int(m.objective_nodes),
        "converged": bool(m.converged),
        "varcorr": [rows(b) for b in m.varcorr],
        "re_groups": [(name, list(terms)) for name, terms in m.re_groups],
        "y": rows(m.y),
        "weights": rows(m.weights),
        "offset": rows(m.offset),
        "dispersion_held": None if m.dispersion_held is None else float(m.dispersion_held),
        "design_summary": None
        if ds is None
        else (rows(ds["counts"]), rows(ds["s"]), rows(ds["c"])),
        "rss": float(m.rss),
        "coef_names": list(m.names),
        "coef": rows(m.beta),
        "vcov": np.asarray(m.vcov, dtype=float).reshape(len(m.beta), len(m.beta)).tolist(),
        "aliased": [bool(a) for a in m.aliased],
    }


def _term_table(m, stored, test):
    """The `TermTable` of a stored per-term result (`Fit.term_table_type3`),
    read for `glmm.anova(m, test=test)`."""
    error = stored["error"]
    if error is not None:
        # A panic caught while the table was built is the crate's fault, not
        # the model's: an error, not a refusal.
        raise (RuntimeError if error.startswith("internal error") else ValueError)(error)
    ss = stored["ss_type"]
    labels = list(stored["label"])
    if labels:
        f_rows = stored["kind"][0] == "F"
        # The heading names Satterthwaite's method only when some row has its
        # df: on a fit with the pass switched off or failed, no row does.
        satterthwaite = any(
            kind == "satterthwaite" and math.isfinite(d)
            for kind, d in zip(stored["den_df_kind"], stored["den_df"])
        )
    else:
        f_rows = m.df_method != "none"
        satterthwaite = m.df_method == "satterthwaite"
    if f_rows:
        heading = f"Type {ss} Analysis of Variance Table" + (
            " with Satterthwaite's method" if satterthwaite else ""
        )
        values = {
            "Sum Sq": stored["sum_sq"],
            "Mean Sq": stored["mean_sq"],
            "NumDF": stored["num_df"],
            "DenDF": stored["den_df"],
            "F value": stored["statistic"],
            "Pr(>F)": stored["p_value"],
        }
    else:
        if stored.get("test") == "lrt" or test == "auto":
            # The second line names the test and why it ran, as the R port's;
            # an explicit "Wald" keeps the one-line heading.
            if stored.get("test") == "lrt":
                line = _LRT_LINE + (" on the reduced design" if ss == "III" else "")
            else:
                line = _WALD_LINE
            why = (
                'test = "LRT"'
                if test == "LRT"
                else f'test = "auto": {_AUTO_REASONS[m.term_test_auto["reason"]]}'
            )
            heading = f"Type {ss} Analysis of Deviance Table\n{line}; {why}"
        else:
            heading = f"Type {ss} Analysis of Deviance Table (Wald χ² tests)"
        values = {
            "Chisq": stored["statistic"],
            "Df": stored["num_df"],
            "Pr(>Chisq)": stored["p_value"],
        }
    row_notes = [
        f"{label}: {note}" for label, notes in zip(labels, stored["row_notes"]) for note in notes
    ]
    notes = ([] if m.converged else [_NOT_CONVERGED]) + list(stored["notes"]) + row_notes
    return TermTable(values, labels, heading, notes, ss)


# One notation for a reference distribution in both tables. A row with no df
# (an aliased term, a row with no denominator df) names none: an empty cell.
def _f_reference(num_df, den_df):
    if math.isnan(num_df) or math.isnan(den_df):
        return ""
    return f"F({fmt_df(num_df)}, {fmt_df(den_df)})"


def _chisq_reference(df):
    return "" if math.isnan(df) else f"χ²({fmt_df(df)})"


def _reference(ref):
    """The distribution a comparison's p-value is read from, in words."""
    kind = ref["kind"]
    if kind == "chisq":
        return _chisq_reference(ref["df"])
    if kind == "chibar":
        k = ref["df"]
        return f"50:50 mixture of {_chisq_reference(k)} and {_chisq_reference(k + 1)}"
    if kind == "exact":
        if ref["nsim"] == 0:
            return f"{_RLRT}; no draws needed: the statistic is 0, so p = 1"
        return f"{_RLRT}, {ref['nsim']} draws"
    return _f_reference(ref["df"], ref["den_df"])


def _heading(r):
    if r["test"] == "f":
        return "Model comparison: F test"
    objective = r["objective"]
    if objective == "reml":
        return "Model comparison: likelihood ratio test on the REML criterion"
    label = {"laplace": "Laplace", "agq": f"AGQ {r['objective_nodes']}"}.get(objective, "ML")
    return f"Model comparison: likelihood ratio test ({label})"


def _comparison_table(r, labels, family):
    """The `ComparisonTable` of a native comparison dict; `labels` names the two
    sides as given, `family` is `compare`'s (None from `anova`)."""
    nan = math.nan
    large = r["larger"]
    order = [1 - large, large]
    sides = [r["sides"][i] for i in order]
    test = [nan, r["statistic"]]
    p = [nan, r["p_value"]]
    df = [nan, r["df"]]
    if r["test"] == "lr":
        values = {"npar": [s["npar"] for s in sides]}
        if all(s["aic"] is not None for s in sides):
            values["AIC"] = [s["aic"] for s in sides]
            values["BIC"] = [s["bic"] for s in sides]
        values["logLik"] = [s["loglik"] for s in sides]
        values.update({"Chisq": test, "Df": df, "Pr(>Chisq)": p})
        blank = {(0, "Chisq"), (0, "Df"), (0, "Pr(>Chisq)")}
        if r["reference"]["kind"] == "exact":
            # The exact reference is not a chi-squared: no Df.
            values["Df"] = [nan, nan]
            blank.add((1, "Df"))
    else:
        # The residual df n - p, p the retained columns (npar counts sigma or phi too).
        values = {"Res.Df": [s["nobs"] - (s["npar"] - 1) for s in sides]}
        ols = all(math.isfinite(s["rss"]) for s in sides)
        if ols:
            values["RSS"] = [s["rss"] for s in sides]
        values["Df"] = df
        if ols:
            values["Sum of Sq"] = [nan, sides[0]["rss"] - sides[1]["rss"]]
        values.update({"F": test, "Pr(>F)": p})
        blank = {(0, c) for c in ("Df", "Sum of Sq", "F", "Pr(>F)")}
    notes = list(r["notes"])
    if family == "gaussian" and r["test"] == "lr" and r["differs"] != "random":
        notes.append("Both models were fitted by ML because their fixed parts differ.")
    return ComparisonTable(
        {k: np.asarray(v, dtype=float) for k, v in values.items()},
        [labels[i] for i in order],
        _heading(r),
        notes,
        [(labels[i], r["sides"][i]["formula"]) for i in order],
        _reference(r["reference"]),
        r["nsim"],
        r["seed"],
        blank,
    )
