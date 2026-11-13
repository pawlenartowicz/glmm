"""lme4-shaped summary of a `Fit`, as plain data plus renderers.

One structure, four renders (`text`, `html`, `latex`, `typst`). Every renderer
formats numbers through `fmt` — the sharing is load-bearing: if the LaTeX and
text paths rounded independently, one fit would report two different numbers
depending on where it was pasted.

Block order follows lme4's `print.summary.merMod`; the R port's
`print.summary.fastglmm` prints the same blocks in the same order — change
together.

The `html`, `latex` and `typst` renderers live on `Rendered`, a base that
reads four hooks: `_header_lines`, `_notes`, `_tables` and `_footer_lines`.
`_notes` holds the printed notes each render puts directly under the title;
`_LEAD_TABLES` names the tables the LaTeX and Typst renders put first, the
others following in `_tables` order. `Summary` and the two tables of
`glmm.anova` (`glmm/anova.py`) share it; each keeps its own `text`.
"""

import html as _html
import math
from dataclasses import dataclass, field

import numpy as np

from glmm import _native

# The footer of a Gaussian mixed model's t tests, as the native summary record
# carries it.
FOOTNOTE = "t tests use Satterthwaite degrees of freedom (finite-difference θ-Hessian)."
# The footer of a Gaussian mixed model fitted with satterthwaite=False; the R port
# prints "df not computed (satterthwaite = FALSE)", its own argument spelling.
DF_NOT_COMPUTED = "df not computed (satterthwaite=False)"


def fmt(x, digits=4):
    """The one number formatter. NaN prints as lme4's NA."""
    x = float(x)
    if math.isnan(x):
        return "NA"
    return f"{x:.{digits}g}"


def fmt_df(x):
    """The one degrees-of-freedom formatter, for the df cells and every
    reference string: a whole df (a residual df, a numerator df, a chi-squared
    df) prints as an integer at any size, a Satterthwaite df with four
    significant digits, or as the nearest whole number from 10000 up, always
    in fixed notation. NaN prints as NA and an infinite df as Inf or -Inf,
    as R prints them."""
    x = float(x)
    if math.isnan(x):
        return "NA"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    if x == round(x):
        return str(int(x))
    if abs(x) >= 1e4:
        return f"{x:.0f}"
    return np.format_float_positional(x, precision=4, unique=False, fractional=False, trim="-")


def fmt_p(p):
    p = float(p)
    if math.isnan(p):
        return "NA"
    if p < 2e-16:
        return "<2e-16"
    return f"{p:.3g}"


def stars(p):
    """R's `Signif. codes` thresholds."""
    p = float(p)
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return " "


def _f64_bytes(a):
    return None if a is None else np.asarray(a, dtype=np.float64).tobytes()


def pearson_residuals(fit):
    """`(y - fitted) / sqrt(phi * V(fitted) / w^weight_power)` per row, from the native
    code, which takes sqrt(V) without forming V so it stays finite at
    extreme means."""
    r = _native.call(
        "pearson_residuals",
        {
            "family": fit.family,
            "y": _f64_bytes(fit.y),
            "fitted": _f64_bytes(fit.fitted),
            "weights": _f64_bytes(fit.weights),
            "dispersion": float(fit.dispersion),
            "weight_power": float(fit.weight_power),
        },
    )
    return np.asarray(r, dtype=float)


def _tex_escape(s):
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("|", "\\textbar{}")
        .replace(">", "\\textgreater{}")
        .replace("<", "\\textless{}")
    )


def _typst_str(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


class Rendered:
    """The `html`, `latex` and `typst` renders of a report, read from four
    hooks a subclass defines: `_header_lines()` (the title, then the lines
    under it), `_notes()` (the boxed notes under the title), `_tables()` (the
    blocks as `(title, header, rows)`, every cell already a string) and
    `_footer_lines()`. `_LEAD_TABLES` names the blocks the LaTeX and Typst
    renders put first."""

    _LEAD_TABLES = ()

    # -- html --------------------------------------------------------------

    def html(self):
        e = _html.escape
        out = [
            (
                "<style>"
                ".glmm-summary table{border-collapse:collapse;margin:0.5em 0;font-family:monospace}"
                ".glmm-summary th,.glmm-summary td{padding:0.1em 0.6em;text-align:right}"
                ".glmm-summary td:first-child,.glmm-summary th:first-child{text-align:left}"
                ".glmm-summary caption{text-align:left;font-weight:bold}"
                ".glmm-summary p{margin:0.2em 0}"
                ".glmm-summary .glmm-note{margin:0.4em 0;padding:0.3em 0.6em;"
                "border-left:4px solid #c88a00;background:rgba(200,138,0,0.12)}"
                "</style>"
            ),
            '<div class="glmm-summary">',
        ]
        title, *rest = self._header_lines()
        out.append(f"<p>{e(title)}</p>")
        out += [f'<div class="glmm-note" role="note">{e(n)}</div>' for n in self._notes()]
        out += [f"<p>{e(l)}</p>" for l in rest]
        for title, header, rows in self._tables():
            out.append(f"<table><caption>{e(title)}</caption>")
            out.append("<tr>" + "".join(f"<th>{e(h)}</th>" for h in header) + "</tr>")
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>")
            out.append("</table>")
        out += [f"<p>{e(l)}</p>" for l in self._footer_lines()]
        out.append("</div>")
        return "\n".join(out)

    # -- latex ---------------------------------------------------------------

    def latex(self):
        """booktabs fragments; the caller's preamble must load `booktabs`.
        The `_LEAD_TABLES` first (on a summary: fixed effects, then random
        effects), then the rest, header and
        footer as `%` comments: a table goes into a document, a header does not.
        A note is a boxed paragraph under the title, in the document, ended
        with `\\par\\medskip` so the first table starts below it with a gap."""
        title, *rest = self._header_lines()
        out = [f"% {title}"]
        out += [
            "\\noindent\\fbox{\\parbox{\\dimexpr\\linewidth-2\\fboxsep-2\\fboxrule\\relax}"
            f"{{{_tex_escape(n)}}}}}\\par\\medskip"
            for n in self._notes()
        ]
        out += [f"% {l}" for l in rest]
        tables = self._tables()
        order = list(self._LEAD_TABLES) + [t for t, _, _ in tables if t not in self._LEAD_TABLES]
        by_title = {t: (h, r) for t, h, r in tables}
        for title in order:
            if title not in by_title:
                continue
            header, rows = by_title[title]
            spec = "l" + "r" * (len(header) - 1)
            out.append(f"% {title}")
            out.append(f"\\begin{{tabular}}{{{spec}}}")
            out.append("\\toprule")
            out.append(" & ".join(_tex_escape(h) for h in header) + " \\\\")
            out.append("\\midrule")
            for r in rows:
                out.append(" & ".join(_tex_escape(c) for c in r) + " \\\\")
            out.append("\\bottomrule")
            out.append("\\end{tabular}")
            out.append("")
        out += [f"% {l}" for l in self._footer_lines()]
        return "\n".join(out)

    # -- typst -----------------------------------------------------------

    def typst(self):
        """`#table(...)` fragments. Written and tested against Typst 0.14.2 —
        table syntax has moved between Typst versions; re-check on bump."""
        title, *rest = self._header_lines()
        out = [f"// {title}"]
        out += [f"#block(stroke: 0.5pt, inset: 6pt, {_typst_str(n)})" for n in self._notes()]
        out += [f"// {l}" for l in rest]
        tables = self._tables()
        order = list(self._LEAD_TABLES) + [t for t, _, _ in tables if t not in self._LEAD_TABLES]
        by_title = {t: (h, r) for t, h, r in tables}
        for title in order:
            if title not in by_title:
                continue
            header, rows = by_title[title]
            n = len(header)
            align = "(left," + ",".join(["right"] * (n - 1)) + ")"
            out.append(f"// {title}")
            out.append(f"#table(columns: {n}, align: {align},")
            # Cells are Typst strings, not `[...]` content blocks: a name like
            # `x_1` or `a*b` is markup inside a content block and plain text
            # inside a string.
            out.append(
                "  " + ", ".join(f"strong({_typst_str(h)})" if h else '""' for h in header) + ","
            )
            for r in rows:
                out.append("  " + ", ".join(_typst_str(c) for c in r) + ",")
            out.append(")")
            out.append("")
        out += [f"// {l}" for l in self._footer_lines()]
        return "\n".join(out)


def build_summary(fit, effect_sizes=None, need_refit=False):
    """The `Summary` of `fit`, from the native summary record. `effect_sizes`
    is the dict of `Fit._effect_sizes(null_refit=False)` (None: no R-squared
    line), and `need_refit` is True where that call gave None, so the summary
    points to `r2()` and `icc()` instead."""
    p = len(fit.names)
    # getattr, not attribute access: a Fit pickled before these attributes
    # existed still summarizes, as the z table its version printed.
    df_method = getattr(fit, "df_method", "none")
    r = _native.call(
        "summary",
        {
            "fit": {
                "formula": fit.formula,
                "family": fit.family,
                "link": fit.link,
                "reml": bool(fit.reml),
                "nagq": int(fit.nagq),
                "nobs": int(fit.nobs),
                "df": int(fit.df),
                "loglik": float(fit.loglik),
                "names": list(fit.names),
                "beta": np.asarray(fit.beta, dtype=float),
                "se": np.asarray(fit.se, dtype=float),
                "vcov": np.asarray(fit.vcov, dtype=float).reshape(p, p),
                "aliased": np.asarray(fit.aliased, dtype=bool),
                "df_method": df_method,
                "coef_df": getattr(fit, "coef_df", None),
                "coef_p": getattr(fit, "coef_p", None),
                "dispersion": float(fit.dispersion),
                "dispersion_held": getattr(fit, "dispersion_held", None),
                "varcorr": [np.asarray(b, dtype=float) for b in fit.varcorr],
                "re_groups": [(name, list(terms)) for name, terms in fit.re_groups],
                "re_term_index": [list(t) for t in fit.re_term_index],
                "ranef_levels": fit.ranef_levels,
                "re_group_levels": (fit._kernel_parts or {}).get("re_group_levels"),
                "y": _f64_bytes(fit.y),
                "fitted": _f64_bytes(fit.fitted),
                "weights": _f64_bytes(fit.weights),
                "weight_power": float(fit.weight_power),
                "weight_power_estimated": bool(fit.estimated_weight_power),
                "n_eval": int(fit.n_eval),
                "converged": bool(fit.converged),
                "singular": bool(fit.singular),
                "rss": getattr(fit, "rss", math.nan),
                "r_squared": getattr(fit, "r_squared", math.nan),
                "adj_r_squared": getattr(fit, "adj_r_squared", math.nan),
                "warnings": list(getattr(fit, "warnings", [])),
            },
            "effect_sizes": effect_sizes,
            "need_refit": need_refit,
        },
    )
    for k in (
        "estimate",
        "std_error",
        "coef_df",
        "statistic",
        "p_value",
        "scaled_residuals",
        "corr_fixed",
    ):
        if r[k] is not None:
            r[k] = np.asarray(r[k], dtype=float)
    for re in r["random_effects"]:
        for k in ("variance", "stddev", "corr"):
            re[k] = np.asarray(re[k], dtype=float)
    if r["test"] == "z":
        # This port prints NA for an infinite z, where the record (as R's
        # 2 * pnorm(-Inf)) has 0.
        r["p_value"][np.isinf(r["statistic"])] = math.nan
    r["groups"] = list(zip(r["groups"]["names"], r["groups"]["levels"]))
    r["aliased"] = np.asarray(r["aliased"], dtype=bool)
    if df_method == "disabled":
        r["footnote"] = DF_NOT_COMPUTED
    return Summary(**r)


@dataclass
class Summary(Rendered):
    _LEAD_TABLES = ("Fixed effects", "Random effects")

    method: str
    formula: str
    family: str
    link: str
    reml: bool
    criterion: dict
    scaled_residuals: np.ndarray | None
    random_effects: list
    residual_variance: float | None
    nobs: int
    groups: list
    coef_names: list
    estimate: np.ndarray
    std_error: np.ndarray
    coef_df: np.ndarray | None  # the df column, printed on Satterthwaite t tests only
    statistic: np.ndarray
    test: str  # "t" or "z": the statistic's column names
    p_value: np.ndarray
    residual_df: (
        int | float | None
    )  # n - p on OLS and the dispersion GLMs (NaN on a fit with no estimates), else None
    rss: float
    r_squared: float
    adj_r_squared: float
    # A mixed model's R-squared pair and adjusted ICC; NaN or None where
    # there are none.
    r2_marginal: float | None
    r2_conditional: float | None
    icc_adjusted: float | None
    # True on a Poisson or negative binomial GLMM, whose values need a refit
    # of the null model: the line then points to r2() and icc().
    r2_need_refit: bool
    aliased: np.ndarray
    corr_fixed: np.ndarray | None
    dispersion_label: str | None
    dispersion: float
    n_eval: int
    converged: bool
    singular: bool
    n_aliased: int
    footnote: str | None
    warnings: list = field(default_factory=list)
    # The weight power when estimated or other than 1, then (when estimated)
    # the line that it left the simulated range and the model SE's footnote.
    weight_power_lines: list = field(default_factory=list)
    # The note under the title of a Gaussian mixed model fitted by ML, else
    # None.
    ml_note: str | None = None

    def __repr__(self):
        return self.text()

    # -- text ------------------------------------------------------------

    def text(self):
        by_title = {t: (h, r) for t, h, r in self._tables()}
        out = [
            self.method,
            *self._notes(),
            f"Formula: {self.formula}",
            f" Family: {self.family} ({self.link})",
            "",
        ]

        def stacked(header, row):
            # One header line over one value line, each column right-aligned.
            w = [max(len(h), len(v)) + 1 for h, v in zip(header, row)]
            return [
                "".join(f"{h:>{wi}}" for h, wi in zip(header, w)),
                "".join(f"{v:>{wi}}" for v, wi in zip(row, w)),
            ]

        header, (row,) = by_title["Criterion"]
        if self.reml:
            out += [f"{header[0]}: {row[0]}", ""]
        else:
            out += stacked(header, row) + [""]

        if "Scaled residuals" in by_title:
            header, (row,) = by_title["Scaled residuals"]
            out += ["Scaled residuals:", *stacked(header, row), ""]

        if "Random effects" in by_title:
            header, rows = by_title["Random effects"]
            w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
            out.append("Random effects:")
            for r in (header, *rows):
                out.append(" " + " ".join(f"{c:<{wi}}" for c, wi in zip(r, w)))
        if self.groups:
            out.append(
                f"Number of obs: {self.nobs}, groups:  "
                + "; ".join(f"{name}, {n}" for name, n in self.groups)
            )
        else:
            out.append(f"Number of obs: {self.nobs}")
        out += self._effect_size_lines()
        out.append("")

        out.append("Fixed effects:")
        header, rows = by_title["Fixed effects"]
        name_w = max(12, max((len(n) for n in self.coef_names), default=4))
        widths = [12, 12] + ([10] if self.coef_df is not None else []) + [10, 10]
        k = len(widths)
        out.append(
            f"{'':<{name_w}} " + " ".join(f"{h:>{wi}}" for h, wi in zip(header[1 : k + 1], widths))
        )
        for r in rows:
            out.append(
                f"{r[0]:<{name_w}} "
                + " ".join(f"{c:>{wi}}" for c, wi in zip(r[1 : k + 1], widths))
                + f" {r[k + 1]}"
            )
        out.append("---")
        out.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        if self.n_aliased:
            out.append(f"({self.n_aliased} coefficient(s) not defined because of singularities)")
        if self._lm_lines():
            out += ["", *self._lm_lines()]
        out.append("")

        if "Correlation of Fixed Effects" in by_title:
            # Full names as row and column labels on both ports; lme4
            # abbreviates the columns (`abbreviate(., 6)`), which is the one
            # deliberate difference in this block.
            header, rows = by_title["Correlation of Fixed Effects"]
            cw = max(6, max(len(n) for n in header[1:]))
            out.append("Correlation of Fixed Effects:")
            out.append(f"{'':<{name_w}} " + " ".join(f"{n:>{cw}}" for n in header[1:]))
            for r in rows:
                cells = " ".join(f"{c:>{cw}}" for c in r[1:] if c)
                out.append(f"{r[0]:<{name_w}} {cells}")
            out.append("")

        if self.dispersion_label is not None:
            out.append(self._dispersion_line())
        out += self.weight_power_lines
        out.append(f"Optimizer evaluations: {self.n_eval}, converged: {self.converged}")
        if self.singular:
            out.append("boundary (singular) fit: see help('isSingular')")
        if self.footnote is not None:
            out.append(self.footnote)
        if self.warnings:
            # Mirrors print.summary.fastglmm (glmm-r/R/fastglmm-methods.R) - change together.
            out += ["", "Warnings:"]
            out += [
                f"{w['tier'].capitalize()}: {w['title']}. {w['message']}" for w in self.warnings
            ]
        return "\n".join(out)

    # -- shared block decomposition ---------------------------------------

    def _effect_size_lines(self):
        """The line of the R-squared pair and the adjusted ICC, under the
        number of observations; the pointer to r2() and icc() where they need
        a refit of the null model; nothing when a value is NaN or absent (a
        fit without random effects, a family they do not cover, a fit pickled
        before its training data was kept). Same lines as the R port's
        print.summary.fastglmm - change together."""
        if self.r2_need_refit:
            return ["R2 and ICC need a refit of the null model: call fit.r2() and fit.icc()"]
        values = (self.r2_marginal, self.r2_conditional, self.icc_adjusted)
        if any(v is None or not math.isfinite(v) for v in values):
            return []
        line = (
            f"Marginal R2: {values[0]:.3f}, conditional R2: {values[1]:.3f}, "
            f"adjusted ICC: {values[2]:.3f}"
        )
        return [line]

    def _lm_lines(self):
        """`summary.lm`'s two footer lines on an OLS fit (it has an R-squared),
        nothing otherwise. Same lines as the R port's print.summary.fastglmm -
        change together - except that R prints a residual standard error of 1e4
        or more in fixed notation (`format(signif(x, 4))`, "12350") where `fmt`
        switches to e-notation ("1.235e+04")."""
        if self.residual_df is None or self.family != "gaussian":
            return []
        sigma = math.sqrt(self.rss / self.residual_df) if self.residual_df > 0 else math.nan
        df = "NA" if math.isnan(self.residual_df) else self.residual_df
        return [
            f"Residual standard error: {fmt(sigma)} on {df} degrees of freedom",
            (
                f"Multiple R-squared:  {fmt(self.r_squared)},\t"
                f"Adjusted R-squared:  {fmt(self.adj_r_squared)}"
            ),
        ]

    def _dispersion_line(self):
        """The dispersion row, with the residual df its t tests use on a
        dispersion GLM with phi estimated."""
        line = f"{self.dispersion_label}: {fmt(self.dispersion, 6)}"
        if self.residual_df is not None:
            df = "NA" if math.isnan(self.residual_df) else self.residual_df
            line += f" on {df} degrees of freedom"
        return line

    def _tables(self):
        """Blocks as (title, header, rows) with every cell already a string via
        `fmt` — the single place the non-text renderers take their numbers from."""
        tables = []
        if self.reml:
            ((label, value),) = self.criterion.items()
            tables.append(("Criterion", [label], [[fmt(value, 5)]]))
        else:
            keys = list(self.criterion)
            tables.append(
                (
                    "Criterion",
                    keys,
                    [
                        [
                            fmt(self.criterion[k], 5)
                            if k != "df.resid"
                            else fmt_df(self.criterion[k])
                            for k in keys
                        ]
                    ],
                )
            )
        if self.scaled_residuals is not None:
            tables.append(
                (
                    "Scaled residuals",
                    ["Min", "1Q", "Median", "3Q", "Max"],
                    [[fmt(v) for v in self.scaled_residuals]],
                )
            )
        if self.random_effects:
            rows = []
            for re in self.random_effects:
                term = re["term_index"]
                for i in range(len(re["stddev"])):
                    # As lme4: a correlation only between two columns of one
                    # term, whose columns are contiguous; two terms are
                    # uncorrelated by the model.
                    corr = " ".join(
                        f"{re['corr'][i, j]:.2f}"
                        for j in range(i)
                        if i >= len(term) or term[j] == term[i]
                    )
                    rows.append(
                        [
                            re["group"] if i == 0 else "",
                            re["terms"][i] if i < len(re["terms"]) else "",
                            fmt(re["variance"][i]),
                            fmt(re["stddev"][i]),
                            corr,
                        ]
                    )
            if self.residual_variance is not None:
                rows.append(
                    [
                        "Residual",
                        "",
                        fmt(self.residual_variance),
                        fmt(math.sqrt(self.residual_variance)),
                        "",
                    ]
                )
            tables.append(
                ("Random effects", ["Groups", "Name", "Variance", "Std.Dev.", "Corr"], rows)
            )
        df_cells = (
            (lambda i: [fmt_df(self.coef_df[i])]) if self.coef_df is not None else (lambda i: [])
        )
        rows = [
            [
                n,
                fmt(self.estimate[i]),
                fmt(self.std_error[i]),
                *df_cells(i),
                fmt(self.statistic[i]),
                fmt_p(self.p_value[i]),
                stars(self.p_value[i]).strip(),
            ]
            for i, n in enumerate(self.coef_names)
        ]
        header = [
            "",
            "Estimate",
            "Std. Error",
            *(["df"] if self.coef_df is not None else []),
            f"{self.test} value",
            f"Pr(>|{self.test}|)",
            "",
        ]
        tables.append(("Fixed effects", header, rows))
        if self.corr_fixed is not None:
            names = self.coef_names
            rows = [
                [names[i]]
                + [f"{self.corr_fixed[i, j]:.3f}" for j in range(i)]
                + [""] * (len(names) - 1 - i)
                for i in range(1, len(names))
            ]
            tables.append(("Correlation of Fixed Effects", [""] + names[:-1], rows))
        return tables

    def _notes(self):
        """The notes printed directly under the title: the ML note on an ML fit
        of a Gaussian mixed model, else none. It cannot be switched off."""
        return [] if self.ml_note is None else [self.ml_note]

    def _header_lines(self):
        lines = [self.method, f"Formula: {self.formula}", f"Family: {self.family} ({self.link})"]
        if self.groups:
            lines.append(
                f"Number of obs: {self.nobs}, groups: "
                + "; ".join(f"{n}, {k}" for n, k in self.groups)
            )
        else:
            lines.append(f"Number of obs: {self.nobs}")
        return lines + self._effect_size_lines()

    def _footer_lines(self):
        # "Signif. codes" rides here (not in `_tables`, it is a legend, not a
        # data row) so its thresholds still reach the non-text renderers: the
        # cross-renderer numeric-parity test checks every number in `text()`
        # shows up somewhere in each render.
        lines = ["Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"]
        lines += self._lm_lines()
        if self.dispersion_label is not None:
            lines.append(self._dispersion_line())
        lines += self.weight_power_lines
        lines.append(f"Optimizer evaluations: {self.n_eval}, converged: {self.converged}")
        if self.singular:
            lines.append("boundary (singular) fit: see help('isSingular')")
        if self.n_aliased:
            lines.append(f"{self.n_aliased} coefficient(s) not defined because of singularities")
        if self.footnote is not None:
            lines.append(self.footnote)
        return lines
