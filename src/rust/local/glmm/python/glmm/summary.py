"""lme4-shaped summary of a `Fit`, as plain data plus renderers.

One structure, four renders (`text`, `html`, `latex`, `typst`). Every renderer
formats numbers through `fmt` — the sharing is load-bearing: if the LaTeX and
text paths rounded independently, one fit would report two different numbers
depending on where it was pasted.

Block order follows lme4's `print.summary.merMod`; the R port's
`print.summary.fastglmm` prints the same blocks in the same order — change
together.
"""

import html as _html
import math
from dataclasses import dataclass

import numpy as np

# Variance functions V(mu) per family, for Pearson residuals. Mirrors the
# kernel's family table (src/family.rs); theta is the negative-binomial shape
# the kernel reports as `dispersion`.
_VARIANCE = {
    "gaussian": lambda mu, theta: np.ones_like(mu),
    "binomial": lambda mu, theta: mu * (1.0 - mu),
    "poisson": lambda mu, theta: mu,
    "gamma": lambda mu, theta: mu * mu,
    "negativebinomial": lambda mu, theta: mu + mu * mu / theta,
    "inversegaussian": lambda mu, theta: mu**3,
}
# Families whose `dispersion` is a free scale phi multiplying V(mu). For the
# others phi == 1 (negbin's `dispersion` is theta, already inside V).
_PHI_FAMILIES = {"gaussian", "gamma", "inversegaussian"}

_DISPERSION_LABEL = {
    "gamma": "Dispersion (phi, Pearson)",
    "inversegaussian": "Dispersion (phi, Pearson)",
    "negativebinomial": "Shape (theta)",
}

# Same sentence the R port prints - change together.
FOOTNOTE = (
    "z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual df is reported."
)


def fmt(x, digits=4):
    """The one number formatter. NaN prints as lme4's NA."""
    x = float(x)
    if math.isnan(x):
        return "NA"
    return f"{x:.{digits}g}"


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


def method_line(fit):
    # Same strings as R's `.method_line()` (r/R/fastglmm-methods.R), with the
    # port name swapped — change together.
    mixed = len(fit.varcorr) > 0
    if not mixed:
        if fit.family == "gaussian":
            return "Linear model fit by least squares [glmm]"
        return "Generalized linear model fit by IRLS [glmm]"
    if fit.family == "gaussian":
        return "Linear mixed model fit by REML [glmm]"
    if fit.nagq > 1:
        return (
            "Generalized linear mixed model fit by maximum likelihood "
            f"(Adaptive Gauss-Hermite Quadrature, nAGQ = {fit.nagq}) [glmm]"
        )
    return "Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) [glmm]"


def pearson_residuals(fit):
    mu = np.asarray(fit.fitted, dtype=float)
    y = np.asarray(fit.y, dtype=float)
    w = np.ones_like(mu) if fit.weights is None else np.asarray(fit.weights, dtype=float)
    phi = fit.dispersion if fit.family in _PHI_FAMILIES else 1.0
    v = _VARIANCE[fit.family](mu, fit.dispersion)
    return (y - mu) * np.sqrt(w) / np.sqrt(phi * v)


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


def build_summary(fit):
    beta = np.asarray(fit.beta, dtype=float)
    se = np.asarray(fit.se, dtype=float)
    aliased = np.asarray(fit.aliased, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = beta / se
    p = np.array([math.erfc(abs(v) / math.sqrt(2.0)) if math.isfinite(v) else math.nan for v in z])
    beta = np.where(aliased, math.nan, beta)
    se = np.where(aliased, math.nan, se)
    z = np.where(aliased, math.nan, z)
    p = np.where(aliased, math.nan, p)

    if fit.reml:
        criterion = {"REML criterion at convergence": -2.0 * fit.loglik}
    else:
        criterion = {
            "AIC": -2.0 * fit.loglik + 2.0 * fit.df,
            "BIC": -2.0 * fit.loglik + fit.df * math.log(fit.nobs) if fit.nobs > 0 else math.nan,
            "logLik": fit.loglik,
            "deviance": -2.0 * fit.loglik,
            "df.resid": fit.nobs - fit.df,
        }

    scaled = None
    if len(fit.fitted) == len(fit.y) and len(fit.fitted) > 0:
        scaled = np.quantile(pearson_residuals(fit), [0.0, 0.25, 0.5, 0.75, 1.0])

    random_effects = []
    theta_off = 0
    for g in range(len(fit.varcorr)):
        sd, corr = fit.stddev_corr(g)
        q = len(sd)
        group, terms = fit.re_groups[g]
        # stddev_se is theta-layout: one entry per vech element per grouping;
        # a stddev's SE sits at its diagonal vech position.
        se_sd = []
        for i in range(q):
            di = theta_off + (i * q - (i * i - i) // 2)
            se_sd.append(fit.stddev_se[di] if di < len(fit.stddev_se) else math.nan)
        theta_off += q * (q + 1) // 2
        random_effects.append(
            {
                "group": group,
                "terms": list(terms),
                "variance": sd * sd,
                "stddev": sd,
                "se": np.array(se_sd),
                "corr": corr,
            }
        )
    mixed = len(fit.varcorr) > 0
    residual_variance = fit.dispersion if (mixed and fit.family == "gaussian") else None

    groups = []
    if len(fit.ranef_levels) == len(fit.re_groups):
        groups = [(name, int(n)) for (name, _), n in zip(fit.re_groups, fit.ranef_levels)]

    corr_fixed = None
    if len(beta) >= 2:
        vc = np.asarray(fit.vcov, dtype=float)
        d = np.sqrt(np.diag(vc))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr_fixed = vc / np.outer(d, d)

    return Summary(
        method=method_line(fit),
        formula=fit.formula,
        family=fit.family,
        link=fit.link,
        reml=fit.reml,
        criterion=criterion,
        scaled_residuals=scaled,
        random_effects=random_effects,
        residual_variance=residual_variance,
        nobs=fit.nobs,
        groups=groups,
        coef_names=list(fit.names),
        estimate=beta,
        std_error=se,
        z_value=z,
        p_value=p,
        aliased=aliased,
        corr_fixed=corr_fixed,
        dispersion_label=_DISPERSION_LABEL.get(fit.family),
        dispersion=fit.dispersion,
        n_eval=fit.n_eval,
        converged=fit.converged,
        singular=fit.singular,
        n_aliased=int(aliased.sum()),
        footnote=FOOTNOTE,
    )


@dataclass
class Summary:
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
    z_value: np.ndarray
    p_value: np.ndarray
    aliased: np.ndarray
    corr_fixed: np.ndarray | None
    dispersion_label: str | None
    dispersion: float
    n_eval: int
    converged: bool
    singular: bool
    n_aliased: int
    footnote: str

    def __repr__(self):
        return self.text()

    # -- text ------------------------------------------------------------

    def text(self):
        by_title = {t: (h, r) for t, h, r in self._tables()}
        out = [self.method, f"Formula: {self.formula}", f" Family: {self.family} ({self.link})", ""]

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
        out.append("")

        out.append("Fixed effects:")
        header, rows = by_title["Fixed effects"]
        name_w = max(12, max((len(n) for n in self.coef_names), default=4))
        widths = [12, 12, 10, 10]
        out.append(
            f"{'':<{name_w}} " + " ".join(f"{h:>{wi}}" for h, wi in zip(header[1:5], widths))
        )
        for r in rows:
            out.append(
                f"{r[0]:<{name_w}} "
                + " ".join(f"{c:>{wi}}" for c, wi in zip(r[1:5], widths))
                + f" {r[5]}"
            )
        out.append("---")
        out.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        if self.n_aliased:
            out.append(f"({self.n_aliased} coefficient(s) not defined because of singularities)")
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
            out.append(f"{self.dispersion_label}: {fmt(self.dispersion, 6)}")
        out.append(f"Optimizer evaluations: {self.n_eval}, converged: {self.converged}")
        if self.singular:
            out.append("boundary (singular) fit: see help('isSingular')")
        out.append(self.footnote)
        return "\n".join(out)

    # -- shared block decomposition ---------------------------------------

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
                            fmt(self.criterion[k], 5) if k != "df.resid" else str(self.criterion[k])
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
                for i in range(len(re["stddev"])):
                    corr = " ".join(f"{re['corr'][i, j]:.2f}" for j in range(i))
                    rows.append(
                        [
                            re["group"] if i == 0 else "",
                            re["terms"][i] if i < len(re["terms"]) else "",
                            fmt(re["variance"][i]),
                            fmt(re["stddev"][i]),
                            fmt(re["se"][i]),
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
                        "",
                    ]
                )
            tables.append(
                ("Random effects", ["Groups", "Name", "Variance", "Std.Dev.", "se", "Corr"], rows)
            )
        rows = [
            [
                n,
                fmt(self.estimate[i]),
                fmt(self.std_error[i]),
                fmt(self.z_value[i]),
                fmt_p(self.p_value[i]),
                stars(self.p_value[i]).strip(),
            ]
            for i, n in enumerate(self.coef_names)
        ]
        tables.append(
            ("Fixed effects", ["", "Estimate", "Std. Error", "z value", "Pr(>|z|)", ""], rows)
        )
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

    def _header_lines(self):
        lines = [self.method, f"Formula: {self.formula}", f"Family: {self.family} ({self.link})"]
        if self.groups:
            lines.append(
                f"Number of obs: {self.nobs}, groups: "
                + "; ".join(f"{n}, {k}" for n, k in self.groups)
            )
        else:
            lines.append(f"Number of obs: {self.nobs}")
        return lines

    def _footer_lines(self):
        # "Signif. codes" rides here (not in `_tables`, it is a legend, not a
        # data row) so its thresholds still reach the non-text renderers: the
        # cross-renderer numeric-parity test checks every number in `text()`
        # shows up somewhere in each render.
        lines = ["Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"]
        if self.dispersion_label is not None:
            lines.append(f"{self.dispersion_label}: {fmt(self.dispersion, 6)}")
        lines.append(f"Optimizer evaluations: {self.n_eval}, converged: {self.converged}")
        if self.singular:
            lines.append("boundary (singular) fit: see help('isSingular')")
        if self.n_aliased:
            lines.append(f"{self.n_aliased} coefficient(s) not defined because of singularities")
        lines.append(self.footnote)
        return lines

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
                "</style>"
            ),
            '<div class="glmm-summary">',
        ]
        out += [f"<p>{e(l)}</p>" for l in self._header_lines()]
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
        Fixed effects first, random effects second, then the rest, header and
        footer as `%` comments — a table goes into a document, a header does not."""
        out = [f"% {l}" for l in self._header_lines()]
        tables = self._tables()
        order = ["Fixed effects", "Random effects"] + [
            t for t, _, _ in tables if t not in ("Fixed effects", "Random effects")
        ]
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
        out = [f"// {l}" for l in self._header_lines()]
        tables = self._tables()
        order = ["Fixed effects", "Random effects"] + [
            t for t, _, _ in tables if t not in ("Fixed effects", "Random effects")
        ]
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
