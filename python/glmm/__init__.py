"""GLMM Python port — formula + data -> fit.

`glmm.fit` parses `formula` against `data`'s columns through the Rust
`glmm::formula` module, fits via the `glmm` kernel (through the `glmm._native`
PyO3 extension), and returns a `Fit`. Four combinations from the API spec's
family/link table are GLMM 0.1.1 ("approved design, not yet implemented in the
kernel" — docs/GLMM/0.1.1/2026-07-04-glmm-0.1.1-families-links-spec.md) and
raise a clean `NotImplementedError`: `family="inversegaussian"`,
`link="cloglog"`, quasi-likelihood `dispersion=` on binomial/poisson, and
`init_theta=<float>` (no kernel hook exists yet to seed the negative-binomial
search — only the default `init_theta=None` cold-start is supported).

Public surface is `fit`, `Fit`, and the warning categories the diagnostics
channel raises (`DiagnosticWarning`, `IllConditionedWarning`,
`PirlsExhaustedWarning`, `UnusedGroupingLevelsWarning`).
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np

from glmm import _native

__all__ = [
    "DiagnosticWarning",
    "Fit",
    "IllConditionedWarning",
    "PirlsExhaustedWarning",
    "UnusedGroupingLevelsWarning",
    "fit",
]


class DiagnosticWarning(UserWarning):
    """Base category for the kernel's per-fit solver notes.

    Every note warns under a subclass of this, so
    `warnings.filterwarnings("ignore", category=glmm.DiagnosticWarning)`
    silences the whole channel. A note kind this version of the wrapper does not
    recognize — the Rust `Note` enum is `#[non_exhaustive]`, so a newer kernel
    may raise one — warns under this base category rather than being dropped.
    """


class IllConditionedWarning(DiagnosticWarning):
    """Two or more fixed-effect columns are entangled — near-collinear, but not
    redundant.

    The fit is real and the estimates are honest; the standard errors are large
    because the data cannot separate the columns. Distinct from an *aliased*
    column, which is exactly redundant and is dropped (`Fit.aliased`).
    """


class PirlsExhaustedWarning(DiagnosticWarning):
    """A GLMM's inner PIRLS solve ran its full iteration cap without converging.

    Two cases, and the message states which one occurred (the note's
    `final_eval` field carries the distinction). On a BOBYQA trial point during
    the search, that point was rejected and the search steered around it —
    observation-only, no fitted number is affected. On the final re-evaluation
    at the converged fit, the reported estimates rest on that truncated solve —
    the more serious of the two cases.
    """


class UnusedGroupingLevelsWarning(DiagnosticWarning):
    """A grouping factor declares levels that carry no row but still occupy
    random-effect columns.

    An empty cluster between two observed ones contributes nothing to the
    likelihood and costs model width anyway, because the block is sized by the
    largest observed code. `ranef` reports such a level with its mode fully
    shrunk to zero; lme4 has no counterpart row at all. Dropping the level
    (pandas `.cat.remove_unused_categories()`, R `droplevels`) removes both the
    row and the wasted width. A level declared *after* the last observed one
    costs nothing and never warns.
    """


# Family table — mirrors the API spec §3.2 and GLMM/src/family.rs.
_FAMILIES = {
    "gaussian": {"default_link": "identity", "links": {"identity"}},
    "binomial": {"default_link": "logit", "links": {"logit", "probit", "cloglog"}},
    "poisson": {"default_link": "log", "links": {"log"}},
    "gamma": {"default_link": "log", "links": {"log", "inverse"}},
    "negativebinomial": {"default_link": "log", "links": {"log"}},
    "inversegaussian": {"default_link": "log", "links": {"log", "inverse_squared"}},
}

# Families where `dispersion=` is meaningful: phi families (gamma,
# inversegaussian) plus binomial/poisson, where "estimate"/float means
# quasi-likelihood (GLM only). gaussian and negativebinomial have no phi
# knob (negbin's parameter is theta).
_DISPERSION_FAMILIES = {"binomial", "poisson", "gamma", "inversegaussian"}

_MAX_NAGQ = 25  # mirrors GLMM/src/consts.rs::MAX_NAGQ — change together


def _columns(data):
    """Extract {name: column} from dict / pandas / polars / pyarrow (spec §3.1)
    without a hard dependency on any dataframe library.

    Columns are returned UNFLATTENED so `fit` can still see a categorical dtype:
    `list(col)` drops it, and with it the level order the caller declared (see
    `_levels_and_codes`)."""
    if isinstance(data, dict):
        return dict(data)
    if hasattr(data, "column_names"):  # pyarrow Table
        return {c: data.column(c) for c in data.column_names}
    if hasattr(data, "columns"):  # pandas / polars DataFrame
        return {c: data[c] for c in data.columns}
    raise TypeError(f"data must be a dict, DataFrame, or pyarrow Table; got {type(data).__name__}")


def _levels_and_codes(col):
    """A factor column as (levels, per-row codes), or None if `col` is not
    categorical.

    A declared level order is the whole point: level 0 is the treatment-contrast
    base, so `pd.Categorical(x, categories=["low","med","high"])` must fit
    against `"low"`, not against whichever label happens to sort first. Rust's
    `Column::Factor` takes the order from us rather than re-deriving it.

    Duck-typed on `.categories`/`.codes` — no hard pandas dependency, matching
    `_columns`' `hasattr(data, "column_names")` style. Covers pandas
    `Categorical`/`Series[category]` (via `.cat`) and pyarrow `DictionaryArray`.
    A plain string column has no declared order and is handled by the caller."""
    cat = getattr(col, "cat", col)  # pandas Series[category] -> .cat accessor
    if hasattr(cat, "categories") and hasattr(cat, "codes"):
        levels = [str(v) for v in cat.categories]
        codes = [int(c) for c in cat.codes]
        # pandas marks a missing value as code -1; there is no level to fit it
        # against, and silently dropping the row would change the model.
        if any(c < 0 for c in codes):
            raise ValueError("categorical column has missing values (code -1); drop or fill them")
        return levels, codes
    if hasattr(col, "dictionary") and hasattr(col, "indices"):  # pyarrow DictionaryArray
        levels = [str(v) for v in col.dictionary.to_pylist()]
        codes = col.indices.to_pylist()
        if any(c is None for c in codes):
            raise ValueError("categorical column has missing values; drop or fill them")
        return levels, [int(c) for c in codes]
    return None


def _sorted_levels_and_codes(labels):
    """A plain string column as (levels, codes), levels lexicographic.

    Mirrors Rust's `Column::factor_from_labels` — the R `factor()` default, and
    all that can be inferred when the caller declared no order. Doing the sort
    here rather than in the parser is what makes it a default the caller can
    override, instead of one imposed on every factor."""
    levels = sorted(set(labels))
    index = {lvl: i for i, lvl in enumerate(levels)}
    return levels, [index[v] for v in labels]


@dataclass
class Fit:
    """Fit result — mirrors the Rust `Fit` (src/fit/mod.rs), plus coefficient
    names from the formula. Returned by `fit`, not constructed by callers.

    `converged`, `singular` and `aliased` are properties over `diagnostics`,
    not dataclass fields: read them off the Fit as before, but reach for
    `diagnostics` when reflecting over the dataclass (`asdict`, `fields`)."""

    beta: np.ndarray  # (p,) fixed-effect estimates
    se: np.ndarray  # (p,) standard errors; NaN where unavailable
    vcov: np.ndarray  # (p, p) full Cov(beta-hat); se is sqrt of its diagonal
    tau2: np.ndarray  # legacy per-element RE variances (q=1 only) — prefer varcorr
    varcorr: list  # per grouping: vech-packed (column-major lower-tri) RE covariance
    stddev_se: (
        np.ndarray
    )  # SE of each RE stddev, theta layout (not beta-aligned); NaN where unavailable
    # Everything the solver reports about the fit itself, mirroring the Rust
    # `Diagnostics` (src/fit/mod.rs) as a plain dict:
    #   converged  bool
    #   singular   bool — boundary fit, >=1 variance component pinned at 0
    #                     (lme4's isSingular)
    #   aliased    (p,) bool — rank-deficient columns dropped (lme4's NA
    #                     coefficients)
    #   boundary   "interior" | "at_boundary" | "no_optimum" | "unknown" —
    #                     where the accepted theta sits. Only the dense LMM and
    #                     dense GLMM routes distinguish all three; elsewhere it
    #                     is back-derived from `singular`, so "no_optimum" is
    #                     unreachable there and "interior" means "not pinned",
    #                     not "verified interior".
    #   pinned     list per grouping (varcorr order) of per-component bools:
    #                     pinned[g][i] pairs with stddev_corr(g)[0][i]. ON A
    #                     CONVERGED FIT, EMPTY MEANS NOTHING WAS PINNED — a
    #                     model with no variance components at all (OLS, GLM,
    #                     fixed-effect-only negative binomial) also reports
    #                     empty, for the same reason: there was nothing to pin.
    #                     `fit()` does not raise on `converged: False`; on that
    #                     path `pinned` is always the empty NaN-fill default
    #                     regardless of what the optimizer was doing when it
    #                     gave up, so it carries no information there.
    #   notes      list of {"kind": str, "columns": [int], "pivot": float,
    #                     "evals": int, "final_eval": bool, "detail": str} —
    #                     observations with no dedicated field,
    #                     each raised as a `DiagnosticWarning` subclass by `fit`.
    #                     Mostly the solver's; the formula lowering contributes
    #                     "unused_grouping_levels", whose names ride in `detail`.
    #                     `columns` are 0-based indices into `names` (the R
    #                     package reports the same thing 1-based, per R's own
    #                     convention). An absent note means "not detected",
    #                     never "checked and clean": the dense GLMM route
    #                     records no pivot and the sparse route refuses rather
    #                     than flagging.
    # `converged`, `singular` and `aliased` also stay readable straight off the
    # Fit (properties below) — one storage location, unchanged ergonomics.
    diagnostics: dict
    dispersion: float  # phi (gamma / inverse-gaussian) / theta (negbin) / residual sigma^2 (gaussian) / 1.0 (binomial, poisson)
    names: list  # coefficient names, aligned with beta
    re_groups: list  # per grouping, in varcorr order: (name, [term names])
    n_eval: int  # optimizer objective evaluations (0 on the closed-form/IRLS paths)
    # Minimized optimizer criterion. NOT comparable across models and NOT an AIC
    # input — it carries the Rust `Fit::deviance` caveat: for an LMM it is lme4's
    # REMLcrit minus a data-independent constant; for a GLMM it is the marginal
    # Laplace deviance, which differs from -2*logLik by a data-only saturated
    # constant. NaN for OLS/GLM and on numerical failure.
    deviance: float
    # Log-likelihood at the fitted parameters, on the logLik() scale (R/lme4).
    # For an LMM this is the REML criterion (see `reml` below); for OLS/GLM/GLMM
    # it is the ordinary log-likelihood. NaN wherever `deviance`'s failure modes
    # apply.
    loglik: float
    # Parameters counted for AIC/BIC: retained fixed effects + RE parameters +
    # 1 if the family estimates a dispersion/scale. 0 on degenerate NaN-fill
    # paths.
    df: int
    # True iff `loglik` is a REML criterion rather than an ML log-likelihood
    # (the Gaussian LMM paths). Model comparisons (AIC/LRT) across fits with
    # different fixed effects are invalid when this is set.
    reml: bool
    # Fitted means mu-hat per row (n,). Empty on non-converged fits.
    fitted: np.ndarray
    # Random-effect conditional modes b-hat, one block per grouping in
    # varcorr/re_groups order, each block level-major (level l's q values at
    # [l*q .. (l+1)*q]). Empty on non-converged fits; see `ranef_levels` for
    # slicing per grouping. This is the raw numbers -- for the labelled form,
    # read `ranef_blocks` instead and do NOT slice this yourself: which layout a
    # grouping lands in is a data-dependent decision inside the kernel.
    ranef: np.ndarray
    # Level count per grouping, for slicing `ranef`: ranef.size == sum(levels *
    # q_per_group). Empty exactly when `ranef` is.
    ranef_levels: np.ndarray
    # The same conditional modes, labelled: a list of dicts, one per grouping in
    # re_groups order, each with
    #   group   str        -- grouping factor name
    #   terms   list[str]  -- column names, "(Intercept)" first
    #   levels  list[str]  -- row labels, one per level
    #   values  ndarray    -- (len(levels), len(terms))
    # Padded slots of a nested grouping are already dropped, so `levels` is
    # exactly the levels that exist. Not a DataFrame: the package's only
    # dependency is numpy, and `fit()` accepts duck-typed data precisely so it
    # does not require pandas. Empty exactly when `ranef` is.
    ranef_blocks: list

    @property
    def converged(self):
        """`diagnostics["converged"]` — the most-read field, kept at the top
        level so moving the storage costs no caller an extra hop."""
        return self.diagnostics["converged"]

    @property
    def singular(self):
        """`diagnostics["singular"]`."""
        return self.diagnostics["singular"]

    @property
    def aliased(self):
        """`diagnostics["aliased"]`."""
        return self.diagnostics["aliased"]

    def stddev_corr(self, group_idx):
        """Split grouping `group_idx`'s vech-packed covariance into
        (stddevs, correlation matrix) — mirrors Rust `Fit::stddev_corr`
        (src/fit/mod.rs): column-major lower-triangular vech."""
        vech = np.asarray(self.varcorr[group_idx], dtype=float)
        m = len(vech)
        q = (math.isqrt(1 + 8 * m) - 1) // 2
        if q * (q + 1) // 2 != m:
            raise ValueError(f"varcorr[{group_idx}] is not a valid vech (len {m})")

        def idx(r, c):
            return c * q - (c * c - c) // 2 + (r - c)

        stddev = np.array([math.sqrt(vech[idx(i, i)]) for i in range(q)])
        corr = np.eye(q)
        for c in range(q):
            for r in range(c + 1, q):
                rho = vech[idx(r, c)] / (stddev[r] * stddev[c])
                corr[r, c] = corr[c, r] = rho
        return stddev, corr

    def summary(self):
        """Build the coefficient table, print it, and return it as text.

        z/p are Wald (z = beta/se, p = 2*(1 - Phi(|z|)) = erfc(|z|/sqrt 2)),
        derived here — the Rust `Fit` carries no p-values. Wald-z (not t)
        matches the GLM/GLMM convention and the absence of a residual-df
        field on the kernel output."""
        beta = np.asarray(self.beta, dtype=float)
        se = np.asarray(self.se, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = beta / se
        p = np.array(
            [math.erfc(abs(v) / math.sqrt(2.0)) if math.isfinite(v) else math.nan for v in z]
        )

        name_w = max(12, max((len(n) for n in self.names), default=4))
        lines = [f"{'name':<{name_w}} {'estimate':>12} {'std.error':>12} {'z':>10} {'p':>10}"]
        for i, name in enumerate(self.names):
            if self.aliased[i]:
                est = se_i = z_i = p_i = math.nan  # lme4 prints NA here
            else:
                est, se_i, z_i, p_i = beta[i], se[i], z[i], p[i]
            lines.append(f"{name:<{name_w}} {est:>12.4g} {se_i:>12.4g} {z_i:>10.4g} {p_i:>10.4g}")

        if len(self.varcorr):
            lines.append("")
            lines.append("Random effects:")
            # stddev_se is theta-layout: one entry per vech element per
            # grouping; a stddev's SE sits at its diagonal vech position.
            theta_off = 0
            for g in range(len(self.varcorr)):
                sd, corr = self.stddev_corr(g)
                q = len(sd)
                # re_groups is emitted in varcorr order (asserted in the Rust
                # shim), so index g names block g.
                group_name, terms = self.re_groups[g]
                lines.append(f"  {group_name}:")
                term_w = max((len(t) for t in terms), default=0)
                for i in range(q):
                    di = theta_off + (i * q - (i * i - i) // 2)
                    sd_se = self.stddev_se[di] if di < len(self.stddev_se) else math.nan
                    corr_cells = " ".join(f"{corr[i][j]:>8.3f}" for j in range(i + 1))
                    term = terms[i] if i < len(terms) else ""
                    lines.append(
                        f"    {term:<{term_w}}  sd {sd[i]:>10.4g}  se {sd_se:>10.4g}"
                        f"  corr {corr_cells}"
                    )
                theta_off += q * (q + 1) // 2

        lines.append("")
        lines.append(
            f"dispersion: {self.dispersion:.6g}   converged: {self.converged}   singular: {self.singular}"
        )
        text = "\n".join(lines)
        print(text)
        return text


def _pinned_detail(res):
    """Names of the RE components the optimizer pinned at the boundary, for the
    singular warning: "sd(term | group) pinned at the variance boundary" each.

    Read straight off `diagnostics["pinned"]`, which is the kernel's own record
    of what it pinned. Do NOT reconstruct it from `varcorr`: on a grouping with
    q >= 2 the pin fixes the diagonal of the Cholesky factor, while the reported
    stddev is sqrt(lambda_offdiag^2 + lambda_diag^2) and so lands at ~1e-9
    rather than at 0 — a scan for exactly-zero stddevs misses those pins.

    That same fact is why the message says "pinned at the variance boundary" and
    not "= 0", which is what it used to say. What is pinned is the Cholesky
    diagonal; the stddev this fit reports for the component keeps whatever the
    off-diagonal settled on, and on a q >= 2 block that has been measured as
    high as 2.2e-3 against a 0.689 sibling — a number that shows up in a printed
    VarCorr at default rounding. A warning must not contradict a number the same
    fit prints.

    Empty means nothing was pinned — including a model with no variance
    components to pin. `singular` can still be `True` with `pinned` empty (the
    post-hoc negligible-stddev check is independent of the optimizer's own pin
    decision), so the bare lme4 text stands regardless; the caller must not
    read an empty `pinned` as "not singular"."""
    pinned = res.diagnostics["pinned"]
    if not pinned:
        return []
    parts = []
    # strict: when the kernel reports pins at all it emits one flag block per
    # varcorr block, so a length mismatch there is a bug to surface, not
    # groupings to drop quietly. The empty case is handled above — it is the
    # documented "nothing was pinned", not a mismatch.
    for flags, (group, terms) in zip(pinned, res.re_groups, strict=True):
        for i, is_pinned in enumerate(flags):
            if is_pinned:
                term = terms[i] if i < len(terms) else f"component {i}"
                parts.append(f"sd({term} | {group}) pinned at the variance boundary")
    return parts


def _note_warning(note, names):
    """One kernel note as (message, warning category).

    The `kind` string, not the English text, is the stable identifier — an
    unrecognized kind comes from a kernel newer than this wrapper (the Rust
    `Note` enum is `#[non_exhaustive]`) and still warns, under the base
    category. The R port maps the same kinds to condition classes
    (r/R/fastglmm.R) — change together."""
    if note["kind"] == "ill_conditioned":
        named = ", ".join(names[i] if i < len(names) else f"column {i}" for i in note["columns"])
        # "entangled with" rather than "the design does not identify <name>":
        # entanglement is symmetric, and the kernel names the column attaining
        # the SMALLEST scaled pivot, which is one member of the group and not a
        # statement about the others. Wording it as a fact about that one column
        # reads as if its partners were exonerated.
        return (
            (
                f"{named} is entangled with one or more other columns (scaled pivot "
                f"{note['pivot']:.3g}): the fit is real and the estimates are honest, "
                "but the standard errors are large. Only the column the pivot search "
                "reached is named; its partners are equally implicated and are not "
                "identified here."
            ),
            IllConditionedWarning,
        )
    if note["kind"] == "unused_grouping_levels":
        return (
            (
                f"grouping levels with no rows occupy random-effect columns "
                f"({note['detail']}); their conditional modes are reported as exactly "
                "zero. Drop the unused levels to remove both the rows and the wasted "
                "model width."
            ),
            UnusedGroupingLevelsWarning,
        )
    if note["kind"] == "pirls_exhausted":
        # `final_eval` is the case split the docstring describes: a rejected
        # trial point is benign, the final re-evaluation at the converged fit
        # feeds the reported estimates.
        if note["final_eval"]:
            return (
                (
                    "the final PIRLS re-evaluation at the converged fit ran its "
                    "full iteration cap without converging: the reported "
                    "estimates rest on that truncated solve."
                ),
                PirlsExhaustedWarning,
            )
        return (
            (
                "a GLMM inner PIRLS solve ran its full iteration cap without "
                "converging. This is observation-only and no fitted number is "
                "affected."
            ),
            PirlsExhaustedWarning,
        )
    return (
        (
            "the kernel reported a solver note this version of glmm does not "
            f"recognize ({note['kind']!r})"
        ),
        DiagnosticWarning,
    )


def fit(
    data,
    formula,
    family="gaussian",
    *,
    link=None,
    dispersion=None,
    init_theta=None,
    weights=None,
    offset=None,
    wald_se="hessian",
    nagq=1,
    warm_start=None,
):
    """Fit `formula` against `data`'s columns and return a `Fit`.

    data: dict[str, array-like]; DataFrame / Arrow Table also accepted (duck-typed).
    formula: R-style string, e.g. "y ~ x + z + (1 + x | g)".
    family: gaussian | binomial | poisson | gamma | negativebinomial | inversegaussian.
    nagq: adaptive Gauss-Hermite quadrature nodes per random-effect dimension
        (odd, 1..=25; default 1 = Laplace). k>1 applies to binomial/Poisson
        models with a single grouping factor and q <= 3 random effects per
        group (temporary cap); any other shape warns and falls back to Laplace.
    init_theta: negative-binomial shape seed, named for `MASS::glm.nb(init.theta=)`
        — the same knob, and the name the R port exposes. Distinct from
        `warm_start["theta"]`, which is the random-effect Cholesky vector
        (lme4's `start=list(theta=)`); they are unrelated parameters and both
        may be passed in one call.
    offset: per-row additive offset on the linear-predictor scale, length n
        (R's `offset=`): eta = offset + X*beta (+ Z*b). A fixed known
        contribution, not a parameter — the canonical use is a Poisson
        exposure, offset = log(exposure). None = no offset.
    warm_start: {"beta": …, "theta": …} optimizer start. See `init_theta` above
        for why "theta" here is NOT the negative-binomial shape.

    A categorical column's level order is honored: level 0 is the
    treatment-contrast base, so `pd.Categorical(x, categories=[…])` fits against
    the first category you list. A plain string column has no declared order and
    is sorted lexicographically (R's `factor()` default).

    See the API spec for the remaining knobs.
    """
    if family not in _FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(_FAMILIES)}")
    fam = _FAMILIES[family]
    if link is None:
        link = fam["default_link"]
    elif link not in fam["links"]:
        raise ValueError(
            f"family {family!r} does not support link {link!r}; "
            f"expected one of {sorted(fam['links'])}"
        )

    # `|` marks a random-effect term, so its presence is the mixed/GLM split
    # — decidable without the (M6, Rust-side) formula parser.
    mixed = "|" in formula

    if family == "inversegaussian" and mixed:
        raise ValueError(
            "family 'inversegaussian' is GLM-only: random-effect terms "
            "(`(... | g)`) are not supported"
        )

    if wald_se not in ("hessian", "rx"):
        raise ValueError(f"wald_se must be 'hessian' or 'rx', got {wald_se!r}")

    if not (
        isinstance(nagq, int)
        and not isinstance(nagq, bool)
        and 1 <= nagq <= _MAX_NAGQ
        and nagq % 2 == 1
    ):
        raise ValueError(f"nagq must be an odd integer in 1..={_MAX_NAGQ}, got {nagq!r}")

    # Valid-but-inapplicable options: warn and strip (spec §3.5). The kernel
    # boundary-faults on inapplicable options and a Rust panic across the FFI
    # is not an acceptable user error, so nothing inapplicable may reach it.
    if dispersion is not None and family not in _DISPERSION_FAMILIES:
        warnings.warn(
            f"dispersion= is not applicable to family {family!r}; ignored",
            stacklevel=2,
        )
        dispersion = None
    if dispersion is not None:
        if not (
            dispersion == "estimate"
            or (isinstance(dispersion, (int, float)) and not isinstance(dispersion, bool))
        ):
            raise ValueError(f"dispersion must be None, 'estimate', or a float, got {dispersion!r}")
        if family in ("binomial", "poisson") and mixed:
            warnings.warn(
                "quasi-likelihood dispersion on binomial/poisson is GLM-only; "
                "ignored for a mixed formula",
                stacklevel=2,
            )
            dispersion = None
    if init_theta is not None and family != "negativebinomial":
        warnings.warn(
            "init_theta= applies only to family 'negativebinomial'; ignored",
            stacklevel=2,
        )
        init_theta = None
    if warm_start is not None:
        if not isinstance(warm_start, dict):
            raise TypeError(
                "warm_start must be a dict with keys 'beta'/'theta', "
                f"got {type(warm_start).__name__}"
            )
        unknown = set(warm_start) - {"beta", "theta"}
        if unknown:
            warnings.warn(f"warm_start keys ignored: {sorted(unknown)}", stacklevel=2)
            warm_start = {k: v for k, v in warm_start.items() if k in ("beta", "theta")}

    # --- kernel-gap checks: GLMM 0.1.1 not yet implemented (see module docstring) ---
    if family == "inversegaussian":
        raise NotImplementedError(
            "family 'inversegaussian' requires GLMM 0.1.1; not yet implemented in the kernel"
        )
    if link == "cloglog":
        raise NotImplementedError(
            "link 'cloglog' requires GLMM 0.1.1; not yet implemented in the kernel"
        )
    if dispersion == "estimate" and family == "gamma":
        # gamma's family default (dispersion=None) already computes the
        # Pearson estimate, so "estimate" needs no distinct kernel state.
        dispersion = None
    if family in ("binomial", "poisson") and dispersion is not None:
        raise NotImplementedError(
            f"quasi-likelihood dispersion on family {family!r} requires GLMM "
            "0.1.1; not yet implemented in the kernel"
        )
    if init_theta is not None:
        raise NotImplementedError(
            "init_theta= (negative-binomial shape seed) has no kernel hook yet; "
            "only init_theta=None (cold-start search) is supported"
        )

    # Classify each column numeric vs factor, and hand factors across as
    # (levels, codes) so the caller's reference level survives into Rust.
    # A declared categorical is checked FIRST: dtype beats value-sniffing, or a
    # categorical of non-strings (pd.Categorical([1, 2, 3])) would land in the
    # numeric branch and be fit as a continuous predictor.
    numeric_columns = {}
    factor_columns = {}
    for name, col in _columns(data).items():
        declared = _levels_and_codes(col)
        if declared is not None:
            factor_columns[name] = declared
            continue
        values = list(col)
        if values and isinstance(values[0], str):
            factor_columns[name] = _sorted_levels_and_codes([str(v) for v in values])
        else:
            numeric_columns[name] = [float(v) for v in values]

    warm_start_pair = None
    if warm_start is not None:
        warm_start_pair = (
            [float(v) for v in warm_start.get("beta", [])],
            [float(v) for v in warm_start.get("theta", [])],
        )

    r = _native.fit(
        formula,
        numeric_columns,
        factor_columns,
        family,
        link,
        wald_se,
        nagq,
        dispersion,
        [float(w) for w in weights] if weights is not None else None,
        [float(v) for v in offset] if offset is not None else None,
        warm_start_pair,
    )
    # nagq's shape eligibility (single grouping factor, binomial/Poisson,
    # q <= 3) is only decidable after the Rust-side formula lowering, so the
    # §3.5 warn-and-strip for it lives in the shared glmm::orchestrate module
    # (GLMM/src/orchestrate.rs); the
    # message comes back here to be raised as the same UserWarning the
    # dispersion/theta strips above use.
    if r["agq_warning"] is not None:
        warnings.warn(r["agq_warning"], stacklevel=2)
    # Wrap the native dict's plain lists back into the array types Fit's
    # dataclass documents (no numpy Rust dep — the native call returns lists).
    res = Fit(
        beta=np.asarray(r["beta"], dtype=float),
        se=np.asarray(r["se"], dtype=float),
        vcov=np.asarray(r["vcov"], dtype=float),
        tau2=np.asarray(r["tau2"], dtype=float),
        varcorr=r["varcorr"],
        stddev_se=np.asarray(r["stddev_se"], dtype=float),
        diagnostics={
            "converged": r["converged"],
            "singular": r["singular"],
            "aliased": np.asarray(r["aliased"], dtype=bool),
            "boundary": r["boundary"],
            "pinned": r["pinned"],
            "notes": r["notes"],
        },
        dispersion=r["dispersion"],
        names=r["names"],
        re_groups=r["re_groups"],
        n_eval=r["n_eval"],
        deviance=r["deviance"],
        loglik=r["loglik"],
        df=r["df"],
        reml=r["reml"],
        fitted=np.asarray(r["fitted"], dtype=float),
        ranef=np.asarray(r["ranef"], dtype=float),
        ranef_levels=np.asarray(r["ranef_levels"], dtype=int),
        ranef_blocks=[
            {
                "group": b["group"],
                "terms": b["terms"],
                "levels": b["levels"],
                "values": np.asarray(b["values"], dtype=float).reshape(
                    len(b["levels"]), len(b["terms"])
                ),
            }
            for b in r["ranef_blocks"]
        ],
    )
    # lme4 agreement (boundary-fits follow-up spec Part B step 4): lme4's exact
    # text, extended with the degenerate components. The R port emits the same
    # message (fastglmm.R) — change together.
    if res.singular:
        warnings.warn(
            "; ".join(["boundary (singular) fit: see help('isSingular')", *_pinned_detail(res)]),
            stacklevel=2,
        )
    # One warning per note, each under its own category so a caller can filter
    # by kind. The R port raises the same set as classed conditions — change
    # together.
    for note in res.diagnostics["notes"]:
        message, category = _note_warning(note, res.names)
        warnings.warn(message, category=category, stacklevel=2)
    return res
