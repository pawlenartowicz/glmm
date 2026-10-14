"""GLMM Python port — formula + data -> fit.

`glmm.fit` parses `formula` against `data`'s columns through the Rust
`glmm::formula` module, fits via the `glmm` kernel (through the `glmm._native`
PyO3 extension), and returns a `Fit`. One option is not yet implemented
in the kernel and raises a clean `NotImplementedError`: quasi-likelihood
`dispersion=` on binomial/poisson.

Public surface is `fit`, `Fit`, the tests of `glmm.anova` (`anova`,
`compare`, `TermTable`, `ComparisonTable`, `Comparison`, in `glmm/anova.py`),
and the warning categories the diagnostics channel raises (`DiagnosticWarning`, `AgqFallbackWarning`,
`ArgumentIgnoredWarning`, `ConstantResponseWarning`, `DesignUnsolvableWarning`,
`ExactProfileFallbackWarning`,
`FitFailedWarning`, `GradientSearchFallbackWarning`, `GlmDivergedWarning`, `HessianSeFallbackWarning`,
`IllConditionedWarning`, `NbShapeUnsettledWarning`, `NoCoefficientsWarning`,
`NonIntegerResponseWarning`, `PirlsExhaustedWarning`, `PrecisionLimitedWarning`,
`RowsDroppedWarning`, `SatterthwaiteCurvatureWarning`,
`SearchLimitWarning`, `SeUnavailableWarning`, `SingleLevelGroupingDroppedWarning`,
`SingularFitWarning`,
`TooFewRowsWarning`, `UnusedGroupingLevelsWarning`).
"""

import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np

from glmm import _native

# This package's directory, with the separator: `_warn` skips its frames.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep

__all__ = [
    "AgqFallbackWarning",
    "ArgumentIgnoredWarning",
    "Comparison",
    "ComparisonTable",
    "ConstantResponseWarning",
    "DesignUnsolvableWarning",
    "DiagnosticWarning",
    "ExactProfileFallbackWarning",
    "Fit",
    "FitFailedWarning",
    "GlmDivergedWarning",
    "GradientSearchFallbackWarning",
    "HessianSeFallbackWarning",
    "IllConditionedWarning",
    "NbShapeUnsettledWarning",
    "NoCoefficientsWarning",
    "NonIntegerResponseWarning",
    "PirlsExhaustedWarning",
    "PrecisionLimitedWarning",
    "RowsDroppedWarning",
    "SatterthwaiteCurvatureWarning",
    "SeUnavailableWarning",
    "SearchLimitWarning",
    "SingleLevelGroupingDroppedWarning",
    "SingularFitWarning",
    "TermTable",
    "TooFewRowsWarning",
    "UnusedGroupingLevelsWarning",
    "anova",
    "compare",
    "fit",
]


class DiagnosticWarning(UserWarning):
    """Base category for every warning `fit` raises.

    Every warning `fit` raises is a subclass of this, so
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
    """The final re-evaluation of a converged fit ran its full PIRLS iteration
    cap without converging.

    Every other cap-out is either observation-only (a rejected trial point
    during the search, kept in the fit's notes and never raised) or folded
    into the non-convergence warning (`search_limit` / `fit_failed`) instead,
    since the non-convergence warning already says not to use them.
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


class SingleLevelGroupingDroppedWarning(DiagnosticWarning):
    """A random-effect term's grouping factor has only one observed level, so
    no between-group variance is identifiable from it.

    lme4 errors here ("grouping factors must have > 1 sampled level"); this
    port drops the term instead and fits the rest of the model. When every
    random-effect term is dropped this way, the model fits as though the
    formula had declared none.
    """


class HessianSeFallbackWarning(DiagnosticWarning):
    """`wald_se="hessian"` was requested, but the joint Hessian was not usable
    (not positive definite, or the fit took the finite-difference Hessian and a
    perturbed deviance evaluation was non-finite).

    The standard errors reported are the RX/Schur ones instead, and
    `Fit.stddev_se` (the random-effect standard deviations' own standard
    errors) comes back `NaN` — only the joint Hessian route fills it.
    """


class GradientSearchFallbackWarning(DiagnosticWarning):
    """The gradient search (used on linear mixed models whose random slopes
    are all on one grouping, on binomial, Poisson and Gamma GLMMs fitted by
    the Laplace approximation, and on logit and probit binomial and Poisson
    GLMMs with a random intercept only and fewer than 50 rows per group,
    fitted with `nagq` above 1) did not converge, so the standard
    derivative-free search continued from the best point it had reached.

    The reported estimates and `converged` come from that second search,
    which ran with its own full evaluation budget; `n_eval` counts both.
    """


class ExactProfileFallbackWarning(DiagnosticWarning):
    """A GLMM's exact-profile outer search did not converge, so the fit reran
    once with a different search strategy (`PqlThenJoint`) from the same
    start.

    The reported estimates come from that rerun when it converged. When the
    rerun also failed, they come from whichever of the two attempts reached
    the lower deviance, and `converged` stays `False` either way. Only
    affects mixed models fitted with the default Laplace approximation, and
    not those with a random slope on a second grouping or too large for the
    dense layout, which take another route.
    """


class SeUnavailableWarning(DiagnosticWarning):
    """No covariance of a mixed model's fixed effects could be formed at the
    search's end point: the information matrix is not
    positive definite to rounding (under `wald_se="hessian"`, after the joint
    Hessian failed too).

    The estimates, the variance components and the deviance are the search's
    own and `converged` is unchanged; `Fit.se`, `Fit.vcov` and
    `Fit.stddev_se` are `NaN`.
    """


class PrecisionLimitedWarning(DiagnosticWarning):
    """A mixed model's deviance is too large for double precision to resolve
    its optimum: the estimated rounding error of the deviance at the end point
    is above 0.01, so the variance estimates may be off by about a tenth of a
    standard error or more.

    Every estimate and `converged` are the search's own. Raised on counts in
    the billions or more, either through the deviance itself (a model that
    misses a large effect) or, with more than one grouping factor, through the
    log-determinant of the random-effect system.
    """


class SatterthwaiteCurvatureWarning(DiagnosticWarning):
    """The Satterthwaite degrees of freedom of a Gaussian linear mixed model are
    approximate or could not be computed.

    They are approximate when the curvature of the deviance in the variance
    parameters has flat directions (left out) or negative ones, or when the
    search stopped at its step limit. They could not be computed when the fit
    has no finite end point or the deviance near it was not finite: then
    `Fit.df_method` is "failed", every coefficient is tested with z and
    `Fit.confint` gives normal intervals.
    """


class SearchLimitWarning(DiagnosticWarning):
    """The search for the variance parameters used its evaluation budget before it
    settled, and `converged` is False. Everything is reported at the best point found;
    nothing checks how close that point is to the optimum."""


class FitFailedWarning(DiagnosticWarning):
    """A mixed-model fit stopped on a degenerate configuration: `converged` is False and
    every estimate is NaN."""


class GlmDivergedWarning(DiagnosticWarning):
    """A GLM without random effects diverged, usually from separation. The coefficients
    are the last IRLS step; the standard errors and the deviance are NaN."""


class DesignUnsolvableWarning(DiagnosticWarning):
    """An OLS fit could not factor X'X (columns redundant up to rounding). No estimates
    are returned."""


class SingularFitWarning(DiagnosticWarning):
    """Boundary (singular) fit, lme4's `isSingular`: a variance component at or near zero
    or a correlation at or near -1 or 1."""


class AgqFallbackWarning(DiagnosticWarning):
    """`nagq` above 1 on a model adaptive quadrature does not cover. The fit ran the
    Laplace approximation and `Fit.nagq` is 1."""


class ArgumentIgnoredWarning(DiagnosticWarning):
    """An argument that does not apply to this model was cleared before fitting."""


class RowsDroppedWarning(DiagnosticWarning):
    """A column the formula uses was missing a value on some rows, and those
    rows were dropped before fitting (R's default na.action = na.omit,
    applied the same way in Python since there is no `model.frame` here to do
    it for free)."""


class ConstantResponseWarning(DiagnosticWarning):
    """The fit did not converge and every row of the response has the same value."""


class TooFewRowsWarning(DiagnosticWarning):
    """The fit did not converge and there are no more rows than estimated coefficients."""


class NoCoefficientsWarning(DiagnosticWarning):
    """A model without random effects and without any fixed-effect column."""


class NbShapeUnsettledWarning(DiagnosticWarning):
    """A negative-binomial GLM's shape search stopped at its round limit before it
    settled. The fit is reported at the last shape value, with the coefficients and
    standard errors refit there."""


class NonIntegerResponseWarning(DiagnosticWarning):
    """A Poisson, negative-binomial or binomial response is not within 1e-3 of an
    integer count (binomial: the derived number of successes, `y * weights`).

    The fit still runs — the deviance and score are continuous in `y` — but the
    reported numbers describe a distribution that assumes integer counts. The 1e-3
    tolerance matches R's `binomial()$initialize` non-integer-count check. R's
    Poisson/NB warning is unrelated and much tighter (it comes from `dpois`/`dnbinom`
    inside `poisson()$aic`, at `1e-7 * max(1, abs(y))`), so R warns on e.g.
    `y = 2.0005` where glmm stays silent.
    """


# Family table — mirrors the kernel's own table in src/family.rs. `links` is
# an ordered tuple, not a set: its order is what the "expected one of" error
# text lists, and it must read the same as the R port's .FAMILIES
# (r/R/fastglmm.R) — change together.
_FAMILIES = {
    "gaussian": {"default_link": "identity", "links": ("identity",)},
    "binomial": {"default_link": "logit", "links": ("logit", "probit", "cloglog")},
    "poisson": {"default_link": "log", "links": ("log",)},
    "gamma": {"default_link": "log", "links": ("log", "inverse")},
    "negativebinomial": {"default_link": "log", "links": ("log",)},
    "inversegaussian": {"default_link": "log", "links": ("log", "inverse_squared")},
}

# Families where `dispersion=` is meaningful: phi families (gamma,
# inversegaussian) plus binomial/poisson, where "estimate"/float means
# quasi-likelihood (GLM only). gaussian and negativebinomial have no phi
# knob (negbin's parameter is theta).
_DISPERSION_FAMILIES = {"binomial", "poisson", "gamma", "inversegaussian"}

_MAX_NAGQ = 25  # mirrors GLMM/src/consts.rs::MAX_NAGQ — change together

# The families nagq > 1 covers, as the fallback message names them. Mirrors the AGQ
# eligibility check in the kernel (src/orchestrate.rs, the `if nagq > 1` block) and the R
# port's .AGQ_FAMILIES - change together.
_AGQ_FAMILIES = "binomial, Poisson, negative-binomial or Gamma"


def _columns(data):
    """Extract {name: column} from dict / pandas / polars / pyarrow
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
    `Categorical`/`Series[category]` (via `.cat`) and pyarrow dictionary
    columns, bare `DictionaryArray` or the `ChunkedArray` a Table's `.column()`
    returns. A plain string column has no declared order and is handled by the
    caller."""
    cat = getattr(col, "cat", col)  # pandas Series[category] -> .cat accessor
    if hasattr(cat, "categories") and hasattr(cat, "codes"):
        levels = [str(v) for v in cat.categories]
        # pandas marks a missing value as code -1; there is no level to fit it
        # against, so the caller drops that row the same way it drops a plain
        # column's missing value, rather than raising here.
        codes = [int(c) for c in cat.codes]
        return levels, codes
    # A pyarrow ChunkedArray has no `.dictionary`/`.indices` of its own, and
    # each chunk may carry a different dictionary. `unify_dictionaries()`
    # rewrites every chunk against one shared dictionary (first chunk's values
    # first, so the declared order survives), after which `combine_chunks()`
    # yields a single `DictionaryArray`. Keyed on the dictionary type's
    # `index_type`, not an import, so pyarrow stays optional.
    if hasattr(col, "unify_dictionaries") and hasattr(getattr(col, "type", None), "index_type"):
        col = col.unify_dictionaries().combine_chunks()
    if hasattr(col, "dictionary") and hasattr(col, "indices"):  # pyarrow DictionaryArray
        dictionary = col.dictionary.to_pylist()
        # A null index comes back as None; -1 makes it the same missing
        # sentinel pandas uses, for one row-dropping rule below. A null can
        # also be a dictionary value of its own (`null_encoding="encode"`):
        # that is a missing row too, not a level, so it is dropped from the
        # levels and the codes after it shift down.
        remap, levels = {}, []
        for i, v in enumerate(dictionary):
            if v is not None:
                remap[i] = len(levels)
                levels.append(str(v))
        codes = [-1 if c is None else remap.get(c, -1) for c in col.indices.to_pylist()]
        return levels, codes
    return None


def _values(col):
    """`col`'s scalars as plain Python values.

    `.to_pylist()` first: a pyarrow `Array`/`ChunkedArray` iterates as
    `pyarrow.Scalar` wrappers, not bare `None`/`float`, so `list(col)` on one
    holds a null as a `Scalar` that `_is_na` doesn't recognize and `float()`
    rejects outright (`TypeError`, not the missing-value path). `.to_pylist()`
    unwraps every element, null included, the same way `_levels_and_codes`
    already does for a pyarrow `DictionaryArray`'s own pieces."""
    if hasattr(col, "to_pylist"):
        return col.to_pylist()
    return list(col)


def _is_bool_column(col):
    """True for a bool column: `numpy.bool_`, pandas `bool` or the nullable
    `boolean` extension dtype, pyarrow `bool_`, or a plain list/tuple of
    Python `bool` (an `object`-dtype array unwraps to the same check).

    Checked on the column object, the way `_levels_and_codes` inspects it
    before `_values` flattens it into scalars that no longer carry a dtype."""
    dtype = getattr(col, "dtype", None)
    if dtype is not None:
        if str(dtype) in ("bool", "boolean"):
            return True
        if str(dtype) != "object":
            return False
        col = list(col)  # object-dtype array — fall through to the element check
    arrow_type = getattr(col, "type", None)  # pyarrow Array/ChunkedArray
    if arrow_type is not None:
        return str(arrow_type) == "bool"
    if isinstance(col, (list, tuple)):
        present = [v for v in col if v is not None]
        return bool(present) and all(isinstance(v, (bool, np.bool_)) for v in present)
    return False


def _bool_values(col):
    """A bool column's scalars as `"FALSE"`/`"TRUE"` strings, missing values
    kept as `None` — lme4's and the R port's own spelling for a logical
    predictor or grouping column (`factor()`'s lexicographic order puts
    `FALSE` before `TRUE`, so `False` is level 0, the treatment-contrast
    base).

    Returned as plain strings, not `(levels, codes)`, so the result flows
    through the SAME "values" path a plain string column does: levels are
    computed once from what the data actually has, after missing rows are
    dropped (`_sorted_levels_and_codes`, called lower in `fit`) — not fixed
    to both `FALSE` and `TRUE` regardless of the data, the way R's own
    `factor(rep(TRUE, n))` gives one level ("TRUE"), not two."""
    return [None if _is_na(v) else ("TRUE" if v else "FALSE") for v in _values(col)]


def _sorted_levels_and_codes(labels):
    """A plain string column as (levels, codes), levels lexicographic.

    Mirrors Rust's `Column::factor_from_labels` — the R `factor()` default, and
    all that can be inferred when the caller declared no order. Doing the sort
    here rather than in the parser is what makes it a default the caller can
    override, instead of one imposed on every factor."""
    levels = sorted(set(labels))
    index = {lvl: i for i, lvl in enumerate(levels)}
    return levels, [index[v] for v in labels]


def _formula_columns(formula, names):
    """`(used, missing, numeric_only)` for `formula` against the column names
    `names`, from one parse in the Rust formula frontend
    (`_native.formula_columns`).

    - `used`: `names` restricted to the ones the formula reads as data,
      mirroring R's own `data[all.vars(formula)]` (`r/R/fastglmm.R`) so a
      column the formula never uses — a datetime, or anything the
      numeric/factor branches below cannot convert — is never read at all.
      It is a filter: a name it cannot resolve is silently dropped.
    - `missing`: names the formula references that are not among `names` at
      all — not even as a transform's underlying column (`log(x)` when `x` is
      absent), so a missing formula column is reported before an unrelated
      `weights=`/`offset=` length happens to fail first.
    - `numeric_only`: names the formula always reads as plain numbers — the
      response (both count columns of a `cbind()` response) and the offset()
      term's column. A bool column crosses as a `FALSE`/`TRUE` factor
      everywhere else (`_is_bool_column`/`_bool_values` below); the R port
      keeps the same columns numeric off its own formula AST
      (`all.vars(formula[[2L]])` plus a small `offset()` walk).

    The Rust frontend answers rather than a scan of the formula text: it
    already knows its own identifier grammar (Unicode-aware, unlike an
    ASCII-only regex) and how to tell `log(x)` (reads `x`) apart from a column
    literally named `log`, which a bare-token scan cannot. `list(names)`:
    `_native` takes a plain list, and `names` here is often a `dict_keys` view.

    A bad formula raises here with the same message `_native.fit` would give
    for it: letting the error through here, rather than falling back to every
    name in `names`, keeps a parse error a parse error instead of routing an
    unrelated column (a datetime, say) into the numeric/factor branches below,
    where it fails to convert and raises a `TypeError` that has nothing to do
    with the formula."""
    return _native.formula_columns(formula, list(names))


def _row_array(values, name, n, missing_ok=False):
    """`weights=`/`offset=` as a float array of one finite entry per data row.

    Mirrors the R port's checks in `fastglmm()` (`r/R/fastglmm.R`): a wrong
    length or a missing entry is refused as R refuses it, here rather than by
    the kernel's entry assert, which reaches Python as a Rust panic. Inf is
    refused too, as that assert would. `n` is None when no data column was
    read; the kernel then reports the missing column first. `missing_ok`
    lets a missing entry through as NaN, for a caller that drops its row."""
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        arr = None
    if arr is None or arr.ndim != 1 or (n is not None and len(arr) != n):
        raise ValueError(f"{name} must be a numeric array with one entry per row of data")
    # `np.asarray(..., dtype=float)` turns a None entry into NaN, so this also
    # catches a missing value, which R's anyNA() rejects the same way.
    if not np.all(np.isfinite(arr) | (missing_ok & np.isnan(arr))):
        if missing_ok:
            raise ValueError(f"{name} must be finite or missing: no inf entries")
        raise ValueError(f"{name} must be finite: no NaN, inf, or missing entries")
    return arr


def _is_na(v):
    """True for a missing scalar: `None`, or NaN (float, numpy) — the only
    ordinary value that compares unequal to itself. A sentinel whose `!=`
    raises on the ambiguous truth value (pandas' `pd.NA`) is missing too,
    which is what that ambiguity itself means."""
    if v is None:
        return True
    try:
        return bool(v != v)  # noqa: PLR0124 -- NaN is the value unequal to itself
    except TypeError:
        return True


@dataclass
class Fit:
    """Fit result — mirrors the Rust `Fit` (src/fit/mod.rs), plus coefficient
    names from the formula. Returned by `fit`, not constructed by callers.

    `converged`, `singular` and `aliased` are properties over `diagnostics`,
    not dataclass fields: read them off the Fit as before, but reach for
    `diagnostics` when reflecting over the dataclass (`asdict`, `fields`).
    `aic` and `bic` are properties too, computed from `loglik`, `df` and
    `nobs`."""

    beta: np.ndarray  # (p,) fixed-effect estimates
    se: np.ndarray  # (p,) standard errors; NaN where unavailable
    vcov: np.ndarray  # (p, p) full Cov(beta-hat); se is sqrt of its diagonal
    tau2: np.ndarray  # legacy per-element RE variances (q=1 only) — prefer varcorr
    varcorr: list  # per grouping, in formula order (lme4 sorts by decreasing level count): vech-packed (column-major lower-tri) RE covariance
    stddev_se: np.ndarray  # SE of each RE stddev at its diagonal slot of the theta layout (not beta-aligned); NaN off the diagonal and where unavailable
    # Everything the solver reports about the fit itself, mirroring the Rust
    # `Diagnostics` (src/fit/mod.rs) as a plain dict:
    #   converged  bool
    #   singular   bool — boundary fit, >=1 variance component pinned at 0
    #                     (lme4's isSingular)
    #   aliased    (p,) bool — rank-deficient columns dropped (lme4's NA
    #                     coefficients)
    #   boundary   "interior" | "at_boundary" | "no_optimum" — where the
    #                     accepted theta sits. Every theta-carrying
    #                     route distinguishes all three (LMM over either
    #                     kernel, GLMM over every layout, negative binomial
    #                     included). OLS and GLM (no theta) always report
    #                     "interior", and so does a fit that failed before any
    #                     search ran (a degenerate guard), reported through
    #                     converged=False.
    #   pinned     list per grouping (varcorr order) of per-component bools:
    #                     pinned[g][i] pairs with stddev_corr(g)[0][i]. ON A
    #                     CONVERGED FIT, EMPTY MEANS NOTHING WAS PINNED — a
    #                     model with no variance components at all (OLS, GLM,
    #                     fixed-effect-only negative binomial) also reports
    #                     empty, for the same reason: there was nothing to pin.
    #                     `fit()` does not raise on `converged: False`:
    #                     `pinned` is empty on every non-converged fit — a
    #                     failed fit, and a fit stopped at its evaluation
    #                     budget, where nothing is pinned at the capped
    #                     endpoint.
    #   notes      list of {"kind": str, "columns": [int], "pivot": float,
    #                     "evals": int, "final_eval": bool, "detail": str}
    #                     — observations with no dedicated field, each raised
    #                     as a `DiagnosticWarning` subclass by `fit`. Mostly
    #                     the solver's; the formula lowering contributes
    #                     "unused_grouping_levels" (grouping and level names
    #                     ride in `detail`).
    #                     `columns` are 0-based indices into `names` (the R
    #                     package reports the same thing 1-based, per R's own
    #                     convention). An absent note means "not detected",
    #                     never "checked and clean": a fit that formed no
    #                     factor records no pivot, and every route flags
    #                     `IllConditioned` rather than refuse.
    # `converged`, `singular` and `aliased` also stay readable straight off the
    # Fit (properties below) — one storage location, unchanged ergonomics.
    diagnostics: dict
    dispersion: float  # phi (gamma / inverse-gaussian) / theta (negbin) / residual sigma^2 (gaussian) / 1.0 (binomial, poisson)
    names: list  # coefficient names, aligned with beta
    re_groups: list  # per grouping, in varcorr order: (name, [term names])
    n_eval: int  # optimizer objective evaluations (0 on the closed-form/IRLS paths)
    # The gradient (BFGS) theta search, on the models GradientSearchFallbackWarning
    # names; they describe that search even when the derivative-free search
    # finished the fit after it. `n_grad` counts its gradient evaluations
    # (`n_eval` does not); `outer_stop` is why it stopped: "fdelta", "pgrad"
    # or "noise_floor" (converged), "line_search", "max_iter",
    # "grad_unavailable" or "non_finite"; `outer_decrement` is its estimate of
    # the deviance decrease left at the endpoint. 0 / None / NaN on every fit
    # that ran no gradient search.
    n_grad: int
    outer_stop: str | None
    outer_decrement: float
    # Minimized optimizer criterion. NOT comparable across models and NOT an AIC
    # input — it carries the Rust `Fit::deviance` caveat: for an LMM it is lme4's
    # REMLcrit (on an ML fit, lme4's ML deviance) minus a data-independent
    # constant; for a GLMM it is the marginal
    # Laplace deviance, which differs from -2*logLik by a data-only saturated
    # constant on binomial/Poisson fits and equals -2*logLik exactly on Gamma
    # and negative-binomial fits. NaN for OLS/GLM and on numerical failure.
    deviance: float
    # Log-likelihood at the fitted parameters, on the logLik() scale (R/lme4).
    # On a REML fit this is the REML criterion (see `reml` below); otherwise
    # it is the ordinary log-likelihood. NaN wherever `deviance`'s failure modes
    # apply.
    loglik: float
    # Parameters counted for AIC/BIC: retained fixed effects + RE parameters +
    # 1 if the family estimates a dispersion/scale. 0 on degenerate NaN-fill
    # paths, and on a GLMM that stopped at MaxFunReached (finite `loglik`).
    df: int
    # True iff `loglik` is a REML criterion rather than an ML log-likelihood
    # (a Gaussian mixed model fitted by REML, or OLS with reml=True). Model
    # comparisons (AIC/LRT) across fits with different fixed effects are
    # invalid when this is set.
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
    # Header inputs for `summary()`. `formula`/`family`/`link` are `fit()`'s
    # own arguments, kept so the report can name what was fitted.
    formula: str
    family: str
    link: str  # resolved link, after the family default is applied
    nagq: int  # quadrature nodes that actually ran (1 after a warn-and-strip)
    # Row count the kernel fitted. Not `len(fitted)`: `fitted` is empty on a
    # non-converged fit and the summary header still prints.
    nobs: int
    # The response as the kernel fitted it (`Lowered::y` in Rust): after
    # lowering, so a `cbind(s, f)` response is the proportion s/(s+f). Backs
    # `residuals()` and the scaled-residuals block.
    y: np.ndarray
    # (n,) prior weights the kernel fitted with, None if unweighted. From the
    # kernel, not the `weights=` argument: a `cbind(s, f)` response lowers to
    # trial-count weights the caller never passed.
    weights: np.ndarray | None
    # Per coefficient, aligned with `names`: the degrees of freedom of its t test
    # (the Satterthwaite df on a Gaussian linear mixed model, n - p on OLS and on
    # a Gamma or inverse-Gaussian GLM with phi estimated), NaN where the test is
    # z and on an OLS or dispersion-GLM fit that did not converge; and that t
    # test's two-sided p-value, computed by the kernel, NaN where there is no df.
    coef_df: np.ndarray
    coef_p: np.ndarray
    # Where `coef_df` comes from: "satterthwaite", "residual" (OLS and the
    # dispersion GLMs, whatever `satterthwaite=` says), "none" (the families
    # with no df: z), "disabled" (a Gaussian mixed model fitted with
    # `satterthwaite=False`: z) or "failed" (its Satterthwaite pass failed: z).
    df_method: str
    # OLS fits only (NaN on every other fit and on a non-converged one): the
    # weighted residual sum of squares, and R's `summary.lm` multiple and
    # adjusted R-squared (0 on an intercept-only model).
    rss: float
    r_squared: float
    adj_r_squared: float
    # Every warning `fit` raised, in raise order: dicts with "tier" ("severe" |
    # "caution" | "note"), "kind" (stable; match on this), "title" and "message".
    # The texts are in documentation/warnings.md.
    warnings: list = field(default_factory=list)
    # The numeric `dispersion=` argument the caller passed (Gamma or
    # inverse-Gaussian only; None otherwise, and None when `dispersion=` was
    # left to estimate). `summary()`'s dispersion label reads this to print
    # "fixed" instead of "ML"/"Pearson". Defaulted, and last, because a
    # dataclass field with a default must follow every field without one.
    dispersion_held: float | None = None
    # The design summary of a Gaussian fit whose only random term is one random
    # intercept `(1 | g)`, with no prior weights and at least one fixed-effect
    # column; None on every other fit. A dict of three flat float arrays, with
    # K = the highest observed level code + 1 (a level below it with no rows
    # counts 0; unused levels after it are not counted) and p retained
    # fixed-effect columns, on the fit's orthogonalized design X~ (it spans the
    # column space of the fixed-effect design):
    #   counts  (K,)    rows per level
    #   s       (p*K,)  the p x K matrix X~'Z, column-major (level a at [a*p:(a+1)*p])
    #   c       (p*p,)  the p x p matrix X~'X~, column-major, symmetric
    design_summary: dict | None = None
    # (n,) offset the kernel fitted with, None if there is none; read from the
    # kernel like `weights`, so `glmm.anova(m0, m1)` can check that two fits
    # share it. With `y` and `weights` it is everything `glmm.anova(m0, m1)`
    # reads per row: the fit keeps no data column. Defaulted, like the fields
    # below, only because a dataclass field with a default comes last.
    offset: np.ndarray | None = None
    # What the fit maximised, and its quadrature node count: "least_squares"
    # (OLS), "maximum_likelihood" (GLM, or a Gaussian mixed model fitted by
    # ML), "reml", "laplace" (1 node) or "agq" (`objective_nodes` nodes); the
    # count is 0 on the first three. `glmm.anova(m0, m1)` labels its test with
    # it.
    objective: str | None = None
    objective_nodes: int = 0
    # The per-term tables `glmm.anova(m)` reads (Type III and Type II), built
    # when the fit was made. Each is a dict: "error" holds the message when the
    # fit gets no table (an interaction without its marginal terms), else None
    # beside the columns, one entry per term in formula order: "ss_type",
    # "label", "kind" ("F" or "chisq"), "statistic", "num_df", "den_df",
    # "den_df_kind" ("satterthwaite", "residual" or "" on a chi-squared row),
    # "p_value", "sum_sq", "mean_sq" (NaN on a chi-squared row), "row_notes"
    # (a list of sentences per row) and "notes" (the table's sentences).
    term_table_type3: dict | None = None
    term_table_type2: dict | None = None

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

    @property
    def aic(self):
        """Akaike information criterion, `-2 * loglik + 2 * df`, on every fit.

        On a REML fit (`reml` is True) `loglik` is the REML criterion, so this
        AIC compares only fits with the same fixed part; to compare fixed
        parts, fit both models with `reml=False`, or use
        `glmm.compare(data, f0, f1)`, which fits both by ML. `summary()` of a
        REML fit prints the REML criterion and no AIC.

        On a fit that did not converge the value is not an AIC: a GLMM that
        stopped at MaxFunReached has a finite `loglik` and `df` 0, so this is
        `-2 * loglik` (as R's `AIC()` gives); check `converged` first."""
        return -2.0 * self.loglik + 2.0 * self.df

    @property
    def bic(self):
        """Bayesian information criterion, `-2 * loglik + df * log(nobs)`, on
        every fit (NaN when `nobs` is 0).

        The REML caveat of `aic` applies: on a REML fit this BIC compares only
        fits with the same fixed part; fit with `reml=False`, or use
        `glmm.compare(data, f0, f1)`, to compare fixed parts. The
        non-converged caveat of `aic` applies too: at MaxFunReached on a GLMM
        `df` is 0 and this is `-2 * loglik`."""
        if self.nobs <= 0:
            return math.nan
        return -2.0 * self.loglik + self.df * math.log(self.nobs)

    def stddev_corr(self, group_idx):
        """Split grouping `group_idx`'s vech-packed covariance into
        (stddevs, correlation matrix) — mirrors Rust `Fit::stddev_corr`
        (src/fit/mod.rs): column-major lower-triangular vech."""
        vech = np.asarray(self.varcorr[group_idx], dtype=float)
        m = len(vech)
        q = (math.isqrt(1 + 8 * m) - 1) // 2
        if q * (q + 1) // 2 != m:
            raise ValueError(f"varcorr[{group_idx}] is not a valid vech (length {m})")

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
        """Print the lme4-shaped summary and return it as text. The blocks as
        data, and the HTML/LaTeX/Typst renders, are on `summary_object()`."""
        text = self.summary_object().text()
        print(text)
        return text

    def summary_object(self):
        """The summary as plain data with `.text()/.html()/.latex()/.typst()`
        renderers — see `glmm.summary.Summary`. Name is provisional until 1.0.0.

        Imported locally: `summary.py` never imports `glmm`, but `__init__`
        must finish defining `Fit` before `summary.py` is needed."""
        from glmm.summary import build_summary

        return build_summary(self)

    def confint(self, level=0.95):
        """Confidence intervals for the fixed effects as a `(p, 2)` array of
        lower and upper bounds, in the order of `names`. Each row follows
        `summary()`: the t quantile on the row's `coef_df` where the summary
        tests with t, the normal quantile where it tests with z (including a
        Gaussian mixed model fitted with `satterthwaite=False`, whose summary
        says "df not computed"). Aliased coefficients give NaN. The R port's
        `confint()` computes the same intervals through the same kernel
        function."""
        # A Fit saved before `coef_df` existed has no df: normal intervals,
        # as such a fit's summary tests with z.
        coef_df = getattr(self, "coef_df", None)
        if coef_df is None:
            coef_df = np.full(len(self.beta), np.nan)
        return np.asarray(
            _native.confint(
                [float(v) for v in self.beta],
                [float(v) for v in self.se],
                [float(v) for v in coef_df],
                float(level),
            ),
            dtype=float,
        ).reshape(len(self.beta), 2)

    def residuals(self, type="response"):
        """Response (`y - fitted`) or Pearson residuals. Deviance and working
        residuals are not offered: they need per-family deviance formulas
        nothing else in the package carries, and a wrong default would
        silently differ from lme4's `residuals(type=)`."""
        if type not in ("response", "pearson"):
            raise ValueError(f"type must be 'response' or 'pearson', got {type!r}")
        if len(self.fitted) != len(self.y) or len(self.fitted) == 0:
            raise ValueError(
                "residuals are unavailable: the fit did not converge, so no fitted "
                "values are available"
            )
        if type == "response":
            return np.asarray(self.y, dtype=float) - np.asarray(self.fitted, dtype=float)
        from glmm.summary import pearson_residuals

        return pearson_residuals(self)


# kind -> (tier, title), one fixed pair per kind. Mirrors the R port's
# .WARNING_KINDS (r/R/fastglmm.R) and documentation/warnings.md, which holds the
# message text - change all three together.
_WARNING_KINDS = {
    "search_limit": ("severe", "Search stopped at its step limit"),
    "fit_failed": ("severe", "Fit failed"),
    "glm_diverged": ("severe", "Fit diverged"),
    "design_unsolvable": ("severe", "Predictors could not be separated"),
    "constant_response": ("severe", "Response does not vary"),
    "too_few_rows": ("severe", "Too few rows"),
    "no_coefficients": ("severe", "Nothing to estimate"),
    "pirls_exhausted": ("caution", "Last fitting step did not finish"),
    "nb_shape_unsettled": ("caution", "Shape search did not settle"),
    "singular": ("caution", "Singular fit"),
    "ill_conditioned": ("caution", "Nearly collinear columns"),
    "hessian_se_fallback": ("caution", "Simpler standard errors used"),
    "se_unavailable": ("caution", "Standard errors not computed"),
    "exact_profile_fallback": ("caution", "Search retried with a different method"),
    "gradient_search_fallback": ("caution", "Search finished by another method"),
    "precision_limited": ("caution", "Numbers too large for full precision"),
    "agq_fallback": ("caution", "Adaptive quadrature not used"),
    "non_integer_response": ("caution", "Non-integer response"),
    "rows_dropped_na": ("caution", "Rows dropped for missing values"),
    "satterthwaite_curvature": ("caution", "Satterthwaite df uncertain"),
    "argument_ignored": ("note", "Argument ignored"),
    "unused_grouping_levels": ("note", "Unused grouping levels"),
    "single_level_grouping_dropped": ("note", "Random effect dropped (single level)"),
}
# A kernel note this wrapper has no entry for keeps its own kind string.
_UNKNOWN_KIND = ("caution", "Unrecognized solver message")


def _warn(store, kind, message, category):
    """Raise one warning as `<Tier>: <title>. <message>` and append it to `store`,
    the list that becomes `Fit.warnings`. The warning points at the line that
    called into glmm (`fit`, `compare`), however deep in the package it is
    raised: the stack level is the first frame outside this directory."""
    tier, title = _WARNING_KINDS.get(kind, _UNKNOWN_KIND)
    store.append({"tier": tier, "kind": kind, "title": title, "message": message})
    level, frame = 1, sys._getframe()
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        level, frame = level + 1, frame.f_back
    warnings.warn(f"{tier.capitalize()}: {title}. {message}", category=category, stacklevel=level)


def _pinned_detail(res):
    """Names of the RE components the optimizer pinned at the boundary, as
    `<term> in <group>`, for the singular warning.

    Read straight off `diagnostics["pinned"]`, which is the kernel's own record
    of what it pinned. Do NOT reconstruct it from `varcorr`: on a grouping with
    q >= 2 the pin fixes the diagonal of the Cholesky factor, while the reported
    stddev is sqrt(lambda_offdiag^2 + lambda_diag^2) and so lands at ~1e-9
    rather than at 0 — a scan for exactly-zero stddevs misses those pins.

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
                parts.append(f"{term} in {group}")
    return parts


def _note_warning(note, names, converged):
    """One kernel note as (message, warning category), or None to raise nothing.

    The `kind` string, not the English text, is the stable identifier — an
    unrecognized kind comes from a kernel newer than this wrapper (the Rust
    `Note` enum is `#[non_exhaustive]`) and still warns, under the base
    category. The R port maps the same kinds to condition classes
    (r/R/fastglmm.R) — change together."""
    kind = note["kind"]
    if kind == "separated":
        # Only on a non-converged fit, whose one warning (`_nonconvergence`) names it.
        return None
    if kind == "ill_conditioned":
        named = ", ".join(names[i] if i < len(names) else f"column {i}" for i in note["columns"])
        return (
            (
                f"{named} is almost a combination of other columns in the model, so its "
                "standard error is large. Its estimate is still correct, but imprecise. The "
                "other columns involved are not named. Consider dropping or combining "
                "predictors that carry the same information."
            ),
            IllConditionedWarning,
        )
    if kind == "unused_grouping_levels":
        # The kernel packs "<group>: <level>, <level>" (src/orchestrate.rs). Split on
        # the first ": " only: a level label may contain one.
        group, _, levels = note["detail"].partition(": ")
        return (
            (
                f"Grouping factor '{group}' has levels with no rows ({levels}). They stay in "
                "the model with random effects of exactly zero, but they are not counted in "
                "the number of groups. Remove unused categories before fitting."
            ),
            UnusedGroupingLevelsWarning,
        )
    if kind == "pirls_exhausted":
        # Raised only when the final re-evaluation of a converged fit hit the cap. A
        # rejected trial point changes no reported number and stays in
        # diagnostics["notes"]; on a non-converged fit the cap-out is one sentence of
        # the non-convergence warning instead (_nonconvergence).
        if note["final_eval"] and converged:
            return (
                (
                    "The final step that computes the reported results ran out of iterations. "
                    "The estimates and their standard errors may be less accurate than usual. "
                    "Try simplifying the random effects."
                ),
                PirlsExhaustedWarning,
            )
        return None
    if kind == "nb_shape_unsettled":
        # The kernel's Note::NbShapeUnsettled (src/fit/mod.rs); `evals` carries the rounds run.
        return (
            (
                "The search for the negative binomial shape parameter stopped at its limit of "
                f"{note['evals']} rounds before it settled. The coefficients and standard errors "
                "are computed at the last value it reached, which may not be the best one."
            ),
            NbShapeUnsettledWarning,
        )
    if kind == "single_level_grouping_dropped":
        return (
            (
                f"Grouping factor '{note['detail']}' has only one level, so no variance "
                "between groups can be estimated from it. Its random effect was dropped; "
                "the rest of the model was fitted without it."
            ),
            SingleLevelGroupingDroppedWarning,
        )
    if kind == "hessian_se_fallback":
        return (
            (
                "The usual standard errors could not be computed, so a simpler method was "
                "used. Its standard errors tend to be too small, so p-values and confidence "
                "intervals may look more precise than they are. Standard errors for the "
                "random-effect standard deviations are not available."
            ),
            HessianSeFallbackWarning,
        )
    if kind == "se_unavailable":
        return (
            (
                "Standard errors could not be computed at the fit's end point: the fixed "
                "effects carry too little separate information. The estimates are usable; "
                "standard errors, p-values and confidence intervals are not reported. Check "
                "for predictors that are copies or near-copies of each other."
            ),
            SeUnavailableWarning,
        )
    if kind == "gradient_search_fallback":
        return (
            (
                "The fast search for this model did not settle on an answer, so the fit "
                "finished with the standard search from the best point it had reached. The "
                "reported estimates come from that search."
            ),
            GradientSearchFallbackWarning,
        )
    if kind == "exact_profile_fallback":
        return (
            (
                "The default search method for this model did not settle on an answer, so "
                "the fit tried a different search method from the same starting point. The "
                "reported estimates come from whichever method reached the better answer."
            ),
            ExactProfileFallbackWarning,
        )
    if kind == "precision_limited":
        return (
            (
                "The numbers in this fit are so large that rounding in the computer's "
                "arithmetic hides small changes in its deviance. The random-effect estimates, "
                "and the fixed effects that depend on them, may be off by a tenth of a "
                "standard error or more. This happens with counts in the billions or more, "
                "or when a model misses a large effect in such data."
            ),
            PrecisionLimitedWarning,
        )
    if kind == "non_integer_response":
        # The kernel's Note::NonIntegerResponse (src/fit/mod.rs); `evals` carries the
        # affected row count (src/orchestrate.rs reuses the same NoteInfo slot as
        # nb_shape_unsettled's `rounds`).
        return (
            (
                f"The response is not a whole number in {_count(note['evals'], 'row')}. This "
                "family expects integer counts (binomial: the number of successes, "
                "y * weights). The fit still ran, but check whether this is the right family "
                "for the data."
            ),
            NonIntegerResponseWarning,
        )
    if kind == "satterthwaite_curvature":
        return _satterthwaite_message(note["detail"]), SatterthwaiteCurvatureWarning
    return (
        (
            f"The solver reported something ('{kind}') that this installed version does not "
            "recognize. Please report it at https://github.com/pawlenartowicz/glmm/issues."
        ),
        DiagnosticWarning,
    )


def _satterthwaite_message(detail):
    """The `satterthwaite_curvature` message from the kernel's detail string,
    "dropped <n>, negative <n>, converged <bool>, nonfinite <bool>"
    (src/orchestrate.rs). Mirrors the R port's .satterthwaite_message - change
    together."""
    fields = dict(part.split(" ", 1) for part in detail.split(", "))
    if fields["nonfinite"] == "true":
        return (
            "The Satterthwaite degrees of freedom could not be computed: the fit has no finite "
            "end point, or the deviance near it was not finite. Every coefficient is tested with "
            "z, and its confidence interval uses the normal distribution."
        )
    parts = []
    dropped, negative = int(fields["dropped"]), int(fields["negative"])
    if dropped:
        parts.append(
            "the curvature of the deviance in the variance parameters has "
            f"{_count(dropped, 'flat direction')}, left out"
        )
    if negative:
        parts.append(
            f"it has {_count(negative, 'negative direction')}, so the fit may not be at a minimum"
        )
    if fields["converged"] != "true":
        parts.append(
            "the variance search stopped at its step limit, so they are computed at its last point"
        )
    return (
        f"The Satterthwaite degrees of freedom are approximate: {'; '.join(parts)}. The t tests "
        "and confidence intervals use them as computed."
    )


_INNER_STEPS = "Some of its inner steps ran out of iterations."

# glm_diverged messages by family: separation only means something for a binomial
# response, and the Gamma and inverse-Gaussian fits skip the linear-predictor check
# (src/glm.rs), so theirs cannot be called a divergence to an extreme.
_DIVERGED_BINOMIAL = (
    "The fit did not converge. This usually means a predictor, or a combination of "
    "predictors, predicts the outcome perfectly (separation), so some fitted probabilities "
    "go to 0 or 1. The coefficients are from the last step; standard errors are not "
    "reported. Check the data for separation."
)
_DIVERGED_COUNTS = (
    "The fit did not converge. This usually means that some category of a predictor, or "
    "some combination of predictors, has only zero counts, so some fitted counts go to 0. "
    "The coefficients are from the last step; standard errors are not reported. Check for "
    "categories whose counts are all zero."
)
_DIVERGED_CONTINUOUS = (
    "The fit did not settle on an answer: the fitting steps stopped before converging. The "
    "coefficients are from the last step; standard errors are not reported. Check predictors "
    "with extreme values; with a link other than log, the log link is usually more stable."
)

# A response far outside the reciprocal links' range: the kernel holds the Gamma 1/mu and
# inverse-Gaussian 1/mu^2 linear predictor inside [1e-75, 1e75] (`INV_ETA_MIN` /
# `INV_ETA_MAX` in src/family.rs - change together), and a fit whose means must sit
# outside it cannot converge.
_ETA_RANGE = (
    "Some response values are so large or so small that the linear predictor of this link "
    "(1/mu for gamma, 1/mu^2 for inverse Gaussian) leaves the range the solver supports, "
    "1e-75 to 1e75. Rescale the response, for example divide it by a power of ten, or use "
    "the log link."
)


def _eta_out_of_range(family, link, y):
    """Whether some positive response value puts the reciprocal link's eta = 1/y (gamma)
    or 1/y^2 (inverse Gaussian) outside [1e-75, 1e75]."""
    power = {("gamma", "inverse"): 1, ("inversegaussian", "inverse_squared"): 2}.get((family, link))
    if power is None:
        return False
    y = y[y > 0]
    with np.errstate(over="ignore", divide="ignore"):
        eta = 1.0 / y**power
    return bool(np.any((eta < 1e-75) | (eta > 1e75)))


def _count(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _nonconvergence(family, mixed, notes, beta, aliased, deviance, y, link=None):
    """The one severe warning of a fit with `converged` False, as (kind, message,
    category). The kernel does not say which stopping rule fired, so the port reads the
    cause off what the fit reports, most specific first: no coefficient, too few rows
    (a one-row response is trivially constant, so this comes first), a constant
    response, then the model. A model with random effects whose fixed-effects-only GLM
    stopped on separation (the kernel's `separated` note) never ran its search; the
    coefficients are that GLM's last step, so it gets the GLM's message. Otherwise, with
    random effects, a finite deviance
    means the kernel reached an end point (the budget stop, reported at its best point),
    and finite estimated coefficients (aliased slots are NaN by contract) confirm it;
    anything else failed. The deviance is what decides a model with no fixed effects. A
    GLMM inner cap-out during the search is one extra sentence rather than a second
    warning. On the reciprocal links a response outside their range (`_eta_out_of_range`)
    names that cause instead of the general advice. Mirrors the R port's .nonconvergence -
    change together."""
    y = np.asarray(y, dtype=float)
    n_est = int(np.count_nonzero(~aliased))
    if not mixed and n_est == 0:
        return (
            "no_coefficients",
            (
                "The model has no coefficients and no random effects, so there is nothing to "
                "estimate. Add an intercept or a predictor."
            ),
            NoCoefficientsWarning,
        )
    if y.size <= n_est:
        return (
            "too_few_rows",
            (
                f"The model has {_count(n_est, 'coefficient')} to estimate but only "
                f"{_count(y.size, 'row')}, so no estimates were computed. Use more rows or "
                "fewer predictors."
            ),
            TooFewRowsWarning,
        )
    if y.size and np.all(y == y[0]):
        return (
            "constant_response",
            (
                f"Every value of the response is {y[0]:g}, so there is nothing to estimate. "
                "Check the response column."
            ),
            ConstantResponseWarning,
        )
    glm_stopped = any(n["kind"] == "separated" for n in notes)
    eta_range = _eta_out_of_range(family, link, y)
    if not mixed or glm_stopped:
        if family == "gaussian":
            return (
                "design_unsolvable",
                (
                    "The predictors could not be separated numerically, so no estimates were "
                    "computed. Check for predictors that are copies or near-copies of each other."
                ),
                DesignUnsolvableWarning,
            )
        if family == "binomial":
            message = _DIVERGED_BINOMIAL
        elif family in ("poisson", "negativebinomial"):
            message = _DIVERGED_COUNTS
        elif eta_range:
            message = (
                "The fit did not converge. The coefficients are from the last step; standard "
                "errors are not reported. " + _ETA_RANGE
            )
        else:
            message = _DIVERGED_CONTINUOUS
        return "glm_diverged", message, GlmDivergedWarning
    inner = [_INNER_STEPS] if any(n["kind"] == "pirls_exhausted" for n in notes) else []
    advice = "Try a simpler random-effects structure or check for nearly collinear predictors."
    if math.isfinite(deviance) and np.all(np.isfinite(beta[~aliased])):
        sentences = [
            "The search for the variance estimates reached its step limit before it settled.",
            *inner,
            (
                "The estimates shown are the best point found; they are often close, but this "
                "is not checked."
            ),
            "Do not use them until the fit converges.",
            advice,
        ]
        return "search_limit", " ".join(sentences), SearchLimitWarning
    sentences = [
        "The fitting algorithm failed and returned no estimates.",
        *inner,
        _ETA_RANGE if eta_range else advice,
    ]
    return "fit_failed", " ".join(sentences), FitFailedWarning


def _check_call(formula, family, link):
    """The checks `fit` and `compare` make on a formula, the family and the
    link before any data is read; returns the link, the family's default when
    `link` is None."""
    # Formula-shape checks the shared Rust parser has no dedicated message for
    # (it would fall through to a generic syntax error): caught here with the
    # same wording the R port uses (r/R/fastglmm.R's .check_formula) — change
    # together.
    if "||" in formula:
        raise ValueError(
            "(x || g) double-bar terms are not supported: the glmm kernel always "
            "fits the full RE correlation structure"
        )
    if re.search(r"(?<![\w.])\.(?![\w.])", formula):
        raise ValueError("'.' is not supported by the formula parser; list the columns explicitly")

    if family not in _FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {', '.join(_FAMILIES)}")
    fam = _FAMILIES[family]
    if link is None:
        link = fam["default_link"]
    elif link not in fam["links"]:
        raise ValueError(
            f"family {family!r} does not support link {link!r}; "
            f"expected one of {', '.join(fam['links'])}"
        )

    if family == "inversegaussian" and "|" in formula:
        raise ValueError(
            "family 'inversegaussian' is GLM-only: random-effect terms are not supported"
        )
    return link


def _check_nagq(nagq):
    if not (
        isinstance(nagq, int)
        and not isinstance(nagq, bool)
        and 1 <= nagq <= _MAX_NAGQ
        and nagq % 2 == 1
    ):
        raise ValueError(f"nagq must be an odd integer in 1..={_MAX_NAGQ}, got {nagq!r}")


def _prepare_columns(data, formulas, weights, offset, store, drop_missing_arguments=False):
    """The columns of `data` that `formulas` use, as `_native.fit` takes them,
    with `weights=`/`offset=` checked: `(numeric_columns, factor_columns,
    weights, offset)`. A row with a missing value in a column any of the
    formulas uses is dropped once, from every column and from the weights and
    the offset, so two formulas (`compare`) are fitted on the same rows. With
    `drop_missing_arguments` (`compare`), a row whose weight or offset is
    missing is dropped the same way; otherwise a missing entry is refused."""
    # Only the columns the formula actually uses are read from here on — see
    # `_formula_columns`.
    used_columns = _columns(data)
    used_names, missing, numeric_only = [], [], []
    for formula in formulas:
        for into, names in zip(
            (used_names, missing, numeric_only), _formula_columns(formula, used_columns.keys())
        ):
            into += [name for name in names if name not in into]
    where = "a column the formula uses"
    if drop_missing_arguments:
        where = "a column either formula uses, the weights or the offset"

    # A name the formula references but `data` does not have resolves to
    # nothing in `used_names` above (it is a filter, not a validator), so it
    # must be caught here — before the weights=/offset= checks below, which
    # are the same length regardless of a missing column and would otherwise
    # fail first with an unrelated message.
    if missing:
        raise ValueError(f"column(s) not found in data: {', '.join(missing)}")

    # The response and the offset() column, if any, always stay numeric —
    # everything else gets the bool check below.
    numeric_only = set(numeric_only)

    # Classify each used column bool vs numeric vs factor, and hand factors
    # across as (levels, codes) so the caller's reference level survives into
    # Rust. Bool is checked FIRST, ahead of the declared-categorical check: a
    # bool column crosses as "FALSE"/"TRUE" strings (`_bool_values`), which
    # fall into the plain-string branch below and so get lme4's own coding —
    # not as numeric 0/1, except in the response/offset positions
    # `numeric_only` names. A declared categorical is checked next: dtype
    # beats value-sniffing, or a categorical of non-strings
    # (pd.Categorical([1, 2, 3])) would land in the numeric branch and be fit
    # as a continuous predictor.
    #
    # A kept row is one where every used column has a value, mirroring R's
    # default na.action = na.omit (`r/R/fastglmm.R`) — there is no
    # `model.frame` here to do that filtering for free. A declared
    # categorical's missing rows (pandas code -1, pyarrow null) feed the same
    # `na_mask` as a plain column's missing values: `_levels_and_codes` never
    # raises on them, it just leaves the sentinel in `codes` for this loop to
    # find. `na_mask` stays None until a missing value is seen anywhere.
    parsed = {}
    na_mask = None
    for name in used_names:
        if name not in numeric_only and _is_bool_column(used_columns[name]):
            values = _bool_values(used_columns[name])
            parsed[name] = ("values", values)
            keep = [not _is_na(v) for v in values]
            na_mask = keep if na_mask is None else [a and b for a, b in zip(na_mask, keep)]
            continue
        declared = _levels_and_codes(used_columns[name])
        if declared is not None:
            levels, codes = declared
            parsed[name] = ("factor", levels, codes)
            keep = [c != -1 for c in codes]
            na_mask = keep if na_mask is None else [a and b for a, b in zip(na_mask, keep)]
            continue
        values = _values(used_columns[name])
        parsed[name] = ("values", values)
        keep = [not _is_na(v) for v in values]
        na_mask = keep if na_mask is None else [a and b for a, b in zip(na_mask, keep)]

    # A dict's columns can differ in length, and the `zip`s above stop at the
    # shortest one, so a ragged dict is refused here, before `na_mask` is used.
    # Same message as `run_fit`'s length check in `src/orchestrate.rs` — change
    # together.
    lengths = sorted((name, len(entry[-1])) for name, entry in parsed.items())
    if len({n for _, n in lengths}) > 1:
        detail = ", ".join(f'"{name}": {n}' for name, n in lengths)
        raise ValueError(
            f"columns have mismatched lengths ({detail}); all columns must have the same length"
        )

    # Checked against data's original rows, before the NA row drop below, as
    # R's fastglmm() checks them before na.action: a NaN weight is a bad
    # argument, not a missing value.
    n_rows = len(na_mask) if na_mask is not None else None
    if weights is not None:
        weights = _row_array(weights, "weights", n_rows, drop_missing_arguments)
        if np.any(weights <= 0):
            # The kernel requires strictly positive weights; a zero weight is a
            # row that should not be in the fit at all.
            raise ValueError("weights must be positive; drop zero-weight rows from data instead")
    if offset is not None:
        offset = _row_array(offset, "offset", n_rows, drop_missing_arguments)
    for arr in (weights, offset):
        if drop_missing_arguments and arr is not None and na_mask is not None:
            na_mask = [k and not math.isnan(v) for k, v in zip(na_mask, arr)]

    if na_mask is not None and not all(na_mask):
        n_kept = sum(na_mask)
        if n_kept == 0:
            raise ValueError(f"every row has a missing value in {where}; no rows left to fit")
        _warn(
            store,
            "rows_dropped_na",
            f"Dropped {len(na_mask) - n_kept} of {len(na_mask)} row(s): {where} had "
            "a missing value there.",
            RowsDroppedWarning,
        )
        for name, entry in parsed.items():
            if entry[0] == "factor":
                _, levels, codes = entry
                parsed[name] = ("factor", levels, [c for c, k in zip(codes, na_mask) if k])
            else:
                _, values = entry
                parsed[name] = ("values", [v for v, k in zip(values, na_mask) if k])
        # weights=/offset= are caller-supplied arrays aligned to data's original
        # rows, not columns of `data` itself, so the row drop above must repeat
        # here.
        if weights is not None:
            weights = weights[na_mask]
        if offset is not None:
            offset = offset[na_mask]

    numeric_columns = {}
    factor_columns = {}
    for name, entry in parsed.items():
        if entry[0] == "factor":
            factor_columns[name] = (entry[1], entry[2])
            continue
        values = entry[1]
        if values and isinstance(values[0], str):
            factor_columns[name] = _sorted_levels_and_codes([str(v) for v in values])
        else:
            numeric_columns[name] = [float(v) for v in values]
    return numeric_columns, factor_columns, weights, offset


def _term_result(t):
    """A stored per-term result (`_native.fit`'s dict) with its numeric columns
    as float arrays; see `Fit.term_table_type3`."""
    if t["error"] is not None:
        return {"error": t["error"]}
    numeric = ("statistic", "num_df", "den_df", "p_value", "sum_sq", "mean_sq")
    return {k: np.asarray(v, dtype=float) if k in numeric else v for k, v in t.items()}


def _fit_from_native(r, formula, family, link, nagq, dispersion, store):
    """The `Fit` of `_native.fit`'s dict `r`, with the fit's warnings raised
    and appended to `store`; the other arguments are the call's own."""
    # nagq's shape eligibility (single grouping factor, binomial/Poisson/NB/Gamma,
    # q <= 3) is only decidable after the Rust-side formula lowering, so the
    # warn-and-strip for it lives in the shared glmm::orchestrate module
    # (src/orchestrate.rs).
    # Whether the FITTED model actually has random effects — not `mixed`,
    # which only says the formula asked for some: a single-level grouping is
    # dropped after lowering (`single_level_grouping_dropped`), and once every
    # random-effect term is dropped this way, `r["re_groups"]` comes back
    # empty even though `mixed` (read off the formula text before fitting)
    # still says True.
    fitted_mixed = bool(r["re_groups"])
    if r["agq_warning"] is not None:
        # Built here, not taken from `agq_warning`: that string spells the R port's
        # argument differently. A Gaussian model is fitted exactly and a model without
        # random effects has no integral, so there nagq changes nothing and is an
        # ignored argument, not a fallback worth a caution.
        if family == "gaussian" or not fitted_mixed:
            reason = (
                "a Gaussian model" if family == "gaussian" else "a model without random effects"
            )
            _warn(
                store,
                "argument_ignored",
                f"nagq={nagq} has no effect for {reason}, because nothing is approximated.",
                ArgumentIgnoredWarning,
            )
        else:
            _warn(
                store,
                "agq_fallback",
                f"nagq={nagq} works only for {_AGQ_FAMILIES} models whose random effects "
                "are in one grouping factor, with at most 3 random effects in it. This "
                "model was fitted without adaptive quadrature.",
                AgqFallbackWarning,
            )
    if r["reml_warning"] is not None:
        # Worded here, as `agq_warning` above: the shared rule in
        # src/orchestrate.rs decides, each port spells its own argument. The R
        # port (r/R/fastglmm.R, `fastglmm`) words the same sentence with
        # `REML = TRUE`; change together.
        _warn(
            store,
            "argument_ignored",
            "reml=True applies only to Gaussian models; it was ignored and the model was "
            "fitted by maximum likelihood.",
            ArgumentIgnoredWarning,
        )
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
        n_grad=r["n_grad"],
        outer_stop=r["outer_stop"],
        outer_decrement=r["outer_decrement"],
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
        formula=formula,
        family=family,
        link=link,
        nagq=nagq if r["agq_warning"] is None else 1,
        nobs=int(r["nobs"]),
        y=np.asarray(r["y"], dtype=float),
        weights=np.asarray(r["weights"], dtype=float) if r["weights"] is not None else None,
        coef_df=np.asarray(r["coef_df"], dtype=float),
        coef_p=np.asarray(r["coef_p"], dtype=float),
        df_method=r["df_method"],
        rss=r["rss"],
        r_squared=r["r_squared"],
        adj_r_squared=r["adj_r_squared"],
        warnings=store,
        dispersion_held=dispersion,
        design_summary=None
        if r["design_summary"] is None
        else {k: np.asarray(v, dtype=float) for k, v in r["design_summary"].items()},
        offset=np.asarray(r["offset"], dtype=float) if r["offset"] is not None else None,
        objective=r["objective"],
        objective_nodes=int(r["objective_nodes"]),
        term_table_type3=_term_result(r["term_table_type3"]),
        term_table_type2=_term_result(r["term_table_type2"]),
    )
    # Singularity is not assessed on a fit that did not converge: the kernel never sets
    # `singular` there (the post-hoc check and the boundary flags are gated on
    # `converged`), so `m.singular` reads False on a non-converged fit. `and res.converged`
    # below is a defensive guard, not load-bearing on the current kernel.
    if res.singular and res.converged:
        affected = _pinned_detail(res)
        _warn(
            store,
            "singular",
            "The random effects are too complex for the data: a variance is estimated at "
            "or near zero, or a correlation at or near −1 or 1. Consider removing the "
            "affected random effect." + (f" Affected: {', '.join(affected)}." if affected else ""),
            SingularFitWarning,
        )
    for note in res.diagnostics["notes"]:
        out = _note_warning(note, res.names, res.converged)
        if out is not None:
            _warn(store, note["kind"], *out)
    if not res.converged:
        _warn(
            store,
            *_nonconvergence(
                family,
                fitted_mixed,
                res.diagnostics["notes"],
                res.beta,
                res.aliased,
                res.deviance,
                res.y,
                link,
            ),
        )
    return res


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
    satterthwaite=True,
    reml=None,
):
    """Fit `formula` against `data`'s columns and return a `Fit`.

    data: dict[str, array-like]; DataFrame / Arrow Table also accepted (duck-typed).
    formula: R-style string, e.g. "y ~ x + z + (1 + x | g)".
    family: gaussian | binomial | poisson | gamma | negativebinomial | inversegaussian.
    nagq: adaptive Gauss-Hermite quadrature nodes per random-effect dimension
        (odd, 1..=25; default 1 = Laplace). k>1 applies to binomial, Poisson,
        negative-binomial and Gamma models with a single grouping factor and
        q <= 3 random effects per
        group (temporary cap); any other shape warns and falls back to Laplace.
    init_theta: negative-binomial shape seed, named for `MASS::glm.nb(init.theta=)`
        — the same knob, and the name the R port exposes: the starting value of
        the shape search (the same slot as `warm_start["dispersion"]`; give one
        of the two). None cold-starts it from the no-random-effect
        negative-binomial fit. Distinct from `warm_start["theta"]`, which is the
        random-effect Cholesky vector (lme4's `start=list(theta=)`); they are
        unrelated parameters and both may be passed in one call.
    offset: per-row additive offset on the linear-predictor scale, length n
        (R's `offset=`): eta = offset + X*beta (+ Z*b). A fixed known
        contribution, not a parameter — the canonical use is a Poisson
        exposure, offset = log(exposure). None = no offset.
    warm_start: {"beta": …, "theta": …, "dispersion": …} optimizer start; any key
        may be left out to cold-start that part. "dispersion" is the estimated
        dispersion as a previous fit reports it (`Fit.dispersion`): the shape
        for negative binomial, phi for Gamma. See `init_theta` above for why
        "theta" here is NOT the negative-binomial shape.
    satterthwaite: compute lmerTest's Satterthwaite degrees of freedom for a
        Gaussian mixed model's t tests (default True). False skips that pass:
        the summary then tests with z and says "df not computed", and
        `Fit.df_method` is "disabled". OLS and the Gamma and inverse-Gaussian
        GLMs keep their residual df whatever it says.
    reml: None, True or False; `Fit.reml` reports the value used.
        Gaussian with random effects (after any single-level grouping is
        dropped): None and True fit by REML, False by ML.
        The summary of an ML fit is titled "Linear mixed model fit by maximum
        likelihood", prints AIC, BIC, logLik, deviance and df.resid, and
        carries the note "p-values can be somewhat anticonservative" under
        the title; a REML fit prints the REML criterion and no AIC.
        Gaussian without (OLS, also when every grouping was dropped): same
        estimates; True reports the REML log-likelihood (the summary prints
        the REML criterion), None and False the ML one.
        Any other family: ML; True is ignored with an ArgumentIgnoredWarning.

    A categorical column's level order is honored: level 0 is the
    treatment-contrast base, so `pd.Categorical(x, categories=[…])` fits against
    the first category you list. A plain string column has no declared order and
    is sorted lexicographically (R's `factor()` default). A bool column (Python
    `bool`, `numpy.bool_`, pandas `bool`/nullable `boolean`, pyarrow `bool_`) is
    treated the same way, with levels `FALSE`/`TRUE` — so a bool `x` gives a
    coefficient `xTRUE`, as in lme4 — except as the response or the `offset()`
    column, which stay 0/1 numbers.

    See the API spec for the remaining knobs.
    """
    link = _check_call(formula, family, link)
    # `|` marks a random-effect term, so its presence is the mixed/GLM split
    # — decidable without the (Rust-side) formula parser.
    mixed = "|" in formula

    if wald_se not in ("hessian", "rx"):
        raise ValueError(f"wald_se must be 'hessian' or 'rx', got {wald_se!r}")

    _check_nagq(nagq)

    if not isinstance(satterthwaite, (bool, np.bool_)):
        raise TypeError(f"satterthwaite must be True or False, got {satterthwaite!r}")
    satterthwaite = bool(satterthwaite)
    if reml is not None and not isinstance(reml, (bool, np.bool_)):
        raise TypeError(f"reml must be None, True or False, got {reml!r}")
    reml = None if reml is None else bool(reml)

    store = []

    # Valid-but-inapplicable options: warn and strip. The kernel
    # boundary-faults on inapplicable options and a Rust panic across the FFI
    # is not an acceptable user error, so nothing inapplicable may reach it.
    if dispersion is not None and family not in _DISPERSION_FAMILIES:
        _warn(
            store,
            "argument_ignored",
            f"dispersion= has no effect for family '{family}'.",
            ArgumentIgnoredWarning,
        )
        dispersion = None
    if dispersion is not None:
        if not (
            dispersion == "estimate"
            or (isinstance(dispersion, (int, float)) and not isinstance(dispersion, bool))
        ):
            raise ValueError(
                f"dispersion must be None, 'estimate', or a number, got {dispersion!r}"
            )
        if family in ("binomial", "poisson") and mixed:
            _warn(
                store,
                "argument_ignored",
                "Quasi-likelihood dispersion= is not supported yet for binomial or Poisson "
                "models. The default dispersion of 1 was used.",
                ArgumentIgnoredWarning,
            )
            dispersion = None
    if init_theta is not None and family != "negativebinomial":
        _warn(
            store,
            "argument_ignored",
            f"init_theta= is not used for family '{family}'.",
            ArgumentIgnoredWarning,
        )
        init_theta = None
    if warm_start is not None:
        if not isinstance(warm_start, dict):
            raise TypeError(
                "warm_start must be a dict with keys 'beta', 'theta' and/or 'dispersion' (theta "
                "is the random-effect Cholesky vector, not the negative-binomial shape - that is "
                f"init_theta or dispersion), got {type(warm_start).__name__}"
            )
        allowed = ("beta", "theta", "dispersion")
        unknown = [k for k in warm_start if k not in allowed]
        if unknown:
            _warn(
                store,
                "argument_ignored",
                "warm_start accepts only 'beta', 'theta' and 'dispersion'; these entries were "
                "ignored: " + ", ".join(map(str, unknown)) + ".",
                ArgumentIgnoredWarning,
            )
            warm_start = {k: v for k, v in warm_start.items() if k in allowed}

    if dispersion == "estimate" and family in ("gamma", "inversegaussian"):
        # The phi families' default (dispersion=None) already estimates phi
        # (Pearson on a Gamma GLM, maximum likelihood on a Gamma GLMM, Pearson
        # on inversegaussian), so "estimate" needs no distinct kernel state.
        dispersion = None
    if family in ("binomial", "poisson") and dispersion is not None:
        raise NotImplementedError(
            f"quasi-likelihood dispersion on family {family!r} is not yet implemented in the kernel"
        )
    # init_theta is the negative-binomial shape seed: the same kernel slot as
    # warm_start["dispersion"], so giving both is ambiguous.
    if init_theta is not None:
        if warm_start is not None and warm_start.get("dispersion") is not None:
            raise ValueError(
                "give the negative-binomial shape seed once: init_theta= or "
                'warm_start["dispersion"], not both'
            )
        warm_start = dict(warm_start or {}, dispersion=init_theta)
    if warm_start is not None and warm_start.get("dispersion") is not None:
        d = warm_start["dispersion"]
        if (
            isinstance(d, bool)
            or not isinstance(d, (int, float))
            or not (math.isfinite(d) and d > 0)
        ):
            raise ValueError(
                f'warm_start["dispersion"] (or init_theta) must be one finite number > 0, got {d!r}'
            )

    numeric_columns, factor_columns, weights, offset = _prepare_columns(
        data, [formula], weights, offset, store
    )

    warm_start_pair = None
    if warm_start is not None:
        d = warm_start.get("dispersion")
        warm_start_pair = (
            [float(v) for v in warm_start.get("beta", [])],
            [float(v) for v in warm_start.get("theta", [])],
            None if d is None else float(d),
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
        satterthwaite,
        reml,
    )
    return _fit_from_native(r, formula, family, link, nagq, dispersion, store)


# Last: `glmm/anova.py` reads `Fit` and the helpers above from this module.
from glmm.anova import Comparison, ComparisonTable, TermTable, anova, compare
