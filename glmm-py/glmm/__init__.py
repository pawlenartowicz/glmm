"""GLMM Python port — formula + data -> fit.

`glmm.fit` parses `formula` against `data`'s columns through the Rust
`glmm::formula` module, fits via the `glmm` kernel (through the `glmm._native`
PyO3 extension), and returns a `Fit`. One option is not yet implemented
in the kernel and raises a clean `NotImplementedError`: quasi-likelihood
`dispersion=` on binomial/poisson.

Public surface is `fit`, `Fit`, `model_matrix`, `term_labels`, the tests of `glmm.anova` (`anova`,
`compare`, `TermTable`, `ComparisonTable`, `Comparison`, in `glmm/anova.py`), `pairwise` and
`PairwiseTable`, `Fit.profile_confint`'s `ProfileTable`,
and the warning categories the diagnostics channel raises (`DiagnosticWarning`, `AgqFallbackWarning`,
`ArgumentIgnoredWarning`, `ConstantResponseWarning`, `DesignUnsolvableWarning`,
`ExactProfileFallbackWarning`, `FewRowsForRandomEffectsWarning`,
`FitFailedWarning`, `GradientSearchFallbackWarning`, `GlmDivergedWarning`, `HessianSeFallbackWarning`,
`IllConditionedWarning`, `ModelBasedJointTestWarning`, `NbShapeUnsettledWarning`,
`NewGroupingLevelsWarning`, `NoCoefficientsWarning`, `NonIntegerResponseWarning`,
`PirlsExhaustedWarning`, `PrecisionLimitedWarning`, `PredictSeModelBasedWarning`,
`RobustDfSmallWarning`, `RobustSeUnidentifiedWarning`, `RobustWithRandomSlopeWarning`,
`RowsDroppedWarning`, `SatterthwaiteCurvatureWarning`,
`SearchLimitWarning`, `SeUnavailableWarning`, `SingleLevelGroupingDroppedWarning`,
`SingularFitWarning`,
`TooFewRowsWarning`, `VarianceAtLimitWarning`, `WeightPowerAtBoundWarning`,
`WeightPowerImplausibleWarning`, `WeightPowerOutsideUnitWarning`,
`WeightPowerSearchFailuresWarning`).
"""

import math
import os
import random
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
    "FewRowsForRandomEffectsWarning",
    "Fit",
    "FitFailedWarning",
    "GlmDivergedWarning",
    "GradientSearchFallbackWarning",
    "HessianSeFallbackWarning",
    "IllConditionedWarning",
    "ModelBasedJointTestWarning",
    "NbShapeUnsettledWarning",
    "NewGroupingLevelsWarning",
    "NoCoefficientsWarning",
    "NonIntegerResponseWarning",
    "PairwiseTable",
    "PirlsExhaustedWarning",
    "PrecisionLimitedWarning",
    "PredictSeModelBasedWarning",
    "ProfileTable",
    "RobustDfSmallWarning",
    "RobustSeUnidentifiedWarning",
    "RobustWithRandomSlopeWarning",
    "RowsDroppedWarning",
    "SatterthwaiteCurvatureWarning",
    "SeUnavailableWarning",
    "SearchLimitWarning",
    "SingleLevelGroupingDroppedWarning",
    "SingularFitWarning",
    "TermTable",
    "TooFewRowsWarning",
    "VarianceAtLimitWarning",
    "WeightPowerAtBoundWarning",
    "WeightPowerImplausibleWarning",
    "WeightPowerOutsideUnitWarning",
    "WeightPowerSearchFailuresWarning",
    "anova",
    "compare",
    "fit",
    "model_matrix",
    "pairwise",
    "term_labels",
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


class NewGroupingLevelsWarning(DiagnosticWarning):
    """`Fit.predict` met grouping levels the fit did not see.

    Those rows get a random effect of zero for that grouping (the population
    value), while their other groupings keep theirs. Raised by `predict`, not
    by `fit`, so it is not in `Fit.warnings`. `allow_new_levels=False` makes
    such a level an error instead.
    """


class SingleLevelGroupingDroppedWarning(DiagnosticWarning):
    """A random-effect term's grouping factor has only one observed level, so
    no between-group variance is identifiable from it.

    lme4 errors here ("grouping factors must have > 1 sampled level"); this
    port drops the term instead and fits the rest of the model. When every
    random-effect term is dropped this way, the model fits as though the
    formula had declared none.
    """


class FewRowsForRandomEffectsWarning(DiagnosticWarning):
    """A random-effect term has at least as many random effects as the fit has
    rows (more than the rows, on a family other than Gaussian), so its variance
    parameters probably cannot be estimated from these data.

    A term's random effects are its observed levels times its columns (the
    intercept, one per numeric slope, one per non-base level of a factor
    slope): `(1 + x | g)` on 10 levels has 20. Each term is checked on its own
    against all rows. lme4 stops on such a model by default
    (`check.nobs.vs.nRE`); this port fits it as usual and warns.
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
    log-determinant of the random-effect system; on a Gaussian mixed model,
    when the residuals are tiny against the response (a response far from
    zero next to its residual spread, or weights spanning many orders of
    magnitude).
    """


class VarianceAtLimitWarning(DiagnosticWarning):
    """A random-effect parameter ended on the upper limit of the variance search
    (1000 on glmm's internal scale), so the estimates are at that limit, not at
    an optimum.

    On a Gaussian model this usually means the residual variance is estimated
    at or near zero; on other families the random-effect variance has no
    finite estimate. `converged` and `singular` are the search's own, and
    `diagnostics["boundary"]` reads "upper_limit" when it would otherwise read
    "interior".
    """


class WeightPowerAtBoundWarning(DiagnosticWarning):
    """The fitted weight power (`weight_power="estimate"`, or `None` resolved
    to it) is not a maximum: the log-likelihood at the end of the search range
    nearer to it is at least its own, so the profile rises, or stays flat,
    toward that end.

    With at most as many rows as coefficients off the common weight the
    profile goes flat and can have no finite maximum, and on weights that vary
    little the maximum can lie past the range.
    """


class WeightPowerSearchFailuresWarning(DiagnosticWarning):
    """Fits inside the weight-power search failed or did not converge.

    A failed one counts as the worst value at its power, which steers the
    search toward 0 (the weights ignored); the reported fit is the one at the
    power found, with its own convergence.
    """


class WeightPowerOutsideUnitWarning(DiagnosticWarning):
    """The fitted weight power is outside [0, 1] (0: the weights carry no
    variance information; 1: the weights as claimed).

    With few rows or a weak fit (low R²) this happens by chance and is not by
    itself a problem. Not raised alongside `WeightPowerImplausibleWarning` or
    `WeightPowerAtBoundWarning`.
    """


class WeightPowerImplausibleWarning(DiagnosticWarning):
    """The fitted weight power is outside [-1, 2], the range the simulations
    behind it covered, and the fitted variances differ from the claimed
    weights by more than 10 times across the data (s^|power - 1| > 10, s the
    largest weight over the smallest).
    """


class RobustWithRandomSlopeWarning(DiagnosticWarning):
    """Robust standard errors (`se_type="robust"`) were asked for on a model
    with a random slope. Their arithmetic is checked there, but in simulations
    with 10 groups their Type I error ran at .056 to .066; the model-based
    standard errors are the default on such a model.
    """


class RobustSeUnidentifiedWarning(DiagnosticWarning):
    """One group alone carries the information on a coefficient, so its robust
    standard error and df are NaN: that group has no residual left to measure
    it with. More groups are needed; `se_type="model"` assumes the groups
    share one variance.
    """


class RobustDfSmallWarning(DiagnosticWarning):
    """Robust degrees of freedom below 4, where the robust test gives no
    assurance on its error rate (Pustejovsky and Tipton 2018). If such a
    predictor is constant within groups, `se_type="model"` is the better route.
    Raised by `fit` per coefficient, and by `Fit.contrast_test` and `pairwise`
    per test (where `RobustSeUnidentifiedWarning` also names a test one group
    alone identifies).
    """


class ModelBasedJointTestWarning(DiagnosticWarning):
    """`Fit.contrast_test` of more than one row on a robust fit: there is no
    evidence for a robust joint test, so the F uses the model-based standard
    errors. Not in `Fit.warnings`.
    """


class PredictSeModelBasedWarning(DiagnosticWarning):
    """`Fit.predict` with `se_fit=True` on a robust fit: the standard error and
    interval are model-based, since their formula holds only with the model's
    covariance. Not in `Fit.warnings`.
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
    """A fit without random effects could not factor X'WX (columns redundant up to
    rounding, on a GLM possibly only under the weights). No estimates are returned."""


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
# (glmm-r/R/fastglmm.R) — change together.
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

_MAX_NAGQ = 25  # mirrors GLMM/glmm-rs/src/consts.rs::MAX_NAGQ — change together

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
    `Categorical`/`Series[category]` (via `.cat`), polars `Enum`, and pyarrow
    dictionary columns, bare `DictionaryArray` or the `ChunkedArray` a Table's
    `.column()` returns. A plain string column has no declared order and is handled by the
    caller."""
    cat = getattr(col, "cat", col)  # pandas Series[category] -> .cat accessor
    if hasattr(cat, "categories") and hasattr(cat, "codes"):
        levels = [str(v) for v in cat.categories]
        # pandas marks a missing value as code -1; there is no level to fit it
        # against, so the caller drops that row the same way it drops a plain
        # column's missing value, rather than raising here.
        codes = np.asarray(cat.codes, dtype=np.int64)
        return levels, codes
    # A polars Enum declares its levels on the dtype; `to_physical()` gives
    # each row's index into them, None for a missing value. A polars
    # Categorical has no declared order (its dtype's `categories` is a string
    # cache, not a level list), so it takes the plain-string path below.
    dtype = getattr(col, "dtype", None)
    if str(dtype).startswith("Enum") and hasattr(col, "to_physical"):
        codes = [-1 if c is None else c for c in col.to_physical().to_list()]
        return [str(v) for v in dtype.categories], codes
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


# Element types `_plain_column` converts in one pass: what the per-row path
# below would pass to `float()` (None as missing), and plain strings. A list
# of these types only (matched exactly, so no subclass's own code runs) goes
# to the native `plain_column`.
_NUMBER_TYPES = {float, int, bool, type(None), np.float64}
_STRING_TYPES = {str, np.str_}
_PLAIN_TYPES = _NUMBER_TYPES | _STRING_TYPES
# Below this many elements numpy's fixed per-call cost exceeds the per-row
# loop's, so a short column without a numeric dtype keeps the per-row path.
_PLAIN_MIN_ROWS = 64


def _plain_column(col):
    """`col` vectorised, when it is all numbers or all strings, reading each
    value exactly as `_prepare_columns`' per-row path (`_values`, `_is_na`,
    `float()`, `str()`) would:

    - `("num", values, keep)`: a numeric dtype, or elements that are all
      `_NUMBER_TYPES`; `values` a float64 array with NaN where missing.
    - `("str", codes, keys, keep)`: non-missing elements all strings; `keys`
      the distinct values in first-seen order, `codes` each row's index into
      them. `_is_na` runs once per distinct value, not once per row.

    `keep` is the bool mask of present rows. None for anything else (mixed
    types, dates, pyarrow, which `_values` unwraps) and for a column of fewer
    than `_PLAIN_MIN_ROWS` elements without a numeric dtype: the caller keeps
    the per-row path for it, so its result never depends on which path ran.

    A list of `_PLAIN_TYPES` is read by the native `plain_column` in one pass
    in Rust; when it answers None, the numpy and dict code below reads the
    list instead."""
    dtype = getattr(col, "dtype", None)
    if getattr(dtype, "kind", None) in ("i", "u", "f"):
        try:
            if hasattr(col, "to_numpy"):  # pandas, incl. nullable Int64/Float64
                values = col.to_numpy(dtype=float, na_value=np.nan)
            else:
                values = np.asarray(col, dtype=float)
        except (TypeError, ValueError):
            return None
        return ("num", values, ~np.isnan(values))
    if hasattr(col, "to_pylist") or hasattr(col, "dictionary"):
        return None
    if isinstance(col, list):
        elements = col
    elif dtype is not None and hasattr(col, "tolist"):
        elements = col.tolist()
    elif dtype is not None or isinstance(col, tuple):
        elements = list(col)
    else:
        return None
    if len(elements) < _PLAIN_MIN_ROWS:
        return None
    native = None
    if set(map(type, elements)) <= _PLAIN_TYPES:
        native = _native.call("plain_column", {"elements": elements})
    if native is not None and native["kind"] == "num":
        values = np.frombuffer(native["values"], dtype=np.float64).copy()
        return ("num", values, ~np.isnan(values))
    if native is not None:
        codes, keys = np.frombuffer(native["codes"], dtype=np.int64), native["keys"]
    elif set(map(type, elements)) <= _NUMBER_TYPES:
        values = np.array(elements, dtype=float)
        return ("num", values, ~np.isnan(values))
    else:
        index = {}
        codes = np.fromiter(
            (index.setdefault(v, len(index)) for v in elements), dtype=np.intp, count=len(elements)
        )
        keys = list(index)
    missing = np.array([_is_na(k) for k in keys], dtype=bool)
    if not all(type(k) in _STRING_TYPES for k, m in zip(keys, missing) if not m):
        return None
    return ("str", codes, keys, ~missing[codes])


def _is_bool_column(col):
    """True for a bool column: `numpy.bool_`, pandas `bool` or the nullable
    `boolean` extension dtype, polars `Boolean`, pyarrow `bool_`, or a plain
    list/tuple of Python `bool` (an `object`-dtype array unwraps to the same
    check).

    Checked on the column object, the way `_levels_and_codes` inspects it
    before `_values` flattens it into scalars that no longer carry a dtype."""
    dtype = getattr(col, "dtype", None)
    if dtype is not None:
        if str(dtype) in ("bool", "boolean", "Boolean"):
            return True
        if str(dtype) != "object":
            return False
        col = list(col)  # object-dtype array — fall through to the element check
    arrow_type = getattr(col, "type", None)  # pyarrow Array/ChunkedArray
    if arrow_type is not None:
        return str(arrow_type) == "bool"
    if isinstance(col, (list, tuple)):
        seen = False
        for v in col:
            if v is None:
                continue
            if not isinstance(v, (bool, np.bool_)):
                return False
            seen = True
        return seen
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

    Mirrors Rust's `Column::factor_from_labels` — the R `factor()` default in
    the C locale (by code point; R in another locale can order mixed-case labels
    differently), and all that can be inferred when the caller declared no order. Doing the sort
    here rather than in the parser is what makes it a default the caller can
    override, instead of one imposed on every factor."""
    levels = sorted(set(labels))
    index = {lvl: i for i, lvl in enumerate(levels)}
    return levels, [index[v] for v in labels]


def _formula_columns(formula, names):
    """`(used, missing, numeric_only)` for `formula` against the column names
    `names`, from one parse in the Rust formula frontend
    (the native `formula_columns`).

    - `used`: `names` restricted to the ones the formula reads as data,
      mirroring R's own `data[all.vars(formula)]` (`glmm-r/R/fastglmm.R`) so a
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

    A bad formula raises here with the same message the native `fit` would give
    for it: letting the error through here, rather than falling back to every
    name in `names`, keeps a parse error a parse error instead of routing an
    unrelated column (a datetime, say) into the numeric/factor branches below,
    where it fails to convert and raises a `TypeError` that has nothing to do
    with the formula."""
    args = {"formula": formula, "available": list(names)}
    return _columns_triple(_native.call("formula_columns", args))


def _columns_triple(r):
    """The native `formula_columns` / `model_matrix_columns` record as
    `(used, missing, numeric_only)`."""
    return r["used"], r["missing"], r["numeric_only"]


def _row_array(values, name, n, missing_ok=False):
    """`weights=`/`offset=` as a float array of one finite entry per data row.

    Mirrors the R port's checks in `fastglmm()` (`glmm-r/R/fastglmm.R`): a wrong
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
    #   boundary   "interior" | "at_boundary" | "no_optimum" | "upper_limit":
    #                     where the accepted theta sits ("upper_limit": on the
    #                     upper limit of the search, otherwise interior).
    #                     Every theta-carrying route distinguishes all four
    #                     (LMM over either
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
    #   notes      list of {"kind": str, "tier": str, "title": str,
    #                     "message": str | None, "columns": [int], "pivot": float,
    #                     "evals": int, "final_eval": bool, "detail": str,
    #                     "delta": float, "factor": float, "lower": float,
    #                     "upper": float,
    #                     "failed": int, "not_converged": int}
    #                     — observations with no dedicated field, each with a
    #                     message raised as a `DiagnosticWarning` subclass by
    #                     `fit`. Mostly
    #                     the solver's; the formula lowering contributes
    #                     "single_level_grouping_dropped" and
    #                     "few_rows_for_random_effects" (names ride in `detail`).
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
    # Per grouping, in re_groups order, the term of each term name: (x || g)
    # gives [0, 1]. summary() prints a correlation only within one term.
    re_term_index: list
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
    #   postvar ndarray    -- (len(levels), len(terms), len(terms)): each
    #                         level's conditional covariance of its random
    #                         effects given the data, lme4's
    #                         `ranef(condVar = TRUE)` "postVar"; None on a fit
    #                         made with condvar=False and on one whose layout
    #                         forms none (an extra grouping with a random
    #                         slope, the sparse route, the packed GLMM)
    #   condsd  ndarray    -- (len(levels), len(terms)): the conditional
    #                         standard deviations, the square roots of the
    #                         diagonal of `postvar`; None where it is
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
    # (n,) the caller's prior weights, None if unweighted: the fit used them
    # raised to `weight_power`. From the kernel, not the `weights=` argument:
    # a `cbind(s, f)` response lowers to trial-count weights the caller never
    # passed.
    weights: np.ndarray | None
    # Per coefficient, aligned with `names`: the degrees of freedom of its t test
    # (the Satterthwaite df on a Gaussian linear mixed model, n - p on OLS and on
    # a Gamma or inverse-Gaussian GLM with phi estimated, the robust df on a
    # robust fit), NaN where the test is
    # z and on an OLS or dispersion-GLM fit that did not converge; and that t
    # test's two-sided p-value, computed by the kernel, NaN where there is no df.
    coef_df: np.ndarray
    coef_p: np.ndarray
    # Where `coef_df` comes from: "satterthwaite", "residual" (OLS and the
    # dispersion GLMs, whatever `satterthwaite=` says), "none" (the families
    # with no df: z), "disabled" (a Gaussian mixed model fitted with
    # `satterthwaite=False`: z), "failed" (its Satterthwaite pass failed: z),
    # and on a robust fit "pustejovsky_tipton" (CR2, with a grouping factor) or
    # "bell_mccaffrey" (HC2, without one).
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
    # K = the number of groups and p retained
    # fixed-effect columns, on the fit's orthogonalized design X~ (it spans the
    # column space of the fixed-effect design):
    #   counts  (K,)    rows per level
    #   s       (p*K,)  the p x K matrix X~'Z, column-major (level a at [a*p:(a+1)*p])
    #   c       (p*p,)  the p x p matrix X~'X~, column-major, symmetric
    design_summary: dict | None = None
    # (n,) offset the kernel fitted with, None if there is none; read from the
    # kernel like `weights`, so `glmm.anova(m0, m1)` can check that two fits
    # share it. With `y` and `weights` it is everything `glmm.anova(m0, m1)`
    # reads per row. Defaulted, like the fields below, only because a
    # dataclass field with a default comes last.
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
    # beside "ss_type", the table's "test" ("F", "chisq" or "lrt") and its
    # "reason" ("requested", "f_test", or what "auto" found: "poisson_mixed",
    # "binomial_mixed", "negative_binomial_mixed", "other_mixed" or
    # "no_random_effects"; see `term_test_auto`), and the columns, one
    # entry per term in formula order: "label", "kind" ("F", "chisq" or
    # "lrt"), "statistic", "num_df", "den_df", "den_df_kind" ("satterthwaite",
    # "residual" or "" on a chi-squared row), "p_value", "sum_sq", "mean_sq"
    # (NaN on a chi-squared row), "row_notes" (a list of sentences per row) and
    # "notes" (the table's sentences).
    term_table_type3: dict | None = None
    term_table_type2: dict | None = None
    # The likelihood-ratio tables `glmm.anova(m, test="LRT")` reads, as the two
    # above: made with the fit under `anova_test="LRT"`, by the first
    # `glmm.anova` that needs them under `anova_test="auto"` (None until then),
    # and never under "Wald". On a fit with F rows each holds the refusal as
    # its "error".
    term_table_type3_lrt: dict | None = None
    term_table_type2_lrt: dict | None = None
    # What `glmm.anova(m)`'s default test="auto" gives on this fit:
    # {"test": "lrt", "reason": "poisson_mixed"} on a Poisson model with random
    # effects (the likelihood-ratio tables above), else the test and reason of
    # `term_table_type3`.
    term_test_auto: dict | None = None
    # The Satterthwaite record `contrast_test` reads, on a fit whose
    # `df_method` is "satterthwaite"; None on every other fit, whose test reads
    # only `beta`, `vcov`, `aliased`, `nobs` and `dispersion`. A dict, every
    # matrix in it a flat row-major float array over the p_r retained columns:
    #   a_eff      ((n_free + 1)**2,)  2 * blockdiag(H_p^+, tau^2 / nu), over
    #                                  the n_free free theta coordinates and tau
    #   jac_theta  list of (p_r**2,)   dC/dtheta_k, one per free coordinate
    #   c          (p_r**2,)           Cov(beta-hat)
    #   tau        float               sigma-hat**2
    #   nu         float               the divisor of sigma-hat**2
    #   retained   list[int] | None    0-based retained columns, None when every
    #                                  column was retained
    # About p**2 * (n_free + 1) floats, kept in a pickle. Mirrors the R port's
    # `object$satterthwaite_record` - change together.
    satterthwaite_record: dict | None = None
    # What `predict`, `simulate`, `pairwise`, `r2` and `icc` read besides the
    # public fields, private: the training columns as the native `fit` took them
    # ("numeric" and "factor", after the missing-value row drop, and the
    # "weights" and "offset" arguments, as bytes or None), lowered again on
    # each call, with "row_index" (each training row's position in the
    # caller's data, as `_prepare_columns` gives it, so an error names the
    # caller's row; absent on a fit pickled before it, None when no row was
    # dropped), the kernel's conditional variances in its own layout
    # ("condvar", "condvar_x", float64 bytes, and "condvar_cross", a dict or
    # None), and on a fit made with `anova_test="auto"` "lrt_call", the rest of
    # the call `_lrt_tables` repeats. None on a fit pickled before them.
    # Mirrors the R port's `.kernel_parts` - change together.
    _kernel_parts: dict | None = field(default=None, repr=False)
    # The power the fit put on `weights` (row i has variance sigma^2 *
    # weights[i]^-weight_power): 1 on the weights as given, NaN without
    # weights. Pearson residuals, `rss`, `r_squared` and `simulate` read the
    # weights raised to it. A fit pickled before it reads 1.
    weight_power: float = 1.0
    # Whether the fit estimated `weight_power` (weight_power="estimate", or None
    # resolved to it) rather than holding it fixed; `glmm.anova(m0, m1)` and
    # `profile_confint` read it. A fit pickled before it reads False.
    estimated_weight_power: bool = False
    # The standard error behind `se`, `vcov` and `coef_df`: "model" or "robust".
    # A fit pickled before it reads "model".
    se_type: str = "model"
    # On a robust fit, the model-based covariance of every retained coefficient
    # (p, p), NaN on an aliased one, which `predict` and the Type II hypotheses
    # read there; None on a model fit, whose `vcov` is it, and on a fit pickled
    # before it. Mirrors the R port's `object$model_vcov` - change together.
    model_vcov: np.ndarray | None = None
    # On a robust fit, its CR2 or HC2 pieces, which the tests of a robust fit
    # read back: a dict of float arrays, every matrix flat and column-major over
    # the p coefficients (zero on an aliased one), cluster k the k-th level of
    # the grouping factor with rows, or row k without one:
    #   er            (p * G,)      E_k r_k, one column per cluster
    #   f, epe        list of (p*p,) F_k and E_k Phi_k E_k', per cluster of
    #                               more than one row
    #   row_e, row_x  (p * n1,)     e_k and x_k of each one-row cluster
    #   row_phi       (n1,)         its scalar Phi_k
    #   null_cluster  list[int]     the cluster of each null direction, 0-based
    #   null          list of (p,)  the null directions
    # None on a model fit and on a fit pickled before it. Mirrors the R port's
    # `object$robust` - change together.
    robust: dict | None = None

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

        Imported locally: `__init__` must finish defining `Fit` before
        `summary.py` is needed."""
        from glmm.summary import build_summary

        if self.re_term_index is None:
            raise ValueError(
                "this fit was made by a glmm version that did not keep re_term_index, "
                "which summary reads. Refit the model with this version."
            )
        # A fit pickled before `_kernel_parts` cannot lower its rows again, so
        # its summary has no R-squared line.
        sizes, need_refit = None, False
        if self._kernel_parts is not None:
            sizes = self._effect_sizes(null_refit=False)
            need_refit = sizes is None
        return build_summary(self, sizes, need_refit)

    def confint(self, level=0.95):
        """Confidence intervals for the fixed effects as a `(p, 2)` array of
        lower and upper bounds, in the order of `names`. Each row follows
        `summary()`: the t quantile on the row's `coef_df` where the summary
        tests with t, the normal quantile where it tests with z (including a
        Gaussian mixed model fitted with `satterthwaite=False`, whose summary
        says "df not computed"). Aliased coefficients give NaN. The R port's
        `confint()` computes the same intervals through the same kernel
        function. For profile-likelihood intervals of a Gaussian mixed model,
        variance parameters included, see `profile_confint`."""
        # A Fit saved before `coef_df` existed has no df: normal intervals,
        # as such a fit's summary tests with z.
        coef_df = getattr(self, "coef_df", None)
        if coef_df is None:
            coef_df = np.full(len(self.beta), np.nan)
        ci = _native.call(
            "confint",
            {
                "beta": [float(v) for v in self.beta],
                "se": [float(v) for v in self.se],
                "coef_df": [float(v) for v in coef_df],
                "level": float(level),
            },
        )
        return np.column_stack([ci["lower"], ci["upper"]])

    def profile_confint(self, level=0.95, parm=None):
        """Profile-likelihood confidence intervals of a Gaussian mixed model,
        lme4's `confint(method = "profile")`, as a `ProfileTable`: one row per
        random-effect SD and correlation, the residual SD and coefficient,
        named as lme4 names them ("sd_(Intercept)|Subject",
        "cor_Days.(Intercept)|Subject", "sigma", then `names`), with the
        columns `estimate`, `lower` and `upper`.

        The profiles are of the ML deviance: the model is refitted once by ML
        from the fit's training rows (a REML fit too, as lme4 does), and the
        rows report the ML estimates; the notes say so. A bound is where the
        signed root of the deviance change reaches the normal quantile of
        `level`. An SD estimated at 0 has 0 as its lower bound, and a
        correlation whose profile stays inside the cutoff up to -1 or 1 has
        that limit. An undetermined bound is NaN, and the notes name it.

        level: the confidence level, strictly between 0 and 1.
        parm: None for every row, or a row name or list of row names; "theta_"
            stands for the SDs, correlations and "sigma", "beta_" for the
            coefficients. The rows come in the table's order whatever the
            order of `parm`.

        Raises `ValueError` on a model without random effects, of a family
        other than Gaussian, or with an aliased coefficient, on a `level`
        outside (0, 1), and on a `parm` name the model does not have (the
        message lists the valid ones); `TypeError` on a `parm` that is not a
        name or a list of names. Name is provisional until 1.0.0."""
        from glmm.anova import _profile_table

        if isinstance(parm, str):
            parm = [parm]
        elif parm is not None:
            if not (isinstance(parm, (list, tuple)) and all(isinstance(p, str) for p in parm)):
                raise TypeError(f"parm must be None, a name or a list of names, got {parm!r}")
            parm = list(parm)
        r = _native.call(
            "profile_intervals",
            {
                **self._model_data(),
                "reml": bool(self.reml),
                "weight_power": float(self.weight_power),
                "weight_power_estimated": bool(self.estimated_weight_power),
                "level": float(level),
                "parm": parm,
            },
        )
        return _profile_table(r, bool(self.reml))

    def contrast_test(self, L):
        """One Wald test of the linear hypothesis `L @ beta = 0`, as a `TermTable`
        with one row named "contrast", laid out as `glmm.anova`'s table of the
        same test.

        L: a float array of shape (rows, p), or a vector of length p, over all
            p coefficients in `names` order, aliased ones included.

        On OLS, a Gaussian mixed model and the Gamma and inverse Gaussian GLMs
        with phi estimated (`df_method` "residual" or "satterthwaite") the row
        is the F test, on Satterthwaite's denominator df on a mixed model and
        n - p otherwise. A Gaussian mixed model fitted with
        `satterthwaite=False`, or whose Satterthwaite pass failed, gets a NaN F
        row with a note saying why. Every other fit gets the Wald chi-squared
        test, also a Poisson GLMM, whose default `glmm.anova` rows are
        likelihood-ratio rows. A non-zero entry on an aliased column gives a
        NaN row and a note; an `L` whose rows span fewer directions than it
        has rows is tested on those directions, with a note. Nothing is
        refitted: there is no likelihood-ratio test here, which is
        `glmm.compare`.

        On a robust fit (`se_type` "robust") a one-row `L` is the robust test,
        the robust t squared on its df, with `satterthwaite=False` too; it
        warns when the df are below 4 (`RobustDfSmallWarning`) and is NaN when
        one group alone identifies it (`RobustSeUnidentifiedWarning`). An `L`
        of more than one row is the model-based F, with a
        `ModelBasedJointTestWarning`: there is no evidence for a robust joint
        test.

        Raises `ValueError` on an `L` of the wrong shape, with no rows or with
        a value that is not finite, and on a fit made by a glmm version that
        did not store what the test reads. The name is provisional until
        1.0.0."""
        from glmm.anova import _check_fields, _term_table

        _check_fields(self, ["df_method"])
        if self.df_method == "satterthwaite":
            _check_fields(self, ["satterthwaite_record"])
        L = np.asarray(L, dtype=float)
        if L.ndim not in (1, 2):
            raise ValueError(f"L must be a vector or a matrix, got {L.ndim} dimensions")
        p = len(self.beta)
        stored = _native.call(
            "contrast_test",
            {
                "df_method": self.df_method,
                "beta": [float(v) for v in self.beta],
                "vcov": np.asarray(self.vcov, dtype=float).reshape(p, p).tolist(),
                **self._robust_args(),
                "aliased": [bool(a) for a in self.aliased],
                "nobs": int(self.nobs),
                "dispersion": float(self.dispersion),
                "record": self._record_arg(),
                "names": list(self.names),
                "l": L.ravel().tolist(),
                "rows": 1 if L.ndim == 1 else L.shape[0],
            },
        )
        _test_warnings(stored)
        # A contrast has no SS type: the crate's table carries a placeholder one.
        stored["ss_type"] = None
        # A contrast is not a term, so its row has no partial eta-squared.
        del stored["eta2_partial"]
        # "Wald": the one-line heading of an explicitly asked chi-squared table.
        return _term_table(self, _term_result(stored), "Wald")

    def predict(
        self,
        newdata=None,
        type="link",
        re_form=None,
        allow_new_levels=True,
        se_fit=False,
        level=0.95,
        future=False,
    ):
        """Predictions on new rows, lme4's `predict(newdata = )`.

        newdata: rows read as `fit` reads `data` (dict, DataFrame, Arrow
            Table), with the columns the formula reads other than the
            response; None (the default) predicts the rows the fit used. A
            factor is coded with the fit's levels by label, so a level the fit
            did not see is an error. A row with a missing value is refused.
        type: "link" (the default) for the linear predictor, "response" for
            the mean.
        re_form: None (the default) for the prediction given the conditional
            modes of every grouping, or "NA" for the fixed effects only, which
            needs no grouping column.
        allow_new_levels: a grouping level the fit did not see gets a random
            effect of zero for that grouping, with a
            `NewGroupingLevelsWarning`; False makes it an error.
        se_fit: also return the standard error and the interval at `level`.
            On the response scale the standard error is the delta method's and
            the interval the inverse link of the link-scale one.
        level: the coverage of the interval and of the future bounds.
        future: also return the bounds of a future observation on the response
            scale (a non-Gaussian mixed model only), the family's quantiles
            mixed over the uncertainty of the linear predictor.

        Returns the predictions as a float array; with `se_fit` or `future`, a
        dict of arrays: "fit", then "se_fit", "lower" and "upper" with
        `se_fit`, and "future_lower" and "future_upper" with `future`.

        The standard error is lme4's (`predict(se.fit = TRUE)`), with the
        variance parameters held at their estimates; on a GLMM it uses the
        fit's fixed-effect covariance (`wald_se`), where lme4 uses the RX one.
        A fit whose random effects route to the sparse solver forms no
        conditional variances, so a conditional standard error there raises.
        On a robust fit (`se_type` "robust") the standard error and interval
        are model-based, as on the same fit with `se_type="model"`, with a
        `PredictSeModelBasedWarning`: their formula holds only with the
        model's covariance.
        A model fitted with the `offset=` argument cannot predict new rows:
        give the offset as an `offset()` term in the formula instead. The name
        is provisional until 1.0.0."""
        if re_form is None:
            population = False
        elif isinstance(re_form, str) and re_form == "NA":
            population = True
        else:
            raise ValueError(
                're_form must be None (the random effects included) or "NA" (fixed effects '
                f"only), got {re_form!r}"
            )
        if newdata is None:
            new_numeric = new_factor = None
            n_rows = 0
        else:
            new_numeric, new_factor, _, _, _ = _prepare_columns(
                newdata, [self.formula], None, None, [], new_data=True
            )
            n_rows = _row_count(newdata, new_numeric, new_factor)
        r = _native.call(
            "predict",
            {
                **self._model_data(),
                "parts": self._stored_numbers(),
                "new_numeric": new_numeric,
                "new_factors": new_factor,
                "n_rows": n_rows,
                "scale": type,
                "population": population,
                "allow_new_levels": bool(allow_new_levels),
                "se": bool(se_fit),
                "level": float(level),
                "future": bool(future),
                # New rows are the caller's own: `_prepare_columns` drops none.
                "row_index": self._kept().get("row_index") if newdata is None else None,
                "row_base": 0,
            },
        )
        for note in r["notes"]:
            if note["kind"] == "predict_se_model_based":
                # Mirrors the R port's predict wording - change together.
                warnings.warn(
                    "The standard errors and intervals are model-based, not robust: their "
                    "formula holds only with the model's covariance.",
                    PredictSeModelBasedWarning,
                    stacklevel=2,
                )
            if note["kind"] == "new_grouping_levels":
                grouping, rows = note["detail"].split(": ", 1)
                name = self.re_groups[int(grouping)][0]
                rows = rows.split(", ")
                n = len(rows)
                rows = ", ".join(rows[:10]) + (f", and {n - 10} more" if n > 10 else "")
                warnings.warn(
                    f"{_count(n, 'row')} of newdata {'has' if n == 1 else 'have'} a level of "
                    f"{name} the fit did not see (rows {rows}). Their random effect for "
                    f"{name} is 0.",
                    NewGroupingLevelsWarning,
                    stacklevel=2,
                )
        fit = _f64(r["fit"])
        if not (se_fit or future):
            return fit
        out = {"fit": fit}
        if se_fit:
            out.update(se_fit=_f64(r["se"]), lower=_f64(r["lower"]), upper=_f64(r["upper"]))
        if future:
            out.update(future_lower=_f64(r["future_lower"]), future_upper=_f64(r["future_upper"]))
        return out

    def simulate(self, nsim=1, seed=None):
        """`nsim` draws of the response on the rows the fit used, lme4's
        `simulate()`, as an `(nobs, nsim)` array. Each draw takes new random
        effects from the fitted covariance and then the family's draw around
        the conditional mean, with the fitted dispersion; a binomial draw is a
        proportion of the row's trials.

        seed: a whole number from 0 to 2**64 - 1; the same seed gives the same
            draws on every platform. None (the default) takes one from Python's
            `random` module.

        The name is provisional until 1.0.0."""
        if isinstance(nsim, bool) or not isinstance(nsim, (int, np.integer)) or nsim < 1:
            raise ValueError(f"nsim must be a whole number of at least 1, got {nsim!r}")
        if seed is None:
            seed = random.getrandbits(64)
        elif isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be None or a whole number, got {seed!r}")
        elif not 0 <= seed < 2**64:
            raise ValueError(f"seed must be from 0 to 2**64 - 1, got {seed}")
        draws = _native.call(
            "simulate",
            {
                **self._model_data(),
                "parts": self._stored_numbers(),
                "nsim": int(nsim),
                "seed": int(seed),
                "row_index": self._kept().get("row_index"),
                "row_base": 0,
            },
        )
        return _f64(draws).reshape(int(nsim), self.nobs).T

    def r2(self):
        """The marginal and conditional R-squared of a mixed model (Nakagawa &
        Schielzeth 2013), as `{"marginal": .., "conditional": ..}`, the two
        values of `performance::r2_nakagawa`. On a binomial model with trial
        counts they are performance's values for a `cbind()` response; a
        `glmer` fitted with `weights =` gets other values there.

        The marginal R-squared is the fixed-effect variance divided by the
        fixed-effect, the random-effect and the residual variance; the
        conditional R-squared also adds the random-effect variance to the
        numerator. A random slope's variance is averaged over the rows
        (Johnson 2014). On a GLMM the residual variance is the lognormal
        approximation of the family's variance on the link scale. A
        random-effect variance of 0 gives a conditional R-squared equal to
        the marginal one. A fit without random effects, a Gamma model with a
        link other than log, an inverse Gaussian model and a fit that did not
        converge give NaN.

        The values are computed when called. On a Poisson or negative
        binomial GLMM each call refits the null model once, so keep the
        result instead of calling again. A fit made by a glmm version that did
        not keep its training data raises `ValueError`: refit it. The name is
        provisional until 1.0.0."""
        e = self._effect_sizes(null_refit=True)
        return {"marginal": e["marginal"], "conditional": e["conditional"]}

    def icc(self):
        """The intraclass correlation of a mixed model over all its groupings,
        as `{"adjusted": .., "unadjusted": ..}`, the two values of
        `performance::icc` (Nakagawa, Johnson & Schielzeth 2017).

        The adjusted ICC is the random-effect variance divided by the random
        and the residual variance; the unadjusted ICC also adds the
        fixed-effect variance to the divisor. The variances are those of
        `r2()`. A random-effect variance of 0 gives 0. A fit without random
        effects, a Gamma model with a link other than log, an inverse Gaussian
        model and a fit that did not converge give NaN.

        The values are computed when called. On a Poisson or negative
        binomial GLMM each call refits the null model once, so keep the
        result instead of calling again. A fit made by a glmm version that did
        not keep its training data raises `ValueError`: refit it. The name is
        provisional until 1.0.0."""
        e = self._effect_sizes(null_refit=True)
        return {"adjusted": e["adjusted"], "unadjusted": e["unadjusted"]}

    def _effect_sizes(self, null_refit):
        """The native `effect_sizes` of this fit: the R-squared pair and the
        two ICCs, or None on a Poisson or negative binomial GLMM when
        `null_refit` is False (they need a refit of the null model)."""
        return _native.call(
            "effect_sizes",
            {
                **self._model_data(),
                "parts": self._stored_numbers(),
                "aliased": [bool(a) for a in self.aliased],
                "converged": bool(self.converged),
                "objective": self.objective,
                "objective_nodes": int(self.objective_nodes),
                "null_refit": null_refit,
            },
        )

    def _lrt_tables(self):
        """Fill `term_table_type3_lrt` and `term_table_type2_lrt` on a fit made
        with `anova_test="auto"`, once. Runs the fit's own call again, which
        is deterministic, so the tables are the ones the fit would have made.
        Mirrors the R port's `.lrt_tables` - change together."""
        call = self._kernel_parts.get("lrt_call") if self._kernel_parts else None
        if call is None or self.term_table_type3_lrt is not None:
            return
        wald_se, nagq, dispersion, warm_start = call
        if isinstance(warm_start, tuple):  # a fit pickled by 0.8.0
            warm_start = dict(zip(("beta", "theta", "dispersion"), warm_start))
        r = _native.call(
            "fit",
            {
                **self._model_data(),
                "wald_se": wald_se,
                "nagq": nagq,
                "dispersion": dispersion,
                "warm_start": warm_start,
                # No chi-squared fit reads satterthwaite, condvar or reml.
                "satterthwaite": False,
                "condvar": False,
                "reml": None,
                "anova_test": "auto" if self.term_test_auto["test"] == "lrt" else "LRT",
                "row_index": self._kernel_parts["row_index"],
                "row_base": 0,
            },
        )
        self.term_table_type3_lrt = _lrt_result(r["term_table_type3_lrt"])
        self.term_table_type2_lrt = _lrt_result(r["term_table_type2_lrt"])

    def _robust_args(self):
        """`model_vcov` and `robust` as the native `contrast_test` and `pairwise`
        take them, None when the fit has none (a model fit, or one pickled
        before them)."""
        rec = self.robust
        return {
            "model_vcov": self._model_vcov_arg(),
            "robust": None
            if rec is None
            else {
                **{
                    k: np.asarray(rec[k], dtype=float).tolist()
                    for k in ("er", "row_e", "row_x", "row_phi")
                },
                **{
                    k: [np.asarray(v, dtype=float).tolist() for v in rec[k]]
                    for k in ("f", "epe", "null")
                },
                "null_cluster": [int(k) for k in rec["null_cluster"]],
            },
        }

    def _model_vcov_arg(self):
        """`model_vcov` as rows, None when the fit has none."""
        mv = self.model_vcov
        return None if mv is None else np.asarray(mv, dtype=float).tolist()

    def _record_arg(self):
        """`satterthwaite_record` as the native `contrast_test` and `pairwise`
        take it, None when the fit has none."""
        rec = self.satterthwaite_record
        if rec is None:
            return None
        return {
            "a_eff": np.asarray(rec["a_eff"], dtype=float).tolist(),
            "jac_theta": [np.asarray(j, dtype=float).tolist() for j in rec["jac_theta"]],
            "c": np.asarray(rec["c"], dtype=float).tolist(),
            "tau": float(rec["tau"]),
            "nu": float(rec["nu"]),
            "retained": rec["retained"],
        }

    def __setstate__(self, state):
        # A fit pickled by 0.6.x has no `re_term_index`; it still unpickles and
        # prints, and `summary_object` refuses it.
        self.__dict__.update({"re_term_index": None, **state})

    def _kept(self):
        """`_kernel_parts`, refusing a fit pickled before it."""
        if self._kernel_parts is None:
            raise ValueError(
                "this fit was made by a glmm version that did not keep what predict, "
                "simulate, pairwise, r2, icc and profile_confint read. Refit the model "
                "with this version."
            )
        return self._kernel_parts

    def _model_data(self):
        """The training data as the native `predict`, `simulate`, `pairwise`,
        `effect_sizes`, `profile_intervals` and `fit` take it."""
        kept = self._kept()
        return {
            "formula": self.formula,
            "numeric": kept["numeric"],
            "factors": kept["factor"],
            "family": self.family,
            "link": self.link,
            "weights": kept["weights"],
            "offset": kept["offset"],
        }

    def _stored_numbers(self):
        """The fit's numbers as the native `predict`, `simulate` and
        `effect_sizes` take them."""
        kept = self._kept()
        p = len(self.beta)
        return {
            "beta": [float(v) for v in self.beta],
            "vcov": np.asarray(self.vcov, dtype=float).reshape(p, p).tolist(),
            "model_vcov": self._model_vcov_arg(),
            "varcorr": [[float(v) for v in b] for b in self.varcorr],
            "ranef": np.asarray(self.ranef, dtype=np.float64).tobytes(),
            "ranef_levels": [int(n) for n in self.ranef_levels],
            "condvar": kept["condvar"],
            "condvar_x": kept["condvar_x"],
            "condvar_cross": kept["condvar_cross"],
            "dispersion": float(self.dispersion),
            "weight_power": float(self.weight_power),
        }

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


# The warning category of each kind the fit record's notes and `failure`
# carry; any other kind raises the base category. The tier, title and message
# come rendered with the record (glmm-bind/src/notes.rs). Mirrors the R port's
# .NOTE_KINDS (glmm-r/R/fastglmm.R) - change together.
_CATEGORIES = {
    "search_limit": SearchLimitWarning,
    "fit_failed": FitFailedWarning,
    "glm_diverged": GlmDivergedWarning,
    "design_unsolvable": DesignUnsolvableWarning,
    "constant_response": ConstantResponseWarning,
    "too_few_rows": TooFewRowsWarning,
    "no_coefficients": NoCoefficientsWarning,
    "ill_conditioned": IllConditionedWarning,
    "pirls_exhausted": PirlsExhaustedWarning,
    "nb_shape_unsettled": NbShapeUnsettledWarning,
    "single_level_grouping_dropped": SingleLevelGroupingDroppedWarning,
    "few_rows_for_random_effects": FewRowsForRandomEffectsWarning,
    "variance_at_limit": VarianceAtLimitWarning,
    "hessian_se_fallback": HessianSeFallbackWarning,
    "se_unavailable": SeUnavailableWarning,
    "gradient_search_fallback": GradientSearchFallbackWarning,
    "exact_profile_fallback": ExactProfileFallbackWarning,
    "precision_limited": PrecisionLimitedWarning,
    "non_integer_response": NonIntegerResponseWarning,
    "satterthwaite_curvature": SatterthwaiteCurvatureWarning,
    "weight_power_at_bound": WeightPowerAtBoundWarning,
    "weight_power_search_failures": WeightPowerSearchFailuresWarning,
    "weight_power_outside_unit": WeightPowerOutsideUnitWarning,
    "weight_power_implausible": WeightPowerImplausibleWarning,
    "robust_with_random_slope": RobustWithRandomSlopeWarning,
    "robust_se_unidentified": RobustSeUnidentifiedWarning,
    "robust_df_small": RobustDfSmallWarning,
}


def _warn(store, kind, message, category):
    """Raise one warning this port words itself, under the tier and title of
    `kind`; see `_raise`."""
    tt = _native.call("warning_kind", {"kind": kind})
    _raise(
        store,
        {"tier": tt["tier"], "kind": kind, "title": tt["title"], "message": message},
        category,
    )


def _raise(store, entry, category):
    """Raise `entry` (its `tier`, `kind`, `title` and `message`) as
    `<Tier>: <title>. <message>` and append those four fields to `store`, the
    list that becomes `Fit.warnings`. The warning points at the line that
    called into glmm (`fit`, `compare`), however deep in the package it is
    raised: the stack level is the first frame outside this directory."""
    entry = {k: entry[k] for k in ("tier", "kind", "title", "message")}
    store.append(entry)
    level, frame = 1, sys._getframe()
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        level, frame = level + 1, frame.f_back
    warnings.warn(
        f"{entry['tier'].capitalize()}: {entry['title']}. {entry['message']}",
        category=category,
        stacklevel=level,
    )


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


def _count(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _check_call(formula, family, link):
    """The checks `fit` and `compare` make on a formula, the family and the
    link before any data is read; returns the link, the family's default when
    `link` is None."""
    # A formula shape the shared Rust parser has no dedicated message for (it
    # would fall through to a generic syntax error): caught here with the same
    # wording the R port uses (glmm-r/R/fastglmm.R's .check_formula) — change
    # together.
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


def _prepare_columns(
    data,
    formulas,
    weights,
    offset,
    store,
    drop_missing_arguments=False,
    model_matrix=False,
    new_data=False,
):
    """The columns of `data` that `formulas` use, as the native `fit` takes them,
    with `weights=`/`offset=` checked: `(numeric_columns, factor_columns,
    weights, offset, row_index)`. A row with a missing value in a column any of the
    formulas uses is dropped once, from every column and from the weights and
    the offset, so two formulas (`compare`) are fitted on the same rows. With
    `drop_missing_arguments` (`compare`), a row whose weight or offset is
    missing is dropped the same way; otherwise a missing entry is refused.
    With `model_matrix`, the columns are looked up as `model_matrix` reads
    them (the native `model_matrix_columns`), and a row with a missing value
    raises instead of being dropped. With `new_data` (`Fit.predict`), the
    columns are the ones the formula reads on new rows, the response left
    out (the native `predict_columns`), a missing one is left to the kernel to
    name (a grouping column may be absent), and a row with a missing value
    raises. `row_index` is None when no row was dropped, and otherwise each
    kept row's 0-based position in `data` as int64 bytes, which the
    `_native` entry points take so an error names the caller's row."""
    # Only the columns the formula actually uses are read from here on — see
    # `_formula_columns`.
    used_columns = _columns(data)
    if model_matrix:

        def lookup(formula, names):
            args = {"formula": formula, "available": names}
            return _columns_triple(_native.call("model_matrix_columns", args))

    elif new_data:

        def lookup(formula, names):
            args = {"formula": formula, "available": names}
            used = _native.call("predict_columns", args)
            return used, [], _formula_columns(formula, names)[2]

    else:
        lookup = _formula_columns
    used_names, missing, numeric_only = [], [], []
    for formula in formulas:
        for into, names in zip(
            (used_names, missing, numeric_only), lookup(formula, list(used_columns))
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
    # default na.action = na.omit (`glmm-r/R/fastglmm.R`) — there is no
    # `model.frame` here to do that filtering for free. A declared
    # categorical's missing rows (pandas code -1, pyarrow null) feed the same
    # `na_mask` as a plain column's missing values: `_levels_and_codes` never
    # raises on them, it just leaves the sentinel in `codes` for this loop to
    # find. `na_mask` is None only when the formulas read no column at all.
    #
    # Numeric and plain-string columns go through `_plain_column` in one
    # pass; any other column keeps the per-row path ("values" below). Both
    # read every value the same way, so the result never depends on the path.
    parsed = {}
    keeps = {}
    for name in used_names:
        col = used_columns[name]
        if name not in numeric_only and _is_bool_column(col):
            col = _bool_values(col)
        else:
            declared = _levels_and_codes(col)
            if declared is not None:
                levels, codes = declared
                codes = np.asarray(codes, dtype=np.int64)
                parsed[name] = ("factor", levels, codes)
                keeps[name] = codes != -1
                continue
        entry = _plain_column(col)
        # float64 holds integers exactly only below 2**53; past that, distinct
        # ids such as hashed grouping keys would merge without a word.
        if (
            entry is not None
            and getattr(getattr(col, "dtype", None), "kind", None) in ("i", "u")
            and entry[2].any()
            and np.abs(entry[1][entry[2]]).max() >= 2**53
        ):
            raise ValueError(
                f"column '{name}' holds integers of 2**53 or more, which a float cannot "
                "store exactly; pass them as strings (for example df[col].astype(str))"
            )
        if entry is None:
            values = _values(col)
            keep = np.fromiter((not _is_na(v) for v in values), dtype=bool, count=len(values))
            entry = ("values", values, keep)
        parsed[name] = entry
        keeps[name] = entry[-1]

    # A dict's columns can differ in length, so a ragged dict is refused here,
    # before the per-column masks are combined.
    _check_lengths({name: len(keep) for name, keep in keeps.items()})
    na_mask = np.logical_and.reduce(list(keeps.values())) if keeps else None

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
            na_mask = na_mask & ~np.isnan(arr)

    drop = na_mask is not None and not na_mask.all()
    if drop:
        n_kept = int(na_mask.sum())
        if model_matrix or new_data:
            # No row is dropped: a numpy array has no row names, so the design
            # (or the predictions) would no longer line up with the caller's
            # own arrays.
            raise ValueError(
                f"{len(na_mask) - n_kept} of {len(na_mask)} row(s) have a missing value in "
                f"{where}; drop those rows from data first"
            )
        if n_kept == 0:
            raise ValueError(f"every row has a missing value in {where}; no rows left to fit")
        _warn(
            store,
            "rows_dropped_na",
            f"Dropped {len(na_mask) - n_kept} of {len(na_mask)} row(s): {where} had "
            "a missing value there.",
            RowsDroppedWarning,
        )
        # weights=/offset= are caller-supplied arrays aligned to data's original
        # rows, not columns of `data` itself, so the row drop must repeat here.
        if weights is not None:
            weights = weights[na_mask]
        if offset is not None:
            offset = offset[na_mask]

    # Columns cross as the raw bytes of float64 values and uint32 factor codes
    # (`from_bytes` in glmm-py/src/lib.rs).
    numeric_columns = {}
    factor_columns = {}
    for name, entry in parsed.items():
        kind = entry[0]
        if kind == "factor":
            codes = entry[2][na_mask] if drop else entry[2]
            factor_columns[name] = (entry[1], codes.astype(np.uint32).tobytes())
        elif kind == "num":
            values = entry[1][na_mask] if drop else entry[1]
            numeric_columns[name] = values.tobytes()
        elif kind == "str":
            _, codes, keys, _ = entry
            if drop:
                codes = codes[na_mask]
            if len(codes) == 0:
                numeric_columns[name] = b""
                continue
            # Levels from the kept rows only, sorted as `_sorted_levels_and_codes`
            # sorts them.
            present = np.bincount(codes, minlength=len(keys)) > 0
            # .tolist(): iterating the numpy array yields numpy integers, which
            # index the `keys` list more slowly than Python ints.
            levels = sorted({str(keys[i]) for i in np.flatnonzero(present).tolist()})
            position = {level: i for i, level in enumerate(levels)}
            # A key with no kept row is never indexed, so its 0 is a placeholder.
            remap = np.array([position.get(str(k), 0) for k in keys], dtype=np.uint32)
            factor_columns[name] = (levels, remap[codes].tobytes())
        else:
            values = entry[1]
            if drop:
                values = [v for v, k in zip(values, na_mask.tolist()) if k]
            if values and isinstance(values[0], str):
                levels, codes = _sorted_levels_and_codes([str(v) for v in values])
                factor_columns[name] = (levels, np.asarray(codes, dtype=np.uint32).tobytes())
            else:
                numeric_columns[name] = np.array([float(v) for v in values], dtype=float).tobytes()
    row_index = np.flatnonzero(na_mask).astype(np.int64).tobytes() if drop else None
    return numeric_columns, factor_columns, weights, offset, row_index


def _check_lengths(lengths):
    """Raise when the `{name: length}` columns of `lengths` differ in length."""
    # Same message as `table_of`'s length check in `src/orchestrate.rs` — change
    # together.
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f'"{name}": {n}' for name, n in sorted(lengths.items()))
        raise ValueError(
            f"columns have mismatched lengths ({detail}); all columns must have the same length"
        )


# The notes of a robust fit's `contrast_test` and `pairwise`, by kind. Mirrors
# the R port's .TEST_WARNINGS - change together.
_TEST_WARNINGS = {
    "robust_se_unidentified": RobustSeUnidentifiedWarning,
    "robust_df_small": RobustDfSmallWarning,
    "model_based_joint_test": ModelBasedJointTestWarning,
}


def _test_warnings(stored):
    """Raise the robust notes of the native `contrast_test` or `pairwise`
    result `stored` as warnings, at the caller of the public function, and drop
    them from it."""
    for w in stored.pop("warnings"):
        warnings.warn(w["message"], _TEST_WARNINGS.get(w["kind"], DiagnosticWarning), stacklevel=3)


def _term_result(t):
    """A stored per-term result (the native `fit`'s dict) with its numeric columns
    as float arrays; see `Fit.term_table_type3`."""
    if t["error"] is not None:
        return {"error": t["error"]}
    numeric = ("statistic", "num_df", "den_df", "p_value", "sum_sq", "mean_sq", "eta2_partial")
    return {k: np.asarray(v, dtype=float) if k in numeric else v for k, v in t.items()}


def _satterthwaite_record(rec):
    """The native `fit`'s `satterthwaite_record` with its matrices as float arrays,
    None when the fit has none; see `Fit.satterthwaite_record`."""
    if rec is None:
        return None
    return {
        "a_eff": np.asarray(rec["a_eff"], dtype=float),
        "jac_theta": [np.asarray(j, dtype=float) for j in rec["jac_theta"]],
        "c": np.asarray(rec["c"], dtype=float),
        "tau": rec["tau"],
        "nu": rec["nu"],
        "retained": rec["retained"],
    }


def _robust_record(rec):
    """The native `fit`'s `robust` record with its numbers as float arrays, None
    when the fit has none; see `Fit.robust`."""
    if rec is None:
        return None
    flat = ("er", "row_e", "row_x", "row_phi")
    return {
        **{k: _f64(rec[k]) for k in flat},
        "f": [np.asarray(v, dtype=float) for v in rec["f"]],
        "epe": [np.asarray(v, dtype=float) for v in rec["epe"]],
        "null_cluster": list(rec["null_cluster"]),
        "null": [np.asarray(v, dtype=float) for v in rec["null"]],
    }


def _lrt_result(t):
    """`_term_result` of a stored likelihood-ratio table, None when the fit
    stored none."""
    return None if t is None else _term_result(t)


def _f64(b):
    """A per-row vector `_native` returns as float64 bytes, as an array that
    owns its data and is writable (`np.frombuffer` alone is a read-only view
    of the bytes)."""
    return np.frombuffer(b, dtype=np.float64).copy()


def _postvar(block):
    """A `ranef_blocks` entry's conditional covariances as `(levels, q, q)`,
    from the row-major per-level vech (column-major lower triangle) the
    native dict carries; None when the fit formed none."""
    levels, q = len(block["levels"]), len(block["terms"])
    if levels == 0 or not block["condvar"]:
        return None
    vech = np.asarray(block["condvar"], dtype=float).reshape(levels, q * (q + 1) // 2)
    # `triu_indices` walks (c, r), c <= r, in the vech's column-major order.
    c, r = np.triu_indices(q)
    out = np.empty((levels, q, q))
    out[:, r, c] = vech
    out[:, c, r] = vech
    return out


def _ranef_block(b):
    """One `ranef_blocks` entry of `Fit` from the native dict's; see
    `Fit.ranef_blocks`."""
    postvar = _postvar(b)
    return {
        "group": b["group"],
        "terms": b["terms"],
        "levels": b["levels"],
        "values": np.asarray(b["values"], dtype=float).reshape(len(b["levels"]), len(b["terms"])),
        "postvar": postvar,
        "condsd": None if postvar is None else np.sqrt(np.diagonal(postvar, axis1=1, axis2=2)),
    }


def _fit_from_native(r, formula, family, link, nagq, dispersion, store, kept):
    """The `Fit` of the native `fit`'s dict `r`, with the fit's warnings raised
    and appended to `store`; `kept` is the training columns and the
    `weights=`/`offset=` bytes the call sent (`Fit._kernel_parts`), the other
    arguments the call's own."""
    # nagq's shape eligibility (single grouping factor, binomial/Poisson/NB/Gamma,
    # q <= 3) is only decidable after the Rust-side formula lowering, so the
    # warn-and-strip for it lives in the shared glmm::orchestrate module
    # (src/orchestrate.rs).
    # Whether the FITTED model actually has random effects — not `mixed`,
    # which only says the formula asked for some: a single-level grouping is
    # dropped after lowering (`single_level_grouping_dropped`), and once every
    # random-effect term is dropped this way, `r["re_group_names"]` comes back
    # empty even though `mixed` (read off the formula text before fitting)
    # still says True.
    fitted_mixed = bool(r["re_group_names"])
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
        # port (glmm-r/R/fastglmm.R, `fastglmm`) words the same sentence with
        # `REML = TRUE`; change together.
        _warn(
            store,
            "argument_ignored",
            "reml=True applies only to Gaussian models; it was ignored and the model was "
            "fitted by maximum likelihood.",
            ArgumentIgnoredWarning,
        )
    # Wrap the native dict's plain lists back into the array types Fit's
    # dataclass documents (no numpy Rust dep — the native call returns lists,
    # and the per-row vectors as bytes).
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
        re_groups=list(zip(r["re_group_names"], r["re_group_terms"], strict=True)),
        re_term_index=r["re_term_index"],
        n_eval=r["n_eval"],
        n_grad=r["n_grad"],
        outer_stop=r["outer_stop"],
        outer_decrement=r["outer_decrement"],
        deviance=r["deviance"],
        loglik=r["loglik"],
        df=r["df"],
        reml=r["reml"],
        fitted=_f64(r["fitted"]),
        ranef=_f64(r["ranef"]),
        ranef_levels=np.frombuffer(r["ranef_levels"], dtype=np.uint64).astype(int),
        ranef_blocks=[_ranef_block(b) for b in r["ranef_blocks"]],
        formula=formula,
        family=family,
        link=link,
        nagq=nagq if r["agq_warning"] is None else 1,
        nobs=int(r["nobs"]),
        y=_f64(r["y"]),
        weights=_f64(r["weights"]) if r["weights"] is not None else None,
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
        offset=_f64(r["offset"]) if r["offset"] is not None else None,
        objective=r["objective"],
        objective_nodes=int(r["objective_nodes"]),
        term_table_type3=_term_result(r["term_table_type3"]),
        term_table_type2=_term_result(r["term_table_type2"]),
        term_table_type3_lrt=_lrt_result(r["term_table_type3_lrt"]),
        term_table_type2_lrt=_lrt_result(r["term_table_type2_lrt"]),
        term_test_auto=dict(r["term_test_auto"]),
        satterthwaite_record=_satterthwaite_record(r["satterthwaite_record"]),
        weight_power=r["weight_power"],
        estimated_weight_power=r["weight_power_estimated"],
        se_type=r["se_type"],
        model_vcov=np.asarray(r["model_vcov"], dtype=float) if r["model_vcov"] else None,
        robust=_robust_record(r["robust"]),
        _kernel_parts={
            **kept,
            "condvar": r["condvar"],
            "condvar_x": r["condvar_x"],
            "condvar_cross": r["condvar_cross"],
        },
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
    # The notes and the non-convergence warning come with their text
    # (glmm-bind/src/notes.rs); a note without a message raises nothing.
    for note in res.diagnostics["notes"]:
        if note["message"] is not None:
            _raise(store, note, _CATEGORIES.get(note["kind"], DiagnosticWarning))
    if r["failure"] is not None:
        _raise(store, r["failure"], _CATEGORIES[r["failure"]["kind"]])
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
    condvar=True,
    reml=None,
    anova_test="auto",
    weight_power=None,
    se_type=None,
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
    condvar: compute the conditional variances of the random effects (default
        True), which `postvar` and `condsd` in `Fit.ranef_blocks` and a
        conditional `predict` with standard errors or future bounds read.
        False skips them, which saves up to about the time of the fit itself
        on a crossed model with many levels: `postvar` and `condsd` are then
        None and such a `predict` raises.
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
    anova_test: when the likelihood-ratio tables of `glmm.anova(m, test=)` are
        made. Every fit stores its Wald chi-squared (or F) tables, with no
        refit. "auto" (default) makes the likelihood-ratio tables the first
        time `glmm.anova` needs them, which on a Poisson model with random
        effects is `glmm.anova(m)` itself; "LRT" makes them with the fit, on
        every fit with chi-squared rows (any GLMM, and every GLM but a Gamma
        or inverse Gaussian one with phi estimated); "Wald" never makes them,
        and `glmm.anova` then refuses a test that needs them. The tables
        refit the model once per distinct reduced design, and under "auto"
        once more for the fit itself. On a fit with F rows the fit stands and
        `glmm.anova(m, test="LRT")` raises.
    weight_power: None, "estimate" or a finite number: the power on
        `weights=`, so row i has variance sigma^2 * weights[i]^-weight_power.
        "estimate" fits the power by REML (by ML with reml=False) and refits at
        it; the power counts as one parameter in the fit's df, and the
        standard errors treat it as known. It needs a Gaussian model with
        weights not all equal, at most one grouping factor and at least 20
        rows. None is "estimate" on such a model fitted by REML (reml None or
        True), and 1 everywhere else. A number fits on weights**weight_power:
        1 takes the weights as given, 0 ignores them; a value other than 0 and
        1 has no simulation evidence behind it. A number other than 1 needs a
        Gaussian model when there are weights; without weights it changes
        nothing and warns (ArgumentIgnoredWarning). `Fit.weight_power` reports
        the value used, the estimate on a fitted power.
    se_type: None, "model" or "robust", the standard error the fit reports.
        None gives "robust" on a REML fit that estimates the weight power
        with one grouping factor and no random slope (CR2, Pustejovsky-Tipton
        df) or with none (HC2, Bell-McCaffrey df), and "model" everywhere
        else. "robust" works on any Gaussian model with at most one grouping
        factor, by REML when it has one or estimates the weight power, and
        warns on a random slope (`RobustWithRandomSlopeWarning`).
        `Fit.se_type` reports the value used, and `Fit.df_method` the df.

    A categorical column's level order is honored: level 0 is the
    treatment-contrast base, so `pd.Categorical(x, categories=[…])` fits against
    the first category you list. A plain string column has no declared order and
    is sorted by code point (R's `factor()` default in the C locale; R in
    another locale can order mixed-case labels such as `"a"`, `"B"` differently). A bool column (Python
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
    if anova_test not in ("auto", "Wald", "LRT"):
        raise ValueError(f"anova_test must be 'auto', 'Wald' or 'LRT', got {anova_test!r}")

    _check_nagq(nagq)

    if not isinstance(satterthwaite, (bool, np.bool_)):
        raise TypeError(f"satterthwaite must be True or False, got {satterthwaite!r}")
    satterthwaite = bool(satterthwaite)
    if not isinstance(condvar, (bool, np.bool_)):
        raise TypeError(f"condvar must be True or False, got {condvar!r}")
    condvar = bool(condvar)
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

    numeric_columns, factor_columns, weights, offset, row_index = _prepare_columns(
        data, [formula], weights, offset, store
    )

    warm_start_record = None
    if warm_start is not None:
        d = warm_start.get("dispersion")
        warm_start_record = {
            "beta": [float(v) for v in warm_start.get("beta", [])],
            "theta": [float(v) for v in warm_start.get("theta", [])],
            "dispersion": None if d is None else float(d),
        }

    kept = {
        "numeric": numeric_columns,
        "factor": factor_columns,
        "weights": weights.tobytes() if weights is not None else None,
        "offset": offset.tobytes() if offset is not None else None,
        "row_index": row_index,
    }
    if anova_test == "auto":
        kept["lrt_call"] = (wald_se, nagq, dispersion, warm_start_record)
    r = _native.call(
        "fit",
        {
            "formula": formula,
            "numeric": numeric_columns,
            "factors": factor_columns,
            "family": family,
            "link": link,
            "wald_se": wald_se,
            "nagq": nagq,
            "dispersion": dispersion,
            "weights": kept["weights"],
            "offset": kept["offset"],
            "warm_start": warm_start_record,
            "satterthwaite": satterthwaite,
            "condvar": condvar,
            "reml": reml,
            "anova_test": "Wald" if anova_test == "auto" else anova_test,
            "weight_power": weight_power,
            "se_type": se_type,
            "row_index": row_index,
            "row_base": 0,
        },
    )
    # The native call has refused anything but None, "estimate" and a number.
    if weight_power not in (None, "estimate", 1) and math.isnan(r["weight_power"]):
        _warn(
            store,
            "argument_ignored",
            f"weight_power={weight_power} has no effect without weights.",
            ArgumentIgnoredWarning,
        )
    return _fit_from_native(r, formula, family, link, nagq, dispersion, store, kept)


def model_matrix(data, formula):
    """The fixed-effects design of `formula` on `data`, as `(X, names)`.

    data: dict[str, array-like]; DataFrame / Arrow Table also accepted, read
        exactly as `fit` reads them.
    formula: R-style string with `~`, e.g. "~ age + gender * condition".
        A response (`"y ~ x"`) is optional; its columns must exist and have
        no missing value, but they are not part of `X`.

    X: float64 array of shape (rows of data, p), the intercept column first
        when the formula has one.
    names: the p column names, the coefficient names `fit` gives the same
        formula.

    `X` and `names` are the design `fit(data, "y ~ <same right side>")`
    builds: factors get the same reference level and coding, and an aliased
    column is kept. A formula with a random-effect or an `offset()` term is
    refused, and so is a row with a missing value in a column the formula
    reads: drop such rows first, so `X` stays aligned with `data`.
    """
    numeric_columns, factor_columns, _, _, _ = _prepare_columns(
        data, [formula], None, None, [], model_matrix=True
    )
    n_rows = _row_count(data, numeric_columns, factor_columns)
    r = _native.call(
        "model_matrix",
        {
            "formula": formula,
            "numeric": numeric_columns,
            "factors": factor_columns,
            "n_rows": n_rows,
            "row_base": 0,
        },
    )
    return _f64(r["x"]).reshape(r["n"], r["p"]), r["names"]


def _row_count(data, numeric_columns, factor_columns):
    """The row count of `data` whose used columns `_prepare_columns` gave as
    `numeric_columns` and `factor_columns`."""
    if isinstance(data, dict):
        # The row count is the length of the formula's columns, which
        # `_prepare_columns` has already checked against each other. `~ 1`
        # reads no column, so then every column of `data` gives it.
        names = [*numeric_columns, *factor_columns] or list(data)
        if not names:
            raise ValueError("data has no columns, so it has no row count")
        lengths = {name: len(data[name]) for name in names}
        _check_lengths(lengths)
        return next(iter(lengths.values()))
    return len(data)


def pairwise(m, factor, by=None, at=None, adjust="holm"):
    """Every pairwise comparison of the levels of one factor of the fixed
    part, emmeans' `pairs(emmeans(m, ~ factor))`, as a `PairwiseTable`.
    Nothing is refitted.

    Each comparison is the difference of two levels' estimated means on the
    link scale. The other factors are averaged with equal weights. Every
    numeric variable is at its mean over the fit's rows. The test is the t
    test on Satterthwaite's df on a Gaussian mixed model, the t test on the
    residual df on OLS and the Gamma and inverse Gaussian GLMs, and the z test
    on every other fit. On a log or logit link each row also has the ratio of
    the two means (the odds ratio on a logit link), `exp(estimate)`.

    factor: the name of a factor of the fixed part.
    by: None, or the name of another variable of the fixed part. The
        comparisons are made within each level of a factor `by`, or at each
        value in `at` of a numeric `by`. The p-values are adjusted within each
        of these groups.
    at: the values of a numeric `by`, a list of numbers.
    adjust: "holm" (the default), "bonferroni", "tukey" or "none". The default
        is Holm, but emmeans' default is Tukey: pass adjust="tukey" to get
        emmeans' p-values.

    A Gaussian mixed model fitted with `satterthwaite=False` has no df: its
    rows have NaN df, statistic and p-values, and a note. A comparison that
    reads an aliased coefficient (an empty cell) is NaN, with a note.
    On a robust fit (`se_type` "robust") each row is the robust t test on its
    robust df, with `satterthwaite=False` too; a comparison one group alone
    identifies is NaN (`RobustSeUnidentifiedWarning`), and df below 4 warn
    (`RobustDfSmallWarning`).
    `confint()` of the table gives the confidence intervals.

    Raises `ValueError` on a name that is not a variable of the fixed part, a
    numeric `factor`, `at` without a numeric `by`, an unknown `adjust`, and a
    fit made by a glmm version that did not keep its training data. The name
    is provisional until 1.0.0."""
    from glmm.anova import _pairwise_table

    if not isinstance(m, Fit):
        raise TypeError(f"glmm.pairwise takes a glmm.Fit, got {type(m).__name__}")
    if not isinstance(factor, str):
        raise TypeError(f"factor must be a variable name, got {factor!r}")
    if by is not None and not isinstance(by, str):
        raise TypeError(f"by must be None or a variable name, got {by!r}")
    if at is not None:
        if by is None:
            raise ValueError("at gives the values of a numeric by, so it needs by")
        at = np.asarray(at, dtype=float).ravel()
        if at.size == 0 or not np.isfinite(at).all():
            raise ValueError("at must be one or more finite numbers")
        at = at.tolist()
    p = len(m.beta)
    stored = _native.call(
        "pairwise",
        {
            **m._model_data(),
            "df_method": m.df_method,
            "beta": [float(v) for v in m.beta],
            "vcov": np.asarray(m.vcov, dtype=float).reshape(p, p).tolist(),
            **m._robust_args(),
            "aliased": [bool(a) for a in m.aliased],
            "nobs": int(m.nobs),
            "record": m._record_arg(),
            "names": list(m.names),
            "factor": factor,
            "by": by,
            "by_values": at,
            "adjust": adjust,
        },
    )
    _test_warnings(stored)
    return _pairwise_table(stored, m.df_method == "none")


def term_labels(formula):
    """The fixed-term labels of `formula` and whether it has an intercept, as
    `(labels, has_intercept)`, read from the formula alone, with no data.

    formula: R-style string, e.g. "y ~ age + sex * condition + (1 | id)". The
        response is optional ("~ x").

    labels: R's `attr(terms(f), "term.labels")`. The variables of a term are
        joined by ":" in the order they first appear in the formula's fixed
        part (`~ a + b + b:a` gives "a:b"); terms are sorted by degree and keep
        their written order within a degree. A parenthesized group is expanded
        as `fit` expands it (`a*(b + c)` gives "a", "b", "c", "a:b", "a:c").
        Random-effect terms and `offset()` are not labels. Match the labels as
        returned rather than building them.
    has_intercept: False for `0 +` or `- 1`.

    A formula that does not parse raises `ValueError` with the parser's
    message. The name is provisional until 1.0.0.
    """
    r = _native.call("term_labels", {"formula": formula})
    return list(r["labels"]), bool(r["has_intercept"])


# Last: `glmm/anova.py` reads `Fit` and the helpers above from this module.
from glmm.anova import (
    Comparison,
    ComparisonTable,
    PairwiseTable,
    ProfileTable,
    TermTable,
    anova,
    compare,
)
