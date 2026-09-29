import datetime
import warnings

import numpy as _np
import pytest

import glmm

DATA = {"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0], "g": ["a", "b", "a"]}

# A larger, family-appropriate dataset for tests whose call now reaches the
# kernel (unlike DATA above, which is deliberately too small/malformed to fit
# — it only exercises the pure-Python validation that runs before any native
# call, so it must stay unchanged for those tests).
_rng = _np.random.default_rng(0)
_N = 200
_GROUPS = _np.repeat(_np.arange(20), _N // 20)
_X = _rng.normal(size=_N)


def _wald_rng(mu, lam, rng):
    # Michael-Schucany-Haas transform: a chi-square(1) draw via a squared
    # normal, folded into the two-root Wald solution by an acceptance test on
    # which root has the right mean (Wald 1947 identity, see also Chhikara &
    # Folks 1989 §4.5). Used only to build a positive, inverse-Gaussian-shaped
    # test fixture — not part of the fitted model.
    v = rng.normal(size=mu.shape) ** 2
    x = mu + mu**2 * v / (2 * lam) - (mu / (2 * lam)) * _np.sqrt(4 * mu * lam * v + mu**2 * v**2)
    return _np.where(rng.uniform(size=mu.shape) <= mu / (mu + x), x, mu**2 / x)


FIT_DATA = {
    "x": _X.tolist(),
    "g": [f"g{i}" for i in _GROUPS.tolist()],
    "y_gauss": (1.0 + 2.0 * _X + _rng.normal(scale=0.5, size=_N)).tolist(),
    # Binomial's kernel domain is {0, 1} (see GLMM/src/glm.rs) — a proper 0/1
    # draw, unlike the count-valued y_pois below, which must never be fit as
    # family="binomial".
    "y_bin": _rng.binomial(1, 1.0 / (1.0 + _np.exp(-(0.2 + 0.8 * _X)))).astype(float).tolist(),
    "y_pois": _rng.poisson(_np.exp(0.5 + 0.3 * _X)).astype(float).tolist(),
    "y_gamma": _rng.gamma(shape=2.0, scale=_np.exp(0.5 + 0.1 * _X) / 2.0).tolist(),
    "y_invgauss": _wald_rng(_np.exp(0.3 + 0.2 * _X), 3.0, _rng).tolist(),
}


def test_unknown_family_message_matches_the_r_port():
    # Exact text, including the family list order: mirrors the R port's
    # equivalent case (r/tests/testthat/test-errors.R has no separate test
    # for this since .normalize_family's string branch shares the same
    # ordering by construction from .FAMILIES).
    with pytest.raises(
        ValueError,
        match=(
            r"^unknown family 'logistic'; expected one of gaussian, binomial, poisson, "
            r"gamma, negativebinomial, inversegaussian$"
        ),
    ):
        glmm.fit(DATA, "y ~ x", "logistic")


def test_link_not_offered_message_matches_the_r_port():
    with pytest.raises(
        ValueError,
        match=r"^family 'binomial' does not support link 'identity'; expected one of logit, probit, cloglog$",
    ):
        glmm.fit(DATA, "y ~ x", "binomial", link="identity")


@pytest.mark.parametrize("formula", ["y ~ .", "y ~ . - x", "y ~ x * ."])
def test_dot_formula_raises(formula):
    with pytest.raises(ValueError, match="'.' is not supported"):
        glmm.fit(DATA, formula)


def test_intercept_suppressed_re_term_raises():
    # Not pre-checked client-side: the shared Rust parser already raises a
    # specific message for this (RandomInterceptSuppressionUnsupported), and
    # it reaches both ports unchanged.
    with pytest.raises(ValueError, match="intercept suppression"):
        glmm.fit(DATA, "y ~ x + (0 + x | g)")


@pytest.mark.parametrize("link", ["cloglog", "probit"])
def test_binomial_glm_link_fits(link):
    result = glmm.fit(FIT_DATA, "y_bin ~ x", "binomial", link=link)
    assert result.converged
    assert len(result.beta) == 2
    assert result.dispersion == 1.0


def test_inversegaussian_mixed_raises():
    # GLM-only family: a mixed formula must be a clean Python error,
    # never a kernel panic.
    with pytest.raises(ValueError, match="GLM-only"):
        glmm.fit(DATA, "y ~ x + (1 | g)", "inversegaussian")


def test_inversegaussian_glm_fits():
    result = glmm.fit(FIT_DATA, "y_invgauss ~ x", "inversegaussian")
    assert result.converged
    assert result.dispersion > 0
    result_inv_sq = glmm.fit(FIT_DATA, "y_invgauss ~ x", "inversegaussian", link="inverse_squared")
    assert result_inv_sq.converged


def test_inversegaussian_dispersion_estimate_is_accepted():
    # "estimate" is the family default for a phi family; it must be stripped
    # to None rather than reaching the kernel as a string.
    result = glmm.fit(FIT_DATA, "y_invgauss ~ x", "inversegaussian", dispersion="estimate")
    assert result.converged


def test_wald_se_invalid_raises():
    with pytest.raises(ValueError, match="wald_se"):
        glmm.fit(DATA, "y ~ x", wald_se="observed")


def test_wald_se_rx_reaches_the_kernel():
    # "rx" bypasses the joint Hessian entirely: unlike the wald_se="hessian"
    # default, stddev_se is never filled on this route (Fit.stddev_se's own
    # docstring: "NaN where unavailable").
    result = glmm.fit(FIT_DATA, "y_bin ~ x + (1 | g)", "binomial", wald_se="rx")
    assert result.converged
    assert _np.all(_np.isfinite(result.se))
    assert _np.all(_np.isnan(result.stddev_se))


@pytest.mark.parametrize("nagq", [0, 2, 4, 26, 27, -1, 1.0])
def test_nagq_invalid_raises(nagq):
    with pytest.raises(ValueError, match="nagq"):
        glmm.fit(DATA, "y ~ x + (1 | g)", "binomial", nagq=nagq)


def test_nagq_max_odd_fits():
    result = glmm.fit(FIT_DATA, "y_bin ~ x + (1 | g)", "binomial", nagq=25)
    assert result.converged


@pytest.mark.parametrize(
    ("family", "response"),
    [("negativebinomial", "y_pois"), ("gamma", "y_gamma")],
)
def test_nagq_on_negbin_and_gamma_fits_quadrature(family, response):
    # NB's theta sits outside the AGQ integral, and Gamma's phi enters it only
    # as a weight on the deviance (the rest of its log-density sits outside),
    # so both families take nagq>1 like binomial/Poisson: no warn-and-strip,
    # and the fit reports the node count it ran.
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = glmm.fit(FIT_DATA, f"{response} ~ x + (1 | g)", family, nagq=7)
    assert not any("nagq" in str(w.message) for w in caught), [str(w.message) for w in caught]
    assert result.nagq == 7
    assert result.converged


def test_dispersion_on_negativebinomial_warns_then_fits():
    # negbin's distribution param is theta, not phi.
    with pytest.warns(glmm.ArgumentIgnoredWarning, match="dispersion"):
        result = glmm.fit(FIT_DATA, "y_pois ~ x", "negativebinomial", dispersion=1.5)
    assert result.converged


def test_quasi_on_glm_poisson_is_a_kernel_gap():
    # Non-mixed: dispersion reaches the family check un-stripped, and
    # quasi-Poisson has no kernel implementation yet.
    with pytest.raises(NotImplementedError, match="quasi-likelihood"):
        glmm.fit(FIT_DATA, "y_pois ~ x", "poisson", dispersion="estimate")


def test_dispersion_bad_value_message_matches_the_r_port():
    with pytest.raises(
        ValueError,
        match=r"^dispersion must be None, 'estimate', or a number, got 'pearson'$",
    ):
        glmm.fit(DATA, "y ~ x", "gamma", dispersion="pearson")


def test_dispersion_bool_raises():
    # bool is an int subclass — must not pass as a numeric dispersion.
    with pytest.raises(ValueError, match="dispersion"):
        glmm.fit(DATA, "y ~ x", "gamma", dispersion=True)


def test_init_theta_on_negbin_is_a_kernel_gap():
    # init_theta=<float> has no kernel hook (no public theta_seed on fit_warm);
    # only the default init_theta=None cold-start search is supported.
    with pytest.raises(NotImplementedError, match="init_theta"):
        glmm.fit(FIT_DATA, "y_pois ~ x", "negativebinomial", init_theta=1.5)


def test_init_theta_and_warm_start_theta_are_independent(recwarn):
    # The collision: `init_theta` (negative-binomial shape) and
    # `warm_start["theta"]` (RE Cholesky vector) are unrelated knobs that may
    # legally appear in one call. `init_theta` on a Gaussian is inapplicable and
    # strips with a warning; the warm start still takes effect.
    result = glmm.fit(
        FIT_DATA,
        "y_gauss ~ x + (1 | g)",
        warm_start={"beta": [0.0, 0.0], "theta": [1.0]},
        init_theta=2.0,
    )
    assert result.converged
    assert any("init_theta" in str(w.message) for w in recwarn)


def test_warm_start_not_dict_message_matches_the_r_port():
    with pytest.raises(
        TypeError,
        match=(
            r"^warm_start must be a dict with keys 'beta' and/or 'theta' \(theta is the "
            r"random-effect Cholesky vector, not the negative-binomial shape - that is "
            r"init_theta\), got list$"
        ),
    ):
        glmm.fit(DATA, "y ~ x", warm_start=[0.0, 0.0])


def test_clean_call_emits_no_warnings_and_fits():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        result = glmm.fit(
            FIT_DATA,
            "y_pois ~ x + (1 | g)",
            "poisson",
            warm_start={"beta": [0.0, 0.0], "theta": [1.0]},
        )
    assert result.converged


@pytest.mark.parametrize("arg", ["weights", "offset"])
def test_wrong_length_weights_or_offset_is_a_plain_valueerror(arg):
    with pytest.raises(ValueError, match=rf"^{arg} must be a numeric array with one entry per row"):
        glmm.fit(DATA, "y ~ x", **{arg: [1.0, 2.0]})


def test_missing_formula_column_is_reported_even_with_wrong_length_weights():
    data = {"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]}
    with pytest.raises(ValueError, match=r"^column\(s\) not found in data: z$"):
        glmm.fit(data, "y ~ x + z", weights=[1.0, 2.0])


def test_missing_formula_column_with_no_weights_is_still_reported():
    data = {"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]}
    with pytest.raises(ValueError, match=r"^column\(s\) not found in data: z$"):
        glmm.fit(data, "y ~ x + z")


def test_missing_column_through_a_transform_names_the_underlying_column():
    # log(z): "z" is missing, not the literal spelling "log(z)".
    data = {"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]}
    with pytest.raises(ValueError, match=r"^column\(s\) not found in data: z$"):
        glmm.fit(data, "y ~ x + log(z)")


def test_one_level_character_factor_in_the_fixed_part_raises():
    data = {"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0], "f": ["z", "z", "z"]}
    with pytest.raises(
        ValueError, match=r"contrasts can be applied only to factors with 2 or more levels"
    ):
        glmm.fit(data, "y ~ f + x")


def test_one_level_bool_factor_in_the_fixed_part_raises():
    # A bool column crosses as a FALSE/TRUE factor (item 1); with every value
    # the same, that factor has one level, same as a one-level character one.
    data = {"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0], "b": [True, True, True]}
    with pytest.raises(
        ValueError, match=r"contrasts can be applied only to factors with 2 or more levels"
    ):
        glmm.fit(data, "y ~ b + x")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None])
@pytest.mark.parametrize("arg", ["weights", "offset"])
def test_non_finite_weights_or_offset_is_a_plain_valueerror(arg, bad, capfd):
    # Checked before the native call, the way the R port checks them, so the
    # kernel's entry assert never fires and prints a Rust panic to stderr.
    kwargs = {arg: [1.0, bad, 1.0]}
    with pytest.raises(ValueError, match=rf"^{arg} must be finite"):
        glmm.fit(DATA, "y ~ x", **kwargs)
    assert "panicked" not in capfd.readouterr().err


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_weights_is_a_plain_valueerror(bad):
    with pytest.raises(ValueError, match=r"^weights must be positive"):
        glmm.fit(DATA, "y ~ x", weights=[1.0, bad, 1.0])


def test_nan_weight_on_a_row_dropped_for_na_is_still_rejected():
    # Validated against data's original rows, before the NA row drop, as R's
    # fastglmm() does: a NaN weight is a bad argument, not a missing value.
    data = {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, float("nan"), 3.0]}
    with pytest.raises(ValueError, match=r"^weights must be finite"):
        glmm.fit(data, "y ~ x", weights=[1.0, 2.0, float("nan"), 4.0])


def test_unused_column_with_an_unconvertible_dtype_is_never_touched():
    # A column the formula does not name must not even be read: a value that
    # crashes both the numeric and factor conversions (float() and str()
    # both "work" on it, but a real datetime/object column with no sane
    # scalar form would not) is fine to leave in `data` as long as the
    # formula never mentions it.
    class Unconvertible:
        def __iter__(self):
            raise AssertionError("an unused column must never be iterated")

    data = {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 2.0, 3.0], "stamp": Unconvertible()}
    result = glmm.fit(data, "y ~ x")
    assert result.converged


def test_column_named_like_a_transform_function_is_not_treated_as_used():
    # `y ~ log(x)` reads "x" through the log transform; a data column
    # literally named "log" must not be pulled in just because that word
    # appears in the formula text (it would crash if it ever reached the
    # numeric/factor conversion).
    data = {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "log": [datetime.date(2020, 1, 1)] * 6,
    }
    result = glmm.fit(data, "y ~ log(x)")
    assert result.converged
    assert result.names == ["(Intercept)", "log(x)"]


def test_column_named_like_a_transform_function_keeps_its_own_missing_values():
    # Same name collision, but the unused "log" column has missing values of
    # its own — a false positive here would drop rows the formula has no
    # reason to drop.
    data = {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "log": ["a", "b", None, "d", "e", "f"],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = glmm.fit(data, "y ~ log(x)")
    assert result.nobs == 6


def test_non_ascii_column_name_is_used():
    # `_formula_columns` asks the Rust parser's own (Unicode-aware)
    # identifier grammar, so a column name outside ASCII keeps working.
    data = {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "café": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    result = glmm.fit(data, "y ~ café")
    assert result.converged
    assert result.names == ["(Intercept)", "café"]


def test_all_rows_missing_in_a_used_column_is_a_plain_error():
    data = {"y": [1.0, 2.0, 3.0], "x": [float("nan"), float("nan"), float("nan")]}
    with pytest.raises(ValueError, match="no rows left to fit"):
        glmm.fit(data, "y ~ x")


def test_bad_formula_syntax_surfaces_the_parse_error_not_an_unrelated_column_error():
    # `glmm`'s `||` uncorrelated-random-effects syntax is not supported, and is
    # caught by fit()'s own formula-shape check before anything else runs. An
    # unrelated column the formula never uses (a date, which crashes both the
    # numeric and factor conversions) must not get a chance to raise first
    # with an error that has nothing to do with the formula.
    data = {
        "y": [1.0, 2.0, 3.0, 4.0],
        "x": [0.0, 1.0, 2.0, 3.0],
        "g": [0, 0, 1, 1],
        "stamp": [datetime.date(2020, 1, 1)] * 4,
    }
    with pytest.raises(ValueError, match="full RE correlation structure"):
        glmm.fit(data, "y ~ (x || g)")


def test_weights_and_offset_are_dropped_in_lockstep_with_na_rows():
    # weights=/offset= are separate arrays aligned to data's original rows;
    # dropping a row for a missing `x` must drop the same position from both,
    # or they silently misalign against the surviving rows.
    data = {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, float("nan"), 3.0]}
    with pytest.warns(glmm.RowsDroppedWarning):
        result = glmm.fit(data, "y ~ x", weights=[1.0, 2.0, 3.0, 4.0], offset=[0.0, 0.0, 5.0, 0.0])
    assert result.converged
    assert result.nobs == 3
    assert list(result.weights) == [1.0, 2.0, 4.0]  # row index 2 (weight 3.0) is the one dropped


def test_pyarrow_null_in_a_used_numeric_column_drops_its_row():
    # A pyarrow Array/ChunkedArray iterates as Scalar wrappers, not bare
    # None/float; `float()` on a null Scalar raises TypeError instead of
    # comparing as missing, so `_is_na` needs the column read through
    # `.to_pylist()` first to see a real `None`.
    pa = pytest.importorskip("pyarrow")
    t = pa.table({"y": [1.0, 2.0, 3.0, 4.0], "x": pa.array([0.0, 1.0, None, 3.0])})
    with pytest.warns(glmm.RowsDroppedWarning, match=r"Dropped 1 of 4 row"):
        result = glmm.fit(t, "y ~ x")
    assert result.converged
    assert result.nobs == 3


def test_pyarrow_null_in_a_used_string_column_drops_its_row():
    pa = pytest.importorskip("pyarrow")
    t = pa.table(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "f": pa.array(["a", "b", None, "a", "b", "a"])}
    )
    with pytest.warns(glmm.RowsDroppedWarning, match=r"Dropped 1 of 6 row"):
        result = glmm.fit(t, "y ~ f")
    assert result.converged
    assert result.nobs == 5


def test_polars_null_in_a_used_column_drops_its_row():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, None, 3.0]})
    with pytest.warns(glmm.RowsDroppedWarning, match=r"Dropped 1 of 4 row"):
        result = glmm.fit(df, "y ~ x")
    assert result.converged
    assert result.nobs == 3
