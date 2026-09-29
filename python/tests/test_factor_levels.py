"""Factor level order.

Level 0 is the treatment-contrast base, so a column's level order picks the
reference level. A declared order (pandas `Categorical`) must survive to the
fit; a plain string column has none and is sorted lexicographically.
"""

import numpy as np
import pytest

import glmm

# y is a clean function of the level: low->1, med->2, high->3 (+/- 0.05), so the
# intercept IS the base level's mean and naming the wrong base is visible.
_LABELS = ["low", "high", "med", "low", "high", "med"]
_Y = [1.0, 3.0, 2.0, 1.1, 3.1, 2.1]


class _FakeCategorical:
    """Duck-typed stand-in for pandas' Categorical: `glmm.fit` keys on
    `.categories`/`.codes`, not on the pandas type, so this exercises the real
    contract without needing an actual pandas Categorical (CI installs pandas
    and pyarrow too, but this stand-in checks the duck-typed path itself)."""

    def __init__(self, categories, codes):
        self.categories = categories
        self.codes = codes

    def __iter__(self):
        return iter([self.categories[c] for c in self.codes])

    def __len__(self):
        return len(self.codes)


class _FakeArrowList:
    """Minimal `.to_pylist()` stand-in for a pyarrow Array —
    `_levels_and_codes` calls nothing else on `.dictionary`/`.indices`."""

    def __init__(self, values):
        self._values = values

    def to_pylist(self):
        return list(self._values)


class _FakeDictionaryArray:
    """Duck-typed stand-in for a bare pyarrow `DictionaryArray` passed in a
    dict: `glmm.fit`'s `_levels_and_codes` keys on `.dictionary`/`.indices`,
    not on the pyarrow type. A pyarrow Table's column is a `ChunkedArray`
    instead; the real-pyarrow tests below cover that shape."""

    def __init__(self, dictionary, indices):
        self.dictionary = _FakeArrowList(dictionary)
        self.indices = _FakeArrowList(indices)


def test_dictionary_array_sets_the_reference_level():
    data = {"y": _Y, "f": _FakeDictionaryArray(["low", "med", "high"], [0, 2, 1, 0, 2, 1])}
    result = glmm.fit(data, "y ~ f")
    assert result.names == ["(Intercept)", "fmed", "fhigh"]
    assert result.beta[0] == pytest.approx(1.05, abs=1e-6)  # mean of "low"


def test_dictionary_array_missing_index_drops_the_row_like_r_na_omit():
    # A null dictionary index drops that row instead of raising, the same
    # rule pandas' code -1 follows above.
    data = {"y": _Y, "f": _FakeDictionaryArray(["low", "med"], [0, 1, None, 0, 1, 0])}
    with pytest.warns(glmm.RowsDroppedWarning, match=r"Dropped 1 of 6 row"):
        result = glmm.fit(data, "y ~ f")
    assert result.names == ["(Intercept)", "fmed"]
    assert result.nobs == 5


def test_pyarrow_table_dictionary_column_keeps_its_declared_level_order():
    pa = pytest.importorskip("pyarrow")
    f = pa.DictionaryArray.from_arrays(
        pa.array([0, 2, 1, 0, 2, 1], pa.int8()), pa.array(["low", "med", "high"])
    )
    result = glmm.fit(pa.table({"y": _Y, "f": f}), "y ~ f")
    # Sorted, "high" would be the base; the dictionary declares "low" first.
    assert result.names == ["(Intercept)", "fmed", "fhigh"]
    assert result.beta[0] == pytest.approx(1.05, abs=1e-6)  # mean of "low"


def test_pyarrow_multi_chunk_dictionary_column_with_different_dictionaries():
    # Each chunk of a ChunkedArray carries its own dictionary. Rows are
    # _LABELS: chunk 1 codes them against [low, med, high], chunk 2 against
    # [high, low, med]; the declared order is the first chunk's.
    pa = pytest.importorskip("pyarrow")
    c1 = pa.DictionaryArray.from_arrays(
        pa.array([0, 2, 1], pa.int8()), pa.array(["low", "med", "high"])
    )
    c2 = pa.DictionaryArray.from_arrays(
        pa.array([1, 0, 2], pa.int8()), pa.array(["high", "low", "med"])
    )
    t = pa.table({"y": pa.chunked_array([_Y[:3], _Y[3:]]), "f": pa.chunked_array([c1, c2])})
    assert t.column("f").num_chunks == 2
    result = glmm.fit(t, "y ~ f")
    assert result.names == ["(Intercept)", "fmed", "fhigh"]
    assert result.beta[0] == pytest.approx(1.05, abs=1e-6)  # mean of "low"
    assert result.beta[1] == pytest.approx(1.0, abs=1e-6)  # med - low
    assert result.beta[2] == pytest.approx(2.0, abs=1e-6)  # high - low


@pytest.mark.parametrize("null_encoding", ["mask", "encode"])
def test_pyarrow_dictionary_null_drops_the_row_like_r_na_omit(null_encoding):
    # "mask" leaves the null in the indices; "encode" makes it a dictionary
    # value of its own. Either way it is a missing row, not a "None" level.
    pa = pytest.importorskip("pyarrow")
    f = pa.array(["low", "med", None, "low", "med", "low"]).dictionary_encode(
        null_encoding=null_encoding
    )
    with pytest.warns(glmm.RowsDroppedWarning, match=r"Dropped 1 of 6 row"):
        result = glmm.fit(pa.table({"y": _Y, "f": f}), "y ~ f")
    assert result.names == ["(Intercept)", "fmed"]
    assert result.nobs == 5


def test_declared_level_order_sets_the_reference_level():
    data = {
        "y": _Y,
        "f": _FakeCategorical(["low", "med", "high"], [0, 2, 1, 0, 2, 1]),
    }
    result = glmm.fit(data, "y ~ f")
    # Base is "low", so the dummies are the other two IN THE DECLARED ORDER.
    assert result.names == ["(Intercept)", "fmed", "fhigh"]
    assert result.beta[0] == pytest.approx(1.05, abs=1e-6)  # mean of "low"


def test_plain_string_column_still_sorts_lexicographically():
    # No declared order -> R's factor() default. "high" sorts first and becomes
    # the base; this is the pre-existing behavior, preserved as a DEFAULT.
    result = glmm.fit({"y": _Y, "f": _LABELS}, "y ~ f")
    assert result.names == ["(Intercept)", "flow", "fmed"]
    assert result.beta[0] == pytest.approx(3.05, abs=1e-6)  # mean of "high"


def test_missing_value_in_a_plain_string_column_drops_the_row_not_a_level():
    # A missing value must drop that row, the way R's na.action = na.omit
    # does, not get stringified into a spurious "fnan" level
    # (`str(float("nan"))` would otherwise become a dummy).
    labels = list(_LABELS)
    labels[3] = float("nan")  # was "low"
    with pytest.warns(glmm.RowsDroppedWarning, match=r"Dropped 1 of 6 row"):
        result = glmm.fit({"y": _Y, "f": labels}, "y ~ f")
    assert result.names == ["(Intercept)", "flow", "fmed"]  # still just the 3 real levels
    assert result.nobs == 5


def test_missing_value_in_the_first_row_does_not_misclassify_the_column():
    # Numeric-vs-factor classification cannot key off `values[0]` alone: a
    # missing first value (a float NaN) would make a string column look
    # numeric and crash float() on the rest of it.
    labels = list(_LABELS)
    labels[0] = float("nan")  # was "low"
    with pytest.warns(glmm.RowsDroppedWarning):
        result = glmm.fit({"y": _Y, "f": labels}, "y ~ f")
    assert result.names == ["(Intercept)", "flow", "fmed"]
    assert result.nobs == 5


def test_categorical_of_non_strings_is_not_fit_as_numeric():
    # Detection must not rely on `isinstance(values[0], str)`: a categorical of
    # ints would fall through to the numeric branch and fit as ONE continuous
    # slope instead of expanding to dummies.
    data = {"y": _Y, "f": _FakeCategorical([10, 20, 30], [0, 2, 1, 0, 2, 1])}
    result = glmm.fit(data, "y ~ f")
    assert result.names == ["(Intercept)", "f20", "f30"]


def test_missing_category_code_drops_the_row_like_r_na_omit():
    # pandas' code -1 (no matching category) drops that row instead of
    # raising, the same rule a plain string column's missing value follows.
    data = {"y": _Y, "f": _FakeCategorical(["low", "med"], [0, 1, -1, 0, 1, 0])}
    with pytest.warns(glmm.RowsDroppedWarning, match=r"Dropped 1 of 6 row"):
        result = glmm.fit(data, "y ~ f")
    assert result.names == ["(Intercept)", "fmed"]
    assert result.nobs == 5


def test_pandas_categorical_round_trips():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "y": _Y,
            "f": pd.Categorical(_LABELS, categories=["low", "med", "high"], ordered=True),
        }
    )
    result = glmm.fit(df, "y ~ f")
    assert result.names == ["(Intercept)", "fmed", "fhigh"]
    assert result.beta[0] == pytest.approx(1.05, abs=1e-6)

    # The same frame with a plain object dtype has no declared order -> sorted.
    df2 = pd.DataFrame({"y": _Y, "f": _LABELS})
    assert glmm.fit(df2, "y ~ f").names == ["(Intercept)", "flow", "fmed"]


def test_vcov_matches_se_and_is_symmetric():
    # vcov is the full p×p, se is its diagonal.
    result = glmm.fit({"y": _Y, "f": _LABELS}, "y ~ f")
    p = len(result.beta)
    assert result.vcov.shape == (p, p)
    assert np.allclose(np.sqrt(np.diag(result.vcov)), result.se)
    assert np.allclose(result.vcov, result.vcov.T)


# A bool column crosses as a factor with levels FALSE/TRUE, lme4's own coding
# (`factor()`'s lexicographic order), everywhere except the response and the
# offset() column, which stay 0/1 numbers.
_B = [True, False, True, False, True, False]


def test_plain_python_bool_list_crosses_as_a_false_true_factor():
    data = {"y": _Y, "b": list(_B)}
    result = glmm.fit(data, "y ~ b")
    assert result.names == ["(Intercept)", "bTRUE"]


def test_numpy_bool_column_crosses_as_a_false_true_factor():
    data = {"y": _Y, "b": np.array(_B, dtype=np.bool_)}
    result = glmm.fit(data, "y ~ b")
    assert result.names == ["(Intercept)", "bTRUE"]


def test_bool_column_with_no_intercept_gives_both_levels():
    data = {"y": _Y, "b": np.array(_B, dtype=np.bool_)}
    result = glmm.fit(data, "y ~ 0 + b")
    assert result.names == ["bFALSE", "bTRUE"]


def test_bool_column_in_an_interaction_keeps_both_levels():
    # `x:b` is the only term: b has no earlier term to fall back on for
    # marginality, so it keeps its full indicator set (R's model.matrix rule).
    data = {"y": _Y, "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0], "b": list(_B)}
    result = glmm.fit(data, "y ~ x:b")
    assert result.names == ["(Intercept)", "x:bFALSE", "x:bTRUE"]


def test_bool_grouping_factor_gets_false_true_levels():
    g = [False, False, False, True, True, True]
    data = {"y": _Y, "g": g}
    result = glmm.fit(data, "y ~ 1 + (1 | g)")
    block = next(b for b in result.ranef_blocks if b["group"] == "g")
    assert block["levels"] == ["FALSE", "TRUE"]


def test_bool_response_stays_a_0_1_number_not_a_factor():
    # The response never becomes a FALSE/TRUE factor, bool-valued or not:
    # glmm.fit would otherwise reject it as a non-numeric response.
    y_bool = [True, False, True, False, True, False]
    result = glmm.fit({"y": y_bool, "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]}, "y ~ x", "binomial")
    assert result.y.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_bool_offset_stays_a_0_1_number_not_a_factor():
    data = {
        "y": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        "o": list(_B),
    }
    result = glmm.fit(data, "y ~ x + offset(o)")
    assert result.names == ["(Intercept)", "x"]


def test_pandas_bool_dtype_crosses_as_a_false_true_factor():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"y": _Y, "b": pd.array(_B, dtype="bool")})
    result = glmm.fit(df, "y ~ b")
    assert result.names == ["(Intercept)", "bTRUE"]


def test_pandas_nullable_boolean_dtype_crosses_as_a_false_true_factor():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"y": _Y, "b": pd.array(_B, dtype="boolean")})
    result = glmm.fit(df, "y ~ b")
    assert result.names == ["(Intercept)", "bTRUE"]


def test_pandas_nullable_boolean_missing_value_drops_the_row():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {"y": _Y, "b": pd.array([True, False, None, True, False, True], dtype="boolean")}
    )
    with pytest.warns(glmm.RowsDroppedWarning, match=r"Dropped 1 of 6 row"):
        result = glmm.fit(df, "y ~ b")
    assert result.nobs == 5


def test_pyarrow_bool_array_crosses_as_a_false_true_factor():
    pa = pytest.importorskip("pyarrow")
    data = {"y": _Y, "b": pa.array(_B, type=pa.bool_())}
    result = glmm.fit(data, "y ~ b")
    assert result.names == ["(Intercept)", "bTRUE"]
