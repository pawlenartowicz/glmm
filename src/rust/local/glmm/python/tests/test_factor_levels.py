"""Factor level order — §6 of the python-port-bugs spec.

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
    contract with no pandas dependency (CI installs only pytest)."""

    def __init__(self, categories, codes):
        self.categories = categories
        self.codes = codes

    def __iter__(self):
        return iter([self.categories[c] for c in self.codes])

    def __len__(self):
        return len(self.codes)


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


def test_categorical_of_non_strings_is_not_fit_as_numeric():
    # The old detection was `isinstance(values[0], str)`, so a categorical of
    # ints fell through to the numeric branch and was fit as ONE continuous
    # slope instead of expanding to dummies.
    data = {"y": _Y, "f": _FakeCategorical([10, 20, 30], [0, 2, 1, 0, 2, 1])}
    result = glmm.fit(data, "y ~ f")
    assert result.names == ["(Intercept)", "f20", "f30"]


def test_missing_category_code_is_rejected():
    data = {"y": _Y, "f": _FakeCategorical(["low", "med"], [0, 1, -1, 0, 1, 0])}
    with pytest.raises(ValueError, match="missing values"):
        glmm.fit(data, "y ~ f")


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
    # §4: vcov is the full p×p, se is its diagonal.
    result = glmm.fit({"y": _Y, "f": _LABELS}, "y ~ f")
    p = len(result.beta)
    assert result.vcov.shape == (p, p)
    assert np.allclose(np.sqrt(np.diag(result.vcov)), result.se)
    assert np.allclose(result.vcov, result.vcov.T)
