import numpy as np
import pytest

import glmm

_rng = np.random.default_rng(7)
_N = 60


def _data():
    return {
        "y": _rng.normal(size=_N).tolist(),
        "v": _rng.normal(size=_N).tolist(),
        "age": _rng.normal(size=_N).tolist(),
        "condition": _rng.choice(["a", "b", "c"], _N).tolist(),
        "gender": _rng.choice(["f", "m"], _N).tolist(),
        "g": _rng.choice(["g1", "g2", "g3"], _N).tolist(),
    }


def test_design_is_the_fixed_design_of_the_fit():
    data = _data()
    x, names = glmm.model_matrix(data, "~ age + gender * condition")
    fit = glmm.fit(data, "y ~ age + gender * condition")
    assert x.shape == (_N, len(fit.names))
    assert x.dtype == np.float64 and x.flags.writeable and x.flags.c_contiguous
    assert names == fit.names
    np.testing.assert_allclose(x @ fit.beta, fit.fitted, rtol=1e-12, atol=1e-12)


def test_intercept_only_has_one_row_per_data_row():
    x, names = glmm.model_matrix({"age": [1.0, 2.0, 3.0]}, "~ 1")
    np.testing.assert_array_equal(x, np.ones((3, 1)))
    assert names == ["(Intercept)"]


def test_an_unused_column_of_another_length_is_ignored_as_fit_ignores_it():
    data = _data()
    x, names = glmm.model_matrix(data, "~ age + gender")
    ragged = {**data, "extra": [1.0, 2.0]}
    assert glmm.fit(ragged, "y ~ age + gender").names == names
    got, got_names = glmm.model_matrix(ragged, "~ age + gender")
    np.testing.assert_array_equal(got, x)
    assert got_names == names


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("age + gender", "needs ~"),
        ("y = age", "needs ~"),
        ("~ age + (1 | g)", "no random-effect terms"),
        ("~ age + offset(v)", "no offset"),
    ],
)
def test_formula_refusals(formula, message):
    with pytest.raises(ValueError, match=message):
        glmm.model_matrix(_data(), formula)


@pytest.mark.parametrize(("formula", "column"), [("~ age", "age"), ("y ~ age", "y")])
def test_a_missing_value_raises(formula, column):
    data = _data()
    data[column][4] = None
    with pytest.raises(ValueError, match=r"1 of 60 row\(s\) have a missing value"):
        glmm.model_matrix(data, formula)


def test_an_absent_response_column_raises():
    data = _data()
    del data["y"]
    with pytest.raises(ValueError, match="not found in data: y"):
        glmm.model_matrix(data, "y ~ age")
