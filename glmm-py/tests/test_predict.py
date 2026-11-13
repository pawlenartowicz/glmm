"""Fit.predict, Fit.simulate and the conditional variances in ranef_blocks:
what crosses the port boundary and how it is worded. The numbers themselves
are checked in the crate (RULE 6)."""

import pickle
import warnings

import numpy as np
import pytest

import glmm

_rng = np.random.default_rng(20261026)
_K, _PER = 12, 10
_N = _K * _PER
_G = np.repeat(np.arange(_K), _PER)
_H = np.tile(np.arange(_PER), _K)
_X = _rng.normal(size=_N)
_O = 0.2 * _rng.normal(size=_N)
_U = _rng.normal(scale=0.8, size=_K)
_V = _rng.normal(scale=0.5, size=_PER)
_S = _rng.normal(scale=0.4, size=_PER)


def _data(**columns):
    d = {
        "y": (1.0 + 0.5 * _X + _O + _U[_G] + 0.5 * _rng.normal(size=_N)).tolist(),
        "yb": _rng.binomial(1, 1.0 / (1.0 + np.exp(-(-0.2 + 0.8 * _X + _U[_G]))))
        .astype(float)
        .tolist(),
        "yc": (1.0 + 0.5 * _X + _U[_G] + _V[_H] + 0.5 * _rng.normal(size=_N)).tolist(),
        "ys": (1.0 + 0.5 * _X + _U[_G] + _S[_H] * _X + 0.5 * _rng.normal(size=_N)).tolist(),
        "x": _X.tolist(),
        "o": _O.tolist(),
        "g": [f"g{i:02d}" for i in _G],
        "h": [f"h{i}" for i in _H],
    }
    d.update(columns)
    return d


D = _data()


def quiet_fit(formula, data=D, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return glmm.fit(data, formula, **kw)


def _rows(d, keep):
    return {k: [v[i] for i in keep] for k, v in d.items()}


def test_predict_on_the_training_rows_at_the_response_scale_is_fitted():
    for formula, family in (("y ~ x + (1 | g)", "gaussian"), ("yb ~ x + (1 | g)", "binomial")):
        m = quiet_fit(formula, family=family)
        np.testing.assert_allclose(m.predict(D, type="response"), m.fitted, rtol=1e-10)


def test_the_default_scale_is_the_link():
    m = quiet_fit("yb ~ x + (1 | g)", family="binomial")
    eta = m.predict(D)
    np.testing.assert_allclose(1.0 / (1.0 + np.exp(-eta)), m.fitted, rtol=1e-10)
    np.testing.assert_array_equal(eta, m.predict(D, type="link"))


def test_population_mode_is_x_beta_plus_offset():
    m = quiet_fit("y ~ x + offset(o) + (1 | g)")
    x, _ = glmm.model_matrix(D, "~ x")
    np.testing.assert_allclose(m.predict(D, re_form="NA"), x @ m.beta + _O, rtol=1e-12)
    # No grouping column at all: population mode needs none.
    no_g = {k: v for k, v in D.items() if k != "g"}
    np.testing.assert_allclose(m.predict(no_g, re_form="NA"), x @ m.beta + _O, rtol=1e-12)


def test_se_fit_returns_the_fit_its_se_and_the_interval():
    m = quiet_fit("y ~ x + (1 | g)")
    out = m.predict(D, se_fit=True)
    assert list(out) == ["fit", "se_fit", "lower", "upper"]
    np.testing.assert_array_equal(out["fit"], m.predict(D))
    assert np.all(out["se_fit"] > 0)
    assert np.all((out["lower"] < out["fit"]) & (out["fit"] < out["upper"]))
    wide = m.predict(D, se_fit=True, level=0.99)
    assert np.all(wide["upper"] - wide["lower"] > out["upper"] - out["lower"])


def test_future_bounds_on_a_glmm():
    m = quiet_fit("yb ~ x + (1 | g)", family="binomial")
    out = m.predict(D, future=True)
    assert list(out) == ["fit", "future_lower", "future_upper"]
    assert set(np.unique(out["future_lower"])) <= {0.0, 1.0}
    with pytest.raises(ValueError, match="future applies only to a non-Gaussian mixed model"):
        quiet_fit("y ~ x + (1 | g)").predict(D, future=True)


def test_an_unseen_grouping_level_warns_and_gets_the_population_prediction():
    m = quiet_fit("y ~ x + (1 | g)")
    new = _rows(D, [0, 1, 15])
    new["g"] = ["g00", "zz", "zz"]
    with pytest.warns(glmm.NewGroupingLevelsWarning, match=r"\bg\b") as caught:
        got = m.predict(new)
    assert issubclass(glmm.NewGroupingLevelsWarning, glmm.DiagnosticWarning)
    assert "rows 1, 2" in str(caught[0].message)
    pop = m.predict(new, re_form="NA")
    np.testing.assert_array_equal(got[1:], pop[1:])
    assert got[0] != pop[0]
    with pytest.raises(ValueError, match="did not see"):
        m.predict(new, allow_new_levels=False)


def _absent_cell_data():
    """36 rows over the cells a:B, b:A and b:B of f1 by f2; a:A has no row."""
    cells = [("a", "B"), ("b", "A"), ("b", "B")]
    return {
        "y": [float(1 + (7 * i) % 5 + 2 * (i % 3)) for i in range(36)],
        "f1": [cells[i % 3][0] for i in range(36)],
        "f2": [cells[i % 3][1] for i in range(36)],
        "g": [f"g{(i // 3) % 6}" for i in range(36)],
    }


def test_a_row_the_fit_cannot_estimate_warns_and_predicts_nan():
    m = quiet_fit("y ~ 0 + f1:f2 + (1 | g)", data=_absent_cell_data())
    new = {"f1": ["a", "b"], "f2": ["A", "B"]}
    match = "Row 0 is not estimable from this rank-deficient fit"
    with pytest.warns(glmm.RowsNotEstimableWarning, match=match):
        got = m.predict(newdata=new, re_form="NA")
    assert np.isnan(got[0]) and np.isfinite(got[1])


def test_a_fixed_factor_level_the_fit_did_not_see_is_refused():
    m = quiet_fit("y ~ x + h + (1 | g)")
    new = _rows(D, [0, 1])
    new["h"] = ["h0", "h_new"]
    with pytest.raises(ValueError, match="h_new"):
        m.predict(new)


def test_bad_arguments_raise():
    m = quiet_fit("y ~ x + (1 | g)")
    with pytest.raises(ValueError, match="type must be") as bad:
        m.predict(D, type="probability")
    assert bad.value.code == "invalid_argument"
    with pytest.raises(ValueError, match="re_form must be"):
        m.predict(D, re_form="~ (1 | g)")
    with pytest.raises(ValueError, match="level must be in"):
        m.predict(D, level=1.5)
    with pytest.raises(ValueError, match="grouping columns"):
        m.predict({k: v for k, v in D.items() if k != "g"})


def test_simulate_is_seeded_and_has_one_row_per_fitted_row():
    y = list(D["y"])
    y[3] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = glmm.fit(_data(y=y), "y ~ x + (1 | g)")
    a = m.simulate(nsim=3, seed=7)
    assert a.shape == (m.nobs, 3) == (_N - 1, 3)
    np.testing.assert_array_equal(a, m.simulate(nsim=3, seed=7))
    assert not np.array_equal(a, m.simulate(nsim=3, seed=8))
    assert not np.array_equal(a[:, 0], a[:, 1])
    assert m.simulate().shape == (_N - 1, 1)
    with pytest.raises(ValueError, match="nsim"):
        m.simulate(nsim=0)


def test_ranef_blocks_carry_the_conditional_variances():
    m = quiet_fit("ys ~ x + (1 + x | g)")
    (block,) = m.ranef_blocks
    levels, q = block["values"].shape
    assert block["postvar"].shape == (levels, q, q)
    np.testing.assert_array_equal(block["postvar"], np.transpose(block["postvar"], (0, 2, 1)))
    np.testing.assert_array_equal(
        block["condsd"], np.sqrt(np.diagonal(block["postvar"], axis1=1, axis2=2))
    )


def test_a_fit_without_conditional_variances_raises_a_clear_error():
    # Two groupings with slopes route to the sparse solver, which forms no
    # conditional variances.
    m = quiet_fit("ys ~ x + (1 + x | g) + (1 + x | h)")
    assert all(b["postvar"] is None and b["condsd"] is None for b in m.ranef_blocks)
    assert len(m.predict(D)) == _N
    with pytest.raises(ValueError, match="no conditional variances"):
        m.predict(D, se_fit=True)
    assert np.all(m.predict(D, re_form="NA", se_fit=True)["se_fit"] > 0)


def test_condvar_false_skips_the_conditional_variances():
    on = quiet_fit("ys ~ x + (1 + x | g)")
    off = quiet_fit("ys ~ x + (1 + x | g)", condvar=False)
    (block,) = off.ranef_blocks
    assert block["postvar"] is None and block["condsd"] is None
    np.testing.assert_array_equal(block["values"], on.ranef_blocks[0]["values"])
    with pytest.raises(ValueError, match="condvar off"):
        off.predict(D, se_fit=True)


def test_a_crossed_fit_round_trips_its_stored_covariances():
    m = quiet_fit("yc ~ x + (1 | g) + (1 | h)")
    back = pickle.loads(pickle.dumps(m))
    want = m.predict(D, se_fit=True)["se_fit"]
    np.testing.assert_array_equal(back.predict(D, se_fit=True)["se_fit"], want)
    np.testing.assert_array_equal(back.simulate(nsim=2, seed=1), m.simulate(nsim=2, seed=1))


def test_a_nested_slope_model_predicts_the_same_in_either_order():
    # Children `n` nested in `g`, 2 or 3 per parent; the slope-last spelling
    # is nested by the fit after it moves `g` first, the slope-first one by
    # the formula front end.
    rng = np.random.default_rng(20261111)
    kids = [(g, k) for g in range(12) for k in range(2 if g < 6 else 3)]
    rows = [(g, k) for g, k in kids for _ in range(10)]
    x = rng.normal(size=len(rows))
    u = rng.normal(scale=0.8, size=12)
    s = rng.normal(scale=0.4, size=12)
    v = rng.normal(scale=0.5, size=len(kids))
    kid = {gk: i for i, gk in enumerate(kids)}
    yn = [
        1.0 + 0.5 * xi + u[g] + s[g] * xi + v[kid[(g, k)]] + 0.5 * rng.normal()
        for xi, (g, k) in zip(x, rows)
    ]
    d = {
        "yn": yn,
        "x": x.tolist(),
        "g": [f"g{g:02d}" for g, _ in rows],
        "n": [f"n{g:02d}{k}" for g, k in rows],
    }
    last = quiet_fit("yn ~ x + (1 | n) + (1 + x | g)", data=d)
    first = quiet_fit("yn ~ x + (1 + x | g) + (1 | n)", data=d)
    assert last.loglik == first.loglik
    pl, pf = last.predict(d, se_fit=True), first.predict(d, se_fit=True)
    np.testing.assert_allclose(pl["fit"], pf["fit"], rtol=1e-12)
    np.testing.assert_allclose(pl["se_fit"], pf["se_fit"], rtol=1e-12)
    blocks = {b["group"]: b for b in first.ranef_blocks}
    for b in last.ranef_blocks:
        twin = blocks[b["group"]]
        assert b["levels"] == twin["levels"]
        np.testing.assert_array_equal(b["values"], twin["values"])
        np.testing.assert_array_equal(b["postvar"], twin["postvar"])
    back = pickle.loads(pickle.dumps(last))
    np.testing.assert_array_equal(back.predict(d, se_fit=True)["se_fit"], pl["se_fit"])
