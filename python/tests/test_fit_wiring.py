import warnings

import numpy as np
import pytest

import glmm

_rng = np.random.default_rng(1)
_N = 300
_GROUPS = np.repeat(np.arange(30), _N // 30)
_X = _rng.normal(size=_N)


def _data(y):
    return {"x": _X.tolist(), "g": [f"g{i}" for i in _GROUPS.tolist()], "y": y}


def test_design_summary_is_kept_on_an_unweighted_random_intercept_fit_only():
    y = 1.0 + 2.0 * _X + np.sin(_GROUPS) + _rng.normal(scale=0.5, size=_N)
    data = _data(y.tolist())
    ds = glmm.fit(data, "y ~ x + (1 | g)").design_summary
    k, p = 30, 2
    assert [len(ds[name]) for name in ("counts", "s", "c")] == [k, p * k, p * p]
    assert all(isinstance(v, np.ndarray) for v in ds.values())
    np.testing.assert_array_equal(ds["counts"], np.full(k, _N // k, dtype=float))
    assert glmm.fit(data, "y ~ x").design_summary is None
    weighted = glmm.fit(data, "y ~ x + (1 | g)", weights=np.linspace(0.5, 1.5, _N))
    assert weighted.design_summary is None


def test_native_fit_refuses_bytes_that_are_not_whole_values():
    def native(x, weights):
        return glmm._native.fit(
            "y ~ x",
            {"x": x, "y": np.ones(4).tobytes()},
            {},
            "gaussian",
            "identity",
            "hessian",
            1,
            None,
            weights,
            None,
            None,
            False,
            True,
            None,
        )

    with pytest.raises(ValueError, match=r'column "x": 31 bytes is not a whole number of 8-byte'):
        native(b"\0" * 31, None)
    with pytest.raises(ValueError, match=r"weights: 7 bytes is not a whole number of 8-byte"):
        native(np.arange(4.0).tobytes(), b"\0" * 7)


def test_per_row_fields_are_writable_float_arrays():
    y = 1.0 + 2.0 * _X + _rng.normal(scale=0.5, size=_N)
    m = glmm.fit(_data(y.tolist()), "y ~ x + (1 | g)", weights=np.ones(_N))
    for name in ("fitted", "y", "ranef", "weights"):
        a = getattr(m, name)
        assert a.dtype == np.float64 and a.flags.writeable and a.flags.owndata, name
    o = glmm.fit(_data(y.tolist()), "y ~ x", offset=0.1 * _X).offset
    assert o.dtype == np.float64 and o.flags.writeable and o.flags.owndata
    np.testing.assert_array_equal(o, 0.1 * _X)
    assert m.ranef_levels.dtype == np.dtype(int)
    assert m.ranef_levels.tolist() == [30]


def test_diagnostics_exposes_the_fields_and_the_flags_still_read_off_fit():
    # Noise centred within each group, so the group variance pins by design
    # rather than by the draw order of the shared generator.
    noise = np.random.default_rng(3).normal(scale=0.5, size=_N)
    for lvl in range(30):
        rows = _GROUPS == lvl
        noise[rows] -= noise[rows].mean()
    y = 1.0 + 2.0 * _X + noise
    result = glmm.fit(_data(y.tolist()), "y ~ x + (1 | g)")
    d = result.diagnostics
    assert set(d) == {"converged", "singular", "aliased", "boundary", "pinned", "notes"}
    # The three moved flags still read straight off the Fit, and off one storage.
    assert result.converged is d["converged"]
    assert result.singular is d["singular"]
    assert result.aliased is d["aliased"]
    assert result.converged
    # This fixture has no real group effect, so the one variance
    # component pins — assert that, not the set of every value the mapper
    # can emit. `singular` gets its own value line for the same reason
    # `converged` does: the identity check above would hold against any value.
    assert result.singular
    assert d["boundary"] == "at_boundary"
    assert d["pinned"] == [[True]]
    assert d["notes"] == []


def test_pinned_detail_survives_a_q2_grouping_where_the_stddev_is_not_zero():
    # On a q>=2 grouping the pin fixes the Cholesky diagonal while the reported
    # stddev inherits the off-diagonal, so it lands at ~1e-10, not at 0.0.
    # Reading `pinned` names the component.
    #
    # Design mirrors the Rust fit::common_tests::diagnostics_pinned_aligns_with_varcorr_blocks:
    # 16 clusters x 16 rows, a real fixed slope but zero cluster-varying slope,
    # residual a +/-0.8 period-4 quadrature block against x's +/-1 alternation so
    # every cluster has sum(resid) = 0 and sum(x*resid) = 0 exactly. The REML
    # slope-variance MLE is then 0 (the slope component pins) while the planted
    # cluster intercepts keep the intercept component interior.
    nc, per = 16, 16
    u0 = 0.6 * np.random.default_rng(5).normal(size=nc)
    xs, ys, gs = [], [], []
    for c in range(nc):
        for k in range(per):
            x1 = 1.0 if k % 2 == 0 else -1.0
            e = 0.8 if (k // 2) % 2 == 0 else -0.8
            xs.append(x1)
            ys.append(0.5 + 0.4 * x1 + u0[c] + e)
            gs.append(f"g{c}")
    with pytest.warns(
        glmm.SingularFitWarning,
        match=r"Affected: .*x in g",
    ):
        result = glmm.fit({"y": ys, "x": xs, "g": gs}, "y ~ x + (1 + x | g)")

    assert result.singular
    assert result.diagnostics["boundary"] == "at_boundary"
    # The slope component is the pinned one, aligned with the varcorr block.
    assert result.diagnostics["pinned"] == [[False, True]]
    sd, _corr = result.stddev_corr(0)
    assert len(result.diagnostics["pinned"][0]) == len(sd)
    # The pinned slot is negligible against its sibling but is NOT exactly 0.
    assert sd[1] != 0.0
    assert sd[1] / sd[0] < 1e-6


def test_ill_conditioned_note_warns_under_its_own_category():
    # x = [1, a, a+delta] with delta living entirely on rows weighted 1e-11:
    # full-rank raw, so nothing is dropped, and near-singular once weighted —
    # the case only the fit can see (mirrors the Rust
    # diagnostics_ill_conditioned_note_through_fit_cold).
    n, split = 60, 40
    a = [((i * 13) % 17) - 8.0 for i in range(n)]
    b = [a[i] + (0.0 if i < split else 1.0) for i in range(n)]
    y = [0.5 + 1.3 * a[i] + 0.477 * b[i] + ((i % 3) - 1.0) for i in range(n)]
    w = [1.0 if i < split else 1e-11 for i in range(n)]
    data = {"y": y, "a": a, "b": b}

    with pytest.warns(
        glmm.IllConditionedWarning, match="b is almost a combination of other columns"
    ) as caught:
        result = glmm.fit(data, "y ~ a + b", weights=w)
    assert result.converged
    assert not result.aliased.any()  # flagged, not dropped
    assert [(e["tier"], e["kind"], e["title"]) for e in result.warnings] == [
        ("caution", "ill_conditioned", "Nearly collinear columns"),
    ]
    (entry,) = result.warnings
    printed = f"{entry['tier'].capitalize()}: {entry['title']}. {entry['message']}"
    assert printed in [str(c.message) for c in caught]
    (note,) = result.diagnostics["notes"]
    assert note["kind"] == "ill_conditioned"
    assert note["columns"] == [2]  # 0-based into `names`
    assert note["pivot"] < 1e-9
    # Filterable as a category, and reachable through the base category too.
    assert issubclass(glmm.IllConditionedWarning, glmm.DiagnosticWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("error", glmm.DiagnosticWarning)
        with pytest.raises(glmm.IllConditionedWarning):
            glmm.fit(data, "y ~ a + b", weights=w)

    # Negative case: the same design at unit weights raises no note at all.
    with warnings.catch_warnings():
        warnings.simplefilter("error", glmm.DiagnosticWarning)
        clean = glmm.fit(data, "y ~ a + b")
    assert clean.diagnostics["notes"] == []


def test_gamma_glm_fixed_dispersion():
    y = _rng.gamma(shape=2.0, scale=np.exp(0.5 + 0.1 * _X) / 2.0)
    result = glmm.fit(_data(y.tolist()), "y ~ x", "gamma", dispersion=1.0)
    assert result.converged
    assert result.dispersion == pytest.approx(1.0)


def test_lmm_ranef_blocks_label_the_flat_vector():
    # The port is a pure renderer: the kernel owns the block layout, so what
    # this checks is that the labelled view and the flat vector are the same
    # numbers, and that the labels reached Python at all. The layout itself is
    # tested crate-side (tests/lmm_ranef.rs).
    rng = np.random.default_rng(7)
    b0 = rng.normal(scale=1.0, size=30)[_GROUPS]
    b1 = rng.normal(scale=1.0, size=30)[_GROUPS]
    y = 1.0 + b0 + (2.0 + b1) * _X + rng.normal(scale=0.5, size=_N)
    result = glmm.fit(_data(y.tolist()), "y ~ x + (x | g)")
    assert result.converged
    assert np.any(result.ranef != 0.0)
    assert len(result.fitted) == _N
    assert len(result.ranef_blocks) == 1
    block = result.ranef_blocks[0]
    assert block["group"] == "g"
    assert block["terms"] == ["(Intercept)", "x"]
    assert sorted(block["levels"]) == sorted(f"g{i}" for i in range(30))
    assert block["values"].shape == (30, 2)
    assert np.allclose(block["values"].reshape(-1), result.ranef)


def test_gradient_search_fields_cross_the_port():
    # A random-slope LMM searches theta with BFGS; a GLM runs no theta search.
    rng = np.random.default_rng(8)
    b0 = rng.normal(scale=1.0, size=30)[_GROUPS]
    b1 = rng.normal(scale=1.0, size=30)[_GROUPS]
    y = 1.0 + b0 + (2.0 + b1) * _X + rng.normal(scale=0.5, size=_N)
    routed = glmm.fit(_data(y.tolist()), "y ~ x + (x | g)")
    assert routed.n_grad > 0
    assert routed.outer_stop in {
        "fdelta",
        "pgrad",
        "noise_floor",
        "line_search",
        "max_iter",
        "grad_unavailable",
        "non_finite",
    }
    assert np.isfinite(routed.outer_decrement)

    plain = glmm.fit(_data(y.tolist()), "y ~ x")
    assert plain.n_grad == 0
    assert plain.outer_stop is None
    assert np.isnan(plain.outer_decrement)


def test_pinned_detail_is_reported_on_a_sparse_route_fit():
    # A slope-carrying extra grouping routes the fit to the sparse solver, which
    # assembles its Fit outside the dense mappers. Design: `g` carries no
    # signal at all, so its single component pins; the residual is centred
    # within each `g` level so not even sampling noise leaves it spurious
    # between-level variance.
    n = 240
    rng = np.random.default_rng(7)
    g = np.array([i % 12 for i in range(n)])
    h = np.array([i // 12 for i in range(n)])
    x = rng.uniform(-1.0, 1.0, size=n)
    noise = 0.1 * rng.uniform(-0.5, 0.5, size=n)
    for lvl in range(12):
        rows = g == lvl
        noise[rows] -= noise[rows].mean()
    y = 1.0 + 0.75 * x + np.sin(h * 0.37) + np.cos(h * 0.91) * x + noise
    data = {
        "y": y.tolist(),
        "x": x.tolist(),
        "g": [f"g{i}" for i in g.tolist()],
        "h": [f"h{i}" for i in h.tolist()],
    }
    with pytest.warns(glmm.SingularFitWarning, match=r"Affected: .*\(Intercept\) in g"):
        result = glmm.fit(data, "y ~ x + (1 | g) + (1 + x | h)")

    assert result.singular
    assert result.diagnostics["pinned"] == [[True], [False, False]]
    # One flag per stddev, per block — the alignment `_pinned_detail` walks.
    for idx, flags in enumerate(result.diagnostics["pinned"]):
        sd, _ = result.stddev_corr(idx)
        assert len(flags) == len(sd)


def test_degenerate_mixed_fit_returns_instead_of_panicking():
    # n <= p leaves the LMM with no finite deviance endpoint, so the kernel takes
    # its degenerate return: NaN beta/se/dispersion and — the crate's
    # numerical-failure convention — no assembled varcorr. A mixed fit lowers one
    # grouping regardless, so the two lengths disagree; that must reach the caller
    # as a returned non-converged result, not a PanicException across the FFI.
    data = {
        "y": [1.0, 2.0, 3.0],
        "x1": [0.1, 0.7, 0.3],
        "x2": [0.5, 0.2, 0.9],
        "g": ["a", "b", "a"],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = glmm.fit(data, "y ~ x1 + x2 + (1 | g)")
    assert not result.converged
    assert result.varcorr == []
    assert result.re_groups == [("g", ["(Intercept)"])]
    # summary() walks varcorr/re_groups together — it must print, not raise.
    assert "Random effects:" not in result.summary()


def test_cbind_response_equals_proportion_plus_weights():
    s = [3.0, 1.0, 4.0, 2.0] * (_N // 4)
    f = [2.0, 4.0, 1.0, 3.0] * (_N // 4)
    x = [0.1 * (i % 7) for i in range(_N)]
    data = {"s": s, "f": f, "x": x, "p": [a / (a + b) for a, b in zip(s, f)]}
    via = glmm.fit(data, "cbind(s, f) ~ x", "binomial")
    hand = glmm.fit(data, "p ~ x", "binomial", weights=[a + b for a, b in zip(s, f)])
    assert via.beta == pytest.approx(hand.beta, rel=1e-12)
    # The trial counts the lowering made up come back as the fit's weights, so
    # Pearson residuals match the hand-weighted fit's instead of silently
    # dropping the sqrt(trials) factor.
    np.testing.assert_allclose(via.weights, [a + b for a, b in zip(s, f)])
    np.testing.assert_allclose(
        via.residuals(type="pearson"), hand.residuals(type="pearson"), rtol=1e-10
    )
    assert via.link == "logit"


def test_fit_carries_header_inputs():
    y = (1.0 + 2.0 * _X + _rng.normal(scale=0.5, size=_N)).tolist()
    w = np.full(_N, 2.0)
    fit = glmm.fit(_data(y), "y ~ x + (1 | g)", weights=w)
    assert fit.names == ["(Intercept)", "x"]
    assert fit.formula == "y ~ x + (1 | g)"
    assert fit.family == "gaussian"
    assert fit.link == "identity"
    assert fit.nagq == 1
    assert fit.nobs == _N
    np.testing.assert_array_equal(fit.y, np.asarray(y))
    np.testing.assert_array_equal(fit.weights, w)
    ols = glmm.fit(_data(y), "y ~ x")
    assert ols.weights is None
    assert ols.df == len(ols.beta) + 1  # p fixed effects + sigma^2
    assert ols.reml is False
    assert len(ols.fitted) == _N
