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


def test_gaussian_lmm():
    y = 1.0 + 2.0 * _X + _rng.normal(scale=0.5, size=_N)
    result = glmm.fit(_data(y.tolist()), "y ~ x + (1 | g)")
    assert result.converged
    assert result.names == ["(Intercept)", "x"]
    assert abs(result.beta[1] - 2.0) < 0.3


def test_binomial_glmm():
    p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * _X)))
    y = _rng.binomial(1, p).astype(float)
    result = glmm.fit(_data(y.tolist()), "y ~ x + (1 | g)", "binomial")
    assert result.converged


def test_singular_fit_warning_names_component():
    # No group effect in tiny clusters: the RE variance pins to 0 and the
    # warning names the pinned component (mirrors the R port's test).
    rng = np.random.default_rng(1)
    x = rng.normal(size=120)
    p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x)))
    data = {
        "x": x.tolist(),
        "g": [f"g{i}" for i in np.repeat(np.arange(30), 4).tolist()],
        "y": rng.binomial(1, p).astype(float).tolist(),
    }
    with pytest.warns(
        UserWarning,
        match=r"boundary \(singular\) fit: see help\('isSingular'\); "
        r"sd\(\(Intercept\) \| g\) pinned at the variance boundary",
    ):
        result = glmm.fit(data, "y ~ x + (1 | g)", "binomial")
    assert result.singular


def test_diagnostics_exposes_the_fields_and_the_flags_still_read_off_fit():
    y = 1.0 + 2.0 * _X + _rng.normal(scale=0.5, size=_N)
    result = glmm.fit(_data(y.tolist()), "y ~ x + (1 | g)")
    d = result.diagnostics
    assert set(d) == {"converged", "singular", "aliased", "boundary", "pinned", "notes"}
    # The three moved flags still read straight off the Fit, and off one storage.
    assert result.converged is d["converged"]
    assert result.singular is d["singular"]
    assert result.aliased is d["aliased"]
    assert result.converged
    # This shared fixture has no real group effect, so the one variance
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
    # Design mirrors the Rust lmm::tests::zero_slope_variance_pins_slope_component:
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
        UserWarning,
        match=r"boundary \(singular\) fit: see help\('isSingular'\); "
        r"sd\(x \| g\) pinned at the variance boundary",
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
        glmm.IllConditionedWarning, match="b is entangled with one or more other columns"
    ):
        result = glmm.fit(data, "y ~ a + b", weights=w)
    assert result.converged
    assert not result.aliased.any()  # flagged, not dropped
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


def test_pirls_exhausted_message_distinguishes_final_eval():
    # No known dataset reaches final_eval=True end-to-end, so both message
    # branches are asserted from constructed notes; the Rust-side test
    # pirls_exhausted_payload_survives_flattening pins the payload itself.
    note = {
        "kind": "pirls_exhausted",
        "columns": [],
        "pivot": float("nan"),
        "evals": 3,
        "final_eval": False,
        "detail": "",
    }
    benign_msg, benign_cat = glmm._note_warning(note, [])
    assert benign_cat is glmm.PirlsExhaustedWarning
    assert "observation-only and no fitted number is affected" in benign_msg

    serious_msg, serious_cat = glmm._note_warning(dict(note, evals=0, final_eval=True), [])
    assert serious_cat is glmm.PirlsExhaustedWarning
    assert "the reported estimates rest on that truncated solve" in serious_msg


def test_re_design_scale_spread_message_names_grouping_and_ratio():
    # No fixture below drives the note through a real fit here (the Rust-side
    # end-to-end test covers that: fit::common_tests::
    # re_design_scale_spread_note_fires_on_mismatched_slope_scale), so the
    # message is asserted from a constructed note, mirroring the
    # pirls_exhausted test above.
    note = {
        "kind": "re_design_scale_spread",
        "columns": [],
        "pivot": float("nan"),
        "evals": 0,
        "final_eval": False,
        "detail": "g",
        "ratio": 4200.0,
    }
    msg, cat = glmm._note_warning(note, [])
    assert cat is glmm.ReDesignScaleWarning
    assert "'g'" in msg
    assert "4.2e+03" in msg
    assert "scales the columns internally" in msg


def test_hessian_se_fallback_message():
    note = {
        "kind": "hessian_se_fallback",
        "columns": [],
        "pivot": float("nan"),
        "evals": 0,
        "final_eval": False,
        "detail": "",
        "ratio": float("nan"),
    }
    msg, cat = glmm._note_warning(note, [])
    assert cat is glmm.HessianSeFallbackWarning
    assert "not positive definite" in msg
    assert "stddev_se is NaN" in msg


def test_poisson_glm():
    y = _rng.poisson(np.exp(0.5 + 0.3 * _X)).astype(float)
    result = glmm.fit(_data(y.tolist()), "y ~ x", "poisson")
    assert result.converged
    assert result.dispersion == pytest.approx(1.0)


def test_gamma_glm_fixed_dispersion():
    y = _rng.gamma(shape=2.0, scale=np.exp(0.5 + 0.1 * _X) / 2.0)
    result = glmm.fit(_data(y.tolist()), "y ~ x", "gamma", dispersion=1.0)
    assert result.converged
    assert result.dispersion == pytest.approx(1.0)


def test_negativebinomial_glm_cold_start():
    y = _rng.negative_binomial(5, 0.5, size=_N).astype(float)
    result = glmm.fit(_data(y.tolist()), "y ~ x", "negativebinomial")
    assert result.converged
    assert result.dispersion > 0  # theta estimate


def test_agq_vector_q2_smoke():
    # q=2 (random intercept + slope) binomial with small clusters, where the
    # Laplace bias AGQ corrects is visible — proves nagq>1 actually routed to
    # the vector-AGQ kernel instead of silently refitting Laplace.
    rng = np.random.default_rng(42)
    n_groups, per = 75, 4
    n = n_groups * per
    g = np.repeat(np.arange(n_groups), per)
    x = rng.normal(size=n)
    b0 = rng.normal(scale=1.2, size=n_groups)
    b1 = rng.normal(scale=0.8, size=n_groups)
    eta = 0.3 + 0.8 * x + b0[g] + b1[g] * x
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    data = {"y": y.tolist(), "x": x.tolist(), "g": [f"g{i}" for i in g.tolist()]}

    laplace = glmm.fit(data, "y ~ x + (1 + x | g)", "binomial")  # nagq=1 default
    agq = glmm.fit(data, "y ~ x + (1 + x | g)", "binomial", nagq=7, wald_se="hessian")
    assert agq.converged
    assert np.all(np.isfinite(agq.se))
    # Gate routed: quadrature moves the answer off the Laplace fit (empirically
    # ~3e-2 in beta and ~0.3 in varcorr on this dataset — far above tolerance).
    assert np.max(np.abs(agq.beta - laplace.beta)) > 1e-6
    assert np.max(np.abs(np.array(agq.varcorr[0]) - np.array(laplace.varcorr[0]))) > 1e-4


def test_gaussian_fixed_only_exposes_loglik_df_reml_fitted():
    y = 1.0 + 2.0 * _X + _rng.normal(scale=0.5, size=_N)
    result = glmm.fit(_data(y.tolist()), "y ~ x")
    assert result.converged
    assert np.isfinite(result.loglik)
    assert result.df == len(result.beta) + 1  # p fixed effects + sigma^2
    assert result.reml is False
    assert len(result.fitted) == _N


def test_mixed_binomial_exposes_ranef_consistent_with_levels():
    p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * _X)))
    y = _rng.binomial(1, p).astype(float)
    result = glmm.fit(_data(y.tolist()), "y ~ x + (1 | g)", "binomial")
    assert result.converged
    assert len(result.ranef_levels) == len(result.varcorr)
    q = 1  # scalar random intercept -> q=1 per level
    assert len(result.ranef) == sum(int(lv) * q for lv in result.ranef_levels)


def test_mixed_poisson_exposes_ranef_consistent_with_levels():
    y = _rng.poisson(np.exp(0.5 + 0.3 * _X)).astype(float)
    result = glmm.fit(_data(y.tolist()), "y ~ x + (1 | g)", "poisson")
    assert result.converged
    assert len(result.ranef_levels) == len(result.varcorr)
    q = 1
    assert len(result.ranef) == sum(int(lv) * q for lv in result.ranef_levels)


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


def test_offset_shifts_poisson_intercept_by_minus_constant():
    y = _rng.poisson(np.exp(0.5 + 0.3 * _X)).astype(float)
    data = _data(y.tolist())
    base = glmm.fit(data, "y ~ x", "poisson")
    c = 1.7
    shifted = glmm.fit(data, "y ~ x", "poisson", offset=[c] * _N)
    assert base.converged and shifted.converged
    assert abs(shifted.beta[0] - (base.beta[0] - c)) < 1e-3
    assert np.max(np.abs(shifted.beta[1:] - base.beta[1:])) < 1e-3


def test_warm_start_reaches_same_answer_as_cold():
    y = 1.0 + 2.0 * _X + _rng.normal(scale=0.5, size=_N)
    cold = glmm.fit(_data(y.tolist()), "y ~ x + (1 | g)")
    warm = glmm.fit(
        _data(y.tolist()),
        "y ~ x + (1 | g)",
        warm_start={"beta": cold.beta.tolist(), "theta": [1.0]},
    )
    assert warm.converged
    assert np.allclose(warm.beta, cold.beta, atol=1e-6)


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
    with pytest.warns(
        UserWarning, match=r"sd\(\(Intercept\) \| g\) pinned at the variance boundary"
    ):
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
