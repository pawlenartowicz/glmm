def test_faststats_alias_ships():
    # faststats/ reaches the wheel only through the `include` glob in pyproject.toml;
    # nothing else imports it, so a missing package would pass every other gate
    # silently. This must run against a wheel built from the sdist too (the release
    # build_sdist job) — that path is what breaks if the include is wheel-scoped. See
    # the include comment in pyproject.toml. ModuleNotFoundError means it stopped matching.
    import faststats.glmm

    import glmm

    assert faststats.glmm is glmm
