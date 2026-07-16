def test_faststats_alias_ships():
    # faststats/ reaches the wheel only through the wheel-scoped `include` glob in
    # pyproject.toml; nothing else imports it, so a missing package would pass every
    # other gate silently. ModuleNotFoundError here means the glob stopped matching.
    import faststats.glmm

    import glmm

    assert faststats.glmm is glmm
