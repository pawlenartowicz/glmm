//! Scratch-bounding capacity constants. `glmm` owns these; they
//! size the solver's stack buffers, so they are HARD ceilings. MCPower's
//! `engine-contract` re-exports MAX_PRIMARY_Q / MAX_EXTRA_GROUPINGS for its
//! `validate()` invariants — single source, no drift.

/// Max primary RE width `q_p = 1 + #slopes` (≤ 7 random slopes).
pub const MAX_PRIMARY_Q: usize = 8;

/// Max extra grouping factors. Mirrors the solver's per-row level-id buffer
/// width `1 + MAX_EXTRA_GROUPINGS`. 6 (not higher) keeps total variance
/// components `MAX_PRIMARY_Q + MAX_EXTRA_GROUPINGS·MAX_EXTRA_Q = 8 + 6·4 = 32`
/// within the 32-bit `pinned_components` bitmask — safe by construction;
/// a future sparse-path implementation may raise this ceiling.
pub const MAX_EXTRA_GROUPINGS: usize = 6;

/// Max random-effect dimension per EXTRA grouping factor: `q_g = 1 + #slopes`
/// (intercept + up to 3 random slopes). Bounds the per-factor `vech(Λ_g)` θ block
/// and every stack buffer sized off `MAX_THETA`. 4 covers the realistic ceiling
/// (a crossed factor with intercept + 3 slopes); raise only with a benchmark.
pub const MAX_EXTRA_Q: usize = 4;

/// Max θ length: full primary `vech(Λ_p)` plus a `vech(Λ_g)` block per extra
/// grouping. For `MAX_EXTRA_Q = 1` this collapses to one scalar per extra — the
/// pre-slope bound `… + MAX_EXTRA_GROUPINGS`.
pub const MAX_THETA: usize = MAX_PRIMARY_Q * (MAX_PRIMARY_Q + 1) / 2
    + MAX_EXTRA_GROUPINGS * (MAX_EXTRA_Q * (MAX_EXTRA_Q + 1) / 2);
