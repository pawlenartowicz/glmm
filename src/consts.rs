//! Scratch-bounding capacity constants. `glmm` owns these (carve spec §6); they
//! size the solver's stack buffers, so they are HARD ceilings. MCPower's
//! `engine-contract` re-exports MAX_PRIMARY_Q / MAX_EXTRA_GROUPINGS for its
//! `validate()` invariants — single source, no drift.

/// Max primary RE width `q_p = 1 + #slopes` (≤ 7 random slopes).
pub const MAX_PRIMARY_Q: usize = 8;

/// Max extra grouping factors. Mirrors the solver's per-row level-id buffer
/// width `1 + MAX_EXTRA_GROUPINGS`.
pub const MAX_EXTRA_GROUPINGS: usize = 7;

/// Max θ length: full primary vech + one variance per extra grouping.
pub const MAX_THETA: usize = MAX_PRIMARY_Q * (MAX_PRIMARY_Q + 1) / 2 + MAX_EXTRA_GROUPINGS;
