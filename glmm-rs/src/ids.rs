//! `GroupIds` — the per-fit random-effect level ids (data companion to `x`/`y`).
//!
//! The stable entry (`crate::fit`/`fit_with_start`) is always ids-driven: every
//! level count the kernels size their random-effect blocks by is derived from
//! these ids. Real-data callers build `GroupIds` directly (the formula frontend
//! and the ports fill it from grouping columns).

/// Per-row random-effect level ids for one fit. Each vector has length `n`.
///
/// - `primary[i]` is the primary grouping's level for row `i` (values
///   `0..max+1`, dense, each below `n`).
/// - `extra[g][i]` is the level of the g-th extra grouping for row `i`, in
///   [`crate::ReStructure::extra_groupings`] declaration order; a crossed one
///   is below `n` too.
///
/// Empty / ignored for fixed-only models (`re: None`). For a **nested** extra
/// grouping the id is
/// `parent · W + position within the parent`, `W` the widest parent's distinct
/// child count, so each parent owns the slots `parent·W .. (parent+1)·W`; the
/// entry sizes the RE block as `n_primary · W` and refuses other ids.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GroupIds {
    /// Primary grouping level per row (length `n`).
    pub primary: Vec<u32>,
    /// One level vector (length `n`) per extra grouping, declaration order.
    pub extra: Vec<Vec<u32>>,
}

impl GroupIds {
    /// The primary grouping's level per row, with no extra groupings.
    pub fn new(primary: Vec<u32>) -> GroupIds {
        GroupIds {
            primary,
            extra: Vec::new(),
        }
    }

    /// Appends the level ids of the next extra grouping, in the order the
    /// groupings are declared in [`crate::ReStructure::extra_groupings`].
    pub fn extra(mut self, ids: Vec<u32>) -> GroupIds {
        self.extra.push(ids);
        self
    }
}
