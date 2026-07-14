//! `GroupIds` — the per-fit random-effect level ids (data companion to `x`/`y`).
//!
//! The stable entry (`crate::fit_cold`/`fit_warm`) is always ids-driven. Real-data
//! callers build `GroupIds` directly (the formula frontend / Python port fill it
//! from grouping columns); the balanced/positional simulator path survives only as
//! [`GroupIds::from_sizing`], which derives ids from a count-parameterized
//! [`crate::ReStructure`] up front, then calls the same entry.

#[cfg(test)]
use crate::{GroupingRelation, ReStructure, Sizing};

/// Per-row random-effect level ids for one fit. Each vector has length `n`.
///
/// - `primary[i]` is the primary grouping's level for row `i` (values
///   `0..max+1`, dense).
/// - `extra[g][i]` is the level of the g-th extra grouping for row `i`, in
///   [`crate::ReStructure::extra_groupings`] declaration order.
///
/// Empty / ignored for fixed-only models (`re: None`). For a **nested** extra
/// grouping the id is the GLOBAL child level (dense over all parents); the entry
/// sizes its RE block as `n_primary · ⌈children / n_primary⌉`, so a balanced
/// nesting maps its `children = n_primary · per_parent` global ids exactly.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GroupIds {
    /// Primary grouping level per row (length `n`).
    pub primary: Vec<u32>,
    /// One level vector (length `n`) per extra grouping, declaration order.
    pub extra: Vec<Vec<u32>>,
}

impl GroupIds {
    /// Balanced/positional path: derive ids from the count-parameterized
    /// `ReStructure` (the simulator's worldview). This is how the count form
    /// "stays for the simulator": tests/balanced-generation build ids from counts,
    /// then call the same entry. Currently `#[cfg(test)]`-gated — the only callers
    /// are the crate's own tests; un-gate to `pub(crate)`/`pub` when a real
    /// consumer wraps `fit_cold` with balanced generation.
    #[cfg(test)]
    pub(crate) fn from_sizing(re: &ReStructure, n: usize) -> Self {
        let primary: Vec<u32> = (0..n).map(|i| re.sizing.cluster_of_row(i) as u32).collect();
        let extra: Vec<Vec<u32>> = (0..re.extra_groupings.len())
            .map(|g| (0..n).map(|i| extra_level_of_row(re, g, i)).collect())
            .collect();
        GroupIds { primary, extra }
    }
}

/// Local level id for extra grouping `g` at row `i`, from the positional sizing.
/// Verbatim from the former `fit::extra_level_of_row`; this is the canonical
/// copy — `test_support::extra_level_of_row` is a thin `&ModelSpec`-unwrapping
/// wrapper that delegates here. Test-gated alongside its only caller,
/// [`GroupIds::from_sizing`].
#[cfg(test)]
pub(crate) fn extra_level_of_row(re: &ReStructure, g: usize, i: usize) -> u32 {
    let rel = &re.extra_groupings[g].relation;
    let level = match &re.sizing {
        Sizing::FixedClusters { n_clusters } => {
            let s = (*n_clusters).max(1) as usize;
            let mut stride = s;
            for h in &re.extra_groupings[..g] {
                stride *= block_levels(&h.relation);
            }
            let within = (i / stride) % block_levels(rel);
            match rel {
                GroupingRelation::Crossed { .. } => within,
                GroupingRelation::NestedWithin { n_per_parent } => {
                    (i % s) * (*n_per_parent).max(1) as usize + within
                }
            }
        }
        Sizing::FixedSize { cluster_size } => {
            let cs = (*cluster_size).max(1) as usize;
            let np = block_levels(rel);
            (i / cs) * np + (i % cs) % np
        }
    };
    level as u32
}

#[cfg(test)]
pub(crate) fn block_levels(rel: &GroupingRelation) -> usize {
    match rel {
        GroupingRelation::Crossed { n_clusters } => (*n_clusters).max(1) as usize,
        GroupingRelation::NestedWithin { n_per_parent } => (*n_per_parent).max(1) as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Grouping, GroupingRelation, ReStructure, Sizing};

    /// `from_sizing` on a 4-cluster primary + one crossed 3-level extra reproduces
    /// the positional layout: primary is `i % 4`; the crossed extra is
    /// `(i / 4) % 3`.
    #[test]
    fn from_sizing_crossed_matches_positional() {
        let re = ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 4 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 3 },
                slopes: vec![],
            }],
        };
        let ids = GroupIds::from_sizing(&re, 12);
        assert_eq!(
            ids.primary,
            (0..12).map(|i| (i % 4) as u32).collect::<Vec<_>>()
        );
        assert_eq!(ids.extra.len(), 1);
        assert_eq!(
            ids.extra[0],
            (0..12).map(|i| ((i / 4) % 3) as u32).collect::<Vec<_>>()
        );
    }
}
