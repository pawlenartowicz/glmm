//! The reader's pure pieces on synthetic input. Declared here, not inside
//! `test_refs.rs`, so the integration side's `include!` of that file does not
//! run them a second time.

use super::test_refs::{covers, stale_entries, Coverage, DivEntry};
use std::collections::BTreeSet;

fn div(id: &str, cell: &str, scope: &str, quantities: &[&str], max_rel: f64) -> DivEntry {
    DivEntry {
        id: id.to_string(),
        cell: cell.to_string(),
        comparison: vec![scope.to_string()],
        quantities: quantities.iter().map(|q| q.to_string()).collect(),
        max_rel,
    }
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| n.to_string()).collect()
}

#[test]
fn covers_matches_cell_quantity_and_scope() {
    let entries = [
        div("other-scope", "ref_a", "glmm-vs-oracle", &["df"], 0.5),
        div("a-df", "ref_a", "oracle-tier", &["df", "p_value"], 0.01),
    ];
    let within = Coverage::Within {
        id: "a-df".to_string(),
    };
    let exceeds = Coverage::Exceeds {
        id: "a-df".to_string(),
        max_rel: 0.01,
    };
    assert_eq!(covers(&entries, "ref_a", "df", 0.005), within);
    assert_eq!(covers(&entries, "ref_a", "p_value", 0.01), within);
    assert_eq!(covers(&entries, "ref_a", "df", 0.02), exceeds);
    assert_eq!(covers(&entries, "ref_a", "df", f64::NAN), exceeds);

    assert_eq!(covers(&entries, "ref_b", "df", 0.005), Coverage::NotCovered);
    assert_eq!(
        covers(&entries, "ref_a", "lrt", 0.005),
        Coverage::NotCovered
    );
    let other_scope_only = [div("x", "ref_a", "glmm-vs-oracle", &["lrt"], 0.5)];
    assert_eq!(
        covers(&other_scope_only, "ref_a", "lrt", 0.005),
        Coverage::NotCovered
    );
}

#[test]
fn stale_entries_are_compared_oracle_tier_entries_that_did_not_fire() {
    let entries = [
        div("unfired", "ref_a", "oracle-tier", &["df"], 0.1),
        div("not-compared", "ref_z", "oracle-tier", &["df"], 0.1),
        div("fired", "ref_b", "oracle-tier", &["df"], 0.1),
        div("other-scope", "ref_a", "glmm-vs-oracle", &["df"], 0.1),
    ];
    let stale = stale_entries(&entries, &set(&["ref_a", "ref_b"]), &set(&["fired"]));
    assert_eq!(stale, vec!["unfired".to_string()]);
}
