use super::random::{grouping_ids, grouping_level_count};
use super::*;
use crate::formula::parse::RandomEffect;
use crate::test_support::{csv_table, with_unused};
use crate::{ColumnId, GroupingRelation, ReStructure, ReTerm};

const FACTORIAL: &str = include_str!("../../../../validation/data/simulated/sim_factorial.csv");

/// Unbalanced nesting — parents "A","B","C" (lexicographic ids 0,1,2) with
/// 1, 2, 3 distinct children respectively (child labels reused across
/// parents, as real nested factors do — e.g. cask "1"/"2"/"3" repeats in
/// every batch). Hand-computed expected layout: contiguous per-parent
/// blocks padded to `W = max_p k_p = 3` — `A`'s single child lands at `0`
/// (slots 1,2 unused padding), `B`'s two children at `[3,4)` (slot 5
/// padding), `C`'s three children fill `[6,9)` exactly.
#[test]
fn grouping_ids_nested_unbalanced_pads_to_max_per_parent() {
    let table = Table {
        columns: vec![
            (
                "g1".into(),
                Column::factor_from_labels(&strs(&["A", "B", "B", "C", "C", "C"])),
            ),
            (
                "g2".into(),
                Column::factor_from_labels(&strs(&["c1", "c1", "c2", "c1", "c2", "c3"])),
            ),
        ],
        n: 6,
    };
    let re = RandomEffect::Intercept {
        group: "g1:g2".to_string(),
        parent: Some("g1".to_string()),
    };
    let layout = grouping_ids(&re, &table).unwrap();
    assert_eq!(layout.ids, vec![0, 3, 4, 6, 7, 8]);
    // The same hand-computed rectangle, read as labels: parent-first joins at
    // every assigned slot, `None` at the two padded ones.
    assert_eq!(
        layout.slot_labels,
        vec![
            Some("A:c1".into()),
            None,
            None,
            Some("B:c1".into()),
            Some("B:c2".into()),
            None,
            Some("C:c1".into()),
            Some("C:c2".into()),
            Some("C:c3".into()),
        ]
    );
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

/// Labels containing `:`: the pairs `("a:b", "c")` and `("a", "b:c")` join
/// to the same `"a:b:c"` but are two levels, in the level count of both the
/// nested and the crossed spelling and in the crossed arm's codes. Codes by
/// hand: equal joined labels, so the pair decides and `"a" < "a:b"` puts
/// `("a", "b:c")` first.
#[test]
fn colon_in_labels_keeps_distinct_pairs_apart() {
    let table = Table {
        columns: vec![
            (
                "g1".into(),
                Column::factor_from_labels(&strs(&["a:b", "a", "a:b", "a"])),
            ),
            (
                "g2".into(),
                Column::factor_from_labels(&strs(&["c", "b:c", "c", "b:c"])),
            ),
        ],
        n: 4,
    };
    let crossed = RandomEffect::Intercept {
        group: "g1:g2".to_string(),
        parent: None,
    };
    let nested = RandomEffect::Intercept {
        group: "g1:g2".to_string(),
        parent: Some("g1".to_string()),
    };
    assert_eq!(grouping_level_count(&crossed, &table).unwrap(), 2);
    assert_eq!(grouping_level_count(&nested, &table).unwrap(), 2);
    let layout = grouping_ids(&crossed, &table).unwrap();
    assert_eq!(layout.ids, vec![1, 0, 1, 0]);
    assert_eq!(
        layout.slot_labels,
        vec![Some("a:b:c".into()), Some("a:b:c".into())]
    );
}

/// Two flat extras that both nest in the primary lower `Crossed`, each on
/// its own lexicographic factor codes with bare labels: only explicit
/// `parent:child` syntax lowers nested, and the fit finds the nests itself.
#[test]
fn flat_nesting_candidates_lower_crossed() {
    let table = Table {
        columns: vec![
            (
                "y".into(),
                Column::Numeric(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            ),
            (
                "g1".into(),
                Column::factor_from_labels(&strs(&["A", "A", "A", "A", "B", "B", "B", "B"])),
            ),
            (
                "g2".into(),
                Column::factor_from_labels(&strs(&[
                    "a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2",
                ])),
            ),
            (
                "g3".into(),
                Column::factor_from_labels(&strs(&[
                    "c1", "c1", "c2", "c2", "d1", "d1", "d2", "d2",
                ])),
            ),
        ],
        n: 8,
    };
    let lo = super::lower("y ~ (1|g1) + (1|g2) + (1|g3)", &table, Family::Gaussian).unwrap();
    let relations: Vec<_> = lo
        .model
        .re
        .as_ref()
        .unwrap()
        .extra_groupings
        .iter()
        .map(|g| g.relation)
        .collect();
    assert_eq!(relations, [GroupingRelation::Crossed; 2]);
    assert_eq!(
        lo.ids.extra[0],
        vec![0, 0, 1, 1, 2, 2, 3, 3],
        "g2's own lexicographic factor codes (a1,a2,b1,b2)"
    );
    assert_eq!(
        lo.re_groups[1].slot_labels,
        vec![
            Some("a1".into()),
            Some("a2".into()),
            Some("b1".into()),
            Some("b2".into())
        ],
        "g2's slot labels are bare child labels, not parent:child"
    );
    assert_eq!(
        lo.ids.extra[1],
        vec![0, 0, 1, 1, 2, 2, 3, 3],
        "g3's own lexicographic factor codes (c1,c2,d1,d2), independent of g1"
    );
    assert_eq!(
        lo.re_groups[2].slot_labels,
        vec![
            Some("c1".into()),
            Some("c2".into()),
            Some("d1".into()),
            Some("d2".into())
        ]
    );
}

/// A formula with the intercept removed and no other fixed term (`y ~ 0`,
/// or `y ~ 0 + (1|g)` with only a random intercept) leaves the design with
/// zero columns — `materialize` refuses this outright rather than handing
/// the fit a `p == 0` design.
#[test]
fn empty_fixed_design_is_refused() {
    let table = Table {
        columns: vec![
            ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0, 4.0])),
            (
                "g".into(),
                Column::factor_from_labels(&strs(&["a", "a", "b", "b"])),
            ),
        ],
        n: 4,
    };
    for formula in ["y ~ 0", "y ~ 0 + (1|g)"] {
        let err = match super::lower(formula, &table, Family::Gaussian) {
            Ok(_) => panic!("{formula}: expected an empty fixed design to be refused"),
            Err(e) => e,
        };
        assert_eq!(err, Error::EmptyDesign, "{formula}");
        assert!(err.to_string().contains("no columns"), "{formula}: {err}");
    }
}

/// A fixed-part factor with a single declared level errors, naming the
/// column — lme4's own "contrasts can be applied only to factors with 2
/// or more levels", with the offending column named. `y ~ f:x` reaches the
/// factor only through an interaction term (no main effect, so `f` is not
/// lowered first): the `interaction_columns` → `expand_var` →
/// `factor_dummies` route.
#[test]
fn one_level_factor_in_fixed_part_errors() {
    let table = Table {
        columns: vec![
            ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0])),
            ("x".into(), Column::Numeric(vec![0.1, 0.2, 0.3])),
            (
                "f".into(),
                Column::factor_from_labels(&strs(&["z", "z", "z"])),
            ),
        ],
        n: 3,
    };
    // One level used of three declared: the unused two are dropped first.
    let declared = Table {
        columns: vec![
            ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0])),
            ("x".into(), Column::Numeric(vec![0.1, 0.2, 0.3])),
            (
                "f".into(),
                Column::Factor {
                    levels: strs(&["z", "a", "b"]),
                    codes: vec![0, 0, 0],
                },
            ),
        ],
        n: 3,
    };
    for table in [&table, &declared] {
        for formula in ["y ~ f + x", "y ~ f:x"] {
            assert_eq!(
                super::lower(formula, table, Family::Gaussian)
                    .err()
                    .unwrap(),
                Error::SingleLevelFactor {
                    name: "f".to_string()
                },
                "{formula}"
            );
        }
    }
}

/// An unused level, first or in the middle, is dropped before coding, as
/// R's model frame does (`drop.unused.levels = TRUE`): the lowering equals
/// that of the same rows without it, in the fixed part, an interaction and
/// a factor slope.
#[test]
fn unused_levels_are_dropped_before_coding() {
    let plain = csv_table(FACTORIAL, &["subj", "g", "h"]);
    let terms = |lo: &Lowered| {
        lo.re_groups
            .iter()
            .map(|g| g.terms.clone())
            .collect::<Vec<_>>()
    };
    for formula in ["y ~ x + g", "y ~ 0 + g", "y ~ x * g + (1 + g | subj)"] {
        let want = super::lower(formula, &plain, Family::Gaussian).unwrap();
        for at in [0, 1] {
            let table = with_unused(&plain, "g", at);
            let got = super::lower(formula, &table, Family::Gaussian).unwrap();
            assert_eq!(got.col_names, want.col_names, "{formula} at {at}");
            assert_eq!(got.x, want.x, "{formula} at {at}");
            assert_eq!(got.re_columns, want.re_columns, "{formula} at {at}");
            assert_eq!(terms(&got), terms(&want), "{formula} at {at}");
            assert_eq!(got.factor_levels, want.factor_levels, "{formula} at {at}");
        }
    }
    // The names a reader checks by eye: base "a", no "gzzz".
    let lo = super::lower("y ~ x + g", &with_unused(&plain, "g", 0), Family::Gaussian).unwrap();
    assert_eq!(lo.col_names, ["(Intercept)", "x", "gb", "gc"]);
}

/// A grouping with a single observed level is dropped with a
/// `SingleLevelGroupingDropped` note rather than fitted, and the other
/// random-effect term is promoted to primary and still fits.
#[test]
fn single_level_grouping_dropped_with_note_keeps_other_groupings() {
    let table = Table {
        columns: vec![
            ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0, 4.0])),
            ("x".into(), Column::Numeric(vec![0.1, 0.2, 0.3, 0.4])),
            (
                "onegroup".into(),
                Column::factor_from_labels(&strs(&["only", "only", "only", "only"])),
            ),
            (
                "h".into(),
                Column::factor_from_labels(&strs(&["a", "a", "b", "b"])),
            ),
        ],
        n: 4,
    };
    let lo = super::lower("y ~ x + (1 | onegroup) + (1 | h)", &table, Family::Gaussian).unwrap();
    assert!(matches!(
        lo.notes.as_slice(),
        [crate::Note::SingleLevelGroupingDropped { grouping }] if grouping == "onegroup"
    ));
    assert_eq!(lo.re_groups.len(), 1);
    assert_eq!(lo.re_groups[0].name, "h");
}

/// Every random-effect term is single-level: the fit routes exactly like a
/// formula that declared no random effects, keeping both warnings.
#[test]
fn every_grouping_single_level_drops_to_no_random_effects() {
    let table = Table {
        columns: vec![
            ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0, 4.0])),
            ("x".into(), Column::Numeric(vec![0.1, 0.2, 0.3, 0.4])),
            (
                "g1".into(),
                Column::factor_from_labels(&strs(&["only", "only", "only", "only"])),
            ),
            (
                "g2".into(),
                Column::factor_from_labels(&strs(&["one", "one", "one", "one"])),
            ),
        ],
        n: 4,
    };
    let lo = super::lower("y ~ x + (1 | g1) + (1 | g2)", &table, Family::Gaussian).unwrap();
    assert!(lo.model.re.is_none());
    assert!(lo.re_groups.is_empty());
    assert_eq!(lo.notes.len(), 2);
    assert!(lo
        .notes
        .iter()
        .all(|n| matches!(n, crate::Note::SingleLevelGroupingDropped { .. })));
}

/// [`unresolved_columns`] names a bare missing column and a transform's
/// missing underlying column (`log(z)` reports `z`, not the call text),
/// and stays empty once everything resolves.
#[cfg(feature = "orchestrate")]
#[test]
fn unresolved_columns_reports_missing_names_including_transform_targets() {
    let ast = parse("y ~ log(z) + (1 | g) + offset(o)").unwrap();
    let available = strs(&["y", "g"]);
    assert_eq!(
        unresolved_columns(&ast, &available),
        vec!["z".to_string(), "o".to_string()]
    );

    let complete = strs(&["y", "z", "g", "o"]);
    assert!(unresolved_columns(&ast, &complete).is_empty());

    // A name spelled in several terms is reported once, in formula order.
    assert_eq!(
        unresolved_columns(&parse("y ~ z + w + z:w").unwrap(), &strs(&["y"])),
        strs(&["z", "w"])
    );

    // A formula with no response leaves the refusal to `materialize`.
    assert!(unresolved_columns(&parse("~ x").unwrap(), &strs(&["x"])).is_empty());
}

/// A formula with no response (`~ x`, or no separator at all) is a syntax
/// error at lowering, not an unknown column named after the parser's
/// placeholder.
#[test]
fn formula_without_response_is_refused() {
    let table = Table {
        columns: vec![("x".into(), Column::Numeric(vec![1.0, 2.0, 3.0]))],
        n: 3,
    };
    let want = Error::Parse(ParseError::Syntax {
        pos: 0,
        msg: "the formula has no response; write it as y ~ x".into(),
    });
    for f in ["~ x", "x"] {
        assert_eq!(
            lower(f, &table, Family::Gaussian).err(),
            Some(want.clone()),
            "{f}"
        );
    }
}

/// 36 rows: six groups `g` of six, a three-level factor `f` (`a`, `b`,
/// `c`) varying within each group, a numeric `x` and a response.
fn factor_table() -> Table {
    let n = 36;
    let (mut y, mut x, mut f, mut g) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        x.push(((i * 7) % 11) as f64 / 10.0);
        y.push(1.0 + ((i * 5) % 13) as f64 / 6.0);
        f.push(["a", "b", "c"][i % 3].to_string());
        g.push(format!("g{}", i / 6));
    }
    Table {
        columns: vec![
            ("y".into(), Column::Numeric(y)),
            ("x".into(), Column::Numeric(x)),
            ("f".into(), Column::factor_from_labels(&f)),
            ("g".into(), Column::factor_from_labels(&g)),
        ],
        n,
    }
}

/// The kernel's grouping layout of a lowering's model.
fn groupings(lo: &Lowered) -> crate::lmm::LmmGroupings {
    let re = lo.model.re.clone().expect("a mixed model");
    let cols = |s: &[ColumnId]| s.iter().map(|&c| c as usize).collect::<Vec<_>>();
    let primary = cols(&re.slopes);
    let extras: Vec<Vec<usize>> = re.extra_groupings.iter().map(|g| cols(&g.slopes)).collect();
    let sized = crate::fit::SizedSpec {
        extra_counts: vec![3; re.extra_groupings.len()],
        spec: ModelSpec::mixed(lo.model.family, re),
        n_primary: 6,
        tail_fill: 0.0,
    };
    crate::lmm::LmmGroupings::from_cluster_spec_ext(&sized, &primary, &extras)
}

/// The labels and the term of each label of the first grouping.
fn labels(lo: &Lowered) -> (Vec<&str>, Vec<usize>) {
    let g = &lo.re_groups[0];
    (
        g.terms.iter().map(String::as_str).collect(),
        g.term_index.clone(),
    )
}

/// `(f || g)` on a factor is lme4's `(1 | g) + (0 + f | g)`; in the crate's
/// K − 1 coding the factor's term already holds the intercept, so the
/// intercept term merges into it and the block is one unstructured
/// `(1 + f | g)`. `(0 + f | g)` lowers to the same block. A numeric slope
/// beside the factor keeps its own term.
#[test]
fn factor_double_bar_is_one_unstructured_term() {
    let t = factor_table();
    let unstructured = lower("y ~ f + (1 + f | g)", &t, Family::Gaussian).unwrap();
    for f in [
        "y ~ f + (f || g)",
        "y ~ f + (0 + f | g)",
        "y ~ f + (1 | g) + (0 + f | g)",
    ] {
        let lo = lower(f, &t, Family::Gaussian).unwrap();
        assert_eq!(lo.model, unstructured.model, "{f}");
        assert_eq!(lo.model.re.as_ref().unwrap().slopes, vec![1, 2], "{f}");
        assert_eq!(
            labels(&lo),
            (vec!["(Intercept)", "fb", "fc"], vec![0, 0, 0]),
            "{f}"
        );
    }
    // Columns: (Intercept) 0, x 1, fb 2, fc 3.
    let lo = lower("y ~ x + f + (1 + x + f || g)", &t, Family::Gaussian).unwrap();
    assert_eq!(
        lo.model.re.as_ref().unwrap(),
        &ReStructure::intercept().terms([
            ReTerm::intercept().slopes([2, 3]),
            ReTerm::no_intercept().slopes([1]),
        ])
    );
    assert_eq!(
        labels(&lo),
        (vec!["(Intercept)", "fb", "fc", "x"], vec![0, 0, 0, 1])
    );
    assert_eq!(groupings(&lo).structural_theta(), &[3, 6, 8][..]);
}

/// `zerocorr(f | g)` is diagonal over every column of the factor's term:
/// one block of width 3, every covariance held at zero. On a factor this
/// is a different model from `(f || g)`.
#[test]
fn factor_zerocorr_is_diagonal_over_every_column() {
    let t = factor_table();
    let lo = lower("y ~ f + zerocorr(f | g)", &t, Family::Gaussian).unwrap();
    let g = groupings(&lo);
    assert_eq!(g.primary_q, 3);
    assert_eq!(g.structural_theta(), &[1, 2, 4][..]);
    assert_eq!(
        labels(&lo),
        (vec!["(Intercept)", "fb", "fc"], vec![0, 1, 2])
    );
    assert_eq!(
        lo.model.re.as_ref().unwrap(),
        &ReStructure::intercept().slopes([1, 2]).uncorrelated()
    );
    // Without the intercept the factor's K − 1 coding cannot write a
    // diagonal over the factor's levels.
    let err = match lower("y ~ f + zerocorr(0 + f | g)", &t, Family::Gaussian) {
        Ok(_) => panic!("expected a refusal"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("zerocorr"), "{err}");
}

fn sleepstudy() -> Table {
    crate::test_support::csv_table(
        include_str!("../../../../validation/data/empirical/sleepstudy.csv"),
        &["Subject"],
    )
}

/// The formula door's `||` and its split spelling build the matrix door's
/// `Re::intercept().slopes([1]).uncorrelated()` and fit it bit for bit.
#[test]
fn split_terms_match_the_matrix_door() {
    let t = sleepstudy();
    let matrix = ModelSpec::mixed(
        Family::Gaussian,
        ReStructure::intercept().slopes([1]).uncorrelated(),
    );
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let opts = FitOptions::default();
    for f in [
        "Reaction ~ Days + (Days || Subject)",
        "Reaction ~ Days + (1 + Days || Subject)",
        "Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)",
        "Reaction ~ Days + zerocorr(Days | Subject)",
    ] {
        let lo = lower(f, &t, Family::Gaussian).unwrap();
        assert_eq!(lo.model, matrix, "{f}");
        assert_eq!(
            labels(&lo),
            (vec!["(Intercept)", "Days"], vec![0, 1]),
            "{f}"
        );
        let by_formula = crate::fit(&lo.data(), lo.spec(), &opts).unwrap();
        let by_matrix = crate::fit(&lo.data(), &matrix, &opts).unwrap();
        assert!(by_formula.converged(), "{f}");
        assert_eq!(
            by_formula.loglik.to_bits(),
            by_matrix.loglik.to_bits(),
            "{f}"
        );
        assert_eq!(bits(&by_formula.beta), bits(&by_matrix.beta), "{f}");
        assert_eq!(bits(&by_formula.se), bits(&by_matrix.se), "{f}");
        assert_eq!(
            bits(&by_formula.varcorr[0]),
            bits(&by_matrix.varcorr[0]),
            "{f}"
        );
        assert_eq!(bits(&by_formula.ranef), bits(&by_matrix.ranef), "{f}");
    }
}

/// Term order does not matter: the intercept term goes first and the
/// labels follow the block. A five-node AGQ fit of an intercept-free
/// block over three slopes (kernel width 4) is refused, not a panic.
#[test]
fn term_order_and_the_agq_width_cap() {
    let t = sleepstudy();
    let a = lower(
        "Reaction ~ Days + (0 + Days | Subject) + (1 | Subject)",
        &t,
        Family::Gaussian,
    )
    .unwrap();
    let b = lower(
        "Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)",
        &t,
        Family::Gaussian,
    )
    .unwrap();
    assert_eq!(a.model, b.model);
    assert_eq!(labels(&a), (vec!["(Intercept)", "Days"], vec![0, 1]));
    assert_eq!(labels(&a), labels(&b));
    // An intercept-free block labels its visible column alone, and its
    // conditional modes have one value per level.
    let lo = lower(
        "Reaction ~ Days + (0 + Days | Subject)",
        &t,
        Family::Gaussian,
    )
    .unwrap();
    assert_eq!(labels(&lo), (vec!["Days"], vec![0]));
    let fit = crate::fit(&lo.data(), lo.spec(), &FitOptions::default()).unwrap();
    let blocks = label_ranef(&fit, &lo.re_groups).unwrap();
    assert_eq!(blocks[0].terms, ["Days"]);
    assert_eq!(blocks[0].values.len(), blocks[0].levels.len());

    let mut t = factor_table();
    let x = |k: usize| -> Vec<f64> { (0..t.n).map(|i| ((i * k) % 7) as f64 / 7.0).collect() };
    let (x1, x2, x3) = (x(3), x(5), x(11));
    let yb: Vec<f64> = (0..t.n).map(|i| ((i * 13) % 5 < 2) as u8 as f64).collect();
    t.columns.push(("x1".into(), Column::Numeric(x1)));
    t.columns.push(("x2".into(), Column::Numeric(x2)));
    t.columns.push(("x3".into(), Column::Numeric(x3)));
    t.columns.push(("yb".into(), Column::Numeric(yb)));
    let opts = FitOptions {
        nagq: 5,
        ..FitOptions::default()
    };
    let binomial = Family::Binomial {
        link: crate::BinomialLink::Logit,
    };
    match super::fit(
        "yb ~ x1 + x2 + x3 + (0 + x1 + x2 + x3 | g)",
        &t,
        binomial,
        &opts,
    ) {
        Err(Error::Fit(crate::FitError::Unsupported(msg))) => {
            assert!(msg.contains("q_p=4"), "{msg}")
        }
        Err(e) => panic!("expected FitError::Unsupported, got {e}"),
        Ok(_) => panic!("expected FitError::Unsupported, got a fit"),
    }
}

/// `y ~ 1 + (x | g)`: `x` is in no fixed term, so it is the one
/// random-effect-only column, read by slope index `p + 0`.
#[test]
fn slope_outside_the_design_is_a_re_column() {
    let t = factor_table();
    let lo = lower("y ~ 1 + (x | g)", &t, Family::Gaussian).unwrap();
    assert_eq!((lo.p, lo.re_p), (1, 1));
    let Some(Column::Numeric(x)) = t.get("x") else {
        unreachable!()
    };
    assert_eq!(&lo.re_columns, x);
    assert_eq!(lo.model.re.as_ref().unwrap().slopes, vec![1]);
    assert_eq!(lo.re_groups[0].terms, ["(Intercept)", "x"]);
}

/// One column set per variable: two groupings that name `x` share it, and
/// a variable in `X` only inside an interaction is still a column of its
/// own, as in lme4.
#[test]
fn one_re_column_per_variable() {
    let t = factor_table();
    let lo = lower("y ~ 1 + (x | g) + (x | f)", &t, Family::Gaussian).unwrap();
    assert_eq!((lo.p, lo.re_p), (1, 1));
    let re = lo.model.re.as_ref().unwrap();
    assert_eq!(re.slopes, vec![1]);
    assert_eq!(re.extra_groupings[0].slopes, vec![1]);

    let lo = lower("y ~ x:f + (x | g)", &t, Family::Gaussian).unwrap();
    assert_eq!(lo.re_p, 1);
    assert_eq!(lo.model.re.as_ref().unwrap().slopes, vec![lo.p as ColumnId]);
}

/// A column of the wrong length or a factor code past its levels is an
/// error, not a panic or a fit on zero-padded rows.
#[test]
fn ragged_columns_and_bad_codes_are_refused() {
    let y = || ("y".to_string(), Column::Numeric(vec![1.0, 2.0, 4.0]));
    for xlen in [2, 4] {
        let t = Table {
            columns: vec![y(), ("x".into(), Column::Numeric(vec![1.0; xlen]))],
            n: 3,
        };
        match lower("y ~ x", &t, Family::Gaussian) {
            Err(Error::Fit(FitError::Shape(m))) => assert!(m.contains("\"x\""), "{m}"),
            other => panic!("x of {xlen} rows: {:?}", other.err()),
        }
    }
    let t = Table {
        columns: vec![
            y(),
            (
                "f".into(),
                Column::Factor {
                    levels: vec!["a".into(), "b".into()],
                    codes: vec![0, 1, 2],
                },
            ),
        ],
        n: 3,
    };
    match lower("y ~ f", &t, Family::Gaussian) {
        Err(Error::Fit(FitError::Shape(m))) => assert!(m.contains("code 2"), "{m}"),
        other => panic!("bad code: {:?}", other.err()),
    }
}

/// `y ~ 1 + (f | g)`: the factor's K − 1 treatment dummies are the
/// random-effect-only columns, the same columns and names `y ~ f` puts in
/// its fixed design, and `(0 + f | g)` takes the intercept as it does with
/// `f` a fixed term.
#[test]
fn factor_slope_outside_the_design() {
    let t = factor_table();
    let lo = lower("y ~ 1 + (f | g)", &t, Family::Gaussian).unwrap();
    assert_eq!((lo.p, lo.re_p), (1, 2));
    let fixed = lower("y ~ f", &t, Family::Gaussian).unwrap();
    let dummies: Vec<f64> = (0..t.n)
        .flat_map(|i| fixed.x[i * 3 + 1..i * 3 + 3].to_vec())
        .collect();
    assert_eq!(lo.re_columns, dummies);
    assert_eq!(fixed.col_names[1..], ["fb", "fc"]);
    assert_eq!(lo.model.re.as_ref().unwrap().slopes, vec![1, 2]);
    assert_eq!(
        labels(&lo),
        (vec!["(Intercept)", "fb", "fc"], vec![0, 0, 0])
    );
    let no_intercept = lower("y ~ 1 + (0 + f | g)", &t, Family::Gaussian).unwrap();
    assert_eq!(no_intercept.model, lo.model);
}

/// A `cbind()` row with a negative count or no trials is named by row,
/// as a field a port can renumber; an error without a row has none.
#[test]
fn cbind_refusals_carry_their_row() {
    let table = |s: Vec<f64>, f: Vec<f64>| Table {
        columns: vec![
            ("s".into(), Column::Numeric(s)),
            ("f".into(), Column::Numeric(f)),
            ("x".into(), Column::Numeric(vec![0.0, 1.0, 2.0, 3.0])),
        ],
        n: 4,
    };
    let binomial = Family::Binomial {
        link: crate::BinomialLink::Logit,
    };
    let refused = |s, f| match lower("cbind(s, f) ~ x", &table(s, f), binomial) {
        Err(e) => e,
        Ok(_) => panic!("accepted"),
    };
    let negative = refused(vec![1.0, 2.0, -1.0, 0.0], vec![1.0; 4]);
    assert!(
        matches!(negative, Error::NegativeCount { row: 2 }),
        "{negative}"
    );
    assert_eq!(negative.row(), Some(2));
    assert_eq!(
        negative.to_string_with_row(3),
        "cbind(): successes and failures must be non-negative at every row; row 3 is not"
    );
    let empty = refused(vec![1.0, 0.0, 1.0, 1.0], vec![1.0, 0.0, 1.0, 1.0]);
    assert!(matches!(empty, Error::ZeroTrials { row: 1 }), "{empty}");
    assert_eq!(empty.row(), Some(1));
    assert_eq!(empty.to_string_with_row(1), empty.to_string());
    let Err(gaussian) = lower(
        "cbind(s, f) ~ x",
        &table(vec![1.0; 4], vec![1.0; 4]),
        Family::Gaussian,
    ) else {
        panic!("cbind() on a Gaussian family accepted");
    };
    assert!(matches!(gaussian, Error::CbindNeedsBinomial), "{gaussian}");
    assert_eq!(gaussian.row(), None);
    assert_eq!(gaussian.to_string_with_row(7), gaussian.to_string());
}

/// [`numeric_only_columns`] names the response and the offset column, and
/// both sides of a `cbind()` response.
#[cfg(feature = "orchestrate")]
#[test]
fn numeric_only_columns_names_response_and_offset() {
    let ast = parse("y ~ x + (1 | g) + offset(o)").unwrap();
    let available = strs(&["y", "x", "g", "o"]);
    let mut names = numeric_only_columns(&ast, &available);
    names.sort();
    assert_eq!(names, vec!["o".to_string(), "y".to_string()]);

    let cbind_ast = parse("cbind(s, f) ~ x").unwrap();
    let cbind_available = strs(&["s", "f", "x"]);
    let mut cbind_names = numeric_only_columns(&cbind_ast, &cbind_available);
    cbind_names.sort();
    assert_eq!(cbind_names, vec!["f".to_string(), "s".to_string()]);
}
