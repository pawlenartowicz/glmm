//! How the fixed parts of two formulas relate, read from [`parse`](super::parse())
//! alone, so no data is needed.

use std::collections::BTreeSet;

use super::parse::{ParsedFormula, Term};

/// How the fixed parts of two formulas relate (see [`fixed_part_relation`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "orchestrate"), allow(dead_code))] // removed by P31-T6
pub(crate) enum FixedPartRelation {
    /// The same fixed terms and the same fixed intercept.
    Same,
    /// The first formula's fixed part lies strictly inside the second's.
    FirstInSecond,
    /// The second formula's fixed part lies strictly inside the first's.
    SecondInFirst,
    /// Neither fixed part contains the other.
    Neither,
}

/// Compares the fixed parts of two parsed formulas.
///
/// Each fixed part is a set of terms, and each term the set of its variable
/// names, as R's `terms.formula` codes it: the order of the terms does not
/// matter, `a:b` and `b:a` are one term, and `a * b` is `a + b + a:b` (the
/// parser has already expanded `*`). The fixed intercept
/// ([`ParsedFormula::has_intercept`]) is one more term, the one with no
/// variables, so `y ~ 0 + x` lies inside `y ~ x`, and `y ~ 1` against
/// `y ~ 0 + f` is [`FixedPartRelation::Neither`] although the two column
/// spaces are nested when `f` is a factor: the answer reads the formulas,
/// never the data.
///
/// The response, the offset and the random part are not read.
#[cfg_attr(not(feature = "orchestrate"), allow(dead_code))] // removed by P31-T6
pub(crate) fn fixed_part_relation(a: &ParsedFormula, b: &ParsedFormula) -> FixedPartRelation {
    let (a, b) = (fixed_terms(a), fixed_terms(b));
    match (a.is_subset(&b), b.is_subset(&a)) {
        (true, true) => FixedPartRelation::Same,
        (true, false) => FixedPartRelation::FirstInSecond,
        (false, true) => FixedPartRelation::SecondInFirst,
        (false, false) => FixedPartRelation::Neither,
    }
}

/// The fixed terms as sets of variable names, the intercept as the empty set.
#[cfg_attr(not(feature = "orchestrate"), allow(dead_code))] // removed by P31-T6
fn fixed_terms(f: &ParsedFormula) -> BTreeSet<BTreeSet<&str>> {
    let mut terms: BTreeSet<BTreeSet<&str>> = f
        .terms
        .iter()
        .map(|t| match t {
            Term::Main { name } => BTreeSet::from([name.as_str()]),
            Term::Interaction { vars } => vars.iter().map(String::as_str).collect(),
        })
        .collect();
    if f.has_intercept {
        terms.insert(BTreeSet::new());
    }
    terms
}

#[cfg(test)]
mod tests {
    use super::FixedPartRelation::{FirstInSecond, Neither, Same, SecondInFirst};
    use super::*;
    use crate::formula::parse;

    /// The relation of `a` to `b`, after checking that swapping the two
    /// formulas gives the mirrored answer.
    fn rel(a: &str, b: &str) -> FixedPartRelation {
        let (pa, pb) = (parse(a).unwrap(), parse(b).unwrap());
        let forward = fixed_part_relation(&pa, &pb);
        let mirrored = match forward {
            FirstInSecond => SecondInFirst,
            SecondInFirst => FirstInSecond,
            other => other,
        };
        assert_eq!(fixed_part_relation(&pb, &pa), mirrored, "{a} against {b}");
        forward
    }

    #[test]
    fn term_order_and_variable_order_do_not_matter() {
        assert_eq!(rel("y ~ a + b + c", "y ~ c + a + b"), Same);
        assert_eq!(rel("y ~ a:b", "y ~ b:a"), Same);
        assert_eq!(rel("y ~ a + b + a:b:c", "y ~ c:b:a + b + a"), Same);
    }

    #[test]
    fn star_equals_its_expansion() {
        assert_eq!(rel("y ~ a * b", "y ~ a + b + a:b"), Same);
        assert_eq!(rel("y ~ a * b", "y ~ b:a + b + a"), Same);
        assert_eq!(rel("y ~ a + b", "y ~ a * b"), FirstInSecond);
    }

    #[test]
    fn the_response_offset_and_random_part_are_not_read() {
        assert_eq!(rel("y ~ x + (1 | g)", "y ~ x + (1 + x | h)"), Same);
        assert_eq!(rel("y ~ x", "y ~ x + (1 | g/h)"), Same);
        assert_eq!(rel("y ~ x + offset(z)", "w ~ x"), Same);
    }

    #[test]
    fn different_term_sets_differ() {
        assert_eq!(rel("y ~ a + b", "y ~ a + c"), Neither);
        assert_eq!(rel("y ~ a", "y ~ a + b"), FirstInSecond);
        assert_eq!(rel("y ~ a:b", "y ~ a * b"), FirstInSecond);
    }

    #[test]
    fn the_fixed_intercept_is_read() {
        assert_eq!(rel("y ~ x", "y ~ 0 + x"), SecondInFirst);
        assert_eq!(rel("y ~ x - 1", "y ~ 0 + x"), Same);
        assert_eq!(rel("y ~ 1 + x", "y ~ x"), Same);
        assert_eq!(rel("y ~ -1 + x", "y ~ 0 + x"), Same);
        assert_eq!(rel("y ~ 1", "y ~ 0 + f"), Neither);
    }
}
