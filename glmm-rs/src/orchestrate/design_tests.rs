//! Tests of the design and term readers.

use super::*;
use std::collections::HashMap;

/// `run_model_matrix` gives the design `run_fit` fits: a level no row has
/// is dropped, the unused first one included, so the base is "a".
#[test]
fn model_matrix_drops_unused_levels() {
    let factors = HashMap::from([(
        "f".to_string(),
        (vec!["zzz".into(), "a".into(), "b".into()], vec![1, 2, 1, 2]),
    )]);
    let (x, n, p, names) = run_model_matrix("~ f", HashMap::new(), factors, 4).unwrap();
    assert_eq!(names, ["(Intercept)", "fb"]);
    assert_eq!((n, p), (4, 2));
    assert_eq!(x, [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn same_fixed_part_reads_the_fixed_terms_and_intercept_only() {
    assert_eq!(
        same_fixed_part("y ~ a * b + (1 | g)", "y ~ b:a + b + a"),
        Ok(true)
    );
    // The response and the offset are not read.
    assert_eq!(same_fixed_part("y ~ x + offset(z)", "w ~ x"), Ok(true));
    assert_eq!(same_fixed_part("y ~ x", "y ~ 0 + x"), Ok(false));
    let err = same_fixed_part("y ~ x", "y ~ x - z").unwrap_err();
    assert!(err.contains("\"y ~ x - z\""), "{err}");
    let err = same_fixed_part("y ~ x + ", "y ~ x").unwrap_err();
    assert!(err.contains("ends with '+'"), "{err}");
}

#[test]
fn term_labels_expand_groups_and_drop_random_terms_and_the_intercept() {
    let cases: [(&str, &[&str], bool); 7] = [
        (
            "y ~ age + sex*_ssd_text + (1 | id)",
            &["age", "sex", "_ssd_text", "sex:_ssd_text"],
            true,
        ),
        ("y ~ a*(b + c)", &["a", "b", "c", "a:b", "a:c"], true),
        ("y ~ 0 + a + b", &["a", "b"], false),
        ("y ~ a + b - 1", &["a", "b"], false),
        ("y ~ x + offset(o) + (1 + x | g) + (1 | h)", &["x"], true),
        // Variables are ordered by first appearance, whatever the
        // interaction's own order; the response is optional.
        ("~ a + b + b:a", &["a", "b", "a:b"], true),
        (
            "y ~ c:b:a + x + a*b",
            &["x", "a", "b", "b:a", "c:b:a"],
            true,
        ),
    ];
    for (formula, labels, intercept) in cases {
        let (got, has_intercept) = term_labels(formula).expect(formula);
        assert_eq!(got, labels, "{formula}");
        assert_eq!(has_intercept, intercept, "{formula}");
    }
    assert!(term_labels("y ~ x + (1 | g").is_err());
}
