//! R-style formula frontend for the `glmm` kernel.
//!
//! Turns `"y ~ x1 * x2 + (1 + x1 | g)"` **plus a data table** into the numeric
//! inputs [`crate::fit`] consumes — a row-major design matrix, a structure-only
//! [`crate::ModelSpec`], per-row [`crate::GroupIds`], and the weights and offset
//! the formula carries. Two stages, cut on whether data is needed:
//!
//! - [`parse`] is pure and data-free: string → [`ParsedFormula`] (the AST). `*` is
//!   already desugared into main effects + interactions here, `A/B` into a nesting
//!   relation.
//! - [`materialize`] is the only data-dependent stage: it discovers factor levels,
//!   builds the numeric design, assigns column ids, finalizes the `ModelSpec`, and
//!   builds `GroupIds`.
//!
//! [`lower`] runs both in one call, and [`fit()`] lowers and fits in one
//! call, returning a [`Model`]:
//! ```
//! # use glmm::formula::{Column, Table};
//! # use glmm::Family;
//! # fn main() -> Result<(), glmm::formula::Error> {
//! # let num = |v: &[f64]| Column::Numeric(v.to_vec());
//! # let g: Vec<String> = ["a", "a", "b", "b", "c", "c", "d", "d"].map(String::from).into();
//! # let table = Table {
//! #     columns: vec![
//! #         ("y".into(), num(&[1.2, 0.7, 2.3, 1.9, 0.4, 1.1, 2.8, 2.0])),
//! #         ("x".into(), num(&[0.1, 0.5, 0.9, 0.2, 0.7, 0.3, 0.8, 0.4])),
//! #         ("z".into(), num(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])),
//! #         ("g".into(), Column::factor_from_labels(&g)),
//! #     ],
//! #     n: 8,
//! # };
//! let opts = glmm::FitOptions::default();
//! let model = glmm::formula::fit("y ~ x*z + (1|g)", &table, Family::Gaussian, &opts)?;
//! assert_eq!(model.fit().beta.len(), model.lowered().p);
//! # Ok(())
//! # }
//! ```
//!
//! Gated by the `formula` cargo feature (on by default). Consumers that need the
//! formula-free kernel — the parse-once/fit-many hot path — take
//! `default-features = false` and link no `regex`.

pub mod anova;
#[cfg(test)]
mod anova_tests;
mod error;
mod fixed_part;
pub(crate) mod hypotheses;
mod materialize;
pub mod pairwise;
mod parse;

#[cfg(all(test, feature = "oracle-tests"))]
mod hypotheses_oracle_tests;
#[cfg(test)]
pub(crate) mod hypotheses_tests;
#[cfg(all(test, feature = "oracle-tests"))]
mod predict_oracle_tests;
#[cfg(test)]
mod predict_tests;

pub use error::{Error, ParseError};
// Crate-private: the nesting check of `anova::nesting` and
// `orchestrate::same_fixed_part` call it.
pub(crate) use fixed_part::{fixed_part_relation, FixedPartRelation};
// Crate-private: `orchestrate::run_model_matrix`,
// `orchestrate::model_matrix_columns`, `orchestrate::term_labels` and
// `orchestrate::predict` call them.
#[cfg(feature = "orchestrate")]
pub(crate) use materialize::{design_columns, fixed_design, lower_new, r_terms};
pub use materialize::{
    fit, fit_lowered, label_ranef, lower, materialize, numeric_only_columns, predict,
    referenced_columns, unresolved_columns, Column, FixedTerm, Lowered, Model, RanefBlock,
    ReGroupInfo, Table,
};
pub use pairwise::{
    pairwise, Adjust, By, Pairwise, PairwiseInterval, PairwiseIntervals, PairwiseRow, RatioKind,
};
pub use parse::{parse, ParsedFormula, RandomEffect, Term};
