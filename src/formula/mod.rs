//! R-style formula frontend for the `glmm` kernel.
//!
//! Turns `"y ~ x1 * x2 + (1 + x1 | g)"` **plus a data table** into the numeric
//! inputs [`crate::fit_cold`] consumes — a row-major design matrix, a structure-only
//! [`crate::ModelSpec`], per-row [`crate::GroupIds`], and a defaulted
//! [`crate::FitOptions`]. Two stages, cut on whether data is needed:
//!
//! - [`parse`] is pure and data-free: string → [`ParsedFormula`] (the AST). `*` is
//!   already desugared into main effects + interactions here, `A/B` into a nesting
//!   relation.
//! - [`materialize`] is the only data-dependent stage: it discovers factor levels,
//!   builds the numeric design, assigns column ids, finalizes the `ModelSpec`, and
//!   builds `GroupIds`.
//!
//! [`lower`] runs both in one call:
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
//! let lo = glmm::formula::lower("y ~ x*z + (1|g)", &table, Family::Gaussian)?;
//! let fit = glmm::fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
//! # let _ = fit;
//! # Ok(())
//! # }
//! ```
//!
//! Gated by the `formula` cargo feature (on by default). Consumers that need the
//! formula-free kernel — the parse-once/fit-many hot path — take
//! `default-features = false` and link no `regex`.

mod error;
mod fixed_part;
pub(crate) mod hypotheses;
mod materialize;
mod parse;

#[cfg(all(test, feature = "oracle-tests"))]
mod hypotheses_oracle_tests;
#[cfg(test)]
pub(crate) mod hypotheses_tests;

pub use error::{Error, ParseError};
// Crate-private: `orchestrate::same_fixed_part` is the only caller so far.
// removed by P31-T6
#[cfg_attr(not(feature = "orchestrate"), allow(unused_imports))]
pub(crate) use fixed_part::{fixed_part_relation, FixedPartRelation};
pub use materialize::{
    label_ranef, lower, materialize, numeric_only_columns, referenced_columns, unresolved_columns,
    Column, FixedTerm, Lowered, RanefBlock, ReGroupInfo, Table,
};
pub use parse::{parse, ParsedFormula, RandomEffect, Term};
