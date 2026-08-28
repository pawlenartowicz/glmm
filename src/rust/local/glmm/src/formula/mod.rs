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
//! ```ignore
//! let lo = glmm::formula::lower("y ~ x*z + (1|g)", &table, Family::Gaussian)?;
//! let fit = glmm::fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
//! ```
//!
//! Gated by the `formula` cargo feature (on by default). Consumers that need the
//! formula-free kernel — the parse-once/fit-many hot path — take
//! `default-features = false` and link no `regex`.

mod error;
mod materialize;
mod parse;

pub use error::{Error, ParseError};
pub use materialize::{
    label_ranef, lower, materialize, Column, Lowered, RanefBlock, ReGroupInfo, Table,
};
pub use parse::{parse, ParsedFormula, RandomEffect, Term};
