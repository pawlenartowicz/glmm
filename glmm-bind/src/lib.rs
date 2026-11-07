//! glmm-bind: the language-neutral half of every glmm port.
//!
//! A port converts its native values into a [`Record`] of [`Value`]s, calls
//! [`call`] with a registered function name, and converts the result (or
//! [`BindError`]) back. Argument checks, defaults, result records and the
//! error codes live here once; the engine work is `glmm::orchestrate`'s.

#![forbid(unsafe_code)]
// Counts cross as i64 and f64; the casts are range-checked where it matters
// (coerce.rs).
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

mod coerce;
mod error;
mod fns;
mod inputs;
mod notes;
mod records;
mod registry;
mod summary;
mod value;

pub use error::BindError;
/// The bound on `nsim` (`compare`, `compare_fits`), for a port's own message
/// on a value it cannot pass.
pub use glmm::orchestrate::MAX_NSIM;
pub use registry::{call, registry, DefaultValue, FnSpec, Param};
pub use value::{Record, Value};
