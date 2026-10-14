//! Hypothesis tests on fitted models: the arithmetic behind t, F and χ² tests
//! of fixed effects and variance components, and the numerical helpers they
//! share.
//!
//! The public items of this module are outside the crate's semver promise
//! until the 1.0 surface review: their names and signatures may change in any
//! release before then.

pub(crate) mod contest;
pub mod contrast;
pub(crate) mod evd;
pub(crate) mod lr;
mod reference;
pub(crate) mod rlrt;
pub(crate) mod tails;
pub(crate) mod wald;

pub use lr::{LrNote, LrTest};
pub use reference::{DenDf, Objective, Reference};
pub use wald::{LinearKind, LinearNote, LinearTest};

// The reader of the frozen inference references, for in-crate tests on
// `pub(crate)` items. `dead_code` is allowed because each test file uses only
// part of it, and its consumers arrive with the steps that freeze references.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod test_refs;
#[cfg(test)]
mod test_refs_tests;
// The exact-RLRT references, typed, over `test_refs`. `dead_code` is allowed
// because the default tier reads only `bands` and `csv`: `entries`, `load`
// and `design` have readers only in the oracle and Monte Carlo tiers.
#[cfg(test)]
#[allow(dead_code)]
mod rlrt_refs;
// The exact RLRT's tests: the default matrix, and the gates against RLRsim's
// frozen references.
#[cfg(all(test, feature = "oracle-tests", feature = "formula"))]
mod rlrt_oracle_tests;
// Gate 7, the level study (Monte Carlo tier). It reads its designs
// through the formula lowering, which `montecarlo-tests` implies.
#[cfg(all(test, feature = "montecarlo-tests"))]
mod rlrt_montecarlo_tests;
// The Monte Carlo calibration tier's design: the grid of cells, the
// generating truth, the seeds and the generator of its datasets, and its
// default-matrix tests (the grid, the layout, the pinned first datasets).
#[cfg(all(test, feature = "formula"))]
mod mc_data;
#[cfg(all(test, feature = "formula"))]
mod mc_data_tests;
// The Monte Carlo calibration tier's driver: per cell and dataset every
// quantity of the tier, counted, printed and written per cell.
#[cfg(all(test, feature = "montecarlo-tests"))]
mod mc_calibration_tests;
#[cfg(test)]
mod rlrt_tests;
