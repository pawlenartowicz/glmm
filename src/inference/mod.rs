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
pub(crate) mod tails;

// The reader of the frozen inference references, for in-crate tests on
// `pub(crate)` items. `dead_code` is allowed because each test file uses only
// part of it, and its consumers arrive with the steps that freeze references.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod test_refs;
#[cfg(test)]
mod test_refs_tests;
