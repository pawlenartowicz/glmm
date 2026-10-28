// The inference reference reader, shared with the crate's in-crate tests: those
// gate `pub(crate)` items and cannot reach a reader under `tests/`, so the one
// implementation lives in the crate and integration tests include it verbatim.
include!("../../src/inference/test_refs.rs");
