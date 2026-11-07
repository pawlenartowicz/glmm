//! Thin staticlib R drives via src/Makevars: every `#[extendr]` function lives
//! in the workspace's `glmm-r` crate; this only stamps the R package's module
//! name onto that surface.
use extendr_api::prelude::*;

extendr_module! {
    mod fastglmm;
    use glmm_r;
}
