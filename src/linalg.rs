//! Linear-algebra primitives shared by the sim and fit layers.

/// In-place lower Crout Cholesky of a `q×q` block stored row-major in `blk`
/// (lower triangle read; on return the lower triangle holds L). Returns false on
/// a non-positive pivot — the module's failure surface. Shared by the GLMM
/// block solver (`glmm::workspace::glmm_block_chol` call sites) and the LMM
/// per-family factor in `lmm::reml_deviance`.
#[inline]
pub(crate) fn block_chol(blk: &mut [f64], q: usize) -> bool {
    for j in 0..q {
        let mut d = blk[j * q + j];
        for k in 0..j {
            d -= blk[j * q + k] * blk[j * q + k];
        }
        if !(d.is_finite() && d > 0.0) {
            return false;
        }
        let l = d.sqrt();
        blk[j * q + j] = l;
        for i in (j + 1)..q {
            let mut v = blk[i * q + j];
            for k in 0..j {
                v -= blk[i * q + k] * blk[j * q + k];
            }
            blk[i * q + j] = v / l;
        }
    }
    true
}
