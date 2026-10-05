//! Linear-algebra primitives shared by the sim and fit layers.

use crate::scalar::Scalar;

/// In-place lower Crout Cholesky of a `q×q` block stored row-major in `blk`
/// (lower triangle read; on return the lower triangle holds L). Returns false on
/// a non-positive pivot — the module's failure surface. Shared by the GLMM
/// block solver (`glmm::workspace::glmm_block_chol` call sites) and the LMM
/// per-family factor in `lmm::reml_deviance`.
#[inline]
pub(crate) fn block_chol<T: Scalar>(blk: &mut [T], q: usize) -> bool {
    for j in 0..q {
        let mut d = blk[j * q + j];
        for k in 0..j {
            d -= blk[j * q + k] * blk[j * q + k];
        }
        if !(d.value().is_finite() && d.value() > 0.0) {
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

/// Stack scratch of [`SeqLlt`]'s factor and solve, in bytes.
const SEQ_LLT_STACK: usize = 2048;

/// Lower Cholesky `A = LLᵀ` of a dense SPD matrix on the serial path:
/// `faer::Mat::llt`'s own steps (`Llt::new`: lower-triangle copy into a zeroed
/// `Mat`, `cholesky_in_place` with the default regularization, strict upper
/// triangle zeroed) and its solve, but at `Par::Seq`. `Mat::llt` reads
/// `faer::get_global_parallelism()` on every call, which on a native build with
/// faer's default `rayon` feature is `Par::rayon(0)`: the first call spawns
/// rayon's global pool although this crate's own `parallel` feature is off, and
/// a dimension past faer's blocking threshold hands the recursion to it. The
/// kernels' hot Choleskys already take `Par` explicitly
/// (`scalar::Scalar::chol_lower`, the PIRLS border); `SeqLlt` takes the same
/// rule to the GLM IRLS and OLS β solves, the GLMM post-fit Choleskys
/// (`fit_glmm`, `glmm::se::rx_cov_into`, `glmm::se::joint_hessian_cov_at`,
/// `glmm::se::packed_schur_fill`), `lmm::joint_wald_chi_sq` and the LMM
/// random-effect recovery (`lmm::recover_ranef`). Below faer's blocking
/// threshold the arithmetic is `Mat::llt`'s.
pub(crate) struct SeqLlt {
    l: faer::Mat<f64>,
}

impl SeqLlt {
    /// Factor the lower triangle of `a`; `None` where faer's `llt` returns
    /// an error (not positive definite).
    pub(crate) fn new(a: faer::MatRef<'_, f64>) -> Option<Self> {
        use faer::dyn_stack::{MemBuffer, MemStack};
        use faer::linalg::cholesky::llt::factor::{cholesky_in_place, cholesky_in_place_scratch};
        let n = a.nrows();
        let mut l = faer::Mat::<f64>::zeros(n, n);
        l.copy_from_triangular_lower(a);
        // Scratch (one `n`-vector) off a stack buffer when it fits (the
        // p×p and m×m post-fit factors), a heap `MemBuffer` otherwise (the
        // larger RE-recovery factors); the factor is the same.
        let req = cholesky_in_place_scratch::<f64>(n, faer::Par::Seq, Default::default());
        let mut buf = [core::mem::MaybeUninit::<u8>::uninit(); SEQ_LLT_STACK];
        let mut heap;
        let stack = if req.unaligned_bytes_required() <= SEQ_LLT_STACK {
            MemStack::new(&mut buf)
        } else {
            heap = MemBuffer::new(req);
            MemStack::new(&mut heap)
        };
        cholesky_in_place(
            l.as_mut(),
            Default::default(),
            faer::Par::Seq,
            stack,
            Default::default(),
        )
        .ok()?;
        for j in 0..n {
            for i in 0..j {
                l[(i, j)] = 0.0;
            }
        }
        Some(Self { l })
    }

    /// The lower factor `L`.
    #[allow(non_snake_case)]
    pub(crate) fn L(&self) -> faer::MatRef<'_, f64> {
        self.l.as_ref()
    }

    /// `rhs ← A⁻¹·rhs`, faer's `Llt::solve_in_place` at `Par::Seq`.
    pub(crate) fn solve_in_place(&self, rhs: faer::MatMut<'_, f64>) {
        use faer::dyn_stack::{MemBuffer, MemStack};
        use faer::linalg::cholesky::llt::solve::{
            solve_in_place_scratch, solve_in_place_with_conj,
        };
        let req = solve_in_place_scratch::<f64>(self.l.nrows(), rhs.ncols(), faer::Par::Seq);
        let mut buf = [core::mem::MaybeUninit::<u8>::uninit(); SEQ_LLT_STACK];
        let mut heap;
        let stack = if req.unaligned_bytes_required() <= SEQ_LLT_STACK {
            MemStack::new(&mut buf)
        } else {
            heap = MemBuffer::new(req);
            MemStack::new(&mut heap)
        };
        solve_in_place_with_conj(self.l.as_ref(), faer::Conj::No, rhs, faer::Par::Seq, stack);
    }
}
