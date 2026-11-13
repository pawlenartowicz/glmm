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
/// ([`chol_lower`] in the LMM kernel, the β-Schur border in `glmm::pirls`);
/// `SeqLlt` takes the same
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

/// Lower Cholesky of the symmetric `dim×dim` `a` (column-major, lower
/// triangle read) into `l_out` (column-major, lower triangle written;
/// strictly upper untouched), by faer's kernel at `Par::Seq`. `false` on a
/// non-positive or non-finite pivot. Out-of-place because the LMM kernel
/// keeps the unfactored matrix for the conditional-mode recovery pass.
pub(crate) fn chol_lower(a: &[f64], dim: usize, l_out: &mut [f64]) -> bool {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::cholesky::llt::factor::{
        cholesky_in_place, cholesky_in_place_scratch, LltRegularization,
    };
    use faer::{Par, Spec};
    // Copy A's lower triangle into `l_out` and factor THAT in place instead of calling
    // `MatRef::llt`. `.llt` reads `faer::get_global_parallelism()` on every call
    // (faer 0.24.4 `linalg/solvers.rs:789`), which is `Par::rayon(0)` in any build with
    // faer's default `rayon` feature: the first call spawns rayon's whole global pool
    // and every later call with dim > 64 hands the blocked recursion to it.
    // `cholesky_in_place` takes `Par` as an argument — the same rule the PIRLS
    // Choleskys and [`syrk_lower_sub`]'s GEMM already follow.
    // The zero-fill mirrors `.llt`'s own `Mat::zeros` + `copy_from_triangular_lower`:
    // faer's kernel also writes the columns above the diagonal, so the input's upper
    // triangle must be zero for the factor to match what `.llt` produced.
    l_out[..dim * dim].fill(0.0);
    for j in 0..dim {
        let col = j * dim;
        l_out[col + j..col + dim].copy_from_slice(&a[col + j..col + dim]);
    }
    // The factor's scratch comes off a stack buffer when it fits (it does
    // on every LMM kernel's `q`-sized block: 132 B at the loop shapes'
    // dims), so a per-eval call does not allocate. A larger request falls
    // back to a heap `MemBuffer`; the factor is the same either way.
    let req = cholesky_in_place_scratch::<f64>(dim, Par::Seq, Spec::default());
    let l = faer::MatMut::from_column_major_slice_mut(&mut l_out[..dim * dim], dim, dim);
    const STACK_SCRATCH: usize = 1024;
    if req.unaligned_bytes_required() <= STACK_SCRATCH {
        let mut buf = [core::mem::MaybeUninit::<u8>::uninit(); STACK_SCRATCH];
        cholesky_in_place(
            l,
            LltRegularization::default(),
            Par::Seq,
            MemStack::new(&mut buf),
            Spec::default(),
        )
        .is_ok()
    } else {
        let mut mem = MemBuffer::new(req);
        cholesky_in_place(
            l,
            LltRegularization::default(),
            Par::Seq,
            MemStack::new(&mut mem),
            Spec::default(),
        )
        .is_ok()
    }
}

/// `tail(lower) -= bt · btᵀ` via faer's blocked triangular `matmul` at
/// `Par::Seq`. `tail` is column-major `t_dim×t_dim` (lower triangle touched
/// and written; faer neither reads nor writes its strict upper triangle), `bt`
/// column-major `t_dim×w_tot`. The LMM kernel's tail downdate.
pub(crate) fn syrk_lower_sub(bt: &[f64], t_dim: usize, w_tot: usize, tail: &mut [f64]) {
    use faer::linalg::matmul::triangular::{matmul, BlockStructure};
    let bt = faer::MatRef::from_column_major_slice(&bt[..t_dim * w_tot], t_dim, w_tot);
    let tail = faer::MatMut::from_column_major_slice_mut(&mut tail[..t_dim * t_dim], t_dim, t_dim);
    matmul(
        tail,
        BlockStructure::TriangularLower,
        faer::Accum::Add,
        bt,
        BlockStructure::Rectangular,
        bt.transpose(),
        BlockStructure::Rectangular,
        -1.0,
        faer::Par::Seq,
    );
}
