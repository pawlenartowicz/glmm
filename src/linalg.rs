//! Linear-algebra primitives shared by the sim and fit layers.

/// Lower Cholesky factor of a symmetric PSD `q×q` matrix (row-major in/out).
/// validate() guarantees PSD, so a zero pivot is treated as exact 0 (its
/// below-diagonal column entries become 0).
pub fn chol_lower(a: &[f64], q: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; q * q];
    for j in 0..q {
        let mut diag = a[j * q + j];
        for k in 0..j {
            diag -= l[j * q + k] * l[j * q + k];
        }
        let ljj = diag.max(0.0).sqrt();
        l[j * q + j] = ljj;
        for i in (j + 1)..q {
            if ljj > 0.0 {
                let mut s = a[i * q + j];
                for k in 0..j {
                    s -= l[i * q + k] * l[j * q + k];
                }
                l[i * q + j] = s / ljj;
            }
        }
    }
    l
}
