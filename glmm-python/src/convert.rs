//! Pure translation helpers — no `pyo3` types anywhere in this file, so these
//! run under plain `cargo test` with no Python interpreter involved.

use glmm::{BinomialLink, Family, GammaLink, NegBinomialLink, PoissonLink};

/// Maps the Python API's `family`/`link` strings to `glmm::Family`. `family`
/// and `link` are already validated against the full (0.1.1-aware) spec table
/// by `glmm/__init__.py` before this ever runs — the "not yet implemented in
/// the kernel" arms below are the FFI-boundary backstop for the two 0.1.1
/// additions (`inversegaussian`, `cloglog`) the current `Family`/`BinomialLink`
/// enums have no variant for; every other `Err` arm is unreachable via
/// `glmm.fit` and exists only for direct callers of this crate.
pub fn family_from_str(family: &str, link: &str) -> Result<Family, String> {
    match family {
        "gaussian" => Ok(Family::Gaussian),
        "binomial" => match link {
            "logit" => Ok(Family::Binomial {
                link: BinomialLink::Logit,
            }),
            "probit" => Ok(Family::Binomial {
                link: BinomialLink::Probit,
            }),
            "cloglog" => Err(
                "link 'cloglog' requires GLMM 0.1.1; not yet implemented in the kernel".to_string(),
            ),
            other => Err(format!("unsupported binomial link {other:?}")),
        },
        "poisson" => match link {
            "log" => Ok(Family::Poisson {
                link: PoissonLink::Log,
            }),
            other => Err(format!("unsupported poisson link {other:?}")),
        },
        "gamma" => match link {
            "log" => Ok(Family::Gamma {
                link: GammaLink::Log,
            }),
            "inverse" => Ok(Family::Gamma {
                link: GammaLink::Inverse,
            }),
            other => Err(format!("unsupported gamma link {other:?}")),
        },
        "negativebinomial" => match link {
            "log" => Ok(Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            }),
            other => Err(format!("unsupported negativebinomial link {other:?}")),
        },
        "inversegaussian" => Err(
            "family 'inversegaussian' requires GLMM 0.1.1; not yet implemented \
             in the kernel"
                .to_string(),
        ),
        other => Err(format!("unknown family {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glmm::{BinomialLink, Family, GammaLink, NegBinomialLink, PoissonLink};

    #[test]
    fn gaussian_maps() {
        assert_eq!(
            family_from_str("gaussian", "identity"),
            Ok(Family::Gaussian)
        );
    }

    #[test]
    fn binomial_logit_and_probit_map() {
        assert_eq!(
            family_from_str("binomial", "logit"),
            Ok(Family::Binomial {
                link: BinomialLink::Logit
            })
        );
        assert_eq!(
            family_from_str("binomial", "probit"),
            Ok(Family::Binomial {
                link: BinomialLink::Probit
            })
        );
    }

    #[test]
    fn binomial_cloglog_is_a_kernel_gap() {
        let err = family_from_str("binomial", "cloglog").unwrap_err();
        assert!(err.contains("not yet implemented in the kernel"), "{err}");
    }

    #[test]
    fn poisson_maps() {
        assert_eq!(
            family_from_str("poisson", "log"),
            Ok(Family::Poisson {
                link: PoissonLink::Log
            })
        );
    }

    #[test]
    fn gamma_log_and_inverse_map() {
        assert_eq!(
            family_from_str("gamma", "log"),
            Ok(Family::Gamma {
                link: GammaLink::Log
            })
        );
        assert_eq!(
            family_from_str("gamma", "inverse"),
            Ok(Family::Gamma {
                link: GammaLink::Inverse
            })
        );
    }

    #[test]
    fn negativebinomial_maps() {
        assert_eq!(
            family_from_str("negativebinomial", "log"),
            Ok(Family::NegativeBinomial {
                link: NegBinomialLink::Log
            })
        );
    }

    #[test]
    fn inversegaussian_is_a_kernel_gap() {
        let err = family_from_str("inversegaussian", "log").unwrap_err();
        assert!(err.contains("not yet implemented in the kernel"), "{err}");
    }
}
