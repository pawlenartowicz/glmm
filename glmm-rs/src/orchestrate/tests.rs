//! Tests of the port strings and tags shared by every entry point.

use super::*;
use crate::inference::Objective;
use crate::{BinomialLink, Family, GammaLink, InverseGaussianLink, NegBinomialLink, PoissonLink};

#[test]
fn family_and_link_strings_map_to_kernel_families() {
    let ok = [
        ("gaussian", "identity", Family::Gaussian),
        (
            "binomial",
            "logit",
            Family::Binomial {
                link: BinomialLink::Logit,
            },
        ),
        (
            "binomial",
            "probit",
            Family::Binomial {
                link: BinomialLink::Probit,
            },
        ),
        (
            "binomial",
            "cloglog",
            Family::Binomial {
                link: BinomialLink::Cloglog,
            },
        ),
        (
            "poisson",
            "log",
            Family::Poisson {
                link: PoissonLink::Log,
            },
        ),
        (
            "gamma",
            "log",
            Family::Gamma {
                link: GammaLink::Log,
            },
        ),
        (
            "gamma",
            "inverse",
            Family::Gamma {
                link: GammaLink::Inverse,
            },
        ),
        (
            "negativebinomial",
            "log",
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
        ),
        (
            "inversegaussian",
            "log",
            Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            },
        ),
        (
            "inversegaussian",
            "inverse_squared",
            Family::InverseGaussian {
                link: InverseGaussianLink::InverseSquared,
            },
        ),
    ];
    for (family, link, expected) in ok {
        assert_eq!(
            family_from_str(family, link),
            Ok(expected),
            "{family}/{link}"
        );
    }
    let err = family_from_str("inversegaussian", "identity").unwrap_err();
    assert!(err.contains("unsupported inversegaussian link"), "{err}");
    let err = family_from_str("gaussian", "log").unwrap_err();
    assert!(err.contains("unsupported gaussian link"), "{err}");
}

/// `objective_parts` on every objective, and `objective_from_parts`
/// back; a pair `objective_parts` never gives is refused.
#[test]
fn objective_parts_round_trip_on_every_objective() {
    let cases = [
        (Objective::LeastSquares, ("least_squares", 0)),
        (Objective::MaximumLikelihood, ("maximum_likelihood", 0)),
        (Objective::Reml, ("reml", 0)),
        (Objective::Laplace, ("laplace", 1)),
        (Objective::Agq { nodes: 5 }, ("agq", 5)),
    ];
    for (objective, (label, nodes)) in cases {
        assert_eq!(objective_parts(objective), (label, nodes));
        assert_eq!(objective_from_parts(label, nodes), Ok(objective));
    }
    assert_eq!(
        objective_from_parts("reml", 1),
        Err("no fit stores the objective \"reml\" with 1 quadrature node".to_string())
    );
    for (label, nodes) in [
        ("least_squares", 1),
        ("reml", 1),
        ("laplace", 0),
        ("laplace", 2),
        ("agq", 0),
        ("agq", 1),
        ("ml", 0),
    ] {
        assert!(
            objective_from_parts(label, nodes).is_err(),
            "{label}, {nodes}"
        );
    }
}
