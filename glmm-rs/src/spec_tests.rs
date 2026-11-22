use super::*;
use crate::{Data, FitError, FitOptions, GroupIds};

/// A spec whose terms do not describe its blocks returns
/// `FitError::Shape` naming the grouping and the reason, on the primary
/// and on an extra. Each row breaks one rule, so the reason names the
/// guard that refused it.
#[test]
fn bad_terms_are_a_shape_error() {
    let n = 12;
    let x: Vec<f64> = (0..n)
        .flat_map(|i| [1.0, (i % 5) as f64, (i % 3) as f64])
        .collect();
    let y: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64).collect();
    let ids = GroupIds::new((0..n as u32).map(|i| i % 4).collect())
        .extra((0..n as u32).map(|i| i % 3).collect());
    let data = Data::new(&x, &y, n, 3).ids(&ids);
    let shape_error = |re: Re, grouping: &str, reason: &str| {
        let spec = ModelSpec::mixed(Family::Gaussian, re);
        match crate::fit(&data, &spec, &FitOptions::default()) {
            Err(FitError::Shape(m)) => {
                assert!(m.contains(grouping) && m.contains(reason), "{m}")
            }
            Err(e) => panic!("{spec:?}: {e:?}"),
            Ok(_) => panic!("{spec:?}: fitted"),
        }
    };
    let crossed = || Re::intercept().extra(Grouping::crossed());
    let not_in_order = "not the terms' columns in term order";
    let empty = "neither the intercept nor a slope";

    // `slopes` changed after the terms were set.
    shape_error(
        Re::intercept()
            .terms([ReTerm::intercept().slopes([1])])
            .slopes([2]),
        "primary grouping",
        not_in_order,
    );
    // A slope column in two terms.
    shape_error(
        Re::intercept().terms([
            ReTerm::intercept().slopes([1]),
            ReTerm::no_intercept().slopes([1]),
        ]),
        "primary grouping",
        "slope column 1 is in two terms",
    );
    // The intercept in two terms.
    shape_error(
        Re::intercept().terms([ReTerm::intercept(), ReTerm::intercept().slopes([1])]),
        "primary grouping",
        "must be in term 0",
    );
    // An empty term.
    shape_error(
        Re::intercept().terms([ReTerm::intercept(), ReTerm::no_intercept()]),
        "primary grouping",
        empty,
    );
    // The intercept term not first, which only a field write can do.
    // `slopes` is swapped too, so it still lists the terms' columns in
    // term order and only the intercept guard can refuse the spec.
    let mut re = Re::intercept().terms([
        ReTerm::intercept().slopes([1]),
        ReTerm::no_intercept().slopes([2]),
    ]);
    re.terms.swap(0, 1);
    re.slopes.swap(0, 1);
    shape_error(re, "primary grouping", "must be in term 0");
    // The same checks reach an extra grouping.
    shape_error(
        crossed().extra(Grouping::crossed().terms([ReTerm::no_intercept()])),
        "extra grouping 1",
        empty,
    );
    let mut re = crossed();
    re.extra_groupings[0] = Grouping::crossed()
        .terms([ReTerm::no_intercept().slopes([1])])
        .slopes([1, 2]);
    shape_error(re, "extra grouping 0", not_in_order);
}
