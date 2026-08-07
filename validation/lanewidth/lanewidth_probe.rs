//! Lane-width probe, copied by `run_lanewidth.sh` into `examples/` of both
//! scratch trees (normal + scalar-forced) and run there via
//! `cargo run --example lanewidth_probe`. Never runs from this location —
//! `validation/lanewidth/` is not on any Cargo build path, this file is a
//! template the script stages into place.
//!
//! Two things printed, one per acceptance-evidence form:
//!   1. `pulp::Arch::new()`'s Debug repr — the printed dispatch marker. On
//!      aarch64 this reads `Neon(..)` normally and `Scalar` once the vendored
//!      patch is wired in via `[patch.crates-io]`.
//!   2. A refit of the same sparse NB design as
//!      `sparse::tests::fit_sparse_nb_glmm_is_pinned`
//!      (`src/sparse/tests.rs`) — beta, se and the NB dispersion (theta).
//!      Diffing this probe's output between the normal and scalar-forced
//!      scratch trees shows the actual numeric movement the harness exists
//!      to measure — see README.md for a worked example.
use glmm::{
    Family, FitOptions, GroupIds, Grouping, GroupingRelation, ModelSpec, NegBinomialLink,
    ReStructure, Sizing, WaldSe,
};

/// Map string factor labels to dense 0-based ids (first-seen order) — same
/// pattern as `sparse::tests::dense_ids`.
fn dense_ids(raw: &[String]) -> Vec<u32> {
    use std::collections::HashMap;
    let mut map: HashMap<String, u32> = HashMap::new();
    let mut next = 0u32;
    raw.iter()
        .map(|r| {
            *map.entry(r.clone()).or_insert_with(|| {
                let v = next;
                next += 1;
                v
            })
        })
        .collect()
}

fn main() {
    println!("pulp::Arch::new() = {:?}", pulp::Arch::new());

    // Same design + data as sparse::tests::fit_sparse_nb_glmm_is_pinned.
    let csv = include_str!("../validation/data/simulated/sim_sparse_nb.csv");
    let mut y = Vec::<f64>::new();
    let mut xcol = Vec::<f64>::new();
    let mut fac: Vec<Vec<String>> = vec![Vec::new(); 8]; // g1, c1..c7
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        xcol.push(f[1].parse().unwrap());
        for k in 0..8 {
            fac[k].push(f[2 + k].to_string());
        }
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = xcol[i];
    }
    let model = ModelSpec {
        family: Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: (0..7)
                .map(|_| Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                })
                .collect(),
        }),
    };
    let ids = GroupIds {
        primary: dense_ids(&fac[0]),
        extra: fac[1..].iter().map(|f| dense_ids(f)).collect(),
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        wald_se: WaldSe::Rx,
        ..FitOptions::default()
    };
    let f = glmm::fit_cold(&x, &y, n, p, &model, &ids, &opts);
    println!("converged      = {:?}", f.converged());
    println!("beta           = {:?}", f.beta);
    println!("se             = {:?}", f.se);
    println!(
        "varcorr[..][0] = {:?}",
        f.varcorr.iter().map(|b| b[0]).collect::<Vec<_>>()
    );
    println!("dispersion     = {:?}", f.dispersion);
}
