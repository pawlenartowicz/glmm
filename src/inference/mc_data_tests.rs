//! The Monte Carlo tier's design: the grid of cells and their seeds, the
//! generator's determinism and layout (and, at `σ = 0`, its linear predictor
//! and offsets exactly), the first dataset of six cells pinned, and the
//! bands and gated set against `validation/grid/tol.R`.

use super::mc_data::{bands, cell_seed, generate, generate_with, Cell, Truth, CELLS};
use crate::formula::{Column, Table};
use std::collections::BTreeSet;

fn cell(name: &str) -> Cell {
    *CELLS
        .iter()
        .find(|c| c.name() == name)
        .unwrap_or_else(|| panic!("no cell {name}"))
}

fn numeric<'a>(t: &'a Table, name: &str) -> &'a [f64] {
    match t.columns.iter().find(|(n, _)| n == name) {
        Some((_, Column::Numeric(v))) => v,
        _ => panic!("no numeric column {name}"),
    }
}

fn factor<'a>(t: &'a Table, name: &str) -> (&'a [String], &'a [u32]) {
    match t.columns.iter().find(|(n, _)| n == name) {
        Some((_, Column::Factor { levels, codes })) => (levels, codes),
        _ => panic!("no factor column {name}"),
    }
}

/// A table as comparable data: per column its name, its values' bits (or
/// its codes) and its levels.
fn contents(t: &Table) -> Vec<(String, Vec<u64>, Vec<String>)> {
    t.columns
        .iter()
        .map(|(name, col)| match col {
            Column::Numeric(v) => (
                name.clone(),
                v.iter().map(|x| x.to_bits()).collect(),
                vec![],
            ),
            Column::Factor { levels, codes } => (
                name.clone(),
                codes.iter().map(|&c| u64::from(c)).collect(),
                levels.clone(),
            ),
        })
        .collect()
}

const RESPONSES: [&str; 5] = ["y_full", "y_0", "y_00", "y_b0", "y_0f"];

#[test]
fn cells_are_the_grid() {
    assert_eq!(CELLS.len(), 216);
    let names: BTreeSet<String> = CELLS.iter().map(Cell::name).collect();
    assert_eq!(names.len(), 216);
    let seeds: BTreeSet<u64> = CELLS.iter().map(|c| cell_seed(&c.name())).collect();
    assert_eq!(seeds.len(), 216);
    let ks: BTreeSet<usize> = CELLS.iter().map(|c| c.k).collect();
    assert_eq!(ks, BTreeSet::from([10, 20, 30, 50, 100, 200]));
    let ms: BTreeSet<usize> = CELLS.iter().map(|c| c.m).collect();
    assert_eq!(ms, BTreeSet::from([3, 9, 17]));
    let responses: BTreeSet<&str> = CELLS.iter().map(|c| c.response.tag()).collect();
    assert_eq!(
        responses,
        BTreeSet::from(["bern54", "bern22", "bern10", "pois165", "pois034", "gauss"])
    );
    let slopes: BTreeSet<bool> = CELLS.iter().map(|c| c.slope).collect();
    assert_eq!(slopes, BTreeSet::from([false, true]));
    // The canonical order: rdi before rdis, then K, M, the response as listed.
    assert_eq!(CELLS[0].name(), "k10_m3_bern54_rdi");
    assert_eq!(CELLS[5].name(), "k10_m3_gauss_rdi");
    assert_eq!(CELLS[6].name(), "k10_m9_bern54_rdi");
    assert_eq!(CELLS[18].name(), "k20_m3_bern54_rdi");
    assert_eq!(CELLS[107].name(), "k200_m17_gauss_rdi");
    assert_eq!(CELLS[108].name(), "k10_m3_bern54_rdis");
    assert_eq!(CELLS[215].name(), "k200_m17_gauss_rdis");
    // The seed is 20261100 plus FNV-1a 64 of the name, wrapping; FNV-1a's
    // published values for "" and "a".
    assert_eq!(
        cell_seed(""),
        20_261_100u64.wrapping_add(0xcbf2_9ce4_8422_2325)
    );
    assert_eq!(
        cell_seed("a"),
        20_261_100u64.wrapping_add(0xaf63_dc4c_8601_ec8c)
    );
}

#[test]
fn generator_is_deterministic() {
    for name in [
        "k10_m3_bern54_rdi",
        "k20_m17_pois034_rdis",
        "k10_m9_gauss_rdis",
    ] {
        let c = cell(name);
        assert!(
            contents(&generate(&c, 3)) == contents(&generate(&c, 3)),
            "{name}"
        );
        assert!(
            contents(&generate(&c, 3)) != contents(&generate(&c, 4)),
            "{name}"
        );
    }
}

#[test]
fn generator_follows_the_design() {
    for name in [
        "k50_m3_bern22_rdi",
        "k50_m9_pois165_rdis",
        "k50_m17_gauss_rdi",
        "k50_m17_bern10_rdis",
    ] {
        let c = cell(name);
        let t = generate(&c, 0);
        for (col, values) in &t.columns {
            let len = match values {
                Column::Numeric(v) => v.len(),
                Column::Factor { codes, .. } => codes.len(),
            };
            assert_eq!(len, t.n, "{name}: {col}");
        }
        let (ids, id) = factor(&t, "id");
        let (f3_levels, f3) = factor(&t, "f3");
        let (time, group) = (numeric(&t, "time"), numeric(&t, "group"));
        assert_eq!(ids.len(), c.k, "{name}");
        assert_eq!(f3_levels.len(), 3, "{name}");
        let mut f3_seen = BTreeSet::new();
        for s in 0..c.k as u32 {
            let rows: Vec<usize> = (0..t.n).filter(|&i| id[i] == s).collect();
            assert!((1..=c.m).contains(&rows.len()), "{name}: subject {s}");
            // Rows of a subject are contiguous, in subject order.
            assert!(
                rows.windows(2).all(|w| w[1] == w[0] + 1),
                "{name}: subject {s}"
            );
            for w in rows.windows(2) {
                assert!(time[w[0]] < time[w[1]], "{name}: subject {s}");
            }
            for &i in &rows {
                let slot = (time[i] * (c.m - 1) as f64).round();
                assert!((0.0..c.m as f64).contains(&slot), "{name}: row {i}");
                assert_eq!(time[i], slot / (c.m - 1) as f64, "{name}: row {i}");
                assert!(group[i] == 0.0 || group[i] == 1.0, "{name}: row {i}");
                assert_eq!(group[i], group[rows[0]], "{name}: row {i}");
                assert_eq!(f3[i], f3[rows[0]], "{name}: row {i}");
                f3_seen.insert(f3[i]);
            }
        }
        assert_eq!(f3_seen, BTreeSet::from([0, 1, 2]), "{name}");
    }

    // At σ = 0 a Gaussian response is its linear predictor, exactly.
    for name in ["k20_m9_gauss_rdis", "k20_m9_gauss_rdi"] {
        let c = cell(name);
        let truth = Truth {
            sigma: 0.0,
            ..Truth::of(&c)
        };
        let (t, b) = generate_with(&c, 0, &truth);
        assert_eq!(b.len(), c.k, "{name}");
        // Slopes are drawn on rdis only.
        assert_eq!(b.iter().any(|u| u[1] != 0.0), c.slope, "{name}");
        let (_, id) = factor(&t, "id");
        let (time, group) = (numeric(&t, "time"), numeric(&t, "group"));
        let y: Vec<&[f64]> = RESPONSES.iter().map(|r| numeric(&t, r)).collect();
        let off: Vec<&[f64]> = (0..4).map(|j| numeric(&t, &format!("off{j}"))).collect();
        let [b0, b1, b2, b3] = truth.beta;
        for i in 0..t.n {
            let (tm, g) = (time[i], group[i]);
            let [u0, u1] = b[id[i] as usize];
            let xb = |c2: f64, c3: f64| b0 + b1 * tm + c2 * g + c3 * tm * g;
            assert_eq!(y[0][i], xb(b2, b3) + u0 + u1 * tm, "{name}: y_full row {i}");
            assert_eq!(y[1][i], xb(b2, 0.0) + u0 + u1 * tm, "{name}: y_0 row {i}");
            assert_eq!(y[2][i], xb(0.0, 0.0) + u0 + u1 * tm, "{name}: y_00 row {i}");
            assert_eq!(y[3][i], xb(b2, b3), "{name}: y_b0 row {i}");
            assert_eq!(y[4][i], xb(b2, 0.0), "{name}: y_0f row {i}");
            // Each offset is its own term of the truth, so a swapped column
            // fails exactly.
            let terms = [b0, b1 * tm, b2 * g, b3 * tm * g];
            for (j, term) in terms.into_iter().enumerate() {
                assert_eq!(off[j][i], term, "{name}: off{j} row {i}");
            }
        }
    }
}

/// Dataset 0 of six cells, one per response level, both random parts: the
/// rows, `Σgroup` over rows, and the sums of `y_full`, `y_0`, `y_00`, `y_b0`
/// and `y_0f`. Measured on 2026-10-13 with `commonstats` 0.2.2 and
/// `rand_philox` 0.2.1. A release of either that moves them is re-pinned here
/// with a provenance line naming both versions, never absorbed by a band.
const PINS: [(&str, usize, f64, [f64; 5]); 6] = [
    ("k10_m3_bern54_rdi", 18, 7.0, [10.0, 10.0, 10.0, 10.0, 13.0]),
    (
        "k20_m9_bern22_rdis",
        91,
        41.0,
        [16.0, 20.0, 30.0, 17.0, 18.0],
    ),
    (
        "k30_m17_bern10_rdi",
        273,
        105.0,
        [27.0, 22.0, 28.0, 19.0, 19.0],
    ),
    (
        "k50_m3_pois165_rdis",
        97,
        26.0,
        [158.0, 130.0, 154.0, 126.0, 131.0],
    ),
    (
        "k100_m9_pois034_rdi",
        561,
        230.0,
        [198.0, 192.0, 188.0, 129.0, 138.0],
    ),
    (
        "k10_m17_gauss_rdis",
        83,
        50.0,
        [
            53.749969221468334,
            52.78317497440115,
            36.72645877318504,
            4.812470037947638,
            16.575905255127687,
        ],
    ),
];

#[test]
fn first_datasets_are_pinned() {
    let mut moved = Vec::new();
    for (name, n, sum_group, sums) in PINS {
        let t = generate(&cell(name), 0);
        let got_group: f64 = numeric(&t, "group").iter().sum();
        let got: Vec<f64> = RESPONSES
            .iter()
            .map(|r| numeric(&t, r).iter().sum())
            .collect();
        // Discrete sums are integers, exact; Gaussian sums within 1e-12 relative.
        let same = |a: f64, b: f64| {
            if name.contains("gauss") {
                (a - b).abs() <= 1e-12 * b.abs()
            } else {
                a == b
            }
        };
        let ok =
            t.n == n && got_group == sum_group && got.iter().zip(&sums).all(|(&a, &b)| same(a, b));
        if !ok {
            moved.push(format!("(\"{name}\", {}, {got_group:?}, {got:?}),", t.n));
        }
    }
    assert!(
        moved.is_empty(),
        "dataset 0 moved; measured:\n{}",
        moved.join("\n")
    );
}

/// One value of `tol.R` as the mirror test reads it: a number, a string,
/// `NA`, a `c(...)` vector, a `list(...)`, or any other expression as text.
#[derive(Debug)]
enum RValue {
    Num(f64),
    Str(String),
    Na,
    Vec(Vec<RValue>),
    List(Vec<(String, RValue)>),
    Expr(String),
}

/// R source as tokens: comments dropped, a string kept with its opening
/// quote, `(`, `)`, `,` and `=` alone, anything else a run of other
/// characters.
fn r_tokens(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
        } else if ch == '#' {
            chars.by_ref().find(|&c| c == '\n');
        } else if ch == '"' {
            chars.next();
            let mut s = String::from("\"");
            s.extend(chars.by_ref().take_while(|&c| c != '"'));
            out.push(s);
        } else if "(),=".contains(ch) {
            out.push(ch.to_string());
            chars.next();
        } else {
            let mut s = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || "(),=#\"".contains(c) {
                    break;
                }
                s.push(c);
                chars.next();
            }
            out.push(s);
        }
    }
    out
}

/// The value starting at token `*i`, leaving `*i` after it.
fn r_value(t: &[String], i: &mut usize) -> RValue {
    if (t[*i] == "list" || t[*i] == "c") && t[*i + 1] == "(" {
        let is_list = t[*i] == "list";
        *i += 2;
        let mut items = Vec::new();
        while t[*i] != ")" {
            let mut name = String::new();
            if t[*i + 1] == "=" {
                name = t[*i].clone();
                *i += 2;
            }
            items.push((name, r_value(t, i)));
            if t[*i] == "," {
                *i += 1;
            }
        }
        *i += 1;
        return if is_list {
            RValue::List(items)
        } else {
            RValue::Vec(items.into_iter().map(|(_, v)| v).collect())
        };
    }
    let (start, mut depth) = (*i, 0);
    while depth > 0 || (t[*i] != "," && t[*i] != ")") {
        match t[*i].as_str() {
            "(" => depth += 1,
            ")" => depth -= 1,
            _ => {}
        }
        *i += 1;
    }
    match &t[start..*i] {
        [one] if one.starts_with('"') => RValue::Str(one[1..].to_string()),
        [one] if one == "NA" => RValue::Na,
        [one] => one
            .parse()
            .map_or_else(|_| RValue::Expr(one.clone()), RValue::Num),
        many => RValue::Expr(many.join(" ")),
    }
}

fn field<'a>(items: &'a [(String, RValue)], name: &str, at: &str) -> &'a RValue {
    items
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v)
        .unwrap_or_else(|| panic!("{at}: no `{name}`"))
}

fn r_num(v: &RValue, at: &str) -> f64 {
    match v {
        RValue::Num(x) => *x,
        RValue::Expr(e) => panic!("{at}: not a number: the expression `{e}`"),
        other => panic!("{at}: not a number: {other:?}"),
    }
}

fn r_strings(v: &RValue, at: &str) -> Vec<String> {
    match v {
        RValue::Vec(items) => items
            .iter()
            .map(|s| match s {
                RValue::Str(s) => s.clone(),
                other => panic!("{at}: not a string: {other:?}"),
            })
            .collect(),
        other => panic!("{at}: not a vector: {other:?}"),
    }
}

/// The `mc_*` rows of `TOL_INFERENCE` in `validation/grid/tol.R`, read as
/// text, against their Rust copies in `mc_data::bands`: every value, every
/// family of the gated set (side, δ, α, quantities, cells, in order), no
/// `mc_*` row without a copy, and every gated cell a cell of the grid.
#[test]
fn bands_match_tol_r() {
    let path = format!("{}/validation/grid/tol.R", env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let head = "TOL_INFERENCE <- ";
    let at = src.find(head).expect("TOL_INFERENCE in tol.R") + head.len();
    let RValue::List(rows) = r_value(&r_tokens(&src[at..]), &mut 0) else {
        panic!("TOL_INFERENCE is not a list");
    };
    let mc: BTreeSet<&str> = rows
        .iter()
        .map(|(n, _)| n.as_str())
        .filter(|n| n.starts_with("mc_"))
        .collect();
    assert_eq!(
        mc,
        BTreeSet::from([
            "mc_r",
            "mc_level_k",
            "mc_cover_k",
            "mc_asym_rel",
            "mc_fail_max",
            "mc_aliased_max",
            "mc_gated"
        ])
    );
    let num = |name: &str| r_num(field(&rows, name, "TOL_INFERENCE"), name);
    assert_eq!(num("mc_r"), bands::MC_R as f64);
    assert_eq!(num("mc_level_k"), bands::MC_LEVEL_K);
    assert_eq!(num("mc_cover_k"), bands::MC_COVER_K);
    assert_eq!(num("mc_asym_rel"), bands::MC_ASYM_REL);
    assert_eq!(num("mc_fail_max"), bands::MC_FAIL_MAX);
    assert_eq!(num("mc_aliased_max"), bands::MC_ALIASED_MAX);

    let RValue::List(families) = field(&rows, "mc_gated", "TOL_INFERENCE") else {
        panic!("mc_gated is not a list");
    };
    let names: Vec<&str> = families.iter().map(|(n, _)| n.as_str()).collect();
    let rust: Vec<&str> = bands::GATED.iter().map(|g| g.family).collect();
    assert_eq!(names, rust, "mc_gated families");
    let grid: BTreeSet<String> = CELLS.iter().map(Cell::name).collect();
    for ((name, family), gate) in families.iter().zip(&bands::GATED) {
        let RValue::List(items) = family else {
            panic!("{name} is not a list");
        };
        let keys: BTreeSet<&str> = items.iter().map(|(k, _)| k.as_str()).collect();
        let mut want = BTreeSet::from(["side", "delta", "quantities", "cells"]);
        if gate.alpha.is_some() {
            want.insert("alpha");
        }
        assert_eq!(keys, want, "{name}: fields");
        let side = match field(items, "side", name) {
            RValue::Str(s) => s.as_str(),
            other => panic!("{name}: side {other:?}"),
        };
        let rust_side = match gate.side {
            bands::Side::Two => "two",
            bands::Side::Upper => "upper",
            bands::Side::Lower => "lower",
        };
        assert_eq!(side, rust_side, "{name}: side");
        assert_eq!(
            r_num(field(items, "delta", name), name),
            gate.delta,
            "{name}"
        );
        if let Some(alpha) = gate.alpha {
            assert_eq!(r_num(field(items, "alpha", name), name), alpha, "{name}");
        }
        assert_eq!(
            r_strings(field(items, "quantities", name), name),
            gate.quantities,
            "{name}: quantities"
        );
        assert_eq!(
            r_strings(field(items, "cells", name), name),
            gate.cells,
            "{name}: cells"
        );
        for c in gate.cells {
            assert!(grid.contains(*c), "{name}: {c} is not a cell of the grid");
        }
    }
}
