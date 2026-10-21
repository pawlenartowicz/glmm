//! The per-term hypothesis matrices (lmerTest's Type III and `XᵀX` Type II,
//! car's covariance-metric Type II), built from a [`Lowered`] design, with
//! the marginal-term check and the missing-cell note of the per-term tests.
//!
//! Ports of lmerTest 3.2.1's `extract_contrasts_type3` (what
//! `get_contrasts_type3` runs on an all-treatment-contrast design, the only
//! coding the formula frontend emits), `get_contrasts_type2_unfolded` and,
//! for the fallback both take on a design with at most one column or one
//! term, `get_contrasts_type1` (Kuznetsova, Brockhoff & Christensen 2017,
//! *J. Stat. Softw.* 82(13)), and of car 3.1.5's Type II (`Anova_II_default`,
//! `Anova_II_lm`, `Anova_II_mer`; Fox & Weisberg 2019, *An R Companion to
//! Applied Regression*, 3rd ed.). lmerTest and car work on the column-dropped
//! design, so the builders take the fit's retained columns, build on those, and
//! return each matrix over all `p` caller columns with zeros on the aliased
//! ones.

use std::cmp::Reverse;

use faer::{Mat, MatRef};

use super::materialize::Lowered;
use crate::inference::evd::sym_eigen;
use crate::linalg::SeqLlt;
use crate::ColumnId;

/// One term's hypothesis matrix, in lmerTest's basis.
pub struct TermHypothesis {
    /// The term label ([`super::FixedTerm::label`]).
    pub term: String,
    /// `r × p` over all `p` caller columns (zero on aliased ones), one row per
    /// hypothesis lmerTest keeps for the term.
    pub l: Mat<f64>,
}

/// lmerTest's Type III hypothesis matrices, one per term of
/// [`Lowered::terms`] in that order (`attr(anova(m), "hypotheses")`).
///
/// `retained` lists the caller columns the fit kept (ascending), on which
/// the design has full column rank. `L = t(ginv(X)·rdX)` there, which is
/// `(XᵀX)⁻¹·(Xᵀ·rdX)` transposed, rounded by `zapsmall`; then, from the
/// highest-order term down, a term's columns are replaced by their residuals
/// on the columns of the terms containing it; a term's rows are the `rdX`
/// rows named like the retained columns, and rows with `Σ|·| ≤ 1e-8` are
/// dropped. With at most one retained column or one term it is Type I, as in
/// lmerTest; when the retained design is the intercept alone, each term gets
/// a 0-row entry, where lmerTest returns one unnamed 0-row matrix. A design
/// that is singular on `retained` (outside the contract) gives NaN entries.
pub(crate) fn type3(lo: &Lowered, retained: &[usize]) -> Vec<TermHypothesis> {
    type3_with(lo, retained, zapsmall)
}

/// [`type3`] with each `zapsmall` replaced by [`zap`]: the entries
/// `zapsmall` would round to zero set to zero, the rest unrounded. A Type III
/// likelihood-ratio row takes its reduced design's null basis from these
/// ([`type3_null_basis`]): where the Type III weights do not terminate (1/3
/// on an unbalanced design), the 7-digit rounding tilts the hypothesis by
/// about 5e-8, and a refit would carry the tilt into its statistic.
pub(crate) fn type3_unrounded(lo: &Lowered, retained: &[usize]) -> Vec<TermHypothesis> {
    type3_with(lo, retained, zap)
}

/// [`type3`]'s construction, each rounding step done by `round`.
fn type3_with(lo: &Lowered, retained: &[usize], round: fn(&mut Mat<f64>)) -> Vec<TermHypothesis> {
    let d = Design::new(lo, retained);
    if d.type1_fallback() {
        return d.type1();
    }
    let (pr, k) = (retained.len(), lo.rdx.names.len());
    // ginv(X)·rdX on the retained columns, pr × k.
    let mut a = Mat::from_fn(pr, k, |i, c| lo.rdx.xt_rdx[(retained[i], c)]);
    match SeqLlt::new(d.gram().as_ref()) {
        Some(llt) => llt.solve_in_place(a.as_mut()),
        None => a.fill(f64::NAN),
    }
    let mut l = a.transpose().to_owned();
    round(&mut l);

    let mut order: Vec<usize> = (0..lo.terms.len()).collect();
    order.sort_by_key(|&t| Reverse(lo.terms[t].vars.len()));
    for t in order {
        if d.contains[t].is_empty() || d.map[t].is_empty() {
            continue;
        }
        let basis: Vec<usize> = d.contains[t]
            .iter()
            .flat_map(|&u| d.map[u].iter().copied())
            .collect();
        let mut res = residuals(&l, &basis, &d.map[t]);
        round(&mut res);
        for (j, &c) in d.map[t].iter().enumerate() {
            for r in 0..k {
                l[(r, c)] = res[(r, j)];
            }
        }
    }

    // `L[colnames(X), ]`: the rdX row of each retained column, by name.
    let rows: Vec<usize> = retained
        .iter()
        .map(|&c| {
            let name = &lo.col_names[c];
            lo.rdx
                .names
                .iter()
                .position(|n| n == name)
                .unwrap_or_else(|| panic!("design column {name} is not an rdX column"))
        })
        .collect();
    lo.terms
        .iter()
        .zip(&d.map)
        .map(|(term, cols)| {
            let kept: Vec<Vec<f64>> = cols
                .iter()
                .map(|&c| rows.iter().map(|&r| l[(r, c)]).collect::<Vec<f64>>())
                .filter(|row| row.iter().map(|v| v.abs()).sum::<f64>() > 1e-8)
                .collect();
            d.hypothesis(&term.label, &kept)
        })
        .collect()
}

/// [`type3`] for the loop tier (`crate::loop_advanced`, no semver promise):
/// the matrices depend on the design and `retained` only, so one build serves
/// every fit of the design that keeps the same columns.
#[cfg(feature = "loop_advanced")]
#[doc(hidden)]
pub fn type3_pub(lo: &Lowered, retained: &[usize]) -> Vec<TermHypothesis> {
    type3(lo, retained)
}

/// lmerTest's Type II hypothesis matrices in the `XᵀX` metric
/// (`get_contrasts_type2_unfolded`), one per term of [`Lowered::terms`] in
/// that order, on the retained columns as [`type3`]: identity rows for a term
/// no other term contains; for a contained term, its rows of
/// `t(doolittle(crossprod(Xnew))$L)`, `Xnew` the design with the term's and
/// its containers' columns moved last. Type I with at most one retained
/// column or one term.
pub(crate) fn type2_xtx(lo: &Lowered, retained: &[usize]) -> Vec<TermHypothesis> {
    let d = Design::new(lo, retained);
    if d.type1_fallback() {
        return d.type1();
    }
    let pr = retained.len();
    let gram = d.gram();
    lo.terms
        .iter()
        .enumerate()
        .map(|(t, term)| {
            if d.contains[t].is_empty() {
                let rows: Vec<Vec<f64>> = d.map[t]
                    .iter()
                    .map(|&c| (0..pr).map(|j| f64::from(u8::from(j == c))).collect())
                    .collect();
                return d.hypothesis(&term.label, &rows);
            }
            let moved: Vec<usize> = std::iter::once(t)
                .chain(d.contains[t].iter().copied())
                .flat_map(|u| d.map[u].iter().copied())
                .collect();
            let perm: Vec<usize> = (0..pr)
                .filter(|c| !moved.contains(c))
                .chain(moved.iter().copied())
                .collect();
            let mut pos = vec![0; pr];
            for (s, &c) in perm.iter().enumerate() {
                pos[c] = s;
            }
            let g = Mat::from_fn(pr, pr, |i, j| gram[(perm[i], perm[j])]);
            let ld = doolittle_l(&g);
            let rows: Vec<Vec<f64>> = d.map[t]
                .iter()
                .map(|&c| (0..pr).map(|j| ld[(pos[j], pos[c])]).collect())
                .collect();
            d.hypothesis(&term.label, &rows)
        })
        .collect()
}

/// car's Type II hypothesis matrices in the covariance metric
/// (`Anova_II_default`, `Anova_II_lm`, `Anova_II_mer`), one per term of
/// [`Lowered::terms`] in that order, on the retained columns as [`type3`].
///
/// A term's relatives are the other terms whose variables include all of its
/// own (`car:::relatives`: every variable `attr(terms, "factors")` marks,
/// numeric or not). With none, the rows are the identity on the term's
/// columns. With some, they span what `ConjComp(t(H1), t(H2), vcov.)`
/// returns: the vectors on the term's and its relatives' columns that are
/// `vcov`-orthogonal to every relative column (`lᵀ·V·e_r = 0`). They are
/// built in the basis that is the identity on the term's columns,
/// `[−V_tr·V_rr⁻¹ | I]`, which spans the same rows as car's QR basis; Wald
/// χ² and F do not depend on the basis. `vcov` is `p × p` over all caller
/// columns (as [`crate::Fit::vcov`]) and only its retained block is read; a
/// relatives block that is not positive definite gives NaN entries.
pub(crate) fn type2_vcov(
    lo: &Lowered,
    retained: &[usize],
    vcov: &[Vec<f64>],
) -> Vec<TermHypothesis> {
    let d = Design::new(lo, retained);
    let v = |a: usize, b: usize| vcov[retained[a]][retained[b]];
    lo.terms
        .iter()
        .enumerate()
        .map(|(t, term)| {
            let rel: Vec<usize> = relatives(lo, t)
                .flat_map(|u| d.map[u].iter().copied())
                .collect();
            let cols = &d.map[t];
            // V_rr⁻¹·V_rt, |rel| × |cols|.
            let mut b = Mat::from_fn(rel.len(), cols.len(), |i, j| v(rel[i], cols[j]));
            if !rel.is_empty() {
                let vrr = Mat::from_fn(rel.len(), rel.len(), |i, j| v(rel[i], rel[j]));
                match SeqLlt::new(vrr.as_ref()) {
                    Some(llt) => llt.solve_in_place(b.as_mut()),
                    None => b.fill(f64::NAN),
                }
            }
            let rows: Vec<Vec<f64>> = cols
                .iter()
                .enumerate()
                .map(|(j, &c)| {
                    let mut row = vec![0.0; retained.len()];
                    row[c] = 1.0;
                    for (i, &r) in rel.iter().enumerate() {
                        row[r] = -b[(i, j)];
                    }
                    row
                })
                .collect();
            d.hypothesis(&term.label, &rows)
        })
        .collect()
}

/// car's relatives of term `t` (`car:::relatives`), the containment of
/// [`type2_vcov`] and [`type2_lr_sets`]: the other terms whose variables
/// include all of its own, numeric or not, in term order.
fn relatives(lo: &Lowered, t: usize) -> impl Iterator<Item = usize> + '_ {
    let vars = &lo.terms[t].vars;
    (0..lo.terms.len())
        .filter(move |&u| u != t && vars.iter().all(|x| lo.terms[u].vars.contains(x)))
}

/// Why a term has no reduced design for a likelihood-ratio row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReducedDesignError {
    /// The term's hypothesis touches a column that carries a random slope
    /// (Type III), or its reduced design drops one (Type II): `Grouping::slopes`
    /// indexes design columns, so the kernel cannot fit a design that mixes or
    /// drops such a column.
    SlopeTouched,
    /// No basis meets `‖E·N‖ ≤ 1e-12·‖E‖` ([`type3_null_basis`]): the
    /// hypothesis has a non-finite entry, the eigensolver failed, the
    /// hypothesis is too ill-conditioned for the bound, or its unrounded and
    /// rounded forms differ in rank.
    NoNullBasis,
}

/// The column sets of car's Type II likelihood-ratio tests
/// (`Anova(type = 2, test.statistic = "LR")`), one `(full, reduced)` pair per
/// term of [`Lowered::terms`] in that order. `full` is the model of every term
/// that is not one of the term's relatives (the containment [`type2_vcov`]
/// reads, so on `y ~ time * group` with a numeric `time`, `time:group`
/// contains `group`), the term itself included; `reduced` is `full` without
/// the term's columns. Both list caller columns of `retained` in ascending
/// order, the intercept kept, so each design is a column subset of
/// [`Lowered::x`]. A term without relatives has the whole retained design as
/// `full`; equal sets recur across terms, and the caller fits each once.
pub(crate) fn type2_lr_sets(lo: &Lowered, retained: &[usize]) -> Vec<(Vec<usize>, Vec<usize>)> {
    (0..lo.terms.len())
        .map(|t| {
            let dropped: Vec<usize> = relatives(lo, t)
                .flat_map(|u| lo.terms[u].columns.clone())
                .collect();
            let full: Vec<usize> = retained
                .iter()
                .copied()
                .filter(|c| !dropped.contains(c))
                .collect();
            let reduced = full
                .iter()
                .copied()
                .filter(|c| !lo.terms[t].columns.contains(c))
                .collect();
            (full, reduced)
        })
        .collect()
}

/// Eigenvalues of `LᵀL` at most this fraction of the largest belong to null
/// directions of `L` ([`type3_null_basis`]).
const NULL_REL_TOL: f64 = 1e-8;

/// The bound on `‖L·N‖_F / ‖L‖_F` that [`type3_null_basis`] checks.
const NULL_CHECK_REL: f64 = 1e-12;

/// The columns of `l` that are zero in every row, ascending: the columns a
/// hypothesis does not touch. They lead [`type3_null_basis`]'s `N` as unit
/// columns, in this order.
pub(crate) fn untouched_columns(l: MatRef<'_, f64>) -> Vec<usize> {
    (0..l.ncols())
        .filter(|&c| l.col(c).iter().all(|&v| v == 0.0))
        .collect()
}

/// The basis of a Type III reduced design (afex's `mixed(type = 3, method =
/// "LRT")`): `N`, `p × (p − r)` with orthonormal columns spanning the null
/// space of the term's hypothesis (`r` its rank), so the fit on `X·N` is the
/// model under the `Lβ = 0` the Wald row tests, and `p − N.ncols()` is the
/// test's df.
///
/// `l` is the hypothesis as [`type3`] gives it, rounded as lmerTest rounds
/// it, and `exact` the same hypothesis from [`type3_unrounded`]. `l` decides
/// what needs exact zeros: the columns it does not touch come first in `N`,
/// as unit columns in the order of [`untouched_columns`], so their meaning
/// and any random slope on them are kept ([`remap_slopes`] with that list);
/// the slope check; and the rank `r`. The rest of `N` is the null space of
/// `exact` on the touched columns, `E`: the eigenvectors of `EᵀE`
/// ([`sym_eigen`]) whose eigenvalue is at most `1e-8` of the largest, as many
/// as `l` leaves on those columns. The result is checked:
/// `‖E·N‖_F ≤ 1e-12·‖E‖_F`.
///
/// Both are over the columns of the design the reduced fit is built from
/// (the fit's retained columns), and `slope_cols` lists that design's
/// random-slope columns. Errors: [`ReducedDesignError::SlopeTouched`] when
/// `l` touches a slope column, [`ReducedDesignError::NoNullBasis`] when no
/// decomposition exists, the two ranks differ or the check fails.
pub(crate) fn type3_null_basis(
    l: MatRef<'_, f64>,
    exact: MatRef<'_, f64>,
    slope_cols: &[usize],
) -> Result<Mat<f64>, ReducedDesignError> {
    let p = l.ncols();
    assert_eq!(exact.ncols(), p, "l and exact are over the same columns");
    let kept = untouched_columns(l);
    let touched: Vec<usize> = (0..p).filter(|c| !kept.contains(c)).collect();
    if slope_cols.iter().any(|c| touched.contains(c)) {
        return Err(ReducedDesignError::SlopeTouched);
    }
    let on_touched =
        |m: MatRef<'_, f64>| Mat::from_fn(m.nrows(), touched.len(), |i, a| m[(i, touched[a])]);
    let e = on_touched(exact);
    let rank = touched.len() - null_directions(on_touched(l).as_ref())?.ncols();
    let null = null_directions(e.as_ref())?;
    if touched.len() - null.ncols() != rank {
        return Err(ReducedDesignError::NoNullBasis);
    }
    let frobenius = |m: MatRef<'_, f64>| {
        m.col_iter()
            .flat_map(|c| c.iter().copied())
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
    };
    let resid = frobenius((&e * &null).as_ref());
    if !resid.is_finite() || resid > NULL_CHECK_REL * frobenius(e.as_ref()) {
        return Err(ReducedDesignError::NoNullBasis);
    }
    let mut n = Mat::<f64>::zeros(p, kept.len() + null.ncols());
    for (j, &c) in kept.iter().enumerate() {
        n[(c, j)] = 1.0;
    }
    for j in 0..null.ncols() {
        for (a, &c) in touched.iter().enumerate() {
            n[(c, kept.len() + j)] = null[(a, j)];
        }
    }
    Ok(n)
}

/// The null directions of `m` (`rows × k`): the eigenvectors of `mᵀm` whose
/// eigenvalue is at most [`NULL_REL_TOL`] of the largest, as the columns of a
/// `k × (k − rank)` matrix; none when `k` is 0.
fn null_directions(m: MatRef<'_, f64>) -> Result<Mat<f64>, ReducedDesignError> {
    let k = m.ncols();
    if k == 0 {
        return Ok(Mat::zeros(0, 0));
    }
    let gram = Mat::from_fn(k, k, |a, b| {
        m.col(a)
            .iter()
            .zip(m.col(b).iter())
            .map(|(u, v)| u * v)
            .sum::<f64>()
    });
    let eig = sym_eigen(gram.as_ref(), true).map_err(|_| ReducedDesignError::NoNullBasis)?;
    let vectors = eig.vectors.expect("eigenvectors were asked for");
    let cut = NULL_REL_TOL * eig.values[k - 1];
    let null: Vec<usize> = (0..k).filter(|&i| eig.values[i] <= cut).collect();
    Ok(Mat::from_fn(k, null.len(), |a, j| vectors[(a, null[j])]))
}

/// A grouping's random-slope columns (`Grouping::slopes`) on a reduced design
/// whose column `j` is caller column `kept[j]` (a Type II set of
/// [`type2_lr_sets`], or the [`untouched_columns`] that lead a Type III `N`):
/// each slope moves to its column's new position, so it stays on the same
/// variable. [`ReducedDesignError::SlopeTouched`] when a slope column is not
/// kept.
pub(crate) fn remap_slopes(
    slopes: &[ColumnId],
    kept: &[usize],
) -> Result<Vec<ColumnId>, ReducedDesignError> {
    slopes
        .iter()
        .map(|&s| {
            kept.iter()
                .position(|&c| c == s as usize)
                .map(|j| ColumnId::try_from(j).expect("a design column index fits a ColumnId"))
                .ok_or(ReducedDesignError::SlopeTouched)
        })
        .collect()
}

/// The per-term tests' refusal of a formula whose interaction lacks a
/// marginal term: lmerTest's containment and both Type II constructions
/// presuppose the margins, and without them the rows test hypotheses that
/// depend on the coding. Each interaction needs every term made of all but
/// one of its variables (so, recursively, every lower-order margin). `Err`
/// is the refusal, naming each missing term once with the first interaction
/// that needs it.
pub(crate) fn check_marginal(lo: &Lowered) -> Result<(), String> {
    let mut missing: Vec<(String, &str)> = Vec::new();
    for term in lo.terms.iter().filter(|t| t.vars.len() > 1) {
        for v in &term.vars {
            let rest: Vec<&str> = term
                .vars
                .iter()
                .filter(|&u| u != v)
                .map(String::as_str)
                .collect();
            let present = lo.terms.iter().any(|u| {
                u.vars.len() == rest.len() && rest.iter().all(|r| u.vars.iter().any(|x| x == r))
            });
            let label = rest.join(":");
            if !present && missing.iter().all(|(m, _)| *m != label) {
                missing.push((label, &term.label));
            }
        }
    }
    if missing.is_empty() {
        return Ok(());
    }
    let list: Vec<String> = missing
        .iter()
        .map(|(m, t)| format!("`{m}` (a margin of `{t}`)"))
        .collect();
    Err(format!(
        "per-term tests need every marginal term of each interaction in the \
         formula; it lacks {}. Add the missing terms.",
        list.join(", ")
    ))
}

/// lmerTest's missing-cell note (`get_rdX`): the `rdX` columns that are zero
/// on every row, named, or `None` when there is none. A column counts as zero
/// when its cross-product with the design is: an `rdX` column lies in the
/// span of the design columns (the full coding of a term whose margins are
/// in the formula), so `Xᵀ·c = 0` only when `c = 0`.
pub(crate) fn missing_cells_note(lo: &Lowered) -> Option<String> {
    let cells: Vec<&str> = lo
        .rdx
        .names
        .iter()
        .enumerate()
        .filter(|&(c, _)| lo.rdx.xt_rdx.col(c).iter().all(|&v| v == 0.0))
        .map(|(_, name)| name.as_str())
        .collect();
    (!cells.is_empty()).then(|| {
        format!(
            "Missing cells for: {}. Interpret type III hypotheses with care.",
            cells.join(", ")
        )
    })
}

/// The pieces every builder shares, on the retained columns.
struct Design<'a> {
    lo: &'a Lowered,
    retained: &'a [usize],
    /// Per term, its retained columns as positions in `retained`
    /// (`term2colX`).
    map: Vec<Vec<usize>>,
    /// Per term, the terms containing it, in term order (`containment`): a
    /// strict superset of its variables with the same numeric ones, both read
    /// as lmerTest does, over the variables R codes 1 in each term.
    contains: Vec<Vec<usize>>,
}

impl<'a> Design<'a> {
    fn new(lo: &'a Lowered, retained: &'a [usize]) -> Self {
        debug_assert!(
            retained.windows(2).all(|w| w[0] < w[1]) && retained.iter().all(|&c| c < lo.p),
            "retained columns must be ascending and below p"
        );
        let pr = retained.len();
        let map = lo
            .terms
            .iter()
            .map(|t| {
                (0..pr)
                    .filter(|&j| t.columns.contains(&retained[j]))
                    .collect()
            })
            .collect();
        // `term_contain` reads a term's variables off `attr(terms, "factors")`
        // where the code is 1, and R's `TermCode` codes a variable 1 when the
        // term without it is empty or lies inside an earlier term (2 otherwise,
        // its margin absent). Per term: those variables, and the numeric ones
        // among them, sorted.
        let coded: Vec<(Vec<&str>, Vec<&str>)> = lo
            .terms
            .iter()
            .enumerate()
            .map(|(t, term)| {
                let mut vars = Vec::new();
                let mut numerics = Vec::new();
                for (v, &num) in term.vars.iter().zip(&term.numeric) {
                    let rest: Vec<&String> = term.vars.iter().filter(|&u| u != v).collect();
                    let margin = rest.is_empty()
                        || lo.terms[..t]
                            .iter()
                            .any(|e| rest.iter().all(|&u| e.vars.contains(u)));
                    if margin {
                        vars.push(v.as_str());
                        if num {
                            numerics.push(v.as_str());
                        }
                    }
                }
                numerics.sort_unstable();
                (vars, numerics)
            })
            .collect();
        let contains = coded
            .iter()
            .map(|(v1, n1)| {
                (0..coded.len())
                    .filter(|&u| {
                        let (v2, n2) = &coded[u];
                        v1.iter().all(|v| v2.contains(v))
                            && v2.iter().any(|v| !v1.contains(v))
                            && n1 == n2
                    })
                    .collect()
            })
            .collect();
        Design {
            lo,
            retained,
            map,
            contains,
        }
    }

    /// `XᵀX` on the retained columns.
    fn gram(&self) -> Mat<f64> {
        let xtx = xtx(self.lo);
        let pr = self.retained.len();
        Mat::from_fn(pr, pr, |a, b| xtx[(self.retained[a], self.retained[b])])
    }

    /// lmerTest's `ncol(X) <= 1L || length(term_names) <= 1L`.
    fn type1_fallback(&self) -> bool {
        self.retained.len() <= 1 || self.lo.terms.len() <= 1
    }

    /// `get_contrasts_type1`: each term's rows of `t(doolittle(XᵀX)$L)`.
    fn type1(&self) -> Vec<TermHypothesis> {
        let ld = doolittle_l(&self.gram());
        let pr = self.retained.len();
        self.lo
            .terms
            .iter()
            .zip(&self.map)
            .map(|(term, cols)| {
                let rows: Vec<Vec<f64>> = cols
                    .iter()
                    .map(|&c| (0..pr).map(|j| ld[(j, c)]).collect())
                    .collect();
                self.hypothesis(&term.label, &rows)
            })
            .collect()
    }

    /// Rows over the retained columns, scattered to all `p` caller columns.
    fn hypothesis(&self, term: &str, rows: &[Vec<f64>]) -> TermHypothesis {
        let mut l = Mat::<f64>::zeros(rows.len(), self.lo.p);
        for (i, row) in rows.iter().enumerate() {
            for (&v, &c) in row.iter().zip(self.retained) {
                l[(i, c)] = v;
            }
        }
        TermHypothesis {
            term: term.to_string(),
            l,
        }
    }
}

/// `XᵀX` over all `p` columns of the design, built on first use and kept in
/// [`Lowered`]. Each entry sums `x[i][a]·x[i][b]` over the rows in order; only
/// `a ≤ b` is accumulated and the lower triangle mirrored, which is exact
/// because the two products are the same double.
fn xtx(lo: &Lowered) -> &Mat<f64> {
    lo.xtx.get_or_init(|| {
        let p = lo.p;
        let x = &lo.x[..lo.n * p];
        let acc = match p {
            1 => xtx_upper::<1>(x),
            2 => xtx_upper::<2>(x),
            3 => xtx_upper::<3>(x),
            4 => xtx_upper::<4>(x),
            5 => xtx_upper::<5>(x),
            6 => xtx_upper::<6>(x),
            7 => xtx_upper::<7>(x),
            8 => xtx_upper::<8>(x),
            _ => {
                // Same sums, in the same order, as `xtx_upper` — change together.
                let mut acc = vec![0.0f64; p * p];
                for i in 0..lo.n {
                    let row = &lo.x[i * p..(i + 1) * p];
                    for (b, &xb) in row.iter().enumerate() {
                        for (slot, &xa) in acc[b * p..=b * p + b].iter_mut().zip(row) {
                            *slot += xa * xb;
                        }
                    }
                }
                acc
            }
        };
        Mat::from_fn(p, p, |a, b| {
            if a <= b {
                acc[b * p + a]
            } else {
                acc[a * p + b]
            }
        })
    })
}

/// [`xtx`]'s accumulation for a design `P` columns wide, row-major `x`, as
/// the flat `P × P` array with entry `(a, b)`, `a ≤ b`, at `b·P + a` (the
/// lower triangle stays 0). With the width fixed at compile time the
/// accumulators live in registers instead of a load-add-store through memory
/// on every row, about 5× faster at `P = 2`; each entry is still one serial
/// sum over the rows in order, so the bits match the general loop in [`xtx`].
fn xtx_upper<const P: usize>(x: &[f64]) -> Vec<f64> {
    let mut acc = [[0.0f64; P]; P];
    for row in x.chunks_exact(P) {
        let row: &[f64; P] = row.try_into().expect("chunks_exact yields P values");
        for b in 0..P {
            for a in 0..=b {
                acc[b][a] += row[a] * row[b];
            }
        }
    }
    acc.concat()
}

/// The unit lower factor `L` of lmerTest's `doolittle(x, eps = 1e-6)`: a
/// pivot `|U[i, i]| < eps` zeroes its column below the diagonal, and every
/// `|L| < eps` is set to zero. Same loop order as the R code.
fn doolittle_l(x: &Mat<f64>) -> Mat<f64> {
    const EPS: f64 = 1e-6;
    let n = x.nrows();
    let mut l = Mat::<f64>::identity(n, n);
    let mut u = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut v = x[(i, j)];
            for k in 0..i {
                v -= l[(i, k)] * u[(k, j)];
            }
            u[(i, j)] = v;
        }
        for j in i + 1..n {
            let mut v = x[(j, i)];
            for k in 0..i {
                v -= l[(j, k)] * u[(k, i)];
            }
            l[(j, i)] = if u[(i, i)].abs() < EPS {
                0.0
            } else {
                v / u[(i, i)]
            };
        }
    }
    for j in 0..n {
        for i in 0..n {
            if l[(i, j)].abs() < EPS {
                l[(i, j)] = 0.0;
            }
        }
    }
    l
}

/// R's `zapsmall(x)` at the default 7 digits: every entry rounded to
/// `max(0, 7 − ⌈log10(max|x|)⌉)` decimal places (not R's rounding algorithm
/// bit for bit). An all-zero or all-NaN matrix is left alone.
fn zapsmall(m: &mut Mat<f64>) {
    let Some(scale) = zapsmall_scale(m) else {
        return;
    };
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            m[(i, j)] = (m[(i, j)] * scale).round() / scale;
        }
    }
}

/// `10^d`, the scale of [`zapsmall`]'s rounding to `d = max(0, 7 −
/// ⌈log10 max|·|⌉)` decimal places; `None` on an all-zero or all-NaN matrix.
fn zapsmall_scale(m: &Mat<f64>) -> Option<f64> {
    let mx = m
        .col_iter()
        .flat_map(|c| c.iter().copied())
        .filter(|v| !v.is_nan())
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    if mx <= 0.0 || !mx.is_finite() {
        return None;
    }
    let digits = (7.0 - mx.log10().ceil()).max(0.0);
    Some(10f64.powf(digits))
}

/// The zeros of [`zapsmall`] without its rounding: every entry `zapsmall`
/// rounds to zero (`|v|·10^d` rounding to 0, so below `0.5·10^-d`, between
/// 5e-8 and 5e-7 of the largest `|·|`) set to zero, every other entry left
/// unrounded. An all-zero or all-NaN matrix is left alone.
fn zap(m: &mut Mat<f64>) {
    let Some(scale) = zapsmall_scale(m) else {
        return;
    };
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            if (m[(i, j)] * scale).round() == 0.0 {
                m[(i, j)] = 0.0;
            }
        }
    }
}

/// `resid(lm.fit(x = l[, basis], y = l[, cols]))`: the columns `cols` of `l`
/// minus their projection on the span of the columns `basis`. The basis is
/// orthonormalized in order (Gram-Schmidt, twice); a column whose norm falls
/// to `1e-7` of its own is left out, the rank rule of `lm.fit`'s `dqrdc2`.
fn residuals(l: &Mat<f64>, basis: &[usize], cols: &[usize]) -> Mat<f64> {
    let k = l.nrows();
    let project_out = |v: &mut [f64], q: &[Vec<f64>]| {
        for _ in 0..2 {
            for qv in q {
                let dot: f64 = v.iter().zip(qv).map(|(a, b)| a * b).sum();
                for (a, b) in v.iter_mut().zip(qv) {
                    *a -= dot * b;
                }
            }
        }
    };
    let norm = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>().sqrt();
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(basis.len());
    for &b in basis {
        let mut v: Vec<f64> = (0..k).map(|r| l[(r, b)]).collect();
        let before = norm(&v);
        project_out(&mut v, &q);
        let after = norm(&v);
        if after > 0.0 && after >= 1e-7 * before {
            v.iter_mut().for_each(|a| *a /= after);
            q.push(v);
        }
    }
    let mut out = Mat::<f64>::zeros(k, cols.len());
    for (j, &c) in cols.iter().enumerate() {
        let mut v: Vec<f64> = (0..k).map(|r| l[(r, c)]).collect();
        project_out(&mut v, &q);
        for (r, a) in v.into_iter().enumerate() {
            out[(r, j)] = a;
        }
    }
    out
}
