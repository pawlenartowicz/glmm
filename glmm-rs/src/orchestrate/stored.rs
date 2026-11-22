//! Readers of what a port stored of one fit: the stored records and their
//! inverses, `confint`, `contrast_test`, `pairwise`, `effect_sizes` and
//! `profile_intervals`.

use std::collections::HashMap;

use faer::Mat;

use crate::fit::CondvarCross;
use crate::formula::anova::{
    fixed_row, linear_note_line, untested_test, ChoiceReason, SsType, TermTable, TestChoice,
};
use crate::formula::pairwise::pairwise_of;
use crate::formula::{Adjust, By, Lowered, Pairwise, PairwiseIntervals, PairwiseRow, RatioKind};
use crate::inference::contrast::{
    contrast_f_of, robust_f_of, ContrastError, DfSource, NoDfReason, RobustInputs,
};
use crate::inference::cr2::Cr2Pieces;
use crate::inference::effect_size::{variance_parts, VarianceParts};
use crate::inference::profile::{
    better_optimum_message, profile_on, profile_params, weight_power_refusal,
};
use crate::inference::tails::{normal_quantile, t_quantile};
use crate::inference::wald::wald_chisq;
use crate::inference::{
    DenDf, Icc, LinearNote, LinearTest, Objective, ProfileError, ProfileParam, R2,
};
use crate::lmm::satterthwaite::SatterthwaiteState;
use crate::predict::FitParts;
use crate::{Family, FitError, FitOptions};

use super::design::table_of;
use super::fit::{lower_with, DfMethod};
use super::predict::relower;
use super::{adjust_tag, ratio_tag};

/// A Gaussian LMM's Satterthwaite record as plain numbers
/// ([`FitResult::satterthwaite_record`](super::FitResult::satterthwaite_record)): a port stores it on its fit and
/// passes it back to [`contrast_test`]. The `p_r × p_r` matrices are over the
/// retained columns, `p_r` of them, and every matrix is row-major.
///
/// The free θ coordinates are in the order the groupings were declared.
#[derive(Clone, Debug, PartialEq)]
pub struct SatterthwaiteRecord {
    /// `A_eff = 2·blockdiag(H_p⁺, τ̂²/ν)`, `(n_free + 1) × (n_free + 1)`, with
    /// `n_free` the free θ coordinates.
    pub a_eff: Vec<f64>,
    /// `∂C/∂θ̃_k`, one `p_r × p_r` matrix per free θ coordinate.
    pub jac_theta: Vec<Vec<f64>>,
    /// `C = Var(β̂)`, `p_r × p_r`.
    pub c: Vec<f64>,
    /// τ̂ = σ̂².
    pub tau: f64,
    /// The divisor of σ̂²: `N − p_r` on a REML fit, `N` on an ML fit.
    pub nu: f64,
    /// The retained columns, 0-based and ascending, on a rank-deficient fit;
    /// `None` when every column was retained.
    pub retained: Option<Vec<usize>>,
}

/// A robust fit's CR2 or HC2 pieces ([`FitResult::robust`](super::FitResult::robust))
/// as plain numbers: a port stores it on its fit and passes it back to the
/// readers of a robust fit. `t` is in the caller's coordinates, `p` rows
/// (zero on an aliased column); every other field is in the whitened
/// coordinates β_Z of the retained columns (β = t β_Z), `p_Z = t.len() / p`
/// wide (0 when `p` is 0). Every matrix is flat and column-major; cluster `k`
/// is the `k`-th level of the grouping factor that has rows, or row `k`
/// without one.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustRecord {
    /// The map β = t β_Z, `p × p_Z`.
    pub t: Vec<f64>,
    /// E_k r_k, `p_Z × G`: column `k` for cluster `k`.
    pub er: Vec<f64>,
    /// F_k = E_k Z_k, `p_Z × p_Z`, per cluster of more than one row in
    /// cluster order.
    pub f: Vec<Vec<f64>>,
    /// E_k Φ_k E_kᵀ, `p_Z × p_Z`, per cluster of `f`.
    pub epe: Vec<Vec<f64>>,
    /// Per one-row cluster in cluster order, a column of `p_Z`: E_k (so
    /// F_k = e_k z_kᵀ and E_kΦ_kE_kᵀ = φ_k e_k e_kᵀ), `p_Z × n₁`.
    pub row_e: Vec<f64>,
    /// The whitened design row z_k of each one-row cluster, `p_Z × n₁`.
    pub row_x: Vec<f64>,
    /// The scalar Φ_k of each one-row cluster.
    pub row_phi: Vec<f64>,
    /// The cluster of each null direction.
    pub null_cluster: Vec<usize>,
    /// Unit directions n, `p_Z` each, with H_(−k) n = 0 for their cluster k:
    /// a contrast c with (tᵀc)ᵀn ≠ 0 has no information outside cluster k.
    pub null: Vec<Vec<f64>>,
}

impl From<Cr2Pieces> for RobustRecord {
    fn from(pc: Cr2Pieces) -> Self {
        let flat = |a: &Mat<f64>| -> Vec<f64> {
            (0..a.ncols())
                .flat_map(|k| a.col_as_slice(k).iter().copied())
                .collect()
        };
        let (f, epe) = pc.whole.iter().map(|(f, e)| (flat(f), flat(e))).unzip();
        let (null_cluster, null) = pc.null.into_iter().unzip();
        RobustRecord {
            t: flat(&pc.t),
            er: flat(&pc.er),
            f,
            epe,
            row_e: flat(&pc.row_e),
            row_x: flat(&pc.row_x),
            row_phi: pc.row_phi,
            null_cluster,
            null,
        }
    }
}

/// A fit's conditional covariance across groupings
/// ([`FitResult::condvar_cross`](super::FitResult::condvar_cross)) as plain numbers: a port stores it on its
/// fit and passes it back to [`predict`](super::predict) in [`StoredParts`]. The fields are
/// those of the crate-private `fit::CondvarCross`, which documents the
/// layout.
#[derive(Clone, Debug, PartialEq)]
pub struct CondvarCrossRecord {
    /// The declaration index of the grouping whose levels are the families.
    pub family_grouping: usize,
    /// Per family the vech of its coefficients' covariance.
    pub family: Vec<f64>,
    /// The vech of the tail levels' covariance.
    pub tail: Vec<f64>,
    /// Per family plus one, where its coupling entries start.
    pub coupling_at: Vec<u32>,
    /// The tail level each coupling entry names.
    pub coupling_cols: Vec<u32>,
    /// Per coupling entry, one value per family coefficient.
    pub coupling: Vec<f64>,
    /// The declaration index of the grouping `nested_slots` maps; 0 when
    /// that is empty.
    pub nested_grouping: usize,
    /// The nested grouping's child slots when it was declared crossed;
    /// empty otherwise.
    pub nested_slots: Vec<u32>,
    /// The declared nested grouping's padded slot per observed pair when the
    /// fit read it crossed; empty otherwise.
    pub pair_slots: Vec<u32>,
}

impl From<CondvarCross> for CondvarCrossRecord {
    fn from(c: CondvarCross) -> Self {
        CondvarCrossRecord {
            family_grouping: c.family_grouping,
            family: c.family,
            tail: c.tail,
            coupling_at: c.coupling_at,
            coupling_cols: c.coupling_cols,
            coupling: c.coupling,
            nested_grouping: c.nested_grouping,
            nested_slots: c.nested_slots,
            pair_slots: c.pair_slots,
        }
    }
}

impl From<&CondvarCrossRecord> for CondvarCross {
    fn from(c: &CondvarCrossRecord) -> Self {
        CondvarCross {
            family_grouping: c.family_grouping,
            family: c.family.clone(),
            tail: c.tail.clone(),
            coupling_at: c.coupling_at.clone(),
            coupling_cols: c.coupling_cols.clone(),
            coupling: c.coupling.clone(),
            nested_grouping: c.nested_grouping,
            nested_slots: c.nested_slots.clone(),
            pair_slots: c.pair_slots.clone(),
        }
    }
}

/// Confidence intervals for the fixed effects, the one rule both ports'
/// `confint` call: per coefficient `β̂ ± q·se`. `df_method` is the fit's, and
/// says how its `summary()` tests. On a fit that tests with t (`residual`:
/// `n − p` on OLS and the dispersion GLMs; `satterthwaite`; and the robust
/// `bell_mccaffrey` and `pustejovsky_tipton`), `q` is the Student t quantile
/// on `coef_df[i]` where that df is finite and positive, and the interval is
/// NaN where it is not: a lost df leaves no interval. On a fit that tests
/// with z (`none`: GLM and GLMM; `disabled` and `failed`: a Gaussian mixed
/// model whose Satterthwaite pass was switched off or failed), `q` is the
/// standard normal quantile, or the t quantile on a finite, positive df. The
/// quantiles are taken at `(1 − level)/2` and `1 − (1 − level)/2`, as
/// `confint.lm` takes them. An interval is also NaN where `β̂` or `se` is (an
/// aliased column).
///
/// # Errors
///
/// When `level` is not strictly between 0 and 1, or the three slices differ
/// in length.
pub fn confint(
    beta: &[f64],
    se: &[f64],
    coef_df: &[f64],
    df_method: DfMethod,
    level: f64,
) -> Result<Vec<[f64; 2]>, String> {
    let tests_with_t = df_method.tests_with_t();
    if !(level > 0.0 && level < 1.0) {
        return Err(format!(
            "level must be a number strictly between 0 and 1, got {level}"
        ));
    }
    if se.len() != beta.len() || coef_df.len() != beta.len() {
        return Err(format!(
            "beta, se and coef_df must have the same length, got {}, {} and {}",
            beta.len(),
            se.len(),
            coef_df.len()
        ));
    }
    let a = (1.0 - level) / 2.0;
    Ok((0..beta.len())
        .map(|i| {
            let df = coef_df[i];
            let (lo, hi) = if df.is_finite() && df > 0.0 {
                (t_quantile(a, df), t_quantile(1.0 - a, df))
            } else if tests_with_t {
                (f64::NAN, f64::NAN)
            } else {
                (normal_quantile(a), normal_quantile(1.0 - a))
            };
            [beta[i] + lo * se[i], beta[i] + hi * se[i]]
        })
        .collect())
}

/// A row-major copy of `m`.
fn row_major(m: &Mat<f64>) -> Vec<f64> {
    (0..m.nrows())
        .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
        .collect()
}

/// A Satterthwaite pass's state as the record a port stores.
pub(super) fn record_of(st: &SatterthwaiteState) -> SatterthwaiteRecord {
    SatterthwaiteRecord {
        a_eff: row_major(&st.a_eff),
        jac_theta: st.jac_theta.iter().map(row_major).collect(),
        c: row_major(&st.c),
        tau: st.tau,
        nu: st.nu,
        retained: st.retained.clone(),
    }
}

/// The inverse of [`record_of`] on a fit with `p` coefficients, refusing a
/// record whose sizes do not fit together or with `p`.
fn record_state(r: SatterthwaiteRecord, p: usize) -> Result<SatterthwaiteState, String> {
    let malformed = || "the fit's satterthwaite_record is malformed: refit the model".to_string();
    let k = (r.a_eff.len() as f64).sqrt().round() as usize;
    if k == 0 || k * k != r.a_eff.len() {
        return Err(malformed());
    }
    let pr = r.retained.as_ref().map_or(p, Vec::len);
    let square = |v: &[f64]| v.len() == pr * pr;
    if !square(&r.c) || r.jac_theta.len() != k - 1 || !r.jac_theta.iter().all(|j| square(j)) {
        return Err(malformed());
    }
    if let Some(kept) = &r.retained {
        if kept.windows(2).any(|w| w[0] >= w[1]) || kept.last().is_some_and(|&j| j >= p) {
            return Err(malformed());
        }
    }
    let mat = |v: &[f64], n: usize| Mat::from_fn(n, n, |i, j| v[i * n + j]);
    Ok(SatterthwaiteState::from_record(
        mat(&r.a_eff, k),
        r.jac_theta.iter().map(|j| mat(j, pr)).collect(),
        mat(&r.c, pr),
        r.tau,
        r.nu,
        r.retained,
    ))
}

/// A robust fit's stored pieces and covariance, flat and row-major.
struct StoredRobust {
    pieces: Option<Cr2Pieces>,
    vcov: Vec<f64>,
}

impl StoredRobust {
    /// `Some` on a robust `df_method`, refusing a `robust` record or a
    /// `model_vcov` whose sizes do not fit `p`. A fit with no endpoint has
    /// neither, and its robust tests are NaN.
    fn new(
        df_method: DfMethod,
        vcov: &[Vec<f64>],
        model_vcov: &[Vec<f64>],
        robust: Option<RobustRecord>,
        p: usize,
    ) -> Result<Option<Self>, String> {
        if !df_method.is_robust() {
            return Ok(None);
        }
        let malformed = || "the fit's robust record is malformed: refit the model".to_string();
        if !(model_vcov.is_empty()
            || model_vcov.len() == p && model_vcov.iter().all(|r| r.len() == p))
        {
            return Err(malformed());
        }
        let pieces = robust
            .map(|r| record_pieces(r, p).ok_or_else(malformed))
            .transpose()?;
        Ok(Some(StoredRobust {
            pieces,
            vcov: vcov.concat(),
        }))
    }

    fn inputs<'a>(&'a self, aliased: &'a [bool]) -> RobustInputs<'a> {
        RobustInputs::new(self.pieces.as_ref(), &self.vcov, aliased)
    }
}

/// The inverse of `RobustRecord::from`, `None` when its sizes do not fit
/// together or with `p`.
fn record_pieces(r: RobustRecord, p: usize) -> Option<Cr2Pieces> {
    let pz = match p {
        0 => r.t.is_empty().then_some(0)?,
        _ => (r.t.len() % p == 0).then(|| r.t.len() / p)?,
    };
    let cols = |v: &[f64]| match pz {
        0 => v.is_empty().then_some(0),
        _ => (v.len() % pz == 0).then(|| v.len() / pz),
    };
    let mat =
        |v: &[f64], nrows: usize, ncols: usize| Mat::from_fn(nrows, ncols, |i, k| v[k * nrows + i]);
    let g = cols(&r.er)?;
    let n1 = r.row_phi.len();
    if r.f.len() != r.epe.len()
        || r.f.iter().chain(&r.epe).any(|m| m.len() != pz * pz)
        || r.row_e.len() != pz * n1
        || r.row_x.len() != r.row_e.len()
        || r.null.len() != r.null_cluster.len()
        || r.null.iter().any(|n| n.len() != pz)
        || r.null_cluster.iter().any(|&k| k >= g)
    {
        return None;
    }
    Some(Cr2Pieces {
        t: mat(&r.t, p, pz),
        er: mat(&r.er, pz, g),
        whole: r
            .f
            .iter()
            .zip(&r.epe)
            .map(|(f, e)| (mat(f, pz, pz), mat(e, pz, pz)))
            .collect(),
        row_e: mat(&r.row_e, pz, n1),
        row_x: mat(&r.row_x, pz, n1),
        row_phi: r.row_phi,
        null: r.null_cluster.into_iter().zip(r.null).collect(),
    })
}

/// The df record of a fit from what a port stored of it, as `Fit`'s own df
/// record: `df_method` ([`FitResult::df_method`]) `"satterthwaite"` rebuilds
/// it from `record`; `"residual"` builds it as the fit does, `C` being `vcov`
/// on the columns not `aliased` and `ν` being `nobs` less those columns;
/// `"disabled"`, `"failed"` and `"none"` give the reason there is none
/// (`"none"` is the families with no df, [`NoDfReason::NotGaussianMixed`]).
/// On a robust fit, the model record its joint tests read, `vcov` its model
/// covariance: `"bell_mccaffrey"` as `"residual"`, `"pustejovsky_tipton"`
/// from `record`, or [`NoDfReason::Disabled`] without one.
///
/// # Errors
///
/// A `"satterthwaite"` fit with no `record`, or a malformed one.
fn stored_df_state(
    df_method: DfMethod,
    beta: &[f64],
    vcov: &[Vec<f64>],
    aliased: &[bool],
    nobs: usize,
    record: Option<SatterthwaiteRecord>,
) -> Result<Result<SatterthwaiteState, NoDfReason>, String> {
    let p = beta.len();
    Ok(match df_method {
        DfMethod::Satterthwaite => {
            let record = record.ok_or_else(|| {
                "a fit whose df_method is \"satterthwaite\" needs its satterthwaite_record"
                    .to_string()
            })?;
            Ok(record_state(record, p)?)
        }
        // shortcut: a failed pass reads as switched off; a record field
        // would tell them apart.
        DfMethod::PustejovskyTipton => match record {
            Some(r) => Ok(record_state(r, p)?),
            None => Err(NoDfReason::Disabled),
        },
        DfMethod::Residual | DfMethod::BellMcCaffrey => {
            let kept: Vec<usize> = (0..p).filter(|&j| !aliased[j]).collect();
            let c = Mat::from_fn(kept.len(), kept.len(), |a, b| vcov[kept[a]][kept[b]]);
            let mut st = SatterthwaiteState::residual(c, nobs as f64 - kept.len() as f64, beta);
            if kept.len() < p {
                st.retained = Some(kept);
            }
            Ok(st)
        }
        DfMethod::Disabled => Err(NoDfReason::Disabled),
        DfMethod::Failed => Err(NoDfReason::Failed),
        DfMethod::NoDf => Err(NoDfReason::NotGaussianMixed),
    })
}

/// One Wald test of `Lβ = 0` on the fields a port stored of one fit, with no
/// refit: `l` is the `rows × p` matrix `L`, row-major over all `p`
/// coefficients, aliased ones included. The test follows `df_method`
/// ([`FitResult::df_method`](super::FitResult::df_method)):
///
/// - `"satterthwaite"` (a Gaussian LMM) and `"residual"` (OLS, a Gamma or
///   inverse Gaussian GLM with φ estimated): the F of [`Fit::contrast_f`](crate::Fit::contrast_f) on
///   the fit's df record, rebuilt from `record` on the first and, on the
///   second, as the fit builds it: `C` is `vcov` on the columns not
///   `aliased`, `ν = nobs` less those columns;
/// - `"disabled"` and `"failed"` (a Gaussian LMM with no pass): the NaN F row
///   with the no-df note and its reason, as the per-term table gives it;
/// - `"none"`: the Wald χ² of the per-term table's Wald rows;
/// - `"pustejovsky_tipton"` and `"bell_mccaffrey"` (a robust fit): the F of
///   [`Fit::contrast_f`](crate::Fit::contrast_f) on a robust fit, one row from `robust` (its pieces),
///   `vcov` and `model_vcov`, more rows model-based, from `record` and
///   `model_vcov` as the first two routes read them.
///
/// The result is a one-row [`TermTable`] labelled `contrast`, with `Sum Sq`
/// and `Mean Sq` from `dispersion` on an F row, and `names`, the fit's
/// coefficient names, as its `coef_names`. A contrast has no SS type, so
/// `ss_type` (Type III) and `choice.reason` (requested) are placeholders,
/// which the ports drop.
///
/// # Errors
///
/// `L` with no row, with a length other than `rows·p` or with a value that is
/// not finite; a `"satterthwaite"` fit with no `record`, or a malformed one;
/// a malformed `robust` or `model_vcov`.
#[allow(clippy::too_many_arguments)]
pub fn contrast_test(
    df_method: DfMethod,
    beta: &[f64],
    vcov: &[Vec<f64>],
    model_vcov: &[Vec<f64>],
    aliased: &[bool],
    nobs: usize,
    dispersion: f64,
    record: Option<SatterthwaiteRecord>,
    robust: Option<RobustRecord>,
    names: &[String],
    l: &[f64],
    rows: usize,
) -> Result<TermTable, String> {
    let p = beta.len();
    if rows == 0 {
        return Err("L must have at least one row".to_string());
    }
    if l.len() != rows * p {
        return Err(format!(
            "L must have {p} columns, one per coefficient, aliased ones included; got {} \
             values in {rows} row(s)",
            l.len()
        ));
    }
    if let Some(v) = l.iter().find(|v| !v.is_finite()) {
        return Err(format!("L must be finite numbers, got {v}"));
    }
    let test = (|| -> Result<LinearTest, String> {
        let den_df = if df_method.residual_df() {
            DenDf::Residual
        } else {
            DenDf::Satterthwaite
        };
        let robust = StoredRobust::new(df_method, vcov, model_vcov, robust, p)?;
        // A robust fit's joint tests read its model covariance; a fit with
        // no endpoint has none, and its `vcov` is NaN.
        let mv = if robust.is_some() && !model_vcov.is_empty() {
            model_vcov
        } else {
            vcov
        };
        let model = stored_df_state(df_method, beta, mv, aliased, nobs, record)?;
        if let Some(r) = &robust {
            let ri = r.inputs(aliased);
            let den_df = den_df.on_robust_fit(rows);
            return Ok(
                match robust_f_of(beta, &ri, model.as_ref().map_err(|e| *e), l, rows) {
                    Ok(cf) => LinearTest::from_contrast_f(cf, den_df),
                    Err(ContrastError::NoDf(reason)) => {
                        untested_test(true, den_df, vec![LinearNote::NoDf { reason }])
                    }
                },
            );
        }
        Ok(match model {
            Ok(st) => LinearTest::from_contrast_f(contrast_f_of(beta, &st, l, rows), den_df),
            Err(NoDfReason::NotGaussianMixed) => wald_chisq(beta, vcov, aliased, l, rows),
            Err(reason) => untested_test(true, den_df, vec![LinearNote::NoDf { reason }]),
        })
    })()?;
    Ok(TermTable {
        ss_type: SsType::Type3,
        choice: TestChoice {
            test: test.kind,
            reason: ChoiceReason::Requested,
        },
        fixed: vec![fixed_row("contrast".to_string(), test, dispersion)],
        notes: Vec::new(),
        coef_names: names.to_vec(),
    })
}

/// The ports' `pairwise`: [`crate::formula::pairwise()`] of a stored fit, with
/// no refit. The training columns and `family` are what the fit was made
/// from, as [`run_fit`](super::run_fit) took them, and are lowered again for the factor
/// levels and the numeric means of the reference grid; the `weights=` and
/// `offset=` arguments are not needed, since a comparison `l'β̂` reads
/// neither. `df_method`, `beta`, `vcov`, `model_vcov`, `aliased`, `nobs`,
/// `record` and `robust` are the fit's stored fields, read as
/// [`contrast_test`] reads them. `by` names
/// the variable the comparisons are made within: its levels, or, with
/// `by_values`, those values of a numeric variable.
///
/// # Errors
///
/// `by_values` without `by`; the errors of
/// [`contrast_test`] on `record`; and every error of
/// [`crate::formula::pairwise()`].
#[allow(clippy::too_many_arguments)]
pub fn pairwise(
    formula: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
    family: Family,
    df_method: DfMethod,
    beta: &[f64],
    vcov: &[Vec<f64>],
    model_vcov: &[Vec<f64>],
    aliased: &[bool],
    nobs: usize,
    record: Option<SatterthwaiteRecord>,
    robust: Option<RobustRecord>,
    factor: &str,
    by: Option<&str>,
    by_values: Option<Vec<f64>>,
    adjust: Adjust,
) -> Result<Pairwise, String> {
    let by = match (by, by_values.as_deref()) {
        (None, None) => None,
        (Some(name), None) => Some(By::Levels(name)),
        (Some(name), Some(values)) => Some(By::Values(name, values)),
        (None, Some(_)) => {
            return Err(
                "the values to compare at need by, the numeric variable they are values of"
                    .to_string(),
            )
        }
    };
    let lowered = relower(formula, numeric_columns, factor_columns, family, None, None)?;
    let robust = StoredRobust::new(df_method, vcov, model_vcov, robust, beta.len())?;
    let (inputs, model);
    let df = match &robust {
        Some(r) => {
            inputs = r.inputs(aliased);
            DfSource::Robust(&inputs)
        }
        None => {
            model = stored_df_state(df_method, beta, vcov, aliased, nobs, record)?;
            DfSource::Model(model.as_ref().map_err(|reason| *reason))
        }
    };
    pairwise_of(
        formula, &lowered, beta, vcov, aliased, df, factor, by, adjust,
    )
    .map_err(|e| e.to_string())
}

/// A [`Pairwise`] as the plain columns a port stores, one entry per row in
/// each vector, flattened by [`pairwise_columns`].
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseColumns {
    /// [`Pairwise::factor`].
    pub factor: String,
    /// [`Pairwise::by`].
    pub by: Option<String>,
    /// [`Pairwise::adjust`], as [`adjust_tag`] names it.
    pub adjust: &'static str,
    /// [`Pairwise::ratio`]: `"odds_ratio"` or `"ratio"`, `None` on a link
    /// that is neither log nor logit.
    pub ratio_kind: Option<&'static str>,
    /// The factor's number of levels, which [`pairwise_confint`] takes back.
    pub levels: usize,
    /// [`Pairwise::family_sizes`], which the port's adjustment note reads.
    pub family_sizes: Vec<usize>,
    /// [`PairwiseRow::contrast`].
    pub contrast: Vec<String>,
    /// [`PairwiseRow::by_level`], `""` without a `by`.
    pub by_level: Vec<String>,
    /// [`PairwiseRow::estimate`].
    pub estimate: Vec<f64>,
    /// [`PairwiseRow::se`].
    pub se: Vec<f64>,
    /// [`PairwiseRow::df`].
    pub df: Vec<f64>,
    /// [`PairwiseRow::statistic`].
    pub statistic: Vec<f64>,
    /// [`PairwiseRow::p_value`], unadjusted.
    pub p_value: Vec<f64>,
    /// [`PairwiseRow::p_adjusted`].
    pub p_adjusted: Vec<f64>,
    /// [`PairwiseRow::ratio`].
    pub ratio: Vec<f64>,
    /// [`PairwiseRow::ratio_se`].
    pub ratio_se: Vec<f64>,
    /// Each row's notes as plain sentences, `names` the fit's coefficient
    /// names (an aliased column; no df and why).
    pub row_notes: Vec<Vec<String>>,
}

/// A [`Pairwise`] as [`PairwiseColumns`], for `glmm-bind`'s records; `names` are the
/// fit's coefficient names, which the notes of an aliased column name.
pub fn pairwise_columns(pw: &Pairwise, names: &[String]) -> PairwiseColumns {
    let col = |f: fn(&PairwiseRow) -> f64| pw.rows.iter().map(f).collect();
    PairwiseColumns {
        factor: pw.factor.clone(),
        by: pw.by.clone(),
        adjust: adjust_tag(pw.adjust),
        ratio_kind: pw.ratio.map(ratio_tag),
        levels: pw.levels,
        family_sizes: pw.family_sizes.clone(),
        contrast: pw.rows.iter().map(|r| r.contrast.clone()).collect(),
        by_level: pw
            .rows
            .iter()
            .map(|r| r.by_level.clone().unwrap_or_default())
            .collect(),
        estimate: col(|r| r.estimate),
        se: col(|r| r.se),
        df: col(|r| r.df),
        statistic: col(|r| r.statistic),
        p_value: col(|r| r.p_value),
        p_adjusted: col(|r| r.p_adjusted),
        ratio: col(|r| r.ratio),
        ratio_se: col(|r| r.ratio_se),
        row_notes: pw
            .rows
            .iter()
            .map(|r| {
                r.notes
                    .iter()
                    .map(|n| linear_note_line(*n, names))
                    .collect()
            })
            .collect(),
    }
}

/// The ports' `confint` of a stored pairwise table: [`Pairwise::confint`] on
/// the rows rebuilt from what [`pairwise_columns`] gave the port (each row's
/// `estimate`, `se`, `df` and unadjusted `p_value`, the factor's `levels`,
/// and the table's `adjust` and `ratio` kind). Nothing is refitted and no
/// grid is built.
///
/// # Errors
///
/// Columns of different lengths, or a row
/// count that is not a whole number of families of `levels·(levels − 1)/2`
/// rows; and the error of [`Pairwise::confint`] on `level`.
#[allow(clippy::too_many_arguments)]
pub fn pairwise_confint(
    estimate: &[f64],
    se: &[f64],
    df: &[f64],
    p_value: &[f64],
    adjust: Adjust,
    levels: usize,
    ratio: Option<RatioKind>,
    level: f64,
) -> Result<PairwiseIntervals, String> {
    let n = estimate.len();
    let family = levels * levels.saturating_sub(1) / 2;
    if se.len() != n || df.len() != n || p_value.len() != n || family == 0 || n % family != 0 {
        return Err(
            "the stored pairwise table is malformed: make it again with pairwise()".to_string(),
        );
    }
    let rows = (0..n)
        .map(|i| PairwiseRow {
            contrast: String::new(),
            by_level: None,
            estimate: estimate[i],
            se: se[i],
            df: df[i],
            statistic: f64::NAN,
            p_value: p_value[i],
            p_adjusted: f64::NAN,
            ratio: f64::NAN,
            ratio_se: f64::NAN,
            notes: Vec::new(),
        })
        .collect();
    let pw = Pairwise {
        factor: String::new(),
        by: None,
        adjust,
        ratio,
        rows,
        family_sizes: Vec::new(),
        levels,
    };
    pw.confint(level).map_err(|e| e.to_string())
}

/// The numbers of one fit that [`predict`](super::predict), [`simulate`](super::simulate) and
/// [`effect_sizes`] read, as a port stored them from its [`FitResult`](super::FitResult):
/// `beta`, `vcov`, `model_vcov`, `varcorr`, `ranef`, `ranef_levels`,
/// `condvar`, `condvar_x`, `condvar_cross` and `dispersion` as that result
/// holds them.
/// A port may store `beta` with NA in place of an aliased coefficient's NaN;
/// both are NaN here, and contribute zero, unless every coefficient is NaN
/// (see [`predict`](super::predict)).
#[derive(Clone, Debug)]
pub struct StoredParts {
    /// [`FitResult::beta`](super::FitResult::beta).
    pub beta: Vec<f64>,
    /// [`FitResult::vcov`](super::FitResult::vcov).
    pub vcov: Vec<Vec<f64>>,
    /// [`FitResult::model_vcov`](super::FitResult::model_vcov): when not empty (a robust fit), what
    /// [`predict`](super::predict) reads in place of `vcov`, noting that its standard error
    /// is model-based.
    pub model_vcov: Vec<Vec<f64>>,
    /// [`FitResult::varcorr`](super::FitResult::varcorr).
    pub varcorr: Vec<Vec<f64>>,
    /// [`FitResult::ranef`](super::FitResult::ranef).
    pub ranef: Vec<f64>,
    /// [`FitResult::ranef_levels`](super::FitResult::ranef_levels).
    pub ranef_levels: Vec<usize>,
    /// [`FitResult::condvar`](super::FitResult::condvar).
    pub condvar: Vec<f64>,
    /// [`FitResult::condvar_x`](super::FitResult::condvar_x).
    pub condvar_x: Vec<f64>,
    /// [`FitResult::condvar_cross`](super::FitResult::condvar_cross).
    pub condvar_cross: Option<CondvarCrossRecord>,
    /// [`FitResult::dispersion`](super::FitResult::dispersion).
    pub dispersion: f64,
    /// [`FitResult::weight_power`](super::FitResult::weight_power).
    pub weight_power: f64,
}

impl StoredParts {
    /// The kernel's view of these numbers for a model of `family`, `cross`
    /// the kernel form of `condvar_cross`.
    pub(super) fn fit_parts<'a>(
        &'a self,
        family: Family,
        cross: Option<&'a CondvarCross>,
    ) -> FitParts<'a> {
        FitParts {
            family,
            beta: &self.beta,
            vcov: std::borrow::Cow::Borrowed(if self.model_vcov.is_empty() {
                &self.vcov
            } else {
                &self.model_vcov
            }),
            robust: !self.model_vcov.is_empty(),
            varcorr: &self.varcorr,
            ranef: &self.ranef,
            ranef_levels: &self.ranef_levels,
            condvar: &self.condvar,
            condvar_x: &self.condvar_x,
            condvar_cross: cross,
            dispersion: self.dispersion,
            weight_power: self.weight_power,
            null_basis: std::borrow::Cow::Borrowed(&[]),
        }
    }
}

/// The ports' `r2()` and `icc()`: [`crate::Fit::r2`] and [`crate::Fit::icc`]
/// of a stored fit, computed when called. The training columns, `family`,
/// `weights` and `offset` are read as [`predict`](super::predict) reads them; `parts`
/// gives `beta`, `varcorr` and `dispersion`, and `aliased`, `converged` and
/// `objective` are the fit's stored fields.
/// A model `Fit::r2` refuses (no random effects, a family and link it does
/// not cover) gets NaN values, as does a fit that did not converge.
///
/// On a converged Poisson or negative binomial model with random effects the
/// values need a refit of the null model. `null_refit = false` returns
/// `Ok(None)` there instead of refitting, so a port's `summary()` shows the
/// values only where they cost no refit.
///
/// # Errors
///
/// The lowering errors of the training columns, and every error of
/// [`crate::Fit::r2`] other than a model it does not cover: stored `beta` or
/// `varcorr` that do not match the lowered model, and the errors of the null
/// model's refit.
#[allow(clippy::too_many_arguments)]
pub fn effect_sizes(
    formula: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
    family: Family,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    parts: &StoredParts,
    aliased: &[bool],
    converged: bool,
    objective: Objective,
    null_refit: bool,
) -> Result<Option<(R2, Icc)>, String> {
    let lowered = relower(
        formula,
        numeric_columns,
        factor_columns,
        family,
        weights,
        offset,
    )?;
    // The families whose residual variance reads the null model's mean in
    // `variance_parts` (its `null_mu`) - change together.
    let refits = converged
        && lowered.model.re.is_some()
        && matches!(
            lowered.model.family,
            Family::Poisson { .. } | Family::NegativeBinomial { .. }
        );
    if refits && !null_refit {
        return Ok(None);
    }
    let parts = match variance_parts(
        lowered.n,
        &parts.beta,
        &parts.varcorr,
        aliased,
        converged,
        parts.dispersion,
        objective,
        &lowered.data(),
        lowered.spec(),
    ) {
        Ok(parts) => parts,
        Err(FitError::Unsupported(_)) => VarianceParts::NAN,
        Err(e) => return Err(e.to_string()),
    };
    Ok(Some((parts.r2(), parts.icc())))
}

/// [`profile_intervals`]' rows, each a parameter named by [`profile_names`].
#[derive(Debug)]
pub struct ProfileResult {
    /// The row names, in glmm's order ([`crate::inference::ProfileIntervals::rows`]):
    /// per grouping the SDs and correlations column-major, `sigma`, then the
    /// coefficients. lme4 2.0 lists a block's SDs before its correlations, so
    /// a comparison with lme4 matches rows by name.
    pub names: Vec<String>,
    /// The ML estimate of each row; NaN for a correlation next to an SD
    /// estimated at 0.
    pub estimate: Vec<f64>,
    /// The lower bound of each row; NaN when undetermined.
    pub lower: Vec<f64>,
    /// The upper bound of each row; NaN when undetermined.
    pub upper: Vec<f64>,
    /// The confidence level.
    pub level: f64,
}

/// The profile-likelihood intervals at `level` of the Gaussian mixed model a
/// port fitted, [`crate::Fit::profile_intervals`]' rows, on the port's stored
/// training columns as [`effect_sizes`] reads them. The model is lowered with
/// the fit's `weights=`, `offset=` and `reml`, then refitted once, by ML, for
/// the profiles; no fit with `reml` is made. `weight_power` and
/// `weight_power_estimated` are the fit's ([`FitResult::weight_power`](super::FitResult::weight_power)): a
/// power that was estimated or is not 1 is refused, as
/// [`crate::Fit::profile_intervals`] refuses it.
///
/// `parm` names the rows: [`profile_names`]' names, `"theta_"` for every
/// random-effect SD and correlation and `sigma` (lme4's shortcut), and
/// `"beta_"` for every coefficient. `None` is every row. The rows come in
/// glmm's order ([`ProfileResult::names`]) whatever the order of `parm`.
///
/// # Errors
///
/// A name in `parm` the model does not have (the message lists the valid
/// ones), the lowering errors of the columns, and every
/// [`crate::inference::ProfileError`] as its message: a model without random
/// effects or of another family than Gaussian, an aliased column, a `level`
/// outside `(0, 1)`.
#[allow(clippy::too_many_arguments)]
pub fn profile_intervals(
    formula: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
    family: Family,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    reml: bool,
    weight_power: f64,
    weight_power_estimated: bool,
    level: f64,
    parm: Option<Vec<String>>,
) -> Result<ProfileResult, String> {
    weight_power_refusal(weight_power, weight_power_estimated).map_err(|e| e.to_string())?;
    let table = table_of(numeric_columns, factor_columns)?;
    let opts = FitOptions {
        satterthwaite: false,
        reml: Some(reml),
        ..FitOptions::default()
    };
    let lowered = lower_with(formula, &table, family, opts, weights, offset)?.lowered;
    let which = match parm {
        None => None,
        Some(parm) => Some(profile_params_named(&lowered, &parm)?),
    };
    let intervals = profile_on(&lowered.data(), lowered.spec(), level, which.as_deref()).map_err(
        |e| match e {
            ProfileError::BetterOptimum { param, drop } => {
                better_optimum_message(&profile_names(&lowered, &[param])[0], drop)
            }
            e => e.to_string(),
        },
    )?;
    let params: Vec<ProfileParam> = intervals.rows.iter().map(|r| r.param).collect();
    Ok(ProfileResult {
        names: profile_names(&lowered, &params),
        estimate: intervals.rows.iter().map(|r| r.estimate).collect(),
        lower: intervals.rows.iter().map(|r| r.lower).collect(),
        upper: intervals.rows.iter().map(|r| r.upper).collect(),
        level: intervals.level,
    })
}

/// The parameters `parm` names on `lowered`, as [`profile_intervals`] reads
/// them; an unknown name is an error listing the valid ones.
fn profile_params_named(lowered: &Lowered, parm: &[String]) -> Result<Vec<ProfileParam>, String> {
    let all = profile_params(lowered.spec(), lowered.col_names.len());
    let names = profile_names(lowered, &all);
    let mut out = Vec::new();
    for name in parm {
        match name.as_str() {
            "beta_" => out.extend(all.iter().filter(|q| matches!(q, ProfileParam::Beta(_)))),
            "theta_" => out.extend(all.iter().filter(|q| !matches!(q, ProfileParam::Beta(_)))),
            _ => match names.iter().position(|n| n == name) {
                Some(i) => out.push(all[i]),
                None => {
                    let valid: Vec<String> = names.iter().map(|n| format!("{n:?}")).collect();
                    return Err(format!(
                        "parm {name:?} is not a parameter of this model; the parameters are {}, \
                         \"theta_\" (the random-effect parameters and sigma) and \"beta_\" (the \
                         coefficients)",
                        valid.join(", ")
                    ));
                }
            },
        }
    }
    Ok(out)
}

/// lme4's `oldNames = FALSE` row names of `params` on `lowered`: `sd_<term>|<group>`,
/// `cor_<row term>.<col term>|<group>`, `sigma`, then `Lowered::col_names`.
///
/// `params` are those of a fit of `lowered` (as [`crate::inference::ProfileIntervals`]
/// rows carry them); `<term>` and `<group>` are [`crate::formula::ReGroupInfo`]'s
/// `terms` and `name`.
///
/// # Panics
///
/// On a parameter `lowered` does not have: a grouping, block column or
/// coefficient past its end.
pub fn profile_names(lowered: &Lowered, params: &[ProfileParam]) -> Vec<String> {
    params
        .iter()
        .map(|&param| match param {
            ProfileParam::Sd { group, term } => {
                let g = &lowered.re_groups[group];
                format!("sd_{}|{}", g.terms[term], g.name)
            }
            ProfileParam::Cor { group, row, col } => {
                let g = &lowered.re_groups[group];
                format!("cor_{}.{}|{}", g.terms[row], g.terms[col], g.name)
            }
            ProfileParam::Sigma => "sigma".to_string(),
            ProfileParam::Beta(j) => lowered.col_names[j].clone(),
        })
        .collect()
}
