//! Typed reads of a call's arguments, or of a record passed back, naming the
//! argument (or `record.field`) in every error.

use std::collections::HashMap;

use crate::coerce;
use crate::error::BindError;
use crate::value::{Record, Value};

/// The data columns as the engine's orchestration functions take them.
pub(crate) type NumericColumns = HashMap<String, Vec<f64>>;
/// Factor columns as `(levels, codes)`.
pub(crate) type FactorColumns = HashMap<String, (Vec<String>, Vec<u32>)>;

/// A record's fields being read. [`crate::call`] has already refused unknown
/// keys and missing required arguments and filled absent optionals.
pub(crate) struct Inputs {
    rec: Record,
    /// `""` for a call's arguments, `"name."` for a nested record.
    ctx: String,
}

macro_rules! read {
    ($(#[$m:meta])* $req:ident, $opt:ident, $t:ty, $f:expr) => {
        $(#[$m])*
        #[allow(dead_code)]
        pub(crate) fn $req(&mut self, name: &str) -> Result<$t, BindError> {
            let v = self.rec.take(name);
            if v.is_null() {
                return Err(self.bad(name, "is missing"));
            }
            $f(v).map_err(|w: String| self.bad(name, &w))
        }

        $(#[$m])*
        #[allow(dead_code)]
        pub(crate) fn $opt(&mut self, name: &str) -> Result<Option<$t>, BindError> {
            match self.rec.take(name) {
                Value::Null => Ok(None),
                v => $f(v).map(Some).map_err(|w: String| self.bad(name, &w)),
            }
        }
    };
}

impl Inputs {
    pub(crate) fn new(rec: Record) -> Self {
        Self {
            rec,
            ctx: String::new(),
        }
    }

    fn bad(&self, name: &str, why: &str) -> BindError {
        BindError::invalid_argument(format!("{}{name} {why}", self.ctx))
    }

    read!(string, opt_string, String, coerce::to_string);
    read!(f64, opt_f64, f64, |v: Value| coerce::to_f64(&v));
    read!(usize, opt_usize, usize, |v: Value| coerce::to_usize(&v));
    read!(bool, opt_bool, bool, |v: Value| coerce::to_bool(&v));
    read!(seed, opt_seed, u64, |v: Value| coerce::to_seed(&v));
    read!(vec, opt_vec, Vec<f64>, coerce::to_vec);
    read!(vecs, opt_vecs, Vec<Vec<f64>>, coerce::to_vecs);
    read!(usizes, opt_usizes, Vec<usize>, coerce::to_usizes);
    read!(i64s, opt_i64s, Vec<i64>, coerce::to_i64s);
    read!(strs, opt_strs, Vec<String>, coerce::to_strs);
    read!(bools, opt_bools, Vec<bool>, coerce::to_bools);
    read!(
        weight_power,
        opt_weight_power,
        glmm::WeightPower,
        coerce::to_weight_power
    );
    read!(se_type, opt_se_type, glmm::SeType, coerce::to_se_type);

    /// A small count (`nagq`, `objective_nodes`).
    pub(crate) fn u8(&mut self, name: &str) -> Result<u8, BindError> {
        let n = self.usize(name)?;
        u8::try_from(n).map_err(|_| self.bad(name, &format!("{n} is out of range for u8")))
    }

    /// A `width`-column matrix as its rows.
    pub(crate) fn rows(&mut self, name: &str, width: usize) -> Result<Vec<Vec<f64>>, BindError> {
        let v = self.rec.take(name);
        coerce::to_rows(v, width).map_err(|w| self.bad(name, &w))
    }

    /// The raw value, `Null` when absent.
    pub(crate) fn value(&mut self, name: &str) -> Value {
        self.rec.take(name)
    }

    /// A nested record, its fields named `name.field` in errors. An empty
    /// list is an empty record (R's `list()` carries no names).
    pub(crate) fn opt_record(&mut self, name: &str) -> Result<Option<Inputs>, BindError> {
        match self.rec.take(name) {
            Value::Null => Ok(None),
            Value::List(x) if x.is_empty() => Ok(Some(Inputs {
                rec: Record::new(),
                ctx: format!("{}{name}.", self.ctx),
            })),
            Value::Record(rec) => Ok(Some(Inputs {
                rec,
                ctx: format!("{}{name}.", self.ctx),
            })),
            other => Err(self.bad(
                name,
                &format!(
                    "must be a record of named values (a dict / named list), got {}",
                    coerce::describe(&other)
                ),
            )),
        }
    }

    /// A nested record that must be present.
    pub(crate) fn record(&mut self, name: &str) -> Result<Inputs, BindError> {
        self.opt_record(name)?
            .ok_or_else(|| self.bad(name, "is missing"))
    }

    /// True when the record has a non-null `name`.
    pub(crate) fn has(&self, name: &str) -> bool {
        self.rec.get(name).is_some_and(|v| !v.is_null())
    }

    /// Numeric columns: a record of name to vector.
    pub(crate) fn numeric(&mut self, name: &str) -> Result<NumericColumns, BindError> {
        let Some(cols) = self.opt_record(name)? else {
            return Ok(HashMap::new());
        };
        cols.rec
            .into_fields()
            .into_iter()
            .map(|(k, v)| {
                let values = coerce::to_vec(v).map_err(|w| {
                    BindError::invalid_argument(format!("numeric column {k:?} {w}"))
                })?;
                Ok((k, values))
            })
            .collect()
    }

    /// Factor columns: a record of name to [`Value::Factor`].
    pub(crate) fn factors(&mut self, name: &str) -> Result<FactorColumns, BindError> {
        let Some(cols) = self.opt_record(name)? else {
            return Ok(HashMap::new());
        };
        cols.rec
            .into_fields()
            .into_iter()
            .map(|(k, v)| match v {
                Value::Factor { levels, codes } => {
                    let codes = codes
                        .into_iter()
                        .map(|c| {
                            u32::try_from(c).map_err(|_| {
                                BindError::invalid_argument(format!(
                                    "factor column {k:?}: negative code {c} (a missing value)"
                                ))
                            })
                        })
                        .collect::<Result<_, _>>()?;
                    Ok((k, (levels, codes)))
                }
                other => Err(BindError::invalid_argument(format!(
                    "factor column {k:?} must be a factor (levels and codes), got {}",
                    coerce::describe(&other)
                ))),
            })
            .collect()
    }
}
