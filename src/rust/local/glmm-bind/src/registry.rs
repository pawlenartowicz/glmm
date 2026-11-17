//! The function registry and the single entry point.

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};

use crate::error::BindError;
use crate::fns;
use crate::inputs::Inputs;
use crate::port::Port;
use crate::value::{Record, Value};

/// A parameter's default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DefaultValue {
    /// No default: the caller must pass it (non-null). An option the caller
    /// must give is declared [`DefaultValue::Null`] instead, so that its
    /// reader refuses a null or `NA` value in the option's own words.
    Required,
    /// `None` / `NULL`.
    Null,
    /// A bool literal.
    Bool(bool),
    /// A string literal.
    Str(&'static str),
}

impl DefaultValue {
    fn to_value(self) -> Value {
        match self {
            Self::Required | Self::Null => Value::Null,
            Self::Bool(b) => Value::Bool(b),
            Self::Str(s) => Value::Str(s.to_owned()),
        }
    }
}

/// One parameter of a registered function.
#[derive(Debug, Clone, Copy)]
pub struct Param {
    /// Name, the same in every port.
    pub name: &'static str,
    /// Default, or [`DefaultValue::Required`].
    pub default: DefaultValue,
}

type RunFn = fn(&mut Inputs) -> Result<Value, BindError>;

/// One registered function.
pub struct FnSpec {
    /// Name, the same in every port.
    pub name: &'static str,
    /// Parameters in signature order.
    pub params: &'static [Param],
    run: RunFn,
}

const fn req(name: &'static str) -> Param {
    Param {
        name,
        default: DefaultValue::Required,
    }
}

const fn opt(name: &'static str) -> Param {
    Param {
        name,
        default: DefaultValue::Null,
    }
}

const fn def(name: &'static str, default: DefaultValue) -> Param {
    Param { name, default }
}

/// The training data of a stored fit, as `fit` took it.
macro_rules! model_data {
    ($($rest:expr),* $(,)?) => {
        &[
            req("formula"), req("numeric"), req("factors"), req("family"), req("link"),
            opt("weights"), opt("offset"), $($rest),*
        ]
    };
}

static REGISTRY: &[FnSpec] = &[
    FnSpec {
        name: "fit",
        params: &[
            req("formula"),
            req("numeric"),
            req("factors"),
            opt("family"),
            opt("link"),
            opt("wald_se"),
            opt("nagq"),
            opt("dispersion"),
            opt("init_theta"),
            opt("weights"),
            opt("offset"),
            opt("warm_start"),
            opt("satterthwaite"),
            opt("condvar"),
            opt("reml"),
            opt("anova_test"),
            opt("weight_power"),
            opt("se_type"),
            opt("row_index"),
            req("row_base"),
        ],
        run: fns::fit::fit,
    },
    FnSpec {
        name: "compare",
        params: &[
            req("f0"),
            req("f1"),
            req("numeric"),
            req("factors"),
            opt("family"),
            opt("link"),
            opt("nagq"),
            opt("weights"),
            opt("offset"),
            opt("nsim"),
            opt("seed"),
            opt("weight_power"),
            opt("se_type"),
            opt("row_index"),
            req("row_base"),
        ],
        run: fns::compare::compare,
    },
    FnSpec {
        name: "compare_fits",
        params: &[req("m0"), req("m1"), opt("nsim"), opt("seed")],
        run: fns::compare::compare_fits,
    },
    FnSpec {
        name: "confint",
        params: &[
            req("beta"),
            req("se"),
            req("coef_df"),
            req("df_method"),
            req("level"),
        ],
        run: fns::inference::confint,
    },
    FnSpec {
        name: "contrast_test",
        params: &[
            req("df_method"),
            req("beta"),
            req("vcov"),
            opt("model_vcov"),
            req("aliased"),
            req("nobs"),
            req("dispersion"),
            opt("record"),
            opt("robust"),
            req("names"),
            req("l"),
            req("rows"),
        ],
        run: fns::inference::contrast_test,
    },
    FnSpec {
        name: "pairwise",
        params: model_data![
            req("df_method"),
            req("beta"),
            req("vcov"),
            opt("model_vcov"),
            req("aliased"),
            req("nobs"),
            opt("record"),
            opt("robust"),
            req("names"),
            opt("factor"),
            opt("by"),
            opt("at"),
            opt("adjust"),
            req("converged"),
        ],
        run: fns::inference::pairwise,
    },
    FnSpec {
        name: "pairwise_confint",
        params: &[req("table"), req("level")],
        run: fns::inference::pairwise_confint,
    },
    FnSpec {
        name: "term_labels",
        params: &[req("formula")],
        run: fns::formula::term_labels,
    },
    FnSpec {
        name: "same_fixed_part",
        params: &[req("f0"), req("f1")],
        run: fns::formula::same_fixed_part,
    },
    FnSpec {
        name: "term_test_refusal",
        params: &[req("test"), req("den_df_kind")],
        run: fns::inference::term_test_refusal,
    },
    FnSpec {
        name: "predict",
        params: model_data![
            req("parts"),
            opt("new_numeric"),
            opt("new_factors"),
            req("n_rows"),
            opt("scale"),
            opt("re_form"),
            opt("allow_new_levels"),
            opt("se"),
            req("level"),
            opt("future"),
            opt("row_index"),
            req("row_base"),
        ],
        run: fns::predict::predict,
    },
    FnSpec {
        name: "predict_columns",
        params: &[req("formula"), req("available")],
        run: fns::predict::predict_columns,
    },
    FnSpec {
        name: "simulate",
        params: model_data![
            req("parts"),
            opt("nsim"),
            opt("seed"),
            opt("row_index"),
            req("row_base"),
        ],
        run: fns::predict::simulate,
    },
    FnSpec {
        name: "effect_sizes",
        params: model_data![
            req("parts"),
            req("aliased"),
            req("converged"),
            req("objective"),
            req("objective_nodes"),
            req("null_refit"),
        ],
        run: fns::inference::effect_sizes,
    },
    FnSpec {
        name: "profile_intervals",
        params: model_data![
            req("reml"),
            opt("weight_power"),
            opt("weight_power_estimated"),
            req("level"),
            opt("parm")
        ],
        run: fns::inference::profile_intervals,
    },
    FnSpec {
        name: "formula_columns",
        params: &[req("formula"), req("available")],
        run: fns::formula::formula_columns,
    },
    FnSpec {
        name: "model_matrix_columns",
        params: &[req("formula"), req("available")],
        run: fns::formula::model_matrix_columns,
    },
    FnSpec {
        name: "model_matrix",
        params: &[
            req("formula"),
            req("numeric"),
            req("factors"),
            req("n_rows"),
            opt("row_index"),
            req("row_base"),
        ],
        run: fns::formula::model_matrix,
    },
    FnSpec {
        name: "plain_column",
        params: &[req("elements")],
        run: fns::formula::plain_column,
    },
    FnSpec {
        name: "pearson_residuals",
        params: &[
            req("family"),
            req("y"),
            req("fitted"),
            opt("weights"),
            req("dispersion"),
            opt("weight_power"),
        ],
        run: crate::summary::pearson_residuals,
    },
    FnSpec {
        name: "deviance_residual_sum",
        params: &[
            req("family"),
            req("y"),
            req("fitted"),
            opt("weights"),
            opt("weight_power"),
            req("dispersion"),
        ],
        run: crate::summary::deviance_residual_sum,
    },
    FnSpec {
        name: "warning_kind",
        params: &[req("kind")],
        run: crate::notes::warning_kind,
    },
    FnSpec {
        name: "note_kinds",
        params: &[],
        run: crate::notes::note_kinds,
    },
    FnSpec {
        name: "summary",
        params: &[
            req("fit"),
            opt("effect_sizes"),
            def("need_refit", DefaultValue::Bool(false)),
        ],
        run: crate::summary::summary,
    },
];

/// Every registered function.
#[must_use]
pub fn registry() -> &'static [FnSpec] {
    REGISTRY
}

/// Run a registered function. A repeated or undeclared key and a missing
/// (or null) required argument raise `invalid_argument`, an absent or null
/// optional argument takes its default, and a panic anywhere below comes
/// back as an `internal` error.
pub fn call(port: Port, name: &str, inputs: Record) -> Result<Value, BindError> {
    let spec = REGISTRY
        .iter()
        .find(|s| s.name == name)
        .ok_or_else(|| BindError::invalid_argument(format!("unknown function '{name}'")))?;
    call_spec(port, spec, inputs)
}

fn call_spec(port: Port, spec: &FnSpec, mut inputs: Record) -> Result<Value, BindError> {
    let keys: Vec<&str> = inputs.keys().collect();
    for (i, key) in keys.iter().enumerate() {
        if !spec.params.iter().any(|p| p.name == *key) {
            return Err(BindError::invalid_argument(format!(
                "{}() got an unexpected argument '{key}'",
                spec.name
            )));
        }
        if keys[..i].contains(key) {
            return Err(BindError::invalid_argument(format!(
                "{}() got the argument '{key}' twice",
                spec.name
            )));
        }
    }
    for p in spec.params {
        let present = inputs.get(p.name).is_some_and(|v| !v.is_null());
        if !present {
            if p.default == DefaultValue::Required {
                return Err(BindError::invalid_argument(format!(
                    "{}() missing required argument '{}'",
                    spec.name, p.name
                )));
            }
            inputs.set(p.name, p.default.to_value());
        }
    }
    let mut inp = Inputs::new(inputs, port);
    match panic::catch_unwind(AssertUnwindSafe(|| (spec.run)(&mut inp))) {
        Ok(r) => r,
        Err(payload) => Err(BindError::internal(format!(
            "panic in {}: {}",
            spec.name,
            panic_message(payload.as_ref())
        ))),
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boom(_: &mut Inputs) -> Result<Value, BindError> {
        panic!("kaboom")
    }

    #[test]
    fn a_panic_becomes_an_internal_error() {
        let spec = FnSpec {
            name: "boom",
            params: &[],
            run: boom,
        };
        let e = call_spec(Port::Python, &spec, Record::new()).unwrap_err();
        assert_eq!(e.code, "internal");
        assert!(e.message.contains("kaboom"), "{}", e.message);
    }
}
