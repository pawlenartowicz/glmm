//! The value tree every port marshals into and out of `glmm-bind`. Values are
//! owned: a port copies its columns in.

/// An ordered list of named values: a call's arguments, a result, or a
/// record passed back (a stored fit's fields).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Record {
    fields: Vec<(String, Value)>,
}

impl Record {
    /// An empty record.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a field. A repeated key is kept; [`crate::call`] refuses one in
    /// its arguments, and [`Record::get`] reads the first.
    pub fn push(&mut self, key: &str, value: Value) {
        self.fields.push((key.to_owned(), value));
    }

    /// Builder form of [`Record::push`], for records built in code.
    #[must_use]
    pub fn with(mut self, key: &str, value: Value) -> Self {
        self.push(key, value);
        self
    }

    /// Replace a field's value in place, or append it when absent.
    pub fn set(&mut self, key: &str, value: Value) {
        match self.fields.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = value,
            None => self.push(key, value),
        }
    }

    /// A field's value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Move a field's value out, leaving `Null`; `Null` when absent.
    pub fn take(&mut self, key: &str) -> Value {
        self.fields
            .iter_mut()
            .find(|(k, _)| k == key)
            .map_or(Value::Null, |(_, v)| std::mem::replace(v, Value::Null))
    }

    /// Field names, in order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.fields.iter().map(|(k, _)| k.as_str())
    }

    /// `(name, value)` pairs, in order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.fields.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// The fields, consuming the record.
    #[must_use]
    pub fn into_fields(self) -> Vec<(String, Value)> {
        self.fields
    }

    /// Number of fields.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// True when there are no fields.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }
}

/// One value crossing a port boundary.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `None` / `NULL`: an absent optional.
    Null,
    /// A boolean.
    Bool(bool),
    /// A signed integer (counts leave bind as this).
    I64(i64),
    /// An unsigned 64-bit integer (seeds leave bind as this).
    U64(u64),
    /// A float.
    F64(f64),
    /// A string.
    Str(String),
    /// A float vector.
    Vec(Vec<f64>),
    /// An integer vector that is not an index (counts, offsets, caller rows).
    IntVec(Vec<i64>),
    /// Positions, 0-based here: the R port writes them +1 and reads them -1,
    /// the Python port passes them through.
    Index(Vec<usize>),
    /// A string vector.
    StrVec(Vec<String>),
    /// A logical vector.
    BoolVec(Vec<bool>),
    /// A factor column: its levels in the caller's order (the first is the
    /// reference level) and each row's 0-based code, -1 for a missing value.
    Factor {
        /// The levels.
        levels: Vec<String>,
        /// Per row, the index into `levels`, or -1.
        codes: Vec<i32>,
    },
    /// A float matrix, column-major: element `(i, j)` is `data[i + j * nrow]`.
    Mat {
        /// Row count.
        nrow: usize,
        /// Column count.
        ncol: usize,
        /// The elements.
        data: Vec<f64>,
    },
    /// R's `NA` given as one value (a length-1 logical `NA`), which a
    /// refusal names as `NA`.
    Na,
    /// A named record.
    Record(Record),
    /// An ordered list.
    List(Vec<Value>),
}

impl Value {
    /// A short name of the variant, for error messages.
    #[must_use]
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "bool",
            Self::I64(_) | Self::U64(_) => "integer",
            Self::F64(_) => "number",
            Self::Str(_) => "string",
            Self::Vec(_) => "numeric vector",
            Self::IntVec(_) => "integer vector",
            Self::Index(_) => "index vector",
            Self::StrVec(_) => "string vector",
            Self::BoolVec(_) => "logical vector",
            Self::Factor { .. } => "factor",
            Self::Mat { .. } => "matrix",
            Self::Na => "missing value",
            Self::Record(_) => "record",
            Self::List(_) => "list",
        }
    }

    /// True for `Null`.
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}
