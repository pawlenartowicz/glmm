//! What a test's statistic is referred to, and what each fit maximised.

/// The distribution a test statistic is referred to for its p-value.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reference {
    /// χ² on `df` degrees of freedom.
    ChiSquared {
        /// The degrees of freedom.
        df: usize,
    },
    /// The 50:50 mixture of χ²_{df_low} and χ²_{df_low + 1}, the reference
    /// of a likelihood-ratio test whose larger model adds one random-effect
    /// dimension. At `df_low = 0` its lower component, χ²₀, is a point mass
    /// at zero.
    ChiBarSquared {
        /// The df of the lower component.
        df_low: usize,
    },
    /// The exact null distribution of the restricted likelihood-ratio test,
    /// simulated from `seed`.
    ExactRlrt {
        /// The number of draws made: 0 when the statistic is 0, which needs
        /// none (p = 1), and 0 when it is NaN (p NaN).
        nsim: usize,
        /// The seed of the simulation.
        seed: u64,
    },
    /// F on `(num_df, den_df)` degrees of freedom.
    F {
        /// The numerator df.
        num_df: f64,
        /// The denominator df.
        den_df: f64,
        /// Where the denominator df comes from.
        den_df_kind: DenDf,
    },
}

/// Where the denominator df of an F reference comes from.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenDf {
    /// Satterthwaite's approximation (a Gaussian linear mixed model).
    Satterthwaite,
    /// The residual df `n − p` (OLS, and the Gamma and inverse Gaussian GLMs
    /// with φ estimated).
    Residual,
}

/// The objective a fit maximised, which a likelihood-ratio test compares.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    /// Least squares (OLS).
    LeastSquares,
    /// The likelihood (GLM, and a Gaussian linear mixed model fitted by ML).
    MaximumLikelihood,
    /// The restricted likelihood (a Gaussian linear mixed model fitted by
    /// REML).
    Reml,
    /// The Laplace approximation to a GLMM's likelihood.
    Laplace,
    /// Adaptive Gauss-Hermite quadrature on `nodes` nodes, the GLMM
    /// likelihood.
    Agq {
        /// The number of quadrature nodes.
        nodes: u8,
    },
}
