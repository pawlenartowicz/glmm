# Supported families and links

Every family below is wired end-to-end in `fit` for both fixed-only models
(`re: None` → OLS/GLM) and mixed models (`re: Some` → LMM/GLMM), including the
sparse over-envelope solver path, except `InverseGaussian`, which is
fixed-only and faults at the model-shape gate for `re: Some`. There are no
`unimplemented!` gaps in the family dispatch.

| Family | Links | Canonical | Fixed-only | Mixed | Validated against |
|---|---|---|---|---|---|
| `Gaussian` | identity (implicit) | yes | OLS | LMM (REML) | R `lm` / `lme4::lmer` |
| `Binomial` | `Logit`, `Probit`, `Cloglog` | `Logit` | logistic/probit/cloglog GLM | GLMM | R `glm(binomial)` / `lme4::glmer` (cloglog goldens `sim_cloglog_glm`, `sim_cloglog_glmm`) |
| `Poisson` | `Log` | yes | GLM | GLMM | R `glm(poisson)` / `lme4::glmer` (goldens `grouseticks_glm`, `grouseticks`) |
| `Gamma` | `Log`, `Inverse` | neither¹ | GLM | GLMM | R `glm(Gamma(link))` / `lme4::glmer(Gamma)` (goldens `sim_gamma_*`) |
| `NegativeBinomial` | `Log` | no² | GLM (θ estimated) | GLMM | R `MASS::glm.nb` / `lme4::glmer.nb` (goldens `sim_nb_*`) |
| `InverseGaussian` | `Log`, `InverseSquared` | neither³ | GLM | **not supported** (faults) | R `glm(inverse.gaussian(link))` (goldens `sim_igauss_glm`, `sim_igauss_inv_sq_glm`) |

¹ The Gamma canonical link is the *negative* inverse (`θ = −1/μ`); the offered
`Inverse` link (`g(μ) = 1/μ`) is therefore non-canonical and uses the general
Fisher-scoring branch. `Log` is the recommended default — `Inverse` can drive
`μ ≤ 0` mid-IRLS and is domain-clamped (`η > 0`).

² The NB canonical link `log(μ/(μ+θ))` is not offered; `Log` uses the general
Fisher-scoring branch.

³ The Inverse-Gaussian canonical link is `θ = −1/(2μ²)`; the offered
`InverseSquared` link (`g(μ) = 1/μ²`) only matches it up to sign and scale, so
both links use the general Fisher-scoring branch.

**Which algorithm fits your model?** The table's row and column pick the
path: fixed-only rows go through OLS or IRLS-based GLM (both in
[`algorithms.md`](algorithms.md), which also carries the full dispatch graph
and the tuning-knob index), Gaussian mixed models go through the profiled-REML
LMM machinery ([`algorithms-lmm.md`](algorithms-lmm.md)), and non-Gaussian
mixed models go through PIRLS with Laplace/AGQ
([`algorithms-glmm.md`](algorithms-glmm.md)).

## Details

- **Gaussian** has no link enum — identity is the only link, selected by
  `Family::Gaussian` alone.
- **Binomial** counts are fit as expanded 0/1 rows (the kernel is Bernoulli),
  or aggregated with `weights` (`y` = success proportion, `weights` = trial
  count — lme4's `cbind(s, m−s)` objective). `Logit` runs on the fused-SIMD
  canonical hot path; `Probit` (`μ = Φ(η)` via the high-precision `phi_hp`)
  and `Cloglog` (`μ = 1−exp(−exp(η))`, asymmetric — μ approaches 1 much faster
  than 0, hence an upper clamp on η at `ln(ETA_MAX)` the other two links don't
  need) both take the general Fisher-scoring branch.
- **Gamma** dispersion `φ` is estimated post-fit as the Pearson moment
  estimator `φ̂ = Σrᵢ²/(n−p)` and scales the SE by `√φ̂`; the estimate-vs-fixed
  directive lives in `FitOptions`, not in the family. In the GLMM objective the
  dispersion enters via lme4's profiled `Gamma()$aic` term.
- **NegativeBinomial** dispersion `θ` is estimated by an outer loop
  (`fit_glm_nb` / `fit_glmm_nb`) and threaded explicitly through the variance
  function `V(μ) = μ + μ²/θ`.
- **InverseGaussian** (`V(μ) = μ³`) is GLM-only: mixed models fault at the
  model-shape gate because the profiled `inverse.gaussian()$aic` objective
  term the GLMM would need is not built. Dispersion `φ` is estimated post-fit
  by the same Pearson moment estimator as Gamma, and the log-likelihood
  follows R's `inverse.gaussian()$aic` convention (dispersion profiled inside
  the term).
- Dispersion is fixed at `φ ≡ 1` for Binomial, Poisson, and NB (NB
  overdispersion lives in `θ`, not `φ`).
- **Prior (case) weights** (`FitOptions::weights`) are honored on every family
  above, fixed-only and mixed alike, at any `nagq`, including AGQ (`nagq > 1`)
  on the binomial/Poisson shapes it covers. See `FitOptions::weights`'s
  rustdoc for the exact per-path convention and oracle citations.
- **Offset** (`FitOptions::offset`) is honored on every family above,
  fixed-only and mixed alike, at any `nagq`. See `FitOptions::offset`'s
  rustdoc for the exact convention.
