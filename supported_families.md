# Supported families and links

Every family below is wired end-to-end in `fit` for both fixed-only models
(`re: None` → OLS/GLM) and mixed models (`re: Some` → LMM/GLMM), including the
sparse over-envelope solver path. There are no `unimplemented!` gaps in the
family dispatch.

| Family | Links | Canonical | Fixed-only | Mixed | Validated against |
|---|---|---|---|---|---|
| `Gaussian` | identity (implicit) | yes | OLS | LMM (REML/MLE) | R `lm` / `lme4::lmer` |
| `Binomial` | `Logit`, `Probit` | `Logit` | logistic/probit GLM | GLMM | R `glm(binomial)` / `lme4::glmer` |
| `Poisson` | `Log` | yes | GLM | GLMM | R `glm(poisson)` / `lme4::glmer` (goldens `grouseticks_glm`, `grouseticks`) |
| `Gamma` | `Log`, `Inverse` | neither¹ | GLM | GLMM | R `glm(Gamma(link))` / `lme4::glmer(Gamma)` (goldens `sim_gamma_*`) |
| `NegativeBinomial` | `Log` | no² | GLM (θ estimated) | GLMM | R `MASS::glm.nb` / `lme4::glmer.nb` (goldens `sim_nb_*`) |

¹ The Gamma canonical link is the *negative* inverse (`θ = −1/μ`); the offered
`Inverse` link (`g(μ) = 1/μ`) is therefore non-canonical and uses the general
Fisher-scoring branch. `Log` is the recommended default — `Inverse` can drive
`μ ≤ 0` mid-IRLS and is domain-clamped (`η > 0`).

² The NB canonical link `log(μ/(μ+θ))` is not offered; `Log` uses the general
Fisher-scoring branch.

## Details

- **Gaussian** has no link enum — identity is the only link, selected by
  `Family::Gaussian` alone.
- **Binomial** counts are fit as expanded 0/1 rows (the kernel is Bernoulli).
  `Logit` runs on the fused-SIMD canonical hot path; `Probit` (`μ = Φ(η)` via
  the high-precision `phi_hp`) takes the general Fisher-scoring branch.
- **Gamma** dispersion `φ` is estimated post-fit as the Pearson moment
  estimator `φ̂ = Σrᵢ²/(n−p)` and scales the SE by `√φ̂`; the estimate-vs-fixed
  directive lives in `FitOptions`, not in the family. In the GLMM objective the
  dispersion enters via lme4's profiled `Gamma()$aic` term.
- **NegativeBinomial** dispersion `θ` is estimated by an outer loop
  (`fit_glm_nb` / `fit_glmm_nb`) and threaded explicitly through the variance
  function `V(μ) = μ + μ²/θ`.
- Dispersion is fixed at `φ ≡ 1` for Binomial, Poisson, and NB (NB
  overdispersion lives in `θ`, not `φ`).
