# Campaigns — archived validation studies

Two finished studies. Their **conclusions** are frozen in each campaign's
committed `reports/`; their raw run output is gitignored (regenerable). The
scripts stay maintained so each can be rerun after major solver work.

## speed-campaign — optimizer cost across the grid

**Question:** where does glmm's BOBYQA spend evaluations, and how does wall
time compare to MixedModels.jl (and lme4 where feasible) across structure ×
family × size × balance?
**Verdict** (`reports/final_analysis.txt`): eval-ratio medians rise with θ
dimension (~0.97 at 1–2 θ params to ~1.8 at 21–40); 491 of the 510 cells the
frozen run covered are ok, every mismatch adjudicated. `manifest.json` has since
grown to 585 cells; the verdict is the 510-cell run's, and the extra cells have
not been fitted through it.
**Machinery:** `manifest.json` (585 cells) + `prep.R` (fixed-seed data into `data/`) +
`run.sh <engine> <tag>` (per-cell wall watchdog with kill-and-resume) +
`fit.{rs,R,jl}` drivers + `analyze.R`/`report.R`/`counters.R` → `reports/`.
`common.R` holds the readers and the beta band the two analysis scripts share.
`theta_eval.rs` and `sweep_fit.jl` are the mismatch-adjudication drivers
(multi-start sweeps re-scored on glmm's objective); their product is the
best-known optima frozen in `optima/`.
**Second cell list:** `estimate-grid-manifest.json` is the answer-agreement pass over
the same grid — 477 Laplace cells against glmer/lmer, 15 scalar-AGQ cells against
glmer(nAGQ=7), 18 vector-AGQ cells against GLMMadaptive(nAGQ=7). Its verdict is in
`reports/`: 22 named failures of 510, every one adjudicated — boundary near-zero-θ
artifacts, GLMMadaptive under-convergence, and one real disagreement against
MixedModels (`lmm_q8_g3000p5_bal_lowsnr`, REML gap +2.03) kept on record. `fit.R`
selects its three engine branches off that manifest's per-cell `nagq`/`structure`.
**Rerun:** `./run.sh glmm <tag>` etc. after `Rscript prep.R`; user locks the
clock first (`bench-l`) for meaningful walls.

## monte_carlo — accuracy against known truth

**Question:** with data *generated from known parameters*, how close does each
engine (glmm, lme4, GLMMadaptive) land to the truth — RMSE, not cross-engine
agreement? Mirrors Li & Signorelli (2026), *A Comparison of R Packages for
Estimating GLMMs*, arXiv:2606.15933v1 — same DGP, cell grid, and published
bias/RMSE baselines (`manifest.json`'s `paper`/`study` fields, `external_repo`
github.com/xanalee/glmmPackCompare).
**Verdict** (`reports/final_analysis.txt`, `reports/glmm_vs_engines_rmse.png`):
see the frozen summaries; convergence is scored by **each engine's own native
flag** (lme4's untouched), boundary fits reported as their own column split by
where each engine files them — this scoring policy is deliberate, keep it.
**Machinery:** `gen.sh` simulates from `manifest.json` truths into `data/`;
`batch.sh` drives `fit_{glmm,lme4,glmmadaptive}.R`; `summarize_accuracy_truth.R`
and `plot_comparison.R` → `reports/`. `external/`, `data/`, `results/` are
gitignored (local, regenerable); `truth/` is committed.
**Rerun:** `./gen.sh && ./batch.sh`, then the summarize/plot scripts.
