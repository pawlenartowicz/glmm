TOL <- list(
  # beta_rel, stddev_rel, se_rel and se_hessian_rel are the corpus-wide cross-engine
  # bands, set before this grid. MEASURED (2026-10-27) on the full grid, the kept
  # 0.7.0 glmm run against the nearest oracle at glmm's optimum (gate 2), Laplace
  # cells: beta worst 5.81e-04 (binb_q4sx2_g3000p20_bal_base), stddev 6.47e-04
  # (binb_cross6_g300p5_bal_base), se_rx 1.71e-04 (sim_collinear_lmm_ml),
  # se_hessian 7.94e-04 (binb_cross4_g3000p20_bal_rare); the correlation, read
  # against near_zero_abs, 9.17e-04 absolute (binb_q8_g3000p5_bal_base). The house
  # rule would give 2e-03 for beta, stddev and se_hessian. They stay at 1e-3:
  # se_rel and beta_rel also set the inference bands below (df_e2e_rel, confint_rel,
  # wald_rel, sum_sq_rel, eta2_rel, r2_icc_rel), each measured far inside them, and
  # tests/oracle_support/mod.rs mirrors all four.
  beta_rel        = 1e-3,   # fixed effects: relative.
  stddev_rel      = 1e-3,   # varcomp std-devs: relative.
  loglik_abs_lmm  = 2e-6,   # LMM REML criterion: near-exact across engines (~1e-9 typical).
                            #   Measured worst 1.23e-6 on sim_max_q_slope (q=8, 36 theta params,
                            #   the corpus's largest LMM covariance; ~6e-10 relative — MixedModels
                            #   sits 5e-7 from lme4 on the same rung), so 2e-6 = measured worst +
                            #   margin, same convention as se_hessian_rel below.
                            #   Nothing here reads it yet; the slot exists so the number has one
                            #   place to live.
  loglik_abs_glmm = 1e-3,   # GLMM Laplace logLik: two optimizers land ~3e-6 relative
                            #   apart on the same surface (beta/varcomp confirm same fit).
                            #   Nothing here reads it yet; the slot exists so the number has one
                            #   place to live.
  se_rel          = 1e-3,   # LMM SE + method-matched GLMM RX: tight (same method, all engines)
  se_hessian_rel  = 1e-3,   # GLMM Hessian pair (lme4 vs glmm), same band as se_rel. lme4.R pins
                            #   tolPwrss=1e-13, so its ldL2 update carries no lag, and glmm's FD
                            #   runs PIRLS at its tight FD-only tol; measured worst cross-engine
                            #   agreement 2e-5 (grouseticks, 2026-07-04), so 1e-3 = measured
                            #   worst + the same margin se_rel carries over ITS measured worst.

  # glmm (Rust) vs the glmm Python port -- a ROUND-OFF band, not an agreement band.
  # The port drives the same kernel through PyO3 (with no start it is the same `glmm::fit` call),
  # with the same lowering and a deterministic optimizer, so every gated quantity is
  # bit-identical and only the JSON round-trip could perturb it (it does not: Rust
  # and Python both emit shortest-round-trip f64, jsonlite parses back exact). Any
  # nonzero value here means the port fed the kernel something different -- a wiring
  # bug -- so this is the one band that is diagnostic at 0 and is NEVER widened.
  # Measured worst across the 26-rung corpus at freeze (2026-07-16): exactly 0.
  port_rel        = 1e-12,

  # Absolute floor below which `rel_max` stops asking a RELATIVE question. Same
  # defect `agq_corr_abs` below addresses -- a relative difference has no
  # meaning once both sides are at zero -- but reached from the other direction:
  # there the whole quantity lives near zero, here a single coordinate does.
  #
  # The case: glmm PINS a variance component to a hard 0.0 while an oracle's
  # optimizer stops on its own residue (1e-5-ish). Both engines say "this
  # component is zero"; `rel_max`'s 1e-12 denominator floor scores it exactly
  # 1.0, and no residue is small enough to escape -- shrinking the oracle's
  # residue shrinks the denominator with it. Two fixtures were already built
  # AROUND this rather than through it: sim_binomial_zerosd's seed is frozen on
  # a draw where BOTH engines return bit-exact 0.0 (tools/prep/gen_large_theta_data.R),
  # and fit_glmm_binomial_zerosd_is_pinned asserts bit-equality instead of a
  # band (src/fit/glmm_tests.rs). Both stay as they are -- they assert something
  # stronger than this floor and lose nothing by it.
  #
  # MEASURED (2026-08-06) over the answer-agreement campaign's 510 cells, every
  # stddev coordinate where both engines sit under 1e-2: 16 coordinates spanning
  # 1.06e-5 .. 9.06e-3, with the pairwise DIFFERENCE at 4.92e-4 or below on ten
  # of them and 1.90e-3 .. 9.06e-3 on the rest. 1e-3 = ceil-to-one-significant-
  # figure(2 x 4.92e-4), the house rule every measured band above follows. It is
  # deliberately the CONSERVATIVE cut of that spread: the four coordinates
  # between 1.9e-3 and 9.1e-3 keep failing, because nothing yet says whether
  # they are residue or a real disagreement. The largest genuine disagreement in
  # that campaign (lmm_q4sx2_g300p20_bal_lowsnr: glmm 0.62484 against lme4's
  # exact 0.0) clears this floor by 600x.
  #
  # Gates on the COORDINATE'S OWN MAGNITUDE (the larger of the two sides), never
  # on the difference. That distinction is the whole constant: a floor on the
  # difference reads "1e-3 apart is close enough", which on an O(1) coefficient
  # IS beta_rel -- measured 2026-08-06, that version drove every cross-engine
  # comparison in the corpus to exactly 0 and left the gate unable to fail. A
  # floor on the magnitude reads "this coordinate is zero in both engines", which
  # touches nothing that carries signal. Exempting a coordinate can only turn a
  # fail into a pass, so raising this is relaxing every band at once.
  near_zero_abs   = 1e-3,

  # sd_boundary_abs  the near-zero floor of a stddev comparison between two fits at
  # the same optimum (aligned deviances within dev_eps; runs.R::sd_floor): an SD
  # below it on both sides scores 0. The deviance is flat in a near-zero SD, so
  # equal deviances with different tiny SDs say the data do not determine them.
  # near_zero_abs above stays the floor of every other comparison, and of a stddev
  # comparison whose fits sit at different optima.
  # Rule: ceil1(2 x the largest such residue). MEASURED (2026-10-27) on the kept
  # 0.7.0 run, every SD glmm pins to exactly 0 where glmmTMB, at glmm's optimum,
  # leaves a residue: 3.72e-03 on gami_q2d_g3000p20_bal_base (glmm's deviance
  # 2.0e-05 the lower), then 2.15e-03 (gami_nest2), 2.03e-03 (gami_nest3) and
  # 1.06e-03 (gami_cross6), all g3000 Gamma-identity cells; ceil1(7.44e-03) = 8e-03.
  sd_boundary_abs = 8e-03,

  # Vector-RE AGQ rungs vs GLMMadaptive (goldens/sim_*_slope*_agq_k*.json; the
  # in-crate gates in src/fit/glmm_tests.rs use these numbers -- change
  # together). Calibrated empirically at freeze (2026-07-13) against goldens
  # frozen with TIGHTENED mixed_model controls (see goldens_agq.R -- the
  # GLMMadaptive DEFAULTS under-converge by ~4e-3 logLik on the low-info rungs,
  # the same artifact class as lme4's default tolPwrss). Measured matched-k
  # worsts: k=11 agrees to <= 7.1e-4 on EVERY quantity; every k=7 worst below
  # comes from the sparsest rung (sim_poisson_slope1, 4 obs/cluster) where each
  # engine still carries its own quadrature truncation error -- GLMMadaptive's
  # own k=7 se[0] sits 1.2e-2 from its own k=25 limit while glmm's k=7 is
  # within 1.1e-3 of that limit, so the matched-k=7 gap is oracle-side
  # k-truncation, not implementation disagreement. Bands = measured worst +
  # the usual ~2x margin.
  agq_beta_rel       = 3e-3,  # worst 1.4e-3 (poisson k=7); binomial rungs <= 6.7e-5
  agq_stddev_rel     = 4e-3,  # worst 1.7e-3 (poisson k=7)
  agq_corr_abs       = 4e-3,  # ABSOLUTE (correlations near 0 break relative); worst 1.6e-3
  agq_se_hessian_rel = 2e-2,  # worst 1.3e-2 (poisson k=7 intercept, the MA k-truncation
                              #   case above); next-worst 2.6e-3, k=11 worst 7.1e-4

  # ── constants measured on the 33-cell pilot, 2026-09-22 ─────────────────────
  # `tol_for` STOPS on an NA rather than gating against it: a band of NA
  # compares false and would turn every gate that reads it into a silent pass,
  # which is the one failure mode a tolerance table must not have. Each number
  # below carries its measurement, its rule and its date -- the same house style
  # as every band above. All three were taken on an UNLOCKED machine, which does
  # not matter: none of them is a timing.
  #
  # dev_eps  ceil1(10 x max benign |Delta dev| over the pilot), benign =
  #          beta/se/stddev all in-band, the same rule the 48-rung corpus's own
  #          floor measurement used. Measured 2026-09-22: 3.8009e-06 on
  #          lmm_q4sx2_g30000p5_bal_base (the 30000-row, 12-theta LMM, the
  #          corpus's heaviest covariance) over the 27 benign converged pilot
  #          cells; per family gaussian 3.8e-06, negativebinomial 1.7e-06,
  #          binomial 8.4e-07, inversegaussian 3.1e-08, poisson 7.5e-09, gamma
  #          4.5e-12 -- no family stands an order of magnitude above the rest,
  #          so ONE value, as the old corpus also used.
  #
  #          MEASURED AGAINST lme4, not against the best oracle, because that is
  #          what the rule measures: glmm's own round-off floor. The same pilot
  #          read against the best oracle gives 10.9197 (gaml_glm_g3000), but
  #          that number is a disagreement BETWEEN ORACLES -- glmm matches
  #          stats::glm there exactly while glmmTMB sits 5.46 logLik away -- and
  #          a floor set from it would be 200 deviance units wide and would
  #          swallow every real deviance loss the gate exists to report.
  dev_eps     = 4e-05,
  # dev_eps_agq_vec  glmm's OWN deviance on a vector-AGQ cell (nagq > 1 with more
  #          than one random effect per group; every such grid cell has a centered
  #          continuous slope), against a reference whose quadrature grid sits
  #          elsewhere. A product Gauss-Hermite grid is not rotation-invariant
  #          (dev_align.R, ga_rule_dev) and glmm places it in the coordinates of
  #          the standardized design, so a change in how the design is
  #          standardized moves this deviance by the quadrature difference at the
  #          same optimum. It is the band for comparing a glmm run cell by cell
  #          with an earlier glmm run on these cells; every other cell keeps
  #          dev_eps there. Gate 1 does not read it: against GLMMadaptive it scores
  #          glmm's point under GLMMadaptive's own rule, and on the cells where
  #          that rule evaluates, glmm sits 2.0e-07 to 7.1e-07 below GLMMadaptive
  #          on both kept runs named below, inside dev_eps.
  #          Rule: ceil1(10 x max benign |Delta dev|) over the 8 vector-AGQ
  #          cells, benign = beta/se/stddev in band against GLMMadaptive.
  #          Measured 2026-10-01 on the P45 glmm run of that day (which
  #          centered the slopes) against the kept glmm run of 2026-09-30
  #          (committed in 27318ef):
  #          4.307e-05 on binb_q2s_g3000p20_bal_base_agq7 (beta 1.3e-05,
  #          se_hessian 2.1e-04, stddev 5.3e-05 from GLMMadaptive); the other
  #          seven gaml_q2s_g60p5 +8.0e-06, nb_q2s_g3000p20 -3.2e-06,
  #          pois_q2s_g3000p20 +3.5e-07, the rest below 6e-08. The P45 follow-up
  #          kept run of 2026-10-01 (committed in b8d2efd, same standardization)
  #          against the P45 run: at most 6.3e-10 on these cells.
  dev_eps_agq_vec = 5e-04,
  # dev_big  a convention-mismatch constant, not a fit difference. It must sit
  #          far above the benign |Delta dev| spread and far below the smallest
  #          real convention constant. Both edges measured 2026-09-22: the
  #          benign spread tops out at 3.8e-06 (above), and the nAGQ saturated
  #          deficit -- a real convention constant -- is 5729.45 deviance units
  #          on pois_int1_g3000p20_bal_base_agq7, where lme4's reported logLik
  #          plus the saturated term lands 5.2e-08 from GLMMadaptive's. The two
  #          AGQ cells the pilot itself carries are Bernoulli, where that term is
  #          identically zero and measures nothing, which is why the deficit was
  #          taken on a Poisson AGQ cell. 0.5 stays clear of both ends, the
  #          argument the 48-rung corpus also settled on.
  dev_big     = 0.5,
  # truth_floor  below this |truth| the truth-gate error is ABSOLUTE, not
  #          relative. Sized from the deliberate near-zero coordinates the grid
  #          generates: the `nearzero` regime sets a true RE sd to 0.02 and
  #          `boundary` sets one to exactly 0.
  #          Rule: ceil1(2 x max |truth| over those coordinates). Measured
  #          2026-09-22 over the pilot: 2 such coordinates, spanning 0 .. 0.02,
  #          one of them exactly zero, so ceil1(2 x 0.02) = 0.04.
  #          NOTE this is NOT near_zero_abs: that is a round-off floor for the
  #          cross-engine gate and is far too small for a nearzero cell.
  truth_floor = 0.04,
  # ci_ref_rel  the band the saved-reference check gates at: tests/grid_reference.rs
  #          (the `fast` cells against the newest committed glmm run). Two homes
  #          hold this number: this entry and CI_REF_REL in
  #          tests/grid_reference.rs. Change both together.
  #
  #          It has to cover two different gaps at once.
  #          (a) A DIFFERENT CPU. This is the crate's own cross-architecture pin
  #              band for the iterative fit paths: PIN_REL_ITER in
  #              src/fit/common_tests.rs, 1e-7. Every Rust-vs-Rust pin is
  #              bit-exact on the anchor machine (x86_64-unknown-linux-gnu,
  #              Intel Core Ultra 7 265H, AVX2 + FMA, no AVX-512); 1e-7 is the
  #              margin that machine's account sizes for another one.
  #          (b) A DIFFERENT BUILD PROFILE. The committed run is fitted by the
  #              release build (lto = "thin", one codegen unit); tests/grid_reference.rs
  #              fits under the test profile ([profile.test] opt-level = 3, no
  #              LTO, Cargo's default codegen-unit count), so multiply-adds
  #              contract at other points. Measured 2026-09-24 on the anchor machine, by
  #              tests/grid_reference.rs itself: release-built fit_cell against
  #              test-built fit_cell, same function, all 58 fast cells, worst
  #              relative gap 1.88e-16 on clla_int1_g3000p20_bal_base.
  #
  #          STILL PENDING: the measurement on CI's own runner pool, which
  #          .github/workflows/pin-bands.yml collects and which only a push plus
  #          a manual dispatch can start. Until that runs, 1e-7 is inherited, not
  #          measured here. If a CI reference check goes red on this band, the
  #          band is what is wrong -- record the deviation the run prints and
  #          re-measure. Never widen it to turn a run green.
  #
  #          aarch64-apple-darwin, measured 2026-10-16 against the 0.6.0 release
  #          run: deviance and loglik stay inside this band on every fast cell,
  #          but the parameters of 10 of the 116 records (four-way crossed and
  #          nested fits, the NB theta search, vector AGQ) drift 1.5e-7 to
  #          3.8e-6. Those cells carry their own measured band in
  #          CROSS_CPU_CELL_BANDS in tests/grid_reference.rs; this number stays.
  ci_ref_rel  = 1e-7,
  # coef_df_rel  compare.R gate 5: glmm's per-coefficient Satterthwaite df against
  #          lmerTest's on the Gaussian LMM cells, by rel_max_by_coef. The df moves
  #          with theta-hat as the SE does, and the two engines agree on theta-hat
  #          only to their optimizers' tolerance, so this is an agreement band
  #          between two fits, not a round-off band on one.
  #          Rule: ceil1(2 x max rel_max_by_coef) over the cells gate 5 compares.
  #          Measured 2026-10-12 (dell-fedora) on the 19 fast Gaussian LMM cells,
  #          every one compared (no failed-pass, no no-ref), a --fast glmm run of the
  #          0.6.0 tree with coef_df (scratch, not kept; any --fast run of that
  #          tree re-measures it) against the first full lmertest run (lmerTest
  #          3.2.1, ra-ml-twins): 3.30e-04 on lmm_cross4_g3000p20_bal_base,
  #          (Intercept) (glmm 87.934, lmerTest 87.905); next Machines 2.96e-04
  #          (MachineC), sim_max_q_slope 1.43e-04, Penicillin 9.2e-05,
  #          lmm_q8_g3000p20_bal_base 6.9e-05, the other 14 cells at or below
  #          1.6e-05. On the seven largest, lmerTest's df evaluated at glmm's own
  #          theta-hat agrees with glmm's to 5.1e-07 or better (Penicillin; the
  #          others at or below 3.2e-07), so the spread is the two theta-hats',
  #          not the df computation. ceil1(2 x 3.30e-04) = 7e-04.
  #          RE-MEASURED (2026-10-27, dell-fedora) on the full grid: the kept 0.7.0
  #          glmm run against the lmerTest 3.2.1 run (sim_dynrange_lmm and its twin
  #          refitted after the engines' weights fix), 257 of the 263 Gaussian mixed
  #          cells (132 REML, 131 ML twins) compared at glmm's optimum; the other six
  #          are lmerTest failures or lmerTest fits above glmm's optimum, listed as
  #          no-ref. Worst 7.50e-04 on lmm_q4sx2_g30000p20_bal_base_ml, (Intercept)
  #          (lme4's deviance 1.05e-05 above glmm's); next 5.92e-04
  #          lmm_q4sx2_g30000p5_bal_base, 5.73e-04 lmm_q4sx2_g3000p20_bal_nearzero_ml,
  #          4.98e-04 lmm_nestmix_g3000p20_bal_base. ceil1(2 x 7.50e-04) = 2e-03, and
  #          ceil1(2 x 5.92e-04) = 2e-03 without the first cell.
  coef_df_rel = 2e-03
)

# ── truth-error band, per family ─────────────────────────────────────────────
# glmm fails a family when err_glmm > err_best + band. The band is TWICE THE
# STANDARD ERROR OF THE MEAN of the paired per-cell difference
# err_glmm(cell) - err_best(cell), both engines on the same data. The pairing is
# what makes it a band about the glmm-versus-oracle gap rather than about the
# spread between a 60-row and a 30000-row cell.
#
# RE-MEASURED (2026-10-27) on the full grid (compare.R --dev-floor, kept 0.7.0 run),
# not re-pinned. The largest best-oracle 2*SEM per family, over groups with n >= 2:
# binomial 1.70e-04 (agq / beta, n = 13), gamma 2.18e-04 (agq / corr, n = 2),
# gaussian 1.06e-05 (laplace / stddev, n = 214), negativebinomial 1.89e-05
# (laplace / corr, n = 46), poisson 1.49e-04 (agq / stddev, n = 3). Bands at
# ceil1 of those fail gate 3 on negativebinomial agq / corr and poisson agq / corr,
# one vector-AGQ cell each, where no SEM exists and glmm sits 2e-05 and 4.5e-04
# above GLMMadaptive (the quadrature-grid placement of dev_align.R's ga_rule_dev).
# A per-family band cannot cover a one-cell group, so the pilot bands below stay
# until the rule says what such a group is read against.
#
# MEASURED ON THE PILOT, 2026-09-22. The gate reads one band per family but
# fires per (family, arm, quantity) group, so a family's band is the LARGEST
# best-oracle 2*SEM over its own groups -- anything smaller would leave a group
# the band does not cover. A family with fewer than 4 paired cells takes the
# POOLED across-family figure instead, because a 2- or 3-cell SEM is not a band:
# pooling the 48 best-oracle paired differences over every group gives
# sd = 7.8443e-03, 2*SEM = 2.2645e-03, ceil1 = 3e-03.
TOL_TRUTH_BAND <- list(
  # largest group 1.4867e-03 (stddev vs glmmTMB, n = 3; the beta group has n = 4,
  # which is what qualifies the family for its own band)
  gaussian = 2e-03,
  # largest group 7.2409e-03 (beta vs glmmTMB, n = 13)
  binomial = 8e-03,
  # pooled: the pilot's poisson groups reach n = 3. Its own largest group is
  # 2.1466e-06, three orders tighter, so this band is loose for poisson.
  poisson = 3e-03,
  # pooled: the pilot's gamma groups reach n = 2. Its own largest group is
  # 1.4382e-02, so this band is TIGHTER than gamma's measured spread.
  gamma = 3e-03,
  # pooled: the pilot's negativebinomial groups reach n = 2 (own largest
  # 1.8987e-02, so this band is tighter than the measured spread).
  negativebinomial = 3e-03,
  # pooled, and NOT measured on this family: the pilot's one inverse-Gaussian
  # cell carries a truth vector but no seed, so the truth gate skips it and
  # there is no paired difference to take a SEM from.
  inversegaussian = 3e-03
)

# The band for one quantity. `cell_id` names the cell asking, so every call site
# reads the same way. An unknown `quantity` stops -- a typo would otherwise gate
# against NULL and silently pass -- and so does an unmeasured band: NA compares
# false, so gating against it passes everything.
tol_for <- function(cell_id, quantity) {
  band <- TOL[[quantity]]
  if (is.null(band)) stop(sprintf("tol_for: no such tolerance `%s`", quantity))
  if (is.na(band))
    stop(sprintf(paste0("tol_for: TOL$%s has not been measured yet. Its comment in ",
                        "grid/tol.R names the run that measures it; write the number ",
                        "there with its measurement."),
                 quantity))
  band
}

# ── inference (0.6.0) ────────────────────────────────────────────────────────
# The bands of the inference tests (df, p-values, test statistics), which fall in
# three classes:
#   fit-free arithmetic  a formula evaluated on frozen inputs: a hard band, and
#                        no registry entry excuses a miss.
#   end-to-end           a fit compared with a frozen reference under
#                        goldens/inference/: a miss passes only under an
#                        'oracle-tier' entry of divergences.json whose `cell` is
#                        the reference's name and whose `quantities` list the
#                        quantity's own label (`df`, `p_value`, `lrt`, `f` or
#                        `confint` end to end; `df_at_lme4`, `dendf_at_lme4` or
#                        `f_at_lme4` for gate 1 at lme4's theta-hat). Each driver
#                        reads only the entries under its labels.
#   Monte Carlo          seeded draws: the band comes from a DKW or binomial
#                        standard error.
# Each step that sets a band enters it here with its measurement, its rule and
# its date, in the house style of every band above.
# "Gate N" in this list is the inference tier's: 1 df and F against lmerTest,
# 2 the contest layer's arithmetic, 3 step stability, 4 dense against sparse.
# compare.R's grid gates (TOL above; its gate 5 is coef_df_rel) are numbered apart.
# A SEPARATE LIST FROM TOL: `tol_for` never reads it, so no grid gate can pick
# up an inference band. Mirrored in tests/oracle_support/mod.rs::tol for the
# integration gates and, for a gate on a crate-private item, beside the in-crate
# test that reads it. Change together.
TOL_INFERENCE <- list(
  # Floor of lmerTest's own FD spread in the per-coefficient Satterthwaite df
  # (end-to-end class; the gate at lme4's theta-hat, whose per-fixture band is
  # max(spread, floor) + df_fd_rel + nonstat on the value route and
  # max(spread, floor) + df_gd_rel + nonstat on the gradient route, which the
  # gate also runs on every reference whose fit arms the per-group gradient).
  # The spread is |df - df_second| / df,
  # the larger of the two settings, read from each reference's frozen `df` and
  # `df_second` (goldens/inference/lmertest/, tools/goldens_lmertest.R: the df
  # recomputed after vcov_varpar and Jac_list are re-derived at numDeriv Hessian
  # d = 0.05 and 0.2, Jacobian d = 5e-5 and 2e-4, r = 4, around lmerTest's
  # d = 0.1 and 1e-4). MEASURED (2026-10-11) over the 52 lmertest references (142
  # coefficients): maximum 2.96e-6 on lmertest_sim_dynrange_lmm_ml, (Intercept);
  # REML maximum 7.56e-7 on lmertest_sim_dynrange_lmm, u; every fixture other
  # than sim_dynrange_lmm at or below 1.5e-7 (lmertest_sim_unbalanced_nested_ml).
  # 1e-6 is a set floor, not derived from the spread: a fixture above it carries
  # its own spread in its band. nonstat = |df(theta0) - df(theta0 - H_p^+ g)|,
  # theta0 lme4's theta-hat mapped to the kernel, is the df's move over one
  # Newton step toward glmm's stationary point. Gated in
  # src/lmm/satterthwaite_tests.rs (gate1_at_lme4_theta, per coefficient,
  # registry quantity `df_at_lme4`), which holds the Rust copy, and
  # src/formula/hypotheses_oracle_tests.rs (the per-term DenDF and F,
  # `dendf_at_lme4` and `f_at_lme4`, the spread read from `DenDF_second`). Its own
  # labels keep an end-to-end `df` or `f` entry from excusing a miss here. MEASURED (2026-10-11, pawel,
  # x86_64) over the 26 REML per-coefficient references (71 coefficients): worst
  # |df - df_lmerTest| / df_lmerTest 9.10e-6 (lmertest_sim_dynrange_lmm, u; band
  # 2.73e-5) and 8.83e-6 (lmertest_sim_nested_crossed_mix, (Intercept); band
  # 2.80e-5), every other reference at or below 1.88e-6
  # (lmertest_sim_wide_crossed, (Intercept)); nonstat up to 2.99e-4
  # (lmertest_Machines, MachineC, where the df error is 5.4e-8). Per-term rows
  # (8 REML references, Type III and II): DenDF worst 1.25e-6
  # (lmertest_anova_sleepstudy, Days), F worst 1.3e-12. The gradient route,
  # MEASURED (2026-10-11, pawel, x86_64) on the 7 references whose fit arms it
  # (sleepstudy, Machines, sim_max_q_slope, lmm_slope, all_ones,
  # sim_corr_boundary, sim_factorial): worst 1.01e-6 (lmertest_all_ones, x; band
  # 7.48e-5), and the closest to its band 9.97e-7 against 1.04e-6
  # (lmertest_sleepstudy, Days: spread 1.36e-8, nonstat 4.1e-8, so the floor
  # carries it). That gap is shared by both routes (value 1.25e-6, gradient
  # 9.97e-7) and lies outside the crate's FD: the routes agree to 4e-9 there once
  # Richardson removes the value route's truncation. lmerTest's df does not move
  # across its numDeriv settings (spread 1.36e-8), so the cause is open; the
  # floor carries it. Per-term rows on the gradient
  # route: DenDF worst 9.97e-7 (lmertest_anova_sleepstudy, Days; band 1.04e-6),
  # F worst 1.3e-12.
  df_lmertest_floor_rel = 1e-6,
  # Value-route finite-difference band of the per-coefficient Satterthwaite df
  # (hard class, no registry entry). Gate 3, step stability: the df at the pass's
  # step eta = 3e-4 (h_k = eta * max(|theta_k|, 1), ETA in src/lmm/satterthwaite.rs)
  # against the df at 1e-3, per coefficient; also gate 4 (dense against sparse
  # kernel at one theta; the sparse kernel has the value route only, so on a
  # per-group design both the dense gradient route and the dense value route
  # are held to this band against it), the df_fd_rel term of the gate at
  # lme4's theta-hat (band max(spread, floor) + df_fd_rel + nonstat), and the
  # gradient route against the value route on the same fit
  # (gradient_route_agrees_with_value_route: every df, and each H_p entry (i, j)
  # at df_fd_rel x sqrt(H_ii H_jj); MEASURED 2026-10-11 at ETA_GRAD = 3e-6, df
  # 4.5e-7 on q3s near-zero, H_p 5.1e-7 on sim_max_q_slope; MEASURED 2026-10-12
  # under ML on its 9 fixtures, df 1.7e-6 and H_p 1.5e-6 on q3s near-zero).
  # Gated in src/lmm/satterthwaite_tests.rs, which holds the Rust copy. MEASURED
  # (2026-10-11, pawel, x86_64) by step_sweep_is_flat_at_eta over its 17 fixtures
  # (the toy 300 + y/4; sim_dynrange_lmm, sim_nested_crossed_mix,
  # sim_near_singular, sim_corr_boundary; the speed-campaign shapes int1, nest2,
  # q2s, q3s, cross4 and cross6, a base and a near-zero draw each): maximum 8.7e-6
  # (sim_nested_crossed_mix), every speed-campaign shape at or below 5.9e-6 (q3s
  # near-zero). Rule ceil1(2 x 8.7e-6) = 2e-5. The smaller steps are not gated:
  # round-off from the random effects moves the df by 2.9e-5 between 1e-4 and
  # 3e-4 (sim_nested_crossed_mix) and 1.5e-3 between 3e-5 and 1e-4
  # (sim_dynrange_lmm). As a diagnostic only, Richardson extrapolation from the
  # steps 5e-4 and 1e-3 puts the error at 3e-4 at 3.0e-6 or below.
  df_fd_rel = 2e-5,
  # Gradient-route band of the per-coefficient Satterthwaite df (hard class, no
  # registry entry): H_p from central differences of the per-group analytic
  # gradient, on the fits that arm it. Gate 3, step stability: the df at the
  # route's step eta = 3e-6 (h_k = eta * max(|theta_k|, 1), ETA_GRAD in
  # src/lmm/satterthwaite.rs) against the df at 1e-5, per coefficient; also the
  # df_gd_rel term of the gate at lme4's theta-hat on that route (band
  # max(spread, floor) + df_gd_rel + nonstat). Gated in
  # src/lmm/satterthwaite_tests.rs, which holds the Rust copy. Guessed a priori
  # at 1e-9.
  # MEASURED (2026-10-11, pawel, x86_64) by gradient_step_sweep_is_flat_at_eta_grad
  # over the steps 1e-6, 3e-6, 1e-5, 3e-5 and 1e-4 on 9 per-group fixtures
  # (sleepstudy, sim_max_q_slope, the weighted lmm_slope and all_ones,
  # sim_corr_boundary; the speed-campaign shapes q2s and q3s, a base and a
  # near-zero draw each), the largest change of a df per pair of steps: 1.5e-9
  # (1e-6 to 3e-6, sim_max_q_slope), 1.2e-9 (3e-6 to 1e-5, q3s near-zero), 6.9e-9
  # (1e-5 to 3e-5) and 7.8e-8 (3e-5 to 1e-4, both q3s near-zero): round-off below
  # 3e-6, truncation growing as eta^2 above 1e-5. Rule ceil1(2 x 1.2e-9) = 3e-9,
  # above the a-priori 1e-9: no pinned step holds 1e-9 against either
  # neighbour. As a diagnostic only, Richardson extrapolation from the steps 1e-5
  # and 3e-5 puts the error at 3e-6 at 3.1e-10 or below (8.7e-10 at 1e-5). On the
  # same fits the gradient route's df is within 4.5e-7 of the value route's (q3s
  # near-zero), inside df_fd_rel: the gap is the value route's error.
  df_gd_rel = 3e-9,
  # Contest layer, fit-free class (gate 2, src/inference/contest_tests.rs, which
  # holds the Rust copies). The crate's contest1D and get_Fstat_ddf, fed lmerTest's
  # frozen beta, vcov_beta, vcov_varpar and Jac_list, against lmerTest's own
  # estimate, std error, df and t and the get_Fstat_ddf case table: the same
  # arithmetic in another language. The inputs carry jsonlite's 15 significant
  # digits, and rounding them to 15 digits moves contest1D's df by at most 1.2e-14
  # relative (MEASURED 2026-10-11 on sleepstudy, sim_factorial, Machines,
  # sim_wide_slopes, sim_max_q_slope). MEASURED (2026-10-11, pawel, x86_64) by the
  # gate over the 52 per-coefficient references (142 coefficients and 26 seeded
  # l): df maximum 1.23e-14 (lmertest_Machines, (Intercept)); t maximum 6.68e-15
  # (lmertest_lmm_crossed, seeded l); case table maximum 1.17e-16 (`general`).
  # The band is set a priori (the same arithmetic), with about 80x room over the
  # df maximum.
  contest_rel = 1e-12,
  # contest1D's p on the same rows. p composes the tail function (itself gated at
  # 1e-12 against R, tests/fixtures/tail_values.json) with t and df, at relative
  # condition numbers kappa_t = |d ln p / d ln t| and kappa_nu = |d ln p / d ln df|
  # read in R off the frozen rows with a finite nonzero p (pt(log.p = TRUE), the
  # df derivative by central differences): kappa_t maximum 268.4
  # (lmertest_sim_unbalanced_nested_ml, x: t = 24.9, df = 473, p = 6.8e-88),
  # kappa_nu maximum 68.5 (lmertest_sim_wide_crossed_ml, x). Budget
  # 1e-12 + 268.4 x 6.68e-15 + 68.5 x 1.23e-14 = 3.64e-12 (the t and df maxima
  # above); rule ceil1(2 x 3.64e-12) = 8e-12. MEASURED (2026-10-11, pawel) worst
  # 5.34e-13 (lmertest_sim_unbalanced_nested_ml, x). The two frozen p of exactly 0
  # (lmertest_sim_dynrange_lmm and _ml, w: t = 313.53 and 313.69, both sides
  # underflow) are
  # compared exactly, not under this band.
  contest_p_rel = 8e-12,
  # contestMD's F, DenDF, p and per-direction nu_m on the same inputs and on the
  # frozen L. The nu_m depend on the eigenvectors of L C L', whose error is about
  # eps * d1 / gap, so every frozen L has a minimum relative eigen-gap of at
  # least 1e-4 (about 2e-12 at that gap); rounding the inputs to 15 digits moves
  # contestMD's DenDF by at most 7.6e-15 relative (MEASURED 2026-10-11, the same
  # five fixtures). MEASURED (2026-10-11, pawel) by the gate over the 67 frozen
  # cases (every REML reference with p >= 2, plus lmertest_anova_sim_factorial's
  # basis pair): F maximum 2.36e-14 (lmertest_sim_wide_crossed, identity L);
  # DenDF maximum 3.02e-14 (lmertest_anova_sim_factorial, basis pair, lmerTest's
  # L); nu_m maximum 4.87e-14 (lmertest_sim_factorial, identity L, sixth nu_m);
  # p maximum 8.14e-13 (lmertest_sim_wide_crossed, identity L). The band is set
  # a priori, about 50x over the eigenvector error at the smallest admitted gap
  # (2e-12 at 1e-4) and about 120x over the largest measured (p, 8.14e-13).
  contest_md_rel = 1e-10,
  # Gate 1 end to end (end-to-end class, registry quantity `df`, and `f` for the
  # per-term F): glmm's per-coefficient Satterthwaite df at its own theta-hat
  # against lmerTest's (tests/inference_oracle.rs::lmertest_end_to_end), and the
  # per-term DenDF and F end to end (src/formula/hypotheses_oracle_tests.rs,
  # which holds its own Rust copy). df moves with theta-hat as the SE does, and
  # the two engines agree on theta-hat only to their optimizers' tolerance, so the
  # band is the cross-engine LMM SE band. MEASURED (2026-10-11, pawel, x86_64)
  # over the 26 REML per-coefficient references (71 coefficients): maximum
  # 2.96e-4 (lmertest_Machines, MachineC; its df at lme4's theta-hat agrees to
  # 5.4e-8, so the gap is the two theta-hats'), every other reference at or below
  # 1.64e-4 (lmertest_sim_wide_crossed, (Intercept)). Per-term rows: DenDF
  # maximum 2.91e-4 and F 1.18e-4 (lmertest_anova_Machines, Machine). Each fit
  # takes its own derivative route here, the gradient route on the 7 fits that
  # arm the per-group gradient; RE-MEASURED (2026-10-11, pawel) on those routes:
  # maximum unchanged at 2.96e-4 (lmertest_Machines, gradient route), the
  # value-route fits at or below 1.64e-4.
  df_e2e_rel = TOL$se_rel,
  # Gate 1's interval comparison (end-to-end class, registry quantity
  # `confint`): the bounds beta-hat -/+ qt(1 - (1 - level)/2, df) * SE at glmm's
  # own theta-hat against lmerTest's contest1D(m, L, confint = TRUE) lower and
  # upper (tests/inference_oracle.rs::lmertest_confint). Each bound's difference
  # is measured in units of lmerTest's interval width, since a bound may sit near
  # 0, where a difference relative to the bound has no fixed scale. A bound moves
  # by d(beta) + q d(SE) + SE d(q); over the width 2 q SE each term is half a
  # relative move of beta (on the SE scale), of SE or of q, each of the SE band's
  # order, so the band is the SE band. MEASURED (2026-10-11, pawel, x86_64)
  # over the same 71 coefficients: maximum 8.61e-5 (lmertest_Machines,
  # MachineB), every other reference at or below 2.61e-5 (lmertest_Penicillin).
  # On the scale of the bound itself the maximum is 6.14e-4 (the same bound,
  # 1.74 against a width of 12.4). RE-MEASURED (2026-10-11, pawel) with each fit
  # on its own derivative route: maximum 8.60e-5 (lmertest_Machines, MachineB,
  # gradient route), the value-route fits at or below 2.61e-5
  # (lmertest_Penicillin).
  confint_rel = TOL$se_rel,
  # Exact RLRT, the design eigenvalues (hard: no registry entry excuses a miss).
  # glmm's normalized nonzero eigenvalues of Z'P0Z, from a (1 | g) fit's design
  # summary, against RLRsim's frozen `mu` (goldens/inference/rlrt/,
  # tools/goldens_rlrt.R), per value in units of the largest, with the same
  # count below 1e-10 (src/inference/rlrt_oracle_tests.rs::rlrt_mu_matches_rlrsim,
  # which reads the Rust copy in src/inference/rlrt_refs.rs::bands). Both routes
  # give the eigenvalues of one symmetric matrix to about K * eps, so the band is
  # set a priori, three to five decades above that. MEASURED (2026-10-12, pawel,
  # x86_64) over the six RLRsim references: worst 8.99e-15 (rlrt_large_k,
  # K = 1000), every other reference at or below 9.99e-16 (rlrt_sleepstudy).
  rlrt_mu_rel = 1e-10,
  # Exact RLRT, the grid kernel against RLRsim's paired draws (hard: no
  # registry entry excuses a miss). RLRsim's frozen `mu` (zeros kept), grid and
  # per-draw `Chi1` and `ChiK` fed to glmm's profile, RLRsim's first-decrease
  # rule applied to it, against RLRsim's `res` within
  # rlrt_paired_rel * |res| + rlrt_paired_abs, and the 1-based index equal to
  # `lambdaind` (src/inference/rlrt_oracle_tests.rs::paired_profile_matches_rlrsim,
  # which reads the Rust copies in src/inference/rlrt_refs.rs::bands). Set from
  # the R replay of the same draws (4.3e-14) and the 15 significant digits the
  # JSONs keep. MEASURED (2026-10-12, pawel, x86_64) over the 2000 paired draws
  # of rlrt_dyestuff, rlrt_sleepstudy, rlrt_cake and rlrt_few_unbal: worst
  # absolute difference 5.91e-14, worst use of the band 5.0% (rlrt_cake, draw
  # 206); the index agrees on every draw, and the global maximum equals the
  # first decrease on every draw. The internal grid against the frozen grid:
  # worst relative difference 4.46e-15, the JSON's 15-digit rounding.
  rlrt_paired_rel = 1e-12,
  rlrt_paired_abs = 1e-13,
  # Exact RLRT, the simulated null against RLRsim's frozen null (Monte Carlo).
  # Per RLRsim reference, glmm's null at B = 1e5 draws (seed 20261021 + the
  # entry's index among the manifest's rlrt entries) against RLRsim's 1e5
  # (src/inference/rlrt_oracle_tests.rs::ks_null_matches_rlrsim, which reads the
  # Rust copies in src/inference/rlrt_refs.rs::bands): the two-sample KS
  # distance within rlrt_ks_coef * sqrt((B + B_ref) / (B * B_ref)), the
  # asymptotic critical value at alpha = 0.001 (0.0087 at 1e5 against 1e5; the
  # atom at zero makes the KS conservative), and the mass at zero within
  # rlrt_zero_mass_se pooled two-proportion standard errors. MEASURED
  # (2026-10-12, pawel, x86_64) over the six RLRsim references: worst KS
  # 4.90e-3, 56% of the band (rlrt_cake), every other reference at or below
  # 4.04e-3 (rlrt_large_k); worst mass-at-zero gap 1.77 standard errors
  # (rlrt_cake), every other reference at or below 1.34 (rlrt_few_unbal).
  rlrt_ks_coef = 1.95,
  rlrt_zero_mass_se = 4,
  # Exact RLRT, the observed statistic against RLRsim's (gate 5,
  # registry-backed, quantity `rlrt`). glmm's statistic, 2 (l_R1 - l_R0) from
  # the (1 | g) fit and the OLS fit made with reml = TRUE, against RLRsim's
  # frozen statistic, absolute (src/inference/rlrt_oracle_tests.rs::
  # rlrt_observed_matches_rlrsim, which reads the Rust copy in
  # src/inference/rlrt_refs.rs::bands). Twice loglik_abs_lmm, since the
  # statistic is twice a difference of REML log-likelihoods. MEASURED
  # (2026-10-12, pawel, x86_64) over the six RLRsim references: worst 8.19e-12
  # (rlrt_sleepstudy), every other reference at or below 5.49e-12
  # (rlrt_large_k); on rlrt_dyestuff2 glmm's pinned fit gives exactly 0 against
  # RLRsim's 5.68e-14, inside the band, so the statistic needs no entry.
  rlrt_stat_abs = 4e-6,
  # Exact RLRT, gate 1's fit-free part (Monte Carlo): on four balanced
  # intercept-only designs, (K, J) = (6, 5), (3, 4), (4, 5) and (10, 3) with
  # seeds 20261011 to 20261014, glmm's null at B = 1e5 against the one-way
  # ANOVA F test, of which the RLRT is a function there: |ECDF(g(F_q)) - q|
  # at the F quantiles above the atom, and the mass at zero against
  # P(F <= 1), each within the DKW band sqrt(log(2 / alpha) / (2 B)) at
  # alpha = rlrt_dkw_alpha, 6.165e-3 at B = 1e5
  # (src/inference/rlrt_tests.rs::balanced_null_matches_anova_f, which reads
  # the Rust copy in src/inference/rlrt_refs.rs::bands). MEASURED
  # (2026-10-12, pawel, x86_64): worst 1.76e-3, 29% of the band (the mass at
  # zero of (K, J) = (10, 3)); every quantile gap at or below 1.48e-3.
  rlrt_dkw_alpha = 1e-3,
  # Exact RLRT, the simulated p-value (Monte Carlo): within rlrt_p_se
  # binomial standard errors plus 1 / (B + 1), the p-value's own floor, at
  # B = 1e5. Against RLRsim's frozen p (gate 6, registry-backed, quantity
  # `p_value`; src/inference/rlrt_oracle_tests.rs::rlrt_observed_matches_rlrsim,
  # seed 20261021 + the entry's index among the manifest's rlrt entries), the
  # standard error of two independent nulls, sqrt(2 p (1 - p) / B) with p
  # RLRsim's, so where RLRsim's p is 0 the band is 1 / (B + 1) and glmm's p
  # must be exactly 1 / (B + 1). Against Dyestuff's exact F-test p (gate 1,
  # hard; src/inference/rlrt_tests.rs::rlrt_dyestuff_p_matches_anova_f, seed
  # 20261015), sqrt(p (1 - p) / B) with p the F test's. Both read the Rust copy
  # in src/inference/rlrt_refs.rs::bands. MEASURED (2026-10-12, pawel, x86_64):
  # against RLRsim, worst 44% of the band (rlrt_few_unbal, 2.540e-3 against
  # 2.170e-3), rlrt_dyestuff 10% (4.380e-3 against 4.260e-3); rlrt_sleepstudy,
  # rlrt_cake and rlrt_large_k at exactly 1 / (B + 1) against RLRsim's 0;
  # rlrt_dyestuff2 exactly 1 against RLRsim's 0.4392, the registered oracle
  # defect (|1 - 0.4392| / 0.4392 = 1.2769). Against the F test, 9% of the
  # band (4.470e-3 against 4.398e-3).
  rlrt_p_se = 4,
  # Exact RLRT, gate 7, the level at nominal alpha (Monte Carlo, feature
  # montecarlo-tests). On three designs under H0, the X and groups of
  # rlrt_dyestuff, rlrt_few_unbal and rlrt_cake with y iid N(0, 1), R = 2000
  # datasets each (seeds 20261051 to 20261053), every statistic against one
  # null of B = 1e5 draws per design (seeds 20261041 to 20261043): the
  # rejection rate at alpha in {0.01, 0.05, 0.10} within
  # rlrt_level_se * sqrt(alpha (1 - alpha) / R_kept) of alpha, R_kept the
  # datasets kept (R less those whose (1 | g) fit did not converge or whose
  # statistic is NaN, at most 1%), +-0.0146 at alpha = 0.05 with all R kept
  # (src/inference/rlrt_montecarlo_tests.rs::level_rejection_rates, which reads
  # the Rust copy in src/inference/rlrt_refs.rs::bands). The shared null's own
  # error at B = 1e5 adds at most sqrt(alpha (1 - alpha) / B), about 7e-4 at
  # alpha = 0.05. Set a priori; the study runs at release.
  rlrt_level_se = 3,
  # The REML log-likelihood at theta = 0 against R (gates 5 and 11, hard): the
  # OLS fit made with reml = TRUE against the frozen
  # logLik(lm, REML = TRUE) on every rlrt reference, the weighted
  # rlrt_reml0_weighted included, absolute
  # (src/inference/rlrt_oracle_tests.rs::ols_reml_loglik_matches_lm_reml, which
  # reads the Rust copy in src/inference/rlrt_refs.rs::bands). One closed form
  # on both sides, so the band is set a priori. MEASURED (2026-10-12, pawel,
  # x86_64) over the seven references: worst 9.09e-13 (rlrt_large_k,
  # n = 3000), the weighted reference 1.14e-13, every other reference at or
  # below 3.13e-13 (rlrt_dyestuff).
  reml0_loglik_abs = 1e-9,
  # The model comparisons end to end (end-to-end class, registry quantities
  # `lrt`, `p_value`, `f`): every pair of goldens/inference/anova/ and every
  # ranova row (full formula against the row's reduced one), both sides fitted
  # by glmm and compared through formula::anova::compare
  # (tests/inference_oracle.rs::anova_pairs). The deviance gate is read first,
  # per side, on -2 logLik from the frozen `logLik`: glmm at most the oracle
  # tier's DEV_EPS = 2e-4 (tests/oracle_support/mod.rs) above it. MEASURED
  # (2026-10-12, pawel, x86_64) over 48 sides: worst |gap| 1.30e-5
  # (anova_glmm_pair_grouseticks, glmm below lme4 on both sides by 1.30e-5 and
  # 1.15e-5), every other side at or below 1.6e-7. Two pairs are not read
  # against their own logLik: anova_glmm_vs_glm_sim_nb's glmer.nb side is read
  # against the glmmTMB fit of the same model (goldens/sim_nb_glmm_tmb.json; gap
  # -1.2e-9), since glmer.nb maximises another objective (lme4's reported
  # deviance 0.0243 below glmm's, the goldens tier's registered sim_nb_glmm
  # entries); the Gamma GLM pair, whose reference is a Wald F, has no deviance
  # gate (R's logLik of a weighted Gamma glm multiplies each log-density by its
  # weight and takes phi as the mean deviance: -358.999 against glmm's
  # precision-weight maximum -330.523, which R reproduces by optimising
  # sum(dgamma(y, w / phi, scale = mu phi / w, log = TRUE)) over phi).
  # The statistic, absolute: twice DEV_EPS, the sum of the two sides' gaps the
  # deviance gate admits. MEASURED (2026-10-12, pawel, x86_64) over the 22
  # likelihood-ratio rows: worst 1.49e-6 (anova_glmm_pair_grouseticks), every
  # other row at or below 1.5e-7 (anova_glmm_pair_sim_binomial_slope1);
  # anova_glmm_vs_glm_sim_nb 2.43e-2 (glmm's statistic the smaller), the
  # registered objective gap above. anova_ml_pair_sleepstudy_fixed, whose
  # smaller model Reaction ~ 1 + (Days | Subject) has a random slope whose
  # variable is no fixed term: MEASURED (2026-10-26, pawel, x86_64) 2.472e-8.
  # Also read, hard, by the in-crate check of the per-term likelihood-ratio
  # rows against afex (src/formula/anova_tests.rs::
  # type_iii_lrt_rows_match_afex_on_the_frozen_glmm, which holds a third copy;
  # anova_lrt3_glmm_sim_factorial_bern: worst 8.9e-8, g). Change together.
  lrt_abs = 4e-4,
  # The likelihood-ratio p-value against the one the entry's `reference` names
  # (p_chisq or p_mixture; none on the two rows inside the exact test's
  # envelope, whose p is gated against RLRsim above), relative. lrt_abs carried
  # through the tail: |d ln p| = h dT, h = -d ln p / dT read in R off the frozen
  # statistic and reference of each gated row: maximum 0.848
  # (anova_ranova_pastes, row 1: T = 0.658, the mixture with k = 0), every other
  # row at or below 0.577. Rule ceil1(0.848 x 4e-4 = 3.39e-4) = 4e-4. MEASURED
  # (2026-10-12, pawel) over 20 rows: worst 7.60e-7
  # (anova_glmm_pair_grouseticks); anova_glmm_vs_glm_sim_nb 1.26e-2, registered.
  lr_p_rel = 4e-4,
  # The per-term likelihood-ratio rows' p-value against the frozen afex, car
  # and car-containment glmer tables (tests/inference_oracle.rs::
  # anova_lrt_rows, the lrt_terms references), relative. The rule of
  # lr_p_rel: lrt_abs carried through the tail, |d ln p| = h dT with
  # h = dchisq(T, df) / pchisq(T, df, lower.tail = FALSE) at each gated row's
  # frozen statistic. These rows hold null-term statistics near 0 on 1 df,
  # where the tail is steep: maximum 7.046 (anova_lrt2_glm_sim_factorial_bern
  # and anova_lrt3_glm_sim_factorial_bern, row x: T = 0.00352), then 3.180
  # (the GLMM x rows, T = 0.0195), 2.877 (anova_lrt3_glmm_sim_factorial_bern,
  # h), 2.445 (anova_lrt2car_glmm_sim_factorial_bern, h), 1.192 and 1.081 (the
  # GLM h rows), every other row at or below 0.51. Rule ceil1(7.046 x 4e-4 =
  # 2.82e-3) = 3e-3. lr_p_rel stays for its own readers (h at or below
  # 0.848). MEASURED (2026-10-13, pawel, x86_64) over the 22 rows of the 8
  # gated references: p worst 7.60e-7 (anova_lrt3_glmm_grouseticks and
  # anova_lrt2_glmm_grouseticks, cHEIGHT); the statistic, read against lrt_abs
  # by the same gate, worst 1.49e-6 (the same rows); the GLM rows at or below
  # 2e-13 on both.
  lt_lr_p_rel = 3e-3,
  # The per-term likelihood-ratio rows against formula::anova::compare on the
  # formula pair each row stands for (hard, fit-free of any oracle: both sides
  # are glmm's own fits; src/formula/anova_tests.rs::
  # likelihood_ratio_rows_equal_compare_on_their_formula_pairs, which holds the
  # Rust copies). Every Type II row, and every Type III row whose reduced
  # design X N is a column drop, within lt_compare_abs + lt_compare_rel * |T|,
  # T compare's statistic. The row's refits start warm from the full fit,
  # compare's fits cold, so the two land on the same optimum to the
  # optimizer's stopping rule: an absolute residue on 2 (l1 - l0) that does
  # not shrink with T, and a relative one on the large statistics. Null-term
  # statistics sit below 0.1, so a relative band alone means nothing there.
  # MEASURED (2026-10-13, pawel) over 40 rows (binomial GLM; binomial GLMM by
  # Laplace, by AGQ with 5 nodes and with a random slope on x; weighted
  # negative binomial GLM; negative binomial GLMM; Gamma GLM with phi held;
  # Gamma GLMM; Poisson GLMM with an offset; Type II and III), on x86_64 and
  # on wasm32-wasip1 plain and +simd128 (the CI lib legs): rows with T < 1
  # worst 8.92e-12 absolute (wasm simd128; x86_64 8.75e-12; binomial GLMM with
  # the slope, Type II h, T = 0.044), rows with T >= 1 worst 1.58e-12 relative
  # (wasm plain; x86_64 1.57e-12; binomial GLMM nAGQ 5, Type II g, T = 12.84,
  # 2.02e-11 absolute); the GLM rows agree exactly. Rule ceil1(2 x worst):
  # lt_compare_abs from the T < 1 rows, lt_compare_rel from the T >= 1 rows.
  # Measured far below the 1e-7 to 1e-6 the plan guessed for two optimizations
  # from different starts. The Type III null basis is taken from the
  # unrounded hypothesis (formula::hypotheses::type3_unrounded): from the
  # zapsmall-rounded one the statistic of anova_lrt3_glmm_sim_factorial_bern's
  # h row is 0.02446281944 against 0.02446285055, a move of 3.1e-8 (afex
  # 0.02446285427: 3.5e-8 from the rounded basis, 3.7e-9 from the unrounded).
  # On wasm32-wasip1 (2026-10-16, pawel) the same row is 0.024462850554584747
  # (plain) and 0.02446285055461317 (+simd128) on the unrounded basis against
  # 0.02446281944392581 and 0.02446281944395423 on the rounded one: the same
  # 3.1e-8 move, above the 1e-8 the pin guard
  # type_iii_lrt_h_row_refits_on_the_unrounded_null_basis needs to tell the
  # two bases apart.
  # Limits: aarch64 (NEON) is not measured; and the absolute residue is the
  # optimizer's stopping residue on these fixtures (it scales with |l|, the
  # curvature and the stopping rule), so a new gate-2 fixture with a much
  # larger |l| needs its own measurement rather than this constant.
  # RE-MEASURED (2026-10-13, pawel, x86_64) after the refits dropped their
  # standard errors (Rx covariance, no SE targets): all 40 statistics
  # unchanged bit for bit.
  lt_compare_abs = 2e-11,
  lt_compare_rel = 4e-12,
  # The F and Wald rows (end-to-end class, registry quantities `f` and
  # `p_value`): the OLS pairs' F against anova.lm's and the Gamma GLM pair's
  # Wald F against car::linearHypothesis(test = "F") (anova_pairs), and per term
  # through formula::anova::term_tests against car's Type II table and the Type
  # III table of its contr.sum refit (car's (Intercept) rows and the
  # treatment-coded Type III recorded only), on the glm and lm twins, the
  # weighted lm on glm_gamma and the Gamma GLM's one linear hypothesis
  # (tests/inference_oracle.rs::anova_term_rows). The statistic, relative: the
  # beta/SE band, as for every other Wald quantity. MEASURED (2026-10-13, pawel,
  # x86_64): worst 2.46e-5 (the Gamma Wald F, 19.04497 against 19.04451, on the
  # pair and on the linear hypothesis alike: glm's IRLS stop); GLM Wald
  # chi-square at or below 9.2e-7; OLS F at or below 3.8e-14 (the weighted lm on
  # glm_gamma at or below 5.7e-15), except car_lm's contr.sum Type III h on
  # sim_factorial at 5.74e-8, where term_tests' Type III L carries lmerTest's
  # zapsmall rounding (7 digits) and car's contr.sum row is exact.
  wald_rel = TOL$se_rel,
  # The GLMM Wald chi-square rows (anova_car_glmm_sim_factorial_bern), relative.
  # An oracle-side floor: the frozen treatment fit (Nelder_Mead, max|grad|
  # 9.7e-4) stops short of lme4's tight optimum (bobyqa, rhoend 1e-12) by, for
  # g, h, x and g:h, 7.1e-5, 5.1e-5, 2.9e-4 and 9.2e-4 on the Type II rows and
  # 3.4e-6, 2.5e-4, 3.4e-5 and 3.6e-5 on the contr.sum Type III rows (MEASURED
  # 2026-10-12 when the reference was frozen), and no logLik of the model is
  # frozen, so no deviance gate can adjudicate. Rule 2 x 9.2e-4, ceil1: 2e-3.
  # MEASURED (2026-10-12, pawel, x86_64), glmm against car: Type II 6.1e-5,
  # 4.3e-5, 2.8e-4 and 9.41e-4; contr.sum Type III 1.0e-5, 2.5e-4, 3.8e-5 and
  # 3.3e-5. Each within 2.2e-5 of the floor: glmm sits on lme4's tight optimum.
  # The fit keeps the default wald_se and no HessianSeFallback note (car's vcov
  # is lme4's finite-difference Hessian; the RX route moves the chi-square by 5
  # to 13%).
  wald_glmm_rel = 2e-3,
  # The p-value of an F or Wald row, relative: the statistic's band carried
  # through the tail, |d ln p| = kappa |d ln s|, kappa = |d ln p / d ln s| read
  # in R off the frozen rows (the df are compared exactly): maximum 33.1
  # (anova_car_lm_glm_gamma, contr.sum Type III x, F = 98.47 on 1 and 196 df;
  # Type II x 30.5) over the wald_rel rows, 3.31e-2 at wald_rel; maximum 5.74
  # (contr.sum Type III g) over the GLMM rows, 1.15e-2 at wald_glmm_rel. Rule
  # ceil1 of the larger: 4e-2, the value of satt_p_rel. MEASURED (2026-10-13,
  # pawel, x86_64): worst 2.24e-4 (the Gamma Wald F); GLMM rows at or below
  # 3.39e-4 (Type II g); GLM rows at or below 5.0e-6; OLS rows at or below
  # 1.1e-7 (the weighted lm on glm_gamma at or below 2.8e-14).
  wald_p_rel = 4e-2,
  # The Satterthwaite per-term rows of the lmerTest tables through term_tests
  # (hard: these names are registry-gated by the in-crate driver of
  # src/formula/hypotheses_oracle_tests.rs alone; tests/inference_oracle.rs::
  # lmertest_term_rows_match_anova): F and DenDF at df_e2e_rel, Sum Sq and
  # Mean Sq at sum_sq_rel, NumDF exactly, and the p-value relative. df_e2e_rel
  # on F and on DenDF carried through the
  # tail: kappa_F + kappa_nu, read in R off the frozen rows (kappa_nu = |d ln p
  # / d ln DenDF| by central differences): maximum 31.7
  # (lmertest_anova_sim_factorial_ml, Type II g: 23.6 + 8.1), 3.17e-2 at 1e-3.
  # Rule ceil1: 4e-2. MEASURED (2026-10-12, pawel, x86_64) over sim_factorial,
  # its _ml twin and sim_factorial_empty, Type III and II (24 rows): F worst
  # 6.21e-6 and DenDF worst 4.41e-6 (_ml, x), p worst 3.02e-5 (Type II g).
  satt_p_rel = 4e-2,
  # lmerTest's Sum Sq and Mean Sq (F times sigma-hat squared, times NumDF) on
  # the Satterthwaite per-term rows, relative: df_e2e_rel on F (written as
  # se_rel, which df_e2e_rel is; a list cannot read its own entries) plus
  # twice stddev_rel on sigma-hat squared. MEASURED (2026-10-13, pawel, x86_64) over
  # the same 24 rows: worst 7.40e-6 on both (_ml, x).
  sum_sq_rel = TOL$se_rel + 2 * TOL$stddev_rel,
  # Partial eta-squared of an F per-term row, F * df1 / (F * df1 + df2), as
  # effectsize::F_to_eta2 computes it, relative: wald_rel on F plus df_e2e_rel
  # on df2 (each written as se_rel, which both are; a list cannot read its own
  # entries), the bound of |d ln eta2| = (1 - eta2) |d ln F - d ln df2| with
  # df1 exact. MEASURED (2026-10-26, pawel, x86_64) on the four rows of
  # effectsize_ols_sim_factorial (tests/formula_anova.rs::
  # partial_eta2_matches_the_frozen_ols_reference): worst 5.58e-8 (h, the
  # zapsmall rounding of term_tests' Type III L noted at wald_rel), the other
  # rows at or below 3.7e-14. Over the 17 rows of the six effectsize references
  # with eta2 (tests/inference_oracle.rs::effect_sizes): worst 2.39e-5
  # (effectsize_glm_glm_gamma, x:grp), the LMM rows at or below 9.95e-6
  # (effectsize_machines, Machine).
  eta2_rel = TOL$se_rel + TOL$se_rel,
  # Marginal and conditional R2 (performance::r2_nakagawa) and the adjusted and
  # unadjusted ICC (performance::icc), relative. Each is a ratio N / D of sums of
  # the fixed, random and residual variances v_k, and d ln(N / D) =
  # sum_k c_k d ln v_k with every |c_k| <= 1, so the error is at most the sum of
  # the three parts': twice beta_rel on var(X beta-hat), twice stddev_rel on
  # each of the random and the residual variance. MEASURED (2026-10-26, pawel,
  # x86_64) over the seven mixed-model effectsize references
  # (tests/inference_oracle.rs::effect_sizes): R2 worst 3.13e-5
  # (effectsize_machines, marginal), ICC worst 3.10e-5
  # (effectsize_glmm_sim_factorial_bern, unadjusted); the other LMMs at or
  # below 6.93e-6, the other GLMMs at or below 1.64e-5.
  r2_icc_rel = 2 * TOL$beta_rel + 4 * TOL$stddev_rel,
  # A Tukey-adjusted pairwise p-value (emmeans' adjust = "tukey"), relative:
  # the row's p band (satt_p_rel on a t row, wald_p_rel on a z row, both 4e-2,
  # written as the literal since a list cannot read its own entries) plus R's
  # ptukey error, 1e-5 relative where it agrees with commonstats'
  # StudentizedRange (df = Inf, or 30 <= df <= 25000, with both tails at least
  # 1e-3: the agreement that type's doc comment gives). Outside that region
  # the Tukey rows are not compared: posthoc_machines (df about 5) and
  # posthoc_oats (df about 10) whole, and every row whose frozen p or 1 - p is
  # below 1e-3, where ptukey's upper tail is 1 - ptukey(lower), cancellation
  # noise. MEASURED (2026-10-27, pawel, x86_64) over the 47 compared rows of
  # the 14 other posthoc references (tests/inference_oracle.rs::posthoc_rows;
  # 13 rows left out for their tail): worst 7.55e-4
  # (posthoc_glmm_sim_factorial_bern_by, a - b within p, a z row); the t rows
  # at or below 7.46e-6 (posthoc_lmm_sim_factorial_by, b - c within p).
  tukey_p_rel = 4e-2 + 1e-5
)

# Max relative difference over two aligned numeric vectors; NA on length mismatch
# so it shows up as a hard failure rather than a silently-recycled false pass.
#
# Coordinates whose own magnitude (the larger of the two sides) is at or below
# TOL$near_zero_abs score 0 rather than a ratio: see that constant for why a
# relative question stops having an answer down there, and for the measurement
# the cut was sized from. Pass
# `atol = 0` to get the pure relative metric back -- the port gate does, because
# TOL$port_rel = 1e-12 is diagnostic at exactly 0 and an absolute grace of 1e-3
# would swallow the entire wiring bug it exists to catch.
rel_max <- function(x, y, atol = TOL$near_zero_abs) {
  if (length(x) != length(y)) return(NA_real_)
  # A coordinate that is NULL ON BOTH SIDES is not a comparison, so it is dropped
  # rather than poisoning the whole vector: a rank-deficient fit records the
  # aliased column it dropped as null, in the same position on every engine, and
  # max() over an NA reports the entire vector as non-comparable. Null on ONE side
  # only is a genuine shape disagreement and still returns NA, and a vector that
  # is null on both sides in every position has nothing left to compare.
  both_null <- is.na(x) & is.na(y)
  if (any(both_null)) {
    if (all(both_null)) return(NA_real_)
    x <- x[!both_null]; y <- y[!both_null]
  }
  s <- pmax(abs(x), abs(y), 1e-12)
  max(ifelse(s <= atol, 0, abs(x - y) / s))
}

# The port gate's metric: same kernel on both sides, so the honest expectation is
# bit-identity and TOL$port_rel = 1e-12 is diagnostic at exactly 0. Any absolute
# grace would launder a wiring bug into a pass, so this one keeps the pure ratio.
# compare.R's two port blocks call this and nothing else -- change together.
port_rel_max <- function(x, y) rel_max(x, y, atol = 0)

# Torn-line tolerant (kill -9 watchdog can truncate the final line) -- a run
# killed mid-write still yields every complete record before the tear.
# Namespaced fromJSON so this file works when sourced on its own, without a
# caller that has already attached jsonlite.
read_jsonl <- function(path) {
  lines <- readLines(path); lines <- lines[nzchar(lines)]
  recs <- list()
  for (ln in lines) {
    rec <- tryCatch(jsonlite::fromJSON(ln, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(rec)) recs[[length(recs) + 1L]] <- rec
  }
  setNames(recs, vapply(recs, `[[`, "", "cell"))
}

# Manifest cells keyed by cell id. simplifyDataFrame = FALSE keeps `cells` a
# list of per-cell lists, so `cell$family` etc. read off one cell; the default
# would collapse the array into a data.frame.
grid_cells_by_id <- function(path) {
  m <- jsonlite::fromJSON(path, simplifyDataFrame = FALSE)
  setNames(m$cells, vapply(m$cells, `[[`, "", "cell"))
}

# ONE SPELLING for a grouping factor's name, and the key every varcomp block is
# ordered and paired by. An interaction grouping is written three ways across the
# engines -- glmm `g1:g2`, lme4 and glmmTMB `g2:g1`, MixedModels `g1 & g2` -- and
# the same factors in any order are the same grouping, so the label is split on
# `:` and `&`, each factor trimmed, then sorted and rejoined with `:`. Without it
# a sort on the raw label puts the same two blocks in opposite orders on the two
# sides (measured on Oats, Pastes, Arabidopsis, cake and six sim_* cells), and a
# positional comparison then reads one engine's first block against the other's
# second.
canon_group <- function(x) {
  vapply(strsplit(x, "[:&]"),
         function(parts) paste(sort(trimws(parts)), collapse = ":"), "")
}

# varcomp -> stddev vector and off-diagonal correlations, flattened across
# grouping factors in CANONICAL NAME order (the deterministic join key; glmm
# records g1,g2,..., an oracle the same factors under its own spelling).
# Positional alignment within a group is the term order both engines emit
# ((Intercept), slope, ...), which varcomp_keys checks.
stddevs_of <- function(rec) {
  vc <- rec$varcomp
  if (is.null(vc) || length(vc) == 0) return(numeric(0))
  if (is.data.frame(vc)) vc <- lapply(seq_len(nrow(vc)), function(i) as.list(vc[i, ]))
  vc <- vc[order(canon_group(vapply(vc, function(g) g$group, "")))]
  unlist(lapply(vc, function(g) as.numeric(unlist(g$stddev))))
}

# The join key stddevs_of and corrs_of flatten against: one entry per grouping
# factor in canonical name order, each carrying that group's term names with the
# cosmetic label formatting removed (MixedModels writes a contrast `Machine: B`
# where glmm writes `MachineB`). Two records whose keys differ are describing
# different covariance blocks, or the same blocks with different terms in them, so
# their flattened vectors are not comparable coordinate by coordinate even when
# they are the same length.
varcomp_keys <- function(rec) {
  vc <- rec$varcomp
  if (is.null(vc) || length(vc) == 0) return(character(0))
  if (is.data.frame(vc)) vc <- lapply(seq_len(nrow(vc)), function(i) as.list(vc[i, ]))
  groups <- canon_group(vapply(vc, function(g) g$group, ""))
  vc <- vc[order(groups)]; groups <- sort(groups)
  paste(groups, vapply(vc, function(g)
    paste(norm_coef(unlist(g$terms)), collapse = ","), ""), sep = "|")
}

# Off-diagonal correlations (upper triangle) flattened across groups, in the same
# canonical name order. Scalar groups ([[1]]) contribute nothing.
corrs_of <- function(rec) {
  vc <- rec$varcomp
  if (is.null(vc) || length(vc) == 0) return(numeric(0))
  if (is.data.frame(vc)) vc <- lapply(seq_len(nrow(vc)), function(i) as.list(vc[i, ]))
  vc <- vc[order(canon_group(vapply(vc, function(g) g$group, "")))]
  unlist(lapply(vc, function(g) {
    m <- g$corr
    if (is.null(m)) return(numeric(0))
    # After the data.frame round-trip above, corr is a length-1 LIST wrapping
    # the k x k matrix; as.matrix on that gives a 1x1 list-matrix and the
    # nrow<2 guard silently drops every correlation. Unwrap first.
    if (is.list(m)) m <- m[[1]]
    m <- as.matrix(m); if (nrow(m) < 2) return(numeric(0))
    m[upper.tri(m)]
  }))
}
