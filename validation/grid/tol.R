TOL <- list(
  beta_rel        = 1e-3,   # fixed effects: relative. The corpus-wide cross-engine band,
                            #   not independently measured on this grid.
  stddev_rel      = 1e-3,   # varcomp std-devs: relative. The corpus-wide cross-engine
                            #   band, not independently measured on this grid.
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
  # The port drives the same kernel through PyO3 (fit_warm(start=NULL) IS fit_cold),
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
  # ci_ref_rel  the band the saved-reference checks gate at: tests/grid_reference.rs
  #          (the `fast` cells against the newest committed glmm run) and the R
  #          package's pins in r/tests/testthat/test-acceptance-pins.R. Both
  #          restate this number as a constant of their own -- change all three
  #          together.
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
  ci_ref_rel  = 1e-7
)

# ── truth-error band, per family ─────────────────────────────────────────────
# glmm fails a family when err_glmm > err_best + band. The band is TWICE THE
# STANDARD ERROR OF THE MEAN of the paired per-cell difference
# err_glmm(cell) - err_best(cell), both engines on the same data. The pairing is
# what makes it a band about the glmm-versus-oracle gap rather than about the
# spread between a 60-row and a 30000-row cell.
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

# ── per-cell tolerance overrides ─────────────────────────────────────────────
# TOL above is grid-wide: one band per quantity, calibrated on the measured
# worst across every cell. That is the right default (a band nobody can point at
# a cell for is not calibrated), but it cannot express a cell whose engines agree
# far better than the grid worst and whose whole reason for existing is that
# tighter agreement. `se_hessian_rel` is the case: the crate documents <= 2e-5
# measured glmm-vs-lme4 agreement (src/glmm/se.rs) while the grid-wide band is
# 1e-3, so a cell added to guard that documented agreement cannot see a 25x
# violation of it.
#
# Keyed by manifest cell id (compare.R's `cell`), then by the TOL key the
# override replaces. A cell lists ONLY the quantities it tightens; everything
# else falls through to TOL.
#
# Not a place to widen: an override that LOOSENS a band is a tolerance relaxed to
# make an engine pass, which the grid forbids outright. Overrides tighten, and
# `validate_tol_per_cell` below enforces that rather than only asking for it.
#
# BOTH KEY LEVELS ARE VALIDATED AT LOAD -- see `validate_tol_per_cell`. Do not
# add an entry expecting `tol_for` to complain about a typo: it cannot, which is
# exactly why the check is where it is.
TOL_PER_CELL <- list()

# The band for one quantity on one cell: the cell's own override when it has one,
# else the grid-wide TOL value. `cell_id` is a manifest cell name; an unknown
# name is not an error (it simply has no override -- and `validate_tol_per_cell`
# has already proved every name in the table IS a cell), but an unknown
# `quantity` is -- a typo there would otherwise gate against NULL and silently
# pass. An unmeasured band stops for the same reason: NA compares false, so
# gating against it passes everything.
tol_for <- function(cell_id, quantity) {
  band <- TOL[[quantity]]
  if (is.null(band)) stop(sprintf("tol_for: no such tolerance `%s`", quantity))
  if (is.na(band))
    stop(sprintf(paste0("tol_for: TOL$%s has not been measured yet. Run the pilot ",
                        "and write the number into grid/tol.R with its measurement."),
                 quantity))
  ov <- TOL_PER_CELL[[cell_id]]
  if (!is.null(ov) && !is.null(ov[[quantity]])) return(ov[[quantity]])
  band
}

# Directory this file lives in -- and therefore where manifest.json is, the two
# being siblings. `source()` records the path it is reading in the sourcing frame's
# `ofile`, which is the only thing a sourced file can learn its own location from:
# the working directory is not an anchor (compare.R sources this by an absolute
# path, other scripts by a relative one) and `--file=` names the SOURCER, not this
# file. Frames are walked innermost-first so a nested source() still resolves to
# the file actually being read, and the basename test keeps a nested source() of
# some other file from answering for this one.
#
# CALL THIS DURING LOAD. The `ofile` frames exist only while the source() call is
# on the stack, which is when its one caller (`validate_tol_per_cell` below) runs.
# Called later the frame walk finds nothing, and the only remaining anchor is
# `--file=`. With neither, this STOPS rather than guessing: a guessed directory
# reads the wrong manifest.json, or none, and validating cell ids against the
# wrong list is worse than not validating them.
tol_suite_dir <- function() {
  for (i in rev(seq_len(sys.nframe()))) {
    of <- sys.frame(i)$ofile
    if (is.character(of) && length(of) == 1L && file.exists(of) &&
        identical(basename(of), "tol.R")) {
      return(dirname(normalizePath(of)))
    }
  }
  arg <- grep("--file=", commandArgs(FALSE), value = TRUE)
  if (length(arg) == 1L) return(dirname(normalizePath(sub("--file=", "", arg))))
  stop("tol_suite_dir: cannot locate tol.R's own directory, so manifest.json ",
       "cannot be read to validate TOL_PER_CELL. Call this from inside the ",
       "source() that loads tol.R, not afterwards")
}

# Every TOL_PER_CELL key, both levels, checked against reality when this file is
# sourced. Called at the bottom of this block; returns invisibly.
#
# WHY LOAD-TIME AND NOT INSIDE `tol_for`. A mistyped key is indistinguishable from
# "this cell has no override": `tol_for` finds nothing and returns the flat TOL
# band, so the gate runs LOOSER than intended and the cell goes green having
# checked nothing. That is the one failure mode a tolerance table must not have,
# and it is silent at both levels -- an outer typo (`bina_int1_g3000p20_bal_basee`)
# misses the cell, an inner typo (`se_hessian` for `se_hessian_rel`) misses the
# quantity, and neither is visible in the output. `tol_for` cannot detect it even
# in principle, and a check there would only fire for the cells a given run
# happens to include. So the whole table is validated here, on every source(),
# regardless of which cells run.
#
# NO-OP WHILE THE TABLE IS EMPTY: an empty list has nothing to check, so nothing
# is read, no manifest is opened and no jsonlite dependency is taken. The manifest
# read begins the first time an override is added, which is precisely when it is
# needed.
validate_tol_per_cell <- function() {
  if (length(TOL_PER_CELL) == 0L) return(invisible(TRUE))

  cells <- names(TOL_PER_CELL)
  if (is.null(cells) || any(is.na(cells)) || !all(nzchar(cells))) {
    stop("validate_tol_per_cell: every TOL_PER_CELL entry must be NAMED with a ",
         "cell id; found an unnamed or empty-named entry")
  }
  dup <- unique(cells[duplicated(cells)])
  if (length(dup) > 0) {
    # `[[` returns the FIRST match, so a duplicated name silently discards the
    # second entry's overrides -- the same silent-loss class as a typo.
    stop(sprintf("validate_tol_per_cell: duplicated cell id(s) in TOL_PER_CELL: %s",
                 paste(sprintf("`%s`", dup), collapse = ", ")))
  }

  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("validate_tol_per_cell: TOL_PER_CELL is non-empty but jsonlite is not ",
         "available to read manifest.json and check the cell ids")
  }
  manifest_path <- file.path(tol_suite_dir(), "manifest.json")
  if (!file.exists(manifest_path)) {
    stop(sprintf("validate_tol_per_cell: manifest.json not found at %s", manifest_path))
  }
  man <- jsonlite::fromJSON(manifest_path, simplifyDataFrame = FALSE)
  known <- vapply(man$cells, `[[`, "", "cell")
  unknown <- setdiff(cells, known)
  if (length(unknown) > 0) {
    stop(sprintf(paste0("validate_tol_per_cell: TOL_PER_CELL names no such cell: ",
                        "%s -- these would silently fall through to the flat TOL ",
                        "band (manifest.json knows %d cells)"),
                 paste(sprintf("`%s`", unknown), collapse = ", "), length(known)))
  }

  for (cell_id in cells) {
    ov <- TOL_PER_CELL[[cell_id]]
    if (!is.list(ov)) {
      stop(sprintf("validate_tol_per_cell: `%s` must map to a list of overrides, got %s",
                   cell_id, class(ov)[1]))
    }
    quantities <- names(ov)
    if (length(ov) == 0 || is.null(quantities) || !all(nzchar(quantities))) {
      stop(sprintf("validate_tol_per_cell: `%s` has an empty or unnamed override list",
                   cell_id))
    }
    for (quantity in quantities) {
      band <- TOL[[quantity]]
      if (is.null(band)) {
        stop(sprintf(paste0("validate_tol_per_cell: `%s` overrides no such ",
                            "tolerance `%s` -- it would silently fall through to ",
                            "the flat TOL band"), cell_id, quantity))
      }
      value <- ov[[quantity]]
      if (!is.numeric(value) || length(value) != 1L || !is.finite(value) || value <= 0) {
        stop(sprintf(paste0("validate_tol_per_cell: `%s`$`%s` must be a single ",
                            "finite positive number, got %s"),
                     cell_id, quantity, paste(format(value), collapse = " ")))
      }
      # The "overrides tighten" rule, enforced rather than merely documented: a
      # per-cell band LOOSER than the grid-wide one is a tolerance relaxed for a
      # single cell, which is the thing the whole grid forbids -- and it is just
      # as invisible in the output as a typo.
      if (value > band) {
        stop(sprintf(paste0("validate_tol_per_cell: `%s`$`%s` = %s is LOOSER than ",
                            "the grid-wide TOL$%s = %s. Overrides tighten; ",
                            "widening a band for one cell is not a per-cell band"),
                     cell_id, quantity, format(value), quantity, format(band)))
      }
    }
  }
  invisible(TRUE)
}
validate_tol_per_cell()

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
