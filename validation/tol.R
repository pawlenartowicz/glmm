TOL <- list(
  beta_rel        = 1e-3,   # fixed effects: relative
  stddev_rel      = 1e-3,   # varcomp std-devs: relative
  loglik_abs_lmm  = 2e-6,   # LMM REML criterion: near-exact across engines (~1e-9 typical).
                            #   Measured worst 1.23e-6 on sim_max_q_slope (q=8, 36 theta params,
                            #   the corpus's largest LMM covariance; ~6e-10 relative — MixedModels
                            #   sits 5e-7 from lme4 on the same rung), so 2e-6 = measured worst +
                            #   margin, same convention as se_hessian_rel below.
  loglik_abs_glmm = 1e-3,   # GLMM Laplace logLik: two optimizers land ~3e-6 relative
                            #   apart on the same surface (beta/varcomp confirm same fit)
  se_rel          = 1e-3,   # LMM SE + method-matched GLMM RX: tight (same method, all engines)
  se_hessian_rel  = 1e-3,   # GLMM Hessian pair (lme4 vs glmm), same band as se_rel. History:
                            #   this sat at 3e-2 while the frozen oracle carried lme4's lagged-
                            #   ldL2 tolPwrss artifact (~1.3%: glmer's Xwts run one PIRLS
                            #   iteration behind the mode). The 2026-07-04 references
                            #   are artifact-free (lme4.R tolPwrss=1e-13, recorded per JSON) and
                            #   glmm's FD runs PIRLS at its tight FD-only tol, so the engines
                            #   agree to worst 2e-5 (grouseticks); 1e-3 = measured-worst + the
                            #   same ~margin se_rel carries over ITS measured worst.
  stddev_se_rel   = 3e-3,   # GLMM RE-stddev SE (lme4 numDeriv vs glmm single-step FD, both on
                            #   the joint (theta,beta) Hessian theta block). Same artifact
                            #   history as se_hessian_rel (was 3e-2). Against the artifact-free
                            #   oracle the worst gap is 8e-4 (sim_sparse_poisson) -- the
                            #   single-step-FD vs numDeriv-Richardson method floor on the theta
                            #   block, noisier than the beta block, hence the wider band.

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
  # defect `agq_corr_abs` below was created for -- a relative difference has no
  # meaning once both sides are at zero -- but reached from the other direction:
  # there the whole quantity lives near zero, here a single coordinate does.
  #
  # The case: glmm PINS a variance component to a hard 0.0 while an oracle's
  # optimizer stops on its own residue (1e-5-ish). Both engines say "this
  # component is zero"; `rel_max`'s 1e-12 denominator floor scores it exactly
  # 1.0, and no residue is small enough to escape -- shrinking the oracle's
  # residue shrinks the denominator with it. Two fixtures were already built
  # AROUND this rather than through it: sim_binomial_zerosd's seed is frozen on
  # a draw where BOTH engines return bit-exact 0.0 (prep/gen_large_theta_data.R),
  # and fit_glmm_binomial_zerosd_is_pinned asserts bit-equality instead of a
  # band (src/fit/glmm_tests.rs). Both stay as they are -- they assert something
  # stronger than this floor and lose nothing by it.
  #
  # MEASURED (2026-08-06) over the estimate-grid campaign's 510 cells, every
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

  # Deviance gate (dev = -2*loglik, dev_align.R's convention). Measured
  # 2026-08-24 over the full 48-rung curated corpus by measure_dev_floor.R.
  # ONE value, not split per
  # family: corpus-wide max benign |Δdev| (beta/se/stddev all in-band vs lme4)
  # is 1.6507e-05 on sim_sparse_binomial_bigsd (rung 46, the sparse large-θ̂
  # rung that already carries its own tightened se_hessian_rel override
  # below) -- every OTHER benign binomial rung sits at 9e-9..4.7e-6 (rung 27,
  # sim_binomial_slope2, is the true second-highest), the same band as
  # gaussian/poisson, so the apparent per-family gap is that one rung's own
  # known-harder numerics, not a family-wide floor.
  dev_eps = 2e-4,  # ceil-to-one-sig-fig(10 x 1.6507e-05), a ~10x margin
  dev_big = 0.5    # unmoved: corpus-wide benign max (1.65e-5) and the largest
                   #   documented-divergence Δdev (1.56e-4, sim_sparse_gamma's
                   #   basin split) sit 4.5 and 3.5 orders of magnitude below it
                   #   respectively; a convention-mismatch constant is itself
                   #   tens of units (the nAGQ saturated-loglik deficit
                   #   dev_align.R corrects), so 0.5 stays clear of both ends
)

# ── per-rung tolerance overrides ─────────────────────────────────────────────
# TOL above is corpus-wide: one band per quantity, calibrated on the measured
# worst across every rung. That is the right default (a band nobody can point at
# a rung for is not calibrated), but it cannot express a rung whose engines agree
# far better than the corpus worst and whose whole reason for existing is that
# tighter agreement. `se_hessian_rel` is the case: the crate documents <= 2e-5
# measured glmm-vs-lme4 agreement (src/glmm/se.rs) while the corpus-wide band is
# 1e-3, so a rung added to guard that documented agreement cannot see a 25x
# violation of it.
#
# Keyed by manifest dataset name (compare.R's `name`), then by the TOL key the
# override replaces. A rung lists ONLY the quantities it tightens; everything
# else falls through to TOL.
#
# Not a place to widen: an override that LOOSENS a band is a tolerance relaxed to
# make an engine pass, which the corpus forbids outright. Overrides tighten, and
# `validate_tol_per_rung` below enforces that rather than only asking for it.
#
# BOTH KEY LEVELS ARE VALIDATED AT LOAD -- see `validate_tol_per_rung`. Do not
# add an entry expecting `tol_for` to complain about a typo: it cannot, which is
# exactly why the check is where it is.
#
# ── se_hessian_rel = 3e-5 on the two large-theta rungs ───────────────────────
# Populated 2026-07-30, after the FD-theta-step fix landed (the step-construction
# comment in src/glmm/se.rs). These are the only two rungs in the corpus whose
# random-effect SD is well above 1 -- 4.5106 and 2.9716 against a corpus that
# otherwise tops out at 1.34 -- and they exist to guard exactly the agreement the
# corpus-wide 1e-3 cannot see.
#
# MEASURED, both sides of the fix, glmm vs the frozen lme4 references in
# validation/results/lme4_simulated/ (glmer, tolPwrss = 1e-13), as
# rel_max over the three se_hessian coordinates:
#
#     rung                 pre-fix     post-fix    at 3e-5
#     sim_binomial_bigsd   1.2680e-5   7.3886e-6   passes before AND after
#     sim_poisson_bigsd    3.8526e-5   1.4439e-5   FAILS before, passes after
#
# 3e-5 = ceil-to-one-significant-figure(2 x 1.4439e-5), the measured post-fix
# worst with the ~2x margin every measured band above carries. The fail-before
# evidence for sim_poisson_bigsd is arithmetic, not a re-run: 3.8526e-5 > 3e-5.
#
# BOTH RUNGS GET THE SAME VALUE ON PURPOSE. Sized per rung, sim_binomial_bigsd
# would take ceil1(2 x 7.3886e-6) = 2e-5; it is the same class of rung, added by
# the same spec, and a band differing by one significant figure between two such
# rungs invites drift. Note also what a per-rung 2e-5 would NOT buy: no honestly
# sized band makes sim_binomial_bigsd fail before the fix, because its pre-fix
# 1.2680e-5 is already below sim_poisson_bigsd's POST-fix 1.4439e-5 -- the two
# intervals do not overlap, so the coverage spec's "R1 and R2 fail before and
# pass after" holds for sim_poisson_bigsd only.
#
# NOT TIGHTER THAN THIS. Of sim_poisson_bigsd's 1.4439e-5 post-fix gap, only
# 7.17e-6 is ours: post-fix glmm sits within 7.2e-6 of its own h->0 stencil limit
# on every moved rung, while lme4's vcov(use.hessian=TRUE) is `lme4:::deriv12` at
# an ABSOLUTE delta = 1e-4 and carries 5e-6..9e-6 of its own FD error (two runs
# of lme4's own stencil differ by 4.5e-7..1.8e-6). 3e-5 is a band around the
# REFERENCE's reproducibility. Tightening it would pin a number lme4 cannot
# itself reproduce.
# ── se_hessian_rel = 3e-4 on sim_sparse_binomial_bigsd (rung 46) ────────────
# Populated at freeze (rung added to cover the sparse solver's large-theta-hat
# behavior, which rungs 44-45 above never exercised -- both are dense-routed
# scalar (1|g)). RECORDED BASELINE, not a fail-before/pass-after band: nothing
# in the change that added this rung touched the sparse FD Hessian's theta
# step, so the rung passes at freeze by construction. The number below exists
# so a later recalibration of that step can say whether it moved agreement in
# the right direction.
#
# MEASURED at freeze, glmm vs the frozen lme4 reference in
# validation/results/lme4_simulated/sim_sparse_binomial_bigsd.json (glmer,
# nAGQ = 1, tolPwrss = 1e-13), as rel_max over each quantity's coordinates:
#
#     quantity     glmm vs glmer at freeze
#     beta         6.410e-4
#     se_hessian   1.086e-4
#     stddev       8.749e-5
#     stddev_se    1.662e-4   (gated by the corpus-wide TOL$stddev_se_rel = 3e-3)
#
# The sparse FD Hessian's step at freeze is SPARSE_FD_STEP_REL = 1e-4 scaled
# by max(1, |gamma-hat|) per component, and this fixture's theta-hat vector is
# one large component (g1, 3.9076) beside seven small ones (0.25..0.67), so
# both sides of that clamp are exercised in a single fit -- which is why this
# rung is the one to read after a step change.
#
# BAND = ceil-to-one-significant-figure(2 x 1.086e-4) = 3e-4, the same rule
# the two dense bigsd rungs above use. Wider than their 3e-5 because this is a
# different fixture family (Bernoulli, seven crossed extras, sparse route) with
# its own measured agreement, not a copy of theirs.
#
# NOT TIGHTER THAN THIS. lme4's own vcov(use.hessian=TRUE) is
# `lme4:::deriv12` at an ABSOLUTE delta = 1e-4 and carries 5e-6..9e-6 of its
# own FD error run to run. 3e-4 stays well clear of that floor.
TOL_PER_RUNG <- list(
  sim_binomial_bigsd       = list(se_hessian_rel = 3e-5),
  sim_poisson_bigsd        = list(se_hessian_rel = 3e-5),
  sim_sparse_binomial_bigsd = list(se_hessian_rel = 3e-4)
)

# The band for one quantity on one rung: the rung's own override when it has one,
# else the corpus-wide TOL value. `rung` is a manifest dataset name; an unknown
# name is not an error (it simply has no override -- and `validate_tol_per_rung`
# has already proved every name in the table IS a rung), but an unknown `quantity`
# is -- a typo there would otherwise gate against NULL and silently pass.
tol_for <- function(rung, quantity) {
  band <- TOL[[quantity]]
  if (is.null(band)) stop(sprintf("tol_for: no such tolerance `%s`", quantity))
  ov <- TOL_PER_RUNG[[rung]]
  if (!is.null(ov) && !is.null(ov[[quantity]])) return(ov[[quantity]])
  band
}

# Directory this file lives in -- and therefore where manifest.json is, the two
# being siblings. `source()` records the path it is reading in the sourcing frame's
# `ofile`, which is the only thing a sourced file can learn its own location from:
# the working directory is not an anchor (compare.R sources this by an absolute
# path, the campaign analyze scripts by a relative one) and `--file=` names the
# SOURCER, not this file. Frames are walked innermost-first so a nested source()
# still resolves to the file actually being read.
#
# CALL THIS DURING LOAD. The `ofile` frames exist only while the source() call is
# on the stack, which is when its one caller (`validate_tol_per_rung` below) runs.
# Called later the frame walk finds nothing and it falls back to the SOURCER's
# directory, accepted only when a manifest.json actually sits there -- true for
# compare.R and engines/lme4.R, false for the campaign scripts two directories
# down, which is why the fallback is guarded rather than trusted.
tol_suite_dir <- function() {
  for (i in rev(seq_len(sys.nframe()))) {
    of <- sys.frame(i)$ofile
    if (is.character(of) && length(of) == 1L && file.exists(of)) {
      return(dirname(normalizePath(of)))
    }
  }
  arg <- grep("--file=", commandArgs(FALSE), value = TRUE)
  if (length(arg) == 1L) {
    dir <- dirname(normalizePath(sub("--file=", "", arg)))
    if (file.exists(file.path(dir, "manifest.json"))) return(dir)
  }
  stop("tol_suite_dir: cannot locate tol.R's own directory, so manifest.json ",
       "cannot be read to validate TOL_PER_RUNG. Call this from inside the ",
       "source() that loads tol.R, not afterwards")
}

# Every TOL_PER_RUNG key, both levels, checked against reality when this file is
# sourced. Called at the bottom of this block; returns invisibly.
#
# WHY LOAD-TIME AND NOT INSIDE `tol_for`. A mistyped key is indistinguishable from
# "this rung has no override": `tol_for` finds nothing and returns the flat TOL
# band, so the gate runs LOOSER than intended and the rung goes green having
# checked nothing. That is the one failure mode a tolerance table must not have,
# and it is silent at both levels -- an outer typo (`sim_binomial_bigsdd`) misses
# the rung, an inner typo (`se_hessian` for `se_hessian_rel`) misses the quantity,
# and neither is visible in the output. `tol_for` cannot detect it even in
# principle, and a check there would only fire for the rungs a given run happens
# to include. So the whole table is validated here, on every source(), regardless
# of which rungs run.
#
# NO-OP WHILE THE TABLE IS EMPTY: an empty list has nothing to check, so nothing
# is read, no manifest is opened and no jsonlite dependency is taken. Every
# existing sourcer of this file therefore behaves exactly as it did before this
# block existed; the manifest read begins the first time an override is added,
# which is precisely when it is needed.
validate_tol_per_rung <- function() {
  if (length(TOL_PER_RUNG) == 0L) return(invisible(TRUE))

  rungs <- names(TOL_PER_RUNG)
  if (is.null(rungs) || any(is.na(rungs)) || !all(nzchar(rungs))) {
    stop("validate_tol_per_rung: every TOL_PER_RUNG entry must be NAMED with a ",
         "rung name; found an unnamed or empty-named entry")
  }
  dup <- unique(rungs[duplicated(rungs)])
  if (length(dup) > 0) {
    # `[[` returns the FIRST match, so a duplicated name silently discards the
    # second entry's overrides -- the same silent-loss class as a typo.
    stop(sprintf("validate_tol_per_rung: duplicated rung name(s) in TOL_PER_RUNG: %s",
                 paste(sprintf("`%s`", dup), collapse = ", ")))
  }

  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("validate_tol_per_rung: TOL_PER_RUNG is non-empty but jsonlite is not ",
         "available to read manifest.json and check the rung names")
  }
  manifest_path <- file.path(tol_suite_dir(), "manifest.json")
  if (!file.exists(manifest_path)) {
    stop(sprintf("validate_tol_per_rung: manifest.json not found at %s", manifest_path))
  }
  man <- jsonlite::fromJSON(manifest_path, simplifyDataFrame = FALSE)
  # Rungs AND goldens: compare.R keys on dataset names, but a per-rung band is a
  # sensible thing to want on a golden too, and accepting only one of the two lists
  # would make the other look like a typo.
  known <- c(vapply(man$datasets, `[[`, "", "name"),
             vapply(man$m3_goldens, `[[`, "", "name"))
  unknown <- setdiff(rungs, known)
  if (length(unknown) > 0) {
    stop(sprintf(paste0("validate_tol_per_rung: TOL_PER_RUNG names no such rung ",
                        "or golden: %s -- these would silently fall through to ",
                        "the flat TOL band (manifest.json knows %d names)"),
                 paste(sprintf("`%s`", unknown), collapse = ", "), length(known)))
  }

  for (rung in rungs) {
    ov <- TOL_PER_RUNG[[rung]]
    if (!is.list(ov)) {
      stop(sprintf("validate_tol_per_rung: `%s` must map to a list of overrides, got %s",
                   rung, class(ov)[1]))
    }
    quantities <- names(ov)
    if (length(ov) == 0 || is.null(quantities) || !all(nzchar(quantities))) {
      stop(sprintf("validate_tol_per_rung: `%s` has an empty or unnamed override list",
                   rung))
    }
    for (quantity in quantities) {
      band <- TOL[[quantity]]
      if (is.null(band)) {
        stop(sprintf(paste0("validate_tol_per_rung: `%s` overrides no such ",
                            "tolerance `%s` -- it would silently fall through to ",
                            "the flat TOL band"), rung, quantity))
      }
      value <- ov[[quantity]]
      if (!is.numeric(value) || length(value) != 1L || !is.finite(value) || value <= 0) {
        stop(sprintf(paste0("validate_tol_per_rung: `%s`$`%s` must be a single ",
                            "finite positive number, got %s"),
                     rung, quantity, paste(format(value), collapse = " ")))
      }
      # The "overrides tighten" rule, enforced rather than merely documented: a
      # per-rung band LOOSER than the corpus-wide one is a tolerance relaxed for a
      # single rung, which is the thing the whole corpus forbids -- and it is just
      # as invisible in the output as a typo.
      if (value > band) {
        stop(sprintf(paste0("validate_tol_per_rung: `%s`$`%s` = %s is LOOSER than ",
                            "the corpus-wide TOL$%s = %s. Overrides tighten; ",
                            "widening a band for one rung is not a per-rung band"),
                     rung, quantity, format(value), quantity, format(band)))
      }
    }
  }
  invisible(TRUE)
}
validate_tol_per_rung()

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
  s <- pmax(abs(x), abs(y), 1e-12)
  max(ifelse(s <= atol, 0, abs(x - y) / s))
}

# The port gate's metric: same kernel on both sides, so the honest expectation is
# bit-identity and TOL$port_rel = 1e-12 is diagnostic at exactly 0. Any absolute
# grace would launder a wiring bug into a pass, so this one keeps the pure ratio.
# compare.R's two port blocks call this and nothing else -- change together.
port_rel_max <- function(x, y) rel_max(x, y, atol = 0)

# Shared join helpers (campaigns/estimate-grid/analyze.R, the speed-grid
# analyze.R/counters.R scripts + verify_boundary37.R -- change together).

# Torn-line tolerant (kill -9 watchdog can truncate the final line) -- same
# discipline as analyze_grid.R's reader.
read_jsonl <- function(path) {
  lines <- readLines(path); lines <- lines[nzchar(lines)]
  recs <- list()
  for (ln in lines) {
    rec <- tryCatch(fromJSON(ln, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(rec)) recs[[length(recs) + 1L]] <- rec
  }
  setNames(recs, vapply(recs, `[[`, "", "case_id"))
}

# Manifest cells keyed by case_id. simplifyDataFrame = FALSE keeps `cells` a
# list of per-cell lists, so `cell$family` etc. read off one cell; the default
# would collapse the array into a data.frame.
manifest_cells <- function(path) {
  m <- fromJSON(path, simplifyDataFrame = FALSE)
  setNames(m$cells, vapply(m$cells, `[[`, "", "case_id"))
}

# varcomp -> stddev vector and off-diagonal correlations, flattened across
# grouping factors in NAME order (deterministic join key; glmm records g1,g2,...,
# oracle records the same factor names). Positional alignment within a group is
# the term order both engines emit ((Intercept), slope, ...).
stddevs_of <- function(rec) {
  vc <- rec$varcomp
  if (is.null(vc) || length(vc) == 0) return(numeric(0))
  if (is.data.frame(vc)) vc <- lapply(seq_len(nrow(vc)), function(i) as.list(vc[i, ]))
  vc <- vc[order(vapply(vc, function(g) g$group, ""))]
  unlist(lapply(vc, function(g) as.numeric(unlist(g$stddev))))
}
# Off-diagonal correlations (upper triangle) flattened across groups, name-order.
# Scalar groups ([[1]]) contribute nothing.
corrs_of <- function(rec) {
  vc <- rec$varcomp
  if (is.null(vc) || length(vc) == 0) return(numeric(0))
  if (is.data.frame(vc)) vc <- lapply(seq_len(nrow(vc)), function(i) as.list(vc[i, ]))
  vc <- vc[order(vapply(vc, function(g) g$group, ""))]
  unlist(lapply(vc, function(g) {
    m <- g$corr
    if (is.null(m)) return(numeric(0))
    # After the data.frame round-trip above, corr is a length-1 LIST wrapping
    # the k x k matrix; as.matrix on that gives a 1x1 list-matrix and the
    # nrow<2 guard silently dropped every correlation (dead corr gate, caught
    # in review 2026-07-14). Unwrap first.
    if (is.list(m)) m <- m[[1]]
    m <- as.matrix(m); if (nrow(m) < 2) return(numeric(0))
    m[upper.tri(m)]
  }))
}
