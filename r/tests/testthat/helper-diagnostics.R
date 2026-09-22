# Shared assertions for the warning/condition tests in test-methods.R. Not a
# data builder, so it stays separate from helper-benchmark.R.

# The three "pins a variance component at the boundary" tests below all check
# the same lme4 text with only the pinned component's name differing, and all
# follow it with isSingular(fit). `component` is a regex fragment such as
# "sd\\(x \\| g\\)"; `info` names the case in a failure message.
expect_pinned_boundary_fit <- function(fit_fn, component, info = component) {
  msg <- paste0("boundary \\(singular\\) fit: see help\\('isSingular'\\); ",
                component, " pinned at the variance boundary")
  fit <- NULL
  expect_warning(fit <- fit_fn(), msg, info = info)
  expect_true(isSingular(fit), info = info)
  fit
}

# The four `.warn_note()` regression tests in test-methods.R all call it with
# the same converged/beta/aliased arguments; only `note` varies per case.
note_condition <- function(note) {
  tryCatch(fastglmm:::.warn_note(note, character(0), TRUE, 1, FALSE),
           warning = identity)
}
