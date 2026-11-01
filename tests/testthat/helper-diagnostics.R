# Shared assertions for the warning/condition tests in test-methods.R. Not a
# data builder, so it stays separate from helper-benchmark.R.

# The boundary-fit tests below all check the same singular-fit text with only
# the pinned component's name differing, and all follow it with
# isSingular(fit). `component` is a regex fragment such as
# "\\(Intercept\\) in g"; `info` names the case in a failure message.
expect_pinned_boundary_fit <- function(fit_fn, component, info = component) {
  msg <- paste0("^Caution: Singular fit\\. The random effects are too complex .* Affected: .*",
                component)
  fit <- NULL
  expect_warning(fit <- fit_fn(), msg, info = info)
  expect_true(isSingular(fit), info = info)
  fit
}
