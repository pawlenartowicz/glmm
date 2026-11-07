#' @keywords internal
#' @import stats
"_PACKAGE"

# Run glmm-bind's function `name` on the named list `args` through the one
# native entry point, raising its error here as an R error (a refusal of
# anova(m0, m1) with this port's pointer to compare(), `.pair_refusal`).
.glmm_call <- function(name, args) {
  r <- glmm_call(name, args)
  if (!is.null(r$error)) stop(.pair_refusal(r$error, r$kind), call. = FALSE)
  r$value
}
