#' @details An error from the native code has the condition class
#'   `"fastglmm_error_<code>"`, `code` one of `invalid_argument` (a bad
#'   argument), `fit_error` (the data cannot be fitted), `compare_error` (two
#'   fits cannot be compared) and `internal` (a bug in fastglmm), so
#'   `tryCatch()` can select it without matching message text.
#' @keywords internal
#' @import stats
"_PACKAGE"

# Run glmm-bind's function `name` on the named list `args` through the one
# native entry point, raising its error here as an R error of class
# "fastglmm_error_<code>".
.glmm_call <- function(name, args) {
  r <- glmm_call(name, args)
  if (!is.null(r$error)) {
    stop(errorCondition(r$error, class = paste0("fastglmm_error_", r$code), call = NULL))
  }
  r$value
}
