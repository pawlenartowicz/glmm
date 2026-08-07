# Formula syntax

The same Rust parser (`glmm::formula`, the `formula` cargo feature, on by
default) backs the formula string wherever `glmm` takes one: Rust callers
using `glmm::formula::lower`, Python's `glmm.fit`, and R's `fastglmm`. It
follows R-style conventions — `~` separates response from predictors, `+`
adds terms, `:` and `*` build interactions — but only accepts bare column
names, not R's function-call syntax inside a formula.

## Accepted

| Construct | Meaning | Example |
|---|---|---|
| Bare column name | A fixed-effect main effect | `x1` |
| `+` | Add another term | `x1 + x2` |
| `:` | Interaction only (no main effects added) | `x1:x2` |
| `*` | Desugars to main effects + interaction | `x1*x2` → `x1 + x2 + x1:x2` |
| `A/B` | Nesting: `A` main effect + `A:B` interaction | `a/b` → `a + a:b` |
| `(1 \| g)` | Random intercept for grouping factor `g` | `(1 \| group)` |
| `(1 + x \| g)` | Correlated random intercept + slope on `x` for `g` | `(1 + x \| group)` |
| additional `(… \| g2)` terms | More grouping factors — crossed or nested | `(1 \| g) + (1 \| g2)` |

`A/B` nesting on the random-effects side also works: `(1 \| A/B)` yields a
random intercept for `A` and a nested random intercept for `A:B`. A grouping
factor can also be the combination of two existing factor columns crossed
together: `(1 \| A:B)`.

## Not accepted (and the workaround)

| Construct | Error behavior | Workaround |
|---|---|---|
| `log(x)`, `I(x^2)`, `poly(x, 2)` | Clear parse error — the parser only accepts bare identifiers, not function calls | Compute the transformed column yourself and pass it as a plain column |
| `cbind(s, f)` | Clear error — the response is looked up as a literal column name, so a call expression is never found in the data table | Pass the proportion as the response column and the trial count as `weights=` |
| `- 1`, `0 +` | Clear error — the parser has no intercept-suppression or term-removal support on the fixed-effects side | Not available; the model always carries an intercept |
| bare `1` (e.g. `y ~ 1 + x`) | Clear error — the parser only accepts bare identifiers as fixed-effect terms, and `1` is not one; the intercept is always carried implicitly, so there is no term for it to spell | Drop the explicit `1 +`; write `y ~ x` |
| `(x || g)` | Clear error — the double-pipe form (uncorrelated random effects) matches none of the random-effect patterns and is rejected as invalid syntax | Not available; random slopes are always fit with a full correlation structure via `(1 + x | g)` |
| `offset()` | Clear error — same as any other function-call syntax, rejected as an invalid identifier | Not available as formula syntax; both the Python and R ports take it as an `offset=`/`offset` argument instead (see `tutorial-python.md`/`tutorial-r.md` §2) |
| `.` | Clear error — `.` (all other columns) is not a valid identifier | Spell out the predictors explicitly |
| `contrasts=` | Not an argument the R or Python wrapper accepts | Relevel the factor (`relevel()` in R; reorder before building a `pandas.Categorical`) so the level you want as base sorts first |

Every one of these is a clear error at parse or fit time — never a silent
reinterpretation of the formula.

## Factor coding

Factors are coded with treatment contrasts: the base level is the column's
**first level**. A plain string column carries no declared order, so it is
sorted lexicographically — the same rule R's `factor()` uses. A
`pandas.Categorical` (Python) or an R `factor()` with an explicit level order
keeps that order, and its first level becomes the base.

## Language notes

**Python.** `data` is duck-typed as a dict of column arrays — anything
column-addressable works, so `pandas.DataFrame`, `polars.DataFrame`, and
`pyarrow.Table` all work with no dataframe dependency in the package itself.

**R.** R's `Gamma()` family object defaults to `link = "inverse"`, and that R
semantics wins when you pass the object: `fastglmm(y ~ x, data,
family = Gamma())` fits with `link = "inverse"`. Passing the string
`"gamma"` instead uses the glmm port's own default, `link = "log"`.
