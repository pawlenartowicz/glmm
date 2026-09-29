# Formula syntax

The same Rust parser (`glmm::formula`, the `formula` cargo feature, on by
default) backs the formula string wherever `glmm` takes one: Rust callers
using `glmm::formula::lower`, Python's `glmm.fit`, and R's `fastglmm`. It
follows R-style conventions — `~` separates response from predictors, `+`
adds terms, `:` and `*` build interactions — plus a small, closed whitelist of
R's function-call syntax: `log()`/`sqrt()`/`exp()`/`I(x^k)` on one bare column,
`offset()`, and `cbind()` on the response.

## Accepted

| Construct | Meaning | Example |
|---|---|---|
| Bare column name | A fixed-effect main effect | `x1` |
| `+` | Add another term | `x1 + x2` |
| `:` | Interaction only (no main effects added) | `x1:x2` |
| `*` | Desugars to main effects + interaction | `x1*x2` → `x1 + x2 + x1:x2` |
| `A/B` (random-effects grouping only, not the fixed part) | Nesting: random intercept for `A` + nested random intercept for `A:B` | `(1 \| a/b)` |
| `1` | The intercept, written out. It is always there unless removed, so `1 + x` and `x` are the same model | `1 + x` |
| `- 1`, `0 +` | Drops the fixed-effect intercept | `x - 1`, `0 + x` |
| `log(x)`, `sqrt(x)`, `exp(x)` | Transform of one bare column, used as its own design column | `log(x)` |
| `I(x^k)`, `k ≥ 2` | Integer power of one bare column | `I(x^2)` |
| `offset(expr)` | A known additive term on the linear-predictor scale; `expr` is a bare column or one of the transforms above | `offset(log(exposure))` |
| `cbind(successes, failures) ~ …` | Aggregated-binomial response — both arguments must be columns (compute `failures` first, e.g. `size - incidence`; arithmetic inside `cbind()` is not accepted). Lowered to a proportion + trial-count weights | `cbind(incidence, failures) ~ period + (1 \| herd)` |
| `(1 \| g)` | Random intercept for grouping factor `g` | `(1 \| group)` |
| `(1 + x \| g)` | Correlated random intercept + slope on `x` for `g` | `(1 + x \| group)` |
| additional `(… \| g2)` terms | More grouping factors — crossed or nested | `(1 \| g) + (1 \| g2)` |

`A/B` nesting on the random-effects side also works: `(1 \| A/B)` yields a
random intercept for `A` and a nested random intercept for `A:B`. A grouping
factor can also be the combination of two existing factor columns crossed
together: `(1 \| A:B)`.

A random slope can be a transform too: `(1 + log(x) | g)` works the same way
a bare-column slope does.

Dropping the intercept follows R's coding: the first factor's main effect
gets a dummy column per level (no reference level), and any later factor
keeps ordinary treatment contrasts. `y ~ 0 + (1 | g)` — no fixed term at all
— is a clear error: the design would have zero columns.

A transform's argument is exactly one bare column name — no arithmetic, no
nesting, no second argument. A row where the transform is non-finite (e.g.
`log(x)` at `x <= 0`) is a clear error naming the term and the row, not a
silently dropped observation.

`offset()` may appear at most once in the formula. Passing both a formula
`offset()` term and the `offset=`/`offset` argument is a clear error asking
you to use one.

`cbind()` requires `family = binomial`; using it with any other family is a
clear error saying so. Using it together with `weights=` is a clear error
asking you to use one. A row whose trial count (`successes + failures`) is
not a positive, finite number is a clear error naming the row.

## Not accepted (and the workaround)

| Construct | Error behavior | Workaround |
|---|---|---|
| `poly(x, 2)`, nested calls (`log(log(x))`), arithmetic inside a call (`log(x+1)`), two-argument calls | Clear parse error — the transform whitelist accepts exactly one bare column name inside `log()`/`sqrt()`/`exp()`/`I(x^k)`, nothing more | Compute the column yourself and pass it as a plain column, or as `I(x^k)`/`log(x)`/etc. if it fits that whitelist |
| `1` together with `- 1` or `0 +` (e.g. `y ~ 1 + x - 1`) | Clear error — the formula both writes the intercept and removes it. R decides this by term order; this parser does not guess | Keep only the one you mean |
| `(x || g)` | Clear error — the double-pipe form (uncorrelated random effects) matches none of the random-effect patterns and is rejected as invalid syntax | Not available; random slopes are always fit with a full correlation structure via `(1 + x | g)` |
| `(0 + x | g)` | Clear error — intercept suppression inside a random-effects term is not supported | Not available; a random slope always carries a random intercept: `(x | g)` or `(1 + x | g)` |
| `.` | Clear error — `.` (all other columns) is not a valid identifier | Spell out the predictors explicitly |
| `contrasts=` | Not an argument the R or Python wrapper accepts | Relevel the factor (`relevel()` in R; reorder before building a `pandas.Categorical`) so the level you want as base sorts first |

Every one of these is a clear error at parse or fit time, never a silent
reinterpretation of the formula.

## Factor coding

Factors are coded with treatment contrasts: the base level is the column's
**first level**. A plain string column carries no declared order, so it is
sorted lexicographically — the same rule R's `factor()` uses. A
`pandas.Categorical` (Python) or an R `factor()` with an explicit level order
keeps that order, and its first level becomes the base.

In an interaction, a factor keeps all its levels unless the rest of the term
(the term without that factor) is empty or is part of an earlier term, with
terms ordered by degree. This is the rule R's `model.matrix` uses: `y ~ x:f`
gives `x:fa x:fb …`, and `y ~ f + f:g` gives `fa:gq fb:gq …`.

## Column order and names

The design columns, and so the coefficients, come in R's order: the
intercept, then the main effects, then the two-way interactions, then the
three-way ones, and so on. Within one degree, terms keep the order they are
written in. An interaction is named after its variables in the order they
first appear in the fixed part of the formula, not in the order the term
spells them, and the first variable's levels vary fastest:

| Formula | Columns |
|---|---|
| `y ~ x*z + f` | `(Intercept) x z fb fc x:z` |
| `y ~ f:g + f` | `(Intercept) fb fc fa:gq fb:gq fc:gq` |
| `y ~ x + f:x` | `(Intercept) x x:fb x:fc` |
| `y ~ a*b*c*d` | `… a:b a:c b:c a:d b:d c:d …`, as R expands `((a*b)*c)*d` |

A term counts once however it is spelled: `g:f` after `f:g` adds nothing,
and a variable repeated inside a term counts once (`x:x` is `x`). Variables
that appear only in random-effect terms take no part in the order, as in
lme4. Read coefficients by name rather than by position.

When two columns are exactly collinear, the later one is dropped and its
coefficient reported as missing (`NaN` in Rust and Python, `NA` in R), so the
order decides which one: with `xz` holding `x * z`, `y ~ x:z + xz` keeps `xz`
and drops `x:z`, as lme4 does.

## Language notes

**Python.** `data` is duck-typed as a dict of column arrays — anything
column-addressable works, so `pandas.DataFrame`, `polars.DataFrame`, and
`pyarrow.Table` all work with no dataframe dependency in the package itself.

**R.** R's `Gamma()` family object defaults to `link = "inverse"`, and that R
semantics wins when you pass the object: `fastglmm(y ~ x, data,
family = Gamma())` fits with `link = "inverse"`. Passing the string
`"gamma"` instead uses the glmm port's own default, `link = "log"`.
