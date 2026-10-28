# Formula syntax

The same Rust parser (`glmm::formula`, the `formula` cargo feature, on by
default) backs the formula string wherever `glmm` takes one: Rust callers
using `glmm::formula::lower`, Python's `glmm.fit`, and R's `fastglmm`. It
follows R-style conventions — `~` separates response from predictors, `+`
adds terms, `:` and `*` build interactions — plus a small, closed whitelist of
R's function-call syntax: `log()`/`sqrt()`/`exp()`/`I(x^k)` on one bare column,
`offset()`, `cbind()` on the response, and `zerocorr()` around a random-effect
term.

## Accepted

| Construct | Meaning | Example |
|---|---|---|
| Bare column name | A fixed-effect main effect | `x1` |
| `+` | Add another term | `x1 + x2` |
| `:` | Interaction only (no main effects added) | `x1:x2` |
| `*` | Desugars to main effects + interaction | `x1*x2` → `x1 + x2 + x1:x2` |
| `( … )^k`, `k ≥ 2` | Every interaction of the terms inside, up to order `k`, as in R | `(a + b + c)^2` → `a + b + c + a:b + a:c + b:c` |
| `( … )` in the fixed part | Groups terms, so `*` and `:` distribute over the sum inside, as in R | `a*(b + c)` → `a + b + c + a:b + a:c`; `a:(b + c)` → `a:b + a:c` |
| `A/B` (random-effects grouping only, not the fixed part) | Nesting: random intercept for `A` + nested random intercept for `A:B` | `(1 \| a/b)` |
| `1` | The intercept, written out. It is always there unless removed, so `1 + x` and `x` are the same model | `1 + x` |
| `- 1`, `0 +` | Drops the fixed-effect intercept | `x - 1`, `0 + x` |
| `log(x)`, `sqrt(x)`, `exp(x)` | Transform of one bare column, used as its own design column | `log(x)` |
| `I(x^k)`, `k ≥ 2` | Integer power of one bare column | `I(x^2)` |
| `offset(expr)` | A known additive term on the linear-predictor scale; `expr` is a bare column or one of the transforms above | `offset(log(exposure))` |
| `cbind(successes, failures) ~ …` | Aggregated-binomial response — both arguments must be columns (compute `failures` first, e.g. `size - incidence`; arithmetic inside `cbind()` is not accepted). Lowered to a proportion + trial-count weights | `cbind(incidence, failures) ~ period + (1 \| herd)` |
| `(1 \| g)` | Random intercept for grouping factor `g` | `(1 \| group)` |
| `(1 + x \| g)` | Correlated random intercept + slope on `x` for `g` | `(1 + x \| group)` |
| `(x \|\| g)` | Uncorrelated random intercept and slope: one term per variable, as lme4 expands it | `(1 + x \|\| group)` is `(1 \| group) + (0 + x \| group)` |
| `zerocorr(… \| g)` | Uncorrelated random effects over every column, as in MixedModels.jl | `zerocorr(1 + x \| group)` |
| `(0 + x \| g)`, `(-1 + x \| g)`, `(x - 1 \| g)`, `(x + 0 \| g)` | A random slope without a random intercept | `(0 + x \| group)` |
| several terms on one grouping | Independent terms: no correlation between two terms | `(1 + x1 \| g) + (0 + x2 \| g)` |
| additional `(… \| g2)` terms | More grouping factors — crossed or nested | `(1 \| g) + (1 \| g2)` |

`A/B` nesting on the random-effects side also works: `(1 \| A/B)` yields a
random intercept for `A` and a nested random intercept for `A:B`. A grouping
factor can also be the combination of two existing factor columns crossed
together: `(1 \| A:B)`.

A random slope can be a transform too: `(1 + log(x) | g)` works the same way
a bare-column slope does.

A random-slope variable need not be a fixed term. `y ~ 1 + (x | g)` fits a
random slope on `x` with no fixed effect for `x`, as lme4 does. So does
`y ~ x:f + (x | g)`, where `x` is in the fixed part only inside an
interaction. A factor works the same way: in `y ~ 1 + (f | g)` the slope
reads the factor's K − 1 treatment dummies. Such a variable gets no
coefficient.

`^` follows R's `terms()`: `(a + b)^3` is `a + b + a:b`, since there is no
third term to add, and `a^2` is `a`. `I(a^2)` stays the power of a column.
The power must be a whole number of 2 or more.

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

### Uncorrelated and intercept-free random effects

The terms on one grouping are independent of each other: `(1 | g) + (0 + x |
g)` and `(x || g)` are the same model, with no correlation between the
intercept and the slope. `(1 || g)` is `(1 | g)`. `||` and `zerocorr()` take
one plain grouping factor; on a nested `A/B` or a crossed `A:B` grouping they
are a syntax error, as a random slope there is. All random-effect columns of
one grouping are reported as one block, intercept term first, then the other
terms as written.

On a numeric slope `||` and `zerocorr()` are the same. **On a factor slope
they differ.** A factor `f` with levels `a`, `b`, `c` is coded with K − 1
dummy columns (`fb`, `fc`) beside the intercept, so its random-effect block is
`(Intercept), fb, fc`:

- `(f || g)` follows lme4: it expands to `(1 | g) + (0 + f | g)`, and the
  factor's block stays unstructured, every correlation in it estimated.
  `(0 + f | g)` is fitted as `(1 + f | g)`, the same model in this coding,
  and an intercept-only `(1 | g)` beside it is merged into it. lme4's fit of
  `(f || g)` has one more variance, which its full set of indicator columns
  leaves unidentified, so the deviance and the fixed effects agree and the
  variance components are laid out differently.
- `zerocorr(f | g)` makes every column uncorrelated: the intercept, `fb` and
  `fc` each get their own variance and no correlation. `zerocorr(0 + f | g)`
  is a syntax error, since the K − 1 coding cannot write a diagonal over the
  factor's levels without the intercept; write `zerocorr(1 + f | g)`.

A column in two terms on one grouping is an error, after factor coding:
`(1 | g) + (1 + x | g)` repeats the intercept, and so does `(1 + x | g) +
(0 + f | g)` with a factor `f`, because the factor's term holds the intercept.
Write `(1 + x + f | g)` for one correlated block, or `(x + f || g)` for an
uncorrelated slope beside the factor's block.

A random slope without a random intercept costs one random effect per level
more than its columns: the kernel keeps the intercept column at zero variance
and hides it from every output. Adaptive quadrature (`nagq > 1`) takes at
most 3 random effects in its one grouping, so it is reached one slope sooner:
`(0 + x1 + x2 | g)` runs it, `(0 + x1 + x2 + x3 | g)` falls back to the
Laplace approximation with the `agq_fallback` warning
([`warnings.md`](warnings.md)). The same holds for the size limits that send a
wide model to the sparse solver.

## Not accepted (and the workaround)

| Construct | Error behavior | Workaround |
|---|---|---|
| `poly(x, 2)`, nested calls (`log(log(x))`), arithmetic inside a call (`log(x+1)`), two-argument calls | Clear parse error — the transform whitelist accepts exactly one bare column name inside `log()`/`sqrt()`/`exp()`/`I(x^k)`, nothing more | Compute the column yourself and pass it as a plain column, or as `I(x^k)`/`log(x)`/etc. if it fits that whitelist |
| `1` together with `- 1` or `0 +` (e.g. `y ~ 1 + x - 1`, or `(1 + x - 1 \| g)` in a random-effect term) | Clear error — the formula both writes the intercept and removes it. R decides this by term order; this parser does not guess | Keep only the one you mean |
| A column in two terms on one grouping (e.g. `(1 \| g) + (1 + x \| g)`, or `(1 + x \| g) + (0 + f \| g)` with a factor `f`) | Clear error naming the column and the grouping | Put each column in one term: `(1 + x + f \| g)` or `(x + f \|\| g)` |
| `\|\|` or `zerocorr()` on a nested or crossed grouping (`(x \|\| a/b)`, `(x \|\| a:b)`) | Clear parse error | Not available: a random slope on `a/b` or `a:b` is not supported either; only a random intercept, `(1 \| a/b)` or `(1 \| a:b)` |
| A random slope on an interaction or nested grouping (`(1 + x \| a:b)`, `(x \| a/b)`) | Clear parse error naming the term | Not available: only a random intercept, `(1 \| a:b)` or `(1 \| a/b)` |
| A grouping of three or more factors (`(1 \| a/b/c)`, `(1 \| a:b:c)`) | Clear parse error naming the term | Not available: a grouping takes at most two factors, `a:b` or `a/b` |
| A random-effect term written right after a name (`foo(1 \| g)`, or `zerocor(1 + x \| g)` misspelling `zerocorr`) | Clear parse error naming the term | Put a `+` between the name and the term, or fix the spelling |
| `zerocorr(0 + f \| g)` on a factor `f` | Clear error: the K − 1 coding cannot write it | `zerocorr(1 + f \| g)` |
| `.` | Clear error — `.` (all other columns) is not a valid identifier | Spell out the predictors explicitly |
| `1`, `0` or `- 1` inside parentheses (e.g. `a*(b + 1)`), or `-` term removal inside them (`a*(b - c)`) | Clear parse error | Write the intercept, or its removal, outside the parentheses: `a*b - 1` |
| A `^` power that is not a whole number of 2 or more (e.g. `(a + b)^1`) | Clear parse error | Leave out `^1`; `(a + b)^2` and up cross the terms |
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
