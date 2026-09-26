#!/usr/bin/env julia
# MixedModels.jl oracle for the accuracy grid: one JSONL record per cell,
# appended to GRID_OUT, in the schema every engine in the grid emits
# field-for-field (cell/engine/engine_version/converged/singular/status/
# message/coef_names/beta/se_rx/varcomp/sigma/nb_theta/loglik/deviance/n_eval/
# wall_seconds/fits_per_sample), plus se_hessian on a non-gaussian GLM cell, where
# GLM.jl's one SE method fills both slots. Adapted from
# validation/campaigns/speed-grid/fit.jl, which is where the manifest read, the
# resume scan, the construct-then-fit! shape, varcomp_of, the has_nonfinite guard,
# the per-cell try/catch and the append-and-flush writer come from.
#
# Run inside the pinned env:
#   julia --project=<grid> <grid>/engines/mixedmodels.jl

using MixedModels, GLM, CSV, DataFrames, JSON3, LinearAlgebra, Statistics

LinearAlgebra.BLAS.set_num_threads(1)

const VERSION = string(pkgversion(MixedModels))

grid_dir = get(ENV, "GRID_DIR", normpath(joinpath(@__DIR__, "..")))
manifest_path = get(ENV, "GRID_MANIFEST", joinpath(grid_dir, "manifest.json"))
out_path = get(ENV, "GRID_OUT", joinpath(grid_dir, "results.jsonl"))
only_ids = filter(!isempty, split(get(ENV, "GRID_CELLS", ""), ","))

# GRID_TIMED protocol, identical across all seven engines: ""/"0" means
# untimed; else the sample count, an integer >= 2. Errors on a malformed value
# rather than silently running untimed.
function timed_samples()
    v = strip(get(ENV, "GRID_TIMED", ""))
    (isempty(v) || v == "0") && return nothing
    n = tryparse(Int, v)
    (n === nothing || n < 2) &&
        error("GRID_TIMED must be 0 or an integer >= 2 (got \"$v\")")
    n
end
const TIMED = timed_samples()

mkpath(dirname(out_path))

manifest = JSON3.read(read(manifest_path, String))
done = Set{String}()
isfile(out_path) && for l in eachline(out_path)
    isempty(l) && continue
    try push!(done, String(JSON3.read(l).cell)) catch end  # kill -9 can truncate the last line
end

const OK_RETURN = Set([:FTOL_REACHED, :XTOL_REACHED, :SUCCESS, :STOPVAL_REACHED])

# JSON has no NaN/Inf literal -- JSON3.write throws on one, which (unlike an
# exception raised inside fit_cell) is NOT caught by the try/catch in the main
# loop below, because it happens on the WRITE, after fit_cell has already
# returned successfully. A degenerate/near-boundary fit (e.g. a near-zero
# variance component) can leave stderror() or the log-likelihood non-finite
# without fit_cell ever throwing, so this is checked explicitly rather than
# relying on the catch.
has_nonfinite(x::Real) = !isfinite(x)
has_nonfinite(x::AbstractArray) = any(has_nonfinite, x)
has_nonfinite(x::NamedTuple) = any(has_nonfinite, values(x))
has_nonfinite(x) = false

# loglik and deviance, together and nowhere else, so the two fields can never
# disagree. deviance = -2*loglik, before any downstream alignment correction;
# both null when the loglik JSON3 would otherwise have to reject as non-finite.
loglik_pair(ll) = isfinite(ll) ? (loglik = ll, deviance = -2 * ll) :
                                 (loglik = nothing, deviance = nothing)

# Per-reterm Sigma = sigma^2 * lambda*lambda' on gaussian (m.sigma is the
# residual sd, lambda the relative covariance Cholesky factor); lambda*lambda'
# alone on every other family, because a GLMM reterm carries no residual scale
# of its own -- sigma folds into the family dispersion, held at 1 upstream of
# lambda. stddev = sqrt(diag Sigma), corr = D^-1 Sigma D^-1 -- the same
# group/terms/stddev/corr schema the Rust and R engines emit.
function varcomp_of(m, gaussian::Bool)
    map(m.reterms) do t
        sigma = gaussian ? m.σ^2 * t.λ * t.λ' : t.λ * t.λ'
        d = sqrt.(diag(sigma))
        corr = sigma ./ (d * d')
        (group = string(MixedModels.fname(t)), terms = t.cnames,
         stddev = collect(d), corr = [collect(r) for r in eachrow(corr)])
    end
end

# jl_formula carries an offset(...) term textually (the R/Julia dialect); the
# MixedModels.jl and GLM.jl constructors have no offset() formula term, so it
# is stripped here and passed through the offset= keyword instead -- the same
# "applied exactly once" rule the manifest documents for r_formula/jl_formula.
function strip_offset(formula_str, cell)
    haskey(cell, :offset_col) || return formula_str
    replace(formula_str, Regex("\\s*\\+\\s*offset\\($(cell.offset_col)\\)") => "")
end

# Trial counts (manifest `weights`, aggregated-binomial) or prior weights
# (`weights_col`) -- mutually exclusive per cell (gen_manifest.R asserts it),
# both reaching the fit through the same weights keyword. A trials cell also
# needs its `prop` response column synthesized here, because jl_formula's LHS
# is "prop", not the raw response count the CSV carries.
function cell_wts!(cell, df)
    if haskey(cell, :weights)
        w = Float64.(df[!, Symbol(cell.weights)])
        df.prop = Float64.(df[!, Symbol(cell.response)]) ./ w
        return w
    elseif haskey(cell, :weights_col)
        return Float64.(df[!, Symbol(cell.weights_col)])
    end
    nothing
end

cell_offset(cell, df) =
    haskey(cell, :offset_col) ? Float64.(df[!, Symbol(cell.offset_col)]) : nothing

# R-origin column names can carry dots (Arabidopsis's total.fruits); Julia's
# @formula macro reads the raw expression before StatsModels sees it, so a dot
# parses as getproperty, not part of an identifier. jl_formula already spells
# the sanitized (underscored) name, so the data is sanitized here to match.
function read_grid_cell(cell, grid_dir)
    df = CSV.read(joinpath(grid_dir, String(cell.data)), DataFrame)
    for f in cell.factors
        df[!, Symbol(f)] = string.(df[!, Symbol(f)])
    end
    rename!(df, [n => replace(string(n), "." => "_") for n in names(df) if occursin(".", string(n))])
    df
end

# GRID_TIMED protocol, run identically after the primary (untimed) fit that
# produced the record's estimates: `n` extra build+fit passes, discard the
# first, median of the rest. `nothing` (untimed) skips the loop entirely.
function timed_median(build_and_fit, n)
    n === nothing && return nothing
    samples = [@elapsed(build_and_fit()) for _ in 1:n]
    median(samples[2:end])
end

# Construct (never fit) the mixed model for one cell. weights/offset are
# passed uniformly to every branch rather than special-cased per family: the
# manifest never pairs an offset with a gaussian cell, and if it ever did,
# LinearMixedModel has no offset= keyword and would raise a clear
# MethodError -- caught by the per-cell try/catch below -- rather than the
# offset being silently dropped.
function build_mixed(cell, df, gaussian)
    f = eval(Meta.parse(strip_offset(String(cell.jl_formula), cell)))
    kw = Dict{Symbol,Any}()
    w = cell_wts!(cell, df)
    w !== nothing && (kw[:wts] = w)
    off = cell_offset(cell, df)
    off !== nothing && (kw[:offset] = off)
    fam = String(cell.family)
    if gaussian
        LinearMixedModel(f, df; kw...)
    elseif fam == "binomial"
        link = cell.link == "logit" ? LogitLink() : ProbitLink()
        GeneralizedLinearMixedModel(f, df, Binomial(), link; kw...)
    elseif fam == "poisson"
        GeneralizedLinearMixedModel(f, df, Poisson(); kw...)
    else
        # oracles_of restricts MixedModels to gaussian/binomial/poisson on the
        # mixed branch; anything else reaching here is a manifest/oracles_of
        # bug, and must fail loudly rather than being fitted as Poisson by the
        # old catch-all `else`.
        error("unsupported family/link for MixedModels' mixed-model branch: $fam/$(cell.link)")
    end
end

function fit_mixed_cell(cell, df, gaussian, timed)
    build() = build_mixed(cell, df, gaussian)
    dofit!(m) = gaussian ?
        fit!(m; REML = cell.reml === true, progress = false) :
        fit!(m; progress = false)
    m = dofit!(build())
    wall = timed_median(() -> dofit!(build()), timed)
    conv = m.optsum.returnvalue in OK_RETURN
    # MixedModels refuses `loglikelihood` on a REML fit (a REML criterion is
    # not a likelihood) and reports the -2*logLik `objective` instead; lme4's
    # `logLik` on a REML fit is that same restricted criterion / -2, so this
    # puts MixedModels on lme4's REML scale rather than a plain likelihood.
    ll = gaussian && cell.reml === true ? -objective(m) / 2 : loglikelihood(m)
    merge(
        (converged = conv, singular = issingular(m),
         status = conv ? "ok" : "engine-fail",
         message = conv ? nothing :
                   "not converged: optsum.returnvalue = $(m.optsum.returnvalue)",
         coef_names = collect(coefnames(m)), beta = collect(coef(m)),
         se_rx = collect(stderror(m)), varcomp = varcomp_of(m, gaussian),
         sigma = gaussian ? m.σ : nothing, nb_theta = nothing,
         n_eval = m.optsum.feval, wall_seconds = wall, fits_per_sample = 1),
        loglik_pair(ll))
end

# The GLM (no random effect) family/link pair MixedModels' `oracles_of` ever
# hands this branch: gaussian routes to `lm` before this is called; binomial
# (logit/probit) and poisson are the only families `oracles_of` lists
# MixedModels for on a fixed-only cell (never gamma, NB or inverse-Gaussian).
function glm_dist_link(cell)
    fam = String(cell.family)
    fam == "binomial" && return Binomial(), (cell.link == "logit" ? LogitLink() : ProbitLink())
    fam == "poisson" && return Poisson(), LogLink()
    error("unsupported GLM family for MixedModels' GLM.jl branch: $fam")
end

function build_glm(cell, df, fam)
    f = eval(Meta.parse(strip_offset(String(cell.jl_formula), cell)))
    kw = Dict{Symbol,Any}()
    w = cell_wts!(cell, df)
    w !== nothing && (kw[:wts] = w)
    off = cell_offset(cell, df)
    off !== nothing && (kw[:offset] = off)
    if fam == "gaussian"
        GLM.lm(f, df; kw...)
    else
        dist, link = glm_dist_link(cell)
        GLM.glm(f, df, dist, link; kw...)
    end
end

# A GLM (no random effect) cell: no optimizer summary to read a convergence
# code from, so `converged`/`singular` are the fixed values IRLS success
# implies -- the fit failing at all raises and is caught upstream as
# engine-fail. `message` names the routing rather than carrying error text on
# a clean fit, so a reader can tell a GLM.jl row from a mixed-model row
# without a second engine name to track.
function fit_glm_cell(cell, df, timed)
    fam = String(cell.family)
    gaussian = fam == "gaussian"
    build() = build_glm(cell, df, fam)
    m = build()
    wall = timed_median(build, timed)
    se = collect(stderror(m))
    base = (converged = true, singular = false, status = "ok", message = "GLM.jl",
            coef_names = collect(coefnames(m)), beta = collect(coef(m)),
            se_rx = se, varcomp = NamedTuple[],
            # `GLM.lm`/`GLM.glm` return a `StatsModels.TableRegressionModel`
            # wrapper; `dispersion` is defined on the bare `LinearModel` it
            # holds, not the wrapper itself (unlike coef/stderror/
            # loglikelihood, which StatsModels delegates through the wrapper
            # automatically).
            sigma = gaussian ? dispersion(m.model) : nothing, nb_theta = nothing,
            n_eval = nothing, wall_seconds = wall, fits_per_sample = 1)
    # GLM.jl has exactly one SE method, so a non-gaussian GLM cell fills BOTH
    # slots with it -- matching the Rust engine, which fits a fixed-only
    # non-gaussian cell under both WaldSe::Hessian and WaldSe::Rx and gets the
    # same numbers back either way. A gaussian GLM cell keeps the
    # single-profiled-SE convention every engine uses on a gaussian cell and
    # has no se_hessian key at all.
    rec = gaussian ? base : merge(base, (se_hessian = se,))
    merge(rec, loglik_pair(loglikelihood(m)))
end

function fit_cell(cell, grid_dir, timed)
    df = read_grid_cell(cell, grid_dir)
    is_glm = haskey(cell, :structure) && String(cell.structure) == "glm"
    if is_glm
        fit_glm_cell(cell, df, timed)
    else
        fit_mixed_cell(cell, df, String(cell.family) == "gaussian", timed)
    end
end

# Seeded on an engine-fail: se_rx is NOT included, matching the schema's
# "absent when the engine has none" rule -- it was never computed.
fail_record(msg = nothing) = (
    converged = false, singular = false, status = "engine-fail", message = msg,
    coef_names = String[], beta = Float64[], varcomp = NamedTuple[],
    sigma = nothing, nb_theta = nothing, loglik = nothing, deviance = nothing,
    n_eval = nothing, wall_seconds = nothing, fits_per_sample = 1)

# Cell lookup by id, and the fit order: GRID_CELLS's own order (manifest order
# when it is empty), never re-derived from the manifest. The runner's watchdog
# blames a timeout on the first cell of GRID_CELLS still missing from the
# output, so fitting in any other order fits the wrong cell first and the
# blame lands on the wrong one (mirrors run.sh's `next_missing` -- change
# together).
cells_by_id = Dict(String(c.cell) => c for c in manifest.cells)
ids = isempty(only_ids) ? [String(c.cell) for c in manifest.cells] : String.(only_ids)

open(out_path, "a") do io
    for cid in ids
        cid in done && continue
        cell = get(cells_by_id, cid, nothing)
        cell === nothing && error("unknown cell id not in manifest: $cid")
        "MixedModels" in String.(cell.oracles) || continue
        base = (cell = cid, engine = "MixedModels", engine_version = VERSION)
        rec = try
            fitted = fit_cell(cell, grid_dir, TIMED)
            if has_nonfinite(fitted)
                @warn "non-finite fit result, recording as engine-fail" cid
                # The reason belongs in the record, not only in an @warn on
                # stderr: the record is what compare.R reads, and a message of
                # `null` there leaves the failure unexplainable afterwards.
                merge(base, fail_record("non-finite fit result"))
            else
                merge(base, fitted)
            end
        catch err
            @warn "engine-fail" cid err = sprint(showerror, err)
            # FIRST LINE ONLY. showerror's remaining lines are the Julia
            # backtrace, whose absolute paths into the depot say nothing about the
            # cell and turn one record into a multi-line JSON field.
            merge(base, fail_record(first(split(sprint(showerror, err), '\n'))))
        end
        println(io, JSON3.write(rec))
        flush(io)   # line-per-fit flush: run.sh's watchdog watches mtime
    end
end
