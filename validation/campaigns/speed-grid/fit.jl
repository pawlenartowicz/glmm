#!/usr/bin/env julia
# MixedModels side of the optimizer-grid campaign -> one JSONL line per cell.
# Resume-safe (skips case_ids already in GRID_OUT); eval cap set per cell from
# the manifest's pre-registered max_fun via optsum.maxfeval. Run inside the
# pinned env: julia --project=GLMM/validation GLMM/validation/campaigns/speed-grid/fit.jl
using MixedModels, CSV, DataFrames, JSON3, LinearAlgebra

manifest_path = get(ENV, "GRID_MANIFEST", joinpath(@__DIR__, "manifest.json"))
out_path = get(ENV, "GRID_OUT", joinpath(@__DIR__, "results", "mixedmodels_shipped.jsonl"))
tag = get(ENV, "GRID_CONFIG_TAG", "")
only_ids = Set(filter(!isempty, split(get(ENV, "GRID_ONLY", ""), ",")))
mkpath(dirname(out_path))

manifest = JSON3.read(read(manifest_path, String))
done = Set{String}()
isfile(out_path) && for l in eachline(out_path)
    isempty(l) && continue
    try push!(done, String(JSON3.read(l).case_id)) catch end  # kill -9 can truncate the last line
end

const OK_RETURN = Set([:FTOL_REACHED, :XTOL_REACHED, :SUCCESS, :STOPVAL_REACHED])

# Per-reterm Σ = σ² λλᵀ (λ = relative covariance Cholesky factor, m.σ = residual
# sd), then stddev = sqrt(diag Σ), corr = D⁻¹ΣD⁻¹ — same schema as fit.R's
# varcomp_of (mirrors lme4's VarCorr: group/terms/stddev/corr) so
# analyze_diligent.R's stddevs_of/corrs_of read this field unmodified. GLM
# reterms carry no residual scale (σ folds into the family dispersion, held at
# 1 upstream of λ) — only called on gaussian LMM cells (boundary-37 spec, Part
# A), so unconditionally uses m.σ.
function varcomp_of(m)
    map(m.reterms) do t
        sigma = m.σ^2 * t.λ * t.λ'
        d = sqrt.(diag(sigma))
        corr = sigma ./ (d * d')
        (group = string(MixedModels.fname(t)), terms = t.cnames,
         stddev = collect(d), corr = [collect(r) for r in eachrow(corr)])
    end
end

function fit_cell(cell)
    df = CSV.read(joinpath(@__DIR__, "data", string(cell.case_id, ".csv")), DataFrame)  # campaign-local, prep.R output
    for f in cell.factors
        df[!, Symbol(f)] = string.(df[!, Symbol(f)])
    end
    f = eval(Meta.parse(String(cell.jl_formula)))
    fam = String(cell.family)
    # construct-then-fit! so the pre-registered eval cap lands in optsum first
    build() = begin
        m = if fam == "gaussian"
            LinearMixedModel(f, df)
        elseif fam == "binomial"
            if haskey(cell, :weights) && cell.weights !== nothing
                w = Float64.(df[!, Symbol(cell.weights)])
                df.prop = Float64.(df.incidence) ./ w
                GeneralizedLinearMixedModel(f, df, Binomial(); wts = w)
            else
                GeneralizedLinearMixedModel(f, df, Binomial())
            end
        else
            GeneralizedLinearMixedModel(f, df, Poisson())
        end
        m.optsum.maxfeval = Int(cell.max_fun)
        m
    end
    # REML iff the manifest cell says so (gaussian grid cells carry reml=true;
    # the glmm engine's LMM path is REML-only) — mirrors fit.jl's `spec.reml === true`.
    dofit!(m) = fam == "gaussian" ?
        fit!(m; REML = get(cell, :reml, false) === true, progress = false) :
        fit!(m; progress = false)
    # Per-cell warm-up: Julia specializes fit! per formula-shape type, so the
    # first fit of each shape JIT-compiles into its wall time. Fit once and
    # discard, then time a second fit on a freshly built model (hot code, cold
    # state; deterministic optimizer ⇒ identical n_eval/beta/deviance).
    # compile_seconds is recorded as PROOF the timed fit was hot — analysis
    # hard-stops on nonzero — never as a subtraction/correction.
    dofit!(build())
    # wall_seconds = model build + θ-solve, so it stays comparable to glmm's
    # wall (which folds its own O(N) suff-stats build into the fit). The two are
    # no longer reported apart — the build/solve split diagnostic was retired.
    build_seconds = @elapsed (m = build())
    t = @timed dofit!(m)
    conv = m.optsum.returnvalue in OK_RETURN
    status = m.optsum.returnvalue == :MAXEVAL_REACHED ? "maxeval" :
             conv ? "ok" : "engine-fail"
    # θ̂ capture (mismatch-adjudication spec): m.θ is the concatenated
    # column-major lower-triangle vech per reterm, MM's term order (descending
    # level count — NOT formula order); re_groups/re_qs make the mapping to
    # glmm's declaration-order layout explicit downstream.
    (optimizer = string(m.optsum.optimizer), n_eval = m.optsum.feval,
     converged = conv, singular = issingular(m), deviance = objective(m),
     beta = collect(coef(m)), se = collect(stderror(m)),
     theta = collect(m.θ),
     re_groups = [string(MixedModels.fname(t)) for t in m.reterms],
     re_qs = [size(t.λ, 1) for t in m.reterms],
     varcomp = fam == "gaussian" ? varcomp_of(m) : NamedTuple[],
     status = status,
     wall_seconds = build_seconds + t.time, compile_seconds = t.compile_time)
end

open(out_path, "a") do io
    for cell in manifest.cells
        cid = String(cell.case_id)
        (!isempty(only_ids) && !(cid in only_ids)) && continue
        cid in done && continue
        base = (case_id = cid, seed = cell.seed, engine = "mixedmodels",
                config_tag = tag)
        rec = try
            merge(base, fit_cell(cell))
        catch err
            @warn "engine-fail" cid err = sprint(showerror, err)
            merge(base, (optimizer = "", n_eval = 0, converged = false,
                         singular = false, deviance = nothing,
                         beta = Float64[], se = Float64[],
                         theta = Float64[], re_groups = String[], re_qs = Int[],
                         varcomp = NamedTuple[],
                         status = "engine-fail", wall_seconds = 0.0))
        end
        println(io, JSON3.write(rec))
        flush(io)   # line-per-fit flush: the watchdog watches mtime
    end
end
