#!/usr/bin/env julia
# MixedModels.jl reference fits over the parity datasets -> results/mixedmodels/<dataset>.json.
#
# THE ORACLE IS SACRED. These JSONs are a frozen reference, the second of the two
# engines glmm is later held to. Never edit a result to make a downstream engine
# pass; regenerate only if the model SPEC is proven wrong, with a recorded reason.
#
# Run inside the pinned env so package versions match Manifest.toml:
#   julia --project=GLMM/parity GLMM/parity/oracle/fit.jl

using MixedModels, CSV, DataFrames, JSON3, Statistics, LinearAlgebra

const PARITY = normpath(joinpath(@__DIR__, ".."))
const N_RUNS = 10  # timing loop; first pass (JIT warm-up) discarded, min of rest reported

manifest = JSON3.read(read(joinpath(PARITY, "manifest.json"), String))
out_dir = joinpath(PARITY, "results", "mixedmodels")
mkpath(out_dir)

function read_dataset(spec)
    df = CSV.read(joinpath(PARITY, "data", string(spec.name, ".csv")), DataFrame)
    # Mirror fit.R: coerce grouping + categorical fixed-effect columns to String so
    # MixedModels treats them as categorical with DummyCoding; sorted-level base =
    # first level matches R's treatment-contrast base (asserted in compare.R).
    for f in spec.factors
        df[!, Symbol(f)] = string.(df[!, Symbol(f)])
    end
    df
end

# Build the fit thunk for one dataset. The binomial rung carries a two-column
# response (successes/total); MixedModels fits the proportion with per-row weights
# (`weights` field in the manifest), which live outside @formula -- so the response
# column `prop` is synthesized here from the manifest's weight column.
function fit_thunk(spec, df)
    f = eval(Meta.parse(String(spec.jl_formula)))
    fam = String(spec.family)
    if fam == "gaussian"
        () -> fit(MixedModel, f, df; REML = spec.reml === true, progress = false)
    elseif fam == "binomial"
        w = Float64.(df[!, Symbol(spec.weights)])
        df.prop = Float64.(df.incidence) ./ w
        () -> fit(MixedModel, f, df, Binomial(); wts = w, progress = false)
    elseif fam == "poisson"
        () -> fit(MixedModel, f, df, Poisson(); progress = false)
    else
        error("unsupported family: $fam")
    end
end

function time_fit(make_fit)
    times = Float64[]
    for _ in 1:N_RUNS
        push!(times, @elapsed make_fit())
    end
    (fit_seconds_min = minimum(@view times[2:end]), n_runs = N_RUNS, warmup_discarded = 1)
end

# VarCorr -> common representation: per grouping factor, RE term names, their
# standard deviations, and the correlation matrix. MixedModels exposes this through
# VarCorr(m).σρ (a NamedTuple per grouping with `.σ` stddevs and `.ρ` correlations),
# already on the absolute σ scale -- the same scale fit.R normalizes lme4 to.
# Only q in {1,2} occurs across rungs 1-6 (sleepstudy is the only q=2); a wider
# grouping would need the full ρ-ordering reconstruction, so we assert against it.
function varcomp_of(m)
    vc = VarCorr(m)
    out = Vector{Any}()
    for (grp, val) in pairs(vc.σρ)
        terms = collect(string.(keys(val.σ)))
        stddev = collect(Float64.(values(val.σ)))
        q = length(stddev)
        if q == 1
            corr = [[1.0]]
        elseif q == 2
            rho = Float64(only(val.ρ))
            corr = [[1.0, rho], [rho, 1.0]]
        else
            error("varcomp_of: grouping $grp has q=$q random effects; only q<=2 handled")
        end
        push!(out, (group = string(grp), terms = terms, stddev = stddev, corr = corr))
    end
    out
end

# MixedModels has no single converged bool; treat any NLopt success return as
# converged. The values below are the codes that mean "stopped because it found
# the optimum", as opposed to hitting an eval/time limit or failing.
const OK_RETURN = Set([:FTOL_REACHED, :XTOL_REACHED, :SUCCESS, :STOPVAL_REACHED])

function fit_one(spec)
    df = read_dataset(spec)
    make_fit = fit_thunk(spec, df)
    m = make_fit()
    gaussian = String(spec.family) == "gaussian"

    est = Dict{Symbol,Any}(
        :beta => collect(coef(m)),
        :se => collect(stderror(m)),
        :loglik => loglikelihood(m),          # logLik scale (not the -2logLik objective)
        :varcomp => varcomp_of(m),
    )
    gaussian && (est[:sigma] = sdest(m))       # residual std; GLMM rungs omit it

    res = Dict{Symbol,Any}(
        :dataset => string(spec.name), :engine => "mixedmodels",
        :engine_version => string(pkgversion(MixedModels)),
        :family => string(spec.family),
        :reml => gaussian ? (spec.reml === true) : nothing,
        :rung => spec.rung,
        :converged => m.optsum.returnvalue in OK_RETURN,
        :singular => issingular(m),
        :coef_names => collect(coefnames(m)),  # contrast-coding assertion vs lme4
        :estimates => est,
        :timing => time_fit(make_fit),
    )

    open(joinpath(out_dir, string(spec.name, ".json")), "w") do io
        JSON3.pretty(io, res)
    end
    println("mixedmodels  $(rpad(spec.name,12))  rung $(spec.rung)  ",
            "converged=$(res[:converged]) singular=$(res[:singular])  ",
            "fit_min=$(round(res[:timing].fit_seconds_min, sigdigits=4))s")
end

for spec in manifest.datasets
    fit_one(spec)
end
