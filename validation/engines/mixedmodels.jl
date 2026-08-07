#!/usr/bin/env julia
# MixedModels.jl reference fits over the validation datasets -> results/mixedmodels_{empirical,simulated}/<dataset>.json.
#
# THE ORACLE IS SACRED. These JSONs are a frozen reference, the second of the two
# engines glmm is later held to. Never edit a result to make a downstream engine
# pass; regenerate only if the model SPEC is proven wrong, with a recorded reason.
#
# Run inside the pinned env so package versions match Manifest.toml:
#   julia --project=GLMM/validation GLMM/validation/engines/mixedmodels.jl

using MixedModels, CSV, DataFrames, JSON3, Statistics, LinearAlgebra

const SUITE_DIR = normpath(joinpath(@__DIR__, ".."))
# Timing is OPT-IN and its sample count lives in run.sh, not here.
#
# THE contract, mirrored in glmm.rs / lme4.R / glmm_python.py / glmm_r.R
# `timing_runs` -- five languages that cannot share code, so change together:
# VALIDATION_TIMINGS unset / "" / "0" means do not time (`timing` is written null,
# which fmt_scalar already emits for `nothing`); otherwise it IS the sample count, an
# integer >= 2, first pass (JIT warm-up) discarded, median of the rest. run.sh
# validates the value; this errors rather than silently skipping timing when the
# engine is run by hand with a malformed one.
#
# Each result JSON still records its own n_runs, so files timed under any count --
# including the old hardcoded 10 and its 100-run predecessor -- stay self-describing.
function timing_runs()
    v = strip(get(ENV, "VALIDATION_TIMINGS", ""))
    (isempty(v) || v == "0") && return nothing
    n = tryparse(Int, v)
    (n === nothing || n < 2) && error(
        "VALIDATION_TIMINGS must be 0 or an integer >= 2 (got \"$v\"); " *
        "N=2 keeps 1 sample after the warm-up discard")
    n
end
const N_RUNS = timing_runs()  # nothing on an untimed run

manifest = JSON3.read(read(joinpath(SUITE_DIR, "manifest.json"), String))
data_dir_of(spec) = joinpath(SUITE_DIR, "data", String(spec.source) == "sim" ? "simulated" : "empirical")
out_dir_of(spec)  = joinpath(SUITE_DIR, "results",
                             string("mixedmodels_", String(spec.source) == "sim" ? "simulated" : "empirical"))
mkpath(joinpath(SUITE_DIR, "results", "mixedmodels_empirical"))
mkpath(joinpath(SUITE_DIR, "results", "mixedmodels_simulated"))

function read_dataset(spec)
    # `data` field: CSV to read when it differs from the rung name (mirrors lme4.R) --
    # a re-linked rung (cbpp_probit) reuses the committed dataset byte-for-byte.
    src_name = haskey(spec, :data) ? spec.data : spec.name
    df = CSV.read(joinpath(data_dir_of(spec), string(src_name, ".csv")), DataFrame)
    # Mirror lme4.R: coerce grouping + categorical fixed-effect columns to String so
    # MixedModels treats them as categorical with DummyCoding; sorted-level base =
    # first level matches R's treatment-contrast base (asserted in compare.R).
    for f in spec.factors
        df[!, Symbol(f)] = string.(df[!, Symbol(f)])
    end
    # R-origin column names can carry dots (e.g. Arabidopsis's `total.fruits`) --
    # Julia's @formula macro parses the raw expression with Julia's own reader
    # before StatsModels sees it, so `total.fruits` reads as a getproperty
    # expression, not an identifier. Sanitize to underscores here; jl_formula
    # (manifest.json) must reference the same underscored name, r_formula keeps
    # the original dot (R's own dialect handles dots in names natively).
    rename!(df, [n => replace(string(n), "." => "_") for n in names(df) if occursin(".", string(n))])
    df
end

# Build the fit thunk for one dataset. The binomial rung carries a two-column
# response (successes/total); MixedModels fits the proportion with per-row weights
# (`weights` field in the manifest), which live outside @formula -- so the response
# column `prop` is synthesized here from the manifest's weight column.
function fit_thunk(spec, df)
    f = eval(Meta.parse(String(spec.jl_formula)))
    fam = String(spec.family)
    # Prior weights (weights suite `weights_col` rungs): MixedModels' wts=
    # kwarg, the counterpart of lme4's weights=. Distinct from the
    # aggregated-binomial `weights` field handled in the binomial branch.
    wcol = haskey(spec, :weights_col) ? Float64.(df[!, Symbol(spec.weights_col)]) : nothing
    # Known per-row additive linear-predictor offset (manifest `offset` key,
    # e.g. a log-exposure column) -- MixedModels' `offset=` kwarg on the
    # GeneralizedLinearMixedModel constructor (confirmed supported since
    # v3.4.1; GLMM only, not LMM). Mirrors wcol's plain named-column lookup.
    ocol = haskey(spec, :offset) ? Float64.(df[!, Symbol(spec.offset)]) : nothing
    if fam == "gaussian"
        wcol === nothing ?
            (() -> fit(MixedModel, f, df; REML = spec.reml === true, progress = false)) :
            (() -> fit(MixedModel, f, df; REML = spec.reml === true, wts = wcol, progress = false))
    elseif fam == "binomial"
        # `link` field: non-canonical link override (cbpp_probit, mirrors lme4.R).
        # Absent = the canonical logit, the pre-existing behavior.
        lnk = haskey(spec, :link) ? String(spec.link) : "logit"
        L = lnk == "logit" ? LogitLink() :
            lnk == "probit" ? ProbitLink() : error("unsupported binomial link: $lnk")
        # Aggregated binomial (cbind(successes, failures) ~ ...): synthesize the
        # proportion response from the manifest's weight column, as before. Plain
        # per-row binary binomial (VerbAgg: y in {0,1}, no `weights` field) has no
        # aggregation to undo -- fit the formula's own 0/1 response directly.
        if haskey(spec, :weights)
            w = Float64.(df[!, Symbol(spec.weights)])
            df.prop = Float64.(df.incidence) ./ w
            () -> fit(MixedModel, f, df, Binomial(), L; wts = w, progress = false)
        else
            () -> fit(MixedModel, f, df, Binomial(), L; progress = false)
        end
    elseif fam == "poisson"
        if wcol === nothing && ocol === nothing
            () -> fit(MixedModel, f, df, Poisson(); progress = false)
        elseif ocol === nothing
            () -> fit(MixedModel, f, df, Poisson(); wts = wcol, progress = false)
        elseif wcol === nothing
            () -> fit(MixedModel, f, df, Poisson(); offset = ocol, progress = false)
        else
            () -> fit(MixedModel, f, df, Poisson(); wts = wcol, offset = ocol, progress = false)
        end
    elseif fam == "gamma"
        # Explicit LogLink: the manifest's gamma rungs pin link "log" (Gamma's
        # canonical inverse link is unstable on these designs, mirroring lme4.R).
        () -> fit(MixedModel, f, df, Gamma(), LogLink(); progress = false)
    else
        error("unsupported family: $fam")
    end
end

# Times `batch` fits per sample so sub-resolution fits stay above the timer floor
# (mirrors lme4.R). fit_seconds_median is the median time for `fits_per_sample` fits.
function time_fit(make_fit; batch = 1)
    times = Float64[]
    for _ in 1:N_RUNS
        t = @elapsed for _ in 1:batch
            make_fit()
        end
        push!(times, t)
    end
    (fit_seconds_median = median(@view times[2:end]), n_runs = N_RUNS,
     warmup_discarded = 1, fits_per_sample = batch)
end

# Timing for one dataset. MixedModels' only SE is the conditional/Rx vcov (no Hessian
# variant), so on GLMM rungs its single timing IS the Rx timing -- relabel the field
# `fit_seconds_median_rx` to line up with the glmm/lme4 Rx/Hessian split. Gaussian
# rungs keep the single `fit_seconds_median`.
function time_one(spec, make_fit)
    N_RUNS === nothing && return nothing   # untimed run: no loops at all
    t = time_fit(make_fit; batch = get(spec, :timing_batch, 1))
    String(spec.family) == "gaussian" && return t
    (fit_seconds_median_rx = t.fit_seconds_median, n_runs = t.n_runs,
     warmup_discarded = t.warmup_discarded, fits_per_sample = t.fits_per_sample)
end

# VarCorr -> common representation: per grouping factor, RE term names, their
# standard deviations, and the correlation matrix. MixedModels exposes this through
# VarCorr(m).σρ (a NamedTuple per grouping with `.σ` stddevs and `.ρ` correlations),
# already on the absolute σ scale -- the same scale lme4.R normalizes lme4 to.
# val.ρ is flat and strictly-lower-triangular only (no diagonal), in row-major
# walk order: for j in 1:q, for l in 1:(j-1), successive entries are corr(j, l)
# (confirmed from MixedModels.jl's varcorr.jl show method, the only documented
# consumer of this ordering) -- e.g. for q=3: ρ[1]=corr(2,1), ρ[2]=corr(3,1),
# ρ[3]=corr(3,2). Reconstructing the q×q matrix from that walk subsumes q in {1,2}.
function varcomp_of(m)
    vc = VarCorr(m)
    out = Vector{Any}()
    for (grp, val) in pairs(vc.σρ)
        terms = collect(string.(keys(val.σ)))
        stddev = collect(Float64.(values(val.σ)))
        q = length(stddev)
        corr = [[Float64(i == j) for i in 1:q] for j in 1:q]
        rho = collect(Float64.(val.ρ))
        k = 1
        for j in 2:q, l in 1:(j - 1)
            corr[j][l] = corr[l][j] = rho[k]
            k += 1
        end
        push!(out, (group = string(grp), terms = terms, stddev = stddev, corr = corr))
    end
    out
end

# MixedModels has no single converged bool; treat any NLopt success return as
# converged. The values below are the codes that mean "stopped because it found
# the optimum", as opposed to hitting an eval/time limit or failing.
const OK_RETURN = Set([:FTOL_REACHED, :XTOL_REACHED, :SUCCESS, :STOPVAL_REACHED])

# Write JSON in lme4.R's jsonlite-pretty layout so the two engines' result files diff
# cleanly side by side. jsonlite's rule, reproduced here: objects always expand
# one-key-per-line at 2-space depth; an array expands only if it holds a non-scalar
# (object or nested array) -- an all-scalar array stays inline `[a, b, c]`. So beta /
# se / stddev / terms render inline, while varcomp (objects) and corr (nested arrays)
# expand. Numeric VALUES still differ from R in the last digits -- that is the
# cross-engine arithmetic the validation check measures, not a formatting choice.
_kv(x::NamedTuple) = pairs(x)
_kv(x::AbstractDict) = x

fmt_scalar(x) =
    x === nothing            ? "null" :
    x isa Bool               ? (x ? "true" : "false") :
    x isa Integer            ? string(x) :
    x isa AbstractFloat      ? (isinteger(x) ? string(Int(x)) : repr(x)) :  # 1.0 -> "1"
                               JSON3.write(x)   # strings: proper JSON quoting/escaping

is_obj(x) = x isa NamedTuple || x isa AbstractDict

function emit_jsonlite(io, x, depth)
    pad, cpad = "  "^depth, "  "^(depth + 1)
    if is_obj(x)
        kvs = collect(_kv(x))
        isempty(kvs) && return print(io, "{}")
        println(io, "{")
        for (i, (k, v)) in enumerate(kvs)
            print(io, cpad, JSON3.write(string(k)), ": ")
            emit_jsonlite(io, v, depth + 1)
            println(io, i < length(kvs) ? "," : "")
        end
        print(io, pad, "}")
    elseif x isa AbstractVector
        isempty(x) && return print(io, "[]")
        if all(e -> !(is_obj(e) || e isa AbstractVector), x)
            print(io, "[", join(fmt_scalar.(x), ", "), "]")     # all-scalar -> inline
        else
            println(io, "[")
            for (i, v) in enumerate(x)
                print(io, cpad)
                emit_jsonlite(io, v, depth + 1)
                println(io, i < length(x) ? "," : "")
            end
            print(io, pad, "]")
        end
    else
        print(io, fmt_scalar(x))
    end
end

function fit_one(spec)
    df = read_dataset(spec)
    make_fit = fit_thunk(spec, df)
    m = make_fit()
    gaussian = String(spec.family) == "gaussian"

    # logLik on the common scale. MixedModels refuses `loglikelihood` on a REML fit
    # (a REML criterion is not a likelihood), and reports the -2*logLik `objective`;
    # lme4's `logLik` on a REML fit is that same restricted criterion / -2. So emit
    # -objective/2 for the REML LMMs, and the genuine Laplace logLik for the ML GLMMs.
    loglik = gaussian ? -objective(m) / 2 : loglikelihood(m)
    # NamedTuples, not Dicts: they preserve field order, so the JSON keys come out
    # in the same order lme4.R writes them (a Dict serializes in arbitrary hash order).
    # MixedModels' stderror is the conditional-on-theta-hat (RX) vcov -- it has no
    # Hessian/coupling variant. Label it se_rx on GLMM rungs so the validation check
    # compares it against lme4's and glmm's RX, not their Hessian SE. Gaussian rungs
    # keep a single profiled `se` (method-agnostic), matching lme4.R.
    base = gaussian ?
        (beta = collect(coef(m)), se = collect(stderror(m)),
         loglik = loglik, varcomp = varcomp_of(m)) :
        (beta = collect(coef(m)), se_rx = collect(stderror(m)),
         loglik = loglik, varcomp = varcomp_of(m))
    est = gaussian ? merge(base, (sigma = sdest(m),)) : base  # sigma last, as in lme4.R

    res = (dataset = string(spec.name), engine = "mixedmodels",
           engine_version = string(pkgversion(MixedModels)),
           family = string(spec.family),
           reml = gaussian ? (spec.reml === true) : nothing,
           rung = spec.rung,
           converged = m.optsum.returnvalue in OK_RETURN,
           singular = issingular(m),
           optimizer = string(m.optsum.optimizer),
           n_eval = m.optsum.feval,
           coef_names = collect(coefnames(m)),   # contrast-coding assertion vs lme4
           estimates = est,
           timing = time_one(spec, make_fit))

    open(joinpath(out_dir_of(spec), string(spec.name, ".json")), "w") do io
        emit_jsonlite(io, res, 0)
        println(io)                              # trailing newline, matching R's write()
    end
    t_disp = res.timing === nothing ? nothing :
             String(spec.family) == "gaussian" ? res.timing.fit_seconds_median :
                                                 res.timing.fit_seconds_median_rx
    println("mixedmodels  $(rpad(spec.name,12))  rung $(spec.rung)  ",
            "converged=$(res.converged) singular=$(res.singular)",
            t_disp === nothing ? "" : "  fit_median=$(round(t_disp, sigdigits=4))s")
end

# VALIDATION_ONLY=<name>[,<name>...]: fit only the named datasets (mirrors lme4.R) —
# lets a NEW rung get its reference generated without rewriting the frozen
# results of the existing ones (the oracle is sacred).
# (named `only_ds`, not `only` — a global `only` would shadow Base.only used above)
only_ds = get(ENV, "VALIDATION_ONLY", "")
specs = isempty(only_ds) ? collect(manifest.datasets) :
    filter(s -> String(s.name) in split(only_ds, ","), collect(manifest.datasets))
# sim_binomial_slope_crossed (rung 18): the pinned MixedModels (v5.7.0) cannot
# CONSTRUCT this shape — PosDefException at construction, before PIRLS runs; a
# package limitation confirmed independent of the data (see validation/README.md,
# "2-way gate"). results/mixedmodels_{empirical,simulated}/<name>.json is
# intentionally absent and compare.R reports the rung as n/a for this engine.
# sim_poisson_bigsd (rung 45): same engine, same exception, later in the fit — the
# model constructs, then PIRLS throws PosDefException (deviance! -> reweight! ->
# updateL! -> cholUnblocked!) on the large-theta-hat Poisson design. Confirmed
# 2026-08-07 on MixedModels v5.7.0; rung 46's `//` comment in manifest.json cites
# the same crash. Skipping it here is what keeps a whole-corpus `run.sh --oracles`
# alive: run.sh is `set -e` and runs the Julia engine before the Rust one, so an
# unskipped crash here kills the run before the Rust leg and compare.R.
const JL_CANNOT_FIT = ("sim_binomial_slope_crossed", "sim_poisson_bigsd")
for spec in specs
    # No jl_formula = "not a Julia rung" (weights suite fixed-only and R-only
    # rungs omit the field): MixedModels does not fit fixed-only models and
    # GLM.jl is not a project dependency, so these rungs are lme4-only.
    if !haskey(spec, :jl_formula)
        println("mixedmodels  $(rpad(spec.name,12))  SKIPPED -- no jl_formula (R-only rung)")
        continue
    end
    if String(spec.name) in JL_CANNOT_FIT
        println("mixedmodels  $(rpad(spec.name,12))  SKIPPED -- MixedModels cannot fit this rung (see comment above)")
        continue
    end
    fit_one(spec)
end
