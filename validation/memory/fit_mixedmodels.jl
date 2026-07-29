#!/usr/bin/env julia
# MixedModels.jl oracle runner for the large synthetic memory-measurement
# models (models.json). NOT a validation rung -- see models.json's header;
# this is not compared to anything, only measured for peak RSS. lme4/
# MixedModels do not change between memory legs, so they are collected once
# into results/memory/oracles.tsv and reused in every leg comparison (Julia
# costs ~830 MB and a slow start per process, so refitting it per leg would
# be the most expensive part of the whole harness for no new information).
#
#   julia --project=<validation dir> fit_mixedmodels.jl <csv> <formula> <factors_csv> [family]
#
# formula: the same R-style string as models.json's "formula" field, but
# MixedModels.jl's/GLM.jl's @formula macro (unlike this crate's parser) has no
# implicit intercept, so an explicit "1 +" is inserted after "~" before parsing
# (mirrors engines/mixedmodels.jl's inverse transform, which strips that same
# "1 +" for the Rust engine).
#
# family (optional, default "binomial"): the original 4 oracle rows (1/4/6/9)
# were all binomial-with-RE, so every existing 3-arg call keeps working
# unchanged. The oracle backfill added three more row shapes this script must
# also route correctly, picked by whether the formula has a "|" RE term AND
# family:
#   RE + binomial   (rows 2/3/5/7/8/10)  -> fit(MixedModel, ..., Bernoulli())
#   RE + gaussian    (row 11)             -> fit(MixedModel, ...) (LMM, identity)
#   no RE + binomial (row 12)             -> GLM.jl's glm(f, df, Bernoulli())
#   no RE + gaussian (row 13)             -> GLM.jl's lm(f, df)
# GLM.jl is only needed for the two no-RE rows -- promoted from a transitive
# MixedModels dep to a direct Project.toml dep for this (see Project.toml's
# comment at the GLM entry), already pinned in Manifest.toml so this adds no
# new resolved version.
using MixedModels, CSV, DataFrames, GLM

csv_path, formula_str, factors_csv = ARGS[1], ARGS[2], ARGS[3]
family_str = length(ARGS) >= 4 ? ARGS[4] : "binomial"
factors = Symbol.(split(factors_csv, ","; keepempty = false))

df = CSV.read(csv_path, DataFrame)
for c in factors
    df[!, c] = string.(df[!, c])
end

has_re = occursin("|", formula_str)
gaussian = family_str == "gaussian"
jl_formula_str = replace(formula_str, " ~ " => " ~ 1 + "; count = 1)
f = eval(Meta.parse("@formula($jl_formula_str)"))

m = if has_re && !gaussian
    fit(MixedModel, f, df, Bernoulli(); progress = false)
elseif has_re && gaussian
    fit(MixedModel, f, df; progress = false)
elseif !has_re && !gaussian
    glm(f, df, Bernoulli())
else
    lm(f, df)
end
println("n=$(nrow(df)) deviance=$(round(deviance(m), digits=6))")
