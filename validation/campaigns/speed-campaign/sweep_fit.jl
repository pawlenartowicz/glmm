#!/usr/bin/env julia
# MixedModels side of the mismatch-adjudication multi-start sweep (oracle spec
# 2026-07-11, Step B). Reads $SWEEP_STARTS (JSON: case_id -> label -> θ₀ in
# glmm's declaration-order vech layout), maps each start to MM's reterm order
# by grouping name, fits under a tightened schedule (maxfeval 100k, ftol/xtol
# tightened — NLopt has no rho_end; endpoints are re-scored on glmm's
# objective anyway), appends one JSONL line per (cell, start) to $SWEEP_OUT.
# Gaussian REML cells only (the 11 adjudication targets).
using MixedModels, CSV, DataFrames, JSON3

starts = JSON3.read(read(ENV["SWEEP_STARTS"], String))
out_path = ENV["SWEEP_OUT"]
manifest = JSON3.read(read(joinpath(@__DIR__, "manifest.json"), String))
cells = Dict(String(c.case_id) => c for c in manifest.cells)
const DECL = ["g1", "g2", "g3"]  # glmm extra-grouping declaration order

open(out_path, "a") do io
    for cid in sort(collect(String.(keys(starts))))
        cell = cells[cid]
        df = CSV.read(joinpath(@__DIR__, "data", string(cid, ".csv")), DataFrame)  # campaign-local, prep.R output
        for f in cell.factors
            df[!, Symbol(f)] = string.(df[!, Symbol(f)])
        end
        fform = eval(Meta.parse(String(cell.jl_formula)))
        build() = begin
            m = LinearMixedModel(fform, df)
            m.optsum.maxfeval = 100_000
            m.optsum.ftol_rel = 1e-14
            m.optsum.ftol_abs = 1e-12
            m.optsum.xtol_rel = 0.0
            m
        end
        m0 = build()
        groups = [string(MixedModels.fname(t)) for t in m0.reterms]
        qs_mm = [size(t.λ, 1) for t in m0.reterms]
        # glmm declaration-order segment widths (primary g1 first, extras after)
        qs_decl = collect(Int, cell.re_q)
        widths = [q * (q + 1) ÷ 2 for q in qs_decl]
        seg_of = Dict{String,UnitRange{Int}}()
        off = 0
        for (g, w) in zip(DECL[1:length(qs_decl)], widths)
            seg_of[g] = (off+1):(off+w)
            off += w
        end
        for label in sort(collect(String.(keys(starts[cid]))))
            th = collect(Float64, starts[cid][label])
            θ0 = vcat((th[seg_of[g]] for g in groups)...)
            rec = try
                m = build()
                copyto!(m.optsum.initial, θ0)
                fit!(m; REML = true, progress = false)
                (deviance = objective(m), theta = collect(m.θ),
                 re_groups = groups, re_qs = qs_mm,
                 n_eval = m.optsum.feval,
                 status = string(m.optsum.returnvalue))
            catch err
                @warn "sweep-fail" cid label err = sprint(showerror, err)
                (deviance = nothing, theta = Float64[],
                 re_groups = groups, re_qs = qs_mm, n_eval = 0,
                 status = "engine-fail")
            end
            println(io, JSON3.write(merge((case_id = cid, label = label, engine = "mixedmodels"), rec)))
            flush(io)
        end
    end
end
