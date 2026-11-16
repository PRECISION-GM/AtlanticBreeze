# Diagnose 17461 (fresh-model restart at a block boundary is NOT exact; block 1 exact, blocks 2/3 ~1e-4).
# Runs the continuous model B steps, then builds a fresh model and restores prognostic fields + clock
# exactly as block_restart_check.jl. Walks BOTH model objects recursively, collects every array leaf by
# property path (auxiliary/diagnostic fields, radiation fluxes/caches, microphysics state, timestepper
# tendencies, forcing caches, clock), and prints every path whose values differ, (a) right after restore
# and (b) after one more step in each. Paths that differ in (a) are state the restart fails to carry.
# ENV: ATLANTIC_INPUT, ATLANTIC_STATE, ATLANTIC_NX (8), ATLANTIC_DYNAMICS, ATLANTIC_BLOCK (60).
include("model.jl")
using Oceananigans: initialize!, prognostic_fields
using Oceananigans.Architectures: on_architecture
using Oceananigans.TimeSteppers: reset!, update_state!
using Breeze.TimeSteppers: prepare_first_time_step!
using Serialization

input = ENV["ATLANTIC_INPUT"]; saved = deserialize(ENV["ATLANTIC_STATE"])
nx = parse(Int, get(ENV, "ATLANTIC_NX", "8")); dx = grid_spacing(100.0); B = parse(Int, get(ENV, "ATLANTIC_BLOCK", "60")); dt = 1.0
t0 = saved.time

function restore!(m, fields, time, iteration)
    for (name, f) in pairs(prognostic_fields(m)); interior(f) .= on_architecture(GPU(), fields[String(name)]); end
    reset!(m.clock); m.clock.time = time; m.clock.iteration = iteration
    initialize!(m); update_state!(m); prepare_first_time_step!(m, dt)
end
step!(c) = (c.refresh!((; model=c.model)); time_step!(c.model, dt))
snapshot(m) = Dict(String(n) => Array(interior(f)) for (n, f) in pairs(prognostic_fields(m)))

# Recursive collection of array leaves (and number leaves) by property path; skips the grid and huge tables once seen.
function leaves!(out, x, path, seen, depth)
    depth > 14 && return out
    if x isa AbstractArray{<:Number}
        a = try Array(parent(x)) catch; try Array(x) catch; return out end end
        out[path] = a; return out
    elseif x isa Number
        out[path] = [x]; return out
    end
    (isbitstype(typeof(x)) && !(x isa Tuple)) && return out
    x isa Union{Module, Function, DataType, String, Symbol} && return out
    x isa Oceananigans.Grids.AbstractGrid && return out
    id = objectid(x); id in seen && return out; push!(seen, id)
    if x isa Union{Tuple, NamedTuple, AbstractVector}
        for (k, v) in pairs(x); leaves!(out, v, "$path[$k]", seen, depth + 1); end
    elseif x isa AbstractDict
        for (k, v) in x; leaves!(out, v, "$path[$k]", seen, depth + 1); end
    else
        for f in fieldnames(typeof(x))
            isdefined(x, f) || continue
            leaves!(out, getfield(x, f), "$path.$f", seen, depth + 1)
        end
    end
    return out
end
collect_leaves(m) = leaves!(Dict{String, Any}(), m, "model", Set{UInt}(), 0)
function report(label, A, Bd)
    n = 0
    for k in sort(collect(keys(A)))
        haskey(Bd, k) || (println("$label ONLY_CONTINUOUS $k"); continue)
        a, b = A[k], Bd[k]
        size(a) == size(b) || (println("$label SHAPE $k $(size(a)) $(size(b))"); continue)
        eltype(a) <: Number || continue
        d = maximum(abs.(a .- b); init=0.0)   # abs handles complex solver buffers (17482)
        if !(d == 0 || (isnan(d) && all(isnan.(a) .== isnan.(b))))
            n += 1; s = maximum(abs.(a); init=0.0)
            println("$label DIFF $k max_abs=$d max_ref=$s size=$(size(a))")
        end
    end
    for k in keys(Bd); haskey(A, k) || println("$label ONLY_FRESH $k"); end
    println("$label differing_paths=$n of $(length(A))")
end

cont = make_case(input; nx, dx, architecture=GPU())
restore!(cont.model, saved.fields, t0, 0)
for _ in 1:B; step!(cont); end
state = snapshot(cont.model)
fresh = make_case(input; nx, dx, architecture=GPU())
restore!(fresh.model, state, t0 + B * dt, B)
report("AFTER_RESTORE", collect_leaves(cont.model), collect_leaves(fresh.model))
step!(cont); step!(fresh)
report("AFTER_ONE_STEP", collect_leaves(cont.model), collect_leaves(fresh.model))
println("RESTART_STATE_DIFF_DONE")
