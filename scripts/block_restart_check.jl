# Precondition for reusable compiled block VJPs with host-side checkpointing: restarting a block from the
# saved PROGNOSTIC fields (+ clock time/iteration) must reproduce the continuous trajectory exactly, i.e.
# no hidden carried state (radiation flux cache, microphysics diagnostics, timestepper tendencies) is lost.
# Native, from a window_state.jls; continuous nb·B steps vs nb restarts of B steps (window_gradient's
# per-step SST / cos(zenith) writes included). Reports per-field max relative difference per block.
# ENV: ATLANTIC_INPUT, ATLANTIC_STATE, ATLANTIC_NX, ATLANTIC_DYNAMICS, ATLANTIC_BLOCK (60), ATLANTIC_BLOCKS (3),
#      ATLANTIC_RESTART (prepare: also call prepare_first_time_step! after restore, as window_objective does).
include("model.jl")
using Oceananigans: initialize!, prognostic_fields
using Oceananigans.Architectures: on_architecture
using Oceananigans.TimeSteppers: reset!, update_state!
using Breeze.TimeSteppers: prepare_first_time_step!
using Serialization

input = ENV["ATLANTIC_INPUT"];
saved = deserialize(ENV["ATLANTIC_STATE"])
nx = parse(Int, get(ENV, "ATLANTIC_NX", "8"));
dx = grid_spacing(100.0)
B = parse(Int, get(ENV, "ATLANTIC_BLOCK", "60"));
nb = parse(Int, get(ENV, "ATLANTIC_BLOCKS", "3"));
dt = 1.0
prep = get(ENV, "ATLANTIC_RESTART", "prepare") == "prepare"
# ATLANTIC_FRESH=true: every restart uses a NEWLY BUILT model (no cache/history from the continuous run)
fresh = get(ENV, "ATLANTIC_FRESH", "false") == "true"
# ATLANTIC_EXTRA_STATE=true: checkpoint/restore the carried non-prognostic state found by restart_state_diff (17482):
# anelastic pressure anomaly, microphysical surface_temperature, clock last_Δt/last_stage_Δt.
extra_mode = get(ENV, "ATLANTIC_EXTRA_STATE", "false");
extra = extra_mode != "false"
# model_re:<regex>: restore only whole-model storage arrays whose property path matches <regex> (bisection, 17510)
model_re = startswith(extra_mode, "model_re:") ? Regex(extra_mode[10:end]) : nothing
rad_all = extra_mode in ("radiation_all", "model_all") || !isnothing(model_re);
model_all = extra_mode == "model_all" || !isnothing(model_re)
# ATLANTIC_EXTRA_STATE=radiation_all: also every array leaf anywhere under model.radiation (workspace incl.
# atmospheric_state / lookup stencil), restored by path — tests whether the radiation solve depends on its own history.
function array_leaves!(out, x, path, seen, depth)
    depth > 12 && return out
    if x isa Union{Array{<:Number},CUDA.CuArray{<:Number}}   # storage only; wrappers (OffsetArray, Field) are walked into
        out[path] = x
        return out
    end
    (
        isbitstype(typeof(x)) ||
        x isa Union{Module,Function,DataType,String,Symbol} ||
        x isa Oceananigans.Grids.AbstractGrid
    ) && return out
    id = objectid(x)
    id in seen && return out
    push!(seen, id)
    if x isa Union{Tuple,NamedTuple}
        for (k, v) in pairs(x)
            array_leaves!(out, v, "$path[$k]", seen, depth + 1)
        end
    else
        for f in fieldnames(typeof(x))
            isdefined(x, f) && array_leaves!(out, getfield(x, f), "$path.$f", seen, depth + 1)
        end
    end
    return out
end
radiation_arrays(m) =
    model_all ?
    filter(
        kv -> isnothing(model_re) || occursin(model_re, first(kv)),
        array_leaves!(Dict{String,Any}(), m, "model", Set{UInt}(), 0),
    ) : array_leaves!(Dict{String,Any}(), m.radiation, "radiation", Set{UInt}(), 0)
# model_all: EVERY storage array of the whole model (prognostic, auxiliary, timestepper, solver, forcing, radiation…)
radiation_field_names(m) = [
    n for n in propertynames(m.radiation) if
    getproperty(m.radiation, n) isa Oceananigans.Fields.AbstractField
]
function extra_snapshot(m)
    d = Dict{String,Any}("last_Δt" => m.clock.last_Δt, "last_stage_Δt" => m.clock.last_stage_Δt)
    hasproperty(m.dynamics, :pressure_anomaly) &&
        (d["p"] = Array(parent(m.dynamics.pressure_anomaly)))
    haskey(m.microphysical_fields, :surface_temperature) &&
        (d["Ts"] = Array(parent(m.microphysical_fields.surface_temperature)))
    # Radiation fluxes from the last scheduled update: stage 1 of the step that begins a block still uses them
    # (radiation_interface.jl: new fluxes enter from stage 2), and a fresh restore at iteration > 0 does not recompute them.
    for n in radiation_field_names(m)
        d["rad.$n"] = Array(parent(getproperty(m.radiation, n)))
    end
    rad_all && for (k, a) in radiation_arrays(m)
        d["all.$k"] = Array(a)
    end
    return d
end
function extra_restore!(m, d)
    haskey(d, "p") && copyto!(parent(m.dynamics.pressure_anomaly), d["p"])
    haskey(d, "Ts") && copyto!(parent(m.microphysical_fields.surface_temperature), d["Ts"])
    for n in radiation_field_names(m)
        copyto!(parent(getproperty(m.radiation, n)), d["rad.$n"])
    end
    if rad_all
        ra = radiation_arrays(m)
        nr = 0
        for (k, a) in ra
            haskey(d, "all.$k") &&
                size(a) == size(d["all.$k"]) &&
                (copyto!(a, d["all.$k"]); nr += 1)
        end
        println("restored $nr radiation array leaves")
    end
    m.clock.last_Δt = d["last_Δt"]
    m.clock.last_stage_Δt = d["last_stage_Δt"]
end
# ATLANTIC_INPUT_DELTA="T:1,q:-1,w:1,ug:1,vg:-1": nonzero forcing controls, multiples of the input-FD scales
# (1 K/day, 1 g/kg/day, 1 mm/s, 0.1 m/s) × (1 − z/3 km)+ at every forcing knot, written into the model's own
# input series (same plumbing as window_gradient inputs mode) for the continuous AND every fresh model.
const INPUT_SCALE =
    Dict("T" => 1 / 86400, "q" => 1e-3 / 86400, "w" => 1e-3, "ug" => 0.1, "vg" => 0.1)
deltas = [
    (String(a), parse(Float64, b)) for (a, b) in
    (split(s, ":") for s in filter(!isempty, split(get(ENV, "ATLANTIC_INPUT_DELTA", ""), ",")))
]
function build_case()
    c = make_case(input; nx, dx, architecture = GPU())
    for (f, a) in deltas
        P = parent(getproperty(c.input_series, Symbol(f)).data)
        nz = length(c.z)
        off = (size(P, 3) - nz) ÷ 2
        h = Array(P)
        h[1, 1, (off+1):(off+nz), :] .+=
            reshape(a .* INPUT_SCALE[f] .* clamp.(1 .- c.z ./ 3000, 0, 1), nz, 1)
        copyto!(P, h)
    end
    return c
end
case = build_case();
m = case.model;
t0 = saved.time
surface_temperature(m) = m.radiation.surface_radiation.surface_temperature
snapshot(m) = Dict(String(n) => Array(interior(f)) for (n, f) in pairs(prognostic_fields(m)))
# ATLANTIC_RAD_FIRST=true: write the radiation flux Fields of the previous block end BEFORE initialize!/update_state!,
# so stage-1 tendencies Gⁿ (incl. radiative heating of ρθ) are built from the carried fluxes (17512: timestepper restore
# fixes the drift; only Gⁿ[ρθ] differed after restore in 17485).
rad_first = get(ENV, "ATLANTIC_RAD_FIRST", "false") == "true"
# ATLANTIC_PRE_RE=<regex>: also write matching storage arrays (from the continuous block end) BEFORE
# initialize!/update_state!, so derived Gⁿ is rebuilt from them (split: time-averaged velocity caches, 17543).
pre_re = haskey(ENV, "ATLANTIC_PRE_RE") ? Regex(ENV["ATLANTIC_PRE_RE"]) : nothing
all_arrays(m) = array_leaves!(Dict{String,Any}(), m, "model", Set{UInt}(), 0)
function restore!(m, fields, time, iteration; radiation = nothing, pre = nothing)
    for (name, f) in pairs(prognostic_fields(m))
        interior(f) .= on_architecture(GPU(), fields[String(name)])
    end
    reset!(m.clock)
    m.clock.time = time
    m.clock.iteration = iteration
    if !isnothing(radiation)
        for (n, a) in radiation
            copyto!(parent(getproperty(m.radiation, n)), a)
        end
    end
    if !isnothing(pre) && !isnothing(pre_re)
        np = 0
        for (k, a) in all_arrays(m)
            occursin(pre_re, k) &&
                haskey(pre, k) &&
                size(a) == size(pre[k]) &&
                (copyto!(a, pre[k]); np += 1)
        end
        println("pre-restored $np arrays matching $(pre_re.pattern)")
    end
    initialize!(m)
    update_state!(m)
    prep && prepare_first_time_step!(m, dt)
end
step!(m) = (case.refresh!((; model = m)); time_step!(m, dt))   # host SST refresh as the native forward

restore!(m, saved.fields, t0, 0)
continuous = Dict{Int,Any}()
extras = Dict{Int,Any}();
pre_snaps = Dict{Int,Any}()
for b = 1:nb
    for _ = 1:B
        step!(m)
    end
    continuous[b] = snapshot(m)
    extras[b] = extra_snapshot(m)
    isnothing(pre_re) ||
        (pre_snaps[b] = Dict(k => Array(a) for (k, a) in all_arrays(m) if occursin(pre_re, k)))
end
state = saved.fields
for b = 1:nb
    fresh && (global case = build_case(); global m = case.model)
    rad =
        rad_first && b > 1 ? Dict(n => extras[b-1]["rad.$n"] for n in radiation_field_names(m)) :
        nothing
    restore!(
        m,
        state,
        t0 + (b - 1) * B * dt,
        (b - 1) * B;
        radiation = rad,
        pre = (b > 1 && !isnothing(pre_re)) ? pre_snaps[b-1] : nothing,
    )
    extra && b > 1 && extra_restore!(m, extras[b-1])   # from the continuous run's block end (what a checkpoint would store)
    for _ = 1:B
        step!(m)
    end
    global state = snapshot(m)
    d = Dict(
        n =>
            maximum(abs.(state[n] .- continuous[b][n])) /
            max(maximum(abs.(continuous[b][n])), eps()) for n in keys(state)
    )
    worst = maximum(values(d))
    println(
        "BLOCK $b worst_rel=$worst ",
        join(
            [
                "$n=$(round(v; sigdigits=3))" for
                (n, v) in sort(collect(d); by = last, rev = true)[1:min(5, end)]
            ],
            " ",
        ),
    )
end
println(
    "BLOCK_RESTART_CHECK_DONE prepare=$prep fresh=$fresh extra=$extra_mode rad_first=$rad_first B=$B deltas=$deltas radiation_fields=$(radiation_field_names(m))",
)
