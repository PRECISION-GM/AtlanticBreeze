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

input = ENV["ATLANTIC_INPUT"]; saved = deserialize(ENV["ATLANTIC_STATE"])
nx = parse(Int, get(ENV, "ATLANTIC_NX", "8")); dx = grid_spacing(100.0)
B = parse(Int, get(ENV, "ATLANTIC_BLOCK", "60")); nb = parse(Int, get(ENV, "ATLANTIC_BLOCKS", "3")); dt = 1.0
prep = get(ENV, "ATLANTIC_RESTART", "prepare") == "prepare"
case = make_case(input; nx, dx, architecture=GPU()); m = case.model; t0 = saved.time
surface_temperature(m) = m.radiation.surface_radiation.surface_temperature
snapshot(m) = Dict(String(n) => Array(interior(f)) for (n, f) in pairs(prognostic_fields(m)))
function restore!(m, fields, time, iteration)
    for (name, f) in pairs(prognostic_fields(m)); interior(f) .= on_architecture(GPU(), fields[String(name)]); end
    reset!(m.clock); m.clock.time = time; m.clock.iteration = iteration
    initialize!(m); update_state!(m); prep && prepare_first_time_step!(m, dt)
end
step!(m) = (case.refresh!((; model=m)); time_step!(m, dt))   # host SST refresh as the native forward

restore!(m, saved.fields, t0, 0)
continuous = Dict{Int, Any}()
for b in 1:nb; for _ in 1:B; step!(m); end; continuous[b] = snapshot(m); end
state = saved.fields
for b in 1:nb
    restore!(m, state, t0 + (b - 1) * B * dt, (b - 1) * B)
    for _ in 1:B; step!(m); end
    global state = snapshot(m)
    d = Dict(n => maximum(abs.(state[n] .- continuous[b][n])) / max(maximum(abs.(continuous[b][n])), eps()) for n in keys(state))
    worst = maximum(values(d))
    println("BLOCK $b worst_rel=$worst ", join(["$n=$(round(v; sigdigits=3))" for (n, v) in sort(collect(d); by=last, rev=true)[1:min(5, end)]], " "))
end
println("BLOCK_RESTART_CHECK_DONE prepare=$prep")
