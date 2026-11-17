# Cold-versus-warm cost of the native window model, measured in a FRESH process with one protocol for every
# grid: package load, first model construction (includes native JIT of construction), first time step
# (JIT of the step kernels; radiation also runs on this iteration), the first radiation-cadence block, then
# synchronized warm steps. Starts from a saved window state (observation_sensitivity.jl). AD/Reactant times
# are not measured here (unavailable, not zero).
# ENV: ATLANTIC_INPUT, ATLANTIC_STATE, ATLANTIC_NX, ATLANTIC_DX, ATLANTIC_OUTPUT, ATLANTIC_WARM_STEPS (300).
t_load = @elapsed include("model.jl")
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: update_state!, time_step!
using Serialization
input = ENV["ATLANTIC_INPUT"];
output = mkpath(ENV["ATLANTIC_OUTPUT"])
saved = deserialize(ENV["ATLANTIC_STATE"]);
nx = saved.nx;
dx = saved.dx
config = TOML.parsefile(joinpath(input, "interval.toml"));
dt = config["dt"]
function build()
    case = make_case(input; nx, dx)
    m = case.model
    for (n, f) in pairs(prognostic_fields(m))
        set!(f, saved.fields[String(n)])
    end
    m.clock.time = saved.time
    m.clock.iteration = saved.iteration
    case.refresh!((; model = m))
    update_state!(m)
    CUDA.synchronize()
    return case
end
sync(f) = (t = time(); f(); CUDA.synchronize(); time() - t)
t_build_cold = @elapsed case = build()
m = case.model
step!() = (time_step!(m, dt); case.refresh!((; model = m)))
t_first_step = sync(step!)                                  # JIT + radiation at this iteration
n_block = case.radiation_iterations
t_first_block = sync(() -> for _ = 2:n_block
    step!()
end)  # remaining JIT of non-radiation steps
warm = parse(Int, get(ENV, "ATLANTIC_WARM_STEPS", "300"))
t_warm = sync(() -> for _ = 1:warm
    step!()
end)
t_build_warm = @elapsed build()
r = Dict(
    "gpu" => CUDA.name(CUDA.device()),
    "nx" => nx,
    "dx" => dx,
    "nz" => size(m.grid, 3),
    "cells" => prod(size(m.grid)),
    "dt" => dt,
    "radiation_iterations" => n_block,
    "load_s" => t_load,
    "build_cold_s" => t_build_cold,
    "build_warm_s" => t_build_warm,
    "first_step_s" => t_first_step,
    "rest_of_first_radiation_block_s" => t_first_block,
    "warm_steps" => warm,
    "warm_s_per_step" => t_warm / warm,
    "commit" => get(ENV, "ATLANTIC_COMMIT", "unknown"),
)
open(io -> TOML.print(io, r), joinpath(output, "timing.toml"), "w")
println("TIMING ", r)
