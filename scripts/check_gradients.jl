# Reactant/Enzyme reverse-mode gradient of a short March-case window with respect to the two training
# controls (tapered dT/dt and dq/dt corrections), checked against native-CUDA central differences.
# Scope: the configured grid, spacing, advection order and window only. A pass here is evidence for that
# configuration and window, not for the 12 h objective or other grids.
#
# Environment: the `gradient/` project (forward Manifest plus Reactant and Enzyme).
#   ATLANTIC_NX (16), ATLANTIC_DX (200), ATLANTIC_ORDER (9), ATLANTIC_SPINUP_SECONDS (3600),
#   ATLANTIC_WINDOW_STEPS (3), ATLANTIC_STATE (spun-up state file, created if absent), ATLANTIC_OUTPUT.
using Reactant, Enzyme
include("model.jl")
using Oceananigans: initialize!, prognostic_fields
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: reset!, update_state!
using Breeze.TimeSteppers: prepare_first_time_step!
using Reactant: @trace
using Serialization, Pkg

input = get(ENV, "ATLANTIC_INPUT", "data/interval")
output = get(ENV, "ATLANTIC_OUTPUT", "outputs/gradients")
nx = parse(Int, get(ENV, "ATLANTIC_NX", "16"))
dx = parse(Float64, get(ENV, "ATLANTIC_DX", "200")); ENV["ATLANTIC_DX"] = string(dx)
order = advection_order()
spinup = parse(Float64, get(ENV, "ATLANTIC_SPINUP_SECONDS", "3600"))
steps = parse(Int, get(ENV, "ATLANTIC_WINDOW_STEPS", "3"))
state_file = get(ENV, "ATLANTIC_STATE", joinpath(output, "spunup_state.jls"))
epsilons = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4]   # normalized control units (1 = 3 K/day or 2 g/kg/day)
limits = [3.0, 2.0]
isdir(output) && !isempty(readdir(output)) && error("Output directory is not empty: $output")
mkpath(output)
Reactant.set_default_backend("gpu")
note(msg; kw...) = (@info msg kw...; flush(stderr))

# Build the native case; spin up from the condensate-free sounding to a cloudy state (or load it).
native = make_case(input; nx, dx, order, architecture=GPU())
model = native.model
Nz = length(native.z); hz = advection_halo(order)[3]
if isfile(state_file)
    saved = deserialize(state_file)
    (saved.nx, saved.dx, saved.order) == (nx, dx, order) || error("State file is for another configuration")
else
    simulation = Simulation(model; Δt=native.config["dt"], stop_time=spinup)
    add_callback!(simulation, native.refresh!, IterationInterval(1))
    spin = @elapsed run!(simulation)
    saved = (; nx, dx, order, time=model.clock.time, spinup_wall=spin,
               sst=Array(interior(model.radiation.surface_radiation.surface_temperature))[1],
               fields=Dict(String(n) => Array(interior(f)) for (n, f) in pairs(prognostic_fields(model))))
    serialize(state_file, saved)
end

# Window inputs, all on the architecture of the model they are used with (no host reads while traced).
function window_inputs(arch)
    on(x) = arch isa ReactantState ? Reactant.to_rarray(x) : on_architecture(arch, x)
    padded(v) = reshape(vcat(zeros(hz), v, zeros(hz)), 1, 1, :, 1)
    keep = native.z .<= TAPER_TOP
    return (; fields = Dict(n => on(a) for (n, a) in saved.fields),
              energy_taper = on(padded(native.taper .* limits[1] ./ 86400)),
              moisture_taper = on(padded(native.taper .* limits[2] ./ (1000 * 86400))),
              liquid_weights = on(reshape(diff(native.faces) ./ (nx^2 * 0.05), 1, 1, :)),          # LWP / 50 g m⁻²
              vapor_weights = on(reshape(diff(native.faces) .* keep ./ (nx^2 * 1.0), 1, 1, :)),     # vapor path / 1 kg m⁻²
              time = saved.time)
end

# The correction forcings are the last terms of ρE and ρqᵛ (see make_case).
energy_correction(model) = getfield(getfield(model.forcing.ρE, 1)[end], 1).tls
moisture_correction(model) = getfield(getfield(model.forcing.ρqᵛ, 1)[end], 1).qls

function set_controls!(model, x, inputs)
    parent(energy_correction(model).data) .= inputs.energy_taper .* reshape(x[1:1], 1, 1, 1, 1)
    parent(moisture_correction(model).data) .= inputs.moisture_taper .* reshape(x[2:2], 1, 1, 1, 1)
    return nothing
end

# Every evaluation restores all prognostic fields, the controls and the clock (iteration 0, stage 1,
# spun-up time), so radiation is recomputed from the restored state and then every 30 steps. SSP-RK3
# keeps no multistep tendencies. SST is held at its spun-up value (it changes < 1e-6 K per window).
function window_objective(model, x, inputs, Δt, steps)
    for (name, field) in pairs(prognostic_fields(model))
        interior(field) .= inputs.fields[String(name)]
    end
    set_controls!(model, x, inputs)
    reset!(model.clock)
    model.clock.time = inputs.time
    initialize!(model)
    update_state!(model)
    prepare_first_time_step!(model, Δt)
    @trace mincut=true checkpointing=true track_numbers=false for step in 1:steps
        time_step!(model, Δt)
    end
    water = model.microphysical_fields
    liquid = interior(water.ρqᶜˡ) .+ interior(water.ρqʳ) .+ interior(water.ρqʷⁱ)
    return sum(liquid .* inputs.liquid_weights) + sum(interior(model.moisture_density) .* inputs.vapor_weights)
end

function gradient!(model, dmodel, x, dx, inputs, Δt, steps)
    Enzyme.remake_zero!(dmodel)
    dx .= 0
    _, J = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), window_objective, Enzyme.Active,
                           Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(x, dx),
                           Enzyme.Const(inputs), Enzyme.Const(Δt), Enzyme.Const(steps))
    return J, dx
end

function with_large_stack(f)
    task = Task(f, 256 * 1024^2); schedule(task); return fetch(task)
end

Δt = native.config["dt"]
set!(model.radiation.surface_radiation.surface_temperature, saved.sst)
inputs = window_inputs(GPU())
J(x) = window_objective(model, on_architecture(GPU(), x), inputs, Δt, steps)
native_seconds = @elapsed J0 = J(zeros(2))
J0_repeat = J(zeros(2))
cloud_lwp = 50 * (J0 - sum(Array(interior(model.moisture_density)) .* Array(inputs.vapor_weights)))
note("native primal"; J0, J0_repeat, cloud_lwp_gm2=cloud_lwp, native_seconds)
cloud_lwp > 1 || error("Window starts from a nearly cloud-free state (LWP $(cloud_lwp) g/m²); spin up longer")
fd = [(k, ε, (J([k == 1 ? ε : 0, k == 2 ? ε : 0]) - J([k == 1 ? -ε : 0, k == 2 ? -ε : 0])) / 2ε) for k in 1:2, ε in epsilons]

rcase = with_large_stack(() -> make_case(input; nx, dx, order, architecture=ReactantState()))
rmodel = rcase.model
set!(rmodel.radiation.surface_radiation.surface_temperature, saved.sst)
rinputs = window_inputs(ReactantState())
x = Reactant.to_rarray(zeros(2)); dx_shadow = Reactant.to_rarray(zeros(2))
options = Reactant.CompileOptions(raise=true, raise_first=true, sync=true, excluded_passes=["concat_insert_dim_elementwise"])
primal_compile = @elapsed compiled_primal = with_large_stack(() ->
    Reactant.@compile compile_options=options window_objective(rmodel, x, rinputs, Δt, steps))
Jc = Float64(compiled_primal(rmodel, x, rinputs, Δt, steps))
parity = abs(Jc - J0) / abs(J0)
note("compiled primal"; Jc, parity, primal_compile)
dmodel = Enzyme.make_zero(rmodel)
gradient_compile = @elapsed compiled_gradient = with_large_stack(() ->
    Reactant.@compile compile_options=options gradient!(rmodel, dmodel, x, dx_shadow, rinputs, Δt, steps))
gradient_seconds = @elapsed (Jg, g) = compiled_gradient(rmodel, dmodel, x, dx_shadow, rinputs, Δt, steps)
g = Array(g); Jg = Float64(Jg)
note("reverse gradient"; Jg, g, gradient_compile, gradient_seconds)

open(joinpath(output, "fd.csv"), "w") do io
    println(io, "control,epsilon,finite_difference,automatic,absolute_error,relative_error")
    for (k, ε, d) in fd
        println(io, join([k, ε, d, g[k], abs(g[k] - d), abs(g[k] - d) / max(abs(d), abs(g[k]), eps())], ','))
    end
end
best = [minimum(abs(g[k] - d) / max(abs(d), abs(g[k]), eps()) for (kk, ε, d) in fd if kk == k) for k in 1:2]
version(name) = string(Pkg.dependencies()[findfirst(p -> p.name == name, Pkg.dependencies())].version)
results = Dict("scope" => "pathwise derivative of a $(steps)-step window from a spun-up state; not the 12 h objective",
    "nx" => nx, "dx_m" => dx, "domain_m" => nx*dx, "nz" => Nz, "order" => order, "halo" => collect(advection_halo(order)),
    "closure" => "none", "precision" => "Float64", "aerosol" => "prognostic aer1", "steps" => steps, "dt" => Δt,
    "radiation_updates_in_window" => 1 + count(n -> n > 0 && n % 30 == 0, 0:steps-1),
    "objective" => "LWP/(50 g m-2) + vapor path below 2.5 km/(1 kg m-2), domain mean, end of window",
    "controls" => "normalized: x1*3 K/day, x2*2 g/kg/day, tapered to zero at 2.5 km",
    "spunup_time_s" => saved.time, "start_lwp_gm2" => cloud_lwp, "sst_K" => saved.sst,
    "J_native" => J0, "J_native_repeat" => J0_repeat, "J_compiled" => Jc, "J_gradient_primal" => Jg, "primal_parity" => parity,
    "gradient" => g, "best_relative_error" => best, "native_window_seconds" => native_seconds,
    "primal_compile_seconds" => primal_compile, "gradient_compile_seconds" => gradient_compile,
    "gradient_run_seconds" => gradient_seconds, "max_rss_GB" => Sys.maxrss() / 1e9,
    "gpu" => CUDA.name(CUDA.device()), "commit" => get(ENV, "ATLANTIC_COMMIT", "unknown"),
    "versions" => Dict(n => version(n) for n in ("Breeze", "Oceananigans", "Reactant", "Enzyme", "CUDA")))
open(io -> TOML.print(io, results), joinpath(output, "gradients.toml"), "w")
passed = all(isfinite, g) && any(!iszero, g) && parity < 1e-8 && all(best .< 1e-2)
println(passed ? "GRADIENT_CHECK_PASS" : "GRADIENT_CHECK_NOT_PASSED", " steps=$steps grid=$(nx)x$(nx)x$(Nz) dx=$(dx) order=$(order) best_rel=$(best) parity=$(parity)")
