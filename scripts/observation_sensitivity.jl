# Native (no AD) directional sensitivities of a combined LWP + DWP observation loss to the forcing
# time series (dT/dt, dq/dt) and the SST time series, by paired central finite differences on a reduced
# case. This is NOT a Jacobian and NOT an AD result: each direction perturbs all hourly forcing knots
# (or one SST knot) at once.
#
# Loss over one cloudy window [t0, t0+window] from a natively spun-up state held FIXED across runs (the
# window's initial condition does not respond to the controls; spinup dependence is excluded):
#   J = mean_L[((MA30 LWP_model - MA30 LWP_paper)/100 g m⁻²)²] + mean_D[((MA30 DWP_model - MA30 DWP_paper)/1 g m⁻²)²]
# The 100 g m⁻² and 1 g m⁻² scales are chosen balancing weights, not measurement errors. MA30 is the centred
# 30-minute piecewise-linear moving mean (≥95% coverage, no bridging gaps >2 min, no extrapolation), as in
# paper-gradient-targets/combined.jl. Time masks L and D are fixed from the baseline run and exclude the
# first hour of the interval.
#   LWP model: domain-mean cloud + rain + ice-liquid water path (as forward.jl).
#   DWP model operator (a proxy, no radar/lidar detection operator): along the XZ transect j = ny÷2, per
#   column the rain path 1000 Σ ρqʳ dz below the lower face of the first cell with qᶜˡ > 1e-5 kg/kg; the
#   conditional mean over cloudy columns; missing (not zero) when no column is cloudy.
#   Two policies are reported: "frozen" freezes the baseline cloud-base index and cloudy-column set at each
#   time (a conditional, frozen-mask derivative, not the derivative of the full operator); "moving" recomputes
#   them in each run (the full operator, which is nondifferentiable where cloud bases or columns switch).
# Production details kept: live ecCKD radiation on its 30 s cadence, time-dependent apparent solar position,
# prognostic aerosol P3, closure=nothing, SST shared by surface fluxes and radiation (one field, refreshed
# every step from the interpolated, perturbed SST knots).
# ENV: ATLANTIC_INPUT (production interval, forcing SHA checked), ATLANTIC_OUTPUT, ATLANTIC_NX (16),
#      ATLANTIC_DX (200), ATLANTIC_WINDOW_START (7200 s), ATLANTIC_WINDOW (5400 s),
#      ATLANTIC_DIRECTIONS (heating,moistening; also sst, sst_knot:<time_s>), ATLANTIC_EPSILONS (1,0.3,0.1),
#      ATLANTIC_STATE (reuse a window state written by an earlier run), ATLANTIC_TARGETS.
include("model.jl")
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: update_state!
using Serialization
note(msg; kw...) = (@info msg kw...; flush(stderr))

const PRODUCTION_FORCING_SHA = "ec1f2c1962fc25138f4fde0028a325404df3fec20536b65dc90ba17bf7770405"
sha(path) = first(split(read(`sha256sum $path`, String)))
input = ENV["ATLANTIC_INPUT"]; output = mkpath(ENV["ATLANTIC_OUTPUT"])
sha(joinpath(input, "forcing.csv")) == PRODUCTION_FORCING_SHA || error("forcing.csv is not the production forcing")
targets = get(ENV, "ATLANTIC_TARGETS", joinpath(homedir(), "AtlanticBreeze-data", "paper-gradient-targets"))
nx = parse(Int, get(ENV, "ATLANTIC_NX", "16")); dx = grid_spacing(200.0)
t0 = parse(Float64, get(ENV, "ATLANTIC_WINDOW_START", "7200")); window = parse(Float64, get(ENV, "ATLANTIC_WINDOW", "5400"))
t0 >= 3600 || error("The window must start after the excluded first hour")
directions = split(get(ENV, "ATLANTIC_DIRECTIONS", "heating,moistening"), ",")
epsilons = parse.(Float64, split(get(ENV, "ATLANTIC_EPSILONS", "1,0.3,0.1"), ","))
state_file = get(ENV, "ATLANTIC_STATE", joinpath(output, "window_state.jls"))
config = TOML.parsefile(joinpath(input, "interval.toml")); dt = config["dt"]
utc(t) = 11.5 + t / 3600   # hours UTC; t is seconds since the 11:30 UTC launch

# 1. Window initial state: native spinup with the production forcing, written once and then reused.
if isfile(state_file)
    saved = deserialize(state_file)
    (saved.nx, saved.dx, saved.time, saved.forcing_sha) == (nx, dx, t0, PRODUCTION_FORCING_SHA) || error("State file is for another configuration")
else
    case = make_case(input; nx, dx)
    simulation = Simulation(case.model; Δt=dt, stop_time=t0)
    add_callback!(simulation, case.refresh!, IterationInterval(1))
    wall = @elapsed run!(simulation)
    m = case.model
    saved = (; nx, dx, order=case.order, time=m.clock.time, iteration=m.clock.iteration, forcing_sha=PRODUCTION_FORCING_SHA,
               wall, fields=Dict(String(n) => Array(interior(f)) for (n, f) in pairs(prognostic_fields(m))))
    serialize(state_file, saved)
    note("spinup"; nx, dx, t0, wall)
end

# 2. One window run with a given forcing perturbation; records every 60 s of model time.
function run_window(perturb)
    wall0 = time()
    case = make_case(input; nx, dx, forcing_perturbation=perturb)
    m = case.model; grid = m.grid; dz = diff(case.faces); nz = length(dz); j = nx ÷ 2
    for (n, f) in pairs(prognostic_fields(m)); set!(f, saved.fields[String(n)]); end
    m.clock.time = saved.time; m.clock.iteration = saved.iteration
    case.refresh!((; model=m)); update_state!(m); CUDA.synchronize()
    build = time() - wall0
    times = Float64[]; lwp = Float64[]; cfl = Float64[]; base = Matrix{Int}(undef, nx, 0); rain = Array{Float64}(undef, nx, nz, 0)
    function record(sim)
        t = sim.model.clock.time
        (!isempty(times) && t <= times[end]) && return
        w = sim.model.microphysical_fields
        cloud = Array(interior(w.ρqᶜˡ)); ρqr = Array(interior(w.ρqʳ))
        liquid = cloud .+ ρqr .+ Array(interior(w.ρqʷⁱ))
        value = 1000sum(vec(mean(liquid; dims=(1, 2))) .* dz)
        isfinite(value) || error("Nonfinite LWP at t=$t")
        U = max(maximum(abs, interior(sim.model.velocities.u)), maximum(abs, interior(sim.model.velocities.v)))
        c = dt * (2U / dx + maximum(abs, interior(sim.model.velocities.w)) / minimum(dz))
        c < 0.8 || error("CFL $c exceeds 0.8 at t=$t"); push!(cfl, c)
        qc = cloud[:, j, :] ./ reshape(case.density, 1, :)
        kb = [something(findfirst(>(1e-5), qc[i, :]), 0) for i in 1:nx]   # 0: clear column
        push!(times, t); push!(lwp, value); base = hcat(base, kb); rain = cat(rain, ρqr[:, j, :]; dims=3)
    end
    simulation = Simulation(m; Δt=dt, stop_time=t0 + window)
    add_callback!(simulation, case.refresh!, IterationInterval(1))
    add_callback!(simulation, record, IterationInterval(round(Int, 60 / dt)))
    record(simulation); t_run = time(); run!(simulation); CUDA.synchronize(); t_run = time() - t_run; record(simulation)
    # build includes model construction (and native JIT on the first run); run is the synchronized window
    # integration (it includes native JIT of the time step on the first run only; later runs are warm).
    return (; times, lwp, base, rain, dz, cfl_max=maximum(cfl), build, run=t_run, steps=round(Int, window / dt), wall=time() - wall0)
end

# DWP along the transect for given cloud-base indices (0 = clear; missing when no column is cloudy).
function dwp(rain, base, dz)
    map(axes(base, 2)) do n
        cols = findall(>(0), base[:, n])
        isempty(cols) ? NaN : mean(1000sum(rain[i, 1:base[i, n]-1, n] .* dz[1:base[i, n]-1]) for i in cols)
    end
end

# Centred moving mean, copied from paper-gradient-targets/combined.jl (hours in, hours out).
function moving(x, y, centres, width)
    result = fill(NaN, length(centres))
    for (j, c) in enumerate(centres)
        a = c - width/2; b = c + width/2; integ = 0.0; covered = 0.0
        (a < x[1] || b > x[end]) && continue
        for i in 1:length(x)-1
            isfinite(y[i]) && isfinite(y[i+1]) && x[i+1] - x[i] <= 2/60 + 1e-8 || continue
            lo = max(a, x[i]); hi = min(b, x[i+1]); hi > lo || continue
            yl = y[i] + (y[i+1] - y[i]) * (lo - x[i]) / (x[i+1] - x[i]); yh = y[i] + (y[i+1] - y[i]) * (hi - x[i]) / (x[i+1] - x[i])
            integ += (yl + yh) / 2 * (hi - lo); covered += hi - lo
        end
        covered >= 0.95width && (result[j] = integ / covered)
    end
    return result
end
paper_lwp = readcsv(joinpath(targets, "paper_fig2_physical_digitized.csv"))
paper_lwp = paper_lwp[(paper_lwp[:, 1] .> 0.1) .& (paper_lwp[:, 1] .< 23.7), :]   # as combined.jl
paper_dwp = readcsv(joinpath(targets, "paper_dwp_pixel_trace.csv"))

function evaluate(run, frozen_base, masks)
    h = utc.(run.times)
    model_L = moving(h, run.lwp, h, 0.5); obs_L = moving(paper_lwp[:, 1], paper_lwp[:, 2], h, 0.5)
    obs_D = moving(paper_dwp[:, 1], paper_dwp[:, 2], h, 0.5)
    out = Dict{String, Any}("utc" => h, "model_lwp_ma30" => model_L, "paper_lwp_ma30" => obs_L, "paper_dwp_ma30" => obs_D)
    for (policy, b) in (("frozen", frozen_base), ("moving", run.base))
        out["model_dwp_ma30_$policy"] = moving(h, dwp(run.rain, b, run.dz), h, 0.5)
    end
    isnothing(masks) && (masks = (; L=isfinite.(model_L) .& isfinite.(obs_L) .& (h .>= 12.5),
                                    D=isfinite.(out["model_dwp_ma30_frozen"]) .& isfinite.(obs_D) .& (h .>= 12.5)))
    rL = model_L[masks.L] .- obs_L[masks.L]
    out["lwp_term"] = mean((rL ./ 100) .^ 2); out["lwp_rmse_gm2"] = sqrt(mean(rL .^ 2)); out["lwp_bias_gm2"] = mean(rL)
    for policy in ("frozen", "moving")
        rD = out["model_dwp_ma30_$policy"][masks.D] .- obs_D[masks.D]
        out["dwp_term_$policy"] = mean(rD .^ 2); out["dwp_bias_gm2_$policy"] = mean(rD)   # NaN if a masked time lost its cloud
        out["J_$policy"] = out["lwp_term"] + out["dwp_term_$policy"]
    end
    return out, masks
end

# 3. Directions in physical units per unit ε: heating 1 K/day and moistening 1 g/kg/day (tapered to zero by
# 2500 m as the training controls), SST 1 K, at every hourly knot or at one SST knot.
taper(z) = clamp((TAPER_TOP - z) / 1000, 0, 1)
function direction(name, ε)
    function perturb(F)
        F = copy(F)
        if name == "heating"; F[:, 3] .+= ε .* taper.(F[:, 2]) ./ 86400
        elseif name == "moistening"; F[:, 4] .+= ε .* taper.(F[:, 2]) ./ (1000 * 86400)
        elseif name == "sst"; F[:, 8] .+= ε
        elseif startswith(name, "sst_knot:"); knot = parse(Float64, split(name, ":")[2])
            any(F[:, 1] .== knot) || error("No forcing knot at $knot s"); F[F[:, 1] .== knot, 8] .+= ε
        else error("Unknown direction $name") end
        return F
    end
    return perturb
end
units = Dict("heating" => "K/day (tapered)", "moistening" => "g/kg/day (tapered)", "sst" => "K")

baseline = run_window(identity)
frozen_base = baseline.base
base_terms, masks = evaluate(baseline, frozen_base, nothing)
note("baseline"; wall=baseline.wall, J_frozen=base_terms["J_frozen"], J_moving=base_terms["J_moving"],
     lwp_term=base_terms["lwp_term"], dwp_term=base_terms["dwp_term_frozen"], nL=count(masks.L), nD=count(masks.D),
     cloudy_fraction=count(>(0), baseline.base) / length(baseline.base))
repeat = run_window(identity); rep_terms, _ = evaluate(repeat, frozen_base, masks)
note("baseline repeat"; ΔJ=rep_terms["J_frozen"] - base_terms["J_frozen"], max_Δlwp=maximum(abs, repeat.lwp .- baseline.lwp))
timing(r, label) = Dict("label" => label, "build_s" => r.build, "run_s" => r.run, "wall_s" => r.wall, "steps" => r.steps,
                         "s_per_step" => r.run / r.steps, "cfl_max" => r.cfl_max)
timings = [timing(baseline, "baseline (cold: includes JIT)"), timing(repeat, "baseline repeat (warm)")]
results = Dict{String, Any}("timings" => timings, "gpu" => CUDA.name(CUDA.device()), "cells" => nx * nx * length(baseline.dz),
    "dt" => dt, "spinup_wall_s" => saved.wall, "baseline" => base_terms, "baseline_repeat_dJ" => rep_terms["J_frozen"] - base_terms["J_frozen"],
    "masks" => masks, "frozen_base" => frozen_base, "window" => (t0, t0 + window), "nx" => nx, "dx" => dx,
    "forcing_sha" => PRODUCTION_FORCING_SHA, "commit" => get(ENV, "ATLANTIC_COMMIT", "unknown"), "directions" => Dict())
save() = serialize(joinpath(output, "observation_sensitivity.jls"), results)
save()
for name in directions, ε in epsilons
    rp = run_window(direction(name, ε)); rm = run_window(direction(name, -ε))
    push!(timings, timing(rp, "$name +$ε")); push!(timings, timing(rm, "$name -$ε"))
    plus, _ = evaluate(rp, frozen_base, masks); minus, _ = evaluate(rm, frozen_base, masks)
    # Mask audit: transect cells (column × time) whose cloud-base index, or cloudy/clear status, differs from
    # the baseline. The frozen policy ignores these switches; the moving policy sees them. (P3 limiter and
    # other internal branch switches are not counted here.)
    switches(r) = (; base=count(r.base .!= frozen_base), cloudy=count((r.base .> 0) .!= (frozen_base .> 0)), total=length(frozen_base))
    row = Dict{String, Any}("plus" => plus, "minus" => minus, "switches_plus" => switches(rp), "switches_minus" => switches(rm))
    # Taylor audit about the baseline J0: one-sided increments and the symmetric (second-order) residual.
    for key in ("J_frozen", "J_moving")
        J0 = base_terms[key]
        row["taylor_$key"] = (; up=plus[key] - J0, down=J0 - minus[key], second=(plus[key] - 2J0 + minus[key]) / ε^2)
    end
    for key in ("J_frozen", "J_moving", "lwp_term", "dwp_term_frozen", "dwp_term_moving")
        row["d$key"] = (plus[key] - minus[key]) / 2ε
    end
    results["directions"][(name, ε)] = row; save()
    unit = get(units, split(name, ":")[1], "K")
    println("FD ", name, " ε=", ε, " [", unit, "] dJ_frozen/dε=", row["dJ_frozen"], " dJ_moving/dε=", row["dJ_moving"],
            " dLWPterm=", row["dlwp_term"], " dDWPterm_frozen=", row["ddwp_term_frozen"],
            " J+=", plus["J_frozen"], " J-=", minus["J_frozen"], " pair_wall_s=", rp.wall + rm.wall)
    println("audit ", name, " ε=", ε, " switches+=", row["switches_plus"], " switches-=", row["switches_minus"],
            " taylor_frozen=", row["taylor_J_frozen"], " taylor_moving=", row["taylor_J_moving"]); flush(stdout)
end
for t in timings; println("timing ", t); end
println("OBSERVATION_SENSITIVITY_DONE directions=", join(directions, ","), " epsilons=", epsilons)
