# Reactant/Enzyme reverse-mode gradient of the 13:30–15:00 UTC LWP+DWP MA30 observation objective with
# respect to a reduced control basis, checked against native central differences of the SAME function.
#
# Objective (as observation_sensitivity.jl): J = mean(((MA30 LWP − paper)/100)²) + mean(((MA30 DWP − paper)/1)²)
# over the saved fixed masks (h ≥ 12.5 UTC, MA30 ≥95% coverage, no bridging gaps > 2 min). DWP is the
# frozen-cloud-base transect rain path: the cloud-base index per column and record is frozen from the
# native baseline, so this is the CONDITIONAL (frozen-mask) derivative. The moving-mask value is reported
# natively as a diagnostic only (it is piecewise constant in the masks and not differentiated).
# MA30 on the uniform 1-minute model series is linear, so it is applied as a fixed matrix built by
# applying the copied moving() operator to unit vectors.
#
# Controls x (reduced basis, explicit):
#   x[1] tapered heating offset [K/day], x[2] tapered moistening offset [g/kg/day] (time constant, as the
#   training controls), x[3:end] SST offsets [K] at the hourly forcing knots that influence the window.
# Time dependence inside the traced program (no host callbacks):
#   - forcing series: Breeze FieldTimeSeries interpolation on the model clock (in time_step!);
#   - SST: per-step values W*(sst_knots + x_sst) (same linear interpolation as refresh!), written to the
#     surface temperature field shared by surface fluxes and radiation before every step;
#   - solar: per-step cos(zenith) from the same ApparentSolarPosition formula, written into a
#     FixedCosineZenith array before every step (radiation reads it on its 30-step schedule).
# Scope: fixed window IC from a native spinup (the dependence of the IC on the controls through the
# spinup is NOT included: conditional on the IC). Experimental Breeze revision (ratio+stable), not production.
#
# ENV: ATLANTIC_INPUT, ATLANTIC_OUTPUT, ATLANTIC_NX (128), ATLANTIC_DX (100), ATLANTIC_DYNAMICS,
#      ATLANTIC_STATE (window_state.jls from observation_sensitivity.jl, same grid and dynamics),
#      ATLANTIC_WINDOW_MINUTES (90), ATLANTIC_LOOP (checkpoint nest "a,b,c,d": a*b minutes, c*d=60 steps),
#      ATLANTIC_FD_DIRECTIONS (e.g. "1:0.01,1:0.003,3:0.01"), ATLANTIC_SKIP_REACTANT.
using Reactant, Enzyme
include("model.jl")
using Oceananigans: initialize!, prognostic_fields
using Oceananigans.Architectures: ReactantState, on_architecture
using Oceananigans.TimeSteppers: reset!, update_state!
using Breeze.TimeSteppers: prepare_first_time_step!
using Reactant: @trace
using Serialization, Pkg
using LinearAlgebra: I

const T0 = time()
note(msg; kw...) = (@info msg elapsed_s=round(time() - T0; digits=1) kw...; flush(stderr))
gpu_memory() = try readchomp(`nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader`) catch; "unavailable" end

input = ENV["ATLANTIC_INPUT"]; output = mkpath(ENV["ATLANTIC_OUTPUT"])
targets = get(ENV, "ATLANTIC_TARGETS", joinpath(homedir(), "AtlanticBreeze-data", "paper-gradient-targets"))
nx = parse(Int, get(ENV, "ATLANTIC_NX", "128")); dx = grid_spacing(100.0)
saved = deserialize(ENV["ATLANTIC_STATE"])
(saved.nx, saved.dx) == (nx, dx) || error("State is for nx=$(saved.nx) dx=$(saved.dx)")
get(saved, :dynamics, "anelastic") == get(ENV, "ATLANTIC_DYNAMICS", "anelastic") || error("State dynamics $(get(saved, :dynamics, "anelastic")) differs from ATLANTIC_DYNAMICS")
minutes = parse(Int, get(ENV, "ATLANTIC_WINDOW_MINUTES", "90"))
loop = parse.(Int, split(get(ENV, "ATLANTIC_LOOP", "10,9,6,10"), ","))
(loop[1] * loop[2] == minutes && loop[3] * loop[4] == 60) || error("ATLANTIC_LOOP $loop must give $minutes minutes of 60 steps")
Reactant.set_default_backend("gpu")
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = mkpath(joinpath(output, "mlir"))
note("start"; gpu=gpu_memory(), nx, dx, minutes, loop)

config = TOML.parsefile(joinpath(input, "interval.toml")); dt = config["dt"]; dt == 1 || error("expects dt = 1 s")
epoch = DateTime(config["epoch"])
nsteps = 60minutes; t0 = saved.time
utc(t) = 11.5 + t / 3600   # as observation_sensitivity.jl

# Per-step SST interpolation weights and cos(zenith), evaluated at the start of each step (refresh! sets SST
# at the current clock time before the step; radiation evaluates the sun at its update times).
forcing_data = readcsv(joinpath(input, "forcing.csv")); ftimes = unique(forcing_data[:, 1])
sst_knots = [forcing_data[findfirst(==(t), forcing_data[:, 1]), 8] for t in ftimes]
step_times = t0 .+ (0:nsteps-1) .* dt
function interp_weights(times, t)
    w = zeros(length(times)); t <= times[1] && (w[1] = 1; return w); t >= times[end] && (w[end] = 1; return w)
    i = searchsortedlast(times, t); a = (t - times[i]) / (times[i+1] - times[i]); w[i] = 1 - a; w[i+1] = a; return w
end
Wsst_full = reduce(vcat, permutedims.(interp_weights.(Ref(ftimes), step_times)))
active_knots = findall(vec(any(Wsst_full .!= 0; dims=1)))
Wsst = Wsst_full[:, active_knots]                       # nsteps × K
cosz = [max(Breeze.AtmosphereModels.cos_solar_zenith_angle(Breeze.AtmosphereModels.compute_datetime(t, epoch), ENA_COORDINATE...), 0.0) for t in step_times]
nx_controls = 2 + length(active_knots)
note("controls"; basis="[heating K/day, moistening g/kg/day, SST K at knots $(ftimes[active_knots]) s]")

# Fixed observation operators.
paper_lwp = readcsv(joinpath(targets, "paper_fig2_physical_digitized.csv"))
paper_lwp = paper_lwp[(paper_lwp[:, 1] .> 0.1) .& (paper_lwp[:, 1] .< 23.7), :]   # as combined.jl
paper_dwp = readcsv(joinpath(targets, "paper_dwp_pixel_trace.csv"))
function moving(x, y, centres, width)   # copied from observation_sensitivity.jl / combined.jl
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
nrec = minutes + 1; h = utc.(t0 .+ 60 .* (0:minutes))
M = reduce(hcat, [moving(h, Float64.(1:nrec .== n), h, 0.5) for n in 1:nrec])   # nrec × nrec, NaN rows: no support
obs_L = moving(paper_lwp[:, 1], paper_lwp[:, 2], h, 0.5); obs_D = moving(paper_dwp[:, 1], paper_dwp[:, 2], h, 0.5)
supported = vec(all(isfinite, M; dims=2))

# Native model with the traced-style solar position (array, updated per step).
lookup_search = NumericalRadiation.FixedBinarySearch()
build_native = @elapsed native = make_case(input; nx, dx, architecture=GPU(), solar_position=FixedCosineZenith(on_architecture(GPU(), fill(cosz[1], nx, nx))), lookup_search)
model = native.model; dz = diff(native.faces); nz = length(dz); jt = nx ÷ 2
note("native model built"; build_native, dynamics=native.dynamics, gpu=gpu_memory())
hz = advection_halo(native.order)[3]

energy_correction(m) = getfield(getfield(m.forcing.ρE, 1)[end], 1).tls
moisture_correction(m) = getfield(getfield(m.forcing.ρqᵛ, 1)[end], 1).qls
surface_temperature(m) = m.radiation.surface_radiation.surface_temperature
cos_zenith_array(m) = m.radiation.solar_position.cos_zenith

# Native diagnostic baseline: frozen cloud base and masks (host loop, same per-step updates as the objective).
function restore!(m, fields, time)
    for (name, f) in pairs(prognostic_fields(m)); interior(f) .= fields[String(name)]; end
    reset!(m.clock); m.clock.time = time
    initialize!(m); update_state!(m)
end
function native_diagnostic(x)
    restore!(model, Dict(n => on_architecture(GPU(), a) for (n, a) in saved.fields), t0)
    parent(energy_correction(model).data) .= on_architecture(GPU(), reshape(vcat(zeros(hz), native.taper .* x[1] ./ 86400, zeros(hz)), 1, 1, :, 1))
    parent(moisture_correction(model).data) .= on_architecture(GPU(), reshape(vcat(zeros(hz), native.taper .* x[2] ./ (1000 * 86400), zeros(hz)), 1, 1, :, 1))
    sst = Wsst * (sst_knots[active_knots] .+ x[3:end])
    prepare_first_time_step!(model, dt)
    lwp = Float64[]; base = Matrix{Int}(undef, nx, 0); rain = Array{Float64}(undef, nx, nz, 0)
    rec() = begin
        w = model.microphysical_fields; cloud = Array(interior(w.ρqᶜˡ)); ρqr = Array(interior(w.ρqʳ))
        push!(lwp, 1000sum(vec(mean(cloud .+ ρqr .+ Array(interior(w.ρqʷⁱ)); dims=(1, 2))) .* dz))
        qc = cloud[:, jt, :] ./ reshape(native.density, 1, :)
        base = hcat(base, [something(findfirst(>(1e-5), qc[i, :]), 0) for i in 1:nx]); rain = cat(rain, ρqr[:, jt, :]; dims=3)
    end
    rec()
    for s in 1:nsteps
        parent(surface_temperature(model)) .= sst[s]; cos_zenith_array(model) .= cosz[s]
        time_step!(model, dt)
        s % 60 == 0 && rec()
    end
    return (; lwp, base, rain)
end
dwp_series(rain, base) = [let c = findall(>(0), base[:, n]); isempty(c) ? NaN : mean(1000sum(rain[i, 1:base[i, n]-1, n] .* dz[1:base[i, n]-1]) for i in c) end for n in axes(base, 2)]
apply_M(y) = [supported[n] ? sum(M[n, :] .* y) : NaN for n in 1:nrec]
x0 = zeros(nx_controls)
diag_time = @elapsed d0 = native_diagnostic(x0)
frozen = d0.base
L_mask = supported .& isfinite.(obs_L) .& (h .>= 12.5)
D_mask = L_mask .& isfinite.(obs_D) .& isfinite.(apply_M(dwp_series(d0.rain, frozen)))
note("native diagnostic baseline"; diag_time, nL=count(L_mask), nD=count(D_mask), utc_scored=(minimum(h[L_mask]), maximum(h[L_mask])))
function J_from_series(lwp, dwp)
    mL = apply_M(lwp); mD = apply_M(dwp)
    return mean(((mL[L_mask] .- obs_L[L_mask]) ./ 100) .^ 2) + mean((mD[D_mask] .- obs_D[D_mask]) .^ 2)
end
J_diag = J_from_series(d0.lwp, dwp_series(d0.rain, frozen))
J_moving = J_from_series(d0.lwp, dwp_series(d0.rain, d0.base))   # equal at baseline by construction

# Weights for the traced objective.
lwp_weights = reshape(1000 .* dz ./ nx^2, 1, 1, :)
ncols = [count(>(0), frozen[:, n]) for n in 1:nrec]
dwp_weights = zeros(nx, nz, nrec)    # transect (i, k) weight per record: 1000 dz_k [k < base] / ncols
for n in 1:nrec, i in 1:nx
    b = frozen[i, n]; (b > 0 && ncols[n] > 0) || continue
    dwp_weights[i, 1:b-1, n] .= 1000 .* dz[1:b-1] ./ ncols[n]
end
ML = M[L_mask, :]; MD = M[D_mask, :]

function window_inputs(arch)
    on(a) = arch isa ReactantState ? Reactant.to_rarray(a) : on_architecture(arch, a)
    padded(v) = reshape(vcat(zeros(hz), v, zeros(hz)), 1, 1, :, 1)
    return (; fields=Dict(n => on(a) for (n, a) in saved.fields), time=t0,
              energy_taper=on(padded(native.taper ./ 86400)), moisture_taper=on(padded(native.taper ./ (1000 * 86400))),
              Wsst=on(Wsst), sst_knots=on(sst_knots[active_knots]), cosz=on(cosz),
              lwp_weights=on(lwp_weights), dwp_weights=on(dwp_weights), ML=on(ML), MD=on(MD),
              obs_L=on(obs_L[L_mask]), obs_D=on(obs_D[D_mask]), onehot=on(Matrix{Float64}(I(nrec))),
              recidx=on(collect(1.0:nrec)), stepidx=on(collect(1.0:length(cosz))))
end
function window_objective(m, x, inp, Δt, loop)
    for (name, f) in pairs(prognostic_fields(m)); interior(f) .= inp.fields[String(name)]; end
    parent(energy_correction(m).data) .= inp.energy_taper .* reshape(x[1:1], 1, 1, 1, 1)
    parent(moisture_correction(m).data) .= inp.moisture_taper .* reshape(x[2:2], 1, 1, 1, 1)
    sst = inp.Wsst * (inp.sst_knots .+ x[3:end])
    reset!(m.clock); m.clock.time = inp.time
    initialize!(m); update_state!(m); prepare_first_time_step!(m, Δt)
    w = m.microphysical_fields
    # selections by masked reductions, never indexing with a traced loop counter
    function record(n)
        sel = ifelse.(inp.recidx .== n, 1.0, 0.0)
        wd = dropdims(sum(inp.dwp_weights .* reshape(sel, 1, 1, :); dims=3); dims=3)
        return (sum((interior(w.ρqᶜˡ) .+ interior(w.ρqʳ) .+ interior(w.ρqʷⁱ)) .* inp.lwp_weights) .* sel,
                sum(view(interior(w.ρqʳ), :, size(inp.dwp_weights, 1) ÷ 2, :) .* wd) .* sel)
    end
    l1, d1 = record(1); lwp = l1; dwp = d1
    a, b, c, e = loop
    @trace mincut=true checkpointing=true track_numbers=false for ia in 1:a
        @trace mincut=true checkpointing=true track_numbers=false for ib in 1:b
            mi = (ia - 1) * b + ib
            @trace mincut=true checkpointing=true track_numbers=false for ic in 1:c
                @trace mincut=true checkpointing=true track_numbers=false for ie in 1:e
                    s = (mi - 1) * 60 + (ic - 1) * e + ie
                    # masked reductions: traced ranges s:s fail (Reactant iota MethodError, job 17047)
                    sel = ifelse.(inp.stepidx .== s, 1.0, 0.0)
                    parent(surface_temperature(m)) .= sum(sst .* sel)
                    cos_zenith_array(m) .= sum(inp.cosz .* sel)
                    time_step!(m, Δt)
                end
            end
            ln, dn = record(mi + 1); lwp = lwp .+ ln; dwp = dwp .+ dn
        end
    end
    rL = (inp.ML * lwp .- inp.obs_L) ./ 100; rD = inp.MD * dwp .- inp.obs_D
    return sum(rL .^ 2) / length(rL) + sum(rD .^ 2) / length(rD)
end
# NOTE: the per-cell transect view uses the centre row like observation_sensitivity.jl (j = nx÷2).

inputs = window_inputs(GPU())
J(x) = window_objective(model, on_architecture(GPU(), x), inputs, dt, loop)
t_native = @elapsed J0 = J(x0)
t_native_warm = @elapsed J0r = J(x0)
note("native objective"; J0, J0_repeat=J0r, J_diag, J_moving_baseline=J_moving, first_s=t_native, warm_s=t_native_warm)
results = Dict{String, Any}("scope" => "frozen-mask conditional derivative of the 13:30-15:00 UTC MA30 LWP+DWP objective, fixed native-spinup IC (spinup dependence excluded); experimental Breeze ratio+stable",
    "dynamics" => string(native.dynamics), "nx" => nx, "dx" => dx, "nz" => nz, "dt" => dt, "steps" => nsteps, "loop" => loop,
    "radiation_updates" => count(n -> n % native.radiation_iterations == 0, 0:nsteps-1),
    "controls" => "heating K/day, moistening g/kg/day (tapered, time-constant), SST K at knots", "sst_knot_times" => ftimes[active_knots],
    "J_native" => J0, "J_native_repeat" => J0r, "J_diagnostic_loop" => J_diag, "nL" => count(L_mask), "nD" => count(D_mask),
    "native_first_s" => t_native, "native_warm_s" => t_native_warm, "build_native_s" => build_native,
    "gpu" => gpu_memory(), "commit" => get(ENV, "ATLANTIC_COMMIT", "unknown"), "breeze" => pkgdir(Breeze))
save() = open(io -> TOML.print(io, results), joinpath(output, "window_gradient.toml"), "w"); save()

# Native central differences of the same objective.
fd = []
for spec in filter(!isempty, split(get(ENV, "ATLANTIC_FD_DIRECTIONS", "1:0.01,1:0.003,3:0.01"), ","))
    k, ε = parse(Int, split(spec, ":")[1]), parse(Float64, split(spec, ":")[2])
    e = zeros(nx_controls); e[k] = ε
    Jp = J(x0 .+ e); Jm = J(x0 .- e)
    push!(fd, (k, ε, (Jp - Jm) / 2ε, Jp, Jm)); note("FD"; k, ε, slope=(Jp - Jm) / 2ε, Jp, Jm)
    results["fd"] = [Dict("control" => a, "epsilon" => b, "slope" => c, "J_plus" => p, "J_minus" => q) for (a, b, c, p, q) in fd]; save()
end
get(ENV, "ATLANTIC_SKIP_REACTANT", "false") == "true" && (println("WINDOW_NATIVE_DONE"); exit())

function with_large_stack(f); task = Task(f, 512 * 1024^2); schedule(task); return fetch(task); end
note("building Reactant model"; gpu=gpu_memory())
build_r = @elapsed rcase = with_large_stack(() -> CUDA.@allowscalar make_case(input; nx, dx, architecture=ReactantState(),
    solar_position=FixedCosineZenith(Reactant.to_rarray(fill(cosz[1], nx, nx))), lookup_search))
rmodel = rcase.model; rinputs = window_inputs(ReactantState())
x = Reactant.to_rarray(x0); gx = Reactant.to_rarray(zeros(nx_controls))
options = Reactant.CompileOptions(raise=true, raise_first=true, sync=true, excluded_passes=["concat_insert_dim_elementwise"])
results["reactant_build_s"] = build_r; save()
note("tracing primal (code_hlo)"; build_r, gpu=gpu_memory())
trace_s = @elapsed with_large_stack(() -> Reactant.@code_hlo compile_options=options window_objective(rmodel, x, rinputs, dt, loop))
results["primal_trace_s"] = trace_s; save()
note("compiling primal"; trace_s)
pc = @elapsed cprimal = with_large_stack(() -> Reactant.@compile compile_options=options window_objective(rmodel, x, rinputs, dt, loop))
t1 = @elapsed Jc = Float64(cprimal(rmodel, x, rinputs, dt, loop))
t2 = @elapsed Jc2 = Float64(cprimal(rmodel, x, rinputs, dt, loop))
results["primal_compile_s"] = pc; results["J_compiled"] = Jc; results["J_compiled_repeat"] = Jc2
results["primal_parity"] = abs(Jc - J0) / abs(J0); results["primal_first_exec_s"] = t1; results["primal_warm_exec_s"] = t2; save()
note("compiled primal"; Jc, Jc2, parity=results["primal_parity"], pc, t1, t2, gpu=gpu_memory())

function gradient!(m, dm, x, gx, inp, Δt, loop)
    Enzyme.remake_zero!(dm); gx .= 0
    _, Jv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), window_objective, Enzyme.Active,
        Enzyme.Duplicated(m, dm), Enzyme.Duplicated(x, gx), Enzyme.Const(inp), Enzyme.Const(Δt), Enzyme.Const(loop))
    return Jv, gx
end
dmodel = Enzyme.make_zero(rmodel)
note("compiling reverse")
gc = @elapsed cgrad = with_large_stack(() -> Reactant.@compile compile_options=options gradient!(rmodel, dmodel, x, gx, rinputs, dt, loop))
results["reverse_compile_s"] = gc; save()
g1 = @elapsed begin (Jg, g) = cgrad(rmodel, dmodel, x, gx, rinputs, dt, loop); g = Array(g); Jg = Float64(Jg) end
g2 = @elapsed begin (Jg2, gg) = cgrad(rmodel, dmodel, x, gx, rinputs, dt, loop); gg = Array(gg) end
results["gradient"] = g; results["gradient_repeat_identical"] = g == gg; results["J_gradient_primal"] = Jg
results["reverse_first_exec_synchronized_s"] = g1; results["reverse_warm_exec_synchronized_s"] = g2
results["gradient_finite_nonzero"] = all(isfinite, g) && any(!iszero, g)
results["fd_vs_ad"] = [Dict("control" => k, "epsilon" => ε, "fd" => s, "ad" => g[k], "rel" => abs(g[k] - s) / max(abs(s), abs(g[k]), eps())) for (k, ε, s) in fd]
results["max_rss_GB"] = Sys.maxrss() / 1e9; results["gpu_end"] = gpu_memory(); save()
note("reverse gradient"; g, Jg, gc, g1, g2)
println("WINDOW_GRADIENT_DONE finite_nonzero=", results["gradient_finite_nonzero"], " parity=", results["primal_parity"], " fd_vs_ad=", results["fd_vs_ad"])
