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
# ATLANTIC_CONTROLS = reduced (default: 2 tapered amplitudes + SST knots) | inputs (native forcing input arrays).
# inputs: x = vcat(vec(δ_f) for f in ATLANTIC_INPUT_FIELDS), δ_f[k, j] the additive change of the forcing.csv
# value of field f at model level k and forcing knot ftimes[active_knots[j]], in that file's native units.
# SST is NOT a control in inputs mode: prescribed SST (unperturbed knots) as in the matched forward run.
const CONTROLS = get(ENV, "ATLANTIC_CONTROLS", "reduced")
const INPUT_UNITS = Dict("T" => "K s-1", "q" => "kg kg-1 s-1", "w" => "m s-1", "ug" => "m s-1", "vg" => "m s-1")
const INPUT_COLUMN = Dict("T" => "dT_Ks", "q" => "dq_kgkgs", "w" => "w_ms", "ug" => "ug_ms", "vg" => "vg_ms")
input_fields = CONTROLS == "inputs" ? String.(split(get(ENV, "ATLANTIC_INPUT_FIELDS", "T,q"), ",")) : String[]
all(f -> haskey(INPUT_UNITS, f), input_fields) || error("ATLANTIC_INPUT_FIELDS must be among $(keys(INPUT_UNITS))")
nk = length(active_knots)
nzc = size(readcsv(joinpath(input, "initial.csv")), 1)
nx_controls = CONTROLS == "inputs" ? length(input_fields) * nzc * nk : 2 + nk
block(x, i) = x[(i - 1) * nzc * nk + 1:i * nzc * nk]
note("controls"; mode=CONTROLS, basis=CONTROLS == "inputs" ?
     "native input arrays $(input_fields) at $nzc levels × knots $(ftimes[active_knots]) s, SST prescribed" :
     "[heating K/day, moistening g/kg/day, SST K at knots $(ftimes[active_knots]) s]", nx_controls)

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
# The model's own references to the native input series (same pattern as energy_correction): order in model.jl
# :E => (thermo.E, correction.E); :qᵛ => (thermo.qᵛ, subsidence, correction.qᵛ); :u => (geostrophic.u, …); :v => (geostrophic.v, …).
# TimeVaryingGeostrophicForcing for u holds vᵍ and for v holds uᵍ. The subsidence velocity series is one object
# shared by every advected field. Identity with native.input_series is asserted below.
forcing_term(m, name, i) = getfield(getfield(getproperty(m.forcing, name), 1)[i], 1)
input_series_of(m) = (; T=forcing_term(m, :ρE, 1).tls, q=forcing_term(m, :ρqᵛ, 1).qls, w=forcing_term(m, :ρqᵛ, 2).velocity,
                       ug=forcing_term(m, :ρv, 1).geostrophic_velocity, vg=forcing_term(m, :ρu, 1).geostrophic_velocity)
let own = input_series_of(model)
    for f in (:T, :q, :w, :ug, :vg)
        parent(getproperty(own, f).data) === parent(getproperty(native.input_series, f).data) ||
            error("model forcing $f does not alias make_case input_series.$f")
    end
    note("input series alias check passed"; fields=(:T, :q, :w, :ug, :vg))
end

# Input-array embedding: parent(fts.data) is (1, 1, Nz + 2H, Nt); E places the nz interior levels, S the active knots.
series_parent(s) = parent(s.data)
base_parent = Dict(f => Array(series_parent(getproperty(native.input_series, Symbol(f)))) for f in input_fields)
for f in input_fields
    sz = size(base_parent[f]); (sz[1], sz[2], sz[4]) == (1, 1, length(ftimes)) && (sz[3] - nz) % 2 == 0 ||
        error("unexpected FieldTimeSeries parent size $sz for $f")
end
function control_inputs(arch)
    on(a) = arch isa ReactantState ? Reactant.to_rarray(a) : on_architecture(arch, a)
    isempty(input_fields) && return (; base=(), E=on(zeros(1, 1)), S=on(zeros(1, 1)))
    Nzp = size(first(values(base_parent)), 3); off = (Nzp - nz) ÷ 2
    E = zeros(Nzp, nz); for k in 1:nz; E[off + k, k] = 1; end
    S = zeros(nk, length(ftimes)); for (j, a) in enumerate(active_knots); S[j, a] = 1; end
    return (; base=Tuple(on(base_parent[f]) for f in input_fields), E=on(E), S=on(S))
end
# Exact embedding (0/1 matrices): with δ = 0 the arrays are the unperturbed forcing.csv values.
function set_inputs!(m, series, x, ci)
    for (i, f) in enumerate(input_fields)
        δ = reshape(block(x, i), nz, nk)
        P = ci.E * δ * ci.S
        series_parent(getproperty(series, Symbol(f))) .= ci.base[i] .+ reshape(P, 1, 1, size(P, 1), size(P, 2))
    end
    return nothing
end

# Native diagnostic baseline: frozen cloud base and masks (host loop, same per-step updates as the objective).
function restore!(m, fields, time)
    for (name, f) in pairs(prognostic_fields(m)); interior(f) .= fields[String(name)]; end
    reset!(m.clock); m.clock.time = time
    initialize!(m); update_state!(m)
end
function native_diagnostic(x)
    restore!(model, Dict(n => on_architecture(GPU(), a) for (n, a) in saved.fields), t0)
    xr = CONTROLS == "inputs" ? zeros(2) : x
    parent(energy_correction(model).data) .= on_architecture(GPU(), reshape(vcat(zeros(hz), native.taper .* xr[1] ./ 86400, zeros(hz)), 1, 1, :, 1))
    parent(moisture_correction(model).data) .= on_architecture(GPU(), reshape(vcat(zeros(hz), native.taper .* xr[2] ./ (1000 * 86400), zeros(hz)), 1, 1, :, 1))
    sst = CONTROLS == "inputs" ? Wsst * sst_knots[active_knots] : Wsst * (sst_knots[active_knots] .+ x[3:end])
    CONTROLS == "inputs" && set_inputs!(model, native.input_series, on_architecture(GPU(), x), control_inputs(GPU()))
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
              recidx=on(collect(1.0:nrec)), stepidx=on(collect(1.0:length(cosz))),
              ci=control_inputs(arch), wLD=on([1.0, 1.0]))   # J = wL·J_LWP + wD·J_DWP (component gradients by reseeding)
end
# ATLANTIC_CKPT: nested (mincut+checkpointing on all four loops) | nomincut | none — trace bisection (17115)
const CKPT = get(ENV, "ATLANTIC_CKPT", "nested")
const MC = CKPT == "nested"; const CP = CKPT != "none"
@eval function window_terms(m, x, inp, Δt, loop)
    for (name, f) in pairs(prognostic_fields(m)); interior(f) .= inp.fields[String(name)]; end
    if $(CONTROLS == "inputs")
        parent(energy_correction(m).data) .= 0; parent(moisture_correction(m).data) .= 0
        set_inputs!(m, input_series_of(m), x, inp.ci)
        sst = inp.Wsst * inp.sst_knots
    else
        parent(energy_correction(m).data) .= inp.energy_taper .* reshape(x[1:1], 1, 1, 1, 1)
        parent(moisture_correction(m).data) .= inp.moisture_taper .* reshape(x[2:2], 1, 1, 1, 1)
        sst = inp.Wsst * (inp.sst_knots .+ x[3:end])
    end
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
    @trace mincut=$MC checkpointing=$CP track_numbers=false for ia in 1:a
        @trace mincut=$MC checkpointing=$CP track_numbers=false for ib in 1:b
            mi = (ia - 1) * b + ib
            @trace mincut=$MC checkpointing=$CP track_numbers=false for ic in 1:c
                @trace mincut=$MC checkpointing=$CP track_numbers=false for ie in 1:e
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
    return sum(rL .^ 2) / length(rL), sum(rD .^ 2) / length(rD)
end
function window_objective(m, x, inp, Δt, loop)
    JL, JD = window_terms(m, x, inp, Δt, loop)
    return sum(inp.wLD .* [1.0, 0.0]) * JL + sum(inp.wLD .* [0.0, 1.0]) * JD
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
    "controls" => CONTROLS == "inputs" ? "native forcing input arrays $(join(input_fields, ",")) (level × knot, native units), SST prescribed" :
                  "heating K/day, moistening g/kg/day (tapered, time-constant), SST K at knots", "sst_knot_times" => ftimes[active_knots],
    "control_mode" => CONTROLS, "input_fields" => input_fields, "active_knot_times_s" => ftimes[active_knots],
    "J_native" => J0, "J_native_repeat" => J0r, "J_diagnostic_loop" => J_diag, "nL" => count(L_mask), "nD" => count(D_mask),
    "native_first_s" => t_native, "native_warm_s" => t_native_warm, "build_native_s" => build_native,
    "gpu" => gpu_memory(), "commit" => get(ENV, "ATLANTIC_COMMIT", "unknown"), "breeze" => pkgdir(Breeze))
save() = open(io -> TOML.print(io, results), joinpath(output, "window_gradient.toml"), "w"); save()

# Input mode: matched directional central differences of J_LWP, J_DWP and J (native), per field.
# Direction v_f: scale_f × smooth profile (1 at the surface, 0 above 3 km) at every active knot, plus a
# knot-contrast direction (+, −, + … across knots) to test the time structure. ε multiplies the direction.
const INPUT_SCALE = Dict("T" => 1 / 86400, "q" => 1e-3 / 86400, "w" => 1e-3, "ug" => 0.1, "vg" => 0.1)  # 1 K/day, 1 g/kg/day, 1 mm/s, 0.1 m/s
zc = native.z
function input_direction(f, kind)
    v = zeros(nx_controls); i = findfirst(==(f), input_fields)
    prof = INPUT_SCALE[f] .* clamp.(1 .- zc ./ 3000, 0, 1)
    tw = kind == "uniform" ? ones(nk) : [(-1.0)^(j - 1) for j in 1:nk]
    v[(i - 1) * nzc * nk + 1:i * nzc * nk] .= vec(prof * tw')
    return v
end
Jt(x) = window_terms(model, on_architecture(GPU(), x), inputs, dt, loop)
input_fd = []
if CONTROLS == "inputs"
    JL0, JD0 = Jt(x0); results["J_LWP_native"] = JL0; results["J_DWP_native"] = JD0; save()
    note("native components"; JL0, JD0, sum=JL0 + JD0, J0)
    for spec in filter(!isempty, split(get(ENV, "ATLANTIC_INPUT_DIRECTIONS", "T:uniform,q:uniform"), ","))
        f, kind = split(spec, ":"); v = input_direction(f, kind)
        for ε in parse.(Float64, split(get(ENV, "ATLANTIC_INPUT_EPSILONS", "1,0.3"), ","))
            Lp, Dp = Jt(x0 .+ ε .* v); Lm, Dm = Jt(x0 .- ε .* v)
            r = Dict("field" => String(f), "direction" => String(kind), "epsilon" => ε, "scale" => INPUT_SCALE[f], "units" => INPUT_UNITS[f],
                     "dLWP" => (Lp - Lm) / 2ε, "dDWP" => (Dp - Dm) / 2ε, "dJ" => (Lp + Dp - Lm - Dm) / 2ε,
                     "J_plus" => Lp + Dp, "J_minus" => Lm + Dm)
            push!(input_fd, r); note("input FD"; (Symbol(k) => v for (k, v) in r)...)
            results["input_fd"] = input_fd; save()
        end
    end
end

# Native central differences of the same objective (reduced mode).
fd = []
for spec in filter(!isempty, split(get(ENV, "ATLANTIC_FD_DIRECTIONS", CONTROLS == "inputs" ? "" : "1:0.01,1:0.003,3:0.01"), ","))
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
results["primal_trace_s"] = trace_s; results["ckpt"] = CKPT; save()
get(ENV, "ATLANTIC_TRACE_ONLY", "false") == "true" && (note("TRACE_OK"; trace_s, ckpt=CKPT); println("WINDOW_TRACE_DONE"); exit())
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
results["gradient_repeat_identical"] = g == gg; results["J_gradient_primal"] = Jg
CONTROLS == "inputs" || (results["gradient"] = g)
if CONTROLS == "inputs"   # component gradients by reseeding the same compiled reverse program
    comp(w) = (ri = merge(rinputs, (; wLD=Reactant.to_rarray(w))); Array(cgrad(rmodel, dmodel, x, gx, ri, dt, loop)[2]))
    gL = comp([1.0, 0.0]); gD = comp([0.0, 1.0])
    results["component_sum_rel"] = maximum(abs.(gL .+ gD .- g)) / max(maximum(abs.(g)), eps())
    results["input_fd_vs_ad"] = [merge(r, Dict("ad_dLWP" => sum(gL .* input_direction(r["field"], r["direction"])),
                                                "ad_dDWP" => sum(gD .* input_direction(r["field"], r["direction"])),
                                                "ad_dJ" => sum(g .* input_direction(r["field"], r["direction"])))) for r in input_fd]
    NCDatasets.NCDataset(joinpath(output, "input_gradient.nc"), "c") do ds
        ds.attrib["scope"] = results["scope"]; ds.attrib["objective"] = "J = J_LWP + J_DWP, J_LWP = mean(((MA30 LWP - paper)/100 g m-2)^2), J_DWP = mean(((MA30 DWP - paper)/1 g m-2)^2), frozen masks, nL=$(count(L_mask)) nD=$(count(D_mask)), 13:30-15:00 UTC"
        ds.attrib["dynamics"] = string(native.dynamics); ds.attrib["commit"] = get(ENV, "ATLANTIC_COMMIT", "unknown")
        ds.attrib["sst"] = "prescribed (not a control)"; ds.attrib["ic"] = "fixed native-spinup window IC; spinup dependence excluded"
        NCDatasets.defDim(ds, "z", nz); NCDatasets.defDim(ds, "knot", nk)
        NCDatasets.defVar(ds, "z", zc, ("z",); attrib=Dict("units" => "m", "long_name" => "model level centre height"))
        NCDatasets.defVar(ds, "knot_time", Float64.(ftimes[active_knots]), ("knot",); attrib=Dict("units" => "s since $(epoch)", "utc_hour" => string(utc.(ftimes[active_knots]))))
        for (i, f) in enumerate(input_fields), (name, gv) in (("dJ", g), ("dJ_LWP", gL), ("dJ_DWP", gD))
            NCDatasets.defVar(ds, "$(name)_d$(f)", reshape(block(gv, i), nz, nk), ("z", "knot");
                attrib=Dict("units" => "1 per ($(INPUT_UNITS[f]))", "input_column" => INPUT_COLUMN[f],
                            "meaning" => "derivative of $name w.r.t. the forcing.csv value of $f at (level, knot); linear time interpolation between knots"))
        end
    end
end
results["reverse_first_exec_synchronized_s"] = g1; results["reverse_warm_exec_synchronized_s"] = g2
results["gradient_finite_nonzero"] = all(isfinite, g) && any(!iszero, g)
results["fd_vs_ad"] = CONTROLS == "inputs" ? [] : [Dict("control" => k, "epsilon" => ε, "fd" => s, "ad" => g[k], "rel" => abs(g[k] - s) / max(abs(s), abs(g[k]), eps())) for (k, ε, s) in fd]
results["max_rss_GB"] = Sys.maxrss() / 1e9; results["gpu_end"] = gpu_memory(); save()
note("reverse gradient"; g, Jg, gc, g1, g2)
println("WINDOW_GRADIENT_DONE finite_nonzero=", results["gradient_finite_nonzero"], " parity=", results["primal_parity"], " fd_vs_ad=", results["fd_vs_ad"])
