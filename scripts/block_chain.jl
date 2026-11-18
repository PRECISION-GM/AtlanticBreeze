# Differentiate a sequence of short atmospheric simulations with respect to forcing inputs.
# Each block uses the same compiled program. Checkpoints carry prognostic fields, radiation
# fluxes, and (for split-explicit dynamics) the previous Runge–Kutta tendencies between blocks.
# The reverse pass propagates their adjoints and accumulates the forcing gradient.
#
# This is an engineering validation with a linear objective built from LWP and a rain transect.
# It does not compute the 90-minute observation loss. The three checks are:
#   P1: native checkpoint/restart agreement;
#   P2: compiled/native agreement from the same checkpoint;
#   P3: chained automatic differentiation against directional finite differences.
# Required paths: ATLANTIC_INPUT, ATLANTIC_STATE, ATLANTIC_OUTPUT.
# ATLANTIC_BLOCK and ATLANTIC_BLOCKS set the steps per block and number of blocks.

using Reactant, Enzyme
include("model.jl")
using Oceananigans: initialize!, prognostic_fields
using Oceananigans.Architectures: ReactantState, on_architecture
using Oceananigans.TimeSteppers: reset!, update_state!
using Breeze.TimeSteppers: prepare_first_time_step!, prepare_acoustic_first_time_step!
using Reactant: @trace
using Serialization, Random

const T0 = time()
note(msg; kw...) = (@info msg elapsed_s=round(time() - T0; digits = 1) kw...; flush(stderr))
input = ENV["ATLANTIC_INPUT"];
output = mkpath(ENV["ATLANTIC_OUTPUT"]);
saved = deserialize(ENV["ATLANTIC_STATE"])
nx = parse(Int, get(ENV, "ATLANTIC_NX", "8"));
dx = grid_spacing(100.0);
dt = 1.0
B = parse(Int, get(ENV, "ATLANTIC_BLOCK", "60"));
nb = parse(Int, get(ENV, "ATLANTIC_BLOCKS", "2"))
phases = split(get(ENV, "ATLANTIC_PHASES", "P1,P2,P3"), ",")
fields_in = String.(split(get(ENV, "ATLANTIC_INPUT_FIELDS", "T,q,w,ug,vg"), ","))
const INPUT_SCALE =
    Dict("T" => 1 / 86400, "q" => 1e-3 / 86400, "w" => 1e-3, "ug" => 0.1, "vg" => 0.1)
const INPUT_UNITS =
    Dict("T" => "K s-1", "q" => "kg kg-1 s-1", "w" => "m s-1", "ug" => "m s-1", "vg" => "m s-1")
Reactant.set_default_backend("gpu")
lookup_search = NumericalRadiation.FixedBinarySearch()
function with_large_stack(f)
    task = Task(f, 512 * 1024^2)
    schedule(task)
    return fetch(task)
end

config = TOML.parsefile(joinpath(input, "interval.toml"));
epoch = DateTime(config["epoch"]);
t0 = saved.time
forcing_data = readcsv(joinpath(input, "forcing.csv"));
ftimes = unique(forcing_data[:, 1])
sst_knots = [forcing_data[findfirst(==(t), forcing_data[:, 1]), 8] for t in ftimes]
function interp_w(t)
    w = zeros(length(ftimes))
    i = clamp(searchsortedlast(ftimes, t), 1, length(ftimes) - 1)
    a = clamp((t - ftimes[i]) / (ftimes[i+1] - ftimes[i]), 0, 1)
    w[i] = 1 - a
    w[i+1] = a
    return w
end
step_times(b) = t0 .+ ((b - 1) * B .+ (0:(B-1))) .* dt
all_times = t0 .+ (0:(nb*B-1)) .* dt
active = findall(vec(any(reduce(hcat, interp_w.(all_times)) .!= 0; dims = 2)));
nk = length(active)
sst_block(b) = [sum(interp_w(t) .* sst_knots) for t in step_times(b)]
cosz_block(b) = [
    max(
        Breeze.AtmosphereModels.cos_solar_zenith_angle(
            Breeze.AtmosphereModels.compute_datetime(t, epoch),
            ENA_COORDINATE...,
        ),
        0.0,
    ) for t in step_times(b)
]

surface_temperature(m) = m.radiation.surface_radiation.surface_temperature
cos_zenith_array(m) = m.radiation.solar_position.cos_zenith
is_split(m) = hasproperty(m.timestepper, :substepper)
radiation_names(m) = Tuple(
    n for n in propertynames(m.radiation) if
    getproperty(m.radiation, n) isa Oceananigans.Fields.AbstractField
)
carried(m) = merge(
    NamedTuple(prognostic_fields(m)),
    NamedTuple{radiation_names(m)}(Tuple(getproperty(m.radiation, n) for n in radiation_names(m))),
)
Gn(m) = m.timestepper.Gⁿ
input_series_of(m) = (;
    T = fterm(m, :ρE, 1).tls,
    q = fterm(m, :ρqᵛ, 1).qls,
    w = fterm(m, :ρqᵛ, 2).velocity,
    ug = fterm(m, :ρv, 1).geostrophic_velocity,
    vg = fterm(m, :ρu, 1).geostrophic_velocity,
)
fterm(m, name, i) = getfield(getfield(getproperty(m.forcing, name), 1)[i], 1)

function make(arch)
    sp =
        arch isa ReactantState ? FixedCosineZenith(Reactant.to_rarray(fill(0.5, nx, nx))) :
        FixedCosineZenith(on_architecture(GPU(), fill(0.5, nx, nx)))
    f =
        () -> CUDA.@allowscalar make_case(
            input;
            nx,
            dx,
            architecture = arch,
            solar_position = sp,
            lookup_search,
        )
    return arch isa ReactantState ? with_large_stack(f) : f()
end
native = make(GPU());
native_model = native.model;
nz = length(native.z);
zc = native.z;
dz = diff(native.faces)
base_parent =
    Dict(f => Array(parent(getproperty(native.input_series, Symbol(f)).data)) for f in fields_in)
number_of_controls = length(fields_in) * nz * nk
blk(x, i) = x[((i-1)*nz*nk+1):(i*nz*nk)]
function control_inputs(on)
    Nzp = size(first(values(base_parent)), 3)
    off = (Nzp - nz) ÷ 2
    E = zeros(Nzp, nz)
    for k = 1:nz
        E[off+k, k] = 1
    end
    S = zeros(nk, length(ftimes))
    for (j, a) in enumerate(active)
        S[j, a] = 1
    end
    return (; base = Tuple(on(base_parent[f]) for f in fields_in), E = on(E), S = on(S))
end
function set_inputs!(m, x, ci)
    for (i, f) in enumerate(fields_in)
        P = ci.E * reshape(blk(x, i), nz, nk) * ci.S
        parent(getproperty(input_series_of(m), Symbol(f)).data) .=
            ci.base[i] .+ reshape(P, 1, 1, size(P, 1), size(P, 2))
    end
end
direction(f) = (
    v = zeros(number_of_controls);
    i = findfirst(==(f), fields_in);
    v[((i-1)*nz*nk+1):(i*nz*nk)] .= vec(
        (INPUT_SCALE[f] .* clamp.(1 .- zc ./ 3000, 0, 1)) * ones(nk)',
    );
    v
)

# ---- block functions (shared by native and traced) ----
lwp_of(m, w) = sum(
    (
        interior(m.microphysical_fields.ρqᶜˡ) .+ interior(m.microphysical_fields.ρqʳ) .+
        interior(m.microphysical_fields.ρqʷⁱ)
    ) .* w,
)
rain_of(m) = view(
    interior(m.microphysical_fields.ρqʳ),
    :,
    size(interior(m.microphysical_fields.ρqʳ), 1) ÷ 2,
    :,
)
# Block entry, written out explicitly so that native and compiled blocks run the same operations.
# `prepare_first_time_step!` is not used here because it differs between architectures
# (BreezeReactantExt/Timesteppers.jl; diagnosed from 17570/17571 P2):
#   anelastic (SSP RK3): the Reactant method recomputes radiation at every call, while the native one
#     does so only at iteration 0. Recomputing at a later block entry discards the carried fluxes
#     (17570 block 2: 6.3e-5). Here: no radiation update at entry; the carried fluxes are used, and for
#     block 1 they are the native initial fluxes of the fixed window state (independent of the controls).
#   split explicit (acoustic RK3): the Reactant `maybe_prepare_first_time_step!` is a no-op, while the
#     native one runs `prepare_acoustic_first_time_step!` (transport caches, last_stage_Δt) at iteration 0
#     (17571 block 1: 4.9e-3). Here: call it explicitly at every entry, then write the carried Gⁿ for later
#     blocks — the sequence validated by the fresh-model restart (17543, restore with prepare at
#     iteration > 0 and non-finite last_stage_Δt).
function enter_block!(m, Δt)
    initialize!(m)
    if is_split(m)
        prepare_acoustic_first_time_step!(m, Δt, ())
    else
        m.clock.last_Δt = Δt
        update_state!(m)
    end
    return nothing
end

function block_core!(m, G, x, inp, Δt)
    set_inputs!(m, x, inp.ci)
    # Step-1 boundary values before the entry: its update_state! builds the stage-1 tendencies, which the
    # compiled first step uses directly (the Reactant maybe_prepare is a no-op; natively the iteration-0
    # repeat inside time_step! hid stale values). 18199/18303: stale cos_zenith/SST at entry gave the
    # 2e-2 ρnᵃ/ρnᶜˡ block-1 mismatch.
    first = ifelse.(inp.stepidx .== 1, 1.0, 0.0)
    parent(surface_temperature(m)) .= sum(inp.sst .* first)
    cos_zenith_array(m) .= sum(inp.cosz .* first)
    enter_block!(m, Δt)
    if is_split(m)   # carried split stage-1 tendencies (17543); useG = 0 in block 1 keeps the derived Gⁿ (its x-dependence differentiated)
        u = sum(inp.useG)
        for (g, gin) in zip(values(Gn(m)), values(G))
            parent(g) .= (1 - u) .* parent(g) .+ u .* gin
        end
    end
    @trace mincut=true checkpointing=true track_numbers=false for i = 1:size(inp.sst, 1)
        sel = ifelse.(inp.stepidx .== i, 1.0, 0.0)
        parent(surface_temperature(m)) .= sum(inp.sst .* sel)
        cos_zenith_array(m) .= sum(inp.cosz .* sel)
        time_step!(m, Δt)
    end
    return nothing
end
function block_primal(m, G, x, inp, Δt)
    block_core!(m, G, x, inp, Δt)
    return lwp_of(m, inp.lwp_weights), rain_of(m) .+ 0
end
function block_seeded(m, G, x, inp, λ, Δt)
    block_core!(m, G, x, inp, Δt)
    s = sum(inp.gL) * lwp_of(m, inp.lwp_weights)
    s += sum(rain_of(m) .* inp.gD)
    for (n, f) in pairs(carried(m))
        s += sum(interior(f) .* λ.state[n])
    end
    if is_split(m)
        for (n, g) in pairs(Gn(m))
            s += sum(parent(g) .* λ.G[n])
        end
    end
    return s
end

# ---- host-side state handling (works for CuArray- and ConcreteRArray-backed models) ----
hostcopy(a) = Array(a)
snapshot(m) = (;
    state = Dict(n => hostcopy(parent(f)) for (n, f) in pairs(carried(m))),
    G = is_split(m) ? Dict(n => hostcopy(parent(g)) for (n, g) in pairs(Gn(m))) : Dict(),
    time = Float64(m.clock.time),
    iteration = Int(m.clock.iteration),
)
function load!(m, s, put)
    for (n, f) in pairs(carried(m))
        put(parent(f), s.state[n])
    end
    if m.clock.time isa Number && !(m.clock.time isa Reactant.ConcreteRNumber)
        m.clock.time = s.time
        m.clock.iteration = s.iteration
    else
        m.clock.time = typeof(m.clock.time)(s.time)
        m.clock.iteration = typeof(m.clock.iteration)(s.iteration)
    end
    m.clock.stage = 1
end
put_native!(d, h) = copyto!(d, on_architecture(GPU(), h))
put_r!(d, h) =
    try
        copyto!(d, h)
    catch
        copyto!(d, Reactant.to_rarray(h))
    end

# Native reference from the saved window IC: radiation initialized at iteration 0 by the restore sequence.
function native_ic!(m)
    for (name, f) in pairs(prognostic_fields(m))
        interior(f) .= on_architecture(GPU(), saved.fields[String(name)])
    end
    reset!(m.clock)
    m.clock.time = t0
    initialize!(m)
    update_state!(m)
    prepare_first_time_step!(m, dt)
end
lwp_weights = reshape(1000 .* dz ./ nx^2, 1, 1, :)
rng = MersenneTwister(1);
aL = randn(rng, nb);
aD = [randn(rng, nx, nz) ./ (nx * nz) for _ = 1:nb]   # linear test functional
on_n(a) = on_architecture(GPU(), a)
native_input_series = control_inputs(on_n)
native_block_inputs(b, gL = 0.0, gD = zeros(nx, nz)) = (;
    useG = on_n([b == 1 ? 0.0 : 1.0]),
    sst = on_n(sst_block(b)),
    cosz = on_n(cosz_block(b)),
    stepidx = on_n(collect(1.0:B)),
    lwp_weights = on_n(lwp_weights),
    ci = native_input_series,
    gL = on_n([gL]),
    gD = on_n(gD),
)
Gzero(m) = is_split(m) ? map(g -> similar(g), Gn(m)) : NamedTuple()
function native_chain(x; restart = false)
    native_ic!(native_model)
    ckpt = [snapshot(native_model)]
    J = 0.0
    recs = []
    for b = 1:nb
        if restart && b > 1
            s = ckpt[end]
            load!(native_model, s, put_native!)
        end
        G =
            is_split(native_model) ?
            NamedTuple{keys(Gn(native_model))}(
                Tuple(on_n(ckpt[end].G[n]) for n in keys(Gn(native_model))),
            ) : NamedTuple()
        L, R = block_primal(native_model, G, on_n(x), native_block_inputs(b), dt)
        J += aL[b] * L + sum(Array(R) .* aD[b])
        push!(recs, (L, Array(R)))
        push!(ckpt, snapshot(native_model))
    end
    return J, ckpt, recs
end
results = Dict{String,Any}(
    "nx" => nx,
    "B" => B,
    "nb" => nb,
    "dynamics" => string(native.dynamics),
    "fields" => fields_in,
    "active_knot_times_s" => ftimes[active],
    "carried" => string(keys(carried(native_model))) * (is_split(native_model) ? " + Gⁿ" : ""),
    "commit" => get(ENV, "ATLANTIC_COMMIT", "unknown"),
)
save() = open(io -> TOML.print(io, results), joinpath(output, "block_chain.toml"), "w")

# Comparison of two checkpoints (Codex 04:10 / 06:14 validation additions).
# For every carried field and, for split-explicit dynamics, every stage tendency in Gⁿ:
#     relative difference = max |candidate − reference| / max(max |reference|, eps())
# (field-wise maximum norm, normalised by the reference field's own maximum), and whether the
# candidate is finite everywhere. The worst value over all fields is what the gate below uses.
function compare_checkpoints(candidate, reference)
    relative_difference(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps())
    state_differences = Dict(String(n) => relative_difference(candidate.state[n], reference.state[n]) for n in keys(reference.state))
    tendency_differences = Dict(String(n) => relative_difference(candidate.G[n], reference.G[n]) for n in keys(reference.G))
    all_finite = all(all(isfinite, a) for a in values(candidate.state)) && all(all(isfinite, a) for a in values(candidate.G))
    worst_state = maximum(values(state_differences))
    worst_tendency = isempty(tendency_differences) ? 0.0 : maximum(values(tendency_differences))
    return (; worst_state, worst_tendency, all_finite, state_differences, tendency_differences)
end
summary_line(c) = "state_worst_rel=$(c.worst_state) Gⁿ_worst_rel=$(c.worst_tendency) finite=$(c.all_finite)"
x0 = zeros(number_of_controls)
J0, native_checkpoints, native_records = native_chain(x0)
note("native chain"; J0, carried = results["carried"])
# ATLANTIC_PROBE_RADINIT=true (native only): rerun the native chain with the traced (Reactant) radiation
# schedule — no radiation update at iteration 0, stage 1 — and compare per field with the normal native
# chain. Tests whether the compiled block-1 mismatch (18199/18304) is the skipped initialization solve.
if get(ENV, "ATLANTIC_PROBE_RADINIT", "false") == "true"
    # Only after the window IC is prepared (native_ic! keeps its initialization solve).
    const TRACED_SCHEDULE = Ref(false)
    @eval Breeze.AtmosphereModels radiation_update_due(schedule, clock, model) =
        $TRACED_SCHEDULE[] ? (clock.iteration > 0 && clock.stage == 2 && schedule(model)) :
        ((clock.iteration == 0 && clock.stage == 1) || (clock.iteration > 0 && clock.stage == 2 && schedule(model)))
    function traced_chain(x)
        native_ic!(native_model)
        TRACED_SCHEDULE[] = true
        ckpt = [snapshot(native_model)]
        recs = []
        for b = 1:nb
            G = NamedTuple{keys(Gn(native_model))}(Tuple(on_n(ckpt[end].G[n]) for n in keys(Gn(native_model))))
            L, R = block_primal(native_model, G, on_n(x), native_block_inputs(b), dt)
            push!(recs, (L, Array(R)))
            push!(ckpt, snapshot(native_model))
        end
        TRACED_SCHEDULE[] = false
        return 0.0, ckpt, recs
    end
    J_traced, traced_checkpoints, traced_records = Base.invokelatest(traced_chain, x0)
    for b = 1:nb
        c = compare_checkpoints(traced_checkpoints[b+1], native_checkpoints[b+1])
        results["PROBE_RADINIT_block$(b)"] = Dict("state_worst_rel" => c.worst_state, "tendency_worst_rel" => c.worst_tendency,
            "lwp_rel" => abs(traced_records[b][1] - native_records[b][1]) / abs(native_records[b][1]),
            "state_by_field" => c.state_differences, "tendency_by_field" => c.tendency_differences)
        println("PROBE_RADINIT block $b traced-schedule vs native: ", summary_line(c))
        for (k, v) in sort(collect(merge(c.state_differences, Dict("Gⁿ " * k => v for (k, v) in c.tendency_differences))); by = last, rev = true)[1:8]
            println("  ", rpad(k, 40), " ", v)
        end
    end
    save()
    println("PROBE_RADINIT_DONE")
    exit()
end
# ATLANTIC_PROBE_FRESH1=true (native only): load the block-1 checkpoint (carried state only) into a FRESH
# native model, enter block 1 as block_core! does, and list every storage array that differs from the
# native chain's model at the same point; then run block 1 and compare with the native block-1 checkpoint.
# Tests whether the compiled block-1 mismatch (18199: ρnᵃ, ρnᶜˡ) is state the checkpoint does not carry.
function array_leaves!(out, x, path, seen, depth)
    depth > 12 && return out
    if x isa Union{Array{<:Number},CUDA.CuArray{<:Number}}
        out[path] = x
        return out
    end
    (isbitstype(typeof(x)) || x isa Union{Module,Function,DataType,String,Symbol} ||
     x isa Oceananigans.Grids.AbstractGrid) && return out
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
function differing_arrays(a, b)
    out = Dict{String,Float64}()
    for (k, x) in a
        haskey(b, k) && size(b[k]) == size(x) || continue
        y = b[k]
        d = maximum(abs.(Array(x) .- Array(y)); init = 0.0) / max(maximum(abs.(Array(y)); init = 0.0), eps())
        (d > 1e-14 || isnan(d)) && (out[k] = d)
    end
    return out
end
if get(ENV, "ATLANTIC_PROBE_FRESH1", "false") == "true"
    raw(m) = Dict(k => Array(v) for (k, v) in array_leaves!(Dict{String,Any}(), m, "model", Set{UInt}(), 0))
    inputs1 = native_block_inputs(1)
    G1 = is_split(native_model) ? NamedTuple{keys(Gn(native_model))}(Tuple(on_n(native_checkpoints[1].G[n]) for n in keys(Gn(native_model)))) : NamedTuple()
    native_ic!(native_model)
    before_native = raw(native_model)
    set_inputs!(native_model, on_n(x0), inputs1.ci); enter_block!(native_model, dt)
    entered_native = raw(native_model)
    fresh = make(GPU()).model
    load!(fresh, native_checkpoints[1], put_native!)
    before_fresh = raw(fresh)
    set_inputs!(fresh, on_n(x0), inputs1.ci); enter_block!(fresh, dt)
    entered_fresh = raw(fresh)
    for (label, a, b) in (("before_entry", before_fresh, before_native), ("after_entry", entered_fresh, entered_native))
        d = differing_arrays(a, b)
        results["PROBE_FRESH1_$(label)"] = d
        println("PROBE_FRESH1 $label: $(length(d)) arrays differ")
        for (k, v) in sort(collect(d); by = last, rev = true)[1:min(end, 60)]
            println("  ", rpad(k, 90), " ", v)
        end
    end
    fresh2 = make(GPU()).model
    load!(fresh2, native_checkpoints[1], put_native!)
    block_primal(fresh2, G1, on_n(x0), native_block_inputs(1), dt)
    c = compare_checkpoints(snapshot(fresh2), native_checkpoints[2])
    results["PROBE_FRESH1_block1"] = Dict("state_by_field" => c.state_differences, "tendency_by_field" => c.tendency_differences)
    println("PROBE_FRESH1 block 1 fresh-loaded vs native: ", summary_line(c))
    save()
    println("PROBE_FRESH1_DONE")
    exit()
end
# ATLANTIC_PROBE_ITER0=true (split, native only): repeat the native chain with the iteration-0 repeat of
# `prepare_acoustic_first_time_step!` inside `time_step!` suppressed (the Reactant method is a no-op), and
# compare per field with the normal native chain. Overwrites the generic method for this process only.
if get(ENV, "ATLANTIC_PROBE_ITER0", "false") == "true" && is_split(native_model)
    @eval Breeze.TimeSteppers function OceananigansTimeSteppers.maybe_prepare_first_time_step!(model::CompressibleAcousticModel, Δt, callbacks)
        isfinite(model.clock.last_stage_Δt) || prepare_acoustic_first_time_step!(model, Δt, callbacks)
        return nothing
    end
    J_single, single_checkpoints, single_records = Base.invokelatest(native_chain, x0)
    for b = 1:nb
        c = compare_checkpoints(single_checkpoints[b+1], native_checkpoints[b+1])
        results["PROBE_ITER0_block$(b)"] = Dict("state_worst_rel" => c.worst_state, "tendency_worst_rel" => c.worst_tendency,
            "lwp_rel" => abs(single_records[b][1] - native_records[b][1]) / abs(native_records[b][1]),
            "state_by_field" => c.state_differences, "tendency_by_field" => c.tendency_differences)
        println("PROBE_ITER0 block $b single-prepare vs native: ", summary_line(c))
    end
    results["PROBE_ITER0_J"] = [J_single, J0]
    save()
    println("PROBE_ITER0_DONE")
    exit()
end
if "P1" in phases
    # Same native model reloaded from its own checkpoints: tests checkpoint/load completeness, not a
    # fresh-model restart (that evidence is block_restart_check.jl with ATLANTIC_FRESH=true).
    J1, restarted_checkpoints, _ = native_chain(x0; restart = true)
    results["P1_J_continuous"] = J0
    results["P1_J_restarted"] = J1
    for b = 1:nb
        comparison = compare_checkpoints(restarted_checkpoints[b+1], native_checkpoints[b+1])
        results["P1_block$(b)"] = Dict(
            "state_worst_rel" => comparison.worst_state,
            "tendency_worst_rel" => comparison.worst_tendency,
            "finite" => comparison.all_finite,
        )
        println("PHASE P1 block $b native restarted vs continuous: ", summary_line(comparison))
    end
    save()
    println("PHASE P1 native continuous vs restarted: J $J0 vs $J1")
end
fd = Dict{String,Any}()
if "P3" in phases
    for f in fields_in, ε in parse.(Float64, split(get(ENV, "ATLANTIC_FD_EPS", "1,0.3"), ","))
        v = direction(f)
        Jp, _, _ = native_chain(x0 .+ ε .* v)
        Jm, _, _ = native_chain(x0 .- ε .* v)
        fd["$f:$ε"] = (Jp - Jm) / 2ε
        note("native FD"; f, ε, slope = fd["$f:$ε"])
    end
    results["P3_fd"] = fd
    save()
end
get(ENV, "ATLANTIC_SKIP_REACTANT", "false") == "true" &&
    (println("BLOCK_CHAIN_NATIVE_DONE"); exit())

# ---- compiled block primal and VJP (each traced once) ----
build = @elapsed rcase = make(ReactantState());
reactant_model = rcase.model;
results["reactant_build_s"] = build
on_r(a) = Reactant.to_rarray(a)
reactant_input_series = control_inputs(on_r)
reactant_block_inputs(b, gL = 0.0, gD = zeros(nx, nz)) = (;
    useG = on_r([b == 1 ? 0.0 : 1.0]),
    sst = on_r(sst_block(b)),
    cosz = on_r(cosz_block(b)),
    stepidx = on_r(collect(1.0:B)),
    lwp_weights = on_r(lwp_weights),
    ci = reactant_input_series,
    gL = on_r([gL]),
    gD = on_r(gD),
)
reactant_tendencies =
    is_split(reactant_model) ?
    NamedTuple{keys(Gn(reactant_model))}(
        Tuple(on_r(zeros(size(parent(g)))) for g in values(Gn(reactant_model))),
    ) : NamedTuple()
# (17560: a misplaced parenthesis closed Tuple before the generator, making each λ a tuple of scalars)
λ0 = (;
    state = NamedTuple{keys(carried(reactant_model))}(
        Tuple(on_r(zeros(size(interior(f)))) for f in values(carried(reactant_model))),
    ),
    G = is_split(reactant_model) ?
        NamedTuple{keys(Gn(reactant_model))}(
        Tuple(on_r(zeros(size(parent(g)))) for g in values(Gn(reactant_model))),
    ) : NamedTuple(),
)
reactant_controls = on_r(x0);
control_adjoint = on_r(zeros(number_of_controls))
opts = Reactant.CompileOptions(
    raise = true,
    raise_first = true,
    sync = true,
    excluded_passes = ["concat_insert_dim_elementwise"],
)
function vjp!(m, model_adjoint, G, tendency_adjoint, x, dx, inp, λ, Δt)
    Enzyme.remake_zero!(model_adjoint)
    dx .= 0
    is_split(m) && map(g -> (g .= 0), values(tendency_adjoint))
    Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.Reverse),
        block_seeded,
        Enzyme.Active,
        Enzyme.Duplicated(m, model_adjoint),
        Enzyme.Duplicated(G, tendency_adjoint),
        Enzyme.Duplicated(x, dx),
        Enzyme.Const(inp),
        Enzyme.Const(λ),
        Enzyme.Const(Δt),
    )
    return nothing
end
load!(reactant_model, native_checkpoints[1], put_r!)
if "P2" in phases
    pc = @elapsed cprimal = with_large_stack(
        () -> Reactant.@compile compile_options=opts block_primal(
            reactant_model,
            reactant_tendencies,
            reactant_controls,
            reactant_block_inputs(1),
            dt,
        )
    )
    results["primal_compile_s"] = pc
    save()
    note("compiled block primal"; pc)
    for b = 1:nb
        load!(reactant_model, native_checkpoints[b], put_r!)
        is_split(reactant_model) && foreach(
            n -> put_r!(reactant_tendencies[n], native_checkpoints[b].G[n]),
            keys(reactant_tendencies),
        )
        tb = @elapsed (L, R) = cprimal(
            reactant_model,
            reactant_tendencies,
            reactant_controls,
            reactant_block_inputs(b),
            dt,
        )
        rel = abs(Float64(L) - native_records[b][1]) / max(abs(native_records[b][1]), eps())
        relR =
            maximum(abs.(Array(R) .- native_records[b][2])) /
            max(maximum(abs.(native_records[b][2])), eps())
        comparison = compare_checkpoints(snapshot(reactant_model), native_checkpoints[b+1])
        results["P2_block$(b)"] = Dict(
            "lwp_rel" => rel,
            "rain_rel" => relR,
            "state_worst_rel" => comparison.worst_state,
            "tendency_worst_rel" => comparison.worst_tendency,
            "finite" => comparison.all_finite && isfinite(Float64(L)) && all(isfinite, Array(R)),
            "state_by_field" => comparison.state_differences,
            "tendency_by_field" => comparison.tendency_differences,
            "exec_s" => tb,
        )
        save()
        println("PHASE P2 block $b compiled vs native (reloaded checkpoint): lwp_rel=$rel rain_rel=$relR ",
                summary_line(comparison), " exec_s=$tb")
    end

    # Continuously chained compiled trajectory: load the window checkpoint once, then let the compiled
    # block carry its own state (and, for split dynamics, its own Gⁿ) from block to block, with no native
    # reloads, and compare every block end with the native chain.
    load!(reactant_model, native_checkpoints[1], put_r!)
    is_split(reactant_model) &&
        foreach(n -> put_r!(reactant_tendencies[n], native_checkpoints[1].G[n]), keys(reactant_tendencies))
    for b = 1:nb
        if b > 1 && is_split(reactant_model)
            compiled_state = snapshot(reactant_model)
            foreach(n -> put_r!(reactant_tendencies[n], compiled_state.G[n]), keys(reactant_tendencies))
        end
        (L, R) = cprimal(reactant_model, reactant_tendencies, reactant_controls, reactant_block_inputs(b), dt)
        comparison = compare_checkpoints(snapshot(reactant_model), native_checkpoints[b+1])
        lwp_relative = abs(Float64(L) - native_records[b][1]) / max(abs(native_records[b][1]), eps())
        rain = Array(R)
        rain_relative = maximum(abs.(rain .- native_records[b][2])) / max(maximum(abs.(native_records[b][2])), eps())
        results["P2_continuous_block$(b)"] = Dict(
            "lwp_rel" => lwp_relative,
            "rain_rel" => rain_relative,
            "state_worst_rel" => comparison.worst_state,
            "tendency_worst_rel" => comparison.worst_tendency,
            "finite" => comparison.all_finite && isfinite(Float64(L)) && all(isfinite, rain),
        )
        save()
        println("PHASE P2c block $b compiled continuous chain vs native: lwp_rel=$lwp_relative rain_rel=$rain_relative ",
                summary_line(comparison))
    end
end

# Gate before the reverse compilation. The VJP is compiled only if every compiled block (reloaded and
# continuously chained) is finite — carried state, Gⁿ, the LWP record and every rain-transect element —
# and agrees with the native chain to within
#   ATLANTIC_PARITY_TOL (default 1e-8) for carried state and Gⁿ (field-wise maximum norm, compare_checkpoints), and
#   ATLANTIC_RECORD_TOL (default 1e-8) for the records that seed the objective: |ΔLWP| / |LWP| and
#   max |Δrain| / max |rain| over the transect (Codex 07:20: state agreement alone does not qualify objective parity).
# ATLANTIC_FORCE_VJP=true compiles anyway and labels the reverse results as compile-only diagnostics.
parity_tolerance = parse(Float64, get(ENV, "ATLANTIC_PARITY_TOL", "1e-8"))
record_tolerance = parse(Float64, get(ENV, "ATLANTIC_RECORD_TOL", "1e-8"))
parity_entries = [v for (k, v) in results if startswith(k, "P2_block") || startswith(k, "P2_continuous_block")]
parity_passed =
    !isempty(parity_entries) && all(
        e["finite"] && e["state_worst_rel"] <= parity_tolerance && e["tendency_worst_rel"] <= parity_tolerance &&
        e["lwp_rel"] <= record_tolerance && e["rain_rel"] <= record_tolerance
        for e in parity_entries
    )
force_vjp = get(ENV, "ATLANTIC_FORCE_VJP", "false") == "true"
results["parity_tolerance"] = parity_tolerance
results["record_tolerance"] = record_tolerance
results["parity_gate_passed"] = parity_passed
results["reverse_scope"] = parity_passed ? "validated-primal" : (force_vjp ? "compile-only diagnostics (parity gate failed)" : "skipped (parity gate failed)")
save()
println("PHASE GATE parity_passed=$parity_passed tolerance=$parity_tolerance reverse_scope=$(results["reverse_scope"])")
if "P3" in phases && !parity_passed && !force_vjp
    println("BLOCK_CHAIN_DONE (reverse skipped: parity gate failed)")
    exit()
end
if "P3" in phases
    model_adjoint = Enzyme.make_zero(reactant_model)
    tendency_adjoint =
        is_split(reactant_model) ? map(g -> on_r(zeros(size(parent(g)))), reactant_tendencies) :
        NamedTuple()
    vc = @elapsed cvjp = with_large_stack(
        () -> Reactant.@compile compile_options=opts vjp!(
            reactant_model,
            model_adjoint,
            reactant_tendencies,
            tendency_adjoint,
            reactant_controls,
            control_adjoint,
            reactant_block_inputs(1, 1.0, aD[1]),
            λ0,
            dt,
        )
    )
    results["vjp_compile_s"] = vc
    save()
    note("compiled block VJP"; vc)
    forcing_gradient = zeros(number_of_controls)
    λ = λ0
    for b = nb:-1:1
        load!(reactant_model, native_checkpoints[b], put_r!)
        is_split(reactant_model) && foreach(
            n -> put_r!(reactant_tendencies[n], native_checkpoints[b].G[n]),
            keys(reactant_tendencies),
        )
        tv = @elapsed cvjp(
            reactant_model,
            model_adjoint,
            reactant_tendencies,
            tendency_adjoint,
            reactant_controls,
            control_adjoint,
            reactant_block_inputs(b, aL[b], aD[b]),
            λ,
            dt,
        )
        forcing_gradient .+= Array(control_adjoint)
        λ = (;
            state = NamedTuple{keys(carried(reactant_model))}(
                Tuple(on_r(Array(interior(f))) for f in values(carried(model_adjoint))),
            ),
            G = is_split(reactant_model) ?
                NamedTuple{keys(tendency_adjoint)}(
                Tuple(on_r(Array(parent(g))) for g in values(tendency_adjoint)),
            ) : NamedTuple(),
        )
        results["P3_vjp_exec_block$(b)_s"] = tv
    end
    cmp = Dict{String,Any}()
    for (k, s) in fd
        f = split(k, ":")[1]
        ad = sum(forcing_gradient .* direction(f))
        cmp[k] = Dict("fd" => s, "ad" => ad, "rel" => abs(ad - s) / max(abs(s), abs(ad), eps()))
        println("PHASE P3 $k fd=$s ad=$ad rel=$(cmp[k]["rel"])")
    end
    results["P3_fd_vs_ad"] = cmp
    results["gradient_finite_nonzero"] =
        all(isfinite, forcing_gradient) && any(!iszero, forcing_gradient)
    save()
    serialize(
        joinpath(output, "block_chain_gradient.jls"),
        (;
            g = forcing_gradient,
            fields = fields_in,
            z = zc,
            knots = ftimes[active],
            units = INPUT_UNITS,
        ),
    )
end
println("BLOCK_CHAIN_DONE")
