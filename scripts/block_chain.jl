# Host-chained reverse through compiled B-step blocks, with the carried state found by block_restart_check.jl
# (nx=8, nonzero controls, fresh models: 17526/17528/17529 anelastic, 17543 split), and the full interior
# forcing inputs as controls. VALIDATION STAGE: small grid, nb blocks; no full-window claim from this script.
#
# Carried block state S_b (checkpointed on the host, and the adjoint passed between blocks):
#   prognostic fields, the radiation flux Fields of the last scheduled update (written BEFORE update_state!),
#   and for split_explicit also timestepper.Gⁿ (written AFTER the entry update_state!/prepare_first_time_step!,
#   because a fresh update_state! does not rebuild it, 17549/17550).
# Controls x = vcat(vec δ_f) over ATLANTIC_INPUT_FIELDS (default T,q,w,ug,vg): native forcing.csv values at
# (model level, forcing knot), written into the model's own input series inside every block (traced).
#
# Block b, compiled once and reused for every b (same shapes; per-block SST/cos(zenith)/seeds are inputs):
#   block_core!(m, G, x, inp): set inputs(x); update_state!; prepare_first_time_step!; [split: Gⁿ .= G];
#                              B steps (prescribed SST, cos zenith per step)
#   seeded(m, G, x, inp, λ)  = gL·LWP(S_{b+1}) + ⟨gD, rain transect(S_{b+1})⟩ + Σ_f ⟨λ_f, f(S_{b+1})⟩ [+ Σ ⟨λG_f, Gⁿ_f⟩]
# Reverse: for b = nb…1 restore S_{b-1} (and clock), VJP with seeds from the records of b and λ_b → λ_{b-1}
# (shadow model fields [+ shadow G]) and ∂J/∂x contribution (shadow x), summed. J here is a linear test
# functional of the records (fixed random weights) so ∂J/∂record is exact; the MA30 objective is quadratic in
# the records and uses the same seeds path with host-computed ∂J/∂record.
#
# Phases checked (printed as PHASE lines):
#   P1 native: continuous nb·B steps vs nb host-restarted blocks from checkpoints (state parity)
#   P2 compiled block primal vs native block from the same checkpoint (per block)
#   P3 chained compiled VJP directional derivative v·∂J/∂x vs native central FD of the same J (per field direction)
# ENV: ATLANTIC_INPUT, ATLANTIC_STATE, ATLANTIC_OUTPUT, ATLANTIC_NX (8), ATLANTIC_DYNAMICS, ATLANTIC_SUBSTEPS,
#      ATLANTIC_BLOCK (60), ATLANTIC_BLOCKS (2), ATLANTIC_INPUT_FIELDS, ATLANTIC_FD_EPS (1,0.3), ATLANTIC_PHASES (P1,P2,P3)
using Reactant, Enzyme
include("model.jl")
using Oceananigans: initialize!, prognostic_fields
using Oceananigans.Architectures: ReactantState, on_architecture
using Oceananigans.TimeSteppers: reset!, update_state!
using Breeze.TimeSteppers: prepare_first_time_step!
using Reactant: @trace
using Serialization, Random

const T0 = time()
note(msg; kw...) = (@info msg elapsed_s=round(time() - T0; digits=1) kw...; flush(stderr))
input = ENV["ATLANTIC_INPUT"]; output = mkpath(ENV["ATLANTIC_OUTPUT"]); saved = deserialize(ENV["ATLANTIC_STATE"])
nx = parse(Int, get(ENV, "ATLANTIC_NX", "8")); dx = grid_spacing(100.0); dt = 1.0
B = parse(Int, get(ENV, "ATLANTIC_BLOCK", "60")); nb = parse(Int, get(ENV, "ATLANTIC_BLOCKS", "2"))
phases = split(get(ENV, "ATLANTIC_PHASES", "P1,P2,P3"), ",")
fields_in = String.(split(get(ENV, "ATLANTIC_INPUT_FIELDS", "T,q,w,ug,vg"), ","))
const INPUT_SCALE = Dict("T" => 1 / 86400, "q" => 1e-3 / 86400, "w" => 1e-3, "ug" => 0.1, "vg" => 0.1)
const INPUT_UNITS = Dict("T" => "K s-1", "q" => "kg kg-1 s-1", "w" => "m s-1", "ug" => "m s-1", "vg" => "m s-1")
Reactant.set_default_backend("gpu")
lookup_search = NumericalRadiation.FixedBinarySearch()
function with_large_stack(f); task = Task(f, 512 * 1024^2); schedule(task); return fetch(task); end

config = TOML.parsefile(joinpath(input, "interval.toml")); epoch = DateTime(config["epoch"]); t0 = saved.time
forcing_data = readcsv(joinpath(input, "forcing.csv")); ftimes = unique(forcing_data[:, 1])
sst_knots = [forcing_data[findfirst(==(t), forcing_data[:, 1]), 8] for t in ftimes]
function interp_w(t)
    w = zeros(length(ftimes)); i = clamp(searchsortedlast(ftimes, t), 1, length(ftimes) - 1)
    a = clamp((t - ftimes[i]) / (ftimes[i+1] - ftimes[i]), 0, 1); w[i] = 1 - a; w[i+1] = a; return w
end
step_times(b) = t0 .+ ((b - 1) * B .+ (0:B-1)) .* dt
all_times = t0 .+ (0:nb*B-1) .* dt
active = findall(vec(any(reduce(hcat, interp_w.(all_times)) .!= 0; dims=2))); nk = length(active)
sst_block(b) = [sum(interp_w(t) .* sst_knots) for t in step_times(b)]
cosz_block(b) = [max(Breeze.AtmosphereModels.cos_solar_zenith_angle(Breeze.AtmosphereModels.compute_datetime(t, epoch), ENA_COORDINATE...), 0.0) for t in step_times(b)]

surface_temperature(m) = m.radiation.surface_radiation.surface_temperature
cos_zenith_array(m) = m.radiation.solar_position.cos_zenith
is_split(m) = hasproperty(m.timestepper, :substepper)
radiation_names(m) = Tuple(n for n in propertynames(m.radiation) if getproperty(m.radiation, n) isa Oceananigans.Fields.AbstractField)
carried(m) = merge(NamedTuple(prognostic_fields(m)), NamedTuple{radiation_names(m)}(Tuple(getproperty(m.radiation, n) for n in radiation_names(m))))
Gn(m) = m.timestepper.Gⁿ
input_series_of(m) = (; T=fterm(m, :ρE, 1).tls, q=fterm(m, :ρqᵛ, 1).qls, w=fterm(m, :ρqᵛ, 2).velocity,
                       ug=fterm(m, :ρv, 1).geostrophic_velocity, vg=fterm(m, :ρu, 1).geostrophic_velocity)
fterm(m, name, i) = getfield(getfield(getproperty(m.forcing, name), 1)[i], 1)

function make(arch)
    sp = arch isa ReactantState ? FixedCosineZenith(Reactant.to_rarray(fill(0.5, nx, nx))) : FixedCosineZenith(on_architecture(GPU(), fill(0.5, nx, nx)))
    f = () -> CUDA.@allowscalar make_case(input; nx, dx, architecture=arch, solar_position=sp, lookup_search)
    return arch isa ReactantState ? with_large_stack(f) : f()
end
native = make(GPU()); nm = native.model; nz = length(native.z); zc = native.z; dz = diff(native.faces)
base_parent = Dict(f => Array(parent(getproperty(native.input_series, Symbol(f)).data)) for f in fields_in)
nxc = length(fields_in) * nz * nk
blk(x, i) = x[(i - 1) * nz * nk + 1:i * nz * nk]
function control_inputs(on)
    Nzp = size(first(values(base_parent)), 3); off = (Nzp - nz) ÷ 2
    E = zeros(Nzp, nz); for k in 1:nz; E[off + k, k] = 1; end
    S = zeros(nk, length(ftimes)); for (j, a) in enumerate(active); S[j, a] = 1; end
    return (; base=Tuple(on(base_parent[f]) for f in fields_in), E=on(E), S=on(S))
end
function set_inputs!(m, x, ci)
    for (i, f) in enumerate(fields_in)
        P = ci.E * reshape(blk(x, i), nz, nk) * ci.S
        parent(getproperty(input_series_of(m), Symbol(f)).data) .= ci.base[i] .+ reshape(P, 1, 1, size(P, 1), size(P, 2))
    end
end
direction(f) = (v = zeros(nxc); i = findfirst(==(f), fields_in); v[(i-1)*nz*nk+1:i*nz*nk] .= vec((INPUT_SCALE[f] .* clamp.(1 .- zc ./ 3000, 0, 1)) * ones(nk)'); v)

# ---- block functions (shared by native and traced) ----
lwp_of(m, w) = sum((interior(m.microphysical_fields.ρqᶜˡ) .+ interior(m.microphysical_fields.ρqʳ) .+ interior(m.microphysical_fields.ρqʷⁱ)) .* w)
rain_of(m) = view(interior(m.microphysical_fields.ρqʳ), :, size(interior(m.microphysical_fields.ρqʳ), 1) ÷ 2, :)
function block_core!(m, G, x, inp, Δt)
    set_inputs!(m, x, inp.ci)
    update_state!(m); prepare_first_time_step!(m, Δt)
    if is_split(m)   # carried split stage-1 tendencies (17543); useG = 0 in block 1 keeps the derived Gⁿ (its x-dependence differentiated)
        u = sum(inp.useG)
        for (g, gin) in zip(values(Gn(m)), values(G)); parent(g) .= (1 - u) .* parent(g) .+ u .* gin; end
    end
    @trace mincut=true checkpointing=true track_numbers=false for i in 1:size(inp.sst, 1)
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
    for (n, f) in pairs(carried(m)); s += sum(interior(f) .* λ.state[n]); end
    if is_split(m); for (n, g) in pairs(Gn(m)); s += sum(parent(g) .* λ.G[n]); end; end
    return s
end

# ---- host-side state handling (works for CuArray- and ConcreteRArray-backed models) ----
hostcopy(a) = Array(a)
snapshot(m) = (; state=Dict(n => hostcopy(parent(f)) for (n, f) in pairs(carried(m))),
                 G=is_split(m) ? Dict(n => hostcopy(parent(g)) for (n, g) in pairs(Gn(m))) : Dict(),
                 time=Float64(m.clock.time), iteration=Int(m.clock.iteration))
function load!(m, s, put)
    for (n, f) in pairs(carried(m)); put(parent(f), s.state[n]); end
    if m.clock.time isa Number && !(m.clock.time isa Reactant.ConcreteRNumber)
        m.clock.time = s.time; m.clock.iteration = s.iteration
    else
        m.clock.time = typeof(m.clock.time)(s.time); m.clock.iteration = typeof(m.clock.iteration)(s.iteration)
    end
    m.clock.stage = 1
end
put_native!(d, h) = copyto!(d, on_architecture(GPU(), h))
put_r!(d, h) = try copyto!(d, h) catch; copyto!(d, Reactant.to_rarray(h)) end

# Native reference from the saved window IC: radiation initialized at iteration 0 by the restore sequence.
function native_ic!(m)
    for (name, f) in pairs(prognostic_fields(m)); interior(f) .= on_architecture(GPU(), saved.fields[String(name)]); end
    reset!(m.clock); m.clock.time = t0; initialize!(m); update_state!(m); prepare_first_time_step!(m, dt)
end
lwp_w = reshape(1000 .* dz ./ nx^2, 1, 1, :)
rng = MersenneTwister(1); aL = randn(rng, nb); aD = [randn(rng, nx, nz) ./ (nx * nz) for _ in 1:nb]   # linear test functional
on_n(a) = on_architecture(GPU(), a)
ci_n = control_inputs(on_n)
inp_n(b, gL=0.0, gD=zeros(nx, nz)) = (; useG=on_n([b == 1 ? 0.0 : 1.0]), sst=on_n(sst_block(b)), cosz=on_n(cosz_block(b)), stepidx=on_n(collect(1.0:B)),
                                      lwp_weights=on_n(lwp_w), ci=ci_n, gL=on_n([gL]), gD=on_n(gD))
Gzero(m) = is_split(m) ? map(g -> similar(g), Gn(m)) : NamedTuple()
function native_chain(x; restart=false)
    native_ic!(nm); ckpt = [snapshot(nm)]; J = 0.0; recs = []
    for b in 1:nb
        if restart && b > 1
            s = ckpt[end]; load!(nm, s, put_native!)
        end
        G = is_split(nm) ? NamedTuple{keys(Gn(nm))}(Tuple(on_n(ckpt[end].G[n]) for n in keys(Gn(nm)))) : NamedTuple()
        L, R = block_primal(nm, G, on_n(x), inp_n(b), dt)
        J += aL[b] * L + sum(Array(R) .* aD[b]); push!(recs, (L, Array(R))); push!(ckpt, snapshot(nm))
    end
    return J, ckpt, recs
end
results = Dict{String, Any}("nx" => nx, "B" => B, "nb" => nb, "dynamics" => string(native.dynamics), "fields" => fields_in,
    "active_knot_times_s" => ftimes[active], "carried" => string(keys(carried(nm))) * (is_split(nm) ? " + Gⁿ" : ""),
    "commit" => get(ENV, "ATLANTIC_COMMIT", "unknown"))
save() = open(io -> TOML.print(io, results), joinpath(output, "block_chain.toml"), "w")
x0 = zeros(nxc)
J0, ck0, rec0 = native_chain(x0)
note("native chain"; J0, carried=results["carried"])
if "P1" in phases
    J1, ck1, _ = native_chain(x0; restart=true)
    worst = maximum(maximum(abs.(ck1[end].state[n] .- ck0[end].state[n])) / max(maximum(abs.(ck0[end].state[n])), eps()) for n in keys(ck0[end].state))
    results["P1_J_continuous"] = J0; results["P1_J_restarted"] = J1; results["P1_state_worst_rel"] = worst; save()
    println("PHASE P1 native continuous vs restarted: J $J0 vs $J1, state worst_rel=$worst")
end
fd = Dict{String, Any}()
if "P3" in phases
    for f in fields_in, ε in parse.(Float64, split(get(ENV, "ATLANTIC_FD_EPS", "1,0.3"), ","))
        v = direction(f); Jp, _, _ = native_chain(x0 .+ ε .* v); Jm, _, _ = native_chain(x0 .- ε .* v)
        fd["$f:$ε"] = (Jp - Jm) / 2ε; note("native FD"; f, ε, slope=fd["$f:$ε"])
    end
    results["P3_fd"] = fd; save()
end
get(ENV, "ATLANTIC_SKIP_REACTANT", "false") == "true" && (println("BLOCK_CHAIN_NATIVE_DONE"); exit())

# ---- compiled block primal and VJP (each traced once) ----
build = @elapsed rcase = make(ReactantState()); rm = rcase.model; results["reactant_build_s"] = build
on_r(a) = Reactant.to_rarray(a)
ci_r = control_inputs(on_r)
inp_r(b, gL=0.0, gD=zeros(nx, nz)) = (; useG=on_r([b == 1 ? 0.0 : 1.0]), sst=on_r(sst_block(b)), cosz=on_r(cosz_block(b)), stepidx=on_r(collect(1.0:B)),
                                      lwp_weights=on_r(lwp_w), ci=ci_r, gL=on_r([gL]), gD=on_r(gD))
rG = is_split(rm) ? NamedTuple{keys(Gn(rm))}(Tuple(on_r(zeros(size(parent(g)))) for g in values(Gn(rm)))) : NamedTuple()
λ0 = (; state=NamedTuple{keys(carried(rm))}(Tuple(on_r(zeros(size(interior(f))))) for f in values(carried(rm))),
        G=is_split(rm) ? NamedTuple{keys(Gn(rm))}(Tuple(on_r(zeros(size(parent(g))))) for g in values(Gn(rm))) : NamedTuple())
xr = on_r(x0); gx = on_r(zeros(nxc))
opts = Reactant.CompileOptions(raise=true, raise_first=true, sync=true, excluded_passes=["concat_insert_dim_elementwise"])
function vjp!(m, dm, G, dG, x, dx, inp, λ, Δt)
    Enzyme.remake_zero!(dm); dx .= 0; is_split(m) && map(g -> (g .= 0), values(dG))
    Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.Reverse), block_seeded, Enzyme.Active,
        Enzyme.Duplicated(m, dm), Enzyme.Duplicated(G, dG), Enzyme.Duplicated(x, dx), Enzyme.Const(inp), Enzyme.Const(λ), Enzyme.Const(Δt))
    return nothing
end
load!(rm, ck0[1], put_r!)
if "P2" in phases
    pc = @elapsed cprimal = with_large_stack(() -> Reactant.@compile compile_options=opts block_primal(rm, rG, xr, inp_r(1), dt))
    results["primal_compile_s"] = pc; save(); note("compiled block primal"; pc)
    for b in 1:nb
        load!(rm, ck0[b], put_r!); is_split(rm) && foreach(n -> put_r!(rG[n], ck0[b].G[n]), keys(rG))
        tb = @elapsed (L, R) = cprimal(rm, rG, xr, inp_r(b), dt)
        rel = abs(Float64(L) - rec0[b][1]) / max(abs(rec0[b][1]), eps()); relR = maximum(abs.(Array(R) .- rec0[b][2])) / max(maximum(abs.(rec0[b][2])), eps())
        st = maximum(maximum(abs.(Array(parent(f)) .- ck0[b+1].state[n])) / max(maximum(abs.(ck0[b+1].state[n])), eps()) for (n, f) in pairs(carried(rm)))
        results["P2_block$(b)"] = Dict("lwp_rel" => rel, "rain_rel" => relR, "state_worst_rel" => st, "exec_s" => tb); save()
        println("PHASE P2 block $b compiled vs native: lwp_rel=$rel rain_rel=$relR state_worst_rel=$st exec_s=$tb")
    end
end
if "P3" in phases
    dm = Enzyme.make_zero(rm); dG = is_split(rm) ? map(g -> on_r(zeros(size(parent(g)))), rG) : NamedTuple()
    vc = @elapsed cvjp = with_large_stack(() -> Reactant.@compile compile_options=opts vjp!(rm, dm, rG, dG, xr, gx, inp_r(1, 1.0, aD[1]), λ0, dt))
    results["vjp_compile_s"] = vc; save(); note("compiled block VJP"; vc)
    gtot = zeros(nxc); λ = λ0
    for b in nb:-1:1
        load!(rm, ck0[b], put_r!); is_split(rm) && foreach(n -> put_r!(rG[n], ck0[b].G[n]), keys(rG))
        tv = @elapsed cvjp(rm, dm, rG, dG, xr, gx, inp_r(b, aL[b], aD[b]), λ, dt)
        gtot .+= Array(gx)
        λ = (; state=NamedTuple{keys(carried(rm))}(Tuple(on_r(Array(interior(f))) for f in values(carried(dm)))),
               G=is_split(rm) ? NamedTuple{keys(dG)}(Tuple(on_r(Array(parent(g)))) for g in values(dG)) : NamedTuple())
        results["P3_vjp_exec_block$(b)_s"] = tv
    end
    cmp = Dict{String, Any}()
    for (k, s) in fd
        f = split(k, ":")[1]; ad = sum(gtot .* direction(f))
        cmp[k] = Dict("fd" => s, "ad" => ad, "rel" => abs(ad - s) / max(abs(s), abs(ad), eps()))
        println("PHASE P3 $k fd=$s ad=$ad rel=$(cmp[k]["rel"])")
    end
    results["P3_fd_vs_ad"] = cmp; results["gradient_finite_nonzero"] = all(isfinite, gtot) && any(!iszero, gtot); save()
    serialize(joinpath(output, "block_chain_gradient.jls"), (; g=gtot, fields=fields_in, z=zc, knots=ftimes[active], units=INPUT_UNITS))
end
println("BLOCK_CHAIN_DONE")
