# Reusable compiled block VJP for the window objective (compile-cost reduction; DRAFT, validation stage).
#
# One B-step block (B = 60 steps = one MA30 record interval, two radiation updates) is compiled ONCE as
#   primal:  state_b ↦ (state_{b+1}, LWP_{b+1}, DWP-transect_{b+1})
#   VJP:     f_b(state_b, x) = Σ_f ⟨λ_f, field_f(state_{b+1})⟩ + gL·LWP_{b+1} + ⟨gD, rain_transect_{b+1}⟩
# and chained on the host with explicit checkpoints: forward stores state_b; reverse, b = nb…1, restores
# state_b, runs the compiled VJP with the incoming adjoint λ_{b+1} and record seeds, and reads λ_b from the
# shadow model and the block's ∂J/∂x from the shadow controls (accumulated). J is quadratic in the MA30
# records, so ∂J/∂record is exact on the host. Every block's VJP differentiates its steps, forcing reads
# and radiation updates; the host only passes adjoints, so no dependence through time is detached.
# Valid only if block restarts from prognostic fields are exact (block_restart_check.jl).
#
# ENV: ATLANTIC_INPUT, ATLANTIC_STATE, ATLANTIC_OUTPUT, ATLANTIC_NX (8), ATLANTIC_DYNAMICS, ATLANTIC_BLOCK (60),
#      ATLANTIC_BLOCKS (2, validation), ATLANTIC_RAISE (true).
using Reactant, Enzyme
include("model.jl")
using Oceananigans: initialize!, prognostic_fields
using Oceananigans.Architectures: ReactantState, on_architecture
using Oceananigans.TimeSteppers: reset!, update_state!
using Reactant: @trace
using Serialization

const T0 = time()
note(msg; kw...) = (@info msg elapsed_s=round(time() - T0; digits=1) kw...; flush(stderr))
input = ENV["ATLANTIC_INPUT"]; output = mkpath(ENV["ATLANTIC_OUTPUT"]); saved = deserialize(ENV["ATLANTIC_STATE"])
nx = parse(Int, get(ENV, "ATLANTIC_NX", "8")); dx = grid_spacing(100.0); dt = 1.0
B = parse(Int, get(ENV, "ATLANTIC_BLOCK", "60")); nb = parse(Int, get(ENV, "ATLANTIC_BLOCKS", "2"))
Reactant.set_default_backend("gpu"); Reactant.MLIR.IR.DUMP_MLIR_DIR[] = mkpath(joinpath(output, "mlir"))
lookup_search = NumericalRadiation.FixedBinarySearch()
function with_large_stack(f); task = Task(f, 512 * 1024^2); schedule(task); return fetch(task); end
surface_temperature(m) = m.radiation.surface_radiation.surface_temperature
cos_zenith_array(m) = m.radiation.solar_position.cos_zenith
# Block state = prognostic fields + the radiation flux Fields of the last scheduled update (17526: a fresh-model
# restart is exact only if those fluxes are written before update_state!, which builds stage-1 Gⁿ incl. radiative
# heating from them). Gⁿ itself is derived inside block! by update_state!, so it needs no checkpoint or adjoint.
radiation_field_names(m) = Tuple(n for n in propertynames(m.radiation) if getproperty(m.radiation, n) isa Oceananigans.Fields.AbstractField)
state_fields(m) = merge(NamedTuple(prognostic_fields(m)), NamedTuple{radiation_field_names(m)}(Tuple(getproperty(m.radiation, n) for n in radiation_field_names(m))))
names_of(m) = Tuple(keys(state_fields(m)))

# Per-step SST/cos(zenith) from the native series (prescribed; controls here are the forcing input arrays).
config = TOML.parsefile(joinpath(input, "interval.toml")); epoch = DateTime(config["epoch"]); t0 = saved.time
forcing_data = readcsv(joinpath(input, "forcing.csv")); ftimes = unique(forcing_data[:, 1])
sst_knots = [forcing_data[findfirst(==(t), forcing_data[:, 1]), 8] for t in ftimes]
interp(times, v, t) = (i = clamp(searchsortedlast(times, t), 1, length(times) - 1); a = clamp((t - times[i]) / (times[i+1] - times[i]), 0, 1); (1 - a) * v[i] + a * v[i+1])
step_times(b) = t0 .+ ((b - 1) * B .+ (0:B-1)) .* dt
sst_block(b) = [interp(ftimes, sst_knots, t) for t in step_times(b)]
cosz_block(b) = [max(Breeze.AtmosphereModels.cos_solar_zenith_angle(Breeze.AtmosphereModels.compute_datetime(t, epoch), ENA_COORDINATE...), 0.0) for t in step_times(b)]

build = @elapsed rcase = with_large_stack(() -> CUDA.@allowscalar make_case(input; nx, dx, architecture=ReactantState(),
    solar_position=FixedCosineZenith(Reactant.to_rarray(fill(0.5, nx, nx))), lookup_search))
m = rcase.model; nz = length(rcase.z); jt = nx ÷ 2
note("built"; build, nx, dynamics=rcase.dynamics)

# One block. Single-level loop, loop-counter indexing only (no values captured from enclosing loops).
function block!(m, inp, Δt)
    update_state!(m)
    @trace mincut=true checkpointing=true track_numbers=false for i in 1:size(inp.sst, 1)
        sel = ifelse.(inp.stepidx .== i, 1.0, 0.0)
        parent(surface_temperature(m)) .= sum(inp.sst .* sel)
        cos_zenith_array(m) .= sum(inp.cosz .* sel)
        time_step!(m, Δt)
    end
    return nothing
end
lwp_of(m, inp) = sum((interior(m.microphysical_fields.ρqᶜˡ) .+ interior(m.microphysical_fields.ρqʳ) .+ interior(m.microphysical_fields.ρqʷⁱ)) .* inp.lwp_weights)
function block_primal(m, inp, Δt)
    block!(m, inp, Δt)
    return lwp_of(m, inp), view(interior(m.microphysical_fields.ρqʳ), :, size(interior(m.microphysical_fields.ρqʳ), 1) ÷ 2, :) .+ 0
end
function block_seeded(m, inp, λ, Δt)
    block!(m, inp, Δt)
    s = sum(inp.gL) * lwp_of(m, inp)   # sum(gL), not gL[1]: no scalar getindex on traced arrays (17387)
    s += sum(view(interior(m.microphysical_fields.ρqʳ), :, size(interior(m.microphysical_fields.ρqʳ), 1) ÷ 2, :) .* inp.gD)   # DWP seed (restored: 8503336 commented it out)
    for (n, f) in pairs(state_fields(m)); s += sum(interior(f) .* λ[n]); end   # adjoint of prognostics AND carried radiation fluxes
    return s
end
function block_vjp!(m, dm, inp, λ, Δt)
    Enzyme.remake_zero!(dm)
    Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.Reverse), block_seeded, Enzyme.Active,
                    Enzyme.Duplicated(m, dm), Enzyme.Const(inp), Enzyme.Const(λ), Enzyme.Const(Δt))
    return nothing
end

dz = diff(rcase.faces)
on(a) = Reactant.to_rarray(a)
inp = (; sst=on(sst_block(1)), cosz=on(cosz_block(1)), stepidx=on(collect(1.0:B)),
         lwp_weights=on(reshape(1000 .* dz ./ nx^2, 1, 1, :)), gL=on([1.0]), gD=on(fill(1.0, nx, nz)))   # nonzero DWP seed so the term is exercised
λ = NamedTuple{names_of(m)}(Tuple(on(zero(Array(interior(f)))) for f in values(state_fields(m))))
mlir_bytes() = sum(filesize(joinpath(output, "mlir", f)) for f in readdir(joinpath(output, "mlir")); init=0)
# ATLANTIC_STRIP=true: strip_llvm_debuginfo (kernel debug info/locations, ~30% of the module text in 17374)
strip = get(ENV, "ATLANTIC_STRIP", "false") == "true"
opts = get(ENV, "ATLANTIC_RAISE", "true") == "true" ?
    Reactant.CompileOptions(raise=true, raise_first=true, sync=true, excluded_passes=["concat_insert_dim_elementwise"], strip_llvm_debuginfo=strip) :
    Reactant.CompileOptions(raise=false, sync=true, strip_llvm_debuginfo=strip)
phases = split(get(ENV, "ATLANTIC_PHASES", "primal,vjp"), ",")
cp = "primal" in phases ? (@elapsed cprimal = with_large_stack(() -> Reactant.@compile compile_options=opts block_primal(m, inp, dt))) : NaN
note("compiled block primal (single trace, no separate code_hlo)"; cp, mlir_MB=mlir_bytes() / 1e6)
dm = Enzyme.make_zero(m)
cv = "vjp" in phases ? (@elapsed cvjp = with_large_stack(() -> Reactant.@compile compile_options=opts block_vjp!(m, dm, inp, λ, dt))) : NaN
note("compiled block VJP"; cv, mlir_MB=mlir_bytes() / 1e6)
# Executable reuse: the same cprimal/cvjp are called for every block with new same-shaped inputs.
results = Dict{String, Any}("nx" => nx, "B" => B, "nb" => nb, "build_s" => build, "primal_compile_s" => cp, "vjp_compile_s" => cv,
                            "dynamics" => string(rcase.dynamics), "strip_llvm_debuginfo" => strip, "phases" => join(phases, ","), "mlir_MB" => mlir_bytes() / 1e6, "commit" => get(ENV, "ATLANTIC_COMMIT", "unknown"))
save() = open(io -> TOML.print(io, results), joinpath(output, "block_vjp.toml"), "w"); save()
println("BLOCK_VJP_COMPILED")
