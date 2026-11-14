# Isolated Reactant test of the P3 + prognostic-aerosol tendency kernel alone (no advection, radiation or
# time stepping). From a small spun-up cloudy state it (1) compiles `compute_microphysical_tendencies!` and
# compares the P3 tendencies with native CUDA, (2) compiles a reverse-mode derivative of a scalar of those
# tendencies with respect to the kernel's moisture inputs (μ.qᵛ, μ.qᶜˡ) and compares its directional
# derivative with native central differences over an epsilon sweep. Kernel-level evidence only: it does not
# validate gradients of the full case.
# ENV: ATLANTIC_STATE (required, from check_gradients.jl), ATLANTIC_DX (must match the state), ATLANTIC_OUTPUT.
using Reactant, Enzyme
include("model.jl")
using Oceananigans: prognostic_fields
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: update_state!
using Serialization, Random
Reactant.set_default_backend("gpu")
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = mkpath(joinpath(get(ENV, "ATLANTIC_OUTPUT", "outputs"), "mlir"))
note(msg; kw...) = (@info msg kw...; flush(stderr))
const AM = Breeze.AtmosphereModels

input = get(ENV, "ATLANTIC_INPUT", "data/interval")
saved = deserialize(ENV["ATLANTIC_STATE"]); nx = saved.nx
dx = grid_spacing(); order = advection_order()
kw = (; nx, dx, order, solar_position=FixedCosineZenith(0.58), lookup_search=NumericalRadiation.FixedBinarySearch())

function prepare!(case)
    model = case.model
    for (n, f) in pairs(prognostic_fields(model)); interior(f) .= saved.fields[String(n)]; end
    update_state!(model)
    return model
end
G(model) = Breeze.Microphysics.PredictedParticleProperties.p3_tendency_fields(model.timestepper.Gⁿ, model.microphysics)
function p3!(model)
    for g in G(model); fill!(parent(g), 0); end
    AM.compute_microphysical_tendencies!(model.microphysics, model)
    return nothing
end
snapshot(model) = Dict(String(n) => Array(interior(g)) for (n, g) in pairs(G(model)))

tn = @elapsed native = prepare!(make_case(input; kw..., architecture=GPU()))
μ = native.microphysical_fields
note("native model"; build_seconds=tn, nx, dx, mu=keys(μ), max_qcl=maximum(interior(μ.qᶜˡ)), max_na=maximum(interior(μ.nᵃ)))
p3!(native); G0 = snapshot(native)

tr = @elapsed rmodel = CUDA.@allowscalar prepare!(make_case(input; kw..., architecture=ReactantState()))
note("reactant model"; build_seconds=tr)
options = Reactant.CompileOptions(raise=true, raise_first=true, sync=true, excluded_passes=["concat_insert_dim_elementwise"])
tc = @elapsed compiled = Reactant.@compile compile_options=options p3!(rmodel)
compiled(rmodel); G1 = snapshot(rmodel)
parity = Dict(n => maximum(abs.(G0[n] .- G1[n])) / max(maximum(abs.(G0[n])), eps()) for n in keys(G0))
worst = maximum(values(parity))
note("compiled P3 kernel"; compile_seconds=tc, worst, parity)
println(worst < 1e-10 ? "P3_PARITY_PASS" : "P3_PARITY_FAIL", " nx=$nx parity=$worst compile_s=$tc")

# Scalar of the tendencies: condensate and aerosol source terms weighted by fixed random patterns.
Random.seed!(1)
weights = Dict(n => randn(size(G0[n])) for n in ("ρqᶜˡ", "ρnᶜˡ", "ρnᵃ", "ρqᵛ"))
J_native(model) = (p3!(model); s = snapshot(model); sum(sum(weights[n] .* s[n]) for n in keys(weights)))
direction = Dict(n => randn(size(interior(getproperty(μ, Symbol(n))))) for n in ("qᵛ", "qᶜˡ"))
scale = Dict(n => max(maximum(abs, interior(getproperty(μ, Symbol(n)))), eps()) for n in keys(direction))
function nudge!(model, ε)
    for (n, d) in direction; f = getproperty(model.microphysical_fields, Symbol(n)); interior(f) .+= ε .* scale[n] .* d; end
end
base = Dict(n => copy(Array(interior(getproperty(μ, Symbol(n))))) for n in keys(direction))
restore!(model) = for (n, b) in base; interior(getproperty(model.microphysical_fields, Symbol(n))) .= b; end
fd = map((1e-2, 1e-3, 1e-4, 1e-5, 1e-6)) do ε
    nudge!(native, ε); Jp = J_native(native); restore!(native)
    nudge!(native, -ε); Jm = J_native(native); restore!(native)
    (ε, (Jp - Jm) / 2ε)
end

wr = Dict(n => Reactant.to_rarray(w) for (n, w) in weights)
function J(model, wr)
    p3!(model); Gm = G(model)
    return sum(sum(interior(getproperty(Gm, Symbol(n))) .* w) for (n, w) in wr)
end
grad!(m, dm, wr) = (Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.Reverse), J, Enzyme.Active, Enzyme.Duplicated(m, dm), Enzyme.Const(wr)); nothing)
dm = Enzyme.make_zero(rmodel)
tg = @elapsed cg = Reactant.@compile compile_options=options grad!(rmodel, dm, wr)
cg(rmodel, dm, wr)
ad = sum(sum(Array(interior(getproperty(dm.microphysical_fields, Symbol(n)))) .* scale[n] .* d) for (n, d) in direction)
for (ε, g) in fd; note("fd"; ε, g, relative_error=abs(g - ad) / max(abs(ad), eps())); end
best = minimum(abs(g - ad) / max(abs(ad), eps()) for (_, g) in fd)
note("P3 reverse"; compile_seconds=tg, ad, best)
println(best < 1e-4 ? "P3_GRADIENT_PASS" : "P3_GRADIENT_FAIL", " ad=$ad best_rel=$best compile_s=$tg")
