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
using Oceananigans.Architectures: ReactantState, on_architecture
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

# Host arrays are moved to the model's architecture before they touch its fields.
on(arch, x) = arch isa ReactantState ? Reactant.to_rarray(x) : on_architecture(arch, x)
function prepare!(case)
    model = case.model; arch = model.grid.architecture
    for (n, f) in pairs(prognostic_fields(model)); interior(f) .= on(arch, saved.fields[String(n)]); end
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

for n in sort(collect(keys(parity))); println("parity ", n, " ", parity[n], " max|G|=", maximum(abs, G0[n])); end

# The kernel is pointwise in its vapour input μ.qᵛ, so with a per-cell scalar L = Σₙ wₙ Gₙ (weights
# normalized by each tendency's magnitude) the reverse derivative at a cell, ∂L/∂qᵛ, can be compared
# cell by cell with native central differences of that cell's own L. (The kernel reads ρqᶜˡ from the
# prognostic fields, not μ.qᶜˡ, so only qᵛ is perturbed.)
Random.seed!(1)
names = ("ρqᵛ", "ρqᶜˡ", "ρnᶜˡ", "ρnᵃ", "ρqʳ", "ρnʳ")
weights = Dict(n => randn(size(G0[n])) ./ max(maximum(abs, G0[n]), eps()) for n in names)
local_L(s) = sum(weights[n] .* s[n] for n in names)
qv = getproperty(μ, :qᵛ); base = Array(interior(qv)); scale = maximum(abs, base)
qc = Array(interior(μ.ρqᶜˡ)); cloudy = qc .> 1e-6 .* maximum(qc)
function fd_cells(ε)
    interior(qv) .= on(GPU(), base .+ ε * scale); p3!(native); Lp = local_L(snapshot(native))
    interior(qv) .= on(GPU(), base .- ε * scale); p3!(native); Lm = local_L(snapshot(native))
    interior(qv) .= on(GPU(), base)
    return (Lp .- Lm) ./ (2ε * scale)
end

wr = Dict(n => Reactant.to_rarray(weights[n]) for n in names)
function J(model, wr)
    p3!(model); Gm = G(model)
    return sum(sum(interior(getproperty(Gm, Symbol(n))) .* w) for (n, w) in wr)
end
grad!(m, dm, wr) = (Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.Reverse), J, Enzyme.Active, Enzyme.Duplicated(m, dm), Enzyme.Const(wr)); nothing)
dm = Enzyme.make_zero(rmodel)
tg = @elapsed cg = Reactant.@compile compile_options=options grad!(rmodel, dm, wr)
cg(rmodel, dm, wr)
ad = Array(interior(dm.microphysical_fields.qᵛ))
floor = 1e-6 * maximum(abs, ad)
best = fill(Inf, size(ad))
for ε in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
    err = abs.(fd_cells(ε) .- ad) ./ (abs.(ad) .+ floor)
    global best = min.(best, err)
    frac(m) = count(err[m] .< 1e-3) / max(count(m), 1)
    note("fd"; ε, median_error=sort(vec(err))[end ÷ 2], within_1e3_all=frac(trues(size(err))),
         within_1e3_cloudy=frac(cloudy), within_1e3_clear=frac(.!cloudy))
end
bad = findall(best .> 1e-3)
note("P3 reverse"; compile_seconds=tg, cells=length(ad), cloudy_cells=count(cloudy), bad_cells=length(bad), bad_cloudy=count(cloudy[bad]),
     max_abs_ad=maximum(abs, ad))
for c in bad[sortperm(best[bad], rev=true)][1:min(10, end)]
    println("bad cell ", Tuple(c), " ad=", ad[c], " best_rel=", best[c], " qv=", base[c], " ρqcl=", qc[c])
end
println(isempty(bad) ? "P3_GRADIENT_PASS" : "P3_GRADIENT_FAIL", " bad_cells=$(length(bad))/$(length(ad)) compile_s=$tg")
