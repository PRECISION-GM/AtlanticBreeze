# CPU per-cell probe of the P3 tendency computation (no Reactant), for the cells where the compiled
# reverse derivative of p3_kernel.jl disagreed with finite differences. For each selected cell it evaluates
# the same weighted scalar L(qᵛ) the kernel test used, through the same functions the kernel calls, and
# compares: native Enzyme forward mode, the saved Reactant reverse derivative, central differences over an
# ε sweep, and a fine scan of L around the base value (to expose kinks or noise).
# ENV: ATLANTIC_STATE (same as p3_kernel.jl), ATLANTIC_GRADIENT_FILE (p3_gradient.jls from p3_kernel.jl),
#      ATLANTIC_DX, ATLANTIC_OUTPUT.
using Enzyme
include("model.jl")
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: update_state!
using Serialization
const AM = Breeze.AtmosphereModels
const PPP = Breeze.Microphysics.PredictedParticleProperties
note(msg; kw...) = (@info msg kw...; flush(stderr))

input = get(ENV, "ATLANTIC_INPUT", "data/interval")
output = mkpath(get(ENV, "ATLANTIC_OUTPUT", "outputs/p3_cells"))
saved = deserialize(ENV["ATLANTIC_STATE"]); nx = saved.nx
g = deserialize(ENV["ATLANTIC_GRADIENT_FILE"])
case = make_case(input; nx, dx=grid_spacing(), order=advection_order(), architecture=CPU(),
                 solar_position=FixedCosineZenith(0.58), lookup_search=NumericalRadiation.FixedBinarySearch())
model = case.model
for (n, f) in pairs(prognostic_fields(model)); interior(f) .= saved.fields[String(n)]; end
update_state!(model)
μ = model.microphysical_fields; p3 = model.microphysics; grid = model.grid; constants = model.thermodynamic_constants
ρ_field = AM.total_density(model.dynamics)
PPP.compute_p3_surface_temperature!(μ.surface_temperature, model.temperature, grid)

# Same steps as p3_grid_process_inputs + p3_tendency_compute, with the vapour input passed explicitly.
# Everything else arrives through one concretely typed context, so Enzyme sees no untyped globals.
const NAMES = ("ρqᵛ", "ρqᶜˡ", "ρnᶜˡ", "ρnᵃ", "ρqʳ", "ρnʳ")
ctx = (; ρ_field, μ, p3, grid, constants, velocities=model.velocities, formulation=model.formulation,
         dynamics=model.dynamics, w=NamedTuple{Tuple(Symbol.(NAMES))}(Tuple(g.weights[n] for n in NAMES)))
function cell_result(q, c, i, j, k)
    ρ = c.ρ_field[i, j, k]
    ℳ = AM.grid_microphysical_state(i, j, k, c.grid, c.p3, c.μ, ρ, nothing, c.velocities)
    qf = AM.moisture_fractions(c.p3, ℳ, q)
    𝒰₀ = AM.diagnose_thermodynamic_state(i, j, k, c.grid, c.formulation, c.dynamics, qf)
    𝒰 = AM.maybe_adjust_thermodynamic_state(𝒰₀, c.p3, q, c.constants)
    tT = PPP.p3_adiabatic_temperature_tendency(ℳ, 𝒰, c.constants)
    tp = PPP.p3_adiabatic_pressure_tendency(ρ, ℳ, c.constants)
    props = PPP.p3_process_properties(c.p3, ρ, ℳ)
    return PPP.p3_tendency_compute(c.p3, ρ, ℳ, 𝒰, c.constants, props, c.μ.surface_temperature[i, j, 1], tT, tp)
end
function cell_L(q, c, i, j, k)
    r = cell_result(q, c, i, j, k); w = c.w
    return w.ρqᵛ[i, j, k] * r.tendency_ρqᵛ + w.ρqᶜˡ[i, j, k] * r.tendency_ρqᶜˡ + w.ρnᶜˡ[i, j, k] * r.tendency_ρnᶜˡ +
           w.ρnᵃ[i, j, k] * r.tendency_ρnᵃ + w.ρqʳ[i, j, k] * r.tendency_ρqʳ + w.ρnʳ[i, j, k] * r.tendency_ρnʳ
end
tendency(r, n) = getfield(r, Symbol("tendency_", n))

# Cells: NaN AD, worst finite disagreements, and agreeing rain cells as controls.
ad = g.ad; floor = 1e-6 * maximum(abs, filter(isfinite, ad))
best = reduce((a, b) -> min.(a, b), [abs.(f .- ad) ./ (abs.(ad) .+ floor) for f in values(g.fds)])
rain = g.qr .> 0
nan = findall(.!isfinite.(ad))
finite_bad = filter(c -> isfinite(ad[c]) && !(best[c] <= 1e-3), CartesianIndices(ad))
finite_bad = finite_bad[sortperm(best[finite_bad], rev=true)]
good = filter(c -> rain[c] && best[c] < 1e-6 && ad[c] != 0, CartesianIndices(ad))
cells = vcat([(:nan, c) for c in nan[1:min(5, end)]], [(:bad, c) for c in finite_bad[1:min(10, end)]],
             [(:good, c) for c in good[1:min(5, end)]])

rows = []
for (kind, c) in cells
    i, j, k = Tuple(c); q0 = Array(interior(μ.qᵛ))[i, j, k]
    native_ad = Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Forward), cell_L, Enzyme.Duplicated(q0, 1.0), Enzyme.Const(ctx), Enzyme.Const(i), Enzyme.Const(j), Enzyme.Const(k))[1]
    fd = [(ε, (cell_L(q0 + ε, ctx, i, j, k) - cell_L(q0 - ε, ctx, i, j, k)) / 2ε) for ε in (1e-6, 1e-8, 1e-10, 1e-12)]
    δ = range(-1e-7, 1e-7; length=401); scan = [cell_L(q0 + d, ctx, i, j, k) for d in δ]
    second = maximum(abs, diff(diff(scan))) / max(maximum(abs, diff(scan)), eps())
    r = cell_result(q0, ctx, i, j, k)
    push!(rows, (; kind, cell=(i, j, k), q0, reactant_ad=ad[c], native_ad, fd, scan_kink=second,
                 result=NamedTuple{Tuple(Symbol("tendency_", n) for n in NAMES)}(Tuple(tendency(r, n) for n in NAMES))))
    println(kind, " ", (i, j, k), " qv=", q0, " ρqr=", g.qr[c], " reactant_ad=", ad[c], " native_enzyme=", native_ad,
            " fd=", [round(f; sigdigits=6) for (_, f) in fd], " scan_kink=", second)
end
serialize(joinpath(output, "p3_cells.jls"), rows)
println("P3_CELLS_DONE cells=", length(rows))
