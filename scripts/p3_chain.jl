# Intermediate-level derivative probe of the P3 rain-evaporation chain at single cells (native Enzyme vs
# central differences, no Reactant). It replays, with the vapour input qᵛ passed explicitly, the steps of
# p3_grid_process_inputs → compute_p3_process_rates → coupled_saturation_adjustment_rates, and reports
# d/dqᵛ of each intermediate so the first divergence from finite differences can be located.
# It first checks that the CPU primal tendencies equal the native GPU kernel's, over the whole field.
# Scope: one weighted vapour partial of one kernel; not the full Jacobian, nor full-case gradients.
# ENV: ATLANTIC_STATE, ATLANTIC_GRADIENT_FILE (p3_gradient.jls), ATLANTIC_DX, ATLANTIC_OUTPUT, ATLANTIC_CELLS
#      ("i,j,k;i,j,k", default: the three worst disagreeing cells and one agreeing rain cell),
#      ATLANTIC_SKIP_GPU=1 (skip the GPU primal parity), ATLANTIC_FLUSH=1 (flush-to-zero test: after the
#      original pass, set every subnormal microphysical input at the cell to 0, as a GPU flush-to-zero mode
#      would, and repeat the chain; the original values are restored afterwards).
using Enzyme
include("model.jl")
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Architectures: on_architecture
using Serialization, Printf
const AM = Breeze.AtmosphereModels
const PPP = Breeze.Microphysics.PredictedParticleProperties
note(msg; kw...) = (@info msg kw...; flush(stderr))

input = get(ENV, "ATLANTIC_INPUT", "data/interval")
output = mkpath(get(ENV, "ATLANTIC_OUTPUT", "outputs/p3_chain"))
saved = deserialize(ENV["ATLANTIC_STATE"]);
nx = saved.nx
g = deserialize(ENV["ATLANTIC_GRADIENT_FILE"])
kw = (;
    nx,
    dx = grid_spacing(),
    order = advection_order(),
    solar_position = FixedCosineZenith(0.58),
    lookup_search = NumericalRadiation.FixedBinarySearch(),
)
function build(arch)
    model = make_case(input; kw..., architecture = arch).model
    for (n, f) in pairs(prognostic_fields(model))
        interior(f) .= on_architecture(arch, saved.fields[String(n)])
    end
    update_state!(model)
    PPP.compute_p3_surface_temperature!(
        model.microphysical_fields.surface_temperature,
        model.temperature,
        model.grid,
    )
    return model
end
G(model) = PPP.p3_tendency_fields(model.timestepper.Gⁿ, model.microphysics)
function p3!(model)
    for f in G(model)
        fill!(parent(f), 0)
    end
    AM.compute_microphysical_tendencies!(model.microphysics, model)
    return Dict(String(n) => Array(interior(f)) for (n, f) in pairs(G(model)))
end

# 1. Primal parity: the CPU cell replay must reproduce the native GPU kernel before any attribution.
model = build(CPU())
cpu = p3!(model)
gpu =
    get(ENV, "ATLANTIC_SKIP_GPU", "0") == "1" ? Dict{String,Array{Float64,3}}() : p3!(build(GPU()))
for n in sort(collect(keys(gpu)))
    d = maximum(abs.(cpu[n] .- gpu[n]))
    s = maximum(abs, gpu[n])
    println(
        @sprintf(
            "primal CPU-vs-GPU %-6s max|Δ|=%.3e max|G|=%.3e rel=%.3e",
            n,
            d,
            s,
            d / max(s, floatmin())
        ),
    )
end

μ = model.microphysical_fields;
p3 = model.microphysics
ctx = (;
    ρ_field = AM.total_density(model.dynamics),
    μ,
    p3,
    grid = model.grid,
    constants = model.thermodynamic_constants,
    velocities = model.velocities,
    formulation = model.formulation,
    dynamics = model.dynamics,
)

# 2. The chain, as in the kernel (p3_driver.jl, process_rates.jl, coupled_saturation_adjustment.jl).
function chain(q, c, i, j, k)
    p3 = c.p3
    constants = c.constants
    FT = typeof(q)
    ρ = c.ρ_field[i, j, k]
    ℳ = AM.grid_microphysical_state(i, j, k, c.grid, p3, c.μ, ρ, nothing, c.velocities)
    qf = AM.moisture_fractions(p3, ℳ, q)
    𝒰₀ = AM.diagnose_thermodynamic_state(i, j, k, c.grid, c.formulation, c.dynamics, qf)
    𝒰 = AM.maybe_adjust_thermodynamic_state(𝒰₀, p3, q, constants)
    tT = PPP.p3_adiabatic_temperature_tendency(ℳ, 𝒰, constants)
    tp = PPP.p3_adiabatic_pressure_tendency(ρ, ℳ, constants)
    props = PPP.p3_process_properties(p3, ρ, ℳ)
    Ts = c.μ.surface_temperature[i, j, 1]

    T_diag = PPP.temperature(𝒰, constants)
    ρᵗʰ = PPP.density(𝒰, constants)
    qsat_diag = PPP.saturation_specific_humidity(T_diag, ρᵗʰ, constants, PPP.PlanarLiquidSurface())
    adj = PPP.prognostic_vapor_excess_adjustment(
        p3,
        ℳ.qᶜˡ,
        qsat_diag,
        ℳ.δqᵛ⁺ˡ,
        T_diag,
        ρᵗʰ,
        𝒰,
        constants,
    )
    transport = PPP.air_transport_properties(adj.T, adj.P, constants)
    qʳ = ℳ.qʳ
    nʳ = ifelse(
        qʳ > p3.minimum_mass_mixing_ratio,
        PPP.rain_number_from_slope(max(0, qʳ), props.λʳ, p3.process_rates),
        zero(FT),
    )
    rain_relaxation = PPP.rain_vapor_relaxation_coefficient(p3, qʳ, nʳ, ρ, transport)
    ℒˡ = PPP.vaporization_latent_heat(constants, adj.T)
    external_driver =
        PPP.external_vapor_excess_forcing(adj.qᵛ⁺ˡ, ℒˡ, adj.T, adj.P, tT, tp, constants)
    # Replay of compute_p3_process_rates' setup (process_rates.jl) to reach phase 1 and the coupled rates.
    parameters = p3.process_rates
    qʷⁱ = PPP.active_liquid_on_ice(p3, ℳ.qʷⁱ)
    cloud = PPP.diagnose_cloud_dsd(p3, adj.qᶜˡ, ℳ.nᶜˡ, ρ)
    ℳa = PPP.P3MicrophysicalState(
        adj.qᶜˡ,
        ℳ.nᶜˡ,
        qʳ,
        ℳ.nʳ,
        ℳ.qⁱ,
        ℳ.nⁱ,
        props.qᶠ,
        props.bᶠ,
        qʷⁱ,
        adj.qᵛ - adj.qᵛ⁺ˡ,
        ℳ.nᵃ,
        ℳ.w,
    )
    qsat_ice = PPP.p3_ice_saturation_specific_humidity(
        adj.T,
        adj.ρᵗʰ,
        constants,
        parameters.freezing_temperature,
        adj.qᵛ⁺ˡ,
    )
    lookups = PPP.p3_ice_lookups(p3, ℳ.qⁱ, qʷⁱ, props.nⁱ, props.Fᶠ, props.Fˡ, props.ρᶠ, ρ)
    state = PPP.P3DerivedState{FT,typeof(adj.q),typeof(lookups)}(
        props.nⁱ,
        nʳ,
        props.qᶠ,
        props.bᶠ,
        props.Fᶠ,
        props.ρᶠ,
        props.Fˡ,
        cloud.Nᶜˡ,
        cloud.nᶜˡ,
        cloud.μᶜˡ,
        cloud.λᶜˡ,
        adj.T,
        adj.P,
        adj.qᵛ,
        adj.qᵛ⁺ˡ,
        qsat_ice,
        adj.q,
        adj.ρᵗʰ,
        transport.Dᵛ,
        transport.Kᵃ,
        transport.ν,
        lookups,
    )
    phase1 = PPP.p3_phase1_rates(p3, ρ, ℳa, constants, state, tT, tp)
    # Internals of coupled_saturation_adjustment_rates for the rain share (coupled_saturation_adjustment.jl).
    τ = max(parameters.sink_limiting_timescale, eps(FT))
    Rᵛ = FT(PPP.vapor_gas_constant(constants))
    cᵖᵈ = PPP.p3_dry_air_heat_capacity(constants, FT)
    ξˡ = 1 + ℒˡ * (adj.qᵛ⁺ˡ * ℒˡ / (Rᵛ * adj.T^2)) / cᵖᵈ
    cloud_relaxation = PPP.cloud_vapor_relaxation_coefficient(
        p3,
        adj.qᶜˡ,
        ρ,
        transport.Dᵛ,
        cloud.μᶜˡ,
        cloud.λᶜˡ,
        cloud.nᶜˡ,
    )
    total_relaxation = max(cloud_relaxation + rain_relaxation, FT(parameters.floors.rate_scale))
    transient_response = (1 - exp(-total_relaxation * τ)) / τ
    raw_rain_growth = PPP.growth_share(
        rain_relaxation,
        ξˡ,
        external_driver,
        total_relaxation,
        adj.qᵛ - adj.qᵛ⁺ˡ,
        transient_response,
    )
    # Algebraically identical, numerically stable rain share: with a = xτ (x = total relaxation),
    # ext·f - (ext/x)·f·transient = ext·f·φ(a), φ(a) = 1 - (1 - e⁻ᵃ)/a, and transient = -expm1(-a)/τ.
    # Diagnostic only: tests whether the FD plateau of raw_rain_growth is a cancellation artifact.
    a = total_relaxation * τ
    φ = a < 1e-3 ? a / 2 - a^2 / 6 + a^3 / 24 - a^4 / 120 : 1 - (-expm1(-a)) / a
    transient_stable = -expm1(-a) / τ
    f_rain = rain_relaxation / total_relaxation
    raw_rain_growth_stable =
        (external_driver * f_rain * φ + (adj.qᵛ - adj.qᵛ⁺ˡ) * f_rain * transient_stable) / ξˡ
    vr = PPP.coupled_saturation_adjustment_rates(
        p3,
        ℳa.qᶜˡ,
        qʳ,
        nʳ,
        ℳa.qⁱ,
        qʷⁱ,
        props.nⁱ,
        adj.qᵛ,
        adj.qᵛ⁺ˡ,
        qsat_ice,
        props.Fᶠ,
        props.ρᶠ,
        adj.T,
        adj.P,
        ρ,
        constants,
        transport,
        adj.q,
        cloud.μᶜˡ,
        cloud.λᶜˡ,
        cloud.nᶜˡ,
        tT,
        tp,
        lookups,
    )
    rates = PPP.compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, props, Ts, tT, tp)
    result = PPP.p3_tendency_compute(p3, ρ, ℳ, 𝒰, constants, props, Ts, tT, tp)
    return (;
        q_in = q,
        qv_moisture = qf.vapor,
        T_diag,
        ρᵗʰ,
        qsat_diag,
        adj_rate = adj.rate,
        adj_qv = adj.qᵛ,
        adj_T = adj.T,
        adj_qsat = adj.qᵛ⁺ˡ,
        vapor_excess = adj.qᵛ - adj.qᵛ⁺ˡ,
        ν = transport.ν,
        Dᵛ = transport.Dᵛ,
        rain_relaxation,
        external_driver,
        ξˡ,
        cloud_relaxation,
        total_relaxation,
        transient_response,
        raw_rain_growth,
        a_relaxation = a,
        transient_stable,
        raw_rain_growth_stable,
        coupled_rain_evaporation = vr.rain_evaporation,
        phase1_rain_evaporation = phase1.rain_evaporation,
        rate_rain_evaporation = rates.rain_evaporation,
        rate_rain_evaporation_number = rates.rain_evaporation_number,
        rate_rain_condensation = rates.rain_condensation,
        tendency_ρqʳ = result.tendency_ρqʳ,
        tendency_ρnʳ = result.tendency_ρnʳ,
        tendency_ρqᵛ = result.tendency_ρqᵛ,
    )
end
component(q, c, i, j, k, ::Val{n}) where {n} = getfield(chain(q, c, i, j, k), n)

# Cells: worst disagreeing finite cells (as in p3_cells.jl) plus one agreeing rain cell.
ad = g.ad;
floor_ = 1e-6 * maximum(abs, filter(isfinite, ad))
best = reduce((a, b) -> min.(a, b), [abs.(f .- ad) ./ (abs.(ad) .+ floor_) for f in values(g.fds)])
bad = filter(c -> isfinite(ad[c]) && !(best[c] <= 1e-3), CartesianIndices(ad));
bad = bad[sortperm(best[bad], rev = true)]
good = filter(c -> g.qr[c] > 0 && best[c] < 1e-6 && ad[c] != 0, CartesianIndices(ad))
nancells = findall(.!isfinite.(ad))
cells =
    get(ENV, "ATLANTIC_CELLS", "") == "nan" ? vcat(nancells[1:min(2, end)], nancells[end:end]) :
    haskey(ENV, "ATLANTIC_CELLS") ?
    [CartesianIndex(parse.(Int, split(s, ","))...) for s in split(ENV["ATLANTIC_CELLS"], ";")] :
    vcat(bad[1:min(3, end)], good[1:min(1, end)])

εs = (1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12)
rows = []
function probe(c, label)
    i, j, k = Tuple(c)
    q0 = Array(interior(μ.qᵛ))[i, j, k]
    base = chain(q0, ctx, i, j, k)
    println(
        "\n== ",
        label,
        " cell ",
        (i, j, k),
        " q0=",
        q0,
        " ρqʳ=",
        g.qr[c],
        " ρqᶜˡ=",
        g.qc[c],
        " saved reactant ad=",
        ad[c],
    )
    for n in keys(base)
        a = Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Forward),
            component,
            Enzyme.Duplicated(q0, 1.0),
            Enzyme.Const(ctx),
            Enzyme.Const(i),
            Enzyme.Const(j),
            Enzyme.Const(k),
            Enzyme.Const(Val(n)),
        )[1]
        fd = [
            (component(q0 + ε, ctx, i, j, k, Val(n)) - component(q0 - ε, ctx, i, j, k, Val(n))) / 2ε for ε in εs
        ]
        # One-sided slopes expose a branch crossing at q0 (left ≠ right while each is stable).
        left = (base[n] - component(q0 - 1e-9, ctx, i, j, k, Val(n))) / 1e-9
        right = (component(q0 + 1e-9, ctx, i, j, k, Val(n)) - base[n]) / 1e-9
        scale = max(abs(a), maximum(abs, fd), floatmin())
        println(
            @sprintf(
                "%-30s value=% .6e enzyme=% .6e fd(1e-8)=% .6e fd(1e-10)=% .6e left=% .4e right=% .4e |Δ|/scale=%.2e",
                n,
                base[n],
                a,
                fd[4],
                fd[6],
                left,
                right,
                abs(a - fd[4]) / scale
            ),
        )
        push!(
            rows,
            (;
                label,
                cell = (i, j, k),
                name = n,
                value = base[n],
                enzyme = a,
                fd = collect(zip(εs, fd)),
                left,
                right,
            ),
        )
    end
end
for c in cells
    probe(c, "original")
    get(ENV, "ATLANTIC_FLUSH", "0") == "1" || continue
    flushed = Pair{Symbol,Float64}[]
    for (n, f) in pairs(μ)
        f isa Oceananigans.AbstractField &&
        ndims(interior(f)) == 3 &&
        size(interior(f), 3) >= c[3] || continue
        v = f[c[1], c[2], c[3]]
        if issubnormal(v)
            push!(flushed, n => v)
            f[c[1], c[2], c[3]] = 0.0
        end
    end
    println("\nflushed subnormal inputs at ", Tuple(c), ": ", flushed)
    isempty(flushed) || probe(c, "flushed")
    for (n, v) in flushed
        μ[n][c[1], c[2], c[3]] = v
    end
end
serialize(joinpath(output, "p3_chain.jls"), rows)
println("P3_CHAIN_DONE")
