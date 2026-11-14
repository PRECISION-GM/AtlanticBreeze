# Minimal Reactant compile test of the ecCKD radiation update alone (no time stepping): it builds the March
# case on a tiny horizontal grid with the full vertical grid, continued column, cloud optics and fixed solar
# position, then compiles `_update_radiation!` and compares the compiled fluxes with native CUDA fluxes.
# ENV: ATLANTIC_NX (8; at least the WENO halo), ATLANTIC_STATE (optional spun-up cloudy state from check_gradients.jl), ATLANTIC_GRADIENT
# ("1" also compiles a reverse-mode derivative of total longwave heating w.r.t. the state).
using Reactant, Enzyme
include("model.jl")
using Oceananigans: prognostic_fields
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: update_state!
using Serialization
Reactant.set_default_backend("gpu")
# Failed pass pipelines dump their MLIR next to the results instead of node-local /tmp.
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = mkpath(joinpath(get(ENV, "ATLANTIC_OUTPUT", "outputs"), "mlir"))
note(msg; kw...) = (@info msg kw...; flush(stderr))

input = get(ENV, "ATLANTIC_INPUT", "data/interval")
nx = parse(Int, get(ENV, "ATLANTIC_NX", "8"))
dx = grid_spacing(); order = advection_order()
solar_position = FixedCosineZenith(0.58)
lookup_search = NumericalRadiation.FixedBinarySearch()
state = get(ENV, "ATLANTIC_STATE", "")
saved = isempty(state) ? nothing : deserialize(state)

function prepare!(case)
    model = case.model
    if saved !== nothing
        saved.nx == nx || error("state is for nx=$(saved.nx)")
        for (n, f) in pairs(prognostic_fields(model)); interior(f) .= saved.fields[String(n)]; end
    end
    update_state!(model)
    return model
end
fluxes(rtm) = [Array(interior(f)) for f in (rtm.upwelling_longwave_flux, rtm.downwelling_longwave_flux,
                                             rtm.upwelling_shortwave_flux, rtm.downwelling_shortwave_flux)]
radiate!(model) = (Breeze.AtmosphereModels._update_radiation!(model.radiation, model); nothing)

native = prepare!(make_case(input; nx, dx, order, architecture=GPU(), solar_position, lookup_search))
radiate!(native); F0 = fluxes(native.radiation)
note("native"; seconds=@elapsed(radiate!(native)), olr=maximum(F0[1][:, :, end]))

rmodel = CUDA.@allowscalar prepare!(make_case(input; nx, dx, order, architecture=ReactantState(), solar_position, lookup_search))
options = Reactant.CompileOptions(raise=true, raise_first=true, sync=true, excluded_passes=["concat_insert_dim_elementwise"])
t = @elapsed compiled = Reactant.@compile compile_options=options radiate!(rmodel)
compiled(rmodel); F1 = fluxes(rmodel.radiation)
parity = maximum(maximum(abs.(a .- b)) / max(maximum(abs.(a)), eps()) for (a, b) in zip(F0, F1))
note("compiled radiation"; compile_seconds=t, parity)
println(parity < 1e-10 ? "RADIATION_PARITY_PASS" : "RADIATION_PARITY_FAIL", " nx=$nx parity=$parity compile_s=$t")

if get(ENV, "ATLANTIC_GRADIENT", "0") == "1"
    heating(model) = (radiate!(model); sum(interior(model.radiation.downwelling_longwave_flux)))
    function grad!(m, dm); Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.Reverse), heating, Enzyme.Active, Enzyme.Duplicated(m, dm)); return nothing; end
    dm = Enzyme.make_zero(rmodel)
    tg = @elapsed cg = Reactant.@compile compile_options=options grad!(rmodel, dm)
    cg(rmodel, dm)
    note("radiation reverse compile"; seconds=tg, dT=maximum(abs, Array(interior(dm.temperature))))
    println("RADIATION_GRADIENT_COMPILED compile_s=$tg")
end
