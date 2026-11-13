# Shared forward model and observation loss. All times are seconds since sounding launch.
using Breeze, Oceananigans, CUDA
import NumericalRadiation
using Oceananigans.Units
using Oceananigans.Grids: znodes
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Dates, DelimitedFiles, Statistics, TOML, Random
include(joinpath(pkgdir(Breeze), "examples", "lasso_ena", "LassoENA.jl"))
const L = LassoENA
readcsv(path) = readdlm(path, ',', Float64; skipstart=1)

# aer1 matches March 4 surface UHSAS N(D>60 nm) ≈ 120–165 cm⁻³; LASSO aer2 is about twice that.
function make_case(input; controls=zeros(2), nx=nothing, architecture=GPU(), aerosol_setting=:aer1)
    config = TOML.parsefile(joinpath(input, "interval.toml"))
    isnothing(nx) && (nx = config["nx"])
    Oceananigans.defaults.FloatType = Float64
    faces = vec(readdlm(joinpath(input, "faces.csv"), ',', Float64))
    initial = readcsv(joinpath(input, "initial.csv"))
    forcing_data = readcsv(joinpath(input, "forcing.csv"))
    z = initial[:, 1]; nz = length(z); times = unique(forcing_data[:, 1])
    grid = RectilinearGrid(architecture, Float64; size=(nx, nx, nz), x=(0, nx*100), y=(0, nx*100),
                          z=faces, halo=(6, 6, 5), topology=(Periodic, Periodic, Bounded))  # minimum halo Oceananigans requires for this WENO9 setup
    constants = Breeze.ThermodynamicConstants(Float64)
    profiles = L.SoundingProfiles(z, initial[:,3], initial[:,4], initial[:,5], initial[:,6],
                                  initial[1,2]*exp(9.80665*z[1]/(287.05*initial[1,3]*(initial[1,2]/1e5)^0.2857)), false)
    reference = L.sounding_reference_state(grid, profiles, constants)
    pressure = Array(interior(reference.pressure, 1, 1, :))
    density = Array(interior(reference.density, 1, 1, :))
    aerosol = L.lasso_aerosol(; setting=aerosol_setting, reference_density=density[1], maximum_supersaturation=0.003)
    microphysics = P3Microphysics(; cloud=L.CloudDroplets(; number_concentration=75e6), aerosol)
    # Controls: added dT/dt [K/day] and dq/dt [g/kg/day], tapered out by 2500 m.
    taper = clamp.((2500 .- z)./1000, 0, 1)
    columns(c) = [forcing_data[forcing_data[:,1].==t,c] for t in times]
    tcols = [c .+ controls[1].*taper./86400 for c in columns(3)]
    qcols = [c .+ controls[2].*taper./(1000*86400) for c in columns(4)]
    series(c) = L.profile_time_series(grid,times,columns(c))
    thermo = L.large_scale_thermodynamic_forcings(L.profile_time_series(grid,times,tcols),
        L.profile_time_series(grid,times,qcols); microphysics, thermodynamic_constants=constants,
        moisture_name=:qᵛ, moisture_basis=:mass_fraction)
    geostrophic = L.time_varying_geostrophic_forcings(series(6),series(7))
    subsidence = L.LargeScaleVerticalAdvection(series(5))
    sponge = L.SAMSponge(; damping_depth_fraction=0.2)
    forcing = Dict{Symbol,Tuple}(:u=>(geostrophic.u,subsidence,sponge),
        :v=>(geostrophic.v,subsidence,sponge), :w=>(sponge,), :E=>(thermo.E,), :θ=>(subsidence,))
    for name in L.specific_prognostic_names(microphysics); forcing[name]=(subsidence,); end
    forcing[:qᵛ]=(thermo.qᵛ,subsidence)
    sst_values=[first(c) for c in columns(8)]
    surface_temperature=Field{Center,Center,Nothing}(grid)
    set!(surface_temperature,L.interpolate_profile(times,sst_values,0))
    fill_halo_regions!(surface_temperature)
    surface_density=L.sam_surface_interface_density(pressure,z,constants.gravitational_acceleration)
    boundary_conditions=L.sam_ocean_surface_flux_boundary_conditions(grid,surface_temperature;
        moisture_name=:qᵛ,microphysics,thermodynamic_constants=constants,surface_density)
    radiation=RadiativeTransferModel(grid,EcCKDOptics(; clouds=CloudScatteringTables()),constants;
        surface_temperature,surface_albedo=0.07,surface_emissivity=0.95,
        background_atmosphere=BackgroundAtmosphere(CO₂=405e-6,CH₄=1.85e-6,N₂O=330e-9),
        solar_position=ApparentSolarPosition(;coordinate=(-28.0257,39.0916),epoch=DateTime(config["epoch"])),
        schedule=IterationInterval(30),liquid_effective_radius=ConstantRadiusParticles(14e-6),
        ice_effective_radius=ConstantRadiusParticles(30e-6),
        # Radiation sees a continued column above the 8 km LES top: 40 layers to 65 km, standard-atmosphere
        # T/q anchored to the grid-top temperature, and the background standard ozone profile.
        overlying_atmosphere=OverlyingAtmosphereColumn(Float64; top=65e3, layers=40))
    model=AtmosphereModel(grid; dynamics=AnelasticDynamics(reference),formulation=:LiquidIcePotentialTemperature,
        microphysics,radiation,forcing=L.compact_forcing(forcing),boundary_conditions,thermodynamic_constants=constants,
        coriolis=FPlane(f=9.19626e-5),closure=nothing,momentum_advection=WENO(order=9),
        scalar_advection=L.scalar_advection_schemes(9,microphysics,:qᵛ))
    state=L.initial_state_columns(profiles,z,pressure;constants)
    # The observations are dry theta, not Breeze liquid-ice theta. Initialize via T.
    state=merge(state,(;T_condensate_free=initial[:,3].*(pressure./1e5).^0.2857))
    L.set_sounding_initial_state!(model,state;perturbation=L.InitialPerturbation(;seed=3042016),
        moisture_basis=:mass_fraction,initialization=:condensate_free)
    # Host interpolation gives set! a scalar, avoiding the GPU-indexing closure in the example.
    function refresh!(sim)
        set!(surface_temperature,L.interpolate_profile(times,sst_values,sim.model.clock.time))
        fill_halo_regions!(surface_temperature)
    end
    return (;model,config,z,faces,pressure,refresh!)
end

function forecast(input, output; controls=zeros(2), stop_time=nothing, nx=nothing, cpu=false,
                  aerosol_setting=Symbol(get(ENV,"ATLANTIC_AEROSOL","aer1")))
    isdir(output) && !isempty(readdir(output)) && error("Output directory is not empty: $output")
    mkpath(output)
    wall0=time()
    case=make_case(input;controls,nx,architecture=cpu ? CPU() : GPU(),aerosol_setting)
    model=case.model; config=case.config
    duration=isnothing(stop_time) ? config["duration"] : min(stop_time,config["duration"])
    meta=Dict{String,Any}("duration"=>duration,"complete_interval"=>false,"loss"=>NaN,
        "temperature_correction_Kday"=>controls[1],"moisture_correction_gkgday"=>controls[2],"nx"=>size(model.grid,1),
        "input"=>abspath(input),"precision"=>"Float64","aerosol"=>"prognostic","aerosol_setting"=>string(aerosol_setting),
        "dt"=>config["dt"],"advection"=>"WENO9 momentum and scalars","closure"=>"none (implicit WENO dissipation)","commit"=>get(ENV,"ATLANTIC_COMMIT","unknown"))
    writemeta()=open(io->TOML.print(io,meta),joinpath(output,"run.toml"),"w")
    writemeta()  # Records an incomplete run until the end is reached.
    simulation=Simulation(model;Δt=config["dt"],stop_time=duration)
    add_callback!(simulation,case.refresh!,IterationInterval(1))
    records=Vector{Float64}[]
    header=["time_s" "LWP_gm2" "max_abs_u_ms" "max_abs_w_ms" "cfl" "wall_s"]
    Δzmin=minimum(diff(case.faces))
    function diagnose(sim)
        m=sim.model; water=m.microphysical_fields
        # Total P3 liquid includes rain and liquid coating on ice.
        liquid=Array(interior(water.ρqᶜˡ))+Array(interior(water.ρqʳ))+Array(interior(water.ρqʷⁱ))
        lwp=1000sum(vec(mean(liquid;dims=(1,2))).*diff(case.faces))
        U=max(maximum(abs,interior(m.velocities.u)),maximum(abs,interior(m.velocities.v)))
        W=maximum(abs,interior(m.velocities.w))
        cfl=config["dt"]*(2U/100+W/Δzmin)
        row=[m.clock.time,lwp,U,W,cfl,time()-wall0]
        all(isfinite,row) || error("Nonfinite diagnostics at t=$(m.clock.time)")
        cfl<0.8 || error("CFL $(cfl) exceeds 0.8 at t=$(m.clock.time)")
        push!(records,row); @info "forward" time=m.clock.time lwp W cfl wall=row[end]
        # Rewritten every minute: an interrupted run keeps its history (no restart is claimed).
        writedlm(joinpath(output,"lwp.csv"),vcat(header,reduce(vcat,permutedims.(records))),',')
    end
    add_callback!(simulation,diagnose,TimeInterval(60))
    run!(simulation)
    (isempty(records) || last(records)[1]!=model.clock.time) && diagnose(simulation)
    result=reduce(vcat,permutedims.(records))
    average(field)=vec(mean(Array(interior(field));dims=(1,2)))
    theta=average(model.temperature).*(1e5./case.pressure).^0.2857
    final=hcat(case.z,theta,average(model.microphysical_fields.qᵛ),average(model.velocities.u),average(model.velocities.v))
    all(isfinite,final) || error("Nonfinite final profile")
    writedlm(joinpath(output,"final.csv"),vcat(["z_m" "theta_d_K" "qv_kgkg" "u_ms" "v_ms"],final),',')
    complete=model.clock.time>=config["duration"]-config["dt"]/2
    if complete
        terms=observation_loss(input,config,result,final,case.z)
        meta["loss_terms"]=terms; meta["loss"]=terms["lwp"]+terms["theta"]+terms["qv"]
    end
    meta["complete_interval"]=complete; meta["wall_seconds"]=time()-wall0; meta["final_time"]=model.clock.time
    writemeta()
    return meta["loss"]
end

# Balancing scales (50 g m⁻², 1 K, 1 g kg⁻¹), not observational error bars.
function observation_loss(input,config,result,final,z)
    obs=readcsv(joinpath(input,"lwp.csv")); errors=Float64[]
    for row in eachrow(obs)
        row[1]>=config["spinup"] && isfinite(row[3]) || continue
        select=(result[:,1].>=row[1]).&(result[:,1].<row[2])
        any(select) && push!(errors,((mean(result[select,2])-row[3])/50)^2)
    end
    isempty(errors) && error("No post-spinup valid LWP comparison bins")
    target=readcsv(joinpath(input,"target.csv")); keep=z.<=2500
    return Dict("lwp"=>mean(errors),"theta"=>mean((final[keep,2]-target[keep,3]).^2),
                "qv"=>mean(((final[keep,3]-target[keep,4])./.001).^2))
end
