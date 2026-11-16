# Small-grid reproducer for the traced-loop compile failure of window_gradient.jl
# (17115, 17183–17185: MLIR pass "raise_triton_custom_call" failed; while-loop operands vs body args).
# Builds the Reactant model at small nx from the interval initial state (no spinup), then runs
# @code_hlo on a list of variants, each in its own try/catch, and prints VARIANT <name> OK|FAIL <s>.
#
# ENV: ATLANTIC_INPUT, ATLANTIC_OUTPUT, ATLANTIC_NX (8), ATLANTIC_DYNAMICS,
#      ATLANTIC_VARIANTS (comma list, default all below), ATLANTIC_OPTIONS (raise | noraise).
using Reactant, Enzyme
include("model.jl")
using Oceananigans.Architectures: ReactantState
using Reactant: @trace

const T0 = time()
note(msg; kw...) = (@info msg elapsed_s=round(time() - T0; digits=1) kw...; flush(stderr))
input = ENV["ATLANTIC_INPUT"]; output = mkpath(ENV["ATLANTIC_OUTPUT"])
nx = parse(Int, get(ENV, "ATLANTIC_NX", "8")); dx = grid_spacing(100.0)
Reactant.set_default_backend("gpu")
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = mkpath(joinpath(output, "mlir"))
lookup_search = NumericalRadiation.FixedBinarySearch()
function with_large_stack(f); task = Task(f, 512 * 1024^2); schedule(task); return fetch(task); end

build = @elapsed case = with_large_stack(() -> CUDA.@allowscalar make_case(input; nx, dx, architecture=ReactantState(),
    solar_position=FixedCosineZenith(Reactant.to_rarray(fill(0.5, nx, nx))), lookup_search))
m = case.model; note("built"; build, nx, dynamics=case.dynamics)
dt = 1.0
# window_gradient.jl accessors (17291/17292: without these, `surface_temperature` resolved to another binding → "Nothing not callable")
surface_temperature(m) = m.radiation.surface_radiation.surface_temperature
cos_zenith_array(m) = m.radiation.solar_position.cos_zenith
w8 = Reactant.to_rarray(rand(nx, nx, size(interior(m.microphysical_fields.ρqʳ), 3)))
sst = Reactant.to_rarray(fill(290.0, 4)); idx = Reactant.to_rarray(collect(1.0:4))

one_step(m, dt) = (time_step!(m, dt); nothing)
loop2(m, dt) = (@trace track_numbers=false for i in 1:2; time_step!(m, dt); end; nothing)
loop2_ckpt(m, dt) = (@trace mincut=true checkpointing=true track_numbers=false for i in 1:2; time_step!(m, dt); end; nothing)
function loop2_sst(m, dt, sst, idx)
    @trace track_numbers=false for i in 1:2
        sel = ifelse.(idx .== i, 1.0, 0.0)
        parent(surface_temperature(m)) .= sum(sst .* sel)
        time_step!(m, dt)
    end
    return nothing
end
function loop_noop(m, dt, sst, idx)   # loop machinery only, no model step
    @trace track_numbers=false for i in 1:2
        parent(surface_temperature(m)) .= parent(surface_temperature(m)) .+ sum(sst .* ifelse.(idx .== i, 1.0, 0.0))
    end
    return nothing
end

function loop2_forcing(m, dt, sst, idx)   # SST + cos(zenith) writes, as window_objective
    @trace track_numbers=false for i in 1:2
        sel = ifelse.(idx .== i, 1.0, 0.0)
        parent(surface_temperature(m)) .= sum(sst .* sel)
        cos_zenith_array(m) .= sum(sst .* sel) ./ 600
        time_step!(m, dt)
    end
    return nothing
end
function loop2_record(m, dt, idx, w8)    # accumulate a field reduction into a carried vector
    acc = idx .* 0
    @trace track_numbers=false for i in 1:2
        time_step!(m, dt)
        acc = acc .+ sum(interior(m.microphysical_fields.ρqʳ) .* w8) .* ifelse.(idx .== i, 1.0, 0.0)
    end
    return sum(acc)
end
function nested_ckpt(m, dt)              # two nested checkpointed loops, as window_objective
    @trace mincut=true checkpointing=true track_numbers=false for i in 1:2
        @trace mincut=true checkpointing=true track_numbers=false for j in 1:1
            time_step!(m, dt)
        end
    end
    return nothing
end
variants = Dict("loop2_forcing" => () -> Reactant.@code_hlo(compile_options=opts, loop2_forcing(m, dt, sst, idx)),
                "loop2_record" => () -> Reactant.@code_hlo(compile_options=opts, loop2_record(m, dt, idx, w8)),
                "nested_ckpt" => () -> Reactant.@code_hlo(compile_options=opts, nested_ckpt(m, dt)),
                "one_step" => () -> Reactant.@code_hlo(compile_options=opts, one_step(m, dt)),
                "loop_noop" => () -> Reactant.@code_hlo(compile_options=opts, loop_noop(m, dt, sst, idx)),
                "loop2" => () -> Reactant.@code_hlo(compile_options=opts, loop2(m, dt)),
                "loop2_ckpt" => () -> Reactant.@code_hlo(compile_options=opts, loop2_ckpt(m, dt)),
                "loop2_sst" => () -> Reactant.@code_hlo(compile_options=opts, loop2_sst(m, dt, sst, idx)))
opts = get(ENV, "ATLANTIC_OPTIONS", "raise") == "raise" ?
    Reactant.CompileOptions(raise=true, raise_first=true, sync=true, excluded_passes=["concat_insert_dim_elementwise"]) :
    Reactant.CompileOptions(raise=false, sync=true)
for name in split(get(ENV, "ATLANTIC_VARIANTS", "one_step,loop_noop,loop2,loop2_ckpt,loop2_sst"), ",")
    t = time()
    try
        with_large_stack(variants[name])
        println("VARIANT $name OK $(round(time() - t; digits=1))")
    catch err
        showerror(stderr, err, catch_backtrace()); println(stderr); flush(stderr)   # full stack for the log
        msg = first(sprint(showerror, err isa TaskFailedException ? err.task.exception : err), 400)
        println("VARIANT $name FAIL $(round(time() - t; digits=1)) :: $msg")
    end
    flush(stdout)
end
println("LOOP_TRACE_REPRO_DONE")
