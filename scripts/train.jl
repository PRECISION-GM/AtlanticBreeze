# Two-control finite-difference baseline, not an adjoint or a claim of generalization.
# Fitting uses the same interval it is scored on: training loss is not independent validation.
haskey(ENV,"ATLANTIC_STOP_SECONDS") && error("Training requires the complete sounding interval; use forward.jl for short tests")
include("model.jl")
include("plot_error.jl")
using LinearAlgebra: norm

function train()
    input=get(ENV,"ATLANTIC_INPUT","data/interval")
    output=get(ENV,"ATLANTIC_OUTPUT","outputs/training")
    isdir(output) && !isempty(readdir(output)) && error("Training output already exists")
    mkpath(output)
    nx=parse(Int,get(ENV,"ATLANTIC_NX","128")); iterations=parse(Int,get(ENV,"ATLANTIC_TRAIN_ITERATIONS","1"))
    # Wall-clock budget (s): stop before an evaluation that would not finish inside the job.
    budget=parse(Float64,get(ENV,"ATLANTIC_BUDGET_SECONDS","82800")); start=time()
    # Controls are normalized by their bounds (±3 K/day, ±2 g/kg/day), so x ∈ [-1, 1]².
    limits=[3.0,2.0]
    # FD half-steps of 1 K/day and 0.5 g/kg/day: ~0.5 K and 0.25 g/kg over 12 h, chosen to exceed
    # the run-to-run sensitivity of a chaotic 12.8 km LES to tiny forcing changes.
    h=[1.0,0.5]./limits
    x=zeros(2); counter=Ref(0); costs=Float64[]
    evaluations=open(joinpath(output,"evaluations.csv"),"w")
    println(evaluations,"evaluation,dT_Kday,dq_gkgday,loss,wall_s,directory"); flush(evaluations)
    function evaluate(x)
        isempty(costs) || time()-start+1.1maximum(costs)<budget || error("Wall-clock budget exhausted before evaluation $(counter[]+1)")
        counter[]+=1; dir=joinpath(output,"evaluation_$(lpad(counter[],3,'0'))"); t=time()
        # Every evaluation reconstructs the same initial model, including its deterministic perturbations.
        loss=forecast(input,dir;controls=x.*limits,nx)
        push!(costs,time()-t)
        println(evaluations,join([counter[],(x.*limits)...,loss,costs[end],basename(dir)],',')); flush(evaluations)
        return loss,dir
    end
    baseline=get(ENV,"ATLANTIC_BASELINE","")
    if isempty(baseline)
        loss,best=evaluate(x)
    else
        # Reuse a completed uncorrected forward run of the same input and settings instead of repeating it.
        meta=TOML.parsefile(joinpath(baseline,"run.toml"))
        meta["complete_interval"] && meta["temperature_correction_Kday"]==0 && meta["moisture_correction_gkgday"]==0 &&
            meta["input"]==abspath(input) && meta["nx"]==nx && meta["aerosol_setting"]==get(ENV,"ATLANTIC_AEROSOL","aer1") ||
            error("ATLANTIC_BASELINE is not a complete matching uncorrected run")
        loss=meta["loss"]; best=baseline; push!(costs,meta["wall_seconds"])
    end
    history=open(joinpath(output,"history.csv"),"w")
    println(history,"iteration,loss,dT_Kday,dq_gkgday,accepted,grad_T_per_unit,grad_q_per_unit"); println(history,"0,$loss,0,0,true,,"); flush(history)
    try
        for iteration in 1:iterations
            g=zeros(2)
            for k in 1:2
                e=zeros(2); e[k]=h[k]
                g[k]=(evaluate(clamp.(x+e,-1,1))[1]-evaluate(clamp.(x-e,-1,1))[1])/(2h[k])
            end
            all(isfinite,g) || error("Nonfinite finite-difference gradient")
            accepted=false
            # Steepest descent in normalized units with a trust radius of at most 0.5 (1.5 K/day, 1 g/kg/day).
            for rate in (0.5,0.25)
                candidate=clamp.(x-rate*g/max(norm(g),eps()),-1,1)
                candidate_loss,dir=evaluate(candidate)
                if isfinite(candidate_loss) && candidate_loss<loss
                    x=candidate; loss=candidate_loss; best=dir; accepted=true; break
                end
            end
            println(history,join([iteration,loss,(x.*limits)...,accepted,g...],',')); flush(history)
            accepted || break
        end
    catch err
        @error "Training stopped early" exception=err
        println(history,"# stopped: ",sprint(showerror,err))
    finally
        close(history); close(evaluations)
    end
    open(joinpath(output,"controls.toml"),"w") do io
        TOML.print(io,Dict("temperature_Kday"=>x[1]*limits[1],"moisture_gkgday"=>x[2]*limits[2],"loss"=>loss,
            "method"=>"central finite differences, normalized steepest descent with backtracking (not an adjoint)",
            "best_evaluation"=>best,"evaluations"=>counter[],"wall_seconds"=>time()-start,
            "validation"=>"in-sample fit to this interval only"))
    end
    plot_error(input,best)
end

train()
