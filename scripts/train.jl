# Two-control finite-difference baseline, not an adjoint or a claim of generalization.
# Fitting uses the same interval it is scored on: training loss is not independent validation.
haskey(ENV, "ATLANTIC_STOP_SECONDS") &&
    error("Training requires the complete sounding interval; use forward.jl for short tests")
include("model.jl")
include("plot_error.jl")
using LinearAlgebra: norm

function train()
    input=get(ENV, "ATLANTIC_INPUT", "data/interval")
    output=get(ENV, "ATLANTIC_OUTPUT", "outputs/training")
    isdir(output) && !isempty(readdir(output)) && error("Training output already exists")
    mkpath(output)
    nx=haskey(ENV, "ATLANTIC_NX") ? parse(Int, ENV["ATLANTIC_NX"]) :
       TOML.parsefile(joinpath(input, "interval.toml"))["nx"]
    iterations=parse(Int, get(ENV, "ATLANTIC_TRAIN_ITERATIONS", "1"))
    # Wall-clock budget (s): stop before an evaluation that would not finish inside the job.
    budget=parse(Float64, get(ENV, "ATLANTIC_BUDGET_SECONDS", "82800"))
    start=time()
    # Controls are normalized by their bounds (±3 K/day, ±2 g/kg/day), so x ∈ [-1, 1]².
    control_scales=[3.0, 2.0]
    # FD half-steps of 1 K/day and 0.5 g/kg/day: ~0.5 K and 0.25 g/kg over 12 h, chosen to exceed
    # the run-to-run sensitivity of a chaotic 12.8 km LES to tiny forcing changes.
    difference_steps=[1.0, 0.5] ./ control_scales
    normalized_controls=zeros(2)
    evaluation_count=Ref(0)
    evaluation_times=Float64[]
    evaluations=open(joinpath(output, "evaluations.csv"), "w")
    println(evaluations, "evaluation,dT_Kday,dq_gkgday,loss,wall_s,directory")
    flush(evaluations)
    function evaluate(normalized_controls)
        isempty(evaluation_times) ||
            time()-start+1.1maximum(evaluation_times)<budget ||
            error("Wall-clock budget exhausted before evaluation $(evaluation_count[]+1)")
        evaluation_count[]+=1
        dir=joinpath(output, "evaluation_$(lpad(evaluation_count[],3,'0'))")
        t=time()
        # Every evaluation reconstructs the same initial model, including its deterministic perturbations.
        loss=forecast(input, dir; controls = normalized_controls .* control_scales, nx)
        push!(evaluation_times, time()-t)
        println(
            evaluations,
            join(
                [
                    evaluation_count[],
                    (normalized_controls .* control_scales)...,
                    loss,
                    evaluation_times[end],
                    basename(dir),
                ],
                ',',
            ),
        )
        flush(evaluations)
        return loss, dir
    end
    baseline=get(ENV, "ATLANTIC_BASELINE", "")
    if isempty(baseline)
        loss, best_directory=evaluate(normalized_controls)
    else
        # Reuse a completed uncorrected forward run of the same input and settings instead of repeating it.
        meta=TOML.parsefile(joinpath(baseline, "run.toml"))
        meta["complete_interval"] &&
        meta["temperature_correction_Kday"]==0 &&
        meta["moisture_correction_gkgday"]==0 &&
        meta["input"]==abspath(input) &&
        meta["nx"]==nx &&
        meta["aerosol_setting"]==get(ENV, "ATLANTIC_AEROSOL", "aer1") &&
        meta["order"]==advection_order() &&
        meta["dx_m"]==grid_spacing(TOML.parsefile(joinpath(input, "interval.toml"))["dx"]) ||
            error("ATLANTIC_BASELINE is not a complete matching uncorrected run")
        loss=meta["loss"]
        best_directory=baseline
        push!(evaluation_times, meta["wall_seconds"])
    end
    history=open(joinpath(output, "history.csv"), "w")
    println(history, "iteration,loss,dT_Kday,dq_gkgday,accepted,grad_T_per_unit,grad_q_per_unit")
    println(history, "0,$loss,0,0,true,,")
    flush(history)
    try
        for iteration = 1:iterations
            gradient=zeros(2)
            for k = 1:2
                perturbation=zeros(2)
                perturbation[k]=difference_steps[k]
                gradient[k]=(
                    evaluate(clamp.(normalized_controls+perturbation, -1, 1))[1]-evaluate(
                        clamp.(normalized_controls-perturbation, -1, 1),
                    )[1]
                )/(2 * difference_steps[k])
            end
            all(isfinite, gradient) || error("Nonfinite finite-difference gradient")
            accepted=false
            # Steepest descent in normalized units with a trust radius of at most 0.5 (1.5 K/day, 1 g/kg/day).
            for rate in (0.5, 0.25)
                candidate=clamp.(
                    normalized_controls-rate*gradient/max(norm(gradient), eps()),
                    -1,
                    1,
                )
                candidate_loss, dir=evaluate(candidate)
                if isfinite(candidate_loss) && candidate_loss<loss
                    normalized_controls=candidate
                    loss=candidate_loss
                    best_directory=dir
                    accepted=true
                    break
                end
            end
            println(
                history,
                join(
                    [
                        iteration,
                        loss,
                        (normalized_controls .* control_scales)...,
                        accepted,
                        gradient...,
                    ],
                    ',',
                ),
            )
            flush(history)
            accepted || break
        end
    catch err
        @error "Training stopped early" exception=err
        println(history, "# stopped: ", sprint(showerror, err))
    finally
        close(history)
        close(evaluations)
    end
    open(joinpath(output, "controls.toml"), "w") do io
        TOML.print(
            io,
            Dict(
                "temperature_Kday"=>normalized_controls[1]*control_scales[1],
                "moisture_gkgday"=>normalized_controls[2]*control_scales[2],
                "loss"=>loss,
                "method"=>"central finite differences, normalized steepest descent with backtracking (not an adjoint)",
                "best_evaluation"=>best_directory,
                "evaluations"=>evaluation_count[],
                "wall_seconds"=>time()-start,
                "validation"=>"in-sample fit to this interval only",
            ),
        )
    end
    plot_error(input, best_directory)
end

train()
