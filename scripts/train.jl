# Two-control finite-difference baseline, not an adjoint or a claim of generalization.
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
    controls=zeros(2);steps=[0.1,0.1];limits=[3.0,2.0];counter=Ref(0)
    function evaluate(x)
     counter[]+=1
     # Every evaluation reconstructs the same initial model, including its random perturbations.
     forecast(input,joinpath(output,"evaluation_$(lpad(counter[],3,'0'))");controls=x,nx)
    end
    loss=evaluate(controls)
    best_output=joinpath(output,"evaluation_$(lpad(counter[],3,'0'))")
    open(joinpath(output,"history.csv"),"w") do io
     println(io,"iteration,loss,dT_Kday,dq_gkgday,accepted")
     println(io,"0,$loss,0,0,true")
     for iteration in 1:iterations
      gradient=zeros(2)
      for k in 1:2
       direction=zeros(2);direction[k]=steps[k]
       gradient[k]=(evaluate(controls+direction)-evaluate(controls-direction))/(2steps[k])
      end
      all(isfinite,gradient) || error("Nonfinite finite-difference gradient")
      direction=gradient/max(norm(gradient),1.0)
      accepted=false
      for rate in (0.5,0.25,0.125)
       candidate=clamp.(controls-rate*direction,-limits,limits)
       candidate_loss=evaluate(candidate)
       if isfinite(candidate_loss) && candidate_loss<loss
        controls=candidate;loss=candidate_loss;accepted=true
        best_output=joinpath(output,"evaluation_$(lpad(counter[],3,'0'))")
        break
       end
      end
      println(io,"$iteration,$loss,$(controls[1]),$(controls[2]),$accepted");flush(io)
      accepted || break
     end
    end
    open(joinpath(output,"controls.toml"),"w") do io
     TOML.print(io,Dict("temperature_Kday"=>controls[1],"moisture_gkgday"=>controls[2],"loss"=>loss,"method"=>"central finite differences with backtracking", "best_evaluation"=>basename(best_output)))
    end
    plot_error(input,best_output)
end

train()
