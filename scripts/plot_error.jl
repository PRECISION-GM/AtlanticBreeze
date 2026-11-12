using CairoMakie, DelimitedFiles, TOML, Statistics
function plot_error(input,output)
 readcsv(p)=readdlm(p,',',Float64;skipstart=1)
 run=readcsv(joinpath(output,"lwp.csv"));obs=readcsv(joinpath(input,"lwp.csv"));target=readcsv(joinpath(input,"target.csv"));final=readcsv(joinpath(output,"final.csv"))
 meta=TOML.parsefile(joinpath(output,"run.toml"))
 f=Figure(size=(1350,800)); ax=Axis(f[1,1:3],xlabel="Hours since 11:30 UTC",ylabel="LWP (g m⁻²)")
 vspan!(ax,[0],[1],color=(:gray,0.15))
 lines!(ax,(obs[:,1]+obs[:,2])/7200,obs[:,3],color=:black,label="ARM MWRRET");lines!(ax,run[:,1]/3600,run[:,2],linestyle=:dash,label="Breeze");axislegend(ax)
 if meta["complete_interval"]
  for (i,fc,tc,label,scale) in [(1,2,3,"θ error (K)",1),(2,3,4,"qᵥ error (g kg⁻¹)",1000),(3,4,5,"u error (m s⁻¹)",1)]
   ax=Axis(f[2,i],xlabel=label,ylabel="Height (km)");lines!(ax,(final[:,fc]-target[:,tc])*scale,final[:,1]/1000);vlines!(ax,[0],color=:gray);ylims!(ax,0,2.5)
  end
 else
  Label(f[2,:],"Short engineering test: final-sounding error is not evaluated.")
 end
 Label(f[0,:],meta["complete_interval"] ? "Forward evaluation • loss = $(round(meta["loss"],digits=3))" : "Short forward test (not a case result)")
 save(joinpath(output,"observation_error.png"),f)
end
