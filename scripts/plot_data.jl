using CairoMakie, DelimitedFiles, TOML
input=length(ARGS)>0 ? ARGS[1] : "data/interval"
output=length(ARGS)>1 ? ARGS[2] : "outputs/data"
mkpath(output)
readcsv(p)=readdlm(joinpath(input,p),',',Float64;skipstart=1)
a=readcsv("initial.csv");b=readcsv("target.csv");forcing=readcsv("forcing.csv");lwp=readcsv("lwp.csv")
f=Figure(size=(1350,850))
for (col,label,scale) in [(3,"Dry potential temperature (K)",1),(4,"Vapor (g kg⁻¹)",1000),(5,"u wind (m s⁻¹)",1)]
 ax=Axis(f[1,col-2],xlabel=label,ylabel="Height (km)")
 lines!(ax,a[:,col]*scale,a[:,1]/1000,label="11:30 UTC");lines!(ax,b[:,col]*scale,b[:,1]/1000,label="23:31 UTC")
 ylims!(ax,0,3);col==3 && axislegend(ax)
end
times=unique(forcing[:,1]);z=a[:,1];nt=length(times)
for (slot,col,label,scale) in [(1,3,"ERA5 thermal advection (K day⁻¹)",86400),(2,4,"ERA5 moisture advection (g kg⁻¹ day⁻¹)",86400*1000)]
 ax=Axis(f[2,slot],xlabel="Hours since initial sounding",ylabel="Height (km)",title=label)
 h=heatmap!(ax,times/3600,z/1000,permutedims(reshape(forcing[:,col]*scale,length(z),nt)),colormap=:balance)
 ylims!(ax,0,3);Colorbar(f[2,slot,Right()],h)
end
ax=Axis(f[2,3],xlabel="Hours since initial sounding",ylabel="LWP (g m⁻²)",title="QC-screened MWRRET observations")
lines!(ax,(lwp[:,1]+lwp[:,2])/7200,lwp[:,3],color=:black)
Label(f[0,:],"March 4, 2016 • one sounding interval • ERA5 initial forcing guess",fontsize=22)
save(joinpath(output,"initial_guess.png"),f)
