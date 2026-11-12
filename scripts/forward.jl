include("model.jl")
input=get(ENV,"ATLANTIC_INPUT","data/interval")
output=get(ENV,"ATLANTIC_OUTPUT","outputs/forward")
stop_time=haskey(ENV,"ATLANTIC_STOP_SECONDS") ? parse(Float64,ENV["ATLANTIC_STOP_SECONDS"]) : nothing
nx=parse(Int,get(ENV,"ATLANTIC_NX","128"))
forecast(input,output;stop_time,nx,cpu=get(ENV,"ATLANTIC_CPU","false")=="true")
include("plot_error.jl")
plot_error(input,output)
