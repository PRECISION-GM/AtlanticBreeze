# Run the March 4, 2016 sounding interval and compare it with observations.
# Input paths and optional engineering-test settings are supplied by the batch job.
include("model.jl")

input = get(ENV, "ATLANTIC_INPUT", "data/interval")
output = get(ENV, "ATLANTIC_OUTPUT", "outputs/forward")

# With no overrides, use the duration and horizontal grid in interval.toml.
stop_time = if haskey(ENV, "ATLANTIC_STOP_SECONDS")
    parse(Float64, ENV["ATLANTIC_STOP_SECONDS"])
else
    nothing
end

nx = if haskey(ENV, "ATLANTIC_NX")
    parse(Int, ENV["ATLANTIC_NX"])
else
    nothing
end

cpu = get(ENV, "ATLANTIC_CPU", "false") == "true"
forecast(input, output; stop_time, nx, cpu)
include("plot_error.jl")
plot_error(input, output)
