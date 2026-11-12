# One entry point for derived inputs and Julia figures; raw observations are unchanged.
data=get(ENV,"ATLANTIC_DATA","data/march2016")
input=get(ENV,"ATLANTIC_INPUT","data/interval")
output=get(ENV,"ATLANTIC_OUTPUT","outputs/data")
python=get(ENV,"ATLANTIC_PYTHON","python")
run(`$python $(joinpath(@__DIR__,"prepare_data.py")) --data-dir $data --output $input`)
empty!(ARGS); append!(ARGS,[input,output])
include("plot_data.jl")
