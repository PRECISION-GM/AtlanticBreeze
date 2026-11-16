# Native check of the input-array plumbing used by window_gradient.jl ATLANTIC_CONTROLS=inputs.
# For each forcing input f in (T, q, w, ug, vg): (a) the model's forcing terms alias make_case's input_series;
# (b) writing δ into the series data in place gives the same trajectory as rebuilding the model with the same
# δ applied to forcing.csv (forcing_perturbation). Small grid, from the interval IC, a few steps.
# ENV: ATLANTIC_INPUT, ATLANTIC_NX (8), ATLANTIC_DYNAMICS, ATLANTIC_CHECK_STEPS (20).
include("model.jl")
using Oceananigans: prognostic_fields

input = ENV["ATLANTIC_INPUT"]; nx = parse(Int, get(ENV, "ATLANTIC_NX", "8")); dx = grid_spacing(100.0)
nsteps = parse(Int, get(ENV, "ATLANTIC_CHECK_STEPS", "20")); dt = 1.0
columns = Dict("T" => 3, "q" => 4, "w" => 5, "ug" => 6, "vg" => 7)
scale = Dict("T" => 1 / 86400, "q" => 1e-3 / 86400, "w" => 1e-3, "ug" => 0.1, "vg" => 0.1)
forcing_term(m, name, i) = getfield(getfield(getproperty(m.forcing, name), 1)[i], 1)
input_series_of(m) = (; T=forcing_term(m, :ρE, 1).tls, q=forcing_term(m, :ρqᵛ, 1).qls, w=forcing_term(m, :ρqᵛ, 2).velocity,
                       ug=forcing_term(m, :ρv, 1).geostrophic_velocity, vg=forcing_term(m, :ρu, 1).geostrophic_velocity)
run!(m) = (for _ in 1:nsteps; time_step!(m, dt); end; Dict(String(n) => Array(interior(f)) for (n, f) in pairs(prognostic_fields(m))))

base = make_case(input; nx, dx, architecture=GPU())
own = input_series_of(base.model)
for f in (:T, :q, :w, :ug, :vg)
    ok = parent(getproperty(own, f).data) === parent(getproperty(base.input_series, f).data)
    println("ALIAS $f $(ok ? "OK" : "FAIL")")
end
z = base.z; times = base.forcing_times; nz = length(z)
prof = clamp.(1 .- z ./ 3000, 0, 1)
reference = run!(make_case(input; nx, dx, architecture=GPU()).model)
worst = 0.0
for (f, c) in columns
    δ = scale[f] .* prof                         # same δ at every knot
    rebuilt = run!(make_case(input; nx, dx, architecture=GPU(),
        forcing_perturbation=d -> (e = copy(d); for t in times; r = e[:, 1] .== t; e[r, c] .+= δ; end; e)).model)
    case = make_case(input; nx, dx, architecture=GPU())
    s = getproperty(input_series_of(case.model), Symbol(f)); P = parent(s.data)
    off = (size(P, 3) - nz) ÷ 2
    host = Array(P); host[1, 1, off+1:off+nz, :] .+= reshape(δ, nz, 1); copyto!(P, host)
    inplace = run!(case.model)
    diff = maximum(maximum(abs.(inplace[n] .- rebuilt[n])) / max(maximum(abs.(rebuilt[n])), eps()) for n in keys(rebuilt))
    effect = maximum(maximum(abs.(inplace[n] .- reference[n])) / max(maximum(abs.(reference[n])), eps()) for n in keys(rebuilt))
    global worst = max(worst, diff)
    println("INPLACE_VS_REBUILT $f rel_diff=$diff perturbation_effect=$effect")
end
println(worst == 0 ? "INPUT_PLUMBING_CHECK_PASS exact" : worst < 1e-12 ? "INPUT_PLUMBING_CHECK_PASS rel<1e-12 ($worst)" : "INPUT_PLUMBING_CHECK_FAIL $worst")
