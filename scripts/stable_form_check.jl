# Correctness checks for the isolated TEST-ONLY stable growth_share (Breeze worktree stable-rain-share-test).
# Run with the env whose Breeze is that worktree. (1) φ(a) from Breeze.transient_deficit vs BigFloat-256 over
# a ∈ [1e-20, 1e3], in ulps; (2) the full species growth expression, old vs new form in Float64 vs a BigFloat
# reference, over random inputs in the cancellation regime; (3) P3 tendencies on a saved window state, written
# for comparison with the production env (p3 tendencies are also written when run with production Breeze).
# ENV: ATLANTIC_INPUT, ATLANTIC_STATE (window_state.jls), ATLANTIC_OUTPUT, ATLANTIC_LABEL.
include("model.jl")
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: update_state!
using Serialization, Printf, Random
const AM = Breeze.AtmosphereModels
const PPP = Breeze.Microphysics.PredictedParticleProperties
output = mkpath(ENV["ATLANTIC_OUTPUT"]); label = ENV["ATLANTIC_LABEL"]
setprecision(BigFloat, 256)
stable = isdefined(PPP, :transient_deficit)
println("label=", label, " stable_form_defined=", stable, " breeze=", pkgdir(Breeze))

if stable
    φexact(a::BigFloat) = 1 - (1 - exp(-a)) / a
    ulps(x, ref) = Float64(abs(BigFloat(x) - ref) / eps(Float64(ref)))
    worst = 0.0; worst_a = 0.0
    for a in exp10.(range(-20, 3; length=20001))
        e = ulps(PPP.transient_deficit(a), φexact(BigFloat(a)))
        e > worst && (worst = e; worst_a = a)
    end
    near = maximum(ulps(PPP.transient_deficit(a), φexact(BigFloat(a))) for a in (prevfloat(1.0), 1.0, nextfloat(1.0), 0.999, 1.001))
    println(@sprintf("phi: max error %.2f ulp at a=%.3e over [1e-20,1e3]; around a=1: %.2f ulp", worst, worst_a, near))

    old(ε, ξ, ext, x, vx, T) = (ext * (ε / x) + (vx - ext / x) * (ε / x) * T) / ξ
    Random.seed!(2)
    errs_old = Float64[]; errs_new = Float64[]
    for _ in 1:100000
        a = exp10(-12 + 10rand()); τ = exp10(rand() * 2); x = a / τ
        ε = x * rand(); ξ = 1 + rand(); ext = (rand() - 0.5) * exp10(-10 + 4rand()); vx = (rand() - 0.5) * exp10(-8 + 4rand())
        ref = (ext * BigFloat(ε) / x * (1 - (1 - exp(-BigFloat(x) * τ)) / (BigFloat(x) * τ)) +
               vx * (BigFloat(ε) / x) * (1 - exp(-BigFloat(x) * τ)) / τ) / ξ
        Told = (1 - exp(-x * τ)) / τ; Tnew = -expm1(-x * τ) / τ
        new = PPP.growth_share(ε, ξ, ext, x, vx, Tnew, PPP.transient_deficit(x * τ))
        push!(errs_old, Float64(abs(old(ε, ξ, ext, x, vx, Told) - ref) / abs(ref)))
        push!(errs_new, Float64(abs(new - ref) / abs(ref)))
    end
    q(v, p) = sort(v)[ceil(Int, p * length(v))]
    println(@sprintf("growth_share rel error vs BigFloat (a in 1e-12..1e-2): old median %.2e p99 %.2e max %.2e | new median %.2e p99 %.2e max %.2e",
                     q(errs_old, .5), q(errs_old, .99), maximum(errs_old), q(errs_new, .5), q(errs_new, .99), maximum(errs_new)))
end

input = ENV["ATLANTIC_INPUT"]; saved = deserialize(ENV["ATLANTIC_STATE"])
model = make_case(input; nx=saved.nx, dx=saved.dx, architecture=CPU()).model
for (n, f) in pairs(prognostic_fields(model)); set!(f, saved.fields[String(n)]); end
update_state!(model)
Gs = PPP.p3_tendency_fields(model.timestepper.Gⁿ, model.microphysics)
for g in Gs; fill!(parent(g), 0); end
AM.compute_microphysical_tendencies!(model.microphysics, model)
tend = Dict(String(n) => Array(interior(g)) for (n, g) in pairs(Gs))
serialize(joinpath(output, "p3_tendencies_$label.jls"), tend)
other = joinpath(output, "p3_tendencies_$(label == "stable" ? "production" : "stable").jls")
if isfile(other)
    o = deserialize(other)
    for n in sort(collect(keys(tend)))
        d = abs.(tend[n] .- o[n]); s = maximum(abs, o[n])
        println(@sprintf("tendency %-6s max|Δ|=%.3e max|G|=%.3e rel=%.3e cells_differing=%d/%d", n, maximum(d), s, maximum(d) / max(s, floatmin()), count(>(0), d), length(d)))
    end
end
println("STABLE_FORM_CHECK_DONE ", label)
