# Analysis of an early-trajectory diagnosis (observation_sensitivity.jl with ATLANTIC_RUNS and
# ATLANTIC_RECORD_STEPS=1): for each tracked field and step, the maximum over levels of the per-level RMS
# divergence from the baseline, its level, and the ratio between runs with different ε. Under a linear
# (differentiable) response Δ ∝ ε, so the ratio Δ(ε₁)/Δ(ε₂) equals ε₁/ε₂.
# ENV: ATLANTIC_RESULTS (observation_sensitivity.jls).
using Serialization, Printf
r = deserialize(ENV["ATLANTIC_RESULTS"])
runs = filter(t -> t.label != "baseline", r["trajectory"])
fields = ("θ", "w", "ρqᶜˡ", "ρqʳ")
steps = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 450, 600)
for f in fields
    println("\n== ", f, ": max-over-levels RMS Δ [level of max] by step")
    println(
        rpad("step", 6),
        join([rpad(t.label, 30) for t in runs]),
        "ratio(1e-6/1e-9) ratio(1e-9/1e-12)",
    )
    for s in steps
        n = s + 1   # record 1 is the initial state
        n <= length(runs[1].divergence) || continue
        vals = [maximum(t.divergence[n][f]) for t in runs]
        levs = [argmax(t.divergence[n][f]) for t in runs]
        byε = Dict(t.ε => v for (t, v) in zip(runs, vals))
        ratio(a, b) = haskey(byε, a) && haskey(byε, b) && byε[b] > 0 ? byε[a] / byε[b] : NaN
        println(
            rpad(s, 6),
            join([rpad(@sprintf("%.3e [%d]", v, l), 30) for (v, l) in zip(vals, levs)]),
            @sprintf("%.3g  %.3g", ratio(1e-6, 1e-9), ratio(1e-9, 1e-12)),
        )
    end
end
# First step at which each run's divergence of each field is nonzero, and the levels where it is nonzero.
println("\n== first nonzero step per field and its nonzero levels")
for t in runs, f in fields
    n = findfirst(d -> maximum(d[f]) > 0, t.divergence)
    isnothing(n) && continue
    levels = findall(>(0), t.divergence[n][f])
    println(
        rpad(t.label, 16),
        rpad(f, 6),
        " step ",
        n - 1,
        " levels ",
        length(levels) > 12 ?
        string(first(levels, 6), "…", last(levels, 6), " (", length(levels), ")") : levels,
        " max ",
        @sprintf("%.3e", maximum(t.divergence[n][f])),
    )
end
# Per-level profile at step 1 and step 10 for the smallest and largest ε (where in the column it starts).
for t in (runs[1], runs[end]), s in (1, 2, 10)
    d = t.divergence[s+1]
    println(
        "\n",
        t.label,
        " step ",
        s,
        " levels with Δρqʳ>0: ",
        findall(>(0), d["ρqʳ"]),
        "  Δθ>0 levels: ",
        count(>(0), d["θ"]),
        " top Δθ level ",
        argmax(d["θ"]),
        @sprintf(" (%.3e)", maximum(d["θ"])),
    )
end
