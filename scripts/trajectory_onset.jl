# Per-record onset analysis of a trajectory-mode observation_sensitivity.jls (signed and RMS divergence
# scaling across ε pairs, DWP/LWP odd parts, mask switches). ENV: ATLANTIC_RESULT.
using Serialization, Statistics, Printf
r = deserialize(ENV["ATLANTIC_RESULT"]);
T = r["trajectory"];
b = T[1]
runs = Dict(x.label => x for x in T[2:end])
eps = (0.01, 0.001, 0.0001)
lab(s, e) = "heating:" * (s > 0 ? "" : "-") * string(e)
nrec = length(b.lwp)
odd(f, e) = (f(runs[lab(1, e)]) .- f(runs[lab(-1, e)])) ./ 2e     # central difference series
even(f, e) = (f(runs[lab(1, e)]) .+ f(runs[lab(-1, e)]) .- 2f(b)) ./ 2   # second-order part
dw(x) = x.dwp_frozen;
lw(x) = x.lwp
rms(x, f) = [maximum(d[f]) for d in x.divergence]
@printf(
    "%4s %6s | %-29s | %-29s | %-29s | %-23s | %s\n",
    "rec",
    "min",
    "maxRMSΔρqʳ/ε (.01 .001 1e-4)",
    "dDWP/dε raw series",
    "dLWP/dε",
    "even DWP (.01 .001 1e-4)",
    "switches(+.01,-.01,+.001,-.001,+1e-4,-1e-4)"
)
for n in [1:2:30; 31:5:nrec]
    q = [rms(runs[lab(1, e)], "ρqʳ")[n] / e for e in eps]
    d = [odd(dw, e)[n] for e in eps]
    l = [odd(lw, e)[n] for e in eps]
    v = [even(dw, e)[n] for e in eps]
    s = [runs[lab(sg, e)].switches_by_record[n] for e in eps for sg in (1, -1)]
    @printf(
        "%4d %6.1f | %9.2e %9.2e %9.2e | %9.3g %9.3g %9.3g | %9.3g %9.3g %9.3g | %7.1e %7.1e %7.1e | %s\n",
        n,
        (n-1),
        q...,
        d...,
        l...,
        v...,
        s
    )
end
# Per-level onset of ρqʳ nonlinearity: first record where RMSΔ(1e-3)/RMSΔ(1e-4) < 5 at each level with signal.
r3 = runs[lab(1, 0.001)].divergence;
r4 = runs[lab(1, 0.0001)].divergence
nz = length(r3[1]["ρqʳ"])
onset = [
    something(
        findfirst(n -> r4[n]["ρqʳ"][k] > 0 && r3[n]["ρqʳ"][k] / r4[n]["ρqʳ"][k] < 5, 2:nrec),
        -1,
    ) + 1 for k = 1:nz
]
println("per-level first record with ratio(1e-3/1e-4)<5 for ρqʳ (levels 1:60): ", onset[1:60])
for f in ("θ", "w", "ρqᶜˡ")
    rr = [maximum(r3[n][f]) / max(maximum(r4[n][f]), 1e-300) for n = 2:nrec]
    println(
        f,
        " first record ratio<5: ",
        something(findfirst(<(5), rr), -1) + 1,
        "; ratios at 10,20,30,45,60: ",
        round.(rr[[9, 19, 29, 44, 59]]; sigdigits = 3),
    )
end
println("ONSET_DONE")
