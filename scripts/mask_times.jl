# Print the UTC span and count of the saved MA30 scoring masks (and per-record times) of an observation_sensitivity.jls.
using Serialization
r = deserialize(ENV["ATLANTIC_RESULT"]); b = r["baseline"]; m = r["masks"]; h = b["utc"]
for (k, v) in (("L", m.L), ("D", m.D))
    t = h[v]; println("mask ", k, ": count=", count(v), " of ", length(v), " utc min=", minimum(t), " max=", maximum(t),
                      " window_min=", round.(((minimum(t), maximum(t)) .- h[1]) .* 60; digits=2))
end
println("record utc first/last ", h[1], " ", h[end], " n=", length(h))
