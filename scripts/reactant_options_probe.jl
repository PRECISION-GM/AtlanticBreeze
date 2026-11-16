# Introspect the installed Reactant's compile pipeline options (no model): which CompileOptions fields exist,
# whether debug locations can be stripped, and which APIs allow reusing a traced module for compilation.
using Reactant
println("Reactant ", pkgversion(Reactant))
println("CompileOptions fields: ", fieldnames(Reactant.CompileOptions))
println("CompileOptions defaults: ", Reactant.CompileOptions())
for s in names(Reactant.Compiler; all=true)
    n = String(s); occursin(r"(?i)debug|loc|strip|cache|compile_mlir|compile_xla|raise|pass", n) && println("Compiler.", n)
end
for s in names(Reactant.MLIR.IR; all=true)
    n = String(s); occursin(r"(?i)debug|loc|dump|strip", n) && println("MLIR.IR.", n)
end
println("PERSISTENT/CACHE globals: ", [n for n in names(Reactant; all=true) if occursin(r"(?i)cache|persist", String(n))])
println("REACTANT_OPTIONS_PROBE_DONE")
