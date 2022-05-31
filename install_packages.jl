using Pkg

Pkg.add(name="Graphs", version="1.5.1")
Pkg.add(name="MHLib", version="0.1.4")
Pkg.add("Random")
Pkg.add(name="JuMP", version="0.22.2")
Pkg.add(name="SCIP", version="0.10.1")
Pkg.add(name="StatsBase", version="0.33.15")
Pkg.add(PackageSpec(name="bliss_jll", version="0.73"))  # necessary to get correct dependency for SCIP