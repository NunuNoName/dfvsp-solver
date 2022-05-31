# dfvsp-solver

This repository provides a heuristic solver for the Directed Feedback Vertex Set problem (DFVSP). The DFVSP works on directed graphs and its aim is to find a minimum subset of vertices such that removing this subset and all its adjacent edges from the graph results in an acyclic graph. The provided solver employs a hybrid metaheuristic based on a large neighborhood search (LNS) as general metaheuristic framework. The LNS repeatedly applies a pair of a destroy and a repair method which first change a part of a given solution and then fix this partial solution to recover a valid solution. The hybridization is achieved by incorporating a mixed integer programming model into the repair operator. A construction heuristic based on a greedy function is used to generate an initial solution. This initial solution is improved by applying a simple local search procedure to it and the resulting solution then serves as the starting point for the LNS procedure. 

## Requirements

- A 64-bit Linux operating system
- [Julia Programming Language](https://julialang.org/) v1.7.1 or higher
- Julia packages (cf. [Installation](/README.md#installation) section)
  - Graphs v1.5.1
  - MHLib v0.1.4
  - JuMP v0.22.2
  - SCIP v0.10.1
  - Random
  - StatsBase v0.33.15
  - bliss_jll v0.73

## Installation

Execute the install script for the Julia packages: 
```
julia install_packages.jl
```

## Run 

The program reads a DFVSP instance from `stdin` and prints the best found solution to `stdout`. For the input and output format, please refer to the [PACE 2022 challenge web site](https://pacechallenge.org/2022/tracks/#input-format).

A usage example of the solver would be:
```
julia  DFVSPSolverMain.jl < instance
```

Per default, the solver runs for 10 minutes and then terminates.
