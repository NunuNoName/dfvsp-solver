#=
Main:
- Julia version: 1.7.1
- Author: Maria Bresich
=#

using Graphs
using MHLib
using MHLib.SubsetVectorSolutions
using MHLib.Schedulers
using MHLib.ALNSs
using Random
using JuMP
using SCIP
using StatsBase

# include("MyGraphs.jl")
# include("DFVSP.jl")
# include("MyUtils.jl")
# include("MyGraphUtils.jl")
# include("ReductionRules.jl")
# include("ConstructionHeuristics.jl")
# include("MyMILPModels.jl")
# include("MyLNS.jl")
# include("MyLS.jl")

# using .DFVSP
# using .MyUtils
# using .MyGraphUtils
# using .ReductionRules
# using .ConstructionHeuristics
# using .MyMILPModels
# using .MyLNS
# using .MyLS

import JuMP.MathOptInterface
const MOI = JuMP.MathOptInterface

global_start_time = -1
global_run_time_limit = -1  # <0 for no run time limit
first_time_limit_mip_solver = 0
second_time_limit_mip_solver = 0

global_orig_stdout = stdout
# this variable holds the global best solution
# each vector is either the solution from reduction rules or the currently best solution of an SCC
# has to be updated after each call of RRs, each CH, each LS, and each LNS
global_best_solution = Vector{Vector{Int}}()

# variable to indicate normal program termination 
# this is needed for the atexit function
global_normal_program_termination = false


"""
    create_graph_from_file(filename::AbstractString)
Read a simple directed unweighted graph from the specified file.
Metis file format:
- ``% <comments>    #`` ignored
- ``<number of vertices> <number of edges> <weight identifier>``
- ``<vertex_1> <vertex_2> ...   #`` for each vertex with outgoing edges, vertices are labeled in 1...number of vertices
"""
function create_graph_from_file(filename::AbstractString) :: SimpleDiGraph{Int}
    #= SimpleDiGraph: Multiple edges between two given vertices are not allowed: an attempt to add
    an edge that already exists in a graph will not raise an error. This event can be detected
    using the return value of add_edge!.
    =#
    graph = SimpleDiGraph()
    found_description = false
    vertex_number = 0
    n = 0
    m = 0
    t = 0

    input = stdin

    # check whether a file was specified 
    # if yes, read from file
    # if not, then read from stdin 
    if !isempty(filename)
        input = filename
    end

    # TODO: make sure "eachline" keeps empty lines
    for line in eachline(input)
        # necessary to check for empty lines before accessing line[1]? -> yes
        linestart = ""

        if (!isempty(line))
            linestart = line[1]
        end

        if (linestart == '%')     # ignore comments
            continue
        elseif(found_description == false)  # find first non-comment line
            if (isempty(line))
                error("line with graph description is empty")
            end
            found_description = true
            line_parts = split(line)

            if (length(line_parts) != 3)
                error("line with graph description contains wrong number of elements")
            end

            n = parse(Int, line_parts[1])
            m = parse(Int, line_parts[2])
            t = parse(Int, line_parts[3])

            # debugging
            # @info "Vertex number in input file: $n"
            # @info "Edge number in input file: $m"
            # @info "Weight identifier in input file: $t"
            # println("Vertex number in input file: $n")
            # println("Edge number in input file: $m")
            # println("Weight identifier in input file: $t")

            graph = SimpleDiGraph(n)
        else    # lines with adjacency lists
            vertex_number += 1  # keep track of current vertex
            if (isempty(line))
                # skip empty lines = nodes with no outgoing edges
                continue
            end
            split_line = split(line)    # get all successor vertices
            # update adjacency list
            for vertex_string in split_line
                vertex = parse(Int, vertex_string)
                @assert add_edge!(graph, vertex_number, vertex)
            end

        end # if-statement
    end # for-loop

     @assert nv(graph) > 0
     @assert nv(graph) == n
     @assert ne(graph) == m

     # debugging - start

     # @assert is_directed(graph)
     # @assert !has_self_loops(graph)
     @debug "Vertex number of created graph: $(nv(graph))"
     # println("Vertex number of created graph: ", nv(graph))
     @debug "Edge number of created graph: $(ne(graph))"
     # println("Edge number of created graph: ", ne(graph))
     # test_edge = has_edge(graph, 8, 270)    # specific for e_001
     # println("Found specific edge from 8 to 270 in graph: $test_edge")

     directed_graph = is_directed(graph)
     self_loops = has_self_loops(graph)
     @debug "Graph is directed: $directed_graph"
     @debug "Graph has self loops: $self_loops"
     # println("Graph is directed: $directed_graph")
     # println("Graph has self loops: $self_loops")

     # debugging - end

     return graph
end # function create_graph_from_file



"""
    DFVSPInstance
Directed feedback vertex set problem (DFVSP) instance.
Given a directed (unweighted) graph, find a minimum cardinality subset of vertices 
such that the removal of these vertices makes the remaining graph acyclic.

Attributes
- `graph`: directed unweighted graph
- `n`: number of nodes
- `m` number of edges
- `t`: weight identifier (always 0)
- `all_nodes`: set of all nodes
"""
struct DFVSPInstance
    graph::SimpleDiGraph{Int}
    n::Int
    m::Int
    t::Int
    all_nodes::Set{Int}
end


"""
    DFVSPInstance(filename)
Read a simple directed unweighted graph from the specified file.
"""
function DFVSPInstance(filename::AbstractString)
    graph = create_graph_from_file(filename)
    n = nv(graph)
    m = ne(graph)
    t = 0
    DFVSPInstance(graph, n, m, t, Set(1:n))
end

"""
    DFVSPInstance(g)
Create an DFVSP instance based on the given graph.
"""
function DFVSPInstance(g::SimpleDiGraph)
    graph = g
    n = nv(graph)
    m = ne(graph)
    t = 0
    DFVSPInstance(graph, n, m, t, Set(1:n))
end

# TODO: keep function?
function Base.show(io::IO, inst::DFVSPInstance)
    println(io, "n=$(inst.n), m=$(inst.m)")
end


"""
    DFVSPSolution
Solution to a DFVSP instance.
It is a `SubsetVectorSolution` in which we do not store unselected elements in the
solution vector behind the selected ones, but instead use the set `all_elements`.
Attributes in addition to those needed by `SubsetVectorSolution`:
- `dfvs_rr`: The preliminary DFVS resulting from the application of the reduction rules.
- `reverse_vmaps_list`: A list of vertex mappings to find the original vertex labels after the application of reduction rules.
- `vmap_orig_new`: A vertex mapping from the original vertex labels to the new (current) labels.
- `vmap_new_orig`: A vertex mapping from the new (current) vertex labels to the original labels.
"""
mutable struct DFVSPSolution <: SubsetVectorSolution{Int}
    inst::DFVSPInstance
    obj_val::Int
    obj_val_valid::Bool
    x::Vector{Int}    # DFVS from metaheuristic
    sel::Int
    dfvs_rr::Vector{Int}    # DFVS from reduction rules
    reverse_vmaps_list::Vector{Vector{Integer}} # list of vertex mappings
    # vmap_orig_new::Vector{Integer}  # vertex mapping original -> new
    vmap_orig_new::Dict{Int, Integer}
    vmap_new_orig::Vector{Integer}  # vertex mapping new -> original 
end

# TODO: correct?
# maybe use Vector[] or Vector{Vector{Integer}}() instead of Vector{Vector{Integer}}[]?
# for mappings: current label of every vertex is the vertex itself
DFVSPSolution(inst::DFVSPInstance) =
    DFVSPSolution(inst, -1, false, collect(1:inst.n), 0, Vector{Int}(), Vector{Vector{Integer}}(), Dict{Int, Integer}(v=>v for v in 1:inst.n), collect(1:inst.n))

unselected_elems_in_x(::DFVSPSolution) = false

MHLib.SubsetVectorSolutions.all_elements(s::DFVSPSolution) = s.inst.all_nodes

"""
    to_maximize(::DFVSPSolution)

Return false because the optimization goal of DFVSP is to minimize the objective function.
"""
MHLib.to_maximize(s::DFVSPSolution) = false

# TODO: keep function?
# TODO: correct copy of reverse_vmaps_list (= vector of vectors)?
function Base.copy!(s1::DFVSPSolution, s2::DFVSPSolution)
    s1.inst = s2.inst
    s1.obj_val = s2.obj_val
    s1.obj_val_valid = s2.obj_val_valid
    # s1.x[:] = s2.x  # leads to DimensionMismatch when solution vectors have different lengths
    s1.x = copy(s2.x)     # TODO: is this correct?
    s1.sel = s2.sel
    s1.dfvs_rr[:] = s2.dfvs_rr
    s1.reverse_vmaps_list[:] = s2.reverse_vmaps_list
    s1.vmap_orig_new = copy(s2.vmap_orig_new)
    s1.vmap_new_orig[:] = s2.vmap_new_orig
end

# TODO: keep function?
# TODO: correct copy of reverse_vmaps_list (= vector of vectors)?
Base.copy(s::DFVSPSolution) =
    DFVSPSolution(s.inst, s.obj_val, s.obj_val_valid, Base.copy(s.x[:]), s.sel,
        Base.copy(s.dfvs_rr[:]), Base.deepcopy(s.reverse_vmaps_list[:]), 
        Base.copy(s.vmap_orig_new), Base.copy(s.vmap_new_orig[:]))

# TODO: keep function?
Base.show(io::IO, s::DFVSPSolution) =
    println(io, s.x)
   
# TODO: correct?
MHLib.calc_objective(s::DFVSPSolution) =
    # length(s.x) + length(s.dfvs_rr) 
    s.sel > 0 ? length(s.x) + length(s.dfvs_rr) : 0
    # s.sel > 0 ? sum(s.inst.p[s.x[1:s.sel]]) : 0


function sort_sel!(s::DFVSPSolution)
    invoke(MHLib.SubsetVectorSolutions.sort_sel!, Tuple{SubsetVectorSolution}, s)
end
    

function clear!(s::DFVSPSolution)
    # set sel to 0 and invalidate s
    invoke(MHLib.SubsetVectorSolutions.clear!, Tuple{SubsetVectorSolution}, s)
end

# only clear the solution from the metaheuristic
function clear_dfvs!(s::DFVSPSolution)
    s.x = Vector{Int}()
    # set sel to 0 and invalidate s
    invoke(MHLib.SubsetVectorSolutions.clear!, Tuple{SubsetVectorSolution}, s)
end

# clear the solution from the reduction rules and the solution from the metaheuristic and all vertex mappings
# TODO: how to correctly clear a vector and a vector of vectors?
function clear_all!(s::DFVSPSolution)
    s.dfvs_rr = Vector{Int}()
    # or?: empty!(s.dfvs_rr)
    s.reverse_vmaps_list = Vector{Vector{Integer}}()
    # or?: s.reverse_vmaps_list = Vector{Vector{Integer}}[]
    # or?: s.reverse_vmaps_list = Vector[]
    # or?: empty!(s.reverse_vmaps_list)
    s.vmap_orig_new = Dict{Int, Integer}()
    s.vmap_new_orig = Vector{Integer}()
    invoke(MHLib.SubsetVectorSolutions.clear!, Tuple{SubsetVectorSolution}, s)
end

# TODO: implement correct check for DAG!
function MHLib.check(s::DFVSPSolution, unsorted::Bool=true)
    invoke(check, Tuple{SubsetVectorSolution, Bool}, s, unsorted)

    g_copy = DiGraph(s.inst.graph)
    dfvs_orig_labels = copy(s.x)
    append!(dfvs_orig_labels, s.dfvs_rr)
    unique!(dfvs_orig_labels)

    # solution contains the original labels of the vertices
    # but the graph might have changed, so the labels need to be mapped
    dfvs_new_labels = Vector{Int}()
    if !isempty(s.vmap_orig_new)   # there is a mapping available
        for orig_label in dfvs_orig_labels
            new_label = s.vmap_orig_new[orig_label]
            # new label might be 0 if the vertex is not contained in the graph anymore
            if new_label != 0
                push!(dfvs_new_labels, new_label)
            end
        end

    else    # no mapping available
        dfvs_new_labels = dfvs_orig_labels
    end

    rem_vertices!(g_copy, dfvs_new_labels)

    if (is_cyclic(g_copy))
        error("Invalid solution - solution set is not a valid DFVS according to is_cyclic.")
    end

    if (length(strongly_connected_components(g_copy)) != nv(g_copy))
        error("Invalid solution - solution set is not a valid DFVS according to SCCs.")
    end

    #= 
    selected = Set(s.x[1:s.sel])
    for e in edges(s.inst.graph)
        if src(e) in selected && dst(e) in selected
            error("Invalid solution - adjacent nodes selected: $(src(e)), $(src(v))")
        end
    end
    new_covered = zeros(Int, s.inst.n)
    for u in s.x[1:s.sel]
        new_covered[u] += 1
        for v in neighbors(s.inst.graph, u)
            new_covered[v] += 1
        end
    end
    if s.covered != new_covered
        error("Invalid covered values in solution: $(self.covered)")
    end
     =#
end

# TODO: keep function? 
# TODO: removing 1 element could lead to an invalid solution, but how to check this?
function MHLib.SubsetVectorSolutions.element_removed_delta_eval!(s::DFVSPSolution; 
    update_obj_val::Bool=true, allow_infeasible::Bool=false)
    if allow_infeasible # TODO: add condition under which removing the element is allowed -> leads to a valid solution (resulting graph is still DAG)
        # accept the move
        if update_obj_val
            s.obj_val -= 1
        end
        return true # TODO: return whether new solution is valid or not
    end
    # revert the move
    s.sel += 1
    return false
end

# TODO: keep function?
# adding an element to the DFVS is always allowed -> solution is still valid
function MHLib.SubsetVectorSolutions.element_added_delta_eval!(s::DFVSPSolution; 
    update_obj_val::Bool=true, allow_infeasible::Bool=false)
    if update_obj_val
        s.obj_val += 1
    end
end



"""
    add_vertices_to_solution!(solution::DFVSPSolution, vs::AbstractVector{<: Integer}) where {T <: Integer}
Add all given vertices to x in solution after finding the original vertex label using reverse_vmaps_list.
Input: DFVSPSolution, list with vertices to add
Steps:
    - find original vertex label
    - add original vertex label to x 
"""
function add_vertices_to_solution!(solution::DFVSPSolution, vs::AbstractVector{<: Integer}
    ) where {T <: Integer}  

    # Sort the vertices to be added and filter out duplicate values
    vertices_to_add = sort(vs)
    unique!(vertices_to_add)

    # println("Adding vertices to the FVS: $vertices_to_add")

    for vertex in vertices_to_add
        original_label = find_original_vertex_label(solution, vertex)
        push!(solution.x, original_label)
        # update the number of selected elements
        solution.sel += 1
    end

end # function add_vertices_to_solution!



"""
    add_vertices_to_solution_new!(solution::DFVSPSolution, vs::AbstractVector{<: Integer}) where {T <: Integer}
Add all given vertices to x in solution after finding the original vertex label using vmap_new_orig.
Input: DFVSPSolution, list with vertices to add
Steps:
    - find original vertex label
    - add original vertex label to x 
"""
function add_vertices_to_solution_new!(solution::DFVSPSolution, vs::AbstractVector{<: Integer}
    ) where {T <: Integer}  

    # Sort the vertices to be added and filter out duplicate values
    vertices_to_add = sort(vs)
    unique!(vertices_to_add)

    # println("Adding vertices to the FVS: $vertices_to_add")

    for vertex in vertices_to_add
        original_label = solution.vmap_new_orig[vertex]
        push!(solution.x, original_label)
        # update the number of selected elements
        solution.sel += 1
    end

end # function add_vertices_to_solution_new!


"""
    add_vertices_to_dfvs_rr!(solution::DFVSPSolution, vs::AbstractVector{<: Integer}) where {T <: Integer}
Add all given vertices to dfvs_rr in solution after finding the original vertex label using reverse_vmaps_list.
Input: DFVSPSolution, list with vertices to add
Steps:
    - find original vertex label
    - add original vertex label to dfvs_rr 
"""
function add_vertices_to_dfvs_rr!(solution::DFVSPSolution, vs::AbstractVector{<: Integer}
    ) where {T <: Integer}  

    # Sort the vertices to be added and filter out duplicate values
    vertices_to_add = sort(vs)
    unique!(vertices_to_add)

    # println("Adding vertices to the preliminary FVS: $vertices_to_add")

    for vertex in vertices_to_add
        original_label = find_original_vertex_label(solution, vertex)
        push!(solution.dfvs_rr, original_label)
    end

end # function add_vertices_to_dfvs_rr!



"""
    add_vertices_to_dfvs_rr_new!(solution::DFVSPSolution, vs::AbstractVector{<: Integer}) where {T <: Integer}
Add all given vertices to dfvs_rr in solution after finding the original vertex label using vmap_new_orig.
Input: DFVSPSolution, list with vertices to add
Steps:
    - find original vertex label
    - add original vertex label to dfvs_rr 
"""
function add_vertices_to_dfvs_rr_new!(solution::DFVSPSolution, vs::AbstractVector{<: Integer}
    ) where {T <: Integer}  

    # Sort the vertices to be added and filter out duplicate values
    vertices_to_add = sort(vs)
    unique!(vertices_to_add)

    # println("Adding vertices to the preliminary FVS: $vertices_to_add")

    for vertex in vertices_to_add
        original_label = solution.vmap_new_orig[vertex]
        push!(solution.dfvs_rr, original_label)
    end

end # function add_vertices_to_dfvs_rr_new!


"""
    find_original_vertex_label(solution::DFVSPSolution, vertex::T) where {T <: Integer}
Go through the mappings in reverse_vmaps_list and find the original vertex label of the given vertex.
Input: DFVSPSolution, vertex
Returns: original vertex label
"""
function find_original_vertex_label(solution::DFVSPSolution, vertex::T) where {T <: Integer}
    # original_label::Int

    # debugging - start
    start_value = vertex
    # debugging - end

    for vmap in Iterators.reverse(solution.reverse_vmaps_list)
        vertex = vmap[vertex]
    end

    # debugging - start
    # println("Vertex $start_value used to be $vertex.")
    # debugging - end

    # return original_label
    return vertex
end



function set_global_start_time()
    global global_start_time = time()
end

function set_global_start_time(start_time::Float64)
    global global_start_time = start_time
end

# function get_start_time()
#     return global start_time
# end

function set_global_run_time_limit(run_time_limit::Int)
    global global_run_time_limit = run_time_limit
end

function set_first_time_limit_mip_solver(time_limit::Float64)
    global first_time_limit_mip_solver = time_limit
end

function set_second_time_limit_mip_solver(time_limit::Float64)
    global second_time_limit_mip_solver = time_limit
end

function set_global_normal_program_termination(normal_termination::Bool)
    global global_normal_program_termination = normal_termination
end

"""
    print_gobal_best_solution()

Used as exit hook -> called by atexit().
Checks whether the program terminated normally 
and if not, it prints the best solution to `stdout`.
"""
function print_gobal_best_solution()
    if !global_normal_program_termination
        # reset stdout
        redirect_stdout(global_orig_stdout)
        # println("interrupt caught")
        # print all parts of the solution
        # counter = 0
        for solution_part in global_best_solution
            # print solution to stdout
            for vertex in solution_part
                println(vertex)
                # counter += 1
            end
        end
        # println("Counter = $counter")
    end
end


"""
    update_global_best_solution_add_part(solution_part::Vector{Int})

Add the given solution part at the end of `global_best_solution`.
Should be called after the application of reduction rules and after the construction heuristic.
"""
function update_global_best_solution_add_part(solution_part::Vector{Int})
    push!(global_best_solution, solution_part)
end


"""
    update_global_best_solution_current_part(solution_part::Vector{Int})

Update the current part of the global best solution with the given solution part but only 
if the given solution part is better (smaller).
The current part is always the last vector.
The last vector in `global_best_solution` is replaced with `solution_part`.
"""
function update_global_best_solution_current_part(solution_part::Vector{Int})
    current_part_index = length(global_best_solution)
    # check if the given solution is better/smaller
    if (length(solution_part) < length(global_best_solution[current_part_index]))
        global_best_solution[current_part_index] = copy(solution_part)
    end
    
end



"""
    has_path_multi_source_multi_dest(g::AbstractGraph, srcs, dests; exclude_vertices=Vector())

Based on the method `has_path(g::AbstractGraph, u, v; exclude_vertices=Vector())` from the Graphs package.
Return `true` if there is a path from any node `u` in `srcs` to any node `v` in `dests` in `g` (while avoiding vertices in
`exclude_vertices`) or `u == v` for any `u` in srcs and any `v` in `dests`. Return false if there is no such path or if `u` or `v`
is in `excluded_vertices`. 
"""
function has_path_multi_source_multi_dest(g::AbstractGraph{T}, srcs::Vector{Int}, dests::Vector{Int}; 
        exclude_vertices::AbstractVector = Vector{T}()) where T
    # TODO more efficient this way or to use sets and set operations?
    seen = zeros(Bool, nv(g))
    for ve in exclude_vertices # mark excluded vertices as seen
        seen[ve] = true
    end
    # TODO are the 2 following checks correct?
    (any([seen[s] for s in srcs]) || any([seen[d] for d in dests])) && return false
    # any(srcs .== dests) && return true # cannot be separated -> WRONG: dimension mismatch when using dot operator
    # version below is supposed to be more efficient than in.(srcs, Ref(dests))
    # in.(srcs, Ref(dests))
    !isempty(findall(in(dests), srcs)) && return true # at least one source is also a destination => path found
    next = Vector{T}()
    append!(next, srcs)
    for s in srcs
        seen[s] = true
    end
    while !isempty(next)
        current_src = popfirst!(next) # get new element from queue
        for vertex in outneighbors(g, current_src)
            # TODO correct?
            # any(vertex .== dests) && return true
            # TODO faster alternative?:
            in(vertex, dests) && return true
            if !seen[vertex]
                push!(next, vertex) # push onto queue
                seen[vertex] = true
            end
            # TODO also check for equality with srcs and remove src from next?
            # might take longer than just leaving it in and checking it
        end
    end
    return false
end # function has_path_multi_source_multi_dest



"""
    apply_reduction_rules_new!(solution::DFVSPSolution)
Apply reduction rules to the graph given in the instance in DFVSPSolution to decrease its size.
Use the reduction rules based on the vertex mappings in solution.

Returns: an array of arrays, each of which is an entire strongly connected component of the reduced graph
"""
function apply_reduction_rules_new!(solution::DFVSPSolution)

    # debugging - start
    # println("Length of reverse_vmaps_list before reduction rules: ", length(solution.reverse_vmaps_list))
    # debugging - end  

    degree_zero_applicable = true
    indegree_one_applicable = true
    outdegree_one_applicable = true
    self_loop_applicable = true
    scc_applicable = true

    sccs = Array{Array{Int, 1}, 1}()

    #= TODO: possible alternative approach
        - use an outer while-loop over scc_applicable and inside the while-loop for the other rules
        - => let the other rules take care of iteratively reducing the graph, call scc-algorithm less often
        - QUESTION: is it more efficient to call the other rules (especially the merging rules) more often, or rather the scc-algorithm more often?
    =#

    # iterate over reduction rules until the graph cannot be reduced any further
    while (degree_zero_applicable || indegree_one_applicable || outdegree_one_applicable || self_loop_applicable || scc_applicable)

        degree_zero_applicable = remove_indegree_outdegree_zero_new!(solution)
        # merge_indegree_one!(solution)
        # merge_outdegree_one!(solution)

        # merge_indegree_one_alternative1!(solution)
        # merge_outdegree_one_alternative1!(solution)

        # indegree_one_applicable = merge_indegree_one_alternative2_new!(solution)
        # outdegree_one_applicable = merge_outdegree_one_alternative2_new!(solution)

        indegree_one_applicable = merge_indegree_one_alternative3_new!(solution)
        outdegree_one_applicable = merge_outdegree_one_alternative3_new!(solution)

        # test remove_self_loops!
        # vertex = first(vertices(solution.inst.graph))
        # add_edge!(solution.inst.graph, vertex, vertex)

        # call rule for self-loops after the other rules as the original graph does not contain self-loops
        self_loop_applicable = remove_self_loops_new!(solution)

        # call rule for SCCs -> returns list of SCCs as well as boolean
        sccs, scc_applicable = reduce_graph_using_strongly_connected_components_new!(solution)
    end

    # debugging - start

    # println("Length of reverse_vmaps_list after reduction rules: ", length(solution.reverse_vmaps_list))

    # debugging - end

    return sccs

end # function apply_reduction_rules_new!


"""
 remove_indegree_outdegree_zero_new!(solution::DFVSPSolution)
Remove all vertices with in-degree = 0 or out-degree = 0 and all their incident edges from the graph given in the instance inside solution.
Update mappings of vertex labels in solution.

Returns: true if rule has been applied, false otherwise
"""
function remove_indegree_outdegree_zero_new!(solution::DFVSPSolution)

    rule_applicable = false

    graph = solution.inst.graph
    outgoing_edges = graph.fadjlist
    incoming_edges = graph.badjlist

    # collect all vertices with in-degree = 0 and outdegree = 0 in a vector
    indegree_zero = [v for v in vertices(graph) if isempty(incoming_edges[v])]
    outdegree_zero = [v for v in vertices(graph) if isempty(outgoing_edges[v])]
    # concatenate the vectors to give all vertices at once to rem_vertices!
    removable_vertices = copy(indegree_zero)
    # rem_vertices! later takes care of possible duplicates that might be introduced here
    append!(removable_vertices, outdegree_zero)

    # debugging - start

    original_vertex_number = nv(graph)
    original_edge_number = ne(graph)

    @debug "Number of vertices with indegree = 0: $(length(indegree_zero))"
    @debug "Number of vertices with outdegree = 0: $(length(outdegree_zero))"
    # println("Vertices with indegree = 0: ", length(indegree_zero))
    # println("Vertices with outdegree = 0: ", length(outdegree_zero))

#= 
    for x in indegree_zero
      println("Vertex to remove: $x")
    end
 =#

    #     for x in outdegree_zero
    #         println("Vertex to remove: $x")
    #     end

    # if (!isempty(indegree_zero))
    #     to_remove = indegree_zero[1]
    #     println("Before remove, indegree = 0: Graph contains vertex $to_remove: ", has_vertex(graph, to_remove))
    # end
    # if (!isempty(outdegree_zero))
    #     to_remove = outdegree_zero[1]
    #     println("Before remove, outdegree = 0: Graph contains vertex $to_remove: ", has_vertex(graph, to_remove))
    # end

    # test_edge = has_edge(graph, 496, 450)    # specific for instance e_001
    # println("Before remove: Found specific edge from 496 to 450 in graph: $test_edge")

    # debugging end

    # remove all these vertices and their incident edges with rem_vertices!(graph, vector)
    # update the vertex mappings in the solution
    # in the end, these mappings are needed to determine the original vertex label of the vertices in the feedback set
    if (!isempty(removable_vertices))
        rule_applicable = true

        # set new label of these vertices to 0 because they will be removed
        for vertex in removable_vertices
            original_label = solution.vmap_new_orig[vertex]
            solution.vmap_orig_new[original_label] = 0
        end

        reverse_vmap = rem_vertices!(graph, removable_vertices)

        # update vertex mappings

        # create temporary mapping of correct new length
        # the old mapping in solution is needed for the update and cannot be changed directly
        # temp_vmap_new_orig = Vector{Integer}(length(reverse_vmap))
        temp_vmap_new_orig = zeros(Integer, length(reverse_vmap))
        for i in 1:length(temp_vmap_new_orig)
            # update original label
            original_label = solution.vmap_new_orig[reverse_vmap[i]]
            temp_vmap_new_orig[i] = original_label
            # update new label
            solution.vmap_orig_new[original_label] = i
        end
        # store new mapping in solution
        solution.vmap_new_orig = copy(temp_vmap_new_orig)

        # store mapping in list of mappings
        # push!(solution.reverse_vmaps_list, reverse_vmap)
    end

    # debugging - start

    # println("Length of reverse_vmaps_list after removing in-/out-degree = 0: ", length(solution.reverse_vmaps_list))

    # has_vertex() still returns true although number of vertices and edges correctly decreases
    # using rem_vertices!(graph, removable_vertices, keep_order = true) does not change the result of the test with has_vertex
    # probably due to the fact that the implementation of rem_vertices! replaces the vertices to remove with the ones at the end
    # so there is a new vertex in the place of the old one with the same index
    # if (!isempty(indegree_zero))
    #     removed = indegree_zero[1]
    #     println("After remove, indegree = 0: Graph contains vertex $removed: ", has_vertex(graph, removed))
    # end
    # if (!isempty(outdegree_zero))
    #     removed = outdegree_zero[1]
    #     println("After remove, outdegree = 0: Graph contains vertex $removed: ", has_vertex(graph, removed))
    # end

    # test_edge = has_edge(graph, 496, 450)    # specific for e_001
    # println("After remove: Found specific edge from 496 to 450 in graph: $test_edge")

    @debug "Expected number of vertices after removal: $(original_vertex_number - length(unique!(removable_vertices)))"
    # println("Expected number of vertices after removal: ", original_vertex_number - length(unique!(removable_vertices)))

    @debug "Vertex number of modified graph: $(nv(graph))"
    @debug "Edge number of modified graph: $(ne(graph))"
    # println("Vertex number of modified graph: ", nv(graph))
    # println("Edge number of modified graph: ", ne(graph))   # could also use "graph.ne"

    # check how vertices are reordered after removal
    #     for i in 1:length(reverse_vmap)
    #         println("First reduction: Vertex $i used to be vertex ", reverse_vmap[i])
    #     end

    # debugging - end

    return rule_applicable

end # function remove_indegree_outdegree_zero_new!


"""
    merge_indegree_one_alternative3_new!(solution::DFVSPSolution)
Merge all vertices with in-degree = 1 with their predecessor.
Update mappings of vertex labels in solution.
Implementation: get all vertices with in-degree = 1 once in the beginning and remove vertices after each merge 
    => next vertex to be merged gets new label and has to be mapped to this new label before merging 
    -> this is done by using the mappings in solution
    -> could be inefficient
Steps:
    - find vertex with in-degree = 1
    - get list of adjacent vertices
    - add edges to adjacent vertices to the predecessor vertex
    - remove vertex

Returns: true if rule has been applied, false otherwise
"""
function merge_indegree_one_alternative3_new!(solution::DFVSPSolution)
    rule_applicable = false
    
    # get all vertices with in-degree = 1 once in the beginning and remove vertices after each merge 
    # => next vertex to be merged gets new label and has to be mapped to this new label before merging -> could be inefficient
    graph = solution.inst.graph
    incoming_edges = graph.badjlist
    outgoing_edges = graph.fadjlist

    # incoming_edges = Base.deepcopy(graph.badjlist)
    # outgoing_edges = Base.deepcopy(graph.fadjlist)

    # collect all vertices with in-degree = 1 in a vector
    indegree_one = [v for v in vertices(graph) if length(incoming_edges[v]) == 1]

    vertices_with_self_loops = Vector{Int}()
    # merging_reverse_vmaps = Vector{Vector{Integer}}()
    # removed_vertices = Set{Int}()

    n = nv(solution.inst.graph)
    # current label of every vertex is the vertex itself
    # merging_vmap = collect(1:n)


    # debugging - start

    # println("Incoming edges: $incoming_edges")
    # println("Outgoing edges: $outgoing_edges")

    original_vertex_number = nv(graph)
    original_edge_number = ne(graph) 
    
    @debug "Number of vertices with indegree = 1: $(length(indegree_one))"
    # println("Vertices with indegree = 1: ", length(indegree_one))   

    #= 
    for x in indegree_one
        # println("Vertex to merge: $x")
    end 

    if (!isempty(indegree_one))
        to_merge = indegree_one[1]
        println("Before merge, indegree = 1: Graph contains vertex $to_merge: ", has_vertex(graph, to_merge))
    end 
     =#

    # debugging - end   

    # store vertex mapping in solution
    # in the end, these mappings are needed to determine the original vertex label of the vertices in the feedback set
    if (!isempty(indegree_one))
        rule_applicable = true

        # Sort the vertices to be merged -> TODO: necessary?
        vertices_to_merge = sort(indegree_one)

        # build list of original labels of the vertices to be merged
        vertices_to_merge_orig_labels = Vector{Int}()
        for vertex in vertices_to_merge
            orig_label = solution.vmap_new_orig[vertex]
            push!(vertices_to_merge_orig_labels, orig_label)
        end

        # merge_vertices! from Graphs merges the vertices in the given list to the vertex with the lowest number
        # BUT: merge_vertices! Supports [`SimpleGraph`](@ref) only.
        # => own implementation necessary!
        # Traverse the list of vertices and add all their outgoing edges to their predecessor
        for vertex_orig in vertices_to_merge_orig_labels

            remove_predecessor = false

            # check if vertex has already been removed in another iteration
            #= 
            if in(vertex, removed_vertices)
                continue
            end
            =#

            # new_vertex_label = find_new_vertex_label(merging_reverse_vmaps, vertex)
            # new_vertex_label = merging_vmap[vertex]

            new_vertex_label = solution.vmap_orig_new[vertex_orig]

            # check if vertex has already been removed in another iteration
            if (new_vertex_label == 0)
                continue
            end
             

            # check if predecessor has already been removed in another iteration => in-degree = 0 which will be caught be another reduction rule
            if isempty(incoming_edges[new_vertex_label])
                continue
            end

            predecessor = first(incoming_edges[new_vertex_label])
            for successor in outgoing_edges[new_vertex_label]

                if (predecessor == successor)    # TODO: correct comparison of ints?
                    # merging would introduce self-loop => cycle found!
                    # => put predecessor into preliminary DFVS and remove it from the graph together with the original vertex to be merged
                    push!(vertices_with_self_loops, predecessor)
                    remove_predecessor = true
                    # skip rest of inner for-loop as the predecessor and all its incident edges will be removed anyway
                    # => no need to add edges that are going to be removed right after
                    break
                end

                # edge might be already in the graph, so add_edge! can return false
                add_edge!(graph, predecessor, successor)
            end

            # remove merged vertex immediately -> this will change the labels
            if remove_predecessor
                # add predecessor to preliminary DFVS, has to be done before calling rem_vertices! because that will change the labels
                add_vertices_to_dfvs_rr_new!(solution, [predecessor])

                # get original labels of the vertices to be removed
                orig_label_pred = solution.vmap_new_orig[predecessor]

                # remove merged vertex and predecessor
                # rem_vertex! swaps the vertex to be removed with the last one and then removes the last vertex
                swapped_vertex = nv(graph)
                removed = rem_vertex!(graph, new_vertex_label)
                if removed
                    # set new label of the vertex to 0 after removal
                    solution.vmap_orig_new[vertex_orig] = 0
                
                    # update mapping for swapped vertex only if it is not the same as the removed vertex
                    # -> this is the case if the last vertex in the graph is removed and would then overwrite the mapping to 0 done above
                    if (swapped_vertex != new_vertex_label)
                        orig_label_swapped = solution.vmap_new_orig[swapped_vertex]
                        solution.vmap_orig_new[orig_label_swapped] = new_vertex_label
                        solution.vmap_new_orig[new_vertex_label] = orig_label_swapped
                    end
                    
                    # shorten mapping new -> orig by removing the last element
                    # TODO: correct?
                    pop!(solution.vmap_new_orig)
                end

                # predecessor might have a new label after the removal above
                new_label_pred = solution.vmap_orig_new[orig_label_pred]
                swapped_vertex = nv(graph)
                removed = rem_vertex!(graph, new_label_pred)
                if removed
                    # set new label of the vertex to 0 after removal
                    solution.vmap_orig_new[orig_label_pred] = 0
                
                    # update mapping for swapped vertex only if it is not the same as the removed vertex
                    # -> this is the case if the last vertex in the graph is removed and would then overwrite the mapping to 0 done above
                    if (swapped_vertex != new_label_pred)
                        orig_label_swapped = solution.vmap_new_orig[swapped_vertex]
                        solution.vmap_orig_new[orig_label_swapped] = new_label_pred
                        solution.vmap_new_orig[new_label_pred] = orig_label_swapped
                    end
                    
                    # shorten mapping new -> orig by removing the last element
                    # TODO: correct?
                    pop!(solution.vmap_new_orig)
                end

                # OLD version:
#= 
                # use the new vertex label as now used in the graph
                reverse_vmap = rem_vertices!(graph, [new_vertex_label, predecessor])
                # add new reverse_vmap for finding new labels
                push!(merging_reverse_vmaps, reverse_vmap)

                # update vmap for merging with the new labels
                merging_vmap[vertex] = 0    # vertex has been removed
                # merging_vmap[predecessor] = 0     -> wrong because predecessor might have a new label and we need to find the old label
                for (i, u) in enumerate(merging_vmap)

                    # compare u and length first to avoid out of bounds for accessing index u 
                    if (merging_vmap[i] != 0 && (u > length(reverse_vmap) || reverse_vmap[u] != u))   # vertex label of existing vertex has changed
                        # this should also take care of the predecessor mapping
                        new_label = findfirst(isequal(u), reverse_vmap)

                        if isnothing(new_label) # vertex has been removed
                            merging_vmap[i] = 0
                        else
                            merging_vmap[i] = new_label
                        end
                    end
                end

                # add new reverse_vmap to list of all vmaps
                push!(solution.reverse_vmaps_list, reverse_vmap)
                # use the original vertex (and not the new label) here -> but predecessor might use a new label => WRONG!
                union!(removed_vertices, [vertex, predecessor])
 =#
                
            else
                # remove merged vertex
                # rem_vertex! swaps the vertex to be removed with the last one and then removes the last vertex
                swapped_vertex = nv(graph)
                removed = rem_vertex!(graph, new_vertex_label)
                if removed
                    # set new label of the vertex to 0 after removal
                    solution.vmap_orig_new[vertex_orig] = 0
                
                    # update mapping for swapped vertex only if it is not the same as the removed vertex
                    # -> this is the case if the last vertex in the graph is removed and would then overwrite the mapping to 0 done above
                    if (swapped_vertex != new_vertex_label)
                        orig_label_swapped = solution.vmap_new_orig[swapped_vertex]
                        solution.vmap_orig_new[orig_label_swapped] = new_vertex_label
                        solution.vmap_new_orig[new_vertex_label] = orig_label_swapped
                    end
                    
                    # shorten mapping new -> orig by removing the last element
                    # TODO: correct?
                    pop!(solution.vmap_new_orig)
                end

                #= OLD version:
                # use the new vertex label as now used in the graph
                reverse_vmap = rem_vertices!(graph, [new_vertex_label])
                # add new reverse_vmap for finding new labels
                push!(merging_reverse_vmaps, reverse_vmap)

                # update vmap for merging with the new labels
                merging_vmap[vertex] = 0    # vertex has been removed
                for (i, u) in enumerate(merging_vmap)

                    # compare u and length first to avoid out of bounds for accessing index u 
                    if (merging_vmap[i] != 0 && (u > length(reverse_vmap) || reverse_vmap[u] != u))   # vertex label of existing vertex has changed
                        # this should also take care of the predecessor mapping
                        new_label = findfirst(isequal(u), reverse_vmap)

                        if isnothing(new_label) # vertex has been removed
                            merging_vmap[i] = 0
                        else
                            merging_vmap[i] = new_label
                        end
                    end
                end

                # add new reverse_vmap to list of all vmaps
                push!(solution.reverse_vmaps_list, reverse_vmap)
                # use the original vertex (and not the new label) here
                union!(removed_vertices, [vertex])
                =#

            end
        end # for-loop

        # add_vertices_to_dfvs_rr!(solution, vertices_with_self_loops) 
        
        # println("Incoming edges: $incoming_edges")
        # println("Outgoing edges: $outgoing_edges")
    end 

    # debugging - start

    # merging could introduce self-loops!!
    # => TODO: during(!) application of reduction rules, check for vertices with self-loops and add them to the FVS
    # or add them to a preliminary FV set, remove them from the graph and add this set to the FVS found by the following metaheuristic
    @assert !has_self_loops(graph)    

    # println("Incoming edges: $incoming_edges")
    # println("Outgoing edges: $outgoing_edges")

    # println("Length of reverse_vmaps_list after merging in-degree = 1: ", length(solution.reverse_vmaps_list))   
    @debug "Number of vertices before merging indegree = 1: $original_vertex_number"
    @debug "Number of edges before merging indegree = 1: $original_edge_number"
    # println("Number of vertices before merging indegree = 1: $original_vertex_number")
    # println("Number of edges before merging indegree = 1: $original_edge_number")
    # TODO: number below might not be correct after also checking for self-loops
    # println("Expected number of vertices after merging indegree = 1: ", original_vertex_number - length(indegree_one))
    
    # WRONG: might be fewer edges because the predecessor can already have edges to the successor vertices of the vertex that is merged
    # println("Expected number of edges after merging indegree = 1: ", original_edge_number - length(indegree_one)) 
    @debug "Vertex number of merged graph (indegree = 1): $(nv(graph))"
    @debug "Edge number of merged graph (indegree = 1): $(ne(graph))"
    # println("Vertex number of merged graph (indegree = 1): ", nv(graph))
    # println("Edge number of merged graph (indegree = 1): ", ne(graph))  

    # check how vertices are reordered after merging
    #=
    for i in 1:length(reverse_vmap)
        println("Merging: Vertex $i used to be vertex ", reverse_vmap[i])
    end
    =#  

    # debugging - end

    return rule_applicable

end # function merge_indegree_one_alternative3_new!


"""
    merge_outdegree_one_alternative3_new!(solution::DFVSPSolution)
Merge all vertices with out-degree = 1 with their successor.
Update mappings of vertex labels in solution.
Implementation: get all vertices with out-degree = 1 once in the beginning and remove vertices after each merge 
    => next vertex to be merged gets new label and has to be mapped to this new label before merging 
    -> this is done by using the mappings in solution
    -> could be inefficient
Steps:
    - find vertex with out-degree = 1
    - get list of adjacent vertices
    - add edges to adjacent vertices to the successor vertex
    - remove vertex
"""
function merge_outdegree_one_alternative3_new!(solution::DFVSPSolution)
    rule_applicable = false
    
    # get all vertices with out-degree = 1 once in the beginning and remove vertices after each merge 
    # => next vertex to be merged gets new label and has to be mapped to this new label before merging -> could be inefficient
    graph = solution.inst.graph
    incoming_edges = graph.badjlist
    outgoing_edges = graph.fadjlist

    # incoming_edges = Base.deepcopy(graph.badjlist)
    # outgoing_edges = Base.deepcopy(graph.fadjlist)

    # collect all vertices with out-degree = 1 in a vector
    outdegree_one = [v for v in vertices(graph) if length(outgoing_edges[v]) == 1]

    vertices_with_self_loops = Vector{Int}()
    # merging_reverse_vmaps = Vector{Vector{Integer}}()
    # removed_vertices = Set{Int}()

    n = nv(solution.inst.graph)
    # current label of every vertex is the vertex itself
    # merging_vmap = collect(1:n)


    # debugging - start

    # println("Incoming edges: $incoming_edges")
    # println("Outgoing edges: $outgoing_edges")

    original_vertex_number = nv(graph)
    original_edge_number = ne(graph) 
    
    @debug "Number of vertices with outdegree = 1: $(length(outdegree_one))"
    # println("Vertices with outdegree = 1: ", length(outdegree_one))   

    #= 
    for x in outdegree_one
        # println("Vertex to merge: $x")
    end 

    if (!isempty(outdegree_one))
        to_merge = outdegree_one[1]
        println("Before merge, outdegree = 1: Graph contains vertex $to_merge: ", has_vertex(graph, to_merge))
    end 
     =#

    # debugging - end   

    # store vertex mapping in solution
    # in the end, these mappings are needed to determine the original vertex label of the vertices in the feedback set
    if (!isempty(outdegree_one))
        rule_applicable = true

        # Sort the vertices to be merged -> TODO: necessary?
        vertices_to_merge = sort(outdegree_one)

        # build list of original labels of the vertices to be merged
        vertices_to_merge_orig_labels = Vector{Int}()
        for vertex in vertices_to_merge
            orig_label = solution.vmap_new_orig[vertex]
            push!(vertices_to_merge_orig_labels, orig_label)
        end

        # merge_vertices! from Graphs merges the vertices in the given list to the vertex with the lowest number
        # BUT: merge_vertices! Supports [`SimpleGraph`](@ref) only.
        # => own implementation necessary!
        # Traverse the list of vertices and add all their outgoing edges to their predecessor
        for vertex_orig in vertices_to_merge_orig_labels

            remove_successor = false

            # check if vertex has already been removed in another iteration
            #= 
            if in(vertex, removed_vertices)
                continue
            end
            =#

            # new_vertex_label = find_new_vertex_label(merging_reverse_vmaps, vertex)
            # new_vertex_label = merging_vmap[vertex]

            new_vertex_label = solution.vmap_orig_new[vertex_orig]

            # check if vertex has already been removed in another iteration            
            if (new_vertex_label == 0)
                # println("SKIP: already removed")
                continue
            end
            
            # println("Number of vertices in graph: ", nv(graph))
            # println("Length of outgoing_edges list: ", length(outgoing_edges))

            # check if successor has already been removed in another iteration => out-degree = 0 which will be caught by another reduction rule
            if isempty(outgoing_edges[new_vertex_label])
                # println("SKIP: no outgoing neighbor")
                continue
            end

            successor = first(outgoing_edges[new_vertex_label])
            for predecessor in incoming_edges[new_vertex_label]

                if (successor == predecessor)    # TODO: correct comparison of ints?
                    # merging would introduce self-loop => cycle found!
                    # => put successor into preliminary DFVS and remove it from the graph together with the original vertex to be merged
                    push!(vertices_with_self_loops, successor)
                    remove_successor = true
                    # skip rest of inner for-loop as the successor and all its incident edges will be removed anyway
                    # => no need to add edges that are going to be removed right after
                    break
                end

                # edge might be already in the graph, so add_edge! can return false
                add_edge!(graph, predecessor, successor)
            end

            # remove merged vertex immediately -> this will change the labels
            if remove_successor
                # add successor to preliminary DFVS, has to be done before calling rem_vertices! because that will change the labels
                add_vertices_to_dfvs_rr_new!(solution, [successor])

                # get original labels of the vertices to be removed
                orig_label_succ = solution.vmap_new_orig[successor]

                # remove merged vertex and successor
                # rem_vertex! swaps the vertex to be removed with the last one and then removes the last vertex
                swapped_vertex = nv(graph)
                removed = rem_vertex!(graph, new_vertex_label)
                if removed
                    # set new label of the vertex to 0 after removal
                    solution.vmap_orig_new[vertex_orig] = 0
                
                    # update mapping for swapped vertex only if it is not the same as the removed vertex
                    # -> this is the case if the last vertex in the graph is removed and would then overwrite the mapping to 0 done above
                    if (swapped_vertex != new_vertex_label)
                        orig_label_swapped = solution.vmap_new_orig[swapped_vertex]
                        solution.vmap_orig_new[orig_label_swapped] = new_vertex_label
                        solution.vmap_new_orig[new_vertex_label] = orig_label_swapped
                    end
                    
                    # shorten mapping new -> orig by removing the last element
                    # TODO: correct?
                    pop!(solution.vmap_new_orig)
                end

                # successor might have a new label after the removal above
                new_label_succ = solution.vmap_orig_new[orig_label_succ]
                swapped_vertex = nv(graph)
                removed = rem_vertex!(graph, new_label_succ)
                if removed
                    # set new label of the vertex to 0 after removal
                    solution.vmap_orig_new[orig_label_succ] = 0
                
                    # update mapping for swapped vertex only if it is not the same as the removed vertex
                    # -> this is the case if the last vertex in the graph is removed and would then overwrite the mapping to 0 done above
                    if (swapped_vertex != new_label_succ)
                        orig_label_swapped = solution.vmap_new_orig[swapped_vertex]
                        solution.vmap_orig_new[orig_label_swapped] = new_label_succ
                        solution.vmap_new_orig[new_label_succ] = orig_label_swapped
                    end
                    
                    # shorten mapping new -> orig by removing the last element
                    # TODO: correct?
                    pop!(solution.vmap_new_orig)
                end

                # OLD version:
#= 
                # use the new vertex label as now used in the graph
                reverse_vmap = rem_vertices!(graph, [new_vertex_label, successor])
                # add new reverse_vmap for finding new labels
                push!(merging_reverse_vmaps, reverse_vmap)

                # update vmap for merging with the new labels
                merging_vmap[vertex] = 0    # vertex has been removed
                # merging_vmap[successor] = 0     -> wrong because successor might have a new label and we need to find the old label
                for (i, u) in enumerate(merging_vmap)

                    # compare u and length first to avoid out of bounds for accessing index u 
                    if (merging_vmap[i] != 0 && (u > length(reverse_vmap) || reverse_vmap[u] != u))   # vertex label of existing vertex has changed
                        # this should also take care of the predecessor mapping
                        new_label = findfirst(isequal(u), reverse_vmap)

                        if isnothing(new_label) # vertex has been removed
                            merging_vmap[i] = 0
                        else
                            merging_vmap[i] = new_label
                        end
                    end
                end

                # add new reverse_vmap to list of all vmaps
                push!(solution.reverse_vmaps_list, reverse_vmap)
                # use the original vertex (and not the new label) here -> but label for successor might already be a new one
                union!(removed_vertices, [vertex, successor])
 =#

            else
                # remove merged vertex
                # rem_vertex! swaps the vertex to be removed with the last one and then removes the last vertex
                swapped_vertex = nv(graph)
                removed = rem_vertex!(graph, new_vertex_label)
                if removed
                    # set new label of the vertex to 0 after removal
                    solution.vmap_orig_new[vertex_orig] = 0
                
                    # update mapping for swapped vertex only if it is not the same as the removed vertex
                    # -> this is the case if the last vertex in the graph is removed and would then overwrite the mapping to 0 done above
                    if (swapped_vertex != new_vertex_label)
                        orig_label_swapped = solution.vmap_new_orig[swapped_vertex]
                        solution.vmap_orig_new[orig_label_swapped] = new_vertex_label
                        solution.vmap_new_orig[new_vertex_label] = orig_label_swapped
                    end
                    
                    # shorten mapping new -> orig by removing the last element
                    # TODO: correct?
                    pop!(solution.vmap_new_orig)
                end

                # OLD version:
#= 
                # use the new vertex label as now used in the graph
                reverse_vmap = rem_vertices!(graph, [new_vertex_label])
                # add new reverse_vmap for finding new labels
                push!(merging_reverse_vmaps, reverse_vmap)

                # update vmap for merging with the new labels
                merging_vmap[vertex] = 0    # vertex has been removed
                for (i, u) in enumerate(merging_vmap)
                    # compare u and length first to avoid out of bounds for accessing index u 
                    if (merging_vmap[i] != 0 && (u > length(reverse_vmap) || reverse_vmap[u] != u))   # vertex label of existing vertex has changed
                        # this should also take care of the predecessor mapping
                        new_label = findfirst(isequal(u), reverse_vmap)

                        if isnothing(new_label) # vertex has been removed
                            merging_vmap[i] = 0
                        else
                            merging_vmap[i] = new_label
                        end
                    end
                end

                # add new reverse_vmap to list of all vmaps
                push!(solution.reverse_vmaps_list, reverse_vmap)
                # use the original vertex (and not the new label) here
                union!(removed_vertices, [vertex])
 =#

            end
        end # for-loop

        # add_vertices_to_dfvs_rr!(solution, vertices_with_self_loops) 
        
        # println("Incoming edges: $incoming_edges")
        # println("Outgoing edges: $outgoing_edges")
    end 

    # debugging - start

    # merging could introduce self-loops!!
    # => TODO: during(!) application of reduction rules, check for vertices with self-loops and add them to the FVS
    # or add them to a preliminary FV set, remove them from the graph and add this set to the FVS found by the following metaheuristic
    @assert !has_self_loops(graph)    

    # println("Incoming edges: $incoming_edges")
    # println("Outgoing edges: $outgoing_edges")

    # println("Length of reverse_vmaps_list after merging out-degree = 1: ", length(solution.reverse_vmaps_list))   
    @debug "Number of vertices before merging outdegree = 1: $original_vertex_number"
    @debug "Number of edges before merging outdegree = 1: $original_edge_number"
    # println("Number of vertices before merging outdegree = 1: $original_vertex_number")
    # println("Number of edges before merging outdegree = 1: $original_edge_number")
    # TODO: number below might not be correct after also checking for self-loops
    # println("Expected number of vertices after merging outdegree = 1: ", original_vertex_number - length(outdegree_one))
    
    # WRONG: might be fewer edges because the predecessor can already have edges to the successor vertices of the vertex that is merged
    # println("Expected number of edges after merging outdegree = 1: ", original_edge_number - length(outdegree_one)) 
    @debug "Vertex number of merged graph (outdegree = 1): $(nv(graph))"
    @debug "Edge number of merged graph (outdegree = 1): $(ne(graph))"
    # println("Vertex number of merged graph (outdegree = 1): ", nv(graph))
    # println("Edge number of merged graph (outdegree = 1): ", ne(graph))  

    # check how vertices are reordered after merging
    #=
    for i in 1:length(reverse_vmap)
        println("Merging: Vertex $i used to be vertex ", reverse_vmap[i])
    end
    =#  

    # debugging - end

    return rule_applicable

end # function merge_outdegree_one_alternative3_new!


"""
    remove_self_loops_new!(solution::DFVSPSolution)
Find all vertices with self-loops and add them to the preliminary DFVS as such vertices have to be in the final DFVS.
Remove all vertices with self-loops and all their incident edges from the graph given in the instance inside solution.
Update mappings of vertex labels in solution.

Returns: true if rule has been applied, false otherwise
"""
function remove_self_loops_new!(solution::DFVSPSolution)
    rule_applicable = false

    graph = solution.inst.graph

    # collect all vertices with self-loops in a vector
    self_loops = [v for v in vertices(graph) if has_edge(graph, v, v)]

    # debugging - start

    original_vertex_number = nv(graph)
    original_edge_number = ne(graph)

    @debug "Number of vertices with self-loops: $(length(self_loops))"
    # println("Vertices with self-loops: ", length(self_loops))

    #= 
    for x in self_loops
      println("Vertex with self-loop: $x")
    end
     =#

    @debug "Number of vertices before removing self-loops: $original_vertex_number"
    @debug "Number of edges before removing self-loops: $original_edge_number"
    # println("Number of vertices before removing self-loops: $original_vertex_number")
    # println("Number of edges before removing self-loops: $original_edge_number")

    # debugging end

    if (!isempty(self_loops))
        rule_applicable = true
        # add vertices to preliminary DFVS -> has to be done before removal as removing changes the vertex labels
        add_vertices_to_dfvs_rr_new!(solution, self_loops)
        # set new label of these vertices to 0 because they will be removed
        for vertex in self_loops
            original_label = solution.vmap_new_orig[vertex]
            solution.vmap_orig_new[original_label] = 0
        end

        # remove vertices and incident edges from the graph
        reverse_vmap = rem_vertices!(graph, self_loops)

        # update vertex mappings

        # create temporary mapping of correct new length
        # the old mapping in solution is needed for the update and cannot be changed directly
        # temp_vmap_new_orig = Vector{Integer}(length(reverse_vmap))
        temp_vmap_new_orig = zeros(Integer, length(reverse_vmap))
        for i in 1:length(temp_vmap_new_orig)
            # update original label
            original_label = solution.vmap_new_orig[reverse_vmap[i]]
            temp_vmap_new_orig[i] = original_label
            # update new label
            solution.vmap_orig_new[original_label] = i
        end
        # store new mapping in solution
        solution.vmap_new_orig = copy(temp_vmap_new_orig)
        
        
        # push!(solution.reverse_vmaps_list, reverse_vmap)
    end

    # debugging - start

    # println("Length of reverse_vmaps_list after removing self-loops: ", length(solution.reverse_vmaps_list))

    @debug "Expected number of vertices after removing self-loops: $(original_vertex_number - length(unique!(self_loops)))"
    # println("Expected number of vertices after removing self-loops: ", original_vertex_number - length(unique!(self_loops)))

    @debug "Vertex number of modified graph: $(nv(graph))"
    @debug "Edge number of modified graph: $(ne(graph))"
    # println("Vertex number of modified graph: ", nv(graph))
    # println("Edge number of modified graph: ", ne(graph))   # could also use "graph.ne"

    # check how vertices are reordered after removal
    #     for i in 1:length(reverse_vmap)
    #         println("Self-loop removal: Vertex $i used to be vertex ", reverse_vmap[i])
    #     end

    # debugging - end

    return rule_applicable

end # function remove_self_loops_new!


"""
    reduce_graph_using_strongly_connected_components_new!(solution::DFVSPSolution)
Find all strongly connected components (SCCs) of a directed graph given in the instance inside solution.
Remove all SCCs that consist of a single vertex from the given graph as such vertices cannot be part of any cycle.
Add an arbitrary vertex from each SCC with 2 vertices to the preliminary DFVS and remove these SCCs and their vertices from the graph.
Update mappings of vertex labels in solution.

Returns: an array of arrays, each of which is an entire strongly connected component, and true if vertices were removed, otherwise false.
"""
function reduce_graph_using_strongly_connected_components_new!(solution::DFVSPSolution)
    rule_applicable = false

    graph = solution.inst.graph

    n = nv(graph)
    # current label of every vertex is the vertex itself
    # sccs_vmap = collect(1:n)

    # find all SCCs
    # 2 options: Tarjan's Algorithm or Kosaraju's Algorithm

    # using Tarjan's Algorithm
    sccs = strongly_connected_components(graph)

    # using Kosaraju's Algorithm
    # Time Complexity : O(|E|+|V|)
    # Space Complexity : O(|V|) {Excluding the memory required for storing graph}
    # sccs_kosaraju = strongly_connected_components_kosaraju(graph)


    # debugging - start

    @debug "Number of SCCs (Tarjan's alg): $(length(sccs))"
    # println("Number of SCCs (Tarjan's alg): ", length(sccs))
    # println("Found SCCs (Tarjan's alg): $sccs")

    # println("Number of SCCs (Kosaraju's alg): ", length(sccs_kosaraju))
    # println("Found SCCs (Kosaraju's alg): $sccs_kosaraju")

    # debugging - end

    # collect indizes of SCCs to delete
    scc_indices_to_delete = Vector{Int}()
    # collect all vertices from SCCs to delete
    vertices_to_delete = Vector{Int}()

    # store original labels for all vertices in the SCCs
    sccs_orig_labels = Array{Array{Int, 1}, 1}(undef, length(sccs))

    # look for SCCs with 1 or 2 vertices
    # build list of SCCs with original vertex labels
    for i in 1:length(sccs)
        scc = sccs[i]
        scc_length = length(scc)

        # add original labels in the same order as the vertices occur in the SCC
        sccs_orig_labels[i] = Array{Int, 1}(undef, scc_length)
        for j in 1:scc_length
            orig_label = solution.vmap_new_orig[scc[j]]
            # push!(sccs_orig_labels[i], orig_label)
            sccs_orig_labels[i][j] = orig_label
        end

        # remove SCC with a single vertex from the graph and also the list of SCCs
        if (scc_length == 1)
            # println("Found SCC with a single vertex: ", scc)
            # store index of this SCC to remove it later from the list of SCCs
            push!(scc_indices_to_delete, i)

            # remove single vertex in the SCC from the graph
            # reverse_vmap = rem_vertices!(graph, scc)
            # store vertex mapping in solution
            # push!(solution.reverse_vmaps_list, reverse_vmap)

            # store vertices of this SCC to remove them later from the graph
            append!(vertices_to_delete, scc)

            rule_applicable = true

        # SCC with 2 vertices: add arbitrary vertex to preliminary DFVS
        # then remove SCC from the graph and also the list of SCCs
        elseif (scc_length == 2)
            # add arbitrary (-> first) vertex from SCC to preliminary DFVS, has to be done before calling rem_vertices! because that will change the labels
            # TODO: maybe remove vertex with bigger degree instead of arbitrarily choosing -> could lead to further/better reductions with other rules
            add_vertices_to_dfvs_rr_new!(solution, [scc[1]])

            # println("Found SCC with 2 vertices: ", scc)
            # store index of this SCC to remove it later from the list of SCCs
            push!(scc_indices_to_delete, i)
            # store vertices of this SCC to remove them later from the graph
            append!(vertices_to_delete, scc)

            rule_applicable = true
        #=
        elseif (scc_length == 3)
            subgraph,vmap = induced_subgraph(graph, scc)
            if ne(subgraph) == 6
                # two vertices have to be in the DFVS in this case, but it does not matter which ones
                add_vertices_to_dfvs_rr_new!(solution, scc[1:2])
            else
                # in this case it should be possible to apply the reduction rule for vertices with in-/outdegree 1
                # anyways the DFVS has to have one vertex and we can just take the one with the highest degree
                dfvs_vertex = findmax(v -> degree(subgraph, v), vertices(subgraph))[2]
                add_vertices_to_dfvs_rr_new!(solution, [vmap[dfvs_vertex]])
            end

            # println("Found SCC with 3 vertices: ", scc)
            # store index of this SCC to remove it later from the list of SCCs
            push!(scc_indices_to_delete, i)
            # store vertices of this SCC to remove them later from the graph
            append!(vertices_to_delete, scc)

            rule_applicable = true
        =#
        end
    end

    # set new label of vertices to 0 because they will be removed
    for vertex in vertices_to_delete
        original_label = solution.vmap_new_orig[vertex]
        solution.vmap_orig_new[original_label] = 0
    end

    # remove vertices from the graph
    reverse_vmap = rem_vertices!(graph, vertices_to_delete)

    # update vertex mappings
    # create temporary mapping of correct new length
    # the old mapping in solution is needed for the update and cannot be changed directly
    # temp_vmap_new_orig = Vector{Integer}(length(reverse_vmap))
    temp_vmap_new_orig = zeros(Integer, length(reverse_vmap))
    for i in 1:length(temp_vmap_new_orig)
        # update original label
        original_label = solution.vmap_new_orig[reverse_vmap[i]]
        temp_vmap_new_orig[i] = original_label
        # update new label
        solution.vmap_orig_new[original_label] = i
    end
    # store new mapping in solution
    solution.vmap_new_orig = copy(temp_vmap_new_orig)

    # OLD: store vertex mapping in solution
    # push!(solution.reverse_vmaps_list, reverse_vmap)

    # remove all SCCs with 1 or 2 vertices from the list of SCCs
    deleteat!(sccs, scc_indices_to_delete)
    deleteat!(sccs_orig_labels, scc_indices_to_delete)

    # TODO: relabel all vertices in remaining SCCs 
    # OR simply use the algorithm again to find all SCCs
    # OR just rebuild SCCs with new labels from scratch using the vertex mappings
    # -> currently rebuilding is done below

    # rebuild the remaining SCCs with the new vertex labels
    sccs_new_labels = Array{Array{Int, 1}, 1}(undef, length(sccs_orig_labels))

    for i in 1:length(sccs_orig_labels)
        scc = sccs_orig_labels[i]
        sccs_new_labels[i] = Array{Int, 1}()

        # add new labels
        for orig_label in scc
            new_label = solution.vmap_orig_new[orig_label]

            # check if SCCs have been correctly removed as well
            (new_label != 0) || 
                throw(ArgumentError("Vertex to relabel must not have been removed."))
            
            push!(sccs_new_labels[i], new_label)
        end

        #= OR: add labels in the same order
        scc_length = length(scc)
        sccs_new_labels[i] = Array{Int, 1}(undef, scc_length)
        for j in 1:scc_length
            new_label = solution.vmap_orig_new[scc[j]]
            # push!(sccs_orig_labels[i], orig_label)
            sccs_new_labels[i][j] = new_label
        end
        =#
    end


    # OLD relabel:
    #= 
    # set label of removed vertices to 0
    for vertex in vertices_to_delete
        sccs_vmap[vertex] = 0        
    end

    # update vmap with the new labels
    for (i, u) in enumerate(sccs_vmap)

        # compare u and length first to avoid out of bounds for accessing index u 
        if (sccs_vmap[i] != 0 && (u > length(reverse_vmap) || reverse_vmap[u] != u))   # vertex label of existing vertex has changed
            new_label = findfirst(isequal(u), reverse_vmap)

            if isnothing(new_label) # vertex has been removed
                sccs_vmap[i] = 0
            else
                sccs_vmap[i] = new_label
            end
        end
    end
    # relabel vertices in the remaining SCCs
    for scc in sccs
        for i in 1:length(scc)
            current_vertex = scc[i]

            # check if SCCs have been correctly removed as well
            (sccs_vmap[current_vertex] != 0) || 
                throw(ArgumentError("Vertex to relabel must not have been removed."))
            
            scc[i] = sccs_vmap[current_vertex]
        end
        
    end
 =#


    # debugging - start

    @debug "Number of SCCs after removal of SCCs with 1 or 2 vertices: $(length(sccs_new_labels))"
    # println("Number of SCCs after removal of SCCs with 1 or 2 vertices: ", length(sccs_new_labels))
    # println("New SCCs: $sccs_new_labels")
    
    # debugging - end

    return sccs_new_labels, rule_applicable
    
end # function reduce_graph_using_strongly_connected_components_new!



"""
    compute_indegree_outdegree_difference(vertex::T, graph::SimpleDiGraph{Int}) where {T <: Integer}
For the given vertex, compute its value using the following greedy function: 
    h(v) = deg^{-}(v) + deg^{+}(v) - λ × |deg^{-}(v) - deg^{+}(v)|
Argument λ (lambda) is optional: If no value is provided, it defaults to 0.3.

Returns: the heuristic value of the vertex.
"""
function compute_indegree_outdegree_difference(vertex::T, graph::SimpleDiGraph{Int}, lambda::Float64=0.3) where {T <: Integer}

    indegree_value = indegree(graph, vertex)
    outdegree_value = outdegree(graph, vertex)

    hvalue = indegree_value + outdegree_value - (lambda * abs(indegree_value - outdegree_value))

    return hvalue
    
end # function compute_indegree_outdegree_difference


"""
    construction_heuristic_indegree_outdegree_difference_alternative3!(solution::DFVSPSolution, lambda::Float64, res::Result) 
Apply the construction heuristic using the indegree_outdegree_difference heuristic.
Sequentially add the vertices in increasing order of their h value (or decreasing order regarding -h(v)) to a topological ordering. 
Add nodes with backward arcs iteratively to the FVS but only if not all their already visited out-neighbors are already in the FVS.
Argument λ (lambda): recommended setting is 0.3.
Argument res is not used and only present to conform with calls used in MHLib.

This method does NOT change the original graph inside solution!

"""
function construction_heuristic_indegree_outdegree_difference_alternative3!(solution::DFVSPSolution, lambda::Float64, res::Result)
    graph = solution.inst.graph

    # reset the current solution to empty
    clear_dfvs!(solution)

    # build a dictionary of vertices => heuristic value
    dict_vertex_hvalue = Dict(i => compute_indegree_outdegree_difference(i, graph, lambda) for i in vertices(graph))

    # sort dictionary by its values (= heuristic values of vertices), output = array of tuples
    sorted = sort(collect(dict_vertex_hvalue), by=x->x[2])

    # construct topological order and get vertices with backward arcs

    # get vertices in increasing order of their h value
    topo_ord = [sorted[i][1] for i in 1:length(sorted)]

    visited = Set{Int}()
    solution_set = Set{Int}()

    for i in 1:length(topo_ord)
        current_vertex = topo_ord[i]
        current_outneighbors = Set(outneighbors(graph, current_vertex))

        # TODO: use intersect or isdisjoint?
        visited_outneighbors = Base.intersect(visited, current_outneighbors)
        if (!isempty(visited_outneighbors))
            # only add vertex to DFVS if not all visited out-neighbors are in the solution
            if (!issubset(visited_outneighbors, solution_set))
                # add vertex to DFVS
                add_vertices_to_solution_new!(solution, [current_vertex])
                push!(solution_set, current_vertex)
            end
            
        end

        push!(visited, current_vertex)
    end

    sort_sel!(solution)

    # update the global best solution
    update_global_best_solution_add_part(solution.x)

    #= 
    # DAG analysis and plotting
    analyse_dag(solution)
    instance_name = "h_199"
    # get current datetime
    df = Dates.DateFormat("dd-mm-yyyy_HH-MM-SS")
    current_datetime = Dates.now()
    formatted_datetime = Dates.format(current_datetime, df)
    plot_dag(solution, instance_name * "-" * formatted_datetime * "-afterCH")
 =#
    
end # function construction_heuristic_indegree_outdegree_difference_alternative3!




"""
    mtz_formulation!(solution::DFVSPSolution)

Build and solve a MILP model for the DFVSP using the MTZ formulation.
Use the model to repair the given solution to be valid again.
"""
function mtz_formulation!(solution::DFVSPSolution)
    # choose which solver to use, options are "gurobi", "scip"
    solver_name = "scip"
    # solver_name = "gurobi"

    # get times and compute time limit for solver
    # also check termination criterion based on time
    # global_start_time = MyUtils.global_start_time
    # global_run_time_limit = MyUtils.global_run_time_limit

    # start with 60 seconds
    solver_time_limit = first_time_limit_mip_solver
    elapsed_time = time() - global_start_time

    if (global_run_time_limit < 0)
        # no global time limit set, so just continue
    elseif (elapsed_time >= global_run_time_limit)
        # run time limit already exceeded

        # TODO: return biggest solution possible (as done below) or just set solver time to 0? -> should be the same outcome?
        # => return all vertices of the graph as solution to avoid returning an invalid solution
        @debug "MIP solving: Global run time limit exceeded."
        # do not clear the current DFVS as it can contain also other vertices that are not part of the graph at hand
        # clear_dfvs!(solution)
        # there should be no overlap between the graph vertices and vertices already contained in the DFVS
        all_vertices = collect(vertices(solution.inst.graph))
        add_vertices_to_solution_new!(solution, all_vertices)

        return
    else
        remaining_time = (global_run_time_limit - elapsed_time)
        # set solver time limit to the smaller one of 60 seconds and the remaining time
        solver_time_limit = min(solver_time_limit, remaining_time)

        @debug "Remaining run time = $remaining_time"
        @debug "First solver time limit = $solver_time_limit"
    end

    graph = solution.inst.graph
    n_original = nv(graph)

    graph_with_source = SimpleDiGraph(graph)
    # add source vertex
    @assert add_vertex!(graph_with_source)
    n_source = nv(graph_with_source)

    # add edges from source vertex to all other vertices
    for i in 1:n_original
        @assert add_edge!(graph_with_source, n_source, i)
    end

    model = missing 

    if (solver_name == "scip")
        # model = Model(SCIP.Optimizer)
        # set max memory usage in MB, type = real, range: [0,8796093022207], default: 8796093022207]
        model = Model(optimizer_with_attributes(SCIP.Optimizer, "limits/memory" => 4000))
        # maximal time in seconds to run
        # -> [type: real, advanced: FALSE, range: [0,1e+20], default: 1e+20]
        set_optimizer_attribute(model, "limits/time", solver_time_limit)

        # verbosity level of output -> [type: int, advanced: FALSE, range: [0,5], default: 4]
        set_optimizer_attribute(model, "display/verblevel", 0)

    elseif (solver_name == "gurobi")
        # reuse the same Gurobi environment
        model = Model(() -> Gurobi.Optimizer(GRB_ENV))
        MOI.set(model, MOI.NumberOfThreads(), 1)

        # set attributes for Gurobi optimizer
        # TimeLimit: total time expended in seconds, takes type double, default = infinity
        set_optimizer_attribute(model, "TimeLimit", solver_time_limit)   
    end

    # model = Model(SCIP.Optimizer)
    # remove symmetry computation to hopefully avoid problems with bliss library -> works only sometimes
    # model = Model(optimizer_with_attributes(SCIP.Optimizer, "misc/usesymmetry" => 0))
    # model = Model(GLPK.Optimizer)
    # model = Model(Gurobi.Optimizer)

    # reuse the same Gurobi environment
    # model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    # MOI.set(model, MOI.NumberOfThreads(), 1)

    # set attributes for Gurobi optimizer
    # TimeLimit: total time expended in seconds, takes type double, default = infinity
    # set_optimizer_attribute(model, "TimeLimit", solver_time_limit)    
    # MIPGap: relative MIP optimality gap between the lower and upper objective bound
    # -> terminate when gap is less than MIPGap times the absolute value of the incumbent objective value
    # type double, default value = 1e-4 = 0.0001 = 0.01%
    # Ex.: h_055 takes a long time (> 10325s) to go to 0.30% = 0.003
    # set_optimizer_attribute(model, "MIPGap", 0.005)
    # LogToConsole: control console logging, type int, default = 1, turn off = 0
    # set_optimizer_attribute(model, "LogToConsole", 0)
    # OutputFlag: control all logging (log to file, log to console), type int, default = 1, turn off = 0
    # set_optimizer_attribute(model, "OutputFlag", 0)

    # add variables:

    # variables for edges
    edges_array = collect(edges(graph_with_source))
    # @variable(model, x[edges_array], Bin)
    @variable(model, x[edges_array] >= 0.0)


    # variables for vertices
    @variable(model, y[1:n_source], Bin)
    # TODO correct or use constraint instead?
    # source vertex always has to be chosen
    fix(y[n_source], 1; force = true)
    # instead of: @constraint(model, select_source, y[n_source] == 1)

    # potential variables for vertices
    @variable(model, Φ[1:n_source] >= 0.0)
    # TODO correct or use constraint instead?
    # potential of source = 0
    fix(Φ[n_source], 0.0; force = true)
    # instead of: @constraint(model, source_potential_zero, Φ[n_source] == 0.0)


    # add constraints:

    # constraint for increasing potential values of vertices along selected arcs (excluding the source vertex)
    @constraint(model, increasing_potential_value[e = edges_array], (Φ[src(e)] - Φ[dst(e)] + (n_source * x[e])) <= (n_source - 1))
    
    # constraint for potential values of selected vertices -> JuMP needs 2 separate constraints
    # @constraint(model, potential_value[i=1:n_original], y[i] <= Φ[i] <= (n_source - 1))
    @constraint(model, potential_value_left_bound[i=1:n_original], y[i] <= Φ[i])
    @constraint(model, potential_value_right_bound[i=1:n_original], Φ[i] <= (n_source - 1))

    # constraints for selecting both endpoints for a selected arc
    @constraint(model, select_source[e = edges_array], x[e] <= y[src(e)])
    @constraint(model, select_destination[e = edges_array], x[e] <= y[dst(e)])

    # constraint for selecting the incident arc of two selected vertices
    @constraint(model, select_arc[e = edges_array], x[e] >= (y[src(e)] + y[dst(e)] - 1))


    # add objective:

    # objective: maximize the selected vertices (belonging to the forest = opposite of DFVS) -> excluding the source vertex
    # => minimizes the DFVS (= the vertices that are not selected)
    @objective(model, Max, sum(y[i] for i in 1:n_original))

    # show model -> can be very big for large graphs
    # print(model)
    @debug "Finished building the model."
    # println("Finished building the model.")

    # start optimization
    optimize!(model)

    # show solution
    # println(solution_summary(model))

    # recommended workflow
    if termination_status(model) == MOI.OPTIMAL
        # println("Solution is optimal")
    elseif termination_status(model) == MOI.TIME_LIMIT && has_values(model)
        # println("Solution is suboptimal due to a time limit, but a primal solution is available")
    elseif termination_status(model) == MEMORY_LIMIT && has_values(model)
        # println("Solution is suboptimal due to a memory limit, but a primal solution is available")
    elseif termination_status(model) == MOI.INTERRUPTED && has_values(model)
        # println("Solution is suboptimal due to an interrupt, but a primal solution is available")

        # after an interrupt, we should terminate -> get solution and return

        # build new solution from model
        # collect vertices for solution in a list
        solution_vertices = Vector{Int}()
        # all variables y_v with value 1 are part of the DAG => NOT part of the solution
        # all variables y_v with value 0 are not part of the DAG => part of the solution
        for i in 1:n_original
            # if (value(y[i]) == 0)   # TODO correct check for value 0 of a binary decision variable?
            if (isapprox(value(y[i]), 0, atol = 1e-1))  # check for value 0 with a certain tolerance
                push!(solution_vertices, i)   
            end
        end

        # alternative?
        # solution_vertices = [i for i in 1:n_original if (value(y[i]) == 0)]

        # add vertices to the solution
        add_vertices_to_solution_new!(solution, solution_vertices)

        return
    else
        # println("No solution found.")

        # to avoid an error or crash, simply return
        # => return all vertices of the graph as solution to avoid returning an invalid solution
        @debug "MIP solving: unknown error."
        # do not clear the current DFVS as it can contain also other vertices that are not part of the graph at hand
        # clear_dfvs!(solution)
        # there should be no overlap between the graph vertices and vertices already contained in the DFVS
        all_vertices = collect(vertices(solution.inst.graph))
        add_vertices_to_solution_new!(solution, all_vertices)

        return

        #= 
        # check for conflicts in the model and constraints
        compute_conflict!(model)
        if MOI.get(model, MOI.ConflictStatus()) != MOI.CONFLICT_FOUND
            error("No conflict could be found for an infeasible model.")
        end

        # collect conflicting constraints and print them
        conflict_constraint_list = ConstraintRef[]
        for (F, S) in list_of_constraint_types(model)
            for con in all_constraints(model, F, S)
                if MOI.get(model, MOI.ConstraintConflictStatus(), con) == MOI.IN_CONFLICT
                    push!(conflict_constraint_list, con)
                    # println(con)
                end
            end
        end

        error("The model was not solved correctly.")
        =#
    end

    # println("  objective value = ", objective_value(model))
    if primal_status(model) == MOI.FEASIBLE_POINT
        # -> large output for large graphs/models
        # println("  primal solution: y = ", value.(y))
    end
    if dual_status(model) == MOI.FEASIBLE_POINT
       # println("  dual solution: c1 = ", dual(c1))
    end


    # EXPERIMENTAL section

    elapsed_time = time() - global_start_time

    # global run time limit is set and already exceeded
    if ((global_run_time_limit > 0) && (elapsed_time >= global_run_time_limit))
        # continue below

    elseif termination_status(model) == MOI.TIME_LIMIT
        # global run time limit not set or not exceeded yet
        # apply gap limit only if solving was terminated because of exceeded time limit; remove/increase time limit
        # @info "Time limit for solving was exceeded, now continue solving with gap limit and higher time limit"

        # TODO maybe do not completely remove time limit but set it to a higher value and still consider global run time limit
        # start with 300 seconds = 5 minutes
        solver_time_limit = second_time_limit_mip_solver

        if (global_run_time_limit < 0)
            # no global time limit set, so just continue
        else
            remaining_time = (global_run_time_limit - elapsed_time)
            # set solver time limit to the smaller one of 300 seconds and the remaining time
            solver_time_limit = min(solver_time_limit, remaining_time)

            @debug "Remaining run time = $remaining_time"
            @debug "Second solver time limit = $solver_time_limit"
        end

        if (solver_name == "scip")
            set_optimizer_attribute(model, "limits/time", solver_time_limit)
            # relative gap = |primal - dual|/MIN(|dual|,|primal|) is below the given value, the gap is 'Infinity', if primal and dual bound have opposite signs
            # -> [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0]
            set_optimizer_attribute(model, "limits/gap", 0.005)
    
        elseif (solver_name == "gurobi")
            # set_optimizer_attribute(model, "TimeLimit", floatmax(Float64)) 
            set_optimizer_attribute(model, "TimeLimit", solver_time_limit) 
            set_optimizer_attribute(model, "MIPGap", 0.005)
        end

        # set_optimizer_attribute(model, "TimeLimit", floatmax(Float64)) 
        # set_optimizer_attribute(model, "TimeLimit", solver_time_limit) 
        # set_optimizer_attribute(model, "MIPGap", 0.005)

        # continue optimization
        optimize!(model)

        # show solution
        # println(solution_summary(model))

        # recommended workflow
        if termination_status(model) == MOI.OPTIMAL
            # println("Solution is optimal")
        elseif termination_status(model) == MOI.TIME_LIMIT && has_values(model)
            # println("Solution is suboptimal due to a time limit, but a primal solution is available")
        elseif termination_status(model) == MEMORY_LIMIT && has_values(model)
            # println("Solution is suboptimal due to a memory limit, but a primal solution is available")
        elseif termination_status(model) == MOI.INTERRUPTED && has_values(model)
            # println("Solution is suboptimal due to an interrupt, but a primal solution is available")

            # after an interrupt, we should terminate -> get solution and return

            # build new solution from model
            # collect vertices for solution in a list
            solution_vertices = Vector{Int}()
            # all variables y_v with value 1 are part of the DAG => NOT part of the solution
            # all variables y_v with value 0 are not part of the DAG => part of the solution
            for i in 1:n_original
                # if (value(y[i]) == 0)   # TODO correct check for value 0 of a binary decision variable?
                if (isapprox(value(y[i]), 0, atol = 1e-1))  # check for value 0 with a certain tolerance
                    push!(solution_vertices, i)   
                end
            end

            # alternative?
            # solution_vertices = [i for i in 1:n_original if (value(y[i]) == 0)]

            # add vertices to the solution
            add_vertices_to_solution_new!(solution, solution_vertices)

            return
        else
            # println("No solution found.")

            # to avoid an error or crash, simply return
            # => return all vertices of the graph as solution to avoid returning an invalid solution
            @debug "MIP solving: unknown error."
            # do not clear the current DFVS as it can contain also other vertices that are not part of the graph at hand
            # clear_dfvs!(solution)
            # there should be no overlap between the graph vertices and vertices already contained in the DFVS
            all_vertices = collect(vertices(solution.inst.graph))
            add_vertices_to_solution_new!(solution, all_vertices)

            return

            #= 
            # check for conflicts in the model and constraints
            compute_conflict!(model)
            if MOI.get(model, MOI.ConflictStatus()) != MOI.CONFLICT_FOUND
                error("No conflict could be found for an infeasible model.")
            end

            # collect conflicting constraints and print them
            conflict_constraint_list = ConstraintRef[]
            for (F, S) in list_of_constraint_types(model)
                for con in all_constraints(model, F, S)
                    if MOI.get(model, MOI.ConstraintConflictStatus(), con) == MOI.IN_CONFLICT
                        push!(conflict_constraint_list, con)
                        # println(con)
                    end
                end
            end

            error("The model was not solved correctly.")
            =#
        end

        # println("  objective value = ", objective_value(model))
        if primal_status(model) == MOI.FEASIBLE_POINT
            # -> large output for large graphs/models
            # println("  primal solution: y = ", value.(y))
        end
        if dual_status(model) == MOI.FEASIBLE_POINT
           # println("  dual solution: c1 = ", dual(c1))
        end
    end


    # debugging - start

    # why did the solver stop:
    # @info "Termination status: $(termination_status(model))"
    # println("Termination status: ", termination_status(model))
    # solver-specific string for reason:
    # @info "Solver-specific reason for termination: $(raw_status(model))"
    # println("Solver-specific reason for termination: ", raw_status(model))

    # show the objective value
    # @info "Objective value: $(objective_value(model))"
    # println("Objective value: ", objective_value(model))

    # get primal solution for vertices by broadcasting
    # -> large output for large graphs/models
    # println("Primal solution for vertices: ", value.(y))


    # debugging - end


    # build new solution from model

    # collect vertices for solution in a list
    solution_vertices = Vector{Int}()
    # all variables y_v with value 1 are part of the DAG => NOT part of the solution
    # all variables y_v with value 0 are not part of the DAG => part of the solution
    for i in 1:n_original
        # if (value(y[i]) == 0)   # TODO correct check for value 0 of a binary decision variable?
        if (isapprox(value(y[i]), 0, atol = 1e-1))  # check for value 0 with a certain tolerance
            push!(solution_vertices, i)   
        end
    end

    # alternative?
    # solution_vertices = [i for i in 1:n_original if (value(y[i]) == 0)]

    # add vertices to the solution
    add_vertices_to_solution_new!(solution, solution_vertices)


end # function mtz_formulation



"""
    mtz_formulation_reduced!(solution::DFVSPSolution)

Build and solve a MILP model for the DFVSP using the MTZ formulation.
Reduced model that does not need decision variables for edges.
Use the model to repair the given solution to be valid again.
"""
function mtz_formulation_reduced!(solution::DFVSPSolution)
    # choose which solver to use, options are "gurobi", "scip"
    solver_name = "scip"
    # solver_name = "gurobi"

    # get times and compute time limit for solver
    # also check termination criterion based on time
    # global_start_time = MyUtils.global_start_time
    # global_run_time_limit = MyUtils.global_run_time_limit

    # start with 60 seconds
    solver_time_limit = first_time_limit_mip_solver
    elapsed_time = time() - global_start_time

    if (global_run_time_limit < 0)
        # no global time limit set, so just continue
    elseif (elapsed_time >= global_run_time_limit)
        # run time limit already exceeded

        # TODO: return biggest solution possible (as done below) or just set solver time to 0? -> should be the same outcome?
        # => return all vertices of the graph as solution to avoid returning an invalid solution
        @debug "MIP solving: Global run time limit exceeded."
        # do not clear the current DFVS as it can contain also other vertices that are not part of the graph at hand
        # clear_dfvs!(solution)
        # there should be no overlap between the graph vertices and vertices already contained in the DFVS
        all_vertices = collect(vertices(solution.inst.graph))
        add_vertices_to_solution_new!(solution, all_vertices)

        return
    else
        remaining_time = (global_run_time_limit - elapsed_time)
        # set solver time limit to the smaller one of 60 seconds and the remaining time
        solver_time_limit = min(solver_time_limit, remaining_time)

        @debug "Remaining run time = $remaining_time"
        @debug "First solver time limit = $solver_time_limit"
    end

    graph = solution.inst.graph
    n_original = nv(graph)

    graph_with_source = SimpleDiGraph(graph)
    # add source vertex
    @assert add_vertex!(graph_with_source)
    n_source = nv(graph_with_source)

    # add edges from source vertex to all other vertices
    for i in 1:n_original
        @assert add_edge!(graph_with_source, n_source, i)
    end

    model = missing 

    if (solver_name == "scip")
        # model = Model(SCIP.Optimizer)
        # set max memory usage in MB, type = real, range: [0,8796093022207], default: 8796093022207]
        model = Model(optimizer_with_attributes(SCIP.Optimizer, "limits/memory" => 4000))
        # maximal time in seconds to run
        # -> [type: real, advanced: FALSE, range: [0,1e+20], default: 1e+20]
        set_optimizer_attribute(model, "limits/time", solver_time_limit)

        # verbosity level of output -> [type: int, advanced: FALSE, range: [0,5], default: 4]
        set_optimizer_attribute(model, "display/verblevel", 0)

    elseif (solver_name == "gurobi")
        # reuse the same Gurobi environment
        model = Model(() -> Gurobi.Optimizer(GRB_ENV))
        MOI.set(model, MOI.NumberOfThreads(), 1)

        # set attributes for Gurobi optimizer
        # TimeLimit: total time expended in seconds, takes type double, default = infinity
        set_optimizer_attribute(model, "TimeLimit", solver_time_limit)   
    end

    # model = Model(SCIP.Optimizer)
    # remove symmetry computation to hopefully avoid problems with bliss library -> works only sometimes
    # model = Model(optimizer_with_attributes(SCIP.Optimizer, "misc/usesymmetry" => 0))
    # model = Model(GLPK.Optimizer)
    # model = Model(Gurobi.Optimizer)

    # reuse the same Gurobi environment
    # model = Model(() -> Gurobi.Optimizer(GRB_ENV))
    # MOI.set(model, MOI.NumberOfThreads(), 1)

    # set attributes for Gurobi optimizer
    # TimeLimit: total time expended in seconds, takes type double, default = infinity
    # set_optimizer_attribute(model, "TimeLimit", solver_time_limit)    
    # MIPGap: relative MIP optimality gap between the lower and upper objective bound
    # -> terminate when gap is less than MIPGap times the absolute value of the incumbent objective value
    # type double, default value = 1e-4 = 0.0001 = 0.01%
    # Ex.: h_055 takes a long time (> 10325s) to go to 0.30% = 0.003
    # set_optimizer_attribute(model, "MIPGap", 0.005)
    # LogToConsole: control console logging, type int, default = 1, turn off = 0
    # set_optimizer_attribute(model, "LogToConsole", 0)
    # OutputFlag: control all logging (log to file, log to console), type int, default = 1, turn off = 0
    # set_optimizer_attribute(model, "OutputFlag", 0)

    # add variables:

    # variables for edges -> not needed anymore
    # but edge collection still needed
    edges_array = collect(edges(graph_with_source))
    # OLD: @variable(model, x[edges_array], Bin)
    # @variable(model, x[edges_array] >= 0.0)


    # variables for vertices
    @variable(model, y[1:n_source], Bin)
    # TODO correct or use constraint instead?
    # source vertex always has to be chosen
    fix(y[n_source], 1; force = true)
    # instead of: @constraint(model, select_source, y[n_source] == 1)

    # potential variables for vertices
    @variable(model, Φ[1:n_source] >= 0.0)
    # TODO correct or use constraint instead?
    # potential of source = 0
    fix(Φ[n_source], 0.0; force = true)
    # instead of: @constraint(model, source_potential_zero, Φ[n_source] == 0.0)


    # add constraints:

    # constraint for increasing potential values of vertices along selected arcs (excluding the source vertex)
    # OLD version: uses edge variable
    # @constraint(model, increasing_potential_value[e = edges_array], (Φ[src(e)] - Φ[dst(e)] + (n_source * x[e])) <= (n_source - 1))
    # NEW version: uses vertex variable of destination vertex instead of edge variable
    @constraint(model, increasing_potential_value[e = edges_array], (Φ[src(e)] - Φ[dst(e)] + (n_source * y[dst(e)])) <= (n_source - 1))
    
    # constraint for potential values of selected vertices -> JuMP needs 2 separate constraints
    # @constraint(model, potential_value[i=1:n_original], y[i] <= Φ[i] <= (n_source - 1))
    @constraint(model, potential_value_left_bound[i=1:n_original], y[i] <= Φ[i])
    @constraint(model, potential_value_right_bound[i=1:n_original], Φ[i] <= (n_source - 1))

    # constraints for selecting both endpoints for a selected arc
    # NEW: no edge variables => constraints not needed
    # @constraint(model, select_source[e = edges_array], x[e] <= y[src(e)])
    # @constraint(model, select_destination[e = edges_array], x[e] <= y[dst(e)])

    # constraint for selecting the incident arc of two selected vertices
    # NEW: no edge variables => constraints not needed
    # @constraint(model, select_arc[e = edges_array], x[e] >= (y[src(e)] + y[dst(e)] - 1))


    # add objective:

    # objective: maximize the selected vertices (belonging to the forest = opposite of DFVS) -> excluding the source vertex
    # => minimizes the DFVS (= the vertices that are not selected)
    @objective(model, Max, sum(y[i] for i in 1:n_original))

    # show model -> can be very big for large graphs
    # print(model)
    @debug "Finished building the model."
    # println("Finished building the model.")

    # start optimization
    optimize!(model)

    # show solution
    # println(solution_summary(model))

    # recommended workflow
    if termination_status(model) == MOI.OPTIMAL
        # println("Solution is optimal")
    elseif termination_status(model) == MOI.TIME_LIMIT && has_values(model)
        # println("Solution is suboptimal due to a time limit, but a primal solution is available")
    elseif termination_status(model) == MEMORY_LIMIT && has_values(model)
        # println("Solution is suboptimal due to a memory limit, but a primal solution is available")
    elseif termination_status(model) == MOI.INTERRUPTED && has_values(model)
        # println("Solution is suboptimal due to an interrupt, but a primal solution is available")

        # after an interrupt, we should terminate -> get solution and return

        # build new solution from model
        # collect vertices for solution in a list
        solution_vertices = Vector{Int}()
        # all variables y_v with value 1 are part of the DAG => NOT part of the solution
        # all variables y_v with value 0 are not part of the DAG => part of the solution
        for i in 1:n_original
            # if (value(y[i]) == 0)   # TODO correct check for value 0 of a binary decision variable?
            if (isapprox(value(y[i]), 0, atol = 1e-1))  # check for value 0 with a certain tolerance
                push!(solution_vertices, i)   
            end
        end

        # alternative?
        # solution_vertices = [i for i in 1:n_original if (value(y[i]) == 0)]

        # add vertices to the solution
        add_vertices_to_solution_new!(solution, solution_vertices)

        return
    else
        # println("No solution found.")

        # to avoid an error or crash, simply return
        # => return all vertices of the graph as solution to avoid returning an invalid solution
        @debug "MIP solving: unknown error."
        # do not clear the current DFVS as it can contain also other vertices that are not part of the graph at hand
        # clear_dfvs!(solution)
        # there should be no overlap between the graph vertices and vertices already contained in the DFVS
        all_vertices = collect(vertices(solution.inst.graph))
        add_vertices_to_solution_new!(solution, all_vertices)

        return

        #= 
        # check for conflicts in the model and constraints
        compute_conflict!(model)
        if MOI.get(model, MOI.ConflictStatus()) != MOI.CONFLICT_FOUND
            error("No conflict could be found for an infeasible model.")
        end

        # collect conflicting constraints and print them
        conflict_constraint_list = ConstraintRef[]
        for (F, S) in list_of_constraint_types(model)
            for con in all_constraints(model, F, S)
                if MOI.get(model, MOI.ConstraintConflictStatus(), con) == MOI.IN_CONFLICT
                    push!(conflict_constraint_list, con)
                    # println(con)
                end
            end
        end

        error("The model was not solved correctly.")
        =#
    end

    # println("  objective value = ", objective_value(model))
    if primal_status(model) == MOI.FEASIBLE_POINT
        # -> large output for large graphs/models
        # println("  primal solution: y = ", value.(y))
    end
    if dual_status(model) == MOI.FEASIBLE_POINT
       # println("  dual solution: c1 = ", dual(c1))
    end


    # EXPERIMENTAL section

    elapsed_time = time() - global_start_time

    # global run time limit is set and already exceeded
    if ((global_run_time_limit > 0) && (elapsed_time >= global_run_time_limit))
        # continue below

    elseif termination_status(model) == MOI.TIME_LIMIT
        # global run time limit not set or not exceeded yet
        # apply gap limit only if solving was terminated because of exceeded time limit; remove/increase time limit
        # @info "Time limit for solving was exceeded, now continue solving with gap limit and higher time limit"

        # TODO maybe do not completely remove time limit but set it to a higher value and still consider global run time limit
        # start with 300 seconds = 5 minutes
        solver_time_limit = second_time_limit_mip_solver

        if (global_run_time_limit < 0)
            # no global time limit set, so just continue
        else
            remaining_time = (global_run_time_limit - elapsed_time)
            # set solver time limit to the smaller one of 300 seconds and the remaining time
            solver_time_limit = min(solver_time_limit, remaining_time)

            @debug "Remaining run time = $remaining_time"
            @debug "Second solver time limit = $solver_time_limit"
        end

        if (solver_name == "scip")
            set_optimizer_attribute(model, "limits/time", solver_time_limit)
            # relative gap = |primal - dual|/MIN(|dual|,|primal|) is below the given value, the gap is 'Infinity', if primal and dual bound have opposite signs
            # -> [type: real, advanced: FALSE, range: [0,1.79769313486232e+308], default: 0]
            set_optimizer_attribute(model, "limits/gap", 0.005)
    
        elseif (solver_name == "gurobi")
            # set_optimizer_attribute(model, "TimeLimit", floatmax(Float64)) 
            set_optimizer_attribute(model, "TimeLimit", solver_time_limit) 
            set_optimizer_attribute(model, "MIPGap", 0.005)
        end

        # set_optimizer_attribute(model, "TimeLimit", floatmax(Float64)) 
        # set_optimizer_attribute(model, "TimeLimit", solver_time_limit) 
        # set_optimizer_attribute(model, "MIPGap", 0.005)

        # continue optimization
        optimize!(model)

        # show solution
        # println(solution_summary(model))

        # recommended workflow
        if termination_status(model) == MOI.OPTIMAL
            # println("Solution is optimal")
        elseif termination_status(model) == MOI.TIME_LIMIT && has_values(model)
            # println("Solution is suboptimal due to a time limit, but a primal solution is available")
        elseif termination_status(model) == MEMORY_LIMIT && has_values(model)
            # println("Solution is suboptimal due to a memory limit, but a primal solution is available")
        elseif termination_status(model) == MOI.INTERRUPTED && has_values(model)
            # println("Solution is suboptimal due to an interrupt, but a primal solution is available")

            # after an interrupt, we should terminate -> get solution and return

            # build new solution from model
            # collect vertices for solution in a list
            solution_vertices = Vector{Int}()
            # all variables y_v with value 1 are part of the DAG => NOT part of the solution
            # all variables y_v with value 0 are not part of the DAG => part of the solution
            for i in 1:n_original
                # if (value(y[i]) == 0)   # TODO correct check for value 0 of a binary decision variable?
                if (isapprox(value(y[i]), 0, atol = 1e-1))  # check for value 0 with a certain tolerance
                    push!(solution_vertices, i)   
                end
            end

            # alternative?
            # solution_vertices = [i for i in 1:n_original if (value(y[i]) == 0)]

            # add vertices to the solution
            add_vertices_to_solution_new!(solution, solution_vertices)

            return
        else
            # println("No solution found.")

            # to avoid an error or crash, simply return
            # => return all vertices of the graph as solution to avoid returning an invalid solution
            @debug "MIP solving: unknown error."
            # do not clear the current DFVS as it can contain also other vertices that are not part of the graph at hand
            # clear_dfvs!(solution)
            # there should be no overlap between the graph vertices and vertices already contained in the DFVS
            all_vertices = collect(vertices(solution.inst.graph))
            add_vertices_to_solution_new!(solution, all_vertices)

            return

            #= 
            # check for conflicts in the model and constraints
            compute_conflict!(model)
            if MOI.get(model, MOI.ConflictStatus()) != MOI.CONFLICT_FOUND
                error("No conflict could be found for an infeasible model.")
            end

            # collect conflicting constraints and print them
            conflict_constraint_list = ConstraintRef[]
            for (F, S) in list_of_constraint_types(model)
                for con in all_constraints(model, F, S)
                    if MOI.get(model, MOI.ConstraintConflictStatus(), con) == MOI.IN_CONFLICT
                        push!(conflict_constraint_list, con)
                        # println(con)
                    end
                end
            end

            error("The model was not solved correctly.")
            =#
        end

        # println("  objective value = ", objective_value(model))
        if primal_status(model) == MOI.FEASIBLE_POINT
            # -> large output for large graphs/models
            # println("  primal solution: y = ", value.(y))
        end
        if dual_status(model) == MOI.FEASIBLE_POINT
           # println("  dual solution: c1 = ", dual(c1))
        end
    end


    # debugging - start

    # why did the solver stop:
    # @info "Termination status: $(termination_status(model))"
    # println("Termination status: ", termination_status(model))
    # solver-specific string for reason:
    # @info "Solver-specific reason for termination: $(raw_status(model))"
    # println("Solver-specific reason for termination: ", raw_status(model))

    # show the objective value
    # @info "Objective value: $(objective_value(model))"
    # println("Objective value: ", objective_value(model))

    # get primal solution for vertices by broadcasting
    # -> large output for large graphs/models
    # println("Primal solution for vertices: ", value.(y))


    # debugging - end


    # build new solution from model

    # collect vertices for solution in a list
    solution_vertices = Vector{Int}()
    # all variables y_v with value 1 are part of the DAG => NOT part of the solution
    # all variables y_v with value 0 are not part of the DAG => part of the solution
    for i in 1:n_original
        # if (value(y[i]) == 0)   # TODO correct check for value 0 of a binary decision variable?
        if (isapprox(value(y[i]), 0, atol = 1e-1))  # check for value 0 with a certain tolerance
            push!(solution_vertices, i)   
        end
    end

    # alternative?
    # solution_vertices = [i for i in 1:n_original if (value(y[i]) == 0)]

    # add vertices to the solution
    add_vertices_to_solution_new!(solution, solution_vertices)


end # function mtz_formulation_reduced!



"""
    one_deletion_neighborhood_search_neighbor_multi_source_multi_dest_reachability_check_sorted_startindex_solution2!(solution::DFVSPSolution, start_index::Int) 

Apply a 1-deletion neighborhood to a given, valid solution for a DFVS problem.
This method uses first improvement as the step function.
This method goes through all vertices in the given solution one by one in increasing 
order of their heuristic value starting at the given `start_index` and tries to delete 
the current vertex from the DFVS (which is equivalent to adding the current vertex to 
the DAG). If the remaining DFVS is still a valid DFVS (which means that the new DAG 
is still a DAG), then this new solution is accepted and the method returns.

"""
function one_deletion_neighborhood_search_neighbor_multi_source_multi_dest_reachability_check_sorted_startindex_solution2!(solution::DFVSPSolution, start_index::Int)
    #= Steps
        - get the current DFVS = original labels of the vertices
        - get the current labels of the vertices in the DFVS
        - get the vertices of the current DAG 
            - using the graph and the current labels of the vertices in the DFVS
        - go through the current labels of the vertices in the DFVS in increasing order of their heuristic value
            - add the current vertex v to the DAG vertices => DAG'
            - get the subgraph induced by DAG'
            - check if there is a path from v to v in the subgraph
                - if yes: cycle introduced => no DAG anymore
                    => try the next vertex
                - if no: no cycle introduced => still DAG
                    => new, better solution found
                    - remove the vertex from the DFVS
                    - return the resulting solution
    =#
    
    current_sol = solution.x 
    original_graph = solution.inst.graph
    n_original_graph = nv(original_graph)

    # get current labels of vertices in solution.x 
    solution_current_labels = Set{Int}()
    for orig_label in current_sol
        current_label = solution.vmap_orig_new[orig_label]
        push!(solution_current_labels, current_label)
    end

    vertices_set = Set(vertices(original_graph))

    # get the vertices of the current DAG = vertices that are in the graph but not in the solution
    dag_vertices_set = setdiff(vertices_set, solution_current_labels)
    # convert set to array
    dag_vertices_array = collect(dag_vertices_set)

    # get DAG subgraph from graph based on the vertices in the solution (vertices must be in an array)
    subgraph_dag, vmap_dag_orig = induced_subgraph(original_graph, dag_vertices_array)

    # create vmap to map vertices from the original graph to the DAG subgraph
    # not all vertices of the original graph are contained in the DAG subgraph: these are given the value 0
    vmap_orig_dag = zeros(Integer, n_original_graph)
    for i in 1:length(vmap_dag_orig)
        orig_vertex = vmap_dag_orig[i]
        vmap_orig_dag[orig_vertex] = i
    end

    # go through solution vertices in increasing order of their heuristic value
    # build a dictionary of vertices => heuristic value
    dict_vertex_hvalue = Dict(i => compute_indegree_outdegree_difference(i, original_graph, 0.3) for i in solution_current_labels)

    # sort dictionary by its values (= heuristic values of vertices) in increasing order, output = array of tuples
    sorted = sort(collect(dict_vertex_hvalue), by=x->x[2])

    # get vertices in increasing order of their h value
    solution_current_labels_sorted = [sorted[i][1] for i in 1:length(sorted)]

    # check if start_index is still in range
    # if not, this indicates that all vertices have been checked and no more vertices can be added to the DAG
    if (start_index > length(solution_current_labels_sorted))
        return start_index
    end

    # go through vertices in sorted order 
    for i in start_index:length(solution_current_labels_sorted)
        dfvs_vertex = solution_current_labels_sorted[i]
        
        v_outneighbors = outneighbors(original_graph, dfvs_vertex)
        v_inneighbors = inneighbors(original_graph, dfvs_vertex)

        cycle_detected = false

        # get the labels of the inneighbors in the DAG
        in_dag_labels = [vmap_orig_dag[inneighbor] for inneighbor in v_inneighbors]
        # filter all inneighbors that are not contained in the DAG
        # TODO more efficient version possible with filter function?
        # in_dag_labels = [in_dag_labels[i] for i in 1:length(in_dag_labels) if in_dag_labels[i] != 0]
        # in_dag_labels = [i for i in in_dag_labels if i != 0]
        filter!(!iszero, in_dag_labels)
        # OR: more efficient?
        # in_dag_labels = [vmap_orig_dag[inneighbor] for inneighbor in v_inneighbors if vmap_orig_dag[inneighbor] != 0]

        out_dag_labels = [vmap_orig_dag[outneighbor] for outneighbor in v_outneighbors]
        filter!(!iszero, out_dag_labels)

        # cycle only possible if at least one inneighbor and at least one outneighbor is in the DAG
        if (!isempty(in_dag_labels) && !isempty(out_dag_labels))

            if (has_path_multi_source_multi_dest(subgraph_dag, out_dag_labels, in_dag_labels))
                cycle_detected = true
            end
            
        end

        # if no potential cycle was detected, we can remove the vertex from the solution and add it to the DAG
        if !cycle_detected

            # avoid having to map all the vertices again when adding them to the solution
            
            # convert array of original labels of the current solution to a set
            current_sol_set = Set(current_sol)
            # @info "Length of starting sol = $(length(current_sol_set))"
            # get the original label of the current vertex
            dfvs_vertex_orig_label = solution.vmap_new_orig[dfvs_vertex]
            # remove the current vertex from the solution
            new_solution_orig_labels_set = setdiff(current_sol_set, dfvs_vertex_orig_label)
            # @info "Length of new sol = $(length(new_solution_orig_labels_set))"
            # set the new solution
            solution.x = collect(new_solution_orig_labels_set)
            solution.sel = length(solution.x)
            sort_sel!(solution)
            # without calling invalidate!() the new solution is not recognized as better
            # above, this is implicitely done by calling clear_dfvs!()
            invalidate!(solution)
            # @info "Length of returned solution = $(length(solution.x))"
            # @info "Number of selected elements = $(solution.sel)"
            
            # return the index of the selected vertex
            return i
    
        end
    end

    # no suitable vertex found that can be removed from the DFVS and added to the DAG
    # => return the index of the last vertex
    return length(solution_current_labels_sorted)

end # function one_deletion_neighborhood_search_neighbor_multi_source_multi_dest_reachability_check_sorted_startindex_solution2!


"""
    local_search_one_deletion!(solution::DFVSPSolution, par::Float64, res::Result) 

Apply a local search to a given valid solution for the DFVS problem using a 1-deletion neighborhood.
This method uses first improvement as the step function.
Argument par: gives the time after which the local search is terminated.
    - if `par < 0`: no time limit set
Argument res is not used and only present to conform with calls used in MHLib.

This method does NOT change the original graph inside solution!

"""
function local_search_one_deletion!(solution::DFVSPSolution, par::Float64, res::Result)
    @debug "Starting LS"
    t_start = time()
    sol_ls = copy(solution)

    improvement_found = true
    # start search at first vertex
    start_index = 1

    # apply the move function until local optimum is found
    while improvement_found
        # start_index = one_deletion_neighborhood_search_neighbor_path_check_sorted_startindex_solution2!(sol_ls, start_index)
        # start_index = one_deletion_neighborhood_search_neighbor_single_source_multi_dest_reachability_check_sorted_startindex_solution2!(sol_ls, start_index)
        start_index = one_deletion_neighborhood_search_neighbor_multi_source_multi_dest_reachability_check_sorted_startindex_solution2!(sol_ls, start_index)

        # check if the new solution is better than the current one
        if is_better(sol_ls, solution)
            # @debug "LS: found better solution of length $(length(sol_ls.x))"
            # @debug "Current start_index = $start_index"
            # store the new, better solution
            copy!(solution, sol_ls)

        else
            # local optimum found => local search can terminate
            improvement_found = false
            
        end

        t_end = time()
        t_used = t_end - t_start
        # check for termination because of time limit
        if (par >= 0) && (t_used >= par)
            @debug "LS: time limit reached"
            break
        end

        # check termination criterion based on global time limit
        # global_start_time = MyUtils.global_start_time
        # global_run_time_limit = MyUtils.global_run_time_limit

        if (global_run_time_limit < 0)
            # no time limit set, so just continue
        elseif ((time()-global_start_time) >= global_run_time_limit)
            # global run time limit already exceeded
            # => break from loop
            @debug "LS: Global run time limit exceeded."
            break 
        end
    end

    # update global best solution
    # TODO: sufficient to do it at the end of the search? (should be okay if the LS time limit is lower than the general time limit)
    # otherwise this update should be done every time a better solution is found
    update_global_best_solution_current_part(solution.x)

#= 
    # DAG analysis and plotting
    analyse_dag(solution)
    instance_name = "h_199"
    # get current datetime
    df = Dates.DateFormat("dd-mm-yyyy_HH-MM-SS")
    current_datetime = Dates.now()
    formatted_datetime = Dates.format(current_datetime, df)
    plot_dag(solution, instance_name * "-" * formatted_datetime * "-afterLS")
 =#

    @debug "Finished LS"
    
end # function local_search_one_deletion!



"""
    check_lns_termination(sol::DFVSPSolution)

Check for termination based on the run time.
If the run time exceeds the limit, set all vertices of the graph as solution to avoid returning an invalid solution.

"""
function check_lns_termination(sol::DFVSPSolution)

    # check termination criterion based on time

    if (global_run_time_limit < 0)
        # no time limit set, so just continue
        return false
    elseif ((time()-global_start_time) >= global_run_time_limit)
        # run time limit already exceeded
        # => return all vertices of the graph as solution to avoid returning an invalid solution (from the destroy method)

        @debug "LNS: Run time limit exceeded."
        # clear the current and possibly invalid DFVS
        clear_dfvs!(sol)
        all_vertices = collect(vertices(sol.inst.graph))
        add_vertices_to_solution_new!(sol, all_vertices)

        return true
    end

    return false

end # function check_lns_termination


"""
    destroy_dag_tournament_selection_fixed_size_smallest_fixed_k!(sol::DFVSPSolution, (par, k)::Tuple{Int, Int}, result::Result)

Destroy operator for LNS selects par*ALNS.get_number_to_destroy elements according to the indegree-outdegree heuristic for removal from solution. 
Tournament selection is used to as the selection method using parameter `k` to determine the tournament size.
    - `k` gives the number of elements to be used in the tournament
    - tournament_size = k
In each tournament, the element with the smallest heuristic value is selected.

Should be used together with `repair_dag_mtz_formulation!()`.

"""
function destroy_dag_tournament_selection_fixed_size_smallest_fixed_k!(sol::DFVSPSolution, (par, k)::Tuple{Int, Int}, result::Result)
    current_sol = sol.x
    current_sol_length = length(current_sol)
    # get_number_to_destroy selects a number based on the given number of elements in the solution and the ratio given in the ALNS settings
    num_to_destroy = min(get_number_to_destroy(current_sol_length) * par, current_sol_length)
    
    # build a dictionary of solution-vertices => heuristic value (using the current label of the vertices in the heuristic)
    dict_vertex_hvalue = Dict(i => compute_indegree_outdegree_difference(sol.vmap_orig_new[i], sol.inst.graph) for i in current_sol)

    selected_vertices_set = Set{Int}()
    population = Set(copy(current_sol))
    tournament_size = k 

    # use tournament selection to choose the vertices
    for i in 1:num_to_destroy

        # ensure that the tournament size is not bigger than the population
        # this is unlikely but could be the case after deleting the selected vertices in previous tournaments
        if tournament_size > length(population)
            # reduce tournament_size until sampling is possible
            while (tournament_size > length(population)) 
                tournament_size -= 1
            end
        end

        # select tournament_size many vertices as candidates
        # TODO? this will throw an error if tournament_size is negative (which should not happen unless the population is empty)
        # sample() works with arrays not sets, so population has to be converted
        tournament_selection = sample(collect(population), tournament_size, replace=false)
        # println("size of tournament: ", length(tournament_selection))
        # sort the vertices by their heuristic value in increasing order
        tournament_selection_sorted = sort(tournament_selection, by= x -> dict_vertex_hvalue[x])
        # select the first vertex = the vertex with the smallest heuristic value
        selected_vertex = first(tournament_selection_sorted)
        # add the selected vertex to the set
        push!(selected_vertices_set, selected_vertex)
        # remove the selected vertex from the population to avoid multiple selections of a single vertex
        population_length_old = length(population)
        delete!(population, selected_vertex)
        population_length_new = length(population)

        # ensure correct deletion of vertex from population
        @assert population_length_new < population_length_old
        
    end

    current_sol_set = Set(current_sol)

    # actually delete the selected elements from the solution
    new_sol_set = setdiff(current_sol_set, selected_vertices_set)
    sol.x = sort!(collect(new_sol_set))
    sol.sel -= num_to_destroy
    # objective value has changed, solution is probably invalid
    invalidate!(sol)

    # debugging - start

    @debug "Destroy method: length of starting solution = $current_sol_length"
    @debug "Destroy method: number of destroyed elements = $num_to_destroy"
    @debug "Destroy method: length of resulting solution = $(length(sol.x))"
    @debug "Destroy method: number of selected elements in solution = $(sol.sel)"
    # println("")
    # println("Destroy method: length of starting solution = ", current_sol_length)
    # println("Destroy method: number of destroyed elements = ", num_to_destroy)
    # println("Destroy method: length of resulting solution = ", length(sol.x))
    # println("Destroy method: number of selected elements in solution = ", sol.sel)
    # println("")

    # debugging - end

end # function destroy_dag_tournament_selection_fixed_size_smallest_fixed_k!


"""
    repair_dag_mtz_formulation!(sol::DFVSPSolution, par::Int, result::Result)

Repair operator for LNS uses the MTZ MILP formulation to solve the DFVS subproblem on the graph resulting from the previous destroy method.

Should be used together with `destroy_dag_random_fixed_k!()`.

"""
function repair_dag_mtz_formulation!(sol::DFVSPSolution, par::Int, result::Result)
    # check for termination
    if (check_lns_termination(sol))
        return
    end
    
    current_sol = sol.x
    current_sol_length = length(current_sol)

    solution_copy = copy(sol)   # only copies reference to instance with graph
    graph_copy = SimpleDiGraph(sol.inst.graph)
    solution_copy.inst = DFVSPInstance(graph_copy)
    # solution_copy.inst.graph = graph_copy

    # remove vertices that are currently part of the solution from the graph
    # x (= solution) contains the original labels of the vertices -> mapping has to be used
    new_labels_to_remove = Vector{Int}()
    for orig_label in solution_copy.x
        new_label = solution_copy.vmap_orig_new[orig_label]
        push!(new_labels_to_remove, new_label)
    end

    reverse_vmap = rem_vertices!(graph_copy, new_labels_to_remove)

    # update vertex mappings
    # create temporary mapping of correct new length
    # the old mapping in solution is needed for the update and cannot be changed directly
    # temp_vmap_new_orig = Vector{Integer}(length(reverse_vmap))
    temp_vmap_new_orig = zeros(Integer, length(reverse_vmap))
    for i in 1:length(temp_vmap_new_orig)
        # update original label
        original_label = solution_copy.vmap_new_orig[reverse_vmap[i]]
        temp_vmap_new_orig[i] = original_label
        # update new label
        solution_copy.vmap_orig_new[original_label] = i
    end
    # store new mapping in solution
    solution_copy.vmap_new_orig = copy(temp_vmap_new_orig)
    
    # call model with mtz formulation of DFVSP
    mtz_formulation!(solution_copy)

    # update the solution
    sol.x = copy(solution_copy.x)
    sol.sel = solution_copy.sel

    # objective value has changed, solution should be valid
    invalidate!(sol)

    # check validity of solution
    # check(sol)

    # update global best solution
    update_global_best_solution_current_part(sol.x)

    # debugging - start

    @debug "Repair method: length of starting solution = $current_sol_length"
    @debug "Repair method: length of resulting solution = $(length(sol.x))"
    @debug "Repair method: number of selected elements in solution = $(sol.sel)"
    # println("")
    # println("Repair method: length of starting solution = ", current_sol_length)
    # println("Repair method: length of resulting solution = ", length(sol.x))
    # println("Repair method: number of selected elements in solution = ", sol.sel)
    # println("")

    # debugging - end

end # function repair_dag_mtz_formulation!



"""
    repair_dag_reduction_rules_mtz_formulation!(sol::DFVSPSolution, par::Int, result::Result)

Repair operator for LNS uses the MTZ MILP formulation to solve the DFVS subproblem on the graph resulting from the previous destroy method.

Should be used together with `destroy_dag_random_fixed_k!()`.

"""
function repair_dag_reduction_rules_mtz_formulation!(sol::DFVSPSolution, par::Int, result::Result)
    # check for termination
    if (check_lns_termination(sol))
        return
    end
    
    current_sol = sol.x
    current_sol_length = length(current_sol)

    solution_copy = copy(sol)   # only copies reference to instance with graph
    graph_copy = SimpleDiGraph(sol.inst.graph)
    solution_copy.inst = DFVSPInstance(graph_copy)
    # solution_copy.inst.graph = graph_copy

    # remove vertices that are currently part of the solution from the graph
    # x (= solution) contains the original labels of the vertices -> mapping has to be used
    new_labels_to_remove = Vector{Int}()
    for orig_label in solution_copy.x
        new_label = solution_copy.vmap_orig_new[orig_label]
        push!(new_labels_to_remove, new_label)
    end

    reverse_vmap = rem_vertices!(graph_copy, new_labels_to_remove)

    # update vertex mappings
    # create temporary mapping of correct new length
    # the old mapping in solution is needed for the update and cannot be changed directly
    # temp_vmap_new_orig = Vector{Integer}(length(reverse_vmap))
    temp_vmap_new_orig = zeros(Integer, length(reverse_vmap))
    for i in 1:length(temp_vmap_new_orig)
        # update original label
        original_label = solution_copy.vmap_new_orig[reverse_vmap[i]]
        temp_vmap_new_orig[i] = original_label
        # update new label
        solution_copy.vmap_orig_new[original_label] = i
    end
    # store new mapping in solution
    solution_copy.vmap_new_orig = copy(temp_vmap_new_orig)

    # empty out the rr solution set
    solution_copy.dfvs_rr = Vector{Int}()
    # apply reduction rules to enlarged DAG, but do not consider any further splitting into SCCs
    # println("Vertex number of enlarged DAG before RRs: $(nv(graph_copy))")
    # println("Edge number of enlarged DAG before RRs: $(ne(graph_copy))")
    
    apply_reduction_rules_new!(solution_copy)
    rr_solution = copy(solution_copy.dfvs_rr)

    # println("Vertex number of enlarge DAG after RRs: $(nv(graph_copy))")
    # println("Edge number of enlarged DAG before RRs: $(ne(graph_copy))")
    @debug "Solution from reduction rules for enlarged DAG: $rr_solution"
    # println("solution from reduction rules: $rr_solution")
    
    # call model with mtz formulation of DFVSP
    mtz_formulation!(solution_copy)

    # build new solution from MILP solution and reduction rules solution
    new_solution = copy(solution_copy.x)
    append!(new_solution, rr_solution)

    # update the solution
    sol.x = copy(new_solution)
    sol.sel = (solution_copy.sel + length(rr_solution))

    # objective value has changed, solution should be valid
    invalidate!(sol)

    # check validity of solution
    # check(sol)

    # update global best solution
    update_global_best_solution_current_part(sol.x)

    # debugging - start

    @debug "Repair method: length of starting solution = $current_sol_length"
    @debug "Repair method: length of resulting solution = $(length(sol.x))"
    @debug "Repair method: number of selected elements in solution = $(sol.sel)"
    # println("")
    # println("Repair method: length of starting solution = ", current_sol_length)
    # println("Repair method: length of resulting solution = ", length(sol.x))
    # println("Repair method: number of selected elements in solution = ", sol.sel)
    # println("")

    # debugging - end

end # function repair_dag_reduction_rules_mtz_formulation!



"""
    repair_dag_reduction_rules_mtz_formulation_reduced!(sol::DFVSPSolution, par::Int, result::Result)

Repair operator for LNS uses the reduced MTZ MILP formulation to solve the DFVS subproblem on the graph resulting from the previous destroy method.

Should be used together with `destroy_dag_random_fixed_k!()`.

"""
function repair_dag_reduction_rules_mtz_formulation_reduced!(sol::DFVSPSolution, par::Int, result::Result)
    # check for termination
    if (check_lns_termination(sol))
        return
    end
    
    current_sol = sol.x
    current_sol_length = length(current_sol)

    solution_copy = copy(sol)   # only copies reference to instance with graph
    graph_copy = SimpleDiGraph(sol.inst.graph)
    solution_copy.inst = DFVSPInstance(graph_copy)
    # solution_copy.inst.graph = graph_copy

    # remove vertices that are currently part of the solution from the graph
    # x (= solution) contains the original labels of the vertices -> mapping has to be used
    new_labels_to_remove = Vector{Int}()
    for orig_label in solution_copy.x
        new_label = solution_copy.vmap_orig_new[orig_label]
        push!(new_labels_to_remove, new_label)
    end

    reverse_vmap = rem_vertices!(graph_copy, new_labels_to_remove)

    # update vertex mappings
    # create temporary mapping of correct new length
    # the old mapping in solution is needed for the update and cannot be changed directly
    # temp_vmap_new_orig = Vector{Integer}(length(reverse_vmap))
    temp_vmap_new_orig = zeros(Integer, length(reverse_vmap))
    for i in 1:length(temp_vmap_new_orig)
        # update original label
        original_label = solution_copy.vmap_new_orig[reverse_vmap[i]]
        temp_vmap_new_orig[i] = original_label
        # update new label
        solution_copy.vmap_orig_new[original_label] = i
    end
    # store new mapping in solution
    solution_copy.vmap_new_orig = copy(temp_vmap_new_orig)

    # empty out the rr solution set
    solution_copy.dfvs_rr = Vector{Int}()
    # apply reduction rules to enlarged DAG, but do not consider any further splitting into SCCs
    # println("Vertex number of enlarged DAG before RRs: $(nv(graph_copy))")
    # println("Edge number of enlarged DAG before RRs: $(ne(graph_copy))")
    @debug "Vertex number of enlarged DAG before RRs: $(nv(graph_copy))"
    @debug "Edge number of enlarged DAG before RRs: $(ne(graph_copy))"

    apply_reduction_rules_new!(solution_copy)
    rr_solution = copy(solution_copy.dfvs_rr)

    # println("Vertex number of enlarge DAG after RRs: $(nv(graph_copy))")
    # println("Edge number of enlarged DAG before RRs: $(ne(graph_copy))")
    @debug "Vertex number of enlarge DAG after RRs: $(nv(graph_copy))"
    @debug "Edge number of enlarged DAG before RRs: $(ne(graph_copy))"
    @debug "Solution from reduction rules for enlarged DAG: $rr_solution"
    # println("solution from reduction rules: $rr_solution")
    
    # call model with mtz formulation of DFVSP
    mtz_formulation_reduced!(solution_copy)

    # build new solution from MILP solution and reduction rules solution
    new_solution = copy(solution_copy.x)
    append!(new_solution, rr_solution)

    # update the solution
    sol.x = copy(new_solution)
    sol.sel = (solution_copy.sel + length(rr_solution))

    # objective value has changed, solution should be valid
    invalidate!(sol)

    # check validity of solution
    # check(sol)

    # update global best solution
    update_global_best_solution_current_part(sol.x)

    # debugging - start

    @debug "Repair method: length of starting solution = $current_sol_length"
    @debug "Repair method: length of resulting solution = $(length(sol.x))"
    @debug "Repair method: number of selected elements in solution = $(sol.sel)"
    # println("")
    # println("Repair method: length of starting solution = ", current_sol_length)
    # println("Repair method: length of resulting solution = ", length(sol.x))
    # println("Repair method: number of selected elements in solution = ", sol.sel)
    # println("")

    # debugging - end

end # function repair_dag_reduction_rules_mtz_formulation_reduced!



"""
    destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!(sol::DFVSPSolution, (par, k)::Tuple{Int, Int}, result::Result)

Destroy operator for LNS selects par*ALNS.get_number_to_destroy elements according to the indegree-outdegree heuristic for 
removal from the given directed acyclic graph (DAG).
These elements are then added to only a part of the DFVS. 
This partial destruction is actually done in the corresponding repair method.

Tournament selection is used to as the selection method using parameter `k` to determine the tournament size.
    - `k` gives the number of elements to be used in the tournament
    - tournament_size = k
In each tournament, the element with the HIGHEST heuristic value is selected.

Should be used together with `repair_dfvs_partially_mtz_formulation_using_transitive_closure_and_selfloop_removal!()`.

"""
function destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!(sol::DFVSPSolution, (par, k)::Tuple{Int, Int}, result::Result)
    current_sol = sol.x
    current_sol_length = length(current_sol)
    current_sol_sel = sol.sel
    # number of vertices in the graph
    n = nv(sol.inst.graph)
    # number of vertices in the current DAG
    n_dag = n - current_sol_length
    # get_number_to_destroy selects a number based on the given number of elements in the DAG and the ratio given in the ALNS settings
    num_to_destroy = min(get_number_to_destroy(n_dag) * par, n_dag)

    # get the current labels of the vertices in the solution
    solution_current_labels = Set{Int}()
    for original_label in current_sol
        current_label = sol.vmap_orig_new[original_label]
        push!(solution_current_labels, current_label)
    end

    vertices_set = Set(vertices(sol.inst.graph))

    # every vertex in the solution should also be in the current graph
    @assert issubset(solution_current_labels, vertices_set)

    # get the vertices of the current DAG = vertices that are in the graph but not in the solution
    dag_vertices_set = setdiff(vertices_set, solution_current_labels)
    # convert set to array to be used in sample()
    dag_vertices_array = collect(dag_vertices_set)

    # build a dictionary of DAG-vertices => heuristic value (using the current label of the vertices in the heuristic)
    dict_vertex_hvalue = Dict(i => compute_indegree_outdegree_difference(i, sol.inst.graph) for i in dag_vertices_array)

    selected_vertices = Vector{Int}()
    population = copy(dag_vertices_set)
    tournament_size = k

    # use tournament selection to choose the vertices
    for i in 1:num_to_destroy

        # ensure that the tournament size is not bigger than the population
        # this is unlikely but could be the case after deleting the selected vertices in previous tournaments
        if tournament_size > length(population)
            # reduce tournament_size until sampling is possible
            while (tournament_size > length(population)) 
                tournament_size -= 1
            end
        end

        # select tournament_size many vertices as candidates
        # TODO? this will throw an error if tournament_size is negative (which should not happen unless the population is empty)
        # sample() works with arrays not sets, so population has to be converted
        tournament_selection = sample(collect(population), tournament_size, replace=false)
        # println("size of tournament: ", length(tournament_selection))
        # sort the vertices by their heuristic value in decreasing order
        tournament_selection_sorted = sort(tournament_selection, by= x -> dict_vertex_hvalue[x], rev=true)
        # select the first vertex = the vertex with the highest heuristic value
        selected_vertex = first(tournament_selection_sorted)
        # add the selected vertex to the set
        push!(selected_vertices, selected_vertex)
        # remove the selected vertex from the population to avoid multiple selections of a single vertex
        population_length_old = length(population)
        delete!(population, selected_vertex)
        population_length_new = length(population)

        # ensure correct deletion of vertex from population
        @assert population_length_new < population_length_old
        
    end

    # add the selected elements to the solution (= removal from DAG), solution is still valid
    add_vertices_to_solution_new!(sol, selected_vertices)
    # objective value has changed, but solution is still valid
    invalidate!(sol)
    # reset the number of selected elements in the solution to enable the partial destruction
    sol.sel = current_sol_sel

    # debugging - start

    @debug "Destroy method: length of starting solution = $current_sol_length"
    @debug "Destroy method: number of destroyed elements = $num_to_destroy"
    @debug "Destroy method: length of resulting solution = $(length(current_sol))"
    @debug "Destroy method: number of selected elements in solution = $(sol.sel)"
    # println("")
    # println("Destroy method: length of starting solution = ", current_sol_length)
    # println("Destroy method: number of destroyed elements = ", num_to_destroy)
    # println("Destroy method: length of resulting solution = ", length(current_sol))
    # println("Destroy method: number of selected elements in solution = ", sol.sel)
    # println("")

    # debugging - end


end # function destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!


"""
    repair_dfvs_partially_tournament_selection_fixed_size_smallest_mtz_formulation_reduced_using_has_path_multi_src_multi_dest_and_selfloop_removal!(sol::DFVSPSolution, (par, k)::Tuple{Int, Int}, result::Result)

Repair operator for LNS uses the reduced MTZ MILP formulation to solve the DFVS subproblem on the DFVS resulting from the previous destroy method.
Reduces runtime by exploiting properties of self-loops with regards to the DFVS problem.
Removes vertices with self-loops immediately and adds them to the solution.
Does reachability checks using a modified version of `has_path()` for multiple sources and multiple destinations.
Reduces the runtime even further by only solving a subproblem of the DFVS problem:
    - only a fixed-size part of the current DFVS (DFVS') is selected
        - this size is determined by the parameter `par`
        - the selection is done using tournament selection with a fixed size for the tournament size
            - this size is determined by the parameter `k` 
            - the element with the smallest heuristic value is selected
    - the elements selected in the destroy method are added to DFVS'
    - the subgraph induced by DFVS' is then constructed according to the neighborhood rule
        - vertices with (potential) self-loops are added to the solution and removed from the graph
        - edges between the remaining vertices are added according to the edge condition 

Should be used together with `destroy_dfvs_partially_random_fixed_k!()`.

"""
function repair_dfvs_partially_tournament_selection_fixed_size_smallest_mtz_formulation_reduced_using_has_path_multi_src_multi_dest_and_selfloop_removal!(sol::DFVSPSolution, (par, k)::Tuple{Int, Int}, result::Result)
    #= STEPS:
    - get DFVS subgraph from graph based on vertices in sol.x
        - for this: get current labels of vertices in sol.x
        - [sol.x[1]; sol.x[sol.sel]] = DFVS (original DFVS)
        - [sol.x[sol.sel + 1]; sol.x[end]] = added elements
        -> select the first `par` elements from DFVS => DFVS'
            - TODO: possible alternatives for selection: random selection, heuristic selection, tournament selection
        - get subgraph for DFVS'
    - get DAG subgraph from graph based on current DFVS
    - compute vertex mappings from graph to DFVS and DAG subgraph
    - add missing edges to the DFVS' subgraph based on the neighborhood rule using the transitive closure of the DAG for reachability checks
        -> then new DFVS has to be found for this subgraph   
        -> MILP model has to be called with this subgraph
    => build new instance + solution with this subgraph
        - solution.x still contains the elements from DFVS that were not selected for DFVS' and also the vertices with self-loops
    =#

    # check for termination
    if (check_lns_termination(sol))
        return
    end

    original_graph = sol.inst.graph
    n_original_graph = nv(original_graph)
    current_sol = sol.x
    current_sol_length = length(current_sol)

    dfvs_orig_labels_array = sol.x[begin:sol.sel]
    # dfvs_orig_labels_set = Set(dfvs_orig_labels_array)
    added_vertices_orig_labels_array = sol.x[(sol.sel + 1):end]
    # added_vertices_orig_labels_set = Set(added_vertices_orig_labels_array)

    # get current labels of vertices in sol.x 
    solution_current_labels = Set{Int}()
    dfvs_current_labels = Set{Int}()
    added_vertices_current_labels = Set{Int}()

    for orig_label in dfvs_orig_labels_array
        current_label = sol.vmap_orig_new[orig_label]
        # @assert current_label != 0
        # @assert !in(current_label, dfvs_current_labels)
        push!(solution_current_labels, current_label)
        push!(dfvs_current_labels, current_label)
    end

    for orig_label in added_vertices_orig_labels_array
        current_label = sol.vmap_orig_new[orig_label]
        # @assert current_label != 0
        push!(solution_current_labels, current_label)
        push!(added_vertices_current_labels, current_label)
    end

    vertices_set = Set(vertices(original_graph))

    # every vertex in the solution should also be in the current graph
    # @assert issubset(solution_current_labels, vertices_set)
    # number of current labels of solution vertices should be equal to number of vertices in the solution
    # @debug "Current sol length = $current_sol_length"
    # @debug "Solution current labels length = $(length(solution_current_labels))"
    # @debug "DFVS size = $(length(dfvs_orig_labels_array))"
    # @debug "Number of added vertices = $(length(added_vertices_orig_labels_array))"
    # @debug "DFVS current labels size = $(length(dfvs_current_labels))"
    # @debug "Number of added vertices, current labels = $(length(added_vertices_current_labels))"
    # @assert isempty(Base.intersect(dfvs_current_labels, added_vertices_current_labels))
    # @assert isdisjoint(dfvs_current_labels, added_vertices_current_labels)
    # @assert isempty(Base.intersect(dfvs_orig_labels_set, added_vertices_orig_labels_set))
    # @assert isdisjoint(dfvs_orig_labels_set, added_vertices_orig_labels_set)
    # @assert current_sol_length == length(solution_current_labels)

    # get the vertices of the current DAG = vertices that are in the graph but not in the solution
    dag_vertices_set = setdiff(vertices_set, solution_current_labels)
    # convert set to array to be used in sample()
    dag_vertices_array = collect(dag_vertices_set)

    # get DAG subgraph from graph based on the vertices in the solution (vertices must be in an array)
    subgraph_dag, vmap_dag_orig = induced_subgraph(sol.inst.graph, dag_vertices_array)

    # create vmap to map vertices from the original graph to the DAG subgraph
    # not all vertices of the original graph are contained in the DAG subgraph: these are given the value 0
    vmap_orig_dag = zeros(Integer, n_original_graph)
    for i in 1:length(vmap_dag_orig)
        orig_vertex = vmap_dag_orig[i]
        vmap_orig_dag[orig_vertex] = i
    end


    partial_dfvs_size = par
    # calculate suitable size for DFVS'
    if (par > sol.sel)
        # given size is bigger than current DFVS, so it has to be reduced
        partial_dfvs_size = sol.sel
    end


    # get DFVS' subgraph from graph based on the current labels of the vertices in the DFVS (vertices must be in an array)
    dfvs_current_labels_array = collect(dfvs_current_labels)

    # use tournament selection to select `partial_dfvs_size` many elements from the solution for DFVS'
    selected_vertices = Vector{Int}()
    population = copy(dfvs_current_labels)
    tournament_size = k

    # build a dictionary of DFVS-vertices => heuristic value (using the current label of the vertices in the heuristic)
    dict_vertex_hvalue = Dict(i => compute_indegree_outdegree_difference(i, original_graph) for i in dfvs_current_labels_array)

    # use tournament selection to choose the vertices
    for i in 1:partial_dfvs_size

        # ensure that the tournament size is not bigger than the population
        # this is unlikely but could be the case after deleting the selected vertices in previous tournaments
        if tournament_size > length(population)
            # reduce tournament_size until sampling is possible
            while (tournament_size > length(population)) 
                tournament_size -= 1
            end
        end

        # select tournament_size many vertices as candidates
        # TODO? this will throw an error if tournament_size is negative (which should not happen unless the population is empty)
        # sample() works with arrays not sets, so population has to be converted
        tournament_selection = sample(collect(population), tournament_size, replace=false)
        # println("size of tournament: ", length(tournament_selection))
        # sort the vertices by their heuristic value in increasing order
        tournament_selection_sorted = sort(tournament_selection, by= x -> dict_vertex_hvalue[x])
        # select the first vertex = the vertex with the smallest heuristic value
        selected_vertex = first(tournament_selection_sorted)
        # add the selected vertex to the set
        push!(selected_vertices, selected_vertex)
        # remove the selected vertex from the population to avoid multiple selections of a single vertex
        # population_length_old = length(population)
        delete!(population, selected_vertex)
        # population_length_new = length(population)

        # ensure correct deletion of vertex from population
        # @assert population_length_new < population_length_old
        
    end

    partial_dfvs_current_labels_array = collect(selected_vertices)
    # get the current labels of the remaining vertices of the DFVS
    # remaining vertices = unselected vertices of the population, what is left in the population
    remaining_dfvs_vertices_current_labels_array = collect(population)

    # get the original labels of the remaining vertices from the DFVS
    # in the end, these vertices will be added to the new solution
    remaining_dfvs_vertices_orig_labels_array = Vector{Int}()
    for current_label in remaining_dfvs_vertices_current_labels_array
        orig_label = sol.vmap_new_orig[current_label]
        push!(remaining_dfvs_vertices_orig_labels_array, orig_label)
    end

    # @assert isdisjoint(Set(remaining_dfvs_vertices_current_labels_array), Set(partial_dfvs_current_labels_array))

    # combine the partial DFVS with the added vertices to get the new subproblem
    partial_dfvs_with_added_vertices_array = Vector{Int}()
    append!(partial_dfvs_with_added_vertices_array, partial_dfvs_current_labels_array)
    append!(partial_dfvs_with_added_vertices_array, collect(added_vertices_current_labels))

    subgraph_dfvs, vmap_dfvs_orig = induced_subgraph(original_graph, partial_dfvs_with_added_vertices_array)

    # @debug "Length of partial DFVS: $(length(partial_dfvs_current_labels_array))"
    # @debug "Number of added elements: $(length(added_vertices_current_labels))"
    # @debug "Vertex number of new subproblem: $(length(partial_dfvs_with_added_vertices_array))"
    # @debug "Vertex number of new subgraph: $(nv(subgraph_dfvs))"
    # @assert nv(subgraph_dfvs) == length(partial_dfvs_with_added_vertices_array)


    # create a new instance and solution with the DFVS subgraph
    subgraph_dfvs_instance = DFVSPInstance(subgraph_dfvs)
    subgraph_dfvs_solution = DFVSPSolution(subgraph_dfvs_instance)

    # set up the vertex mappings
    # subgraph_dfvs_solution.vmap_orig_new = zeros(Int, length(sol.vmap_orig_new))
    subgraph_dfvs_solution.vmap_orig_new = Dict{Int, Int}()
    
    # update the mappings according to the new mapping from the induced subgraph
    # create temporary mapping of correct new length
    # the old mapping in solution is needed for the update and cannot be changed directly
    temp_vmap_new_orig = zeros(Integer, length(vmap_dfvs_orig))
    for i in 1:length(temp_vmap_new_orig)
        # update original label
        original_label = sol.vmap_new_orig[vmap_dfvs_orig[i]]
        temp_vmap_new_orig[i] = original_label
        # update new label
        subgraph_dfvs_solution.vmap_orig_new[original_label] = i
    end
    # store new mapping in solution
    subgraph_dfvs_solution.vmap_new_orig = copy(temp_vmap_new_orig)

    # reset the current solution to empty
    clear_dfvs!(subgraph_dfvs_solution)

    # add missing edges to build the graph suitable for the MILP model
    # E'' = (E ∩ S'²) ∪ {vw | v, w ∈ S', an in-neighbor of w is reachable from an outneighbor of v in G[D']}.
   
    # @info "Repair method: Starting graph construction."
    # println("Repair method: Starting graph construction.")

    n_subgraph_dfvs = nv(subgraph_dfvs)

    num_added_edges = 0

    self_loops = Vector{Int}()

    # find all vertices that already have or would get self-loops
    for v in 1:n_subgraph_dfvs
        # if edge is already there, add vertex to vertices with self-loops
        if (has_edge(subgraph_dfvs, v, v))
            push!(self_loops, v)  
        end

        # check for termination
        if (check_lns_termination(sol))
            return
        end

        # get in- and outneighbors from original_graph while using the mapping to get the original vertex label
        # check whether neighbors are in DAG / not in DFVS (using the mapping to get the corresponding vertex labels)
        
        # if neighbors are in DAG, then check if there is a path between them
        v_orig_label = vmap_dfvs_orig[v]

        v_outneighbors = outneighbors(original_graph, v_orig_label)
        v_inneighbors = inneighbors(original_graph, v_orig_label)

        # get the labels of the inneighbors in the DAG
        in_dag_labels = [vmap_orig_dag[inneighbor] for inneighbor in v_inneighbors]
        # filter all inneighbors that are not contained in the DAG
        filter!(!iszero, in_dag_labels)

        # get the labels of the outneighbors in the DAG
        out_dag_labels = [vmap_orig_dag[outneighbor] for outneighbor in v_outneighbors]
        # filter all outneighbors that are not contained in the DAG
        filter!(!iszero, out_dag_labels)

        # cycle only possible if at least one inneighbor and at least one outneighbor is in the DAG
        if (!isempty(in_dag_labels) && !isempty(out_dag_labels))

            # check whether any inneighbor is reachable from any outneighbor => there is a path from out to in in the DAG subgraph
            # if there is a path, add the vertex to the vertices with self-loops
            if (has_path_multi_source_multi_dest(subgraph_dag, out_dag_labels, in_dag_labels))
                push!(self_loops, v)
            end
            
        end

    end

    # add vertices with (potential) self-loops to the solution
    add_vertices_to_solution_new!(subgraph_dfvs_solution, self_loops)
    # set new label of these vertices to 0 because they will be removed
    for vertex in self_loops
        original_label = subgraph_dfvs_solution.vmap_new_orig[vertex]
        subgraph_dfvs_solution.vmap_orig_new[original_label] = 0
    end

    # remove vertices and incident edges from the graph
    reverse_vmap_self_loops = rem_vertices!(subgraph_dfvs, self_loops)

    # update vertex mappings

    # create temporary mapping of correct new length
    # the old mapping in solution is needed for the update and cannot be changed directly
    temp_vmap_new_orig = zeros(Integer, length(reverse_vmap_self_loops))
    temp_vmap_dfvs_orig = zeros(Integer, length(reverse_vmap_self_loops))
    for i in 1:length(temp_vmap_new_orig)
        # update original label
        original_label = subgraph_dfvs_solution.vmap_new_orig[reverse_vmap_self_loops[i]]
        temp_vmap_new_orig[i] = original_label
        # update new label
        subgraph_dfvs_solution.vmap_orig_new[original_label] = i
        # update mapping from dfvs subgraph to original graph of given solution 
        temp_vmap_dfvs_orig[i] = vmap_dfvs_orig[reverse_vmap_self_loops[i]]
    end
    # store new mapping in solution
    subgraph_dfvs_solution.vmap_new_orig = copy(temp_vmap_new_orig)
    vmap_dfvs_orig = copy(temp_vmap_dfvs_orig)

    # create vmap to map vertices from the original graph to the DFVS subgraph
    # not all vertices of the original graph are contained in the DFVS subgraph: these are given the value 0
    vmap_orig_dfvs = zeros(Integer, n_original_graph)
    for i in 1:length(vmap_dfvs_orig)
        orig_vertex = vmap_dfvs_orig[i]
        vmap_orig_dfvs[orig_vertex] = i
    end

    #= 
    # debug: ensure correct behaviour
    if (!isempty(self_loops))
        @assert n_subgraph_dfvs > nv(subgraph_dfvs)
        @assert n_subgraph_dfvs > nv(subgraph_dfvs_solution.inst.graph)
        @assert !has_self_loops(subgraph_dfvs)
    end
     =#

    # @info "Repair method - graph construction: finished self-loop checks."
    # @info "Number of found vertices with (potential) self-loops: $(length(self_loops))"

    # update vertex number
    n_subgraph_dfvs = nv(subgraph_dfvs)

    # adding edges according to neighborhood edge condition:
    # only add edges if involved vertices do not have self-loops as these vertices have to be in the DFVS anyways
    for v in 1:n_subgraph_dfvs
        for w in 1:n_subgraph_dfvs

            # check for termination
            if (check_lns_termination(sol))
                return
            end

            # if v = w -> case of self-loop already checked above, so move on the next pair of vertices 
            if (v == w)
                continue
            end

            # if edge is already there or any involved vertex has a self-loop, move on to the next pair of vertices
            if (has_edge(subgraph_dfvs, v, w))
                continue  
            end

            # get in- and outneighbors from original_graph while using the mapping to get the original vertex label
            # check whether neighbors are in DAG / not in DFVS (using the mapping to get the corresponding vertex labels)
            
            # if neighbors are in DAG, then check if there is a path between them
            v_orig_label = vmap_dfvs_orig[v]
            w_orig_label = vmap_dfvs_orig[w]

            v_outneighbors = outneighbors(original_graph, v_orig_label)
            w_inneighbors = inneighbors(original_graph, w_orig_label)

            # get the labels of the outneighbors of v in the DAG
            out_dag_labels = [vmap_orig_dag[outneighbor] for outneighbor in v_outneighbors]
            # filter all outneighbors that are not contained in the DAG
            filter!(!iszero, out_dag_labels)

            # get the labels of the inneighbors of w in the DAG
            in_dag_labels = [vmap_orig_dag[inneighbor] for inneighbor in w_inneighbors]
            # filter all inneighbors that are not contained in the DAG
            filter!(!iszero, in_dag_labels)

            # check whether any inneighbor is reachable from any outneighbor => there is a path from out to in in the DAG subgraph
            # if there is a path, add an edge from v to w in the DFVS subgraph
            if (!isempty(in_dag_labels) && !isempty(out_dag_labels))
            
                if (has_path_multi_source_multi_dest(subgraph_dag, out_dag_labels, in_dag_labels))
                    # edge cannot already exist according to check above
                    @assert add_edge!(subgraph_dfvs, v, w)
                    num_added_edges += 1
                end
                
            end

        end
        
    end

    # @info "Number of added edges: $num_added_edges"
    # @info "Repair method: Finished graph construction."
    # println("Repair method: Finished graph construction.")

    # @info "Repair method: finished preprocessing."
    # println("Repair method: finished preprocessing.")

    # call model with mtz formulation of DFVSP
    mtz_formulation_reduced!(subgraph_dfvs_solution)

    # update the solution
    sol.x = copy(subgraph_dfvs_solution.x)
    sol.sel = subgraph_dfvs_solution.sel
    # add the vertices from the original DFVS that were not selected for the subproblem to the solution
    # @debug "Forbidden overlap between solution and remaining vertices: $(Base.intersect(Set(sol.x), Set(remaining_dfvs_vertices_orig_labels_array)))"
    # @assert isdisjoint(Set(sol.x), Set(remaining_dfvs_vertices_orig_labels_array))
    append!(sol.x, remaining_dfvs_vertices_orig_labels_array)
    sol.sel += length(remaining_dfvs_vertices_orig_labels_array)

    # objective value has changed, solution should be valid
    invalidate!(sol)

    # debugging - start

    @debug "Repair method: length of starting solution = $current_sol_length"
    @debug "Repair method: length of resulting solution = $(length(sol.x))"
    @debug "Repair method: number of selected elements in solution = $(sol.sel)"
    # println("")
    # println("Repair method: length of starting solution = ", current_sol_length)
    # println("Repair method: length of resulting solution = ", length(sol.x))
    # println("Repair method: number of selected elements in solution = ", sol.sel)
    # println("")

    # debugging - end

end # function repair_dfvs_partially_tournament_selection_fixed_size_smallest_mtz_formulation_reduced_using_has_path_multi_src_multi_dest_and_selfloop_removal!


"""
    process_start_time()

Return time when the current Julia process was started.
"""
function process_start_time()
    @assert Sys.islinux()
    local boot_time
    for line in eachline("/proc/stat")
        if startswith(line, "btime")
            boot_time = parse(Int, split(line)[2])
            break
        end
    end
    rel_start_time = parse(Int, split(read("/proc/self/stat", String) )[22]) / 100
    return boot_time + rel_start_time
end

process_running_time() = time() - process_start_time()


function isdfvs(g::DiGraph, dfvs::Vector{Int})
    gcopy = DiGraph(g)
    rem_vertices!(gcopy, dfvs)
    return length(strongly_connected_components(gcopy)) == nv(gcopy)
end

function main()
    # enable debug output
    # debuglogger = SimpleLogger(Logging.Debug)
    # global_logger(debuglogger)

    # start_time = time()
    start_time = process_start_time()
    set_global_start_time(start_time)    
    # start_time = get_start_time()
    # start_time = MyUtils.global_start_time
    @debug "Start time = $(global_start_time)"
    # println(start_time)

    # set allowed runtime in seconds
    current_run_time = process_running_time()
    run_time_limit = floor(Int, 560 - current_run_time)
    mh_run_time_limit = run_time_limit
    set_global_run_time_limit(run_time_limit)
    @debug "Global run time limit = $(global_run_time_limit)"
    # println(MyUtils.global_run_time_limit)

    # set time limits for MIP solver
    set_first_time_limit_mip_solver(60.0)
    set_second_time_limit_mip_solver(90.0)

    Random.seed!(1)
    # suppress output from MHLib
    # orig_stdout = stdout  # TODO: not necessary because global variable in MyUtils is initialized with "stdout"?
    redirect_stdout(devnull)

    # register atexit hook -> function to be called when exiting after SIGTERM
    atexit(print_gobal_best_solution)
    
    # read command line arguments
    arguments = ARGS
    # for x in arguments
    #     println("Argument: $x")
    # end

    # instance_directory = "instances/"
    # instance_directory = arguments[1]

    # instance_name = "h_199"
    # instance_name = "simple_instance_2"
    # instance_name = arguments[2]

    instance_file = ""

    if !isempty(arguments)
        instance_file = arguments[1]
    end
    

    # dfvsp_instance = DFVSPInstance(arguments[1])
    # dfvsp_instance = DFVSPInstance("instances/h_001")
    # dfvsp_instance = DFVSPInstance("instances/simple_instance_2")

    # dfvsp_instance = DFVSPInstance(instance_directory * instance_name)
    dfvsp_instance = DFVSPInstance(instance_file)

    dfvsp_solution = DFVSPSolution(dfvsp_instance)
    # println(dfvsp_solution)

    # visualize the original graph
    # plot = plot_graph(dfvsp_instance.graph)
    # draw(SVG(instance_name * "-graph.svg", 16cm, 16cm), plot)

    # apply reduction rules and get strongly connected components (SCCs) of the remaining graph
    # sccs = apply_reduction_rules!(dfvsp_solution)
    # new implementation with vertex mappings
    sccs = apply_reduction_rules_new!(dfvsp_solution)

    # update the global best solution
    if (!isempty(dfvsp_solution.dfvs_rr))
        update_global_best_solution_add_part(dfvsp_solution.dfvs_rr)
    end

    # debugging - start

    # @info "After reduction rules: vertex number = $(nv(dfvsp_solution.inst.graph))"
    # @info "After reduction rules: edge number = $(ne(dfvsp_solution.inst.graph))"
    # println("After reduction rules: vertex number = ", nv(dfvsp_solution.inst.graph))
    # println("After reduction rules: edge number = ", ne(dfvsp_solution.inst.graph))

    # @info "After reduction rules: number of SCCs = $(length(sccs))"

    # println("Size of reverse_vmaps_list: ", size(dfvsp_solution.reverse_vmaps_list))
    # println("Length of reverse_vmaps_list: ", length(dfvsp_solution.reverse_vmaps_list))
    # for list in dfvsp_solution.reverse_vmaps_list
    #     println("Length of list element: ", length(list))
    # end

    #=
    for i in 1:length(dfvsp_solution.reverse_vmaps_list)
        for j in 1:length(dfvsp_solution.reverse_vmaps_list[i])
            println("$i-th reduction: Vertex $j used to be vertex ", dfvsp_solution.reverse_vmaps_list[i][j])
        end
    end
    =#

    # println("Preliminary DFVS from reduction rules: ", sort(dfvsp_solution.dfvs_rr))
    # @info "Length of preliminary DFVS from reduction rules: $(length(dfvsp_solution.dfvs_rr))"
    # println("Length of preliminary DFVS from reduction rules: ", length(dfvsp_solution.dfvs_rr))

    # debugging - end

    # get strongly connected components (SCCs) of the remaining graph -> done above by calling the reduction rules
    # sccs = reduce_graph_using_strongly_connected_components!(dfvsp_solution)
    # TODO apply reduction rules again? apply SCC rule inside other reduction rules iteration? -> currently applied inside rr iteration

    # debugging - start
#= 
    println("After SCCs: vertex number = ", nv(dfvsp_solution.inst.graph))
    println("After SCCs: edge number = ", ne(dfvsp_solution.inst.graph))

    println("Size of reverse_vmaps_list: ", size(dfvsp_solution.reverse_vmaps_list))
    println("Length of reverse_vmaps_list: ", length(dfvsp_solution.reverse_vmaps_list))

    println("Preliminary DFVS from reduction rules + SCCs: ", sort(dfvsp_solution.dfvs_rr))
    println("Length of preliminary DFVS from reduction rules + SCCs: ", length(dfvsp_solution.dfvs_rr))
 =#
    # debugging - end

    #=
        LNS
    =#

    # TODO: configure settings
    # general settings: seed(?), mh_ttime [s] (Float, <0 to turn off), mh_titer (iterations, Int, <0 to turn off)
    # "--mh_tciter" = "maximum number of iterations without improvement (<0: turned off)", Int, default = -1
    # "--mh_checkit" = "call `check` for each solution after each method application", Bool, default = false
    # "--mh_lnewinc" = "always write iteration log if new incumbent solution", Bool, default = true
    #= (A)LNS settings: 
        --alns_dest_min_abs: Int, min num of elements to destroy, default = 5
        --alns_dest_max_abs: Int, max num of elements to destroy, default = 100
        --alns_dest_min_ratio: Float64, min ratio of elements to destroy, default = 0.1
        --alns_dest_max_ratio: Float64, max ratio of elements to destroy, default = 0.35
        --alns_init_temp_factor: Float64, factor for determining initial temperature, default = 0.0
        --alns_temp_dec_factor: Float64, factor for decreasing the temperature, default = 0.99
    =#


    # using default settings: no time limit, iterations = 100
    # parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.ALNSs.settings_cfg], 
    # ["--seed=1"])

    # no time limit, limited number of iterations = 10 (+1 for LS)
    # parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.ALNSs.settings_cfg],
    # ["--seed=1", "--mh_titer=11"])

    # no time limit, limited number of iterations = 10 (+1 for LS), changed max number to destroy from 100 to 1000
    # parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.ALNSs.settings_cfg],
    # ["--seed=1", "--mh_titer=11", "--alns_dest_max_abs=1000"])

    
    # reduce run time limit for MHLib by the already elapsed time to solve discrepancy between my timer and MHLib timer
    # there is still is difference as my time limit is exceeded earlier than MHLibs
    # but this might be preferable to getting a timeout and terminating without a valid solution
    if (run_time_limit > 0)
        elapsed_time = ceil(Int, (time() - global_start_time))
        mh_run_time_limit = Int(run_time_limit - elapsed_time)
        @debug "Run time limit for MHLib: $mh_run_time_limit"
    end 

    # time limit, limited number of iterations = 10 (+1 for LS)
    # parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.ALNSs.settings_cfg],
    # ["--seed=1", "--mh_titer=11", "--mh_ttime=" * string(mh_run_time_limit)])  

    # time limit, unlimited number of iterations, changed max number to destroy from 100 to 1000
    # limited number of iterations without improvement = 20
    parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.ALNSs.settings_cfg],
    ["--seed=1", "--mh_lnewinc=false", "--mh_titer=-1", "--mh_tciter=20", "--mh_ttime=" * string(mh_run_time_limit), "--alns_dest_max_abs=1000"])

    # time limit, unlimited number of iterations, changed max number to destroy from 100 to 1000
    # limited number of iterations without improvement = 10
    # check validity of solution after each method
    # parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.ALNSs.settings_cfg],
    # ["--seed=1", "--mh_titer=-1", "--mh_tciter=10", "--mh_ttime=" * string(mh_run_time_limit), "--alns_dest_max_abs=1000", "--mh_checkit=true"])

    tournament_sel_percentage = 0.025
    tournament_sel_size = 3
    partial_dfvs_size = 3000
    local_search_time_limit = 60.0

    # no time limit, limited number of iterations = 1
    # parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.ALNSs.settings_cfg], 
    # ["--seed=1", "--mh_titer=1"])

    # time limit = 10 min, iteration limit turned off
    # parse_settings!([MHLib.Schedulers.settings_cfg, MHLib.ALNSs.settings_cfg], 
    # ["--seed=1", "--mh_ttime=600.0", "--mh_titer=-1"])


    final_solution_vector = Vector{Int}()
    compound_solution = Vector{Int}()

    # if multiple SCCs found, create a DFVSP Instance + solution for each subgraph induced by a SCC; otherwise the graph in solution can be used directely
    if (length(sccs) > 1)
        reduced_sccs = 0

        # sort SCCs by increasing size
        sorted_sccs = sort(sccs, by=length)

        # if too many SCCs are found, put some smaller ones together into a package to reduce memory consumption
        # then create a DFVSP Instance + solution for each subgraph induced by the vertices of a package
        if (length(sccs) > 1000)
            # put SCCs together until they reach 100 vertices -> limit to be solved directly with MIP model
            packaged_sccs = Vector{Vector{Int}}()
            # initialize first position
            push!(packaged_sccs, Vector{Int}())
            scc_index = 1

            # go through all SCCs and put them together
            # this also creates sorted packages with increasing size
            for scc in sorted_sccs
                # check whether the SCC can be added to the current package without exceeding 100 vertices
                if ((length(packaged_sccs[scc_index]) + length(scc)) <= 100)
                    append!(packaged_sccs[scc_index], scc)
                else
                    # adding the SCC would surpass the limit of 100 vertices => start next package 
                    push!(packaged_sccs, Vector{Int}())
                    scc_index += 1
                    append!(packaged_sccs[scc_index], scc)
                end

            end

            @debug "Number of SCC packages = $(length(packaged_sccs))"
            @debug "Min SCC package length = $(minimum(length, packaged_sccs))"
            @debug "Max SCC package length = $(maximum(length, packaged_sccs))"

            # now point sorted_sccs to the sorted packages
            sorted_sccs = packaged_sccs
        end

        @debug "Number of sorted SCCs = $(length(sorted_sccs))"

        for scc in sorted_sccs
            # get induced subgraph and vertex mapping
            subgraph, vmap = induced_subgraph(dfvsp_solution.inst.graph, scc)

            # create a new instance and solution
            subgraph_instance = DFVSPInstance(subgraph)
            subgraph_solution = DFVSPSolution(subgraph_instance)

            # set up the vertex mappings
            # subgraph_solution.vmap_orig_new = copy(dfvsp_solution.vmap_orig_new)
            # subgraph_solution.vmap_orig_new = zeros(Int, length(dfvsp_solution.vmap_orig_new))
            # subgraph_solution.vmap_new_orig = copy(dfvsp_solution.vmap_new_orig)
            subgraph_solution.vmap_orig_new = Dict{Int, Int}()

            # update the mappings according to the new mapping from the induced subgraph
            scc_orig_labels = Vector{Int}()
            # create temporary mapping of correct new length
            # the old mapping in solution is needed for the update and cannot be changed directly
            temp_vmap_new_orig = zeros(Integer, length(vmap))
            for i in 1:length(temp_vmap_new_orig)
                # update original label
                original_label = dfvsp_solution.vmap_new_orig[vmap[i]]
                # push!(scc_orig_labels, original_label)
                temp_vmap_new_orig[i] = original_label
                # update new label
                subgraph_solution.vmap_orig_new[original_label] = i
            end
            # store new mapping in solution
            subgraph_solution.vmap_new_orig = copy(temp_vmap_new_orig)

            @debug "Before SCC reduction rules: vertex number = $(nv(subgraph_solution.inst.graph))"
            @debug "Before reduction rules: edge number = $(ne(subgraph_solution.inst.graph))"
            orig_vn = nv(subgraph_solution.inst.graph)

            # apply reduction rules again to each SCC, but do not consider any further splitting into SCCs
            apply_reduction_rules_new!(subgraph_solution)
            # add solution from reduction rules to solutions from subproblems
            append!(compound_solution, subgraph_solution.dfvs_rr)
            # update the global best solution
            if (!isempty(subgraph_solution.dfvs_rr))
                update_global_best_solution_add_part(subgraph_solution.dfvs_rr)
            end
        
            @debug "After SCC reduction rules: vertex number = $(nv(subgraph_solution.inst.graph))"
            @debug "After reduction rules: edge number = $(ne(subgraph_solution.inst.graph))"
            # @assert nv(subgraph_solution.inst.graph) == nv(subgraph)

            new_vn = nv(subgraph_solution.inst.graph)
            if orig_vn != new_vn
                @debug "Successfully reduced SCC"
                reduced_sccs += 1
            end

            if nv(subgraph) == 3 && ne(subgraph) == 6
                # directly add 2 of the 3 vertices to the solution
                append!(compound_solution, [subgraph_solution.vmap_new_orig[v] for v in 1:2])
                # update the global best solution
                update_global_best_solution_add_part([subgraph_solution.vmap_new_orig[v] for v in 1:2])

            elseif nv(subgraph) <= 100

                # first reset the current (=default) solution to empty as no valid solution that should be extended is currently known
                clear_dfvs!(subgraph_solution)
                # directly solve the problem with the MIP model, do not use LNS
                # mtz_formulation!(subgraph_solution)
                mtz_formulation_reduced!(subgraph_solution)
                append!(compound_solution, subgraph_solution.x)
                # update the global best solution
                update_global_best_solution_add_part(subgraph_solution.x)

            else


                # call construction heuristic for subgraph_solution
                # based on cycle checks:
                # construction_heuristic_indegree_outdegree_difference_new!(subgraph_solution, 0.3, Result())
                # based on topological ordering properties:
                # construction_heuristic_indegree_outdegree_difference_alternative!(subgraph_solution, 0.3, Result())

                @debug "Starting SCC of length = $(length(scc))"
                #println("")
                #println("Starting SCC of length = ", length(scc))
                #println("")

                # build and call LNS algorithm
                # -> call only possible with working destroy and repair methods

                # LNS with enlarge DAG neighborhood
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord", construction_heuristic_indegree_outdegree_difference_alternative!, 0.3)],
                #     [MHMethod("destroy_dag_random_fixed_k", destroy_dag_random_fixed_k!, 1)],
                #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

                # LNS with enlarge DAG neighborhood and heuristic element selection -> decreasing order
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord", construction_heuristic_indegree_outdegree_difference_alternative!, 0.3)],
                #     [MHMethod("destroy_dag_heuristic_selection_decreasing_fixed_k", destroy_dag_heuristic_selection_decreasing_fixed_k!, 1)],
                #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

                # LNS with enlarge DAG neighborhood and heuristic element selection -> increasing order
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord", construction_heuristic_indegree_outdegree_difference_alternative!, 0.3)],
                #     [MHMethod("destroy_dag_heuristic_selection_increasing_fixed_k", destroy_dag_heuristic_selection_increasing_fixed_k!, 1)],
                #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

                # using CH with topo ord + neighbor check, LNS with enlarge DAG neighborhood and random element selection
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)],
                #     [MHMethod("destroy_dag_random_fixed_k", destroy_dag_random_fixed_k!, 1)],
                #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

                # using CH with topo ord + neighbor check, LNS with enlarge DAG neighborhood and tournament selection -> smallest hvalue
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)],
                #     [MHMethod("destroy_dag_tournament_selection_smallest_fixed_k", destroy_dag_tournament_selection_smallest_fixed_k!, (1, tournament_sel_percentage))],
                #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

                # using CH with topo ord + neighbor check, LNS with enlarge DAG neighborhood and tournament selection with fixed size -> smallest hvalue
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)],
                #     [MHMethod("destroy_dag_tournament_selection_fixed_size_smallest_fixed_k", destroy_dag_tournament_selection_fixed_size_smallest_fixed_k!, (1, tournament_sel_size))],
                #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

                # using CH with topo ord + IMPROVED neighbor check and subsequent local search mit multi-src multi-dest path, LNS with enlarge DAG neighborhood and tournament selection with fixed size -> smallest hvalue
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
                #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
                #     [MHMethod("destroy_dag_tournament_selection_fixed_size_smallest_fixed_k", destroy_dag_tournament_selection_fixed_size_smallest_fixed_k!, (1, tournament_sel_size))],
                #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

                # using CH with topo ord + IMPROVED neighbor check and subsequent local search mit multi-src multi-dest path, LNS with enlarge DAG neighborhood and tournament selection with fixed size -> smallest hvalue 
                # and application of reduction rules to enlarged DAG
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
                #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
                #     [MHMethod("destroy_dag_tournament_selection_fixed_size_smallest_fixed_k", destroy_dag_tournament_selection_fixed_size_smallest_fixed_k!, (1, tournament_sel_size))],
                #     [MHMethod("repair_dag_rrs_mtz", repair_dag_reduction_rules_mtz_formulation!, 0)])

                # using CH with topo ord + IMPROVED neighbor check and subsequent local search mit multi-src multi-dest path, LNS with enlarge DAG neighborhood and tournament selection with fixed size -> smallest hvalue 
                # and application of reduction rules to enlarged DAG, reduced MTZ model
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
                #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
                #     [MHMethod("destroy_dag_tournament_selection_fixed_size_smallest_fixed_k", destroy_dag_tournament_selection_fixed_size_smallest_fixed_k!, (1, tournament_sel_size))],
                #     [MHMethod("repair_dag_rrs_mtz_reduced", repair_dag_reduction_rules_mtz_formulation_reduced!, 0)])

                # using CH with topo ord + neighbor check, LNS with enlarge DAG neighborhood and tournament selection -> highest hvalue
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)],
                #     [MHMethod("destroy_dag_tournament_selection_highest_fixed_k", destroy_dag_tournament_selection_highest_fixed_k!, (1, tournament_sel_percentage))],
                #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

                # LNS with enlarge DFVS neighborhood
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord", construction_heuristic_indegree_outdegree_difference_alternative!, 0.3)],
                #     [MHMethod("destroy_dfvs_random_fixed_k", destroy_dfvs_random_fixed_k!, 1)],
                #     [MHMethod("repair_dfvs_mtz", repair_dfvs_mtz_formulation!, 0)])

                # using CH with topo ord + neighbor check, LNS with enlarge DFVS neighborhood using transitive closure computation
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)],
                #     [MHMethod("destroy_dfvs_random_fixed_k", destroy_dfvs_random_fixed_k!, 1)],
                #     [MHMethod("repair_dfvs_mtz_trans_closure", repair_dfvs_mtz_formulation_alternative_transitive_closure!, 0)])

                # using CH with topo ord + neighbor check, LNS with enlarge DFVS neighborhood using transitive closure computation and self-loop checks
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)],
                #     [MHMethod("destroy_dfvs_random_fixed_k", destroy_dfvs_random_fixed_k!, 1)],
                #     [MHMethod("repair_dfvs_mtz_trans_closure_selfloops", repair_dfvs_mtz_formulation_using_transitive_closure_and_selfloops!, 0)])

                # using CH with topo ord + neighbor check, LNS with enlarge DFVS neighborhood using transitive closure computation and self-loop removal
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)],
                #     [MHMethod("destroy_dfvs_random_fixed_k", destroy_dfvs_random_fixed_k!, 1)],
                #     [MHMethod("repair_dfvs_mtz_trans_closure_selfloop_removal", repair_dfvs_mtz_formulation_using_transitive_closure_and_selfloop_removal!, 0)])

                # using CH with topo ord + IMPROVED neighbor check and subsequent local search, LNS with enlarge DFVS neighborhood using transitive closure computation and self-loop removal
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
                #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
                #     [MHMethod("destroy_dfvs_random_fixed_k", destroy_dfvs_random_fixed_k!, 1)],
                #     [MHMethod("repair_dfvs_mtz_trans_closure_selfloop_removal", repair_dfvs_mtz_formulation_using_transitive_closure_and_selfloop_removal!, 0)])

                # using CH with topo ord + IMPROVED neighbor check and subsequent local search, LNS with enlarge DFVS neighborhood using has_path and self-loop removal
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
                #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
                #     [MHMethod("destroy_dfvs_random_fixed_k", destroy_dfvs_random_fixed_k!, 1)],
                #     [MHMethod("repair_dfvs_mtz_path_selfloop_removal", repair_dfvs_mtz_formulation_using_has_path_and_selfloop_removal!, 0)])

                # using CH with topo ord + neighbor check, LNS with enlarge DFVS neighborhood using transitive closure computation, self-loop removal and partial destroy/repair
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)],
                #     [MHMethod("destroy_dfvs_partially_random_fixed_k", destroy_dfvs_partially_random_fixed_k!, 1)],
                #     [MHMethod("repair_dfvs_partially_mtz_trans_closure_selfloop_removal", repair_dfvs_partially_mtz_formulation_using_transitive_closure_and_selfloop_removal!, partial_dfvs_size)])

                # using CH with topo ord + IMPROVED neighbor check and subsequent local search, LNS with enlarge DFVS neighborhood using transitive closure computation, self-loop removal 
                # and partial destroy/repair with tournament selection with fixed size (destroy: highest value, repair: highest value)
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
                #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
                #     [MHMethod("destroy_dfvs_partially_ts_fixed_size_highest_fixed_k", destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!, (1, tournament_sel_size))],
                #     [MHMethod("repair_dfvs_partially_ts_fixed_size_highest_mtz_trans_closure_selfloop_removal", repair_dfvs_partially_tournament_selection_fixed_size_highest_mtz_formulation_using_transitive_closure_and_selfloop_removal!, (partial_dfvs_size, tournament_sel_size))])

                # using CH with topo ord + IMPROVED neighbor check and subsequent local search, LNS with enlarge DFVS neighborhood using transitive closure computation, self-loop removal 
                # and partial destroy/repair with tournament selection with fixed size (destroy: highest value, repair: smallest value)
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
                #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
                #     [MHMethod("destroy_dfvs_partially_ts_fixed_size_highest_fixed_k", destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!, (1, tournament_sel_size))],
                #     [MHMethod("repair_dfvs_partially_ts_fixed_size_smallest_mtz_trans_closure_selfloop_removal", repair_dfvs_partially_tournament_selection_fixed_size_smallest_mtz_formulation_using_transitive_closure_and_selfloop_removal!, (partial_dfvs_size, tournament_sel_size))])

                # using CH with topo ord + IMPROVED neighbor check and subsequent local search, LNS with enlarge DFVS neighborhood using has_path, self-loop removal 
                # and partial destroy/repair with tournament selection with fixed size (destroy: highest value, repair: highest value)
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
                #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
                #     [MHMethod("destroy_dfvs_partially_ts_fixed_size_highest_fixed_k", destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!, (1, tournament_sel_size))],
                #     [MHMethod("repair_dfvs_partially_ts_fixed_size_highest_mtz_path_selfloop_removal", repair_dfvs_partially_tournament_selection_fixed_size_highest_mtz_formulation_using_has_path_and_selfloop_removal!, (partial_dfvs_size, tournament_sel_size))])

                # using CH with topo ord + IMPROVED neighbor check and subsequent local search, LNS with enlarge DFVS neighborhood using has_path, self-loop removal 
                # and partial destroy/repair with tournament selection with fixed size (destroy: highest value, repair: smallest value)
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
                #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
                #     [MHMethod("destroy_dfvs_partially_ts_fixed_size_highest_fixed_k", destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!, (1, tournament_sel_size))],
                #     [MHMethod("repair_dfvs_partially_ts_fixed_size_smallest_mtz_path_selfloop_removal", repair_dfvs_partially_tournament_selection_fixed_size_smallest_mtz_formulation_using_has_path_and_selfloop_removal!, (partial_dfvs_size, tournament_sel_size))])

                # using CH with topo ord + IMPROVED neighbor check and subsequent local search mit multi-src multi-dest path, LNS with enlarge DFVS neighborhood using has_path multi-src multi-dest, self-loop removal 
                # and partial destroy/repair with tournament selection with fixed size (destroy: highest value, repair: smallest value)
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
                #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
                #     [MHMethod("destroy_dfvs_partially_ts_fixed_size_highest_fixed_k", destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!, (1, tournament_sel_size))],
                #     [MHMethod("repair_dfvs_partially_ts_fixed_size_smallest_mtz_path_multi_src_multi_dest_selfloop_removal", repair_dfvs_partially_tournament_selection_fixed_size_smallest_mtz_formulation_using_has_path_multi_src_multi_dest_and_selfloop_removal!, (partial_dfvs_size, tournament_sel_size))])

                # using CH with topo ord + IMPROVED neighbor check and subsequent local search mit multi-src multi-dest path, LNS with enlarge DFVS neighborhood using has_path multi-src multi-dest, self-loop removal, REDUCED MTZ formulation 
                # and partial destroy/repair with tournament selection with fixed size (destroy: highest value, repair: smallest value)
                lns_alg = ALNS(subgraph_solution,
                    [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
                        MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
                    [MHMethod("destroy_dfvs_partially_ts_fixed_size_highest_fixed_k", destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!, (1, tournament_sel_size))],
                    [MHMethod("repair_dfvs_partially_ts_fixed_size_smallest_mtz_reduced_path_multi_src_multi_dest_selfloop_removal", repair_dfvs_partially_tournament_selection_fixed_size_smallest_mtz_formulation_reduced_using_has_path_multi_src_multi_dest_and_selfloop_removal!, (partial_dfvs_size, tournament_sel_size))])

                # using only CH with cycle check, dummy methods for destroy + repair
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_cycle_check", construction_heuristic_indegree_outdegree_difference_new!, 0.3)],
                #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
                #     [MHMethod("repair_test", repair_test!, 0)])

                # using only CH with topo ord, dummy methods for destroy + repair
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord", construction_heuristic_indegree_outdegree_difference_alternative!, 0.3)],
                #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
                #     [MHMethod("repair_test", repair_test!, 0)])

                # using only CH with topo ord + neighbor check, dummy methods for destroy + repair
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)],
                #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
                #     [MHMethod("repair_test", repair_test!, 0)])

                # using only CH with topo ord + IMPROVED neighbor check, dummy methods for destroy + repair
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3)],
                #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
                #     [MHMethod("repair_test", repair_test!, 0)])

                # using only CH with topo ord + IMPROVED neighbor check + transitive closure, dummy methods for destroy + repair
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_trans_closure", construction_heuristic_indegree_outdegree_difference_alternative_transitive_closure!, 0.3)],
                #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
                #     [MHMethod("repair_test", repair_test!, 0)])

                # using only CH with topo ord + IMPROVED neighbor check + has_path check, dummy methods for destroy + repair
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_path_check", construction_heuristic_indegree_outdegree_difference_alternative_path_check!, 0.3)],
                #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
                #     [MHMethod("repair_test", repair_test!, 0)])

                # using only CH with topo ord + neighbor check and subsequent local search, dummy methods for destroy + repair
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3), 
                #         MHMethod("local_search_one_deletion", local_search_one_deletion!, 0)],
                #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
                #     [MHMethod("repair_test", repair_test!, 0)])

                # using only CH with topo ord + IMPROVED neighbor check and subsequent local search, dummy methods for destroy + repair
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
                #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
                #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
                #     [MHMethod("repair_test", repair_test!, 0)])

                # using only CH with topo sorting, dummy methods for destroy + repair
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_sorting", construction_heuristic_topological_sort!, 0)],
                #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
                #     [MHMethod("repair_test", repair_test!, 0)])

                # using only CH with topo sorting + neighbor check, dummy methods for destroy + repair
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_sorting_neighbor", construction_heuristic_topological_sort_neighbor_check!, 0)],
                #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
                #     [MHMethod("repair_test", repair_test!, 0)])

                # using only CH with topo sorting in heuristic order, dummy methods for destroy + repair
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_sorting_heuristic_order", construction_heuristic_topological_sort_heuristic!, 0.3)],
                #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
                #     [MHMethod("repair_test", repair_test!, 0)])

                # using only CH with topo sorting in heuristic order + neighbor check, dummy methods for destroy + repair
                # lns_alg = ALNS(subgraph_solution,
                #     [MHMethod("construct_topo_sorting_heuristic_order_neighbor", construction_heuristic_topological_sort_heuristic_neighbor_check!, 0.3)],
                #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
                #     [MHMethod("repair_test", repair_test!, 0)])

                run!(lns_alg)

                # add solution from LNS to solutions from subproblems
                append!(compound_solution, subgraph_solution.x)

                @debug "SCC has solution of length = $(length(subgraph_solution.x))"
                @debug "SCC has solution with number of selected elements = $(subgraph_solution.sel)"
                @debug "Finished SCC of length = $(length(scc))"
                # println("")
                # println("SCC has solution of length = ", length(subgraph_solution.x))
                # println("SCC has solution with number of selected elements = ", subgraph_solution.sel)
                # println("Finished SCC of length = ", length(scc))
                # println("")
            end
        end

        # add compound solution from subproblems to final solution for whole problem
        append!(final_solution_vector, sort(unique!(compound_solution)))

        @debug "Number of further reduced SCCs = $reduced_sccs"
        # println("Solution after LNS: ", final_solution_vector)
        # @info "Length of solution after LNS: $(length(final_solution_vector))"
        # println("Length of solution after LNS: ", length(final_solution_vector))

    else    # only 1 SCC -> dfvsp_solution can be used

        if nv(dfvsp_solution.inst.graph) == 3 && ne(dfvsp_solution.inst.graph) == 6
            # directly add 2 of the 3 vertices to the solution
            append!(final_solution_vector, [dfvsp_solution.vmap_new_orig[v] for v in 1:2])
            # update the global best solution
            update_global_best_solution_add_part([dfvsp_solution.vmap_new_orig[v] for v in 1:2])

        elseif nv(dfvsp_solution.inst.graph) <= 100

            # first reset the current (=default) solution to empty as no valid solution that should be extended is currently known
            clear_dfvs!(dfvsp_solution)
            # directly solve the problem with the MIP model, do not use LNS
            # mtz_formulation!(dfvsp_solution)
            mtz_formulation_reduced!(dfvsp_solution)
            append!(final_solution_vector, dfvsp_solution.x)
            # update the global best solution
            update_global_best_solution_add_part(dfvsp_solution.x)

        else

            # call construction heuristic
            # based on cycle checks
            # construction_heuristic_indegree_outdegree_difference_new!(dfvsp_solution, 0.3, Result())
            # based on topological ordering properties
            # construction_heuristic_indegree_outdegree_difference_alternative!(dfvsp_solution, 0.3, Result())

            # build and call LNS algorithm
            # -> call only possible with working destroy and repair methods 

            # LNS with enlarge DAG neighborhood
            # lns_alg = ALNS(dfvsp_solution, 
            #     [MHMethod("construct_topo_ord", construction_heuristic_indegree_outdegree_difference_alternative!, 0.3)], 
            #     [MHMethod("destroy_dag_random_fixed_k", destroy_dag_random_fixed_k!, 1)],
            #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

            # LNS with enlarge DAG neighborhood and heuristic element selection -> decreasing order
            # lns_alg = ALNS(dfvsp_solution, 
            #     [MHMethod("construct_topo_ord", construction_heuristic_indegree_outdegree_difference_alternative!, 0.3)], 
            #     [MHMethod("destroy_dag_heuristic_selection_decreasing_fixed_k", destroy_dag_heuristic_selection_decreasing_fixed_k!, 1)],
            #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

            # LNS with enlarge DAG neighborhood and heuristic element selection -> increasing order
            # lns_alg = ALNS(dfvsp_solution, 
            #     [MHMethod("construct_topo_ord", construction_heuristic_indegree_outdegree_difference_alternative!, 0.3)], 
            #     [MHMethod("destroy_dag_heuristic_selection_increasing_fixed_k", destroy_dag_heuristic_selection_increasing_fixed_k!, 1)],
            #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

            # using CH with topo ord + neighbor check, LNS with enlarge DAG neighborhood and random element selection
            # lns_alg = ALNS(dfvsp_solution, 
            #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)], 
            #     [MHMethod("destroy_dag_random_fixed_k", destroy_dag_random_fixed_k!, 1)],
            #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

            # using CH with topo ord + neighbor check, LNS with enlarge DAG neighborhood and tournament selection -> smallest hvalue
            # lns_alg = ALNS(dfvsp_solution, 
            #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)], 
            #     [MHMethod("destroy_dag_tournament_selection_smallest_fixed_k", destroy_dag_tournament_selection_smallest_fixed_k!, (1, tournament_sel_percentage))],
            #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

            # using CH with topo ord + neighbor check, LNS with enlarge DAG neighborhood and tournament selection with fixed size -> smallest hvalue
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)],
            #     [MHMethod("destroy_dag_tournament_selection_fixed_size_smallest_fixed_k", destroy_dag_tournament_selection_fixed_size_smallest_fixed_k!, (1, tournament_sel_size))],
            #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

            # using CH with topo ord + IMPROVED neighbor check and subsequent local search mit multi-src multi-dest path, LNS with enlarge DAG neighborhood and tournament selection with fixed size -> smallest hvalue
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
            #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
            #     [MHMethod("destroy_dag_tournament_selection_fixed_size_smallest_fixed_k", destroy_dag_tournament_selection_fixed_size_smallest_fixed_k!, (1, tournament_sel_size))],
            #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

            # using CH with topo ord + IMPROVED neighbor check and subsequent local search mit multi-src multi-dest path, LNS with enlarge DAG neighborhood and tournament selection with fixed size -> smallest hvalue 
            # and application of reduction rules to enlarged DAG
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
            #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
            #     [MHMethod("destroy_dag_tournament_selection_fixed_size_smallest_fixed_k", destroy_dag_tournament_selection_fixed_size_smallest_fixed_k!, (1, tournament_sel_size))],
            #     [MHMethod("repair_dag_rrs_mtz", repair_dag_reduction_rules_mtz_formulation!, 0)])

            # using CH with topo ord + IMPROVED neighbor check and subsequent local search mit multi-src multi-dest path, LNS with enlarge DAG neighborhood and tournament selection with fixed size -> smallest hvalue 
            # and application of reduction rules to enlarged DAG, reduced MTZ model
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
            #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
            #     [MHMethod("destroy_dag_tournament_selection_fixed_size_smallest_fixed_k", destroy_dag_tournament_selection_fixed_size_smallest_fixed_k!, (1, tournament_sel_size))],
            #     [MHMethod("repair_dag_rrs_mtz_reduced", repair_dag_reduction_rules_mtz_formulation_reduced!, 0)])
            
            # using CH with topo ord + neighbor check, LNS with enlarge DAG neighborhood and tournament selection -> highest hvalue
            # lns_alg = ALNS(dfvsp_solution, 
            #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)], 
            #     [MHMethod("destroy_dag_tournament_selection_highest_fixed_k", destroy_dag_tournament_selection_highest_fixed_k!, (1, tournament_sel_percentage))],
            #     [MHMethod("repair_dag_mtz", repair_dag_mtz_formulation!, 0)])

            # LNS with enlarge DFVS neighborhood
            # lns_alg = ALNS(dfvsp_solution, 
            #     [MHMethod("construct_topo_ord", construction_heuristic_indegree_outdegree_difference_alternative!, 0.3)], 
            #     [MHMethod("destroy_dfvs_random_fixed_k", destroy_dfvs_random_fixed_k!, 1)],
            #     [MHMethod("repair_dfvs_mtz", repair_dfvs_mtz_formulation!, 0)])

            # using CH with topo ord + neighbor check, LNS with enlarge DFVS neighborhood using transitive closure computation
            # lns_alg = ALNS(dfvsp_solution, 
            #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)], 
            #     [MHMethod("destroy_dfvs_random_fixed_k", destroy_dfvs_random_fixed_k!, 1)],
            #     [MHMethod("repair_dfvs_mtz_trans_closure", repair_dfvs_mtz_formulation_alternative_transitive_closure!, 0)])

            # using CH with topo ord + neighbor check, LNS with enlarge DFVS neighborhood using transitive closure computation and self-loop checks
            # lns_alg = ALNS(dfvsp_solution, 
            #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)], 
            #     [MHMethod("destroy_dfvs_random_fixed_k", destroy_dfvs_random_fixed_k!, 1)],
            #     [MHMethod("repair_dfvs_mtz_trans_closure_selfloops", repair_dfvs_mtz_formulation_using_transitive_closure_and_selfloops!, 0)])

            # using CH with topo ord + neighbor check, LNS with enlarge DFVS neighborhood using transitive closure computation and self-loop removal
            # lns_alg = ALNS(dfvsp_solution, 
            #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)], 
            #     [MHMethod("destroy_dfvs_random_fixed_k", destroy_dfvs_random_fixed_k!, 1)],
            #     [MHMethod("repair_dfvs_mtz_trans_closure_selfloop_removal", repair_dfvs_mtz_formulation_using_transitive_closure_and_selfloop_removal!, 0)])

            # using CH with topo ord + IMPROVED neighbor check and subsequent local search, LNS with enlarge DFVS neighborhood using transitive closure computation and self-loop removal
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
            #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
            #     [MHMethod("destroy_dfvs_random_fixed_k", destroy_dfvs_random_fixed_k!, 1)],
            #     [MHMethod("repair_dfvs_mtz_trans_closure_selfloop_removal", repair_dfvs_mtz_formulation_using_transitive_closure_and_selfloop_removal!, 0)])

            # using CH with topo ord + IMPROVED neighbor check and subsequent local search, LNS with enlarge DFVS neighborhood using has_path and self-loop removal
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
            #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
            #     [MHMethod("destroy_dfvs_random_fixed_k", destroy_dfvs_random_fixed_k!, 1)],
            #     [MHMethod("repair_dfvs_mtz_path_selfloop_removal", repair_dfvs_mtz_formulation_using_has_path_and_selfloop_removal!, 0)])

            # using CH with topo ord + neighbor check, LNS with enlarge DFVS neighborhood using transitive closure computation, self-loop removal and partial destroy/repair
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)],
            #     [MHMethod("destroy_dfvs_partially_random_fixed_k", destroy_dfvs_partially_random_fixed_k!, 1)],
            #     [MHMethod("repair_dfvs_partially_mtz_trans_closure_selfloop_removal", repair_dfvs_partially_mtz_formulation_using_transitive_closure_and_selfloop_removal!, partial_dfvs_size)])

            # using CH with topo ord + IMPROVED neighbor check and subsequent local search, LNS with enlarge DFVS neighborhood using transitive closure computation, self-loop removal 
            # and partial destroy/repair with tournament selection with fixed size (destroy: highest value, repair: highest value)
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
            #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
            #     [MHMethod("destroy_dfvs_partially_ts_fixed_size_highest_fixed_k", destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!, (1, tournament_sel_size))],
            #     [MHMethod("repair_dfvs_partially_ts_fixed_size_highest_mtz_trans_closure_selfloop_removal", repair_dfvs_partially_tournament_selection_fixed_size_highest_mtz_formulation_using_transitive_closure_and_selfloop_removal!, (partial_dfvs_size, tournament_sel_size))])

            # using CH with topo ord + IMPROVED neighbor check and subsequent local search, LNS with enlarge DFVS neighborhood using transitive closure computation, self-loop removal 
            # and partial destroy/repair with tournament selection with fixed size (destroy: highest value, repair: smallest value)
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
            #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
            #     [MHMethod("destroy_dfvs_partially_ts_fixed_size_highest_fixed_k", destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!, (1, tournament_sel_size))],
            #     [MHMethod("repair_dfvs_partially_ts_fixed_size_smallest_mtz_trans_closure_selfloop_removal", repair_dfvs_partially_tournament_selection_fixed_size_smallest_mtz_formulation_using_transitive_closure_and_selfloop_removal!, (partial_dfvs_size, tournament_sel_size))])

            # using CH with topo ord + IMPROVED neighbor check and subsequent local search, LNS with enlarge DFVS neighborhood using has_path, self-loop removal 
            # and partial destroy/repair with tournament selection with fixed size (destroy: highest value, repair: highest value)
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
            #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
            #     [MHMethod("destroy_dfvs_partially_ts_fixed_size_highest_fixed_k", destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!, (1, tournament_sel_size))],
            #     [MHMethod("repair_dfvs_partially_ts_fixed_size_highest_mtz_path_selfloop_removal", repair_dfvs_partially_tournament_selection_fixed_size_highest_mtz_formulation_using_has_path_and_selfloop_removal!, (partial_dfvs_size, tournament_sel_size))])

            # using CH with topo ord + IMPROVED neighbor check and subsequent local search, LNS with enlarge DFVS neighborhood using has_path, self-loop removal 
            # and partial destroy/repair with tournament selection with fixed size (destroy: highest value, repair: smallest value)
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
            #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
            #     [MHMethod("destroy_dfvs_partially_ts_fixed_size_highest_fixed_k", destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!, (1, tournament_sel_size))],
            #     [MHMethod("repair_dfvs_partially_ts_fixed_size_smallest_mtz_path_selfloop_removal", repair_dfvs_partially_tournament_selection_fixed_size_smallest_mtz_formulation_using_has_path_and_selfloop_removal!, (partial_dfvs_size, tournament_sel_size))])

            # using CH with topo ord + IMPROVED neighbor check and subsequent local search mit multi-src multi-dest path, LNS with enlarge DFVS neighborhood using has_path multi-src multi-dest, self-loop removal 
            # and partial destroy/repair with tournament selection with fixed size (destroy: highest value, repair: smallest value)
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
            #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
            #     [MHMethod("destroy_dfvs_partially_ts_fixed_size_highest_fixed_k", destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!, (1, tournament_sel_size))],
            #     [MHMethod("repair_dfvs_partially_ts_fixed_size_smallest_mtz_path_multi_src_multi_dest_selfloop_removal", repair_dfvs_partially_tournament_selection_fixed_size_smallest_mtz_formulation_using_has_path_multi_src_multi_dest_and_selfloop_removal!, (partial_dfvs_size, tournament_sel_size))])

            # using CH with topo ord + IMPROVED neighbor check and subsequent local search mit multi-src multi-dest path, LNS with enlarge DFVS neighborhood using has_path multi-src multi-dest, self-loop removal, REDUCED MTZ formulation 
            # and partial destroy/repair with tournament selection with fixed size (destroy: highest value, repair: smallest value)
            lns_alg = ALNS(dfvsp_solution,
                [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
                    MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
                [MHMethod("destroy_dfvs_partially_ts_fixed_size_highest_fixed_k", destroy_dfvs_partially_tournament_selection_fixed_size_highest_fixed_k!, (1, tournament_sel_size))],
                [MHMethod("repair_dfvs_partially_ts_fixed_size_smallest_mtz_reduced_path_multi_src_multi_dest_selfloop_removal", repair_dfvs_partially_tournament_selection_fixed_size_smallest_mtz_formulation_reduced_using_has_path_multi_src_multi_dest_and_selfloop_removal!, (partial_dfvs_size, tournament_sel_size))])
            
            # using only CH with cycle check, dummy methods for destroy + repair
            # lns_alg = ALNS(dfvsp_solution, 
            #         [MHMethod("construct_cycle_check", construction_heuristic_indegree_outdegree_difference_new!, 0.3)], 
            #         [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
            #         [MHMethod("repair_test", repair_test!, 0)])

            # using only CH with topo ord, dummy methods for destroy + repair
            # lns_alg = ALNS(dfvsp_solution, 
            #          [MHMethod("construct_topo_ord", construction_heuristic_indegree_outdegree_difference_alternative!, 0.3)], 
            #          [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
            #          [MHMethod("repair_test", repair_test!, 0)])

            # using only CH with topo ord + neighbor check, dummy methods for destroy + repair
            # lns_alg = ALNS(dfvsp_solution, 
            #          [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3)], 
            #          [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
            #          [MHMethod("repair_test", repair_test!, 0)])

            # using only CH with topo ord + IMPROVED neighbor check, dummy methods for destroy + repair
            # lns_alg = ALNS(dfvsp_solution, 
            #          [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3)], 
            #          [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
            #          [MHMethod("repair_test", repair_test!, 0)])

            # using only CH with topo ord + IMPROVED neighbor check + transitive closure, dummy methods for destroy + repair
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor_trans_closure", construction_heuristic_indegree_outdegree_difference_alternative_transitive_closure!, 0.3)],
            #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
            #     [MHMethod("repair_test", repair_test!, 0)])

            # using only CH with topo ord + IMPROVED neighbor check + transitive closure, dummy methods for destroy + repair
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor_path_check", construction_heuristic_indegree_outdegree_difference_alternative_path_check!, 0.3)],
            #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
            #     [MHMethod("repair_test", repair_test!, 0)])

            # using only CH with topo ord + neighbor check and subsequent local search, dummy methods for destroy + repair
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor", construction_heuristic_indegree_outdegree_difference_alternative2!, 0.3), 
            #         MHMethod("local_search_one_deletion", local_search_one_deletion!, 0)],
            #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
            #     [MHMethod("repair_test", repair_test!, 0)])

            # using only CH with topo ord + IMPROVED neighbor check and subsequent local search, dummy methods for destroy + repair
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_ord_neighbor_improved", construction_heuristic_indegree_outdegree_difference_alternative3!, 0.3), 
            #         MHMethod("local_search_one_deletion", local_search_one_deletion!, local_search_time_limit)],
            #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
            #     [MHMethod("repair_test", repair_test!, 0)])

            # using only CH with topo sorting, dummy methods for destroy + repair
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_sorting", construction_heuristic_topological_sort!, 0)],
            #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
            #     [MHMethod("repair_test", repair_test!, 0)])

            # using only CH with topo sorting + neighbor check, dummy methods for destroy + repair
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_sorting_neighbor", construction_heuristic_topological_sort_neighbor_check!, 0)],
            #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
            #     [MHMethod("repair_test", repair_test!, 0)])

            # using only CH with topo sorting in heuristic order, dummy methods for destroy + repair
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_sorting_heuristic_order", construction_heuristic_topological_sort_heuristic!, 0.3)],
            #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
            #     [MHMethod("repair_test", repair_test!, 0)])

            # using only CH with topo sorting in heuristic order + neighbor check, dummy methods for destroy + repair
            # lns_alg = ALNS(dfvsp_solution,
            #     [MHMethod("construct_topo_sorting_heuristic_order_neighbor", construction_heuristic_topological_sort_heuristic_neighbor_check!, 0.3)],
            #     [MHMethod("destroy_test", destroy_random_fixed_k_test!, 1)],
            #     [MHMethod("repair_test", repair_test!, 0)])
            
            run!(lns_alg)

            # println("Solution from construction heuristic: ", dfvsp_solution.x)
            # println("Length of solution from construction heuristic: ", length(dfvsp_solution.x))
            # @info "Length of solution after LNS: $(length(dfvsp_solution.x))"
            # println("Length of solution after LNS: ", length(dfvsp_solution.x))

            # add solution to final solution for whole problem
            append!(final_solution_vector, dfvsp_solution.x)

        end
    end


    # append preliminary DFVS from reduction rules
    append!(final_solution_vector, dfvsp_solution.dfvs_rr)

    # testing - add all remaining vertices after reduction rules to solution -> wrong!! needs to be done using the method in DFVSP for correct labeling
    # for vertex in vertices(dfvsp_solution.inst.graph)
    #     push!(final_solution_vector, vertex)
    # end

    # remove duplicate vertices and sort vertices in increasing order
    final_solution_vector = sort(unique!(final_solution_vector))

    # @info "Length of preliminary DFVS from reduction rules: $(length(dfvsp_solution.dfvs_rr))"
    # println("Length of preliminary DFVS from reduction rules: ", length(dfvsp_solution.dfvs_rr))
    # println("Final solution: ", final_solution_vector)
    # @info "Length of final solution: $(length(final_solution_vector))"
    # println("Length of final solution: ", length(final_solution_vector))

    # get current datetime
    # df = Dates.DateFormat("dd-mm-yyyy_HH-MM-SS")
    # current_datetime = Dates.now()
    # formatted_datetime = Dates.format(current_datetime, df)

    # print solution to output file in following format
    # print one vertex of solution in each line followed by new line character "\n"
    # => use println which automatically appends a newline
    # output_directory = arguments[3]
    # output_filename = instance_name * "-" * formatted_datetime
    # output_file = output_directory * output_filename * ".txt"
    # OLD: output_filename = output_directory * instance_name * "-" * formatted_datetime * ".txt"

    # open(output_file, "w") do io
    #     for vertex in final_solution_vector
    #         println(io, vertex)
    #     end
    # end

    # recover original stdout
    # redirect_stdout(orig_stdout)
    redirect_stdout(global_orig_stdout)
    # print solution to stdout
    for vertex in final_solution_vector
        println(vertex)
    end

    # deactivate exit hook for atexit()
    set_global_normal_program_termination(true)

    # old: graph = DFVSPInstance(instance_directory * instance_name).graph
    # graph = DFVSPInstance(instance_file).graph
    # @assert isdfvs(graph, final_solution_vector)

    # plot final DAG
    # plot_solution = DFVSPSolution(DFVSPInstance(instance_directory * instance_name))
    # clear_all!(plot_solution)
    # plot_solution.x = copy(final_solution_vector)
    # analyse_dag(plot_solution)
    # plot_dag(plot_solution, output_filename)

end


main()



