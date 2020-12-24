module SparseDistanceMatrices

using Distances, LinearAlgebra, NearestNeighbors, ProgressMeter

import Distances: deprecated_dims, result_type

export SparseDistanceMatrix, countnt, symmetrize!, adjacency_matrix, pairwise

struct SparseDistanceMatrix{T} <: AbstractMatrix{T}
    n::Int
    colindices::Vector{Int}
    rowindices::Vector{Int}
    ndval::Vector{T}
    defaultval::T
end
SparseDistanceMatrix(n::Int, colindices::Vector{Int}, rowindices::Vector{Int},
                     ndval::Vector{T}) where T = SparseDistanceMatrix(n,
                                                                      colindices,
                                                                      rowindices,
                                                                      ndval,
                                                                      typemax(eltype(ndval)))
SparseDistanceMatrix(n::Int, T::Type) = SparseDistanceMatrix(n, Int[], Int[], T[])
SparseDistanceMatrix(n::Int, defaultval::T) where T = SparseDistanceMatrix(n, Int[], Int[], T[], defaultval)

Base.size(D::SparseDistanceMatrix) = (D.n, D.n)
function Base.getindex(D::SparseDistanceMatrix{T}, i::Integer, j::Integer) where T
    i == j && return T(0.0)
    index = findfirst((i .== D.rowindices) .& (j .== D.colindices))
    index === nothing && return D.defaultval
    return D.ndval[index]
end
function Base.setindex!(D::SparseDistanceMatrix{T}, v::T, i::Integer, j::Integer) where T
    if i == j
        v == zero(T) ? (return v) : throw(ArgumentError("Diagonol element of a distance matrix has to be $(zero(T))"))
    end
    index = findfirst((i .== D.rowindices) .& (j .== D.colindices))
    if index  === nothing
        push!(D.rowindices, i)
        push!(D.colindices, j)
        push!(D.ndval, v)
    else
        D.ndval[index] = v
    end
    v
end

Base.transpose(D::SparseDistanceMatrix{T}) where T = SparseDistanceMatrix(D.n, D.rowindices, D.colindices, D.ndval, D.defaultval)
function LinearAlgebra.adjoint(D::SparseDistanceMatrix{T}) where T
    T <: Complex ? SparseDistanceMatrix(D.n, D.rowindices, D.colindices, conj.(D.ndval), conj(D.defaultval)) : SparseDistanceMatrix(D.n, D.rowindices, D.colindices, D.ndval, D.defaultval)
end

"""
Count the number of non-trivial elements.
"""
countnt(D::SparseDistanceMatrix{T}) where T = length(D.ndval)

function symmetrize!(D::SparseDistanceMatrix{T}) where T
    rowindicestmp = Int[]
    colindicestmp = Int[]
    ndvaltmp = T[]
    for (i, j, v) in zip(D.rowindices, D.colindices, D.ndval)
        index = findfirst((j .== D.rowindices) .& (i .== D.colindices))
        if index === nothing
            push!(rowindicestmp, j)
            push!(colindicestmp, i)
            push!(ndvaltmp, v)
        end
    end
    append!(D.rowindices, rowindicestmp)
    append!(D.colindices, colindicestmp)
    append!(D.ndval, ndvaltmp)
    D
end

function adjacency_matrix(D::SparseDistanceMatrix{T}) where T
     SparseDistanceMatrix(copy(D.n), copy(D.colindices), copy(D.rowindices), copy(D.ndval), zero(T))
end

function _pairwise!(D::SparseDistanceMatrix{T}, 
                    metric::PreMetric, 
                    a::AbstractMatrix, 
                    k::Int; 
                    map::Function=map, 
                    showprogress::Bool=false) where T
    n = size(a, 2)
    maxndval = typemax(T)
    oneton = 1:n
    showprogress && (p = Progress(n, 1, "Pairwise distance matrix..."))
    for i in 1:n
        d = map(j -> metric(view(a, :, i), view(a, :, j)), 1:n)
        d[i] = typemax(T)
        fillbindices = d .< maxndval
        if any(fillbindices)
            append!(D.rowindices, fill(i, sum(fillbindices)))
            append!(D.colindices, oneton[fillbindices])
            append!(D.ndval, d[fillbindices])
        end
        if countnt(D) > k
            ind = partialsortperm(D.ndval, 1:k)
            discardindices = [discardindex for discardindex in 1:length(D.ndval) if discardindex ∉ ind]
            deleteat!(D.rowindices, discardindices)
            deleteat!(D.colindices, discardindices)
            deleteat!(D.ndval, discardindices)
        end
        maxndval = maximum(D.ndval)
        showprogress && next!(p)
    end
    D
end

function Distances.pairwise(metric::PreMetric, 
                            a::AbstractMatrix, 
                            k::Int; 
                            dims::Union{Nothing,Integer}=nothing, 
                            map::Function=map, 
                            showprogress::Bool=false)
    dims = deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    n = size(a, dims)
    T = result_type(metric, a, a)
    D = SparseDistanceMatrix(n, T)
    if dims == 1
        _pairwise!(D, metric, transpose(a), k; map=map, showprogress=showprogress)
    else
        _pairwise!(D, metric, a, k; map=map, showprogress=showprogress)
    end
end

function _pairwise!(D::SparseDistanceMatrix{T},
                    tree::TT,
                    a::AbstractMatrix,
                    k::Int) where {T,TT<:NNTree}
    idxs, dists = knn(tree, a, k+1)
    for (i, (idx, dist)) ∈ enumerate(zip(idxs, dists))
        append!(D.rowindices, repeat([i], k))
        append!(D.colindices, idx[2:end])
        append!(D.ndval, dist[2:end])
    end
    D
end

function Distances.pairwise(tree::TT, 
                            a::AbstractMatrix, 
                            k::Int; 
                            dims::Union{Nothing,Integer}=nothing) where TT<:NNTree
    dims = deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    n = size(a, dims)
    T = result_type(tree.metric, a, a)
    D = SparseDistanceMatrix(n, T)
    idxs, dists = knn(tree, a, k+1, true)
    if dims == 1
        _pairwise!(D, tree, transpose(a), k)
    else
        _pairwise!(D, tree, a, k)
    end
end


end # module
