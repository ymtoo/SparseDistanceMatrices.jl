module SparseDistanceMatrices

using LinearAlgebra

export SparseDistanceMatrix, countnt, symmetrize!, adjacency_matrix

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
SparseDistanceMatrix(n::Int, T::DataType) = SparseDistanceMatrix(n, Int[], Int[], T[])
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
        v == T(0.0) ? (return v) : throw(ArgumentError("Diagonol element of a distance matrix has to be $(T(0.0))"))
    end
    index = findfirst((i .== D.rowindices) .& (j .== D.colindices))
    if index === nothing
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
     SparseDistanceMatrix(copy(D.n), copy(D.colindices), copy(D.rowindices), copy(D.ndval), T(0.0))
end


end # module
