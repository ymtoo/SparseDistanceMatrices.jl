module SparseDistanceMatrices

export SparseDistanceMatrix, symmetrize!

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

Base.size(D::SparseDistanceMatrix) = (D.n, D.n)
function Base.getindex(D::SparseDistanceMatrix{T}, i::Integer, j::Integer) where T
    index = findall((i .== D.rowindices) .& (j .== D.colindices))
    if length(index) == 1
        return D.ndval[index[1]]
    end
    D.defaultval
end
function Base.setindex!(D::SparseDistanceMatrix{T}, v::T, i::Integer, j::Integer) where T
    push!(D.rowindices, i)
    push!(D.colindices, j)
    push!(D.ndval, v)
    D
end

function symmetrize!(D::SparseDistanceMatrix{T}) where T
    rowindicestmp = Int[]
    colindicestmp = Int[]
    ndvaltmp = T[]
    for (i, j, v) in zip(D.rowindices, D.colindices, D.ndval)
        index = findall((j .== D.rowindices) .& (i .== D.colindices))
        if length(index) == 0
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

end # module
