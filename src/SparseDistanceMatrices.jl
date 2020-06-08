module SparseDistanceMatrices

export SparseDistanceMatrix, symmetrize!

struct SparseDistanceMatrix{T} <: AbstractMatrix{T}
    n::Int
    colindices::Vector{Int}
    rowindices::Vector{Int}
    nzval::Vector{T}
    defaultval::T
end
SparseDistanceMatrix(n::Int, colindices::Vector{Int}, rowindices::Vector{Int},
                     nzval::Vector{T}) where T = SparseDistanceMatrix(n,
                                                                      colindices,
                                                                      rowindices,
                                                                      nzval,
                                                                      typemax(eltype(nzval)))

Base.size(D::SparseDistanceMatrix) = (D.n, D.n)
function Base.getindex(D::SparseDistanceMatrix{T}, i::Integer, j::Integer) where T
    index = findall(x->x==1, (i .== D.rowindices) .& (j .== D.colindices))
    if length(index) == 1
        return D.nzval[index[1]]
    end
    D.defaultval
end
function Base.setindex!(D::SparseDistanceMatrix{T}, v::T, i::Integer, j::Integer) where T
    push!(D.rowindices, i)
    push!(D.colindices, j)
    push!(D.nzval, v)
    D
end

function symmetrize!(D::SparseDistanceMatrix{T}) where T
    rowindicestmp = Int[]
    colindicestmp = Int[]
    nzvaltmp = T[]
    for (i, j, v) in zip(D.rowindices, D.colindices, D.nzval)
        index = findall(x->x==1, (j .== D.rowindices) .& (i .== D.colindices))
        if length(index) == 0
            push!(rowindicestmp, j)
            push!(colindicestmp, i)
            push!(nzvaltmp, v)
        end
    end
    append!(D.rowindices, rowindicestmp)
    append!(D.colindices, colindicestmp)
    append!(D.nzval, nzvaltmp)
    D
end

end # module
