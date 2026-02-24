using LinearAlgebra

"""
    contract(A, modesA, B, modesB, modesC; compute_type=nothing)

High-level tensor contraction: `C = A * B` with Einstein summation semantics.
Returns a new `ROCArray`.
"""
function contract(
    A::ROCArray, modesA::Vector{<:Integer},
    B::ROCArray, modesB::Vector{<:Integer},
    modesC::Vector{<:Integer};
    compute_type::Union{DataType, Nothing} = nothing,
)
    T = promote_type(eltype(A), eltype(B))
    # Infer output shape from modes
    mode_sizes = Dict{Int, Int}()
    for (i, m) in enumerate(modesA)
        mode_sizes[m] = size(A, i)
    end
    for (i, m) in enumerate(modesB)
        if haskey(mode_sizes, m)
            size(B, i) == mode_sizes[m] || throw(DimensionMismatch(
                "mode $m has size $(mode_sizes[m]) in A but $(size(B, i)) in B"))
        else
            mode_sizes[m] = size(B, i)
        end
    end
    out_shape = Tuple(mode_sizes[m] for m in modesC)
    C = ROCArray{T}(undef, out_shape)
    fill!(C, zero(T))
    contract!(one(T), A, modesA, B, modesB, zero(T), C, modesC;
              compute_type=compute_type)
    return C
end

"""
    permutedims!(dest::ROCArray, src::ROCArray, perm)

Tensor permutation using hipTensor. `perm` is a tuple or vector of dimension indices.
"""
function Base.permutedims!(dest::ROCArray, src::ROCArray, perm)
    ndims(src) == length(perm) || throw(ArgumentError(
        "perm must have same length as ndims(src)"))
    modesA = collect(Int, 1:ndims(src))
    modesB = collect(Int, perm)
    permute!(one(eltype(src)), src, modesA, dest, modesB)
    return dest
end
