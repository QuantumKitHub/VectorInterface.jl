module VectorInterfacecuBLASExt

using VectorInterface, cuBLAS, LinearAlgebra
using cuBLAS.CUDACore: CuVector, StridedCuVector, DenseCuArray

const cuBLASVector{T <: LinearAlgebra.BlasFloat} = Union{DenseCuArray{T}, StridedCuVector{T}}

function VectorInterface.inner(x::cuBLASVector{T}, y::cuBLASVector{T}) where {T <: Real}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    result = Scalar(CuVector{T}(undef, 1))
    cuBLAS.dot(n, x, y, result)
    return result
end

function VectorInterface.inner(x::cuBLASVector{T}, y::cuBLASVector{T}) where {T <: Complex}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    result = Scalar(CuVector{T}(undef, 1))
    cuBLAS.dotc(n, x, y, result)
    return result
end

function VectorInterface.norm(x::cuBLASVector{T}, p::Real = 2) where {T}
    result = Scalar(CuVector{real(T)}(undef, 1))
    if p == 2
        cuBLAS.nrm2(length(x), x, result)
    else
        result[] = LinearAlgebra.norm(x, p)
    end
    return result
end

end
