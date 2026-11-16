module VectorInterfacecuBLASExt

using VectorInterface, cuBLAS, LinearAlgebra
using cuBLAS.CUDACore: CuVector

function VectorInterface.inner!(result::Scalar{T, <:CuVector{T}}, x::VectorInterface.BLASVector, y::VectorInterface.BLASVector) where {T <: Real}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    cuBLAS.dot(n, x, y, result)
    return result
end

function VectorInterface.inner!(result::Scalar{T, <:CuVector{T}}, x::VectorInterface.BLASVector, y::VectorInterface.BLASVector) where {T <: Complex}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dot product arguments have lengths $(length(x)) and $(length(y))"))
    cuBLAS.dotc(n, x, y, result)
    return result
end

function VectorInterface.norm!(result::Scalar{T, <:CuVector{T}}, x::VectorInterface.BLASVector, p::Real = 2) where {T}
    if p == 2
        cuBLAS.nrm2(length(x), x, result)
    else
        result[] = LinearAlgebra.norm(x, p)
    end
    return result
end

end
