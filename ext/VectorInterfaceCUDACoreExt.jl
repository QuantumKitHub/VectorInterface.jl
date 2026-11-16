module VectorInterfaceCUDACoreExt

using VectorInterface, CUDACore

import CUDACore: CuRef

CUDACore.CuRef{T}(x::VectorInterface.Scalar{T, <:CuVector{T}}) where {T} = Base.unsafe_convert(CuRef{T}, pointer(x.a))

end
