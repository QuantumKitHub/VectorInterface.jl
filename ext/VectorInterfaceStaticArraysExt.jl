module VectorInterfaceStaticArraysExt

using VectorInterface
using StaticArrays: SArray

# `SArray` is immutable so make sure !! methods route to non-inplace methods
VectorInterface.zerovector!!(x::SArray) = zerovector(x)
VectorInterface.scale!!(x::SArray, α::Number) = scale(x, α)
VectorInterface.scale!!(y::SArray, x::SArray, α::Number) = scale(x, α * one(scalartype(y)))
VectorInterface.add!!(y::SArray, x::SArray, α::Number, β::Number) = add(y, x, α, β)

end
