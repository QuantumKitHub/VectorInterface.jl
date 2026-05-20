module MooncakeTests

using VectorInterface
using VectorInterface: MinimalMVec, MinimalSVec, MinimalVec
using Test, TestExtras
using Mooncake
using Random

rng = Random.default_rng()

precision(::Type{T}) where {T <: Union{Float32, ComplexF32}} = sqrt(eps(Float32))
precision(::Type{T}) where {T <: Union{Float64, ComplexF64}} = sqrt(eps(Float64))

eltypes = (Float32, Float64, ComplexF64)

@testset "scale ($T)" for T in eltypes
    n = 12
    atol = rtol = n * precision(T)

    # Vector
    x = randn(T, n)
    y = randn(T, n)
    α = randn(T)
    Mooncake.TestUtils.test_rule(rng, scale, x, α; atol, rtol, is_primitive = false)
    Mooncake.TestUtils.test_rule(rng, scale!!, x, α; atol, rtol, is_primitive = false)
    Mooncake.TestUtils.test_rule(rng, scale!!, y, x, α; atol, rtol, is_primitive = false)

    # MinimalMVec
    mx = MinimalMVec(x)
    my = MinimalMVec(y)
    Mooncake.TestUtils.test_rule(rng, scale, mx, α; atol, rtol, is_primitive = false)
    Mooncake.TestUtils.test_rule(rng, scale!!, mx, α; atol, rtol, is_primitive = false)
    Mooncake.TestUtils.test_rule(rng, scale!!, my, mx, α; atol, rtol, is_primitive = false)

    # MinimalSVec
    mx = MinimalSVec(x)
    my = MinimalSVec(y)
    Mooncake.TestUtils.test_rule(rng, scale, mx, α; atol, rtol, is_primitive = false)
    Mooncake.TestUtils.test_rule(rng, scale!!, mx, α; atol, rtol, is_primitive = false)
    Mooncake.TestUtils.test_rule(rng, scale!!, my, mx, α; atol, rtol, is_primitive = false)
end

@testset "add pullbacks ($T)" for T in eltypes
    n = 12
    atol = rtol = n * precision(T)

    # Vector
    x = randn(T, n)
    y = randn(T, n)
    α = randn(T)
    β = randn(T)
    Mooncake.TestUtils.test_rule(rng, add, y, x, α, β; atol, rtol, is_primitive = false)
    Mooncake.TestUtils.test_rule(rng, add!!, y, x, α, β; atol, rtol, is_primitive = false)

    # MinimalMVec
    mx = MinimalMVec(x)
    my = MinimalMVec(y)
    Mooncake.TestUtils.test_rule(rng, add, my, mx, α, β; atol, rtol, is_primitive = false)
    Mooncake.TestUtils.test_rule(rng, add!!, my, mx, α, β; atol, rtol, is_primitive = false)

    # MinimalSVec
    mx = MinimalSVec(x)
    my = MinimalSVec(y)
    Mooncake.TestUtils.test_rule(rng, add, my, mx, α, β; atol, rtol, is_primitive = false)
    Mooncake.TestUtils.test_rule(rng, add!!, my, mx, α, β; atol, rtol, is_primitive = false)
end

@testset "inner pullbacks ($T)" for T in eltypes
    n = 12
    atol = rtol = n * precision(T)

    # Vector
    x = randn(T, n)
    y = randn(T, n)
    Mooncake.TestUtils.test_rule(rng, inner, x, y; atol, rtol, is_primitive = false)

    # MinimalMVec
    mx = MinimalMVec(x)
    my = MinimalMVec(y)
    Mooncake.TestUtils.test_rule(rng, inner, mx, my; atol, rtol, is_primitive = false)

    # MinimalSVec
    mx = MinimalSVec(x)
    my = MinimalSVec(y)
    Mooncake.TestUtils.test_rule(rng, inner, mx, my; atol, rtol, is_primitive = false)
end

end
