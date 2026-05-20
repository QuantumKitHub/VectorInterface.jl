module MooncakeTests

using VectorInterface
using VectorInterface: MinimalMVec, MinimalSVec, MinimalVec
using Test, TestExtras
using Mooncake
using Random

rng = Random.default_rng()

precision(::Type{T}) where {T <: Union{Float32, ComplexF32}} = sqrt(eps(Float32))
precision(::Type{T}) where {T <: Union{Float64, ComplexF64}} = sqrt(eps(Float64))

# Small adaptations to make tests work with MinimalVec
#=function ChainRulesTestUtils.test_approx(::AbstractZero, x::MinimalVec, msg = ""; kwargs...)
    return test_approx(zerovector(x), x, msg; kwargs...)
end
function ChainRulesTestUtils.test_approx(x::MinimalVec, ::AbstractZero, msg = ""; kwargs...)
    return test_approx(x, zerovector(x), msg; kwargs...)
end
Base.collect(x::MinimalVec) = x.vec=#

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
