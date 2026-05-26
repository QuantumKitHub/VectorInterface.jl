using VectorInterface
using VectorInterface: MinimalMVec, MinimalSVec, MinimalVec
using Test, TestExtras
using ChainRulesTestUtils
using ChainRulesCore: ChainRulesCore, AbstractZero

Base.collect(x::MinimalVec) = x.vec

precision(::Type{T}) where {T <: Union{Float32, ComplexF32}} = sqrt(eps(Float32))
precision(::Type{T}) where {T <: Union{Float64, ComplexF64}} = sqrt(eps(Float64))

# Small adaptations to make tests work with MinimalVec
function ChainRulesTestUtils.test_approx(::AbstractZero, x::MinimalVec, msg = ""; kwargs...)
    return test_approx(zerovector(x), x, msg; kwargs...)
end
function ChainRulesTestUtils.test_approx(x::MinimalVec, ::AbstractZero, msg = ""; kwargs...)
    return test_approx(x, zerovector(x), msg; kwargs...)
end

eltypes = (Float64, ComplexF64)

@testset "scale pullbacks ($T)" for T in eltypes
    n = 12
    atol = rtol = n * precision(T)

    # Vector
    x = randn(T, n)
    y = randn(T, n)
    α = randn(T)
    test_rrule(scale, x, α; atol, rtol)
    test_rrule(scale!!, x, α; atol, rtol)
    test_rrule(scale!!, y, x, α; atol, rtol)

    # MinimalMVec
    mx = MinimalMVec(x)
    my = MinimalMVec(y)
    test_rrule(scale, mx, α; atol, rtol, check_inferred = false)
    test_rrule(scale!!, mx, α; atol, rtol, check_inferred = false)
    test_rrule(scale!!, my, mx, α; atol, rtol, check_inferred = false)

    # MinimalSVec
    mx = MinimalSVec(x)
    my = MinimalSVec(y)
    test_rrule(scale, mx, α; atol, rtol, check_inferred = false)
    test_rrule(scale!!, mx, α; atol, rtol, check_inferred = false)
    test_rrule(scale!!, my, mx, α; atol, rtol, check_inferred = false)
end

@testset "add pullbacks ($T)" for T in eltypes
    n = 12
    atol = rtol = n * precision(T)

    # Vector
    x = randn(T, n)
    y = randn(T, n)
    α = randn(T)
    β = randn(T)
    test_rrule(add, y, x, α, β; atol, rtol)
    test_rrule(add!!, y, x, α, β; atol, rtol)

    # MinimalMVec
    mx = MinimalMVec(x)
    my = MinimalMVec(y)
    test_rrule(add, my, mx, α, β; atol, rtol, check_inferred = false)
    test_rrule(add!!, my, mx, α, β; atol, rtol, check_inferred = false)

    # MinimalSVec
    mx = MinimalSVec(x)
    my = MinimalSVec(y)
    test_rrule(add, my, mx, α, β; atol, rtol, check_inferred = false)
    test_rrule(add!!, my, mx, α, β; atol, rtol, check_inferred = false)
end

@testset "inner pullbacks ($T)" for T in eltypes
    n = 12
    atol = rtol = n * precision(T)

    # Vector
    x = randn(T, n)
    y = randn(T, n)
    test_rrule(inner, x, y; atol, rtol)

    # MinimalMVec
    mx = MinimalMVec(x)
    my = MinimalMVec(y)
    test_rrule(inner, mx, my; atol, rtol, check_inferred = false)

    # MinimalSVec
    mx = MinimalSVec(x)
    my = MinimalSVec(y)
    test_rrule(inner, mx, my; atol, rtol, check_inferred = false)
end
