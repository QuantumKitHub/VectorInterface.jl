module StaticSVec
using VectorInterface
using StaticArrays
using Test
using TestExtras

deepcollect(x::StaticArray) = collect(x)
deepcollect(x::Number) = x

x = SVector{3}(randn(3))
y = SVector{3}(randn(3))

@testset "scalartype" begin
    s = @constinferred scalartype(x)
    @test s == Float64
end

@testset "zerovector" begin
    z = @constinferred zerovector(x)
    @test z isa SVector{3, Float64}
    @test all(iszero, deepcollect(z))
    @test all(deepcollect(z) .=== zero(scalartype(x)))
    z1 = @constinferred zerovector!!(x)
    @test z1 isa SVector{3, Float64}
    @test all(deepcollect(z1) .=== zero(scalartype(x)))

    z3 = @constinferred zerovector(x, ComplexF64)
    @test z3 isa SVector{3, ComplexF64}
    @test all(deepcollect(z3) .=== zero(ComplexF64))
    z4 = @constinferred zerovector!!(x, ComplexF64)
    @test z4 isa SVector{3, ComplexF64}
    @test all(deepcollect(z4) .=== zero(ComplexF64))
end

@testset "scale" begin
    α = randn()
    z = @constinferred scale(x, α)
    @test z isa SVector{3, Float64}
    @test all(deepcollect(z) .== α .* deepcollect(x))

    z2 = @constinferred scale!!(x, α)
    @test z2 isa SVector{3, Float64}
    @test deepcollect(z2) ≈ (α .* deepcollect(x))
    z2 = @constinferred scale!!(y, x, α)
    @test z2 isa SVector{3, Float64}
    @test deepcollect(z2) ≈ (α .* deepcollect(x))

    α = randn(ComplexF64)
    z4 = @constinferred scale(x, α)
    @test z4 isa SVector{3, ComplexF64}
    @test deepcollect(z4) ≈ (α .* deepcollect(x))
    z5 = @constinferred scale!!(x, α)
    @test z5 isa SVector{3, ComplexF64}
    @test deepcollect(z5) ≈ (α .* deepcollect(x))

    z6 = @constinferred scale!!(zerovector(x), x, α)
    @test z6 isa SVector{3, ComplexF64}
    @test deepcollect(z6) ≈ (α .* deepcollect(x))

    ycomplex = zerovector(y, ComplexF64)
    α = randn(Float64)
    z8 = @constinferred scale!!(ycomplex, x, α)
    @test scalartype(z8) == ComplexF64
    @test all(deepcollect(z8) .== α .* deepcollect(x))
end

@testset "add" begin
    α, β = randn(2)
    z = @constinferred add(y, x)
    @test z isa SVector{3, Float64}
    @test all(deepcollect(z) .== deepcollect(x) .+ deepcollect(y))
    z = @constinferred add(y, x, α)
    @test deepcollect(z) ≈ muladd.(deepcollect(x), α, deepcollect(y))
    z = @constinferred add(y, x, α, β)
    @test deepcollect(z) ≈ muladd.(deepcollect(x), α, deepcollect(y) .* β)

    z2 = @constinferred add!!(y, x)
    @test z2 isa SVector{3, Float64}
    @test deepcollect(z2) ≈ (deepcollect(x) .+ deepcollect(y))
    z2 = @constinferred add!!(y, x, α)
    @test deepcollect(z2) ≈ (muladd.(deepcollect(x), α, deepcollect(y)))
    z2 = @constinferred add!!(y, x, α, β)
    @test deepcollect(z2) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* β))

    α, β = randn(ComplexF64, 2)
    z4 = @constinferred add(y, x, α)
    @test z4 isa SVector{3, ComplexF64}
    @test deepcollect(z4) ≈ (muladd.(deepcollect(x), α, deepcollect(y)))
    z4 = @constinferred add(y, x, α, β)
    @test deepcollect(z4) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* β))

    z5 = @constinferred add!!(y, x, α)
    @test z5 isa SVector{3, ComplexF64}
    @test deepcollect(z5) ≈ (muladd.(deepcollect(x), α, deepcollect(y)))
    z5 = @constinferred add!!(y, x, α, β)
    @test deepcollect(z5) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* β))
end

@testset "inner" begin
    s = @constinferred inner(x, y)
    @test s ≈ inner(deepcollect(x), deepcollect(y))

    α, β = randn(ComplexF64, 2)
    s2 = @constinferred inner(scale(x, α), scale(y, β))
    @test s2 ≈ inner(α * deepcollect(x), β * deepcollect(y))
end

end
