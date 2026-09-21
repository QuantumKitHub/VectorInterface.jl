module StaticSVec
using VectorInterface
using JLArrays
using Test
using TestExtras

deepcollect(x::JLArray) = collect(x)
deepcollect(x::Number) = x

x = JLVector(randn(3))
y = JLVector(randn(3))
nan_y = JLVector(vcat(randn(2), NaN))

@testset "scalartype" begin
    s = @constinferred scalartype(x)
    @test s == Float64
end

@testset "zerovector" begin
    z = @constinferred zerovector(x)
    @test z isa JLVector{Float64}
    @test all(iszero, deepcollect(z))
    @test all(deepcollect(z) .=== zero(scalartype(x)))
    z1 = @constinferred zerovector!!(x)
    @test z1 isa JLVector{Float64}
    @test all(deepcollect(z1) .=== zero(scalartype(x)))

    z3 = @constinferred zerovector(x, ComplexF64)
    @test z3 isa JLVector{ComplexF64}
    @test all(deepcollect(z3) .=== zero(ComplexF64))
    z4 = @constinferred zerovector!!(x, ComplexF64)
    @test z4 isa JLVector{ComplexF64}
    @test all(deepcollect(z4) .=== zero(ComplexF64))
end

@testset "scale" begin
    α = randn()
    z = @constinferred scale(x, α)
    @test z isa JLVector{Float64}
    @test all(deepcollect(z) .== α .* deepcollect(x))

    z2 = @constinferred scale!!(x, α)
    @test z2 isa JLVector{Float64}
    @test deepcollect(z2) ≈ (α .* deepcollect(x))
    z2 = @constinferred scale!!(y, x, α)
    @test z2 isa JLVector{Float64}
    @test deepcollect(z2) ≈ (α .* deepcollect(x))

    α = randn(ComplexF64)
    z4 = @constinferred scale(x, α)
    @test z4 isa JLVector{ComplexF64}
    @test deepcollect(z4) ≈ (α .* deepcollect(x))
    z5 = @constinferred scale!!(x, α)
    @test z5 isa JLVector{ComplexF64}
    @test deepcollect(z5) ≈ (α .* deepcollect(x))

    z6 = @constinferred scale!!(zerovector(x), x, α)
    @test z6 isa JLVector{ComplexF64}
    @test deepcollect(z6) ≈ (α .* deepcollect(x))

    ycomplex = zerovector(y, ComplexF64)
    α = randn(Float64)
    z8 = @constinferred scale!!(ycomplex, x, α)
    @test scalartype(z8) == ComplexF64
    @test all(deepcollect(z8) .== α .* deepcollect(x))
end

@testset "add" begin
    α, β = randn(2)
    z = add(y, x)
    @test z isa JLVector{Float64}
    @test all(deepcollect(z) .== deepcollect(x) .+ deepcollect(y))
    z = add(y, x, α)
    @test deepcollect(z) ≈ muladd.(deepcollect(x), α, deepcollect(y))
    z = add(y, x, α, β)
    @test deepcollect(z) ≈ muladd.(deepcollect(x), α, deepcollect(y) .* β)

    z2 = @constinferred add!!(y, x)
    @test z2 isa JLVector{Float64}
    @test deepcollect(z2) ≈ (deepcollect(x) .+ deepcollect(y))
    z2 = @constinferred add!!(y, x, α)
    @test deepcollect(z2) ≈ (muladd.(deepcollect(x), α, deepcollect(y)))
    z2 = @constinferred add!!(y, x, α, β)
    @test deepcollect(z2) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* β))

    α, β = randn(ComplexF64, 2)
    z4 = add(y, x, α)
    @test z4 isa JLVector{ComplexF64}
    @test deepcollect(z4) ≈ (muladd.(deepcollect(x), α, deepcollect(y)))
    z4 = add(y, x, α, β)
    @test deepcollect(z4) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* β))

    z5 = @constinferred add!!(y, x, α)
    @test z5 isa JLVector{ComplexF64}
    @test deepcollect(z5) ≈ (muladd.(deepcollect(x), α, deepcollect(y)))
    z5 = @constinferred add!!(y, x, α, β)
    @test deepcollect(z5) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* β))

    # test strong zero
    α = randn(ComplexF64)
    z6 = add(y, x, α, Zero())
    @test deepcollect(z6) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* Zero()))
    z6 = add(y, x, α, false)
    @test deepcollect(z6) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* false))

    α = randn(scalartype(x))
    z6 = deepcopy(nan_y)
    z6 = @constinferred add!(z6, x, α, Zero())
    @test !any(isnan, z6)
    @test deepcollect(z6) ≈ (muladd.(deepcollect(x), α, deepcollect(nan_y) .* Zero()))
    z6 = deepcopy(nan_y)
    z6 = @constinferred add!(z6, x, α, false)
    @test !any(isnan, z6)
    @test deepcollect(z6) ≈ (muladd.(deepcollect(x), α, deepcollect(nan_y) .* false))
    z6 = deepcopy(nan_y)
    z6 = @constinferred add!(z6, x, α, 0.0)
    @test any(isnan, z6)
end

@testset "inner" begin
    s = @constinferred inner(x, y)
    @test s ≈ inner(deepcollect(x), deepcollect(y))

    α, β = randn(ComplexF64, 2)
    s2 = @constinferred inner(scale(x, α), scale(y, β))
    @test s2 ≈ inner(α * deepcollect(x), β * deepcollect(y))
end

end
