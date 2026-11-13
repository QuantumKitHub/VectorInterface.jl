module StaticSVec
using VectorInterface
using JLArrays
using Test
using TestExtras

deepcollect(x::JLArray) = collect(x)
deepcollect(x::Number) = x
deepcollect(x::Scalar) = deepcollect(x.a)

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
    for α in (randn(), Scalar(JLVector([randn()])))
        z = @constinferred scale(x, α)
        @test z isa JLVector{Float64}
        @test all(deepcollect(z) .== deepcollect(x) .* α)

        z = @constinferred scale(x, α)
        @test z isa JLVector{Float64}
        @test all(deepcollect(z) .== deepcollect(x) .* α)

        z2 = @constinferred scale!!(x, α)
        @test z2 isa JLVector{Float64}
        @test deepcollect(z2) ≈ deepcollect(x) .* deepcollect(α)
        z2 = @constinferred scale!!(y, x, α)
        @test z2 isa JLVector{Float64}
        @test deepcollect(z2) ≈ deepcollect(x) .* deepcollect(α)
    end

    for α in (randn(ComplexF64), Scalar(JLVector([randn(ComplexF64)])))
        z4 = @constinferred scale(x, α)
        @test z4 isa JLVector{ComplexF64}
        @test deepcollect(z4) ≈ (α .* deepcollect(x))
        z5 = @constinferred scale!!(x, α)
        @test z5 isa JLVector{ComplexF64}
        @test deepcollect(z5) ≈ (α .* deepcollect(x))

        z6 = @constinferred scale!!(zerovector(x), x, α)
        @test z6 isa JLVector{ComplexF64}
        @test deepcollect(z6) ≈ (α .* deepcollect(x))
    end

    for α in (randn(Float64), Scalar(JLVector([randn(Float64)])))
        ycomplex = zerovector(y, ComplexF64)
        z8 = @constinferred scale!!(ycomplex, x, α)
        @test scalartype(z8) == ComplexF64
        @test all(deepcollect(z8) .== α .* deepcollect(x))
    end
end

@testset "add" begin
    for (α, β) in (
            (randn(), randn()),
            (Scalar(JLVector([randn()])), Scalar(JLVector([randn()]))),
        )
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
    end

    for (α, β) in (
            (randn(ComplexF64), randn(ComplexF64)),
            (Scalar(JLVector([randn(ComplexF64)])), Scalar(JLVector([randn(ComplexF64)]))),
        )
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
    end

    for α in (randn(ComplexF64), Scalar(JLVector([randn(ComplexF64)])))
        # test strong zero
        α = randn(ComplexF64)
        z6 = add(y, x, α, Zero())
        @test deepcollect(z6) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* Zero()))
        z6 = add(y, x, α, false)
        @test deepcollect(z6) ≈ (muladd.(deepcollect(x), α, deepcollect(y) .* false))
    end

    for α in (randn(scalartype(x)), Scalar(JLVector([randn(scalartype(x))])))
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
end

@testset "inner" begin
    s = @constinferred inner(x, y)
    @test s ≈ inner(deepcollect(x), deepcollect(y))

    result = Scalar(JLVector(zeros(scalartype(x), 1)))
    inner!(result, x, y)
    @test result[] ≈ inner(deepcollect(x), deepcollect(y))

    for (α, β) in (
            (randn(ComplexF64), randn(ComplexF64)),
            (Scalar(JLVector([randn(ComplexF64)])), Scalar(JLVector([randn(ComplexF64)]))),
        )
        s2 = @constinferred inner(scale(x, α), scale(y, β))
        @test s2 ≈ inner(α * deepcollect(x), β * deepcollect(y))

        result = Scalar(JLVector(zeros(ComplexF64, 1)))
        inner!(result, scale(x, α), scale(y, β))
        @test result[] ≈ s2
        # hit the AbstractArray case too
        inner!(result, adjoint(scale(x, α)), adjoint(scale(y, β)))
        @test result[] ≈ s2
    end
end

@testset "norm" begin
    for p in (-1, 1, 2, 3.5, Inf)
        s = @constinferred norm(x, p)
        @test s ≈ norm(deepcollect(x), p)

        result = Scalar(JLVector(zeros(scalartype(x), 1)))
        norm!(result, x, p)
        @test result[] ≈ norm(deepcollect(x), p)
        # hit the AbstractArray case too
        norm!(result, adjoint(x), p)
        @test result[] ≈ norm(adjoint(deepcollect(x)), p)
    end
end

end
