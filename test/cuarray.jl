module CuVec
using VectorInterface
using CUDA
using Test
using TestExtras

deepcollect(x::CuArray) = collect(x)
deepcollect(x::Number) = x
deepcollect(x::Scalar) = deepcollect(x.a)

if CUDA.functional()

    x = CuVector(randn(3))
    y = CuVector(randn(3))
    nan_y = CuVector(vcat(randn(2), NaN))

    @testset "scalartype" begin
        s = @testinferred scalartype(x)
        @test s == Float64
    end

    @testset "zerovector" begin
        z = @testinferred zerovector(x)
        @test z isa CuVector{Float64}
        @test all(iszero, deepcollect(z))
        @test all(deepcollect(z) .=== zero(scalartype(x)))
        z1 = @testinferred zerovector!!(deepcopy(x))
        @test z1 isa CuVector{Float64}
        @test all(deepcollect(z1) .=== zero(scalartype(x)))

        z3 = @testinferred zerovector(x, ComplexF64)
        @test z3 isa CuVector{ComplexF64}
        @test all(deepcollect(z3) .=== zero(ComplexF64))
        z4 = @testinferred zerovector!!(x, ComplexF64)
        @test z4 isa CuVector{ComplexF64}
        @test all(deepcollect(z4) .=== zero(ComplexF64))
    end

    @testset "scale" begin
        for α in (randn(), Scalar(CuVector([randn()])))
            z = @testinferred scale(x, α)
            @test z isa CuVector{Float64}
            @test all(deepcollect(z) .== deepcollect(x) .* deepcollect(α))

            z = @testinferred scale(x, α)
            @test z isa CuVector{Float64}
            @test all(deepcollect(z) .== deepcollect(x) .* deepcollect(α))

            z2 = @testinferred scale!!(deepcopy(x), α)
            @test z2 isa CuVector{Float64}
            @test deepcollect(z2) ≈ deepcollect(x) .* deepcollect(α)
            z2 = @testinferred scale!!(deepcopy(y), x, α)
            @test z2 isa CuVector{Float64}
            @test deepcollect(z2) ≈ deepcollect(x) .* deepcollect(α)
        end

        for α in (randn(ComplexF64), Scalar(CuVector([randn(ComplexF64)])))
            z4 = @testinferred scale(x, α)
            @test z4 isa CuVector{ComplexF64}
            @test deepcollect(z4) ≈ (deepcollect(α) .* deepcollect(x))
            z5 = @testinferred scale!!(x, α)
            @test z5 isa CuVector{ComplexF64}
            @test deepcollect(z5) ≈ (deepcollect(α) .* deepcollect(x))

            z6 = @testinferred scale!!(zerovector(x), x, α)
            @test z6 isa CuVector{ComplexF64}
            @test deepcollect(z6) ≈ (deepcollect(α) .* deepcollect(x))
        end

        for α in (randn(Float64), Scalar(CuVector([randn(Float64)])))
            ycomplex = zerovector(y, ComplexF64)
            z8 = @testinferred scale!!(ycomplex, x, α)
            @test scalartype(z8) == ComplexF64
            @test all(deepcollect(z8) .== deepcollect(α) .* deepcollect(x))
        end
    end

    @testset "add" begin
        for (α, β) in (
                (randn(), randn()),
                (Scalar(CuVector([randn()])), Scalar(CuVector([randn()]))),
            )
            z = add(y, x)
            @test z isa CuVector{Float64}
            @test all(deepcollect(z) .== deepcollect(x) .+ deepcollect(y))
            z = add(y, x, α)
            @test deepcollect(z) ≈ muladd.(deepcollect(x), deepcollect(α)[], deepcollect(y))
            z = add(y, x, α, β)
            @test deepcollect(z) ≈ muladd.(deepcollect(x), deepcollect(α)[], deepcollect(y) .* deepcollect(β)[])

            z2 = @testinferred add!!(deepcopy(y), x)
            @test z2 isa CuVector{Float64}
            @test deepcollect(z2) ≈ (deepcollect(x) .+ deepcollect(y))
            z2 = @testinferred add!!(deepcopy(y), x, α)
            @test deepcollect(z2) ≈ (muladd.(deepcollect(x), deepcollect(α)[], deepcollect(y)))
            z2 = @testinferred add!!(deepcopy(y), x, α, β)
            @test deepcollect(z2) ≈ (muladd.(deepcollect(x), deepcollect(α)[], deepcollect(y) .* deepcollect(β)[]))
        end

        for (α, β) in (
                (randn(ComplexF64), randn(ComplexF64)),
                (Scalar(CuVector([randn(ComplexF64)])), Scalar(CuVector([randn(ComplexF64)]))),
            )
            z4 = add(y, x, α)
            @test z4 isa CuVector{ComplexF64}
            @test deepcollect(z4) ≈ (muladd.(deepcollect(x), deepcollect(α), deepcollect(y)))
            z4 = add(y, x, α, β)
            @test deepcollect(z4) ≈ (muladd.(deepcollect(x), deepcollect(α), deepcollect(y) .* deepcollect(β)))

            z5 = @testinferred add!!(deepcopy(y), x, α)
            @test z5 isa CuVector{ComplexF64}
            @test deepcollect(z5) ≈ (muladd.(deepcollect(x), deepcollect(α), deepcollect(y)))
            z5 = @testinferred add!!(deepcopy(y), x, α, β)
            @test deepcollect(z5) ≈ (muladd.(deepcollect(x), deepcollect(α), deepcollect(y) .* deepcollect(β)))
        end

        for α in (randn(ComplexF64), Scalar(CuVector([randn(ComplexF64)])))
            # test strong zero
            α = randn(ComplexF64)
            z6 = add(y, x, α, Zero())
            @test deepcollect(z6) ≈ (muladd.(deepcollect(x), deepcollect(α), deepcollect(y) .* Zero()))
            z6 = add(y, x, α, false)
            @test deepcollect(z6) ≈ (muladd.(deepcollect(x), deepcollect(α), deepcollect(y) .* false))
        end

        for α in (randn(scalartype(x)), Scalar(CuVector([randn(scalartype(x))])))
            z6 = deepcopy(nan_y)
            z6 = @testinferred add!(z6, x, α, Zero())
            @test !any(isnan, z6)
            @test deepcollect(z6) ≈ (muladd.(deepcollect(x), deepcollect(α), deepcollect(nan_y) .* Zero()))
            z6 = deepcopy(nan_y)
            z6 = @testinferred add!(z6, x, α, false)
            @test !any(isnan, z6)
            @test deepcollect(z6) ≈ (muladd.(deepcollect(x), deepcollect(α), deepcollect(nan_y) .* false))
            z6 = deepcopy(nan_y)
            z6 = @testinferred add!(z6, x, α, 0.0)
            @test any(isnan, z6)
        end
    end

    @testset "inner" begin
        s = @testinferred inner(x, y)
        @test convert(eltype(x), s) ≈ inner(deepcollect(x), deepcollect(y))

        for (α, β) in (
                (randn(ComplexF64), randn(ComplexF64)),
                (Scalar(CuVector([randn(ComplexF64)])), Scalar(CuVector([randn(ComplexF64)]))),
            )
            s2 = @testinferred inner(scale(x, α), scale(y, β))
            @test convert(ComplexF64, s2) ≈ inner(deepcollect(α) .* deepcollect(x), deepcollect(β) .* deepcollect(y))
        end
    end

    @testset "norm" begin
        for p in (-1, 1, 2, 3.5, Inf)
            s = @testinferred norm(x, p)
            @test convert(real(eltype(x)), s) ≈ norm(deepcollect(x), p)
        end
    end
end

end
