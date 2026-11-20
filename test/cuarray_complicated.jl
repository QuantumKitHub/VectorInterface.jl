using VectorInterface
using Test
using TestExtras
using CUDA

using VectorInterface: One

if CUDA.functional()

    deepcollect(x::CuArray) = collect(x)
    deepcollect(x) = vcat(map(deepcollect, x)...)
    deepcollect(x::Number) = x
    deepcollect(x::Scalar) = deepcollect(x.a)

    x = (CUDA.randn(Float64, 2, 2), CUDA.rand(Float64, 2, 2), (CUDA.randn(Float64, 5, 2), CUDA.randn(Float64, 3, 2), CUDA.randn(Float64, 2, 2)), CUDA.randn(Float64, 1, 2), (CUDA.randn(Float64, 4, 2),))
    y = (CUDA.randn(Float64, 2, 2), CUDA.rand(Float64, 2, 2), (CUDA.randn(Float64, 5, 2), CUDA.randn(Float64, 3, 2), CUDA.randn(Float64, 2, 2)), CUDA.randn(Float64, 1, 2), (CUDA.randn(Float64, 4, 2),))


    @testset "scalartype" begin
        s = @constinferred scalartype(x)
        @test s == Float64
        @test_throws ArgumentError scalartype(())
        @test_throws ArgumentError scalartype((randn(Float64, 2), randn(ComplexF64, 3)))
    end

    @testset "zerovector" begin
        z = @constinferred zerovector(x)
        @test all(iszero, deepcollect(z))
        @test all(deepcollect(z) .=== zero(scalartype(x)))
        z1 = @constinferred zerovector!!(deepcopy(x))
        @test all(deepcollect(z1) .=== zero(scalartype(x)))

        z2 = @constinferred zerovector(x, ComplexF64)
        @test all(deepcollect(z2) .=== zero(ComplexF64))
        z3 = @constinferred zerovector!!(deepcopy(x), ComplexF64)
        @test all(deepcollect(z3) .=== zero(ComplexF64))

        xy = [deepcopy(x), deepcopy(y)]
        z4 = @constinferred zerovector!(xy)
        @test all(deepcollect(z4) .=== zero(scalartype(xy)))
        @test_throws MethodError zerovector!(xy, ComplexF64)
    end

    @testset "scale" begin
        for α in (randn(), Scalar(CuVector([randn()])))
            z = @constinferred scale(x, α)
            @test all(deepcollect(z) .== deepcollect(α) .* deepcollect(x))
            z = @constinferred scale(x, One())
            @test all(deepcollect(z) .== deepcollect(x))
            z2 = @constinferred scale!!(deepcopy(x), α)
            @test deepcollect(z2) ≈ (deepcollect(α) .* deepcollect(x))
            z2 = @constinferred scale!!(deepcopy(x), One())
            @test all(deepcollect(z2) .== deepcollect(x))
            xcopy = deepcopy(x)
            z2 = @constinferred scale!!(deepcopy(y), xcopy, α)
            @test deepcollect(z2) ≈ (deepcollect(α) .* deepcollect(x))
            @test all(deepcollect(xcopy) .== deepcollect(x))

            xy = [deepcopy(x), deepcopy(y)]
            z3 = @constinferred scale!(deepcopy(xy), α)
            @test deepcollect(z3) ≈ (deepcollect(α) .* deepcollect(xy))
            xycopy = deepcopy(xy)
            z3 = @constinferred scale!(zerovector(xy), xycopy, α)
            @test deepcollect(z3) ≈ (deepcollect(α) .* deepcollect(xy))
            @test all(deepcollect(xycopy) .== deepcollect(xy))
        end

        for α in (randn(ComplexF64), Scalar(CuVector([randn(ComplexF64)])))
            z4 = @constinferred scale(x, α)
            @test deepcollect(z4) ≈ (deepcollect(α) .* deepcollect(x))
            xcopy = deepcopy(x)
            z5 = @constinferred scale!!(xcopy, α)
            @test deepcollect(z5) ≈ (deepcollect(α) .* deepcollect(x))
            @test all(deepcollect(xcopy) .== deepcollect(x))

            xcopy = deepcopy(x)
            z6 = @constinferred scale!!(zerovector(x), xcopy, α)
            @test deepcollect(z6) ≈ (deepcollect(α) .* deepcollect(x))
            @test all(deepcollect(xcopy) .== deepcollect(x))
            xz = @constinferred zerovector(x, ComplexF64)
            z6 = @constinferred scale!!(xz, xcopy, α)
            @test deepcollect(z6) ≈ (deepcollect(α) .* deepcollect(x))
            @test all(deepcollect(xcopy) .== deepcollect(x))
        end
    end

    @testset "add" begin
        z = @constinferred add(y, x)
        @test all(deepcollect(z) .== deepcollect(x) .+ deepcollect(y))
        z = @constinferred add(y, x, One(), One())
        @test all(deepcollect(z) .== deepcollect(x) .+ deepcollect(y))
        z = @constinferred add(y, x, One())
        @test all(deepcollect(z) .== deepcollect(x) .+ deepcollect(y))
        z2 = @constinferred add!!(deepcopy(y), deepcopy(x))
        @test deepcollect(z2) ≈ (deepcollect(x) .+ deepcollect(y))
        z2 = @constinferred add!!(deepcopy(y), deepcopy(x), One(), One())
        @test all(deepcollect(z2) .== deepcollect(x) .+ deepcollect(y))
        z2 = @constinferred add!!(deepcopy(y), deepcopy(x), One())
        @test all(deepcollect(z2) .== deepcollect(x) .+ deepcollect(y))
        xy = [deepcopy(x), deepcopy(y)]
        yx = [deepcopy(y), deepcopy(x)]
        z3 = @constinferred add!(deepcopy(xy), deepcopy(yx))
        @test deepcollect(z3) ≈ (deepcollect(xy) .+ deepcollect(yx))

        for (α, β) in (
                (randn(), randn()),
                (Scalar(CuVector([randn()])), Scalar(CuVector([randn()]))),
            )
            a, b = deepcollect(α)[], deepcollect(β)[]
            # for some reason, on some Julia versions on some platforms, but only in test mode
            # there is a small floating point discrepancy, which makes exact comparisons fail
            z = @constinferred add(y, x, α)
            @test deepcollect(z) ≈ muladd.(deepcollect(x), a, deepcollect(y))
            z = @constinferred add(y, x, α, β)
            @test deepcollect(z) ≈ muladd.(deepcollect(x), a, deepcollect(y) .* b)
            z = @constinferred add(y, x, One(), β)
            @test deepcollect(z) ≈ muladd.(deepcollect(y), b, deepcollect(x))
            z = @constinferred add(y, x, α, One())
            @test deepcollect(z) ≈ muladd.(deepcollect(x), a, deepcollect(y))

            z2 = @constinferred add!!(deepcopy(y), deepcopy(x), α)
            @test deepcollect(z2) ≈ (muladd.(deepcollect(x), a, deepcollect(y)))
            z2 = @constinferred add!!(deepcopy(y), deepcopy(x), α, β)
            @test deepcollect(z2) ≈ (muladd.(deepcollect(x), a, deepcollect(y) .* b))
            z2 = @constinferred add!!(deepcopy(y), deepcopy(x), One(), β)
            @test deepcollect(z2) ≈ muladd.(deepcollect(y), b, deepcollect(x))
            z2 = @constinferred add!!(deepcopy(y), deepcopy(x), α, One())
            @test deepcollect(z2) ≈ muladd.(deepcollect(x), a, deepcollect(y))

            z3 = @constinferred add!(deepcopy(xy), deepcopy(yx), α)
            @test deepcollect(z3) ≈ (muladd.(deepcollect(yx), a, deepcollect(xy)))
            z3 = @constinferred add!(deepcopy(xy), deepcopy(yx), α, β)
            @test deepcollect(z3) ≈ (muladd.(deepcollect(yx), a, deepcollect(xy) .* b))
            z3 = @constinferred add!!(deepcopy(xy), deepcopy(yx), α, β)
            @test deepcollect(z3) ≈ (muladd.(deepcollect(yx), a, deepcollect(xy) .* b))
        end

        for (α, β) in (
                (randn(ComplexF64), randn(ComplexF64)),
                (Scalar(CuVector([randn(ComplexF64)])), Scalar(CuVector([randn(ComplexF64)]))),
            )
            a, b = deepcollect(α)[], deepcollect(β)[]
            z4 = @constinferred add(y, x, α)
            @test deepcollect(z4) ≈ (muladd.(deepcollect(x), a, deepcollect(y)))
            z4 = @constinferred add(y, x, α, β)
            @test deepcollect(z4) ≈ (muladd.(deepcollect(x), a, deepcollect(y) .* b))

            z5 = @constinferred add!!(deepcopy(y), deepcopy(x), α)
            @test deepcollect(z5) ≈ (muladd.(deepcollect(x), a, deepcollect(y)))
            z5 = @constinferred add!!(deepcopy(y), deepcopy(x), α, β)
            @test deepcollect(z5) ≈ (muladd.(deepcollect(x), a, deepcollect(y) .* b))

            xy = [deepcopy(x), deepcopy(y)]
            yx = [deepcopy(y), deepcopy(x)]
            z5 = @constinferred add!!(deepcopy(xy), deepcopy(yx), α, β)
            @test deepcollect(z5) ≈ (muladd.(deepcollect(yx), a, deepcollect(xy) .* b))

            # test strong zero
            z6 = @constinferred add(y, x, α, Zero())
            @test deepcollect(z6) ≈ (muladd.(deepcollect(x), a, deepcollect(y) .* Zero()))
            z6 = @constinferred add(y, x, α, false)
            @test deepcollect(z6) ≈ (muladd.(deepcollect(x), a, deepcollect(y) .* false))
        end
    end

    @testset "inner" begin
        s = @constinferred inner(x, y)
        @test convert(Float64, s) ≈ inner(deepcollect(x), deepcollect(y))

        for (α, β) in (
                (randn(ComplexF64), randn(ComplexF64)),
                (Scalar(CuVector([randn(ComplexF64)])), Scalar(CuVector([randn(ComplexF64)]))),
            )
            s2 = @testinferred inner(scale(x, α), scale(y, β))
            @test convert(ComplexF64, s2) ≈ inner(deepcollect(α)[] .* deepcollect(x), deepcollect(β)[] .* deepcollect(y))
        end
    end

    @testset "norm" begin
        @testset for p in (-1, 1, 2, 3.5, Inf)
            # LinearAlgebra.norm on a nested container takes the p-norm of the
            # 2-norms of its elements, not the p-norm of the flattened vector
            ref = norm(map(xi -> norm(deepcollect(xi)), x), p)
            # LinearAlgebra.norm on heterogeneous tuples only infers from Julia 1.12 on
            s = VERSION ≥ v"1.12" ? (@testinferred norm(x, p)) : norm(x, p)
            @test convert(Float64, s) ≈ ref
        end
    end
end
