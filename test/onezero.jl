using Test
using VectorInterface
using LinearAlgebra

const Z = Zero()
const I = One()
const typelist = (
    Int32, Int64, Float16, Float32, Float64, ComplexF16, ComplexF32,
    ComplexF64, BigFloat,
)

@testset "equalities" begin
    @test I == 1
    @test Z == 0
    @test I != Z
    @test I == I
    @test Z == Z

    @test iszero(Z) == true
    @test isone(I) == true
    @test isone(Z) == false
    @test iszero(I) == false

    @test isreal(Z)
    @test isreal(I)
    @test isinteger(Z)
    @test isinteger(I)
    @test isfinite(Z)
    @test isfinite(I)
    @test !isinf(Z)
    @test !isinf(I)
    @test !isnan(Z)
    @test !isnan(I)
    @test !signbit(Z)
    @test !signbit(I)
end

@testset "traits" begin
    @test @inferred(real(Z)) === Z
    @test @inferred(real(I)) === I
    @test @inferred(imag(Z)) === Z
    @test @inferred(imag(I)) === Z
    @test real(Zero) === Zero
    @test real(One) === One

    @test @inferred(abs(Z)) === Z
    @test @inferred(abs(I)) === I
    @test @inferred(abs2(Z)) === Z
    @test @inferred(abs2(I)) === I
    @test @inferred(sign(Z)) === Z
    @test @inferred(sign(I)) === I

    # inherited from `Real`/`Number`, pinned here so a Base change shows up as a failure
    @test @inferred(conj(Z)) === Z
    @test @inferred(conj(I)) === I
    @test @inferred(adjoint(Z)) === Z
    @test @inferred(adjoint(I)) === I
    @test @inferred(transpose(Z)) === Z
    @test @inferred(transpose(I)) === I
    @test @inferred(oneunit(Z)) === I
    @test @inferred(oneunit(I)) === I
    @test Z^2 === Z  # not `@inferred`: `Zero()^n` is `Union{Zero, One}`, since `Zero()^0 == One()`
    @test @inferred(I^2) === I
    @test Z^0 === I
    @test @inferred(float(Z)) === 0.0
    @test @inferred(float(I)) === 1.0
end

@testset "ordering" begin
    @test !(Z < Z)
    @test !(I < I)
    @test Z < I
    @test !(I < Z)
    @test Z <= Z
    @test I <= I
    @test Z <= I
    @test !(I <= Z)

    @test isless(Z, I)
    @test !isless(I, Z)
    @test !isless(Z, Z)
    @test cmp(Z, I) == -1
    @test cmp(I, Z) == 1

    @test @inferred(min(Z, I)) === Z
    @test @inferred(min(I, Z)) === Z
    @test @inferred(max(Z, I)) === I
    @test @inferred(max(I, Z)) === I

    for T in typelist
        T <: Real || continue
        @test Z < one(T)
        @test !(Z < zero(T))
        @test Z <= zero(T)
        @test one(T) > Z
        @test zero(T) < I
        @test isless(Z, one(T))
        @test isless(zero(T), I)
    end
end

@testset "hashing" begin
    @test hash(Z) == hash(0)
    @test hash(I) == hash(1)
    @test isequal(Z, 0)
    @test isequal(I, 1)
    @test !isequal(Z, I)
    # the reason the hashes have to agree in the first place
    d = Dict(0 => :a, 1 => :b)
    @test d[Z] === :a
    @test d[I] === :b
    @test length(Set([Z, 0, false])) == 1
end

@testset "isapprox" begin
    @test Z ≈ Z
    @test I ≈ I
    @test !(Z ≈ I)

    for T in typelist
        @test Z ≈ zero(T)
        @test I ≈ one(T)
        @test !(Z ≈ one(T))
        @test !(I ≈ zero(T))
        @test zero(T) ≈ Z
        @test one(T) ≈ I
        if T <: AbstractFloat
            @test isapprox(Z, T(1.0e-3); atol = 1.0e-1)
            @test !isapprox(Z, T(1.0e-3); atol = 1.0e-6)
        end
    end
end

@testset "arithmetic" begin
    @test Z + Z === Z
    @test I + Z === I
    @test Z + I === I
    @test I + I == 2

    @test @inferred(-Z) == Z
    @test @inferred(-I) == -1
    @test @inferred(I - I) === Z
    @test @inferred(I - Z) === I
    @test @inferred(Z - I) == -I

    @test @inferred(Z * Z) === Z
    @test @inferred(I * Z) === Z
    @test @inferred(Z * I) === Z
    @test @inferred(I * I) === I

    @test @inferred(Z / I) === Z
    @test @inferred(I / I) === I
    @test_throws DivideError @inferred(I / Z)
    @test_throws DivideError @inferred(Z / Z)

    for T in typelist
        x = rand(T)
        while iszero(x)
            x = rand(T)
        end

        @test @inferred(Z + x) == x
        @test @inferred(x + Z) == x
        @test @inferred(I + x) == x + 1
        @test @inferred(x + I) == x + 1

        @test @inferred(Z - x) == -x
        @test @inferred(x - Z) == x
        @test @inferred(I - x) == 1 - x
        @test @inferred(x - I) == x - 1

        @test @inferred(Z * x) == zero(x)
        @test @inferred(x * Z) == zero(x)
        @test @inferred(I * x) == x
        @test @inferred(x * I) == x

        @test_throws DivideError @inferred(x / Z)
        @test @inferred(x / I) == x
        @test @inferred(Z / x) == zero(x)
        @test @inferred(I / x) == inv(x)
    end
end

@testset "promotion" begin
    for T in typelist
        @test @inferred(promote_type(typeof(I), T)) == T
        @test @inferred(promote_type(typeof(Z), T)) == T
        @test @inferred(promote_type(T, typeof(I))) == T
        @test @inferred(promote_type(T, typeof(Z))) == T

        @test @inferred(promote_type(One, Zero, T)) == T
        @test @inferred(promote_type(One, T, Zero)) == T
        @test @inferred(promote_type(T, One, Zero)) == T
        @test @inferred(promote_type(Zero, One, T)) == T
        @test @inferred(promote_type(Zero, T, One)) == T
        @test @inferred(promote_type(T, Zero, One)) == T

        @test @inferred(T(I)) == one(T)
        @test @inferred(T(Z)) == zero(T)
        @test @inferred(convert(T, I)) == one(T)
        @test @inferred(convert(T, Z)) == zero(T)
    end
end

@testset "complex disambiguation" begin
    # `<: Real` makes Base's Real-vs-Complex methods applicable; these cover the methods
    # added to resolve the resulting ambiguities.
    for x in (im, ComplexF32(1.0f0, 2.0f0), ComplexF64(1.0, 2.0), Complex{Int}(3, -4))
        @test @inferred(Z + x) == x
        @test @inferred(x + Z) == x
        @test @inferred(Z - x) == -x
        @test @inferred(x - Z) == x
        @test @inferred(Z * x) == zero(x)
        @test @inferred(x * Z) == zero(x)
        @test @inferred(I * x) == x
        @test @inferred(x * I) == x
        @test @inferred(x / I) == x
        @test @inferred(I / x) == inv(x)
        @test @inferred(Z / x) == zero(x)
        @test_throws DivideError x / Z
    end

    @test @inferred(Bool(Z)) === false
    @test @inferred(Bool(I)) === true
    @test @inferred(Complex(Z)) == 0
    @test @inferred(Complex(I)) == 1
    @test @inferred(Complex{Float64}(Z)) === ComplexF64(0.0)
    @test @inferred(Complex{Float64}(I)) === ComplexF64(1.0)
end

@testset "LinearAlgebra scalars" begin
    @test norm(Z) === 0.0
    @test norm(I) === 1.0
    @test @inferred(dot(Z, I)) === Z
    @test @inferred(dot(I, I)) === I

    # 5-arg `mul!` is what motivated widening the utility surface in the first place
    for T in (Float64, ComplexF64)
        A, B = rand(T, 3, 3), rand(T, 3, 3)
        @test mul!(zeros(T, 3, 3), A, B, I, Z) ≈ A * B
        C = rand(T, 3, 3)
        @test mul!(copy(C), A, B, I, I) ≈ C + A * B

        x, y = rand(T, 3), rand(T, 3)
        @test mul!(zeros(T, 3), A, x, I, Z) ≈ A * x
        @test axpy!(I, x, copy(y)) ≈ y + x
        @test rmul!(copy(x), I) ≈ x
        @test lmul!(I, copy(x)) ≈ x
        @test rmul!(copy(x), Z) ≈ zero(x)
    end
end
