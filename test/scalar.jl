module ScalarTests
using VectorInterface
using VectorInterface: storagetype
using Test
using TestExtras

@testset "construction" begin
    s = Scalar(fill(1.5))
    @test s isa Scalar{Float64, Array{Float64, 0}}
    @test s[] === 1.5
    @test Scalar(s) === s

    s = Scalar([2.0;;])
    @test s isa Scalar{Float64, Array{Float64, 0}}
    @test s[] === 2.0

    v = [1.0, 2.0, 3.0]
    s = Scalar(view(v, 2:2))
    @test s[] === 2.0
    s[] = 5.0 # writes through to the parent array
    @test v[2] === 5.0

    s = Scalar(Ref(3))
    @test s isa Scalar{Int, Base.RefValue{Int}}
    @test Scalar(3) isa Scalar{Int, Base.RefValue{Int}}
    @test Scalar{Float64}(3) isa Scalar{Float64, Base.RefValue{Float64}}
    @test Scalar{Float32}(Scalar(fill(1.0))) isa Scalar{Float32, Array{Float32, 0}}
    @test Scalar{Float64, Array{Float64, 0}}(2) isa Scalar{Float64, Array{Float64, 0}}
    @test Scalar{Float64, Array{Float64, 0}}(One())[] === 1.0
    @test Scalar{Float64}(Zero())[] === 0.0
    @test Scalar(One())[] === 1

    @test_throws DimensionMismatch Scalar([1.0, 2.0])
    @test_throws ArgumentError Scalar{Float32, Array{Float64, 0}}(1)

    @test sprint(show, Scalar(fill(1.5))) == "Scalar(fill(1.5))"
end

@testset "storage types" begin
    @test storagetype(Array{Float64, 0}, ComplexF64) === Array{ComplexF64, 0}
    @test storagetype(Base.RefValue{Float64}, Int) === Base.RefValue{Int}
    A = typeof(reshape(view(randn(3), 1:1), ()))
    @test storagetype(A, Float32) === Array{Float32, 0}
end

@testset "conversion and promotion" begin
    s = Scalar(fill(2.0))
    S = typeof(s)
    @test promote_type(S, Float64) === S
    @test promote_type(Float64, S) === S
    @test promote_type(S, ComplexF32) === Scalar{ComplexF64, Array{ComplexF64, 0}}
    @test promote_type(S, Int) === S
    @test promote_type(S, One) === S
    @test promote_type(S, Bool) === S
    @test promote_type(Scalar{Float32, Base.RefValue{Float32}}, S) === S
    @test promote_type(S, Scalar{Float32, Base.RefValue{Float32}}) === S

    a, b = @constinferred promote(s, 1.0im)
    @test a isa Scalar{ComplexF64, Array{ComplexF64, 0}}
    @test b isa Scalar{ComplexF64, Array{ComplexF64, 0}}
    @test b[] === 1.0im

    @test convert(Float64, s) === 2.0
    @test convert(ComplexF64, s) === 2.0 + 0.0im
    @test convert(S, s) === s
    @test convert(S, 3) isa S
    @test convert(Scalar{Float32, Array{Float32, 0}}, s)[] === 2.0f0
    @test Float64[s, 1] == [2.0, 1.0]
    @test [s, 1.0] isa Vector{S}

    @test real(Scalar{ComplexF64, Array{ComplexF64, 0}}) === Scalar{Float64, Array{Float64, 0}}
    @test float(Scalar{Int, Array{Int, 0}}) === Scalar{Float64, Array{Float64, 0}}
    @test complex(S) === Scalar{ComplexF64, Array{ComplexF64, 0}}
end

@testset "arithmetic with $(typeof(s))" for s in (Scalar(fill(2.0)), Scalar(Ref(2.0)))
    S = typeof(s)
    x = 3.0
    for f in (+, -, *, /, \, ^, min, max, hypot)
        r = @constinferred f(s, x)
        @test r isa S
        @test r[] ≈ f(2.0, x)
        r = @constinferred f(x, s)
        @test r isa S
        @test r[] ≈ f(x, 2.0)
        r = @constinferred f(s, s)
        @test r isa S
        @test r[] ≈ f(2.0, 2.0)
    end
    @test (s * 1im)[] === 2.0im
    @test (s^2)[] === 4.0
    @test (s^(1 // 2))[] ≈ sqrt(2.0)
    @test (ℯ^s)[] ≈ exp(2.0)
    for f in (-, conj, real, imag, abs, abs2, inv, sqrt, exp, log, sin, cos, tanh, sign, float)
        r = @constinferred f(s)
        @test r[] ≈ f(2.0)
    end
    @test (+s) === s
    @test complex(s)[] === 2.0 + 0.0im
    @test muladd(s, 3.0, 1.0)[] === 7.0
    @test muladd(2.0, s, s)[] === 6.0
    @test fma(s, s, s)[] === 6.0

    # interaction with `One` and `Zero`
    @test One() * s === s
    @test s * One() === s
    @test s / One() === s
    @test (One() / s)[] === 0.5
    @test (Zero() * s)[] === 0.0
    @test (s * Zero())[] === 0.0
    @test Zero() + s === s
    @test s - Zero() === s
    @test (Zero() - s)[] === -2.0
    @test_throws DivideError s / Zero()

    @test zero(s)[] === 0.0
    @test one(s)[] === 1.0
    @test zero(S)[] === 0.0
    @test one(S)[] === 1.0
    @test zero(s) isa S

    c = copy(s)
    c[] = 5.0
    @test s[] === 2.0
end

@testset "predicates and comparisons" begin
    s = Scalar(fill(2.0))
    @test s == 2
    @test 2 == s
    @test s == Scalar(Ref(2))
    @test isequal(s, 2.0)
    @test hash(s) == hash(2.0)
    @test s < 3
    @test 1 <= s
    @test isless(s, Scalar(Ref(3.0)))
    @test s ≈ 2.0 + 1.0e-12
    @test !iszero(s)
    @test iszero(zero(s))
    @test isone(one(s))
    @test isfinite(s)
    @test !isnan(s)
    @test isnan(Scalar(fill(NaN)))
    @test isreal(s)
    @test isinteger(s)
    @test Dict(s => 1)[2.0] == 1
end

@testset "broadcasting" begin
    s = Scalar(fill(2.0))
    x = [1.0, 2.0, 3.0]
    @test x .* s == [2.0, 4.0, 6.0]
    y = similar(x)
    y .= x .+ s
    @test y == [3.0, 4.0, 5.0]
end

@testset "vector interface for $(typeof(s))" for s in (Scalar(fill(2.0)), Scalar(Ref(2.0)))
    S = typeof(s)
    @test scalartype(s) === Float64
    @test scalartype(S) === Float64
    @test scalartype([s, s]) === Float64

    z = @constinferred zerovector(s)
    @test z isa S
    @test iszero(z)
    @test z !== s
    z = @constinferred zerovector(s, ComplexF64)
    @test z isa Scalar{ComplexF64}
    @test iszero(z)
    z = zerovector(s, S)
    @test z isa S
    t = copy(s)
    @test zerovector!(t) === t
    @test iszero(t)
    t = copy(s)
    @test zerovector!!(t) === t
    @test iszero(t)

    # scale
    t = @constinferred scale(s, 3.0)
    @test t isa S
    @test t[] === 6.0
    @test scale(Scalar(fill(NaN)), 0.0)[] === 0.0 # strong zero
    @test scale(NaN, Scalar(fill(0.0)))[] === 0.0
    t = @constinferred scale(s, 1.0im)
    @test t[] === 2.0im
    t = @constinferred scale(1.0, s)
    @test t isa S
    t = copy(s)
    @test scale!(t, 3.0) === t
    @test t[] === 6.0
    @test scale!(t, s, 0.5) === t
    @test t[] === 1.0
    @test_throws InexactError scale!(t, 1.0im)
    t = copy(s)
    @test scale!!(t, 3.0) === t
    @test t[] === 6.0
    @test scale!!(t, One()) === t
    t2 = @constinferred scale!!(t, 1.0im)
    @test t2 !== t
    @test t2[] === 6.0im
    @test t[] === 6.0
    t = copy(s)
    @test scale!!(t, s, 2.0) === t
    @test t[] === 4.0
    t2 = scale!!(t, s, 2.0im)
    @test t2[] === 4.0im
    @test scale!!(1.0, s) isa S
    @test scale!!(1.0, 2.0, s) isa S

    # add
    t = @constinferred add(s, 1.0)
    @test t isa S
    @test t[] === 3.0
    t = @constinferred add(s, 1.0, 2.0)
    @test t[] === 4.0
    t = @constinferred add(s, 1.0, 2.0, 3.0)
    @test t[] === 8.0
    t = @constinferred add(1.0, 1.0, s, 3.0)
    @test t isa S
    @test t[] === 5.0
    @test add(Scalar(fill(NaN)), 1.0, 1.0, Zero())[] === 1.0 # strong zero
    @test add(Scalar(fill(NaN)), 1.0, 1.0, Scalar(fill(0.0)))[] === 1.0
    @test add(1.0, Scalar(fill(NaN)), Scalar(fill(0.0)))[] === 1.0
    t = copy(s)
    @test add!(t, 1.0, 2.0, 3.0) === t
    @test t[] === 8.0
    @test add!(t, t) === t
    @test t[] === 16.0
    @test_throws InexactError add!(t, 1.0im)
    t = copy(s)
    @test add!!(t, s, s, s) === t
    @test t[] === 8.0
    t2 = @constinferred add!!(t, 1.0im)
    @test t2 !== t
    @test t2[] === 8.0 + 1.0im
    @test add!!(1.0, 1.0, s) isa S

    # inner
    t = @constinferred inner(s, s)
    @test t isa S
    @test t[] === 4.0
    t = @constinferred inner(s, 1.0im)
    @test t[] === 2.0im
    @test norm(s) == 2.0
end

@testset "inner! and inner!!" begin
    x = randn(4, 3)
    y = randn(4, 3)
    xc = randn(ComplexF64, 4, 3)
    for s in (Scalar(fill(0.0)), Scalar(Ref(0.0)))
        @test inner!(s, x, y) === s
        @test s[] ≈ inner(x, y)
        @test_throws InexactError inner!(s, xc, y)
        s2 = @constinferred inner!!(s, x, y)
        @test s2 === s
        s3 = inner!!(s, xc, y)
        @test s3 isa Scalar{ComplexF64}
        @test s3[] ≈ inner(xc, y)
        @test s[] ≈ inner(x, y)
    end
    # zero-dimensional arrays
    s = Scalar(fill(0.0))
    @test inner!(s, fill(2.0), fill(3.0))[] === 6.0
    # nested vectors and tuples
    s = Scalar(fill(0.0))
    @test inner!(s, [x, y], [y, x])[] ≈ 2 * inner(x, y)
    @test inner!(s, (x, 1.0), (y, 2.0))[] ≈ inner(x, y) + 2
    # plain numbers
    @test @constinferred(inner!!(0.0, x, y)) ≈ inner(x, y)
    @test_throws DimensionMismatch inner!(s, x, randn(3, 4))
end

@testset "arrays with Scalar coefficients" begin
    x = randn(3)
    y = randn(3)
    s = Scalar(fill(2.0))
    sc = Scalar(fill(2.0im))

    z = @constinferred scale(x, s)
    @test z isa Vector{Float64}
    @test z ≈ 2 .* x
    z = @constinferred scale(x, sc)
    @test z isa Vector{ComplexF64}
    z = copy(x)
    @test scale!(z, s) === z
    @test z ≈ 2 .* x
    z = copy(x)
    @test scale!!(z, s) === z
    z2 = scale!!(z, sc)
    @test z2 isa Vector{ComplexF64}
    @test z2 ≈ 4im .* x

    z = @constinferred add(y, x, s, s)
    @test z isa Vector{Float64}
    @test z ≈ 2 .* (x .+ y)
    z = copy(y)
    @test add!(z, x, s, s) === z
    @test z ≈ 2 .* (x .+ y)
    z = copy(y)
    @test add!!(z, x, s) === z
    @test z ≈ y .+ 2 .* x
    z2 = add!!(copy(y), x, sc)
    @test z2 isa Vector{ComplexF64}
    @test z2 ≈ y .+ 2im .* x

    # nested arrays: coefficients are passed on to the elements
    xx = [randn(2), randn(3)]
    z = @constinferred scale(xx, s)
    @test z ≈ 2 .* xx
    zz = deepcopy(xx)
    scale!(zz, s)
    @test zz ≈ 2 .* xx
    zz = deepcopy(xx)
    add!(zz, xx, s, s)
    @test zz ≈ 4 .* xx

    # tuples
    t = scale((1.0, 2.0), s)
    @test t isa Tuple{typeof(s), typeof(s)}
    @test t[1][] === 2.0
end

end
