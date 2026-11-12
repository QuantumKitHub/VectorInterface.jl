module JLScalarTests
using VectorInterface
using JLArrays
using Test
using TestExtras

JLArrays.GPUArrays.allowscalar(false)

# count the number of device-to-host transfers of `Scalar` values
const DOWNLOADS = Ref(0)
function VectorInterface._hostvalue(data::JLArray)
    DOWNLOADS[] += 1
    return only(Array(data))
end
macro nosync(ex)
    return quote
        local n = DOWNLOADS[]
        local r = $(esc(ex))
        @test DOWNLOADS[] == n
        r
    end
end

dscalar(x::Number) = Scalar(JLArray(fill(x)))

x = JLVector(randn(5))
y = JLVector(randn(5))
xh = collect(x)
yh = collect(y)

@testset "construction and access" begin
    s = dscalar(2.0)
    @test s isa Scalar{Float64, JLArray{Float64, 0}}
    @test Scalar(JLArray([3.0])) isa Scalar{Float64, JLArray{Float64, 0}}
    n = DOWNLOADS[]
    @test s[] === 2.0
    @test DOWNLOADS[] == n + 1
    @nosync s[] = 3.0
    @test s[] === 3.0
    @test convert(Float64, s) === 3.0
end

@testset "promotion" begin
    s = dscalar(2.0)
    S = typeof(s)
    @test promote_type(S, ComplexF32) === Scalar{ComplexF64, JLArray{ComplexF64, 0}}
    # device storage wins over host storage
    @test promote_type(S, Scalar{Float32, Array{Float32, 0}}) === S
    @test promote_type(Scalar{Float32, Base.RefValue{Float32}}, S) === S
    a, b = @nosync promote(s, 1.0)
    @test b isa S
    @test b[] === 1.0
end

@testset "arithmetic without synchronisation" begin
    s = dscalar(2.0)
    S = typeof(s)
    r = @nosync @constinferred s + 1.0
    @test r isa S
    r = @nosync @constinferred 3.0 * s * s
    @test r isa S
    @test r[] === 12.0
    r = @nosync @constinferred s / 1im
    @test r isa Scalar{ComplexF64, JLArray{ComplexF64, 0}}
    r = @nosync @constinferred muladd(s, 2.0, s)
    @test r[] === 6.0
    r = @nosync @constinferred sqrt(abs(-s))
    @test r[] ≈ sqrt(2.0)
    r = @nosync @constinferred conj(s * 1im)
    @test r[] == -2.0im
    # host storage is cheaply uploaded
    r = @nosync Scalar(fill(3.0)) * s
    @test r isa S
    r = @nosync Scalar(Ref(3.0)) - s
    @test r isa S
    @test r[] === 1.0
end

@testset "vector interface on Scalar" begin
    s = dscalar(2.0)
    S = typeof(s)
    z = @nosync @constinferred zerovector(s, ComplexF64)
    @test z isa Scalar{ComplexF64, JLArray{ComplexF64, 0}}
    @test iszero(z)
    t = @nosync @constinferred scale(s, 3.0)
    @test t isa S
    @test t[] === 6.0
    t = copy(s)
    @test (@nosync scale!!(t, 3.0)) === t
    @test t[] === 6.0
    t2 = @nosync scale!!(t, 1im)
    @test t2 isa Scalar{ComplexF64, JLArray{ComplexF64, 0}}
    t = copy(s)
    @test (@nosync add!!(t, s, s, s)) === t
    @test t[] === 8.0
    @test (@nosync add(dscalar(NaN), 1.0, 1.0, Zero()))[] === 1.0
    t = @nosync @constinferred inner(s, s)
    @test t[] === 4.0
end

@testset "arrays with Scalar coefficients" begin
    s = dscalar(2.0)
    sc = dscalar(2.0im)
    z = @nosync @constinferred scale(x, s)
    @test z isa JLVector{Float64}
    @test collect(z) ≈ 2 .* xh
    z = @nosync @constinferred scale(x, sc)
    @test z isa JLVector{ComplexF64}
    @test collect(z) ≈ 2im .* xh
    z = copy(x)
    @test (@nosync scale!(z, s)) === z
    @test collect(z) ≈ 2 .* xh
    z = copy(y)
    @test (@nosync scale!(z, x, s)) === z
    @test collect(z) ≈ 2 .* xh
    z = copy(x)
    @test (@nosync scale!!(z, s)) === z
    z = @nosync scale!!(copy(x), sc)
    @test z isa JLVector{ComplexF64}
    @test collect(z) ≈ 2im .* xh

    z = @nosync @constinferred add(y, x, s, s)
    @test z isa JLVector{Float64}
    @test collect(z) ≈ 2 .* (xh .+ yh)
    z = @nosync @constinferred add(y, x, s)
    @test collect(z) ≈ yh .+ 2 .* xh
    z = copy(y)
    @test (@nosync add!(z, x, s, s)) === z # bypasses BLAS
    @test collect(z) ≈ 2 .* (xh .+ yh)
    z = copy(y)
    @test (@nosync add!(z, x, s, Zero())) === z
    @test collect(z) ≈ 2 .* xh
    z = copy(y)
    @test (@nosync add!!(z, x, s)) === z
    @test collect(z) ≈ yh .+ 2 .* xh
    z = @nosync add!!(copy(y), x, sc)
    @test z isa JLVector{ComplexF64}
    @test collect(z) ≈ yh .+ 2im .* xh

    # host scalars are passed as host values
    z = @nosync scale(x, Scalar(fill(2.0)))
    @test collect(z) ≈ 2 .* xh
    z = copy(y)
    @nosync add!(z, x, Scalar(Ref(2.0)))
    @test collect(z) ≈ yh .+ 2 .* xh
end

@testset "inner! and inner!!" begin
    r = dscalar(0.0)
    @test (@nosync inner!(r, x, y)) === r
    @test r[] ≈ inner(xh, yh)
    r = dscalar(0.0)
    @test (@nosync inner!!(r, x, y)) === r
    xc = JLVector(randn(ComplexF64, 5))
    r2 = @nosync inner!!(r, xc, y)
    @test r2 isa Scalar{ComplexF64, JLArray{ComplexF64, 0}}
    @test r2[] ≈ inner(collect(xc), yh)
    X = JLArray(randn(3, 4))
    Y = JLArray(randn(3, 4))
    r = dscalar(0.0)
    @nosync inner!(r, X, Y)
    @test r[] ≈ inner(collect(X), collect(Y))

    # typical use: device-side Gram-Schmidt step
    r = dscalar(0.0)
    xn = scale(x, inv(norm(x)))
    r = @nosync inner!!(r, xn, y)
    z = @nosync add!!(copy(y), xn, -r)
    @test abs(inner(xn, z)) < 1.0e-12
end

end
