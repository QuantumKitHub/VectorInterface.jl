using VectorInterface
using VectorInterface: MinimalMVec, MinimalSVec, MinimalVec
using Enzyme, EnzymeTestUtils
using Test, TestExtras
using Random

Base.collect(x::MinimalVec) = x.vec

rng = Random.default_rng()

precision(::Type{T}) where {T <: Union{Float32, ComplexF32}} = sqrt(eps(Float32))
precision(::Type{T}) where {T <: Union{Float64, ComplexF64}} = 2 * sqrt(eps(Float64))

eltypes = (Float64, ComplexF64)

@testset "scale ($T)" for T in eltypes
    n = 12
    atol = rtol = n * precision(T)

    # Vector
    x = randn(T, n)
    y = randn(T, n)
    α = randn(T)
    for Tα in (Const, Active)
        test_reverse(scale, Duplicated, (x, Duplicated), (α, Tα); atol, rtol)
        test_reverse(scale!!, Duplicated, (x, Duplicated), (α, Tα); atol, rtol)
        test_reverse(scale!!, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα); atol, rtol)
    end
    for Tα in (Const, Duplicated)
        test_forward(scale, Duplicated, (x, Duplicated), (α, Tα); atol, rtol)
        test_forward(scale!!, Duplicated, (x, Duplicated), (α, Tα); atol, rtol)
        test_forward(scale!!, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα); atol, rtol)
    end

    # MinimalMVec
    mx = MinimalMVec(x)
    my = MinimalMVec(y)
    for Tα in (Const, Active)
        test_reverse(scale, Duplicated, (mx, Duplicated), (α, Tα); atol, rtol)
        test_reverse(scale!!, Duplicated, (mx, Duplicated), (α, Tα); atol, rtol)
        test_reverse(scale!!, Duplicated, (my, Duplicated), (mx, Duplicated), (α, Tα); atol, rtol)
    end
    for Tα in (Const, Duplicated)
        test_forward(scale, Duplicated, (mx, Duplicated), (α, Tα); atol, rtol)
        test_forward(scale!!, Duplicated, (mx, Duplicated), (α, Tα); atol, rtol)
        test_forward(scale!!, Duplicated, (my, Duplicated), (mx, Duplicated), (α, Tα); atol, rtol)
    end

    # MinimalSVec
    mx = MinimalSVec(x)
    my = MinimalSVec(y)
    for Tα in (Const, Active)
        test_reverse(scale, Duplicated, (mx, Duplicated), (α, Tα); atol, rtol)
        test_reverse(scale!!, Duplicated, (mx, Duplicated), (α, Tα); atol, rtol)
        test_reverse(scale!!, Duplicated, (my, Duplicated), (mx, Duplicated), (α, Tα); atol, rtol)
    end
    for Tα in (Const, Duplicated)
        test_forward(scale, Duplicated, (mx, Duplicated), (α, Tα); atol, rtol)
        test_forward(scale!!, Duplicated, (mx, Duplicated), (α, Tα); atol, rtol)
        test_forward(scale!!, Duplicated, (my, Duplicated), (mx, Duplicated), (α, Tα); atol, rtol)
    end
end

@testset "add ($Tx, $Ty)" for Ty in eltypes, Tx in eltypes
    n = 12
    atol = rtol = n * max(precision(Tx), precision(Ty))
    T = Base.promote_op(+, Tx, Ty)

    # Vector
    x = randn(Tx, n)
    y = randn(Ty, n)
    α = randn(T)
    β = randn(T)
    for Tα in (Const, Active), Tβ in (Const, Active)
        test_reverse(add, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα), (β, Tβ); atol, rtol)
        test_reverse(add!!, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα), (β, Tβ); atol, rtol)
    end
    for Tα in (Const, Duplicated), Tβ in (Const, Duplicated)
        test_forward(add, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα), (β, Tβ); atol, rtol)
        test_forward(add!!, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα), (β, Tβ); atol, rtol)
    end

    # MinimalMVec
    mx = MinimalMVec(x)
    my = MinimalMVec(y)
    for Tα in (Const, Active), Tβ in (Const, Active)
        test_reverse(add, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα), (β, Tβ); atol, rtol)
        test_reverse(add!!, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα), (β, Tβ); atol, rtol)
    end
    for Tα in (Const, Duplicated), Tβ in (Const, Duplicated)
        test_forward(add, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα), (β, Tβ); atol, rtol)
        test_forward(add!!, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα), (β, Tβ); atol, rtol)
    end

    # MinimalSVec
    mx = MinimalSVec(x)
    my = MinimalSVec(y)
    for Tα in (Const, Active), Tβ in (Const, Active)
        test_reverse(add, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα), (β, Tβ); atol, rtol)
        test_reverse(add!!, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα), (β, Tβ); atol, rtol)
    end
    for Tα in (Const, Duplicated), Tβ in (Const, Duplicated)
        test_forward(add, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα), (β, Tβ); atol, rtol)
        test_forward(add!!, Duplicated, (y, Duplicated), (x, Duplicated), (α, Tα), (β, Tβ); atol, rtol)
    end
end

@testset "inner ($Tx, $Ty)" for Tx in eltypes, Ty in eltypes
    n = 12
    atol = rtol = n * max(precision(Tx), precision(Ty))

    # Vector
    x = randn(Tx, n)
    y = randn(Ty, n)
    for RT in (Const, Active)
        test_reverse(inner, RT, (x, Duplicated), (y, Duplicated); atol, rtol)
    end
    for RT in (Const, Duplicated)
        test_forward(inner, RT, (x, Duplicated), (y, Duplicated); atol, rtol)
    end

    # MinimalMVec
    mx = MinimalMVec(x)
    my = MinimalMVec(y)
    for RT in (Const, Active)
        test_reverse(inner, RT, (x, Duplicated), (y, Duplicated); atol, rtol)
    end
    for RT in (Const, Duplicated)
        test_forward(inner, RT, (x, Duplicated), (y, Duplicated); atol, rtol)
    end

    # MinimalSVec
    mx = MinimalSVec(x)
    my = MinimalSVec(y)
    for RT in (Const, Active)
        test_reverse(inner, RT, (x, Duplicated), (y, Duplicated); atol, rtol)
    end
    for RT in (Const, Duplicated)
        test_forward(inner, RT, (x, Duplicated), (y, Duplicated); atol, rtol)
    end
end

@testset "inactive arguments under runtime activity ($T)" for T in eltypes
    n = 5
    xs = [randn(T, n) for _ in 1:3]
    ws = [randn(T, n) for _ in 1:3]
    ws0 = deepcopy(ws)
    mode = set_runtime_activity(Reverse)

    f(x, w) = real(sum(map(inner, w, x)))
    dxs = Enzyme.make_zero(xs)
    autodiff(mode, f, Active, Duplicated(xs, dxs), Const(ws))
    @test ws == ws0
    @test all(dxs .≈ ws0)

    g(x, w) = real(sum(map((wi, xi) -> inner(xi, add!(copy(xi), wi, 2, 1)), w, x)))
    dxs = Enzyme.make_zero(xs)
    autodiff(mode, g, Active, Duplicated(xs, dxs), Const(ws))
    @test ws == ws0
    @test all(dxs .≈ 2 .* xs .+ 2 .* ws0)
end
