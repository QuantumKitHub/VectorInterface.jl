using Test
using VectorInterface

@testset "Scalar basics" begin
    s = Scalar([2.0])
    @test s * 0.5 == Scalar{Float64, Vector{Float64}}(One())
    @test s * Scalar([0.5]) == Scalar{Float64, Vector{Float64}}(One())
    @test s * One() === One() * s === s
    @test s / One() === s
    @test One() / s == inv(s)
    @test s / 0.5 == Scalar{Float64, Vector{Float64}}([4.0])
    @test s / Scalar([0.5]) == Scalar{Float64, Vector{Float64}}([4.0])
    @test 0.5 / s == Scalar{Float64, Vector{Float64}}([0.25])
    @test s * Zero() == Zero() * s == Scalar{Float64, Vector{Float64}}(Zero())
    @test s + Zero() === Zero() + s === s
    @test s + 2s == Scalar{Float64, Vector{Float64}}(6.0)
    @test s + 4 == Scalar{Float64, Vector{Float64}}(6.0)
    @test 4 + s == Scalar{Float64, Vector{Float64}}(6.0)
    @test s - 2s == Scalar{Float64, Vector{Float64}}(-2.0)
    @test s - 4 == Scalar{Float64, Vector{Float64}}(-2.0)
    @test s - Zero() == s
    @test 4 - s == Scalar{Float64, Vector{Float64}}(2.0)
    @test -s == Scalar{Float64, Vector{Float64}}(-2.0)
    @test inv(s) == Scalar{Float64, Vector{Float64}}(0.5)
    @test iszero(s * Zero())
    @test hypot(s) == Scalar([hypot(2.0)])
    @test hypot(s, 2.0) == Scalar([hypot(2.0, 2.0)])
    @test hypot(s, 2.0, 4.0) == Scalar([hypot(2.0, 2.0, 4.0)])
    @test eps(typeof(s)) == eps(typeof(2.0))
    s2 = Scalar(fill(2.0, 1, 1))
    @test s2 isa Scalar{Float64, Vector{Float64}}
    @test s2 == s
    sc = Scalar([4.0 - 5.0 * im])
    @test real(sc) == Scalar{ComplexF64, Vector{ComplexF64}}(4.0)
    @test imag(sc) == Scalar([-5.0])
    @test conj(sc) == Scalar([4.0 + 5.0im])
    @test eps(typeof(sc)) == eps(typeof(4.0 - 5.0 * im))
end
