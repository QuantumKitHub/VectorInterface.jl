using Test
using VectorInterface

@testset "Scalar basics" begin
    s = Scalar([2.0])
    @test s * 0.5 == Scalar{Float64, Vector{Float64}}(One())
    @test s * Scalar([0.5]) == Scalar{Float64, Vector{Float64}}(One())
    @test s * One() === One() * s === s
    @test s * Zero() == Zero() * s == Scalar{Float64, Vector{Float64}}(Zero())
    @test s + Zero() === Zero() + s === s
    @test s + 2s == Scalar{Float64, Vector{Float64}}(6.0)
    @test s + 4 == Scalar{Float64, Vector{Float64}}(6.0)
    @test 4 + s == Scalar{Float64, Vector{Float64}}(6.0)
    @test iszero(s * Zero())
    s2 = Scalar(fill(2.0, 1, 1))
    @test s2 isa Scalar{Float64, Vector{Float64}}
    @test s2 == s

    sc = Scalar([4.0 - 5.0 * im])
    @test real(sc) == Scalar{ComplexF64, Vector{ComplexF64}}(4.0)
    @test imag(sc) == Scalar([-5.0])
    @test conj(sc) == Scalar([4.0 + 5.0im])
end
