using Test
using VectorInterface

@testset "Scalar basics" begin
    s = Scalar([2.0])
    @test s * 0.5 == Scalar{Float64, Vector{Float64}}(One())
    @test s * One() === One() * s === s
    @test s * Zero() == Zero() * s == Scalar{Float64, Vector{Float64}}(Zero())
    @test s + Zero() === Zero() + s === s
    @test s + 2s == Scalar([6.0])

    sc = Scalar([4.0 - 5.0 * im])
    @test real(sc) == Scalar([4.0])
    @test imag(sc) == Scalar([-5.0])
    @test conj(sc) == Scalar([4.0 + 5.0im])
end
