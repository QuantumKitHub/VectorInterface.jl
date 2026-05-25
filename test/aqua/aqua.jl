module AquaVectorInterface

using Test
using VectorInterface
using Aqua

@testset "Aqua" begin
    Aqua.test_all(VectorInterface)
end

end
