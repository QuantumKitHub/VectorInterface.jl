using VectorInterface
using Test
println("Testing One and Zero")
println("====================")
include("onezero.jl")

println("Testing with simple numerical array")
println("===================================")
include("simple.jl")

println("Testing with MinimalMVec")
println("=======================")
include("minimalmvec.jl")
println("Testing with MinimalSVec")
println("=======================")
include("minimalsvec.jl")

println("Testing with complicated composite object")
println("=========================================")
include("complicated.jl")

println("Testing fallbacks with unsupported object")
println("=========================================")
include("unsupported.jl")

println("Testing previously reported issues")
println("==================================")
include("issues.jl")

println("Quality control test with Aqua.jl")
println("=================================")
module AquaVectorInterface
    using VectorInterface
    using Aqua
    Aqua.test_all(VectorInterface)
end

@static if isdefined(Base, :get_extension) && isempty(VERSION.prerelease)
    println("Testing AD rules")
    println("================")
    println("Testing ChainRules")
    println("==================")
    include("chainrules.jl")
    println("Testing Mooncake")
    println("==================")
    include("mooncake.jl")
end
