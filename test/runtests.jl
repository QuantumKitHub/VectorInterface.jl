using ParallelTestRunner
using VectorInterface

testsuite = ParallelTestRunner.find_tests(@__DIR__)

# Extension tests skip Julia prereleases (extensions are unstable there)
if !isempty(VERSION.prerelease)
    filter!(!startswith("chainrules") ∘ first, testsuite)
    filter!(!startswith("mooncake") ∘ first, testsuite)
    filter!(!startswith("enzyme") ∘ first, testsuite)
    filter!(!startswith("staticsvec") ∘ first, testsuite)
end

args = parse_args(ARGS; custom = ["fast"])
fast = !isnothing(args.custom["fast"])

# On Buildkite (GPU CI runner): only run CUDA and AMDGPU tests
if get(ENV, "BUILDKITE", "false") == "true"
    f(str) = startswith(first(str), "cuarray") || startswith(first(str), "rocarray")
    filter!(f, testsuite)
end

ParallelTestRunner.runtests(VectorInterface, args; testsuite)
