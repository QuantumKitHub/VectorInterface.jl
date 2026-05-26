using ParallelTestRunner
using VectorInterface

testsuite = ParallelTestRunner.find_tests(@__DIR__)

# Skip setup code
filter!(!startswith("simplevec") ∘ first, testsuite)

# Extension tests skip Julia prereleases (extensions are unstable there)
if !isempty(VERSION.prerelease)
    filter!(!startswith("chainrules") ∘ first, testsuite)
    filter!(!startswith("mooncake") ∘ first, testsuite)
    filter!(!startswith("enzyme") ∘ first, testsuite)
    filter!(!startswith("staticsvec") ∘ first, testsuite)
end

args = parse_args(ARGS; custom = ["fast"])
fast = !isnothing(args.custom["fast"])

ParallelTestRunner.runtests(VectorInterface, args; testsuite)
