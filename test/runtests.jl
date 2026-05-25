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

const init_code = quote
    const fast_tests = $fast
end

ParallelTestRunner.runtests(VectorInterface, args; testsuite, init_code)
