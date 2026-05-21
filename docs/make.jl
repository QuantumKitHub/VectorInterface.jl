using Documenter
using VectorInterface

DocMeta.setdocmeta!(VectorInterface, :DocTestSetup, :(using VectorInterface); recursive = true)

makedocs(
    sitename = "VectorInterface.jl",
    modules = [VectorInterface],
    authors = "Jutho Haegeman and contributors",
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Motivation" => "man/motivation.md",
            "Interface" => "man/interface.md",
        ],
        "API reference" => "api.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://quantumkithub.github.io/VectorInterface.jl",
    ),
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/QuantumKitHub/VectorInterface.jl.git",
    devbranch = "main",
    push_preview = true,
)
