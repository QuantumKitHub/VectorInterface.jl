```@meta
CurrentModule = VectorInterface
```

# VectorInterface.jl

A small Julia interface for vector-like objects.

`VectorInterface.jl` proposes a minimal set of methods that any type representing an element of a vector space should support.
Code written against this interface can then operate uniformly on `Number`s, `AbstractArray`s, nested arrays, tuples, named tuples, and arbitrary user-defined types — without conflating vector-space operations with the iteration, container, or `LinearAlgebra` interfaces.

## Installation

```julia
julia> using Pkg; Pkg.add("VectorInterface")
```

## What's provided

- A scalar-type query: [`scalartype`](@ref).
- Construction of zero vectors: [`zerovector`](@ref), [`zerovector!`](@ref), [`zerovector!!`](@ref).
- Scalar multiplication: [`scale`](@ref), [`scale!`](@ref), [`scale!!`](@ref).
- Linear combinations: [`add`](@ref), [`add!`](@ref), [`add!!`](@ref).
- Inner product and norm: [`inner`](@ref), and `norm` is re-exported from `LinearAlgebra`.
- Singleton helpers [`One`](@ref) and [`Zero`](@ref) for hard-coded coefficients in linear combinations.

Each `!`-suffixed method modifies its first argument in place.
Each `!!`-suffixed method tries to do so but falls back to allocating when in-place updates are not possible (e.g. for immutable types or incompatible scalar types).
This convention follows [`BangBang.jl`](https://github.com/JuliaFolds2/BangBang.jl).

## Contents

```@contents
Pages = ["man/motivation.md", "man/interface.md", "api.md"]
Depth = 2
```
