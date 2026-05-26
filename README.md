# VectorInterface


| **Documentation** | **Build Status** | **License** |
|:-----------------:|:----------------:|:-----------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![][aqua-img]][aqua-url] [![CI][github-img]][github-url] [![][codecov-img]][codecov-url] | [![license][license-img]][license-url] |

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://quantumkithub.github.io/VectorInterface.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://quantumkithub.github.io/VectorInterface.jl/stable

[github-img]: https://github.com/QuantumKitHub/VectorInterface.jl/workflows/CI/badge.svg
[github-url]: https://github.com/QuantumKitHub/VectorInterface.jl/actions?query=workflow%3ACI

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[codecov-img]: https://codecov.io/gh/QuantumKitHub/VectorInterface.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/QuantumKitHub/VectorInterface.jl

[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat
[license-url]: LICENSE.md

---

A small Julia interface for vector-like objects.
`VectorInterface.jl` proposes a minimal, unified set of methods that any type representing an element of a vector space should support, so that generic algorithms (Krylov methods, ODE integrators, gradient optimizers, ...) can be written once and reused across `Number`s, `AbstractArray`s, nested arrays, tuples, named tuples, and arbitrary user-defined types — without conflating vector-space operations with the iteration, container, or `LinearAlgebra` interfaces.

## Installation

```julia
julia> using Pkg; Pkg.add("VectorInterface")
```

## Quick example

```julia
using VectorInterface

v = [1.0, 2.0, 3.0]
w = [4.0, 5.0, 6.0]

scalartype(v)            # Float64
zerovector(v)            # [0.0, 0.0, 0.0]
scale!!(v, 2.0)          # tries in place: [2.0, 4.0, 6.0]
add!!(v, w, 0.5)         # v <- v + 0.5 * w
inner(v, w)              # conjugate-linear inner product
norm(v)                  # re-exported from LinearAlgebra
```

Every mutating operation comes in three variants: `f` (allocating), `f!` (strict in-place), and `f!!` (try in-place, fall back to allocating), following the convention from [`BangBang.jl`](https://github.com/JuliaFolds2/BangBang.jl).

