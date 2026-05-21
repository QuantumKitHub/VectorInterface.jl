```@meta
CurrentModule = VectorInterface
DocTestSetup = quote
    using VectorInterface
end
```

# Interface

`VectorInterface.jl` exports a small, fixed set of methods that together constitute the vector-space interface.
This page walks through each method group, with examples.

For every mutating operation, three variants are provided:

- `f(args...)` — pure, allocating, always returns a new object.
- `f!(args...)` — strict in-place, errors if the in-place update is not possible (e.g. the target is immutable or the scalar types are incompatible).
- `f!!(args...)` — tries in-place, falls back to allocating when needed.
  This follows the convention of [`BangBang.jl`](https://github.com/JuliaFolds2/BangBang.jl) and is the right choice when writing generic algorithms that should be both efficient on mutable inputs and correct on immutable ones.

## Scalar type

```@docs; canonical=false
scalartype
```

`scalartype` returns the type of scalars over which the vector-like object behaves as a vector.
For an `AbstractArray{T <: Number}` this is `T`, and for a nested array `Vector{Vector{Float64}}` it is still `Float64` (whereas `eltype` would return `Vector{Float64}`).

```jldoctest
julia> scalartype([1.0, 2.0, 3.0])
Float64

julia> scalartype(Vector{Vector{Float64}})
Float64
```

Custom types should implement the type-domain method `scalartype(::Type{MyType})`.
The value-domain method `scalartype(x)` is defined generically as `scalartype(typeof(x))`.

## Zero vector

```@docs; canonical=false
zerovector
zerovector!
zerovector!!
```

`zerovector(v)` returns a new zero vector of the same type as `v`.
An optional second argument `S <: Number` overrides the scalar type of the result:

```jldoctest
julia> zerovector([1.0, 2.0, 3.0])
3-element Vector{Float64}:
 0.0
 0.0
 0.0

julia> zerovector([1.0, 2.0, 3.0], ComplexF64)
3-element Vector{ComplexF64}:
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
```

`zerovector!(v)` zeros `v` in place, and `zerovector!!(v)` does the same when possible and otherwise allocates.
For nested vectors, the in-place variants recurse, so `zerovector!(v)` on a `Vector{Vector{Float64}}` does not allocate any new inner arrays.

## Scaling

```@docs; canonical=false
scale
scale!
scale!!
```

`scale(v, α)` rescales the vector `v` by the scalar `α` and returns a new object.
`scale!(v, α)` and `scale!(w, v, α)` perform the rescaling in place (into `v` or into `w`, respectively), and require the scalar types to be compatible.
`scale!!` tries to do the operation in place and falls back to allocating otherwise.

```jldoctest
julia> v = [1.0, 2.0, 3.0];

julia> scale(v, 2.0)
3-element Vector{Float64}:
 2.0
 4.0
 6.0

julia> scale!(v, 0.5); v
3-element Vector{Float64}:
 0.5
 1.0
 1.5
```

A key design point: `scale!` on a **nested** vector recurses into the inner vectors and operates fully in place, in contrast to `LinearAlgebra.rmul!` which allocates new inner arrays.
For `v = [[1.0, 2.0], [3.0, 4.0]]`, `scale!(v, 0.5)` does not allocate.

## Linear combinations

```@docs; canonical=false
add
add!
add!!
```

`add(y, x, α, β)` computes `y * β + x * α` and returns a new object.
The in-place variants `add!`/`add!!` overwrite `y` with the result.
The default coefficients are [`One()`](@ref), so the short forms `add(y, x)` and `add(y, x, α)` compute `y + x` and `y * 1 + x * α` respectively.

```jldoctest
julia> add([1.0, 2.0], [10.0, 20.0])
2-element Vector{Float64}:
 11.0
 22.0

julia> add([1.0, 2.0], [10.0, 20.0], 0.1)
2-element Vector{Float64}:
 2.0
 4.0

julia> add([1.0, 2.0], [10.0, 20.0], 0.5, 2.0)
2-element Vector{Float64}:
  7.0
 14.0
```

Custom types should implement only the four-argument form.
When desired, the `One()`-coefficient case can be specialized for efficiency by dispatching on `α::One` and/or `β::One`.

## Inner product and norm

```@docs; canonical=false
inner
```

`inner(x, y)` is similar to `LinearAlgebra.dot` but is stricter about the arguments it accepts: both `x` and `y` must be elements of the same vector space, with compatible scalar types and shapes.
The norm is re-exported directly from `LinearAlgebra`:

```jldoctest
julia> inner([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
32.0

julia> norm([3.0, 4.0])
5.0
```

For complex vectors, `inner(x, y)` is conjugate-linear in its first argument, matching the mathematical convention.

## Hard-coded coefficients: `One` and `Zero`

```@docs; canonical=false
One
Zero
```

`One` and `Zero` are singleton subtypes of `Number` used to represent hard-coded constant coefficients in linear combinations.
They allow methods like [`add`](@ref) to dispatch on a unit coefficient at compile time, avoiding unnecessary multiplications.
They are the default values for the `α` and `β` coefficients in [`add`](@ref), [`add!`](@ref), and [`add!!`](@ref).

## Supported types

Out of the box, `VectorInterface.jl` provides implementations for:

- `<:Number` — scalars are also vectors over themselves.
- `<:AbstractArray` — both `AbstractArray{<:Number}` and nested arrays, with recursive in-place behaviour for the latter.
- `Tuple` and `NamedTuple` with a homogeneous element scalar type, again with arbitrary nesting.
  For example, `Vector{NTuple{3, Matrix{Float64}}}` is supported.

A general fallback exists for types that already implement `LinearAlgebra.rmul!`, `axpy!`, `axpby!`, `mul!`, and `LinearAlgebra.dot`, but this fallback emits a warning and is intended only as a transitional measure that may be removed in a future release.
New types should implement the `VectorInterface.jl` methods directly.

## Implementing the interface for a new type

A new vector-like type `T` should implement, at a minimum:

- `scalartype(::Type{T})`
- `zerovector(x::T, ::Type{S})` where `S <: Number`
- `scale(x::T, α::Number)`
- `add(y::T, x::T, α::Number, β::Number)`
- `inner(x::T, y::T)`
- `LinearAlgebra.norm(x::T)`

If the type is mutable, it should additionally implement the `!`-variants (`zerovector!`, `scale!`, `add!`).
The `!!`-variants then have generic fallbacks: they delegate to the `!`-variant when the type is mutable and to the non-mutating variant otherwise.

The `MinimalVec` type included in the package's source (`src/minimalvec.jl`) is a worked example: it wraps an `AbstractVector` and implements exactly the `VectorInterface.jl` API, deliberately excluding all other `AbstractArray` methods.
It is useful both as a reference and as a test harness for checking that an algorithm depends only on the minimal interface.
