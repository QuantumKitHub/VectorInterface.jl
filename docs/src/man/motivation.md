# Motivation

This page explains why `VectorInterface.jl` exists, what gap it fills in the Julia ecosystem, and why existing interfaces (`AbstractArray`, `LinearAlgebra`, iteration) are not adequate for writing generic algorithms that operate on vector-like objects.

## What is a vector?

Recall the basic properties of vectors in a given vector space.
A vector space is a set of objects — called vectors — with the following structure:

- Vectors can be **added**, and this addition is commutative, has a neutral element (the **zero vector**), and admits inverses.
- Vectors can be **rescaled** by a scalar coefficient, taken from an underlying scalar field.
  Scalar multiplication interacts with addition through the usual distributivity laws.

Together, these two operations give rise to the construction of **linear combinations**.
Beyond this minimum structure, many vector spaces used in practice carry additional structure:

- An **inner product**, or at least a **norm**.
- A finite **basis**, so that any vector can be written as a finite linear combination of basis vectors.

Many quantities in science and engineering — and therefore in scientific computing — behave as vectors, typically over the real or complex numbers.
Even for quantities that do not (but rather belong to a manifold), their derivatives (tangents and cotangents) do, which makes a vector interface important for automatic differentiation.

More importantly, **many algorithms can be formulated using just these basic operations**: gradient optimization, ODE integrators, Krylov methods, and so on.
A standardized vector interface enables a single implementation of such an algorithm to be applied to any vector-like type.

## Problems with the current Julia situation

The most elementary Julia type that acts as a vector is `Vector{T<:Number}`, and by extension almost any subtype of `AbstractArray{T<:Number}`.
However, many other types which are not subtypes of `AbstractArray` are also vectors conceptually — types representing functions (the `ApproxFun` ecosystem), `Tangent` types in the automatic-differentiation ecosystem, and many more.

The Julia interface to access the basic vector operations has several rough edges that make writing generic algorithms hard:

1.  **Vector addition and rescaling don't have a clean in-place interface.**
    `v + w` and `v * α` work, but for efficiency we often want in-place operations.
    For `AbstractArray`, this requires `using LinearAlgebra`, which pulls in a lot of additional functionality.
    Within `LinearAlgebra`, scaling is done via `rmul!(v, α)` / `lmul!(α, v)` / `mul!(w, v, α)` — an interface conflated with matrix multiplication, although scaling a vector and multiplying by a matrix are conceptually very different operations.
    For in-place addition, the only options are the cryptic `axpy!` / `axpby!`, whose argument order (the modified vector is last) is not Julian.

2.  **`eltype` is overloaded between vectors and iterators.**
    For an instance `v` of `AbstractArray{T<:Number}`, the scalar type is `T = eltype(v)`.
    But `eltype` is also part of the iteration interface, so types whose iteration behaviour differs from their vector-like behaviour cannot consistently overload `eltype`.
    A simple example from `Base` is a nested array: the type `Vector{Vector{T<:Number}}` still represents vectors with scalar type `T`, but `eltype` reports `Vector{T}`.

3.  **`zero` and `fill!` don't compose well.**
    `zero(v)` is fine in principle — it returns the additive neutral element — but the default implementation might fail on more complicated data structures (it used to fail nested arrays in older Julia versions).
    For an in-place zero, `fill!(v, 0)` works for `v::AbstractArray{T<:Number}` but is a very `AbstractArray`-specific interface.
    The most general workaround is to scale by zero with `rmul!`, when available.

4.  **`similar` is array-specific.**
    Constructing an equivalent vector with a modified scalar type is done via `similar(v, T′)` for `AbstractArray`s, but again this fails for nested arrays and is array-shaped in spirit.

5.  **`length` is overloaded.**
    For `v::AbstractArray{T<:Number}`, the vector-space dimension is `length(v)`, but `length` is also part of the iteration interface.
    New types may face an `eltype`-style incompatibility.
    For structured arrays, `length(v)` may not even reflect the vector-space dimension — for `UpperTriangular{T<:Number}`, the natural dimension is `n*(n+1)/2`, not `n*n`.

6.  Inner products and norms can be computed with `LinearAlgebra.dot` and `LinearAlgebra.norm`, thus requiring to load in all of `LinearAlgebra`.
    Unlike some of the previous issues, these methods natively support nested arrays, but `dot` is arguably too permissive.
    For example, `dot` happily computes an inner product between things which are probably not vectors from the same vector space, e.g. `dot((1, 2, [3, 4]), [[1], (2,), (3, 4)])`.
    It is also quite inconsistent that `dot` and `norm` accept tuple arguments, even though tuples do not behave as vectors with respect to `+`, `*`, or `zero`.

In summary: there is no formal standardized vector interface in Julia, even though one has broad applicability for writing generic algorithms.
The interfaces for **containers** (`AbstractArray`) and **iterators** are well defined, but they have become conflated with a hypothetical vector interface.

## Existing approaches in the ecosystem

Different ecosystems have responded to this gap in different ways:

- Many Krylov and optimization packages restrict their applicability to `(Abstract)Array{T<:Number}` or even simply `Vector{T<:Number}`, much like their Fortran and C analogues.
- The DifferentialEquations.jl ecosystem also restricts to `AbstractArray`, with helpers such as `RecursiveArrayTools.jl` and `ArrayInterface.jl` to accommodate more complex use cases.
- The automatic-differentiation ecosystem (Zygote.jl and ChainRules.jl) uses custom `Tangent` types for which the necessary operations are defined, relying on internal machinery to destructure custom types.

Forcing everything to be a subtype of `AbstractArray` is an unsatisfactory solution.
Some vector-like objects are not naturally represented with respect to a basis, have no notion of indexing, and may not even be finite-dimensional.
The `AbstractArray` (container) interface is — and should be — distinct from a vector-space interface.

## How `VectorInterface.jl` addresses this

`VectorInterface.jl` avoids a design centered around types, and instead provides a small set of methods that together constitute a minimal vector-space interface.
This way, any object can be made to work with this interface without conflating the notion of a container or iteration, and without the requirement for wrapper types to bypass the single supertype restriction in Julia.
The interface is designed to be standalone and offers several features out of the box, such as recursive behavior for nested `Base` containers, as well as mutating and non-mutating functionality.
Additionally, there is also a "maybe-mutating" version of each function to enable library developers to write code that is agnostic to mutability of the underlying objects.


See the [Interface](@ref) page for the full set of methods, with examples.
