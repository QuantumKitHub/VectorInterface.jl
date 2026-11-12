# Scalar: a number that lives in some (mutable, possibly device-side) storage
###############################################################################
"""
    Scalar(data::AbstractArray)
    Scalar(data::Ref)
    Scalar(x::Number)

Number type that wraps a scalar value that is stored in `data`, which is either an array
of length one, or a `Ref`. Arrays with `ndims(data) > 0` are reshaped to zero-dimensional
arrays, which share memory with the original array. `Scalar(x::Number)` stores `x` in a
`Base.RefValue`.

The main purpose of `Scalar` is to represent and manipulate scalar values that reside in
device (e.g. GPU) memory, e.g. the result of an inner product, without transferring them to
host memory. Arithmetic with `Scalar` values is implemented by broadcasting over their
storage, and thus never requires such a transfer. In contrast, operations that need to
return a host value, such as comparisons (`==`, `<`, `isless`, `iszero`, `isfinite`, ...),
`hash`, `convert(T, s)` for a non-`Scalar` type `T`, or explicitly accessing the value
using `s[]`, do require a transfer and thus a synchronisation point.

`Scalar` takes precedence over other `Number` types in promotion, as converting a host
number into a device scalar is cheap, whereas the reverse requires synchronisation.

`Scalar` objects are immutable, but their storage is mutable. In-place updates of `s` are
possible using `s[] = x` or using the in-place methods of the VectorInterface, such as
`scale!(s, α)`, `add!(s, x, α, β)` and `inner!(s, x, y)`. The recommended pattern for
generic code is however to use the `!!` methods, such as `s = inner!!(s, x, y)`, which work
both for `Scalar` and for plain `Number` instances.

When interpreted as a vector in the sense of VectorInterface.jl, `scalartype(s)` returns
the type of the wrapped value, not `typeof(s)`.

See also: [`storagetype`](@ref VectorInterface.storagetype)
"""
struct Scalar{T <: Number, A} <: Number
    data::A
    global _unsafe_scalar(data::A) where {A} = new{eltype(A), A}(data)
end

# Constructors
#--------------
function Scalar(data::AbstractArray{T}) where {T <: Number}
    length(data) == 1 ||
        throw(DimensionMismatch("`Scalar` requires storage of length one, got size $(size(data))"))
    return _unsafe_scalar(ndims(data) == 0 ? data : reshape(data, ()))
end
Scalar(data::Ref{T}) where {T <: Number} = _unsafe_scalar(data)
Scalar(s::Scalar) = s
Scalar(x::Number) = Scalar(Ref(x))
Scalar{T}(x::Number) where {T <: Number} = Scalar(Ref{T}(convert(T, x)))
Scalar{T}(s::Scalar) where {T <: Number} = _scalar_map(Base.Fix1(convert, T), s)
function Scalar{T, A}(x::Number) where {T <: Number, A}
    eltype(A) === T ||
        throw(ArgumentError("storage type $A does not have element type $T"))
    return _unsafe_scalar(_assign!(_allocate(A), identity, x))
end

# disambiguate with `(::Type{<:Number})(::One)` and `(::Type{<:Number})(::Zero)`
for Z in (:One, :Zero)
    @eval begin
        Scalar(x::$Z) = Scalar(Int(x))
        Scalar{T}(x::$Z) where {T <: Number} = Scalar{T}(T(x))
        Scalar{T, A}(x::$Z) where {T <: Number, A} = Scalar{T, A}(T(x))
    end
end

# Storage utilities
#-------------------
"""
    storagetype(A::Type, T::Type{<:Number})

Return the type of the storage that is obtained by changing the element type of storage
type `A` (of a [`Scalar`](@ref)) to `T`. This is used in the promotion rules of `Scalar`.
The default implementation for `AbstractArray` types relies on inferring the return type
of `similar`; new array types can overload this method if that does not produce a concrete
type.
"""
storagetype(::Type{<:Ref}, ::Type{T}) where {T} = Base.RefValue{T}
storagetype(::Type{<:Array}, ::Type{T}) where {T} = Array{T, 0}
function storagetype(::Type{A}, ::Type{T}) where {A <: AbstractArray, T}
    return Base.promote_op(similar, A, Type{T})
end

_allocate(::Type{A}) where {A <: Ref} = A()
_allocate(::Type{A}) where {A <: AbstractArray} = A(undef, ())

_similar(data::Ref, ::Type{T}) where {T} = Base.RefValue{T}()
_similar(data::AbstractArray, ::Type{T}) where {T} = similar(data, T, ())

# whether the storage lives in host memory
_ishost(::Type{<:Ref}) = true
_ishost(::Type{A}) where {A <: AbstractArray} = Broadcast.BroadcastStyle(A) isa Broadcast.DefaultArrayStyle

# priority of storage types in promotion: device arrays > host arrays > Refs
_rank(::Type{<:Ref}) = 0
_rank(::Type{A}) where {A <: AbstractArray} = _ishost(A) ? 1 : 2

# whether storage of type `B` can participate in a broadcast with style `style`
_iscompatible(style, ::Type{<:Ref}) = style isa Broadcast.DefaultArrayStyle
function _iscompatible(style, ::Type{B}) where {B <: AbstractArray}
    stB = Broadcast.BroadcastStyle(B)
    style isa Broadcast.DefaultArrayStyle && return stB isa Broadcast.DefaultArrayStyle
    return Broadcast.result_style(style, stB) === style
end

# read host value: this is the synchronisation point
_hostvalue(data::Ref) = data[]
_hostvalue(data::AbstractArray) = _ishost(typeof(data)) ? data[] : only(Array(data))

_hostarg(s::Scalar) = s[]
_hostarg(x) = x

# argument for a broadcast into a destination with broadcast style `style`
_bcarg(style, s::Scalar) = _iscompatible(style, typeof(s.data)) ? s.data : s[]
_bcarg(style, x) = x

# assign `f(args...)` to `dst`, where `args` can contain `Scalar` instances
function _assign!(dst::AbstractArray, f::F, args::Vararg{Any, N}) where {F, N}
    style = Broadcast.BroadcastStyle(typeof(dst))
    bcargs = map(a -> _bcarg(style, a), args)
    Broadcast.materialize!(dst, Broadcast.broadcasted(f, bcargs...))
    return dst
end
function _assign!(dst::Ref, f::F, args::Vararg{Any, N}) where {F, N}
    dst[] = f(map(_hostarg, args)...)
    return dst
end

_fill!(dst::AbstractArray, x) = fill!(dst, x)
_fill!(dst::Ref, x) = (dst[] = x; dst)

# select the storage of the dominant `Scalar` among `args`, used as prototype for the result
_prototype(x::Scalar, rest...) = _pick(x.data, _prototype(rest...))
_prototype(x, rest...) = _prototype(rest...)
_prototype() = nothing
_pick(a, ::Nothing) = a
_pick(a, b) = _rank(typeof(b)) > _rank(typeof(a)) ? b : a

_valtype(::Type{S}) where {S <: Scalar} = scalartype(S)
_valtype(::Type{T}) where {T} = T

_hasscalar(args...) = _hasscalar_type(map(typeof, args)...)
_hasscalar_type(::Type{<:Scalar}, rest...) = true
_hasscalar_type(::Type, rest...) = _hasscalar_type(rest...)
_hasscalar_type() = false

# compute `f(args...)` and store the result in a new `Scalar`, with storage similar to the
# dominant `Scalar` argument
function _scalar_map(f::F, args::Vararg{Any, N}) where {F, N}
    proto = _prototype(args...)
    T = Base.promote_op(f, map(a -> _valtype(typeof(a)), args)...)
    return _unsafe_scalar(_assign!(_similar(proto, T), f, args...))
end

# Accessing the value
#---------------------
"""
    s[]

Return the value of `s::Scalar` as a host value. This requires a synchronisation point
when the storage of `s` lives in device memory.
"""
Base.getindex(s::Scalar) = _hostvalue(s.data)

"""
    s[] = x

Overwrite the value of `s::Scalar` with `x`. This does not require a synchronisation
point.
"""
function Base.setindex!(s::Scalar, x::Number)
    _assign!(s.data, identity, x)
    return s
end

Base.copy(s::Scalar) = _scalar_map(identity, s)
Base.Broadcast.broadcastable(s::Scalar) = s.data

function Base.show(io::IO, s::Scalar)
    print(io, "Scalar(")
    show(io, s.data)
    return print(io, ")")
end

# Conversion and promotion
#--------------------------
Base.convert(::Type{S}, s::S) where {S <: Scalar} = s
Base.convert(::Type{S}, s::Scalar) where {S <: Scalar} = S(s)
Base.convert(::Type{S}, x::Number) where {S <: Scalar} = S(x)
Base.convert(::Type{T}, s::Scalar) where {T <: Number} = convert(T, s[])

function Base.promote_rule(::Type{Scalar{T, A}}, ::Type{S}) where {T, A, S <: Number}
    T′ = promote_type(T, S)
    return Scalar{T′, storagetype(A, T′)}
end
function Base.promote_rule(::Type{Scalar{T₁, A₁}}, ::Type{Scalar{T₂, A₂}}) where {T₁, A₁, T₂, A₂}
    T = promote_type(T₁, T₂)
    S₁ = storagetype(A₁, T)
    S₂ = storagetype(A₂, T)
    S₁ === S₂ && return Scalar{T, S₁}
    r₁ = _rank(A₁)
    r₂ = _rank(A₂)
    r₁ > r₂ && return Scalar{T, S₁}
    r₂ > r₁ && return Scalar{T, S₂}
    return Scalar{T} # incompatible storage types: no common concrete type
end

for f in (:real, :float, :complex)
    @eval function Base.$f(::Type{Scalar{T, A}}) where {T, A}
        T′ = $f(T)
        return Scalar{T′, storagetype(A, T′)}
    end
end
Base.rtoldefault(::Type{<:Scalar{T}}) where {T} = Base.rtoldefault(T)

Base.zero(s::Scalar) = _scalar_map(zero, s)
Base.one(s::Scalar) = _scalar_map(one, s)
Base.zero(::Type{Scalar{T, A}}) where {T, A} = Scalar{T, A}(zero(T))
Base.one(::Type{Scalar{T, A}}) where {T, A} = Scalar{T, A}(one(T))
Base.oneunit(s::Scalar) = _scalar_map(oneunit, s)

# Arithmetic: implemented via broadcasting over the storage, no synchronisation
#--------------------------------------------------------------------------------
Base.:+(s::Scalar) = s
for f in (
        :-, :conj, :real, :imag, :abs, :abs2, :inv, :sqrt, :cbrt, :exp, :exp2, :exp10, :expm1,
        :log, :log2, :log10, :log1p, :sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh,
        :tanh, :asinh, :acosh, :atanh, :sign, :angle, :cis, :float, :complex,
    )
    @eval Base.$f(s::Scalar) = _scalar_map($f, s)
end

for f in (:+, :-, :*, :/, :\, :^, :min, :max, :hypot)
    @eval begin
        Base.$f(a::Scalar, b::Scalar) = _scalar_map($f, a, b)
        Base.$f(a::Scalar, b::Number) = _scalar_map($f, a, b)
        Base.$f(a::Number, b::Scalar) = _scalar_map($f, a, b)
    end
end
Base.:^(a::Scalar, b::Integer) = _scalar_map(^, a, b)
Base.:^(a::Scalar, b::Rational) = _scalar_map(^, a, b)
Base.:^(::Irrational{:ℯ}, s::Scalar) = exp(s)

# disambiguate with `Zero` and `One` methods
Base.:+(::Zero, s::Scalar) = s
Base.:+(s::Scalar, ::Zero) = s
Base.:-(::Zero, s::Scalar) = -s
Base.:-(s::Scalar, ::Zero) = s
Base.:*(::One, s::Scalar) = s
Base.:*(s::Scalar, ::One) = s
Base.:*(::Zero, s::Scalar) = zero(s)
Base.:*(s::Scalar, ::Zero) = zero(s)
Base.:/(s::Scalar, ::One) = s
Base.:/(::One, s::Scalar) = inv(s)
Base.:/(::Scalar, ::Zero) = throw(DivideError())
# no check for `iszero(s)`, as that would require synchronisation
Base.:/(::Zero, s::Scalar) = zero(s)

for f in (:muladd, :fma)
    @eval begin
        Base.$f(a::Scalar, b::Scalar, c::Scalar) = _scalar_map($f, a, b, c)
        Base.$f(a::Scalar, b::Number, c::Number) = _scalar_map($f, a, b, c)
        Base.$f(a::Number, b::Scalar, c::Number) = _scalar_map($f, a, b, c)
        Base.$f(a::Number, b::Number, c::Scalar) = _scalar_map($f, a, b, c)
        Base.$f(a::Scalar, b::Scalar, c::Number) = _scalar_map($f, a, b, c)
        Base.$f(a::Scalar, b::Number, c::Scalar) = _scalar_map($f, a, b, c)
        Base.$f(a::Number, b::Scalar, c::Scalar) = _scalar_map($f, a, b, c)
    end
end

# Predicates and comparisons: these require synchronisation
#------------------------------------------------------------
for f in (:iszero, :isone, :isfinite, :isinf, :isnan, :isreal, :isinteger, :signbit)
    @eval Base.$f(s::Scalar) = $f(s[])
end
for f in (:(==), :isequal, :<, :<=, :isless)
    @eval begin
        Base.$f(a::Scalar, b::Scalar) = $f(a[], b[])
        Base.$f(a::Scalar, b::Number) = $f(a[], b)
        Base.$f(a::Number, b::Scalar) = $f(a, b[])
    end
end
Base.hash(s::Scalar, h::UInt) = hash(s[], h)

# VectorInterface
#-----------------
scalartype(::Type{<:Scalar{T}}) where {T} = T

function zerovector(s::Scalar, ::Type{S}) where {S <: Number}
    T = scalartype(S)
    return _unsafe_scalar(_fill!(_similar(s.data, T), zero(T)))
end
function zerovector!(s::Scalar)
    _fill!(s.data, zero(scalartype(s)))
    return s
end
zerovector!!(s::Scalar) = zerovector!(s)

function _check_inplace(s::Scalar, ::Type{T}) where {T}
    return T <: scalartype(s) || throw(
        InexactError(:convert, scalartype(s), T)
    )
end

function scale!(s::Scalar, α::Number)
    α === One() && return s
    return scale!(s, s, α)
end
function scale!(s::Scalar, x::Number, α::Number)
    _check_inplace(s, promote_scale(s, x, α))
    _assign!(s.data, scale, x, α)
    return s
end
function scale!!(s::Scalar, α::Number)
    α === One() && return s
    return promote_scale(s, α) <: scalartype(s) ? scale!(s, α) : scale(s, α)
end
function scale!!(s::Scalar, x::Number, α::Number)
    return promote_scale(s, x, α) <: scalartype(s) ? scale!(s, x, α) : scale(s, x, α)
end

function add!(s::Scalar, x::Number, α::Number, β::Number)
    _check_inplace(s, promote_add(s, x, α, β))
    _assign!(s.data, add, s, x, α, β)
    return s
end
function add!!(s::Scalar, x::Number, α::Number, β::Number)
    return promote_add(s, x, α, β) <: scalartype(s) ? add!(s, x, α, β) : add(s, x, α, β)
end

function inner!(s::Scalar, x, y)
    _check_inplace(s, promote_inner(x, y))
    _assign!(s.data, identity, inner(x, y))
    return s
end
# reduction for arrays of numbers that keeps the result in device memory
function inner!(s::Scalar{<:Number, <:AbstractArray}, x::AbstractArray{<:Number}, y::AbstractArray{<:Number})
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("Non-matching axes $ax and $ay"))
    T = promote_inner(x, y)
    _check_inplace(s, T)
    if ndims(x) == 0
        _assign!(s.data, inner, Scalar(x), Scalar(y))
    else
        dims = ntuple(identity, Val(ndims(x)))
        r = mapreduce(inner, +, x, y; dims, init = zero(T))
        _assign!(s.data, identity, Scalar(r))
    end
    return s
end
function inner!!(s::Scalar, x, y)
    T = promote_inner(x, y)
    T <: scalartype(s) && return inner!(s, x, y)
    return inner!(_unsafe_scalar(_similar(s.data, T)), x, y)
end
