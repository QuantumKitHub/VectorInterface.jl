"""
    struct One end

Singleton type for representing a hard-coded constant 1 in vector addition / linear
combinations.
"""
struct One <: Real end

"""
    struct Zero end
    
Singleton type for representing a hard-coded constant 0 in vector addition / linear
combinations.

```quote
    Now I am become Zero, the destroyer of NaN. - Vishnu
```
"""
struct Zero <: Real end

const ZeroOne = Union{Zero, One}

# Base Arithmetic
# ---------------

Base.:(-)(::One) = -1
Base.:(-)(::Zero) = Zero()

Base.:(+)(::Zero, x::Number) = x
Base.:(+)(x::Number, ::Zero) = x
Base.:(+)(::Zero, ::Zero) = Zero()
Base.:(+)(::One, ::One) = 2

Base.:(-)(x::Number, ::Zero) = x
Base.:(-)(::Zero, x::Number) = -x
Base.:(-)(::Zero, ::Zero) = Zero()
Base.:(-)(::One, ::One) = Zero()

Base.:(*)(::One, x::Number) = x
Base.:(*)(::Zero, x::Number) = zero(x)
Base.:(*)(x::Number, ::One) = x
Base.:(*)(x::Number, ::Zero) = zero(x)
Base.:(*)(::Zero, ::Zero) = Zero()
Base.:(*)(::One, ::One) = One()
Base.:(*)(::Zero, ::One) = Zero()
Base.:(*)(::One, ::Zero) = Zero()

Base.:(/)(::Zero, ::Zero) = throw(DivideError())
Base.:(/)(::One, ::One) = One()
Base.:(/)(::Zero, ::One) = Zero()
Base.:(/)(::One, ::Zero) = throw(DivideError())
Base.:(/)(::Number, ::Zero) = throw(DivideError())
Base.:(/)(::Zero, x::Number) = iszero(x) ? throw(DivideError()) : zero(x)
Base.:(/)(x::Number, ::One) = x
Base.:(/)(::One, x::Number) = inv(x)

Base.inv(::One) = One()

Base.conj(::One) = One()
Base.conj(::Zero) = Zero()

Base.one(::Type{One}) = One()
Base.one(::Type{Zero}) = One()
Base.one(::ZeroOne) = One()
Base.zero(::Type{One}) = Zero()
Base.zero(::Type{Zero}) = Zero()
Base.zero(::ZeroOne) = Zero()

Base.:(==)(::One, ::One) = true
Base.:(==)(::Zero, ::Zero) = true
Base.:(==)(::One, ::Zero) = false
Base.:(==)(::Zero, ::One) = false

# Promotion
# ---------
Base.promote_rule(::Type{Zero}, ::Type{One}) = Bool
Base.promote_rule(::Type{One}, ::Type{Zero}) = Bool
Base.promote_rule(::Type{One}, ::Type{T}) where {T <: Number} = T
Base.promote_rule(::Type{Zero}, ::Type{T}) where {T <: Number} = T
# disambiguate:
Base.promote_rule(::Type{Bool}, ::Type{One}) = Bool
Base.promote_rule(::Type{Bool}, ::Type{Zero}) = Bool

# NOTE: deliberately no `convert(::Type{T}, ::Zero) where {T <: Number}` methods here.
# Since `Zero`/`One` are `<: Real`, such methods supersede `convert(::Type{T}, x::Number)`
# for real argument types and invalidate a large amount of precompiled Base code (~200
# invalidations, mostly under `convert(::Type{Int64}, ::Real)`). Base's own
# `convert(::Type{T}, x::Number) = T(x)::T` routes through the constructors below instead,
# which is equivalent and invalidation-free. Do not reintroduce them.
(T::Type{<:Number})(::One) = one(T)
(T::Type{<:Number})(::Zero) = zero(T)

# Disambiguation
# --------------
# Being `<: Real` makes Base's Real-vs-Complex methods applicable, which collide with the
# broad `::Number` methods above. The bodies match those methods; these only fix dispatch.
for C in (:Complex, :(Complex{Bool}))
    @eval begin
        Base.:(+)(::Zero, x::$C) = x
        Base.:(+)(x::$C, ::Zero) = x
        Base.:(-)(::Zero, x::$C) = -x
        Base.:(-)(x::$C, ::Zero) = x
        Base.:(*)(::Zero, x::$C) = zero(x)
        Base.:(*)(x::$C, ::Zero) = zero(x)
        Base.:(*)(::One, x::$C) = x
        Base.:(*)(x::$C, ::One) = x
        Base.:(/)(::$C, ::Zero) = throw(DivideError())
        Base.:(/)(x::$C, ::One) = x
    end
end
Base.:(/)(::Zero, x::S) where {S <: Complex} = iszero(x) ? throw(DivideError()) : zero(x)
Base.:(/)(::One, x::S) where {S <: Complex} = inv(x)

# against `Bool(::Real)`, `Complex(::Real)` and `Complex{T}(::Real)`
Base.Bool(::One) = true
Base.Bool(::Zero) = false
Base.Complex(::One) = Complex(true)
Base.Complex(::Zero) = Complex(false)
Base.Complex{T}(::One) where {T <: Real} = Complex{T}(one(T))
Base.Complex{T}(::Zero) where {T <: Real} = Complex{T}(zero(T))

# Utility
# -------
# `real`, `imag`, `abs`, `abs2`, `signbit`, `isreal`, `isnan` and mixed-type comparisons are
# inherited from the `Real` fallbacks. What follows is what `Real` does not provide.

Base.isinteger(::ZeroOne) = true
Base.isfinite(::ZeroOne) = true
Base.isinf(::ZeroOne) = false

Base.iszero(::Zero) = true
Base.iszero(::One) = false
Base.isone(::Zero) = false
Base.isone(::One) = true

# `isequal(Zero(), 0)` holds through promotion, so the hashes have to agree as well
Base.hash(::Zero, h::UInt) = hash(0, h)
Base.hash(::One, h::UInt) = hash(1, h)

# same-type comparisons: `<(x::T, y::T) where {T <: Real}` is a Base no-op error, and
# `promote(Zero(), Zero())` is a fixed point, so these cannot be inherited
Base.isless(::Zero, ::Zero) = false
Base.isless(::One, ::One) = false
Base.isless(::Zero, ::One) = true
Base.isless(::One, ::Zero) = false
Base.:(<)(::Zero, ::Zero) = false
Base.:(<)(::One, ::One) = false
Base.:(<)(::Zero, ::One) = true
Base.:(<)(::One, ::Zero) = false
Base.:(<=)(::Zero, ::Zero) = true
Base.:(<=)(::One, ::One) = true
Base.:(<=)(::Zero, ::One) = true
Base.:(<=)(::One, ::Zero) = false

# `sign(x::Real)` compares `x` against itself and hits the same no-op error
Base.sign(::Zero) = Zero()
Base.sign(::One) = One()

# identity-preserving: the inherited versions promote through `Bool`
Base.min(::Zero, ::One) = Zero()
Base.min(::One, ::Zero) = Zero()
Base.max(::Zero, ::One) = One()
Base.max(::One, ::Zero) = One()
