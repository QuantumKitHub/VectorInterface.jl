# Vector interface implementation for subtypes of `Number`
###########################################################

# scalartype
#------------
scalartype(::Type{T}) where {T <: Number} = T

# zerovector & zerovector!!
#---------------------------
@inline zerovector(::Number, ::Type{S}) where {S <: Number} = zero(S)
@inline zerovector!!(x::Number) = zero(x)

# scale & scale!!
#-----------------
# note: the `_hasscalar` branches are resolved at compile time; for `Scalar` arguments, the
# computation is performed elementwise on the storage, in order to avoid synchronisation
# note: required to make scale(NaN, 0) = 0
@inline function scale(x::Number, α::Number)
    _hasscalar(x, α) && return _scalar_map(scale, x, α)
    return ifelse(iszero(α), zero(x), x) * α
end
@inline function scale(y::Number, x::Number, α::Number)
    _hasscalar(y, x, α) && return _scalar_map(scale, y, x, α)
    T = Base.promote_op(*, typeof(y), typeof(x), typeof(α))
    return convert(T, scale(x, α))
end
@inline scale!!(x::Number, α::Number) = scale(x, α)
@inline scale!!(y::Number, x::Number, α::Number) = scale(y, x, α)

# add & add!!
#-------------
@inline add(y::Number, x::Number, ::One, ::One) = y + x
# @inline add(y::Number, x::Number, ::One, β::Number) = add(x, y, β)
@inline function add(y::Number, x::Number, α::Number, β::One)
    _hasscalar(y, x, α) && return _scalar_map(add, y, x, α, β)
    return iszero(α) ? muladd(zero(x), α, y) : muladd(x, α, y) # required to make zero coefficients kill NaN values in x or y
end
@inline function add(y::Number, x::Number, α::Number, β::Number)
    _hasscalar(y, x, α, β) && return _scalar_map(add, y, x, α, β)
    return add(scale(y, β), x, α)
end

@inline add!!(y::Number, x::Number, α::Number, β::Number) = add(y, x, α, β)

# inner
#-------
@inline inner(x::Number, y::Number) = conj(x) * y
