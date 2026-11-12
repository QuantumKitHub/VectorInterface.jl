# Vector interface implementation for subtypes of `AbstractArray`
##################################################################
const BLASVector{T <: BlasFloat} = Union{DenseArray{T}, StridedVector{T}}

# coefficients
#--------------
# `Scalar` coefficients are only kept for arrays of numbers whose storage is compatible with
# that of the coefficient, i.e. device arrays and scalars on the same device. For host arrays
# of numbers, the coefficient is converted into a host value (which is cheap if the scalar
# itself lives in host memory). For nested arrays, the coefficient is passed on as is to the
# elements.
_coef(x::AbstractArray, α::Number) = α
function _coef(x::AbstractArray, α::Scalar)
    eltype(x) <: Number || return α
    _ishost(typeof(x)) && return α[]
    style = Broadcast.BroadcastStyle(typeof(x))
    return _iscompatible(style, typeof(α.data)) ? α : α[]
end
# coefficient as broadcast argument: `Scalar` coefficients should broadcast as their storage
# for arrays of numbers, but should be passed on as a whole to elements of nested arrays
_bccoef(x::AbstractArray, α) = eltype(x) <: Number ? α : Ref(α)

# scalartype
#------------
scalartype(::Type{A}) where {T, A <: AbstractArray{T}} = scalartype(T) # recursive

# zerovector & zerovector!!
#---------------------------
# because broadcasting treats zero-dimensional arrays special
function zerovector(x::AbstractArray{<:Any, 0}, ::Type{S}) where {S <: Number}
    return fill(zerovector(x[], S))
end
function zerovector(x::AbstractArray, ::Type{S}) where {S <: Number}
    return zerovector.(x, S)
end

function zerovector!(x::AbstractArray{<:Number})
    return fill!(x, zero(scalartype(x)))
end
function zerovector!(x::AbstractArray)
    T = eltype(x)
    for I in eachindex(x)
        if isbitstype(T) || isassigned(x, I)
            x[I] = zerovector!!(x[I])
        else
            x[I] = zero(eltype(x))
        end
    end
    return x
end

zerovector!!(x::AbstractArray) = zerovector!(x)

# scale, scale! & scale!!
#-------------------------
# because broadcasting treats zero-dimensional arrays special, use map
function scale(x::AbstractArray, α::Number)
    α′ = _coef(x, α)
    if α′ isa Scalar && eltype(x) <: Number # device array with device scalar
        y = similar(x, promote_scale(x, α′))
        y .= scale.(x, α′)
        return y
    end
    return map(Base.Fix2(scale, α′), x)
end
function scale!(x::AbstractArray, α::Number)
    α === One() && return x
    x .= scale!!.(x, _bccoef(x, _coef(x, α)))
    return x
end
function scale!(y::AbstractArray, x::AbstractArray, α::Number)
    y .= scale!!.(y, x, _bccoef(y, _coef(y, α)))
    return y
end

function scale!!(x::AbstractArray, α::Number)
    α === One() && return x
    if promote_scale(x, α) <: scalartype(x)
        return scale!(x, α)
    else
        return scale!!.(x, _bccoef(x, _coef(x, α)))
    end
end
function scale!!(y::AbstractArray, x::AbstractArray, α::Number)
    if promote_scale(y, x, α) <: scalartype(y)
        return scale!(y, x, α)
    else
        return scale!!.(y, x, _bccoef(y, _coef(y, α)))
    end
end

# add, add! & add!!
#-------------------
# because broadcasting treats zero-dimensional arrays special, use map
function add(y::AbstractArray, x::AbstractArray, α::Number, β::Number)
    α′ = _coef(y, α)
    β′ = _coef(y, β)
    if _hasscalar(α′, β′) && eltype(y) <: Number # device array with device scalars
        ax = axes(x)
        ay = axes(y)
        ax == ay || throw(DimensionMismatch("Output axes $ay differ from input axes $ax"))
        z = similar(y, promote_add(y, x, α′, β′))
        z .= add.(y, x, α′, β′)
        return z
    end
    _add(y, x) = add(y, x, α′, β′)
    return map(_add, y, x)
end

# Special case: simple numerical arrays with BLAS-compatible floating point type
function add!(
        y::BLASVector{T}, x::BLASVector{T}, α::Number, β::Number
    ) where {T <: BlasFloat}
    α′ = _coef(y, α)
    β′ = _coef(y, β)
    # device scalars: avoid BLAS, which would require host values
    _hasscalar(α′, β′) && return _add_generic!(y, x, α′, β′)
    if β === One()
        LinearAlgebra.axpy!(convert(T, α′), x, y)
    elseif β === Zero() || β === false
        scale!(y, x, convert(T, α′))
    else
        LinearAlgebra.axpby!(convert(T, α′), x, convert(T, β′), y)
    end
    return y
end
# General case:
function add!(y::AbstractArray, x::AbstractArray, α::Number, β::Number)
    return _add_generic!(y, x, _coef(y, α), _coef(y, β))
end
function _add_generic!(y::AbstractArray, x::AbstractArray, α::Number, β::Number)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("Output axes $ay differ from input axes $ax"))
    y .= add!!.(y, x, _bccoef(y, α), _bccoef(y, β)) # might error
    return y
end

function add!!(y::AbstractArray, x::AbstractArray, α::Number, β::Number)
    if promote_add(y, x, α, β) <: scalartype(y)
        return add!(y, x, α, β)
    else
        ax = axes(x)
        ay = axes(y)
        ax == ay || throw(DimensionMismatch("Output axes $ay differ from input axes $ax"))
        return add!!.(y, x, _bccoef(y, _coef(y, α)), _bccoef(y, _coef(y, β)))
    end
end

# inner
#-------
function inner(x::BLASVector{T}, y::BLASVector{T}) where {T <: BlasFloat}
    return LinearAlgebra.dot(x, y)
end
function inner(x::AbstractArray, y::AbstractArray)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("Non-matching axes $ax and $ay"))
    T = promote_inner(x, y)
    return mapreduce(inner, +, x, y; init = zero(T))
    # this version is friendlier to GPU arrays with mixed element types
end
