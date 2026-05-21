module VectorInterfaceMooncakeExt

using VectorInterface
using Mooncake
using Mooncake: @is_primitive, DefaultCtx,
    NoFData, NoRData, NoTangent,
    CoDual, Dual, arrayify, primal, extract

# Projection
# ----------
"""
    project_scalar(x::Number, dx::Number)

Project a computed tangent `dx` onto the correct tangent type for `x`.
For example, we might compute a complex `dx` but only require the real part.
"""
project_scalar(x::Number, dx::Number) = oftype(x, dx)
project_scalar(x::Real, dx::Complex) = project_scalar(x, real(dx))

_needs_tangent(x) = _needs_tangent(typeof(x))
_needs_tangent(::Type{T}) where {T <: Number} =
    Mooncake.rdata_type(Mooncake.tangent_type(T)) !== NoRData

# scale
# -----
@is_primitive DefaultCtx Tuple{typeof(scale!), Any, Number}
function Mooncake.rrule!!(::CoDual{typeof(scale!)}, C_ΔC::CoDual, α_Δα::CoDual{<:Number})
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    α = primal(α_Δα)

    # primal call
    C_cache = copy(C)
    scale!(C, α)

    function scale_pullback(::NoRData)
        copy!(C, C_cache)
        Δαr = _needs_tangent(α) ? project_scalar(α, inner(C, ΔC)) : NoRData()
        scale!(ΔC, conj(α))
        return NoRData(), NoRData(), Δαr
    end

    return C_ΔC, scale_pullback
end

function Mooncake.frule!!(::Dual{typeof(scale!)}, C_ΔC::Dual, α_Δα::Dual{<:Number})
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    α, Δα = extract(α_Δα)

    if !isa(Δα, NoTangent)
        add!(ΔC, C, Δα, α)
    else
        scale!(ΔC, α)
    end
    scale!(C, α)

    return C_ΔC
end

@is_primitive DefaultCtx Tuple{typeof(scale!), Any, Any, Number}

function Mooncake.rrule!!(::CoDual{typeof(scale!)}, C_ΔC::CoDual, A_ΔA::CoDual, α_Δα::CoDual{<:Number})
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    A, ΔA = arrayify(A_ΔA)
    α = primal(α_Δα)

    # primal call
    C_cache = copy(C)
    scale!(C, A, α)

    function scale_pullback(::NoRData)
        copy!(C, C_cache)
        add!(ΔA, ΔC, conj(α))
        Δαr = _needs_tangent(α) ? project_scalar(α, inner(A, ΔC)) : NoRData()
        zerovector!(ΔC)
        return NoRData(), NoRData(), NoRData(), Δαr
    end

    return C_ΔC, scale_pullback
end

function Mooncake.frule!!(::Dual{typeof(scale!)}, C_ΔC::Dual, A_ΔA::Dual, α_Δα::Dual{<:Number})
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    A, ΔA = arrayify(A_ΔA)
    α, Δα = extract(α_Δα)

    scale!(ΔC, ΔA, α)
    !isa(Δα, NoTangent) && add!(ΔC, A, Δα, One())
    scale!(C, A, α)
    return C_ΔC
end

# add
# ---

@is_primitive DefaultCtx Tuple{typeof(add!), Any, Any, Number, Number}

function Mooncake.rrule!!(::CoDual{typeof(add!)}, C_ΔC::CoDual, A_ΔA::CoDual, α_Δα::CoDual{<:Number}, β_Δβ::CoDual{<:Number})
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    A, ΔA = arrayify(A_ΔA)
    α = primal(α_Δα)
    β = primal(β_Δβ)

    # primal call
    C_cache = copy(C)
    add!(C, A, α, β)

    function add_pullback(::NoRData)
        copy!(C, C_cache)

        Δαr = _needs_tangent(α) ? project_scalar(α, inner(A, ΔC)) : NoRData()
        Δβr = _needs_tangent(β) ? project_scalar(β, inner(C, ΔC)) : NoRData()
        add!(ΔA, ΔC, conj(α))
        scale!(ΔC, conj(β))

        return NoRData(), NoRData(), NoRData(), Δαr, Δβr
    end

    return C_ΔC, add_pullback
end

function Mooncake.frule!!(::Dual{typeof(add!)}, C_ΔC::Dual, A_ΔA::Dual, α_Δα::Dual{<:Number}, β_Δβ::Dual{<:Number})
    # prepare arguments
    C, ΔC = arrayify(C_ΔC)
    A, ΔA = arrayify(A_ΔA)
    α, Δα = extract(α_Δα)
    β, Δβ = extract(β_Δβ)
    add!(ΔC, ΔA, α, β)
    !isa(Δα, NoTangent) && add!(ΔC, A, Δα, One())
    !isa(Δβ, NoTangent) && add!(ΔC, C, Δβ, One())
    add!(C, A, α, β)
    return C_ΔC
end


# inner
# -----

@is_primitive DefaultCtx Tuple{typeof(inner), Any, Any}

function Mooncake.rrule!!(::CoDual{typeof(inner)}, A_ΔA::CoDual, B_ΔB::CoDual)
    # prepare arguments
    A, ΔA = arrayify(A_ΔA)
    B, ΔB = arrayify(B_ΔB)

    # primal call
    s = inner(A, B)

    function inner_pullback(Δs)
        add!(ΔA, B, conj(Δs))
        add!(ΔB, A, Δs)
        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(s, NoFData()), inner_pullback
end

function Mooncake.frule!!(::Dual{typeof(inner)}, A_ΔA::Dual, B_ΔB::Dual)
    # prepare arguments
    A, ΔA = arrayify(A_ΔA)
    B, ΔB = arrayify(B_ΔB)

    s = inner(A, B)
    Δs = inner(A, ΔB) + inner(ΔA, B)

    return Dual(s, Δs)
end

end
