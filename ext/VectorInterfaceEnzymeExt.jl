module VectorInterfaceEnzymeExt

using VectorInterface
using Enzyme
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore: EnzymeRules

"""
    project_scalar(x::Number, dx::Number)

Project a computed tangent `dx` onto the correct tangent type for `x`.
For example, we might compute a complex `dx` but only require the real part.
"""
project_scalar(x::Number, dx::Number) = oftype(x, dx)
project_scalar(x::Real, dx::Complex) = project_scalar(x, real(dx))

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(scale!)},
        ::Type{RT},
        C::Annotation,
        α::Annotation{<:Number},
    ) where {RT}
    dret = !isa(C, Const) ? C.dval : nothing
    cacheα = EnzymeRules.overwritten(config)[3] ? copy(α.val) : α.val
    cache = (cacheα, copy(C.val)) # is this better than just unscaling?
    ret = scale!(C.val, α.val)
    shadow = EnzymeRules.needs_shadow(config) ? dret : nothing
    primal = EnzymeRules.needs_primal(config) ? ret : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(scale!)},
        ::Type{RT},
        cache,
        C::Annotation,
        α::Annotation{<:Number},
    ) where {RT}
    αval, Cval = cache
    Δα = if !isa(α, Const) && !isa(C, Const)
        project_scalar(α.val, inner(Cval, C.dval))
    elseif !isa(α, Const)
        zero(α.val)
    else
        nothing
    end
    scale!(C.dval, conj(αval))
    return (nothing, Δα)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        func::Const{typeof(scale!)},
        ::Type{RT},
        C::Annotation,
        α::Annotation{<:Number},
    ) where {RT}
    if !isa(α, Const) && !isa(C, Const)
        add!(C.dval, C.val, α.dval, α.val)
    elseif !isa(C, Const)
        scale!(C.dval, α.val)
    end
    scale!(C.val, α.val)
    if EnzymeRules.needs_primal(config) &&  EnzymeRules.needs_shadow(config)
        return C
    elseif EnzymeRules.needs_primal(config)
        return C.val
    elseif EnzymeRules.needs_shadow(config)
        return C.dval
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(scale!)},
        ::Type{RT},
        C::Annotation,
        A::Annotation,
        α::Annotation{<:Number},
    ) where {RT}
    cacheA = EnzymeRules.overwritten(config)[3] ? copy(A.val) : A.val
    cacheα = EnzymeRules.overwritten(config)[4] ? copy(α.val) : α.val
    cache = (cacheA, cacheα)
    ret = scale!(C.val, A.val, α.val)
    dret = !isa(C, Const) ? C.dval : nothing
    shadow = EnzymeRules.needs_shadow(config) ? dret : nothing
    primal = EnzymeRules.needs_primal(config) ? ret : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(scale!)},
        ::Type{RT},
        cache,
        C::Annotation,
        A::Annotation,
        α::Annotation{<:Number},
    ) where {RT}
    Aval, αval = cache
    !isa(A, Const) && !isa(C, Const) && add!(A.dval, C.dval, conj(αval))
    Δα = if !isa(α, Const) && !isa(C, Const)
        project_scalar(α.val, inner(Aval, C.dval))
    elseif !isa(α, Const)
        zero(α.val)
    else
        nothing
    end
    zerovector!(C.dval)
    return (nothing, nothing, Δα)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        func::Const{typeof(scale!)},
        ::Type{RT},
        C::Annotation,
        A::Annotation,
        α::Annotation{<:Number},
    ) where {RT}
    scale!(C.val, A.val, α.val)
    !isa(C, Const) && !isa(A, Const) && scale!(C.dval, A.dval, α.val)
    !isa(α, Const) && !isa(C, Const) && add!(C.dval, A.val, α.dval, One())
    if EnzymeRules.needs_primal(config) &&  EnzymeRules.needs_shadow(config)
        return C
    elseif EnzymeRules.needs_primal(config)
        return C.val
    elseif EnzymeRules.needs_shadow(config)
        return C.dval
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(add!)},
        ::Type{RT},
        C::Annotation,
        A::Annotation,
        α::Annotation{<:Number},
        β::Annotation{<:Number},
    ) where {RT}
    dret = !isa(C, Const) ? C.dval : nothing
    # only need copy of A if α is not constant
    cacheA = !isa(α, Const) && EnzymeRules.overwritten(config)[3] ? copy(A.val) : A.val
    cacheα = EnzymeRules.overwritten(config)[4] ? copy(α.val) : α.val
    cacheβ = EnzymeRules.overwritten(config)[5] ? copy(β.val) : β.val
    # only need copy of C if β is not constant
    cacheC = !isa(β, Const) ? copy(C.val) : C.val
    cache = (cacheA, cacheα, cacheβ, cacheC)
    ret = add!(C.val, A.val, α.val, β.val)
    shadow = EnzymeRules.needs_shadow(config) ? dret : nothing
    primal = EnzymeRules.needs_primal(config) ? ret : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(add!)},
        ::Type{RT},
        cache,
        C::Annotation,
        A::Annotation,
        α::Annotation{<:Number},
        β::Annotation{<:Number},
    ) where {RT}
    Aval, αval, βval, Cval = cache
    Δα = if !isa(α, Const) && !isa(C, Const)
        project_scalar(α.val, inner(Aval, C.dval))
    elseif !isa(α, Const)
        zero(α.val)
    else
        nothing
    end
    Δβ = if !isa(β, Const) && !isa(C, Const)
        project_scalar(β.val, inner(Cval, C.dval))
    elseif !isa(β, Const)
        zero(β.val)
    else
        nothing
    end
    !isa(A, Const) && !isa(C, Const) && add!(A.dval, C.dval, conj(αval))
    !isa(C, Const) && scale!(C.dval, conj(βval))
    return (nothing, nothing, Δα, Δβ)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        func::Const{typeof(add!)},
        ::Type{RT},
        C::Annotation,
        A::Annotation,
        α::Annotation{<:Number},
        β::Annotation{<:Number},
    ) where {RT}
    !isa(C, Const) && !isa(A, Const) && add!(C.dval, A.dval, α.val, β.val)
    !isa(C, Const) && !isa(α, Const) && add!(C.dval, A.val, α.dval, One())
    !isa(C, Const) && !isa(β, Const) && add!(C.dval, C.val, β.dval, One())
    add!(C.val, A.val, α.val, β.val)
    if EnzymeRules.needs_primal(config) &&  EnzymeRules.needs_shadow(config)
        return C
    elseif EnzymeRules.needs_primal(config)
        return C.val
    elseif EnzymeRules.needs_shadow(config)
        return C.dval
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(inner)},
        ::Type{RT},
        A::Annotation,
        B::Annotation,
    ) where {RT}
    cacheA = EnzymeRules.overwritten(config)[2] ? copy(A.val) : A.val
    cacheB = EnzymeRules.overwritten(config)[3] ? copy(B.val) : B.val
    cache = (cacheA, cacheB)
    ret = inner(A.val, B.val)
    shadow = EnzymeRules.needs_shadow(config) ? zero(ret) : nothing
    primal = EnzymeRules.needs_primal(config) ? ret : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(inner)},
        dret::Active,
        cache,
        A::Annotation,
        B::Annotation,
    )
    ΔS = dret.val
    Aval, Bval = cache
    !isa(A, Const) && add!(A.dval, Bval, conj(ΔS))
    !isa(B, Const) && add!(B.dval, Aval, ΔS)
    return (nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(inner)},
        RT::Type{<:Const},
        cache,
        A::Annotation,
        B::Annotation,
    )
    return (nothing, nothing)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        func::Const{typeof(inner)},
        ::Type{RT},
        A::Annotation,
        B::Annotation,
    ) where {RT}
    ret = inner(A.val, B.val)
    if EnzymeRules.needs_shadow(config) # only compute this if actually needed
        dret = zero(ret)
        !isa(A, Const) && (dret += inner(A.dval, B.val))
        !isa(B, Const) && (dret += inner(A.val, B.dval))
    else
        dret = nothing
    end
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(ret, dret)
    elseif EnzymeRules.needs_primal(config)
        return ret
    elseif EnzymeRules.needs_shadow(config)
        return dret
    else
        return nothing
    end
end

end
