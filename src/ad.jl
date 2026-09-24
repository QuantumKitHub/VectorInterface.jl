function project_add!(C, A, α)
    TC = Base.promote_op(+, scalartype(A), scalartype(α))
    return if !(TC <: Real) && scalartype(C) <: Real
        add!(C, real(add!(zerovector(C, TC), A, α)))
    else
        add!(C, A, α)
    end
end

"""
    project_scalar(x::Number, dx::Number)

Project a computed tangent `dx` onto the correct tangent type for `x`.
For example, we might compute a complex `dx` but only require the real part.
"""
project_scalar(x::Number, dx::Number) = oftype(x, dx)
project_scalar(x::Real, dx::Complex) = project_scalar(x, real(dx))
