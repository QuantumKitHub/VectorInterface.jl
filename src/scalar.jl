mutable struct Scalar{T, A} <: Number
    a::A
    Scalar{T, A}(a::A) where {T <: Number, A <: AbstractVector{T}} = new(a)
    Scalar{T, A}(s::T) where {T <: Number, A <: AbstractVector{T}} = Scalar{T, A}(fill!(A(undef, 1), s))
    Scalar{T, A}(s::T2) where {T <: Number, T2 <: Number, A <: AbstractVector{T}} = Scalar{T, A}(fill!(A(undef, 1), convert(T, s)))
    Scalar{T, A}(s::Scalar{T, A}) where {T <: Number, A <: AbstractVector{T}} = s
    Scalar{T, A}(::One) where {T <: Number, A <: AbstractVector{T}} = new(fill!(A(undef, 1), one(T)))
    Scalar{T, A}(::Zero) where {T <: Number, A <: AbstractVector{T}} = new(fill!(A(undef, 1), zero(T)))
end
Scalar{T}(a::A) where {T <: Number, A <: AbstractVector{T}} = Scalar{T, A}(a)
Scalar{T}(s::Scalar{T, A}) where {T <: Number, A <: AbstractVector{T}} = s
function Scalar(a::A) where {T <: Number, A <: AbstractArray{T}}
    length(a) == 1 || throw(ArgumentError(lazy"underlying array for Scalar must have length 1"))
    return Scalar{T}(vec(a))
end
Scalar(s::Scalar{T, A}) where {T <: Number, A <: AbstractVector{T}} = s

scalartype(::Scalar{T, A}) where {T, A} = T

Base.setindex!(s::Scalar{T, A}, v::T) where {T <: Number, A} = fill!(s.a, v)
Base.:*(s::Scalar, α::Number) = Scalar(scale(s.a, α))
Base.:*(s::Scalar, α::One) = s
Base.:*(α::One, s::Scalar) = s
Base.:*(s::Scalar, α::Zero) = Scalar(scale(s.a, α))
Base.:*(α::Zero, s::Scalar) = Scalar(scale(s.a, α))
Base.:*(α::Number, s::Scalar) = Scalar(scale(s.a, α))
Base.:*(s1::Scalar, s2::Scalar) = Scalar(s1.a .* s2.a)
Base.:/(s::Scalar, α::Number) = Scalar(s.a ./ α)
Base.:/(s::Scalar, α::One) = s
Base.:/(s::Scalar, α::Zero) = throw(DivideError()) 
Base.:/(α::One, s::Scalar) = inv(s)
Base.:/(α::Zero, s::Scalar) = s * α
Base.:/(α::Number, s::Scalar) = scale(inv(s), α)
Base.:/(s1::Scalar, s2::Scalar) = Scalar(s1.a ./ s2.a)
Base.:+(s::Scalar, α::Number) = Scalar(s.a .+ α)
Base.:+(α::Number, s::Scalar) = Scalar(s.a .+ α)
Base.:+(s::Scalar, α::Zero) = s
Base.:+(α::Zero, s::Scalar) = s
Base.:+(s1::Scalar, s2::Scalar) = Scalar(s1.a .+ s2.a)
Base.:-(s::Scalar, α::Number) = Scalar(s.a .- α)
Base.:-(α::Number, s::Scalar) = Scalar(α .- s.a)
Base.:-(s::Scalar, α::Zero) = s
Base.:-(α::Zero, s::Scalar) = -s
Base.:-(s1::Scalar, s2::Scalar) = Scalar(s1.a .- s2.a)
Base.:-(s::Scalar) = Scalar(-1 .* s.a)
Base.:(==)(s1::Scalar, s2::Scalar) = all(s1.a .== s2.a)
Base.inv(s::Scalar) = Scalar(inv.(s.a))
Base.abs(s::Scalar) = Scalar(abs.(s.a))
Base.isless(s::Scalar{T}, v::T) where {T <: Number} = all(Base.Fix2(isless, v), s.a)
Base.copyto!(s::Scalar, v) = (s.a .= v; return s)

Base.broadcastable(s::Scalar) = s.a

Base.iszero(s::Scalar) = all(iszero, s.a)
Base.real(s::Scalar) = Scalar(Base.real.(s.a))
Base.imag(s::Scalar) = Scalar(Base.imag.(s.a))
Base.conj(s::Scalar) = Scalar(Base.conj.(s.a))
Base.rtoldefault(::Type{<:Scalar{T}}) where {T} = Base.rtoldefault(T)
Base.isapprox(x::Scalar, y::Number; kwargs...) = all(_x -> isapprox(_x, y; kwargs...), x.a)
Base.isapprox(x::Scalar, y::Scalar; kwargs...) = all(splat((a, b) -> isapprox(a, b; kwargs...)), zip(x.a, y.a))

# split this in two to avoid ambiguity
(::Type{T})(s::Scalar) where {T <: Real} = T(first(collect(s.a)))
(::Type{T})(s::Scalar) where {T <: Complex} = T(first(collect(s.a)))
