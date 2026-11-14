mutable struct Scalar{T, A} <: Number
    a::A
    Scalar{T, A}(a::A) where {T <: Number, A <: AbstractVector{T}} = new(a)
    Scalar{T, A}(s::T) where {T <: Number, A <: AbstractVector{T}} = Scalar{T, A}(fill!(A(undef, 1), s))
    Scalar{T, A}(s::T2) where {T <: Number, T2 <: Number, A <: AbstractVector{T}} = Scalar{T, A}(fill!(A(undef, 1), convert(T, s)))
    Scalar{T, A}(::One) where {T <: Number, A <: AbstractVector{T}} = new(fill!(A(undef, 1), one(T)))
    Scalar{T, A}(::Zero) where {T <: Number, A <: AbstractVector{T}} = new(fill!(A(undef, 1), zero(T)))
end
Scalar{T}(a::A) where {T <: Number, A <: AbstractVector{T}} = Scalar{T, A}(a)
function Scalar(a::A) where {T <: Number, A <: AbstractArray{T}}
    length(a) == 1 || throw(ArgumentError(lazy"underlying array for Scalar must have length 1"))
    return Scalar{T}(vec(a))
end

scalartype(::Scalar{T, A}) where {T, A} = T

Base.setindex!(s::Scalar{T, A}, v::T) where {T <: Number, A} = fill!(s.a, v)
Base.:*(s::Scalar, α::Number) = Scalar(scale(s.a, α))
Base.:*(s::Scalar, α::One) = s
Base.:*(α::One, s::Scalar) = s
Base.:*(s::Scalar, α::Zero) = Scalar(scale(s.a, α))
Base.:*(α::Zero, s::Scalar) = Scalar(scale(s.a, α))
Base.:*(α::Number, s::Scalar) = Scalar(scale(s.a, α))
Base.:*(s1::Scalar, s2::Scalar) = Scalar(s1.a .* s2.a)
Base.:+(s::Scalar, α::Number) = Scalar(s.a .+ α)
Base.:+(α::Number, s::Scalar) = Scalar(s.a .+ α)
Base.:+(s::Scalar, α::Zero) = s
Base.:+(α::Zero, s::Scalar) = s
Base.:+(s1::Scalar, s2::Scalar) = Scalar(s1.a .+ s2.a)
Base.:(==)(s1::Scalar, s2::Scalar) = all(s1.a .== s2.a)

Base.broadcastable(s::Scalar) = s.a

Base.iszero(s::Scalar) = all(iszero, s.a)
Base.real(s::Scalar) = Scalar(Base.real.(s.a))
Base.imag(s::Scalar) = Scalar(Base.imag.(s.a))
Base.conj(s::Scalar) = Scalar(Base.conj.(s.a))
Base.rtoldefault(::Type{<:Scalar{T}}) where {T} = Base.rtoldefault(T)
Base.isapprox(x::Scalar, y::Number; kwargs...) = all(_x -> isapprox(_x, y; kwargs...), x.a)
Base.convert(::Type{T}, x::Scalar) where {T <: Number} = Scalar(convert.(T, x.a))
