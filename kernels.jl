"""
Abstract kernel
"""
abstract type Kernel end
Base.length(k::Kernel) = Base.length(fieldnames(typeof(k)))

"""
Covariance matrix
"""
function cov(k::Kernel, xs1::AbstractVector, xs2::AbstractVector)
    n1 = size(xs1, 1)
    n2 = size(xs2, 1)
    c = zeros(n1, n2)
    for i in 1:n1
        for j in 1:n2
            c[i, j] = ker(k, xs1[i], xs2[j])
        end
    end
    c
end

cov(k::Kernel, xs::AbstractVector) = cov(k, xs, xs)


abstract type BaseKernel <: Kernel end

"""
Gaussian kernel / radial basis function, RBF
"""
mutable struct GaussianKernel <: BaseKernel
    θ::Float64
    function GaussianKernel(θ::Float64)
        @assert θ != 0
        new(θ)
    end
end

# Outer constructors
GaussianKernel(θ::Real) = GaussianKernel(Float64(θ))
GaussianKernel() = GaussianKernel(1.0)

ker(k::GaussianKernel, x1::Real, x2::Real) = exp(-(x1 - x2)^2 / k.θ)

function ker(k::GaussianKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T <: Real,S <: Real}
    exp(-sum((x1 - x2).^2) / k.θ)
end

function logderiv(k::GaussianKernel, x1::Real, x2::Real)
    # derivative for parameter estimation
    # return ∂g/∂τ, where
    # g(τ) = ker(exp(τ)), exp(τ)=θ
    [ker(k, x1, x2) / k.θ * (x1 - x2)^2]
end

function logderiv(k::GaussianKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T <: Real,S <: Real}
    [ker(k, x1, x2) / k.θ * sum((x1 - x2).^2)]
end

function update!(k::GaussianKernel, θ::Real)
    @assert θ != 0
    k.θ = Float64(θ)
    k
end


"""
Linear kernel
"""
struct LinearKernel <: BaseKernel end

ker(k::LinearKernel, x1::Real, x2::Real) = 1 + x1 * x2

function ker(k::LinearKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T <: Real,S <: Real}
    1 + dot(x1, x2)
end

logderiv(k::LinearKernel, x1::Real, x2::Real) =  []

function logderiv(k::LinearKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T <: Real,S <: Real}
    []
end

update!(k::LinearKernel) = k


"""
Exponential kernel, Ornstein-Uhlenbeck
"""
mutable struct ExponentialKernel <: BaseKernel
    θ::Float64
    function ExponentialKernel(θ::Float64)
        @assert θ != 0
        new(θ)
    end
end

# Outer constructors
ExponentialKernel(θ::Real) = ExponentialKernel(Float64(θ))
ExponentialKernel() = ExponentialKernel(1.0)

ker(k::ExponentialKernel, x1::Real, x2::Real) = exp(-abs(x1 - x2) / k.θ)

function ker(k::ExponentialKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T <: Real,S <: Real}
    exp(-sum(abs.(x1 - x2)) / k.θ)
end

function logderiv(k::ExponentialKernel, x1::Real, x2::Real)
    [ker(k, x1, x2) / k.θ * abs(x1 - x2)]
end

function logderiv(k::ExponentialKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T <: Real,S <: Real}
    [ker(k, x1, x2) / k.θ * sum(abs.(x1 - x2))]
end

function update!(k::ExponentialKernel, θ::Real)
    k.θ = Float64(θ)
    k
end


"""
Periodic kernel
"""
mutable struct PeriodicKernel <: BaseKernel
    θ1::Float64
    θ2::Float64
    function PeriodicKernel(θ1::Float64, θ2::Float64)
        @assert θ2 != 0
        new(θ1, θ2)
    end
end

# Outer constructors
PeriodicKernel(θ1::Real, θ2::Real) = PeriodicKernel(Float64(θ1), Float64(θ2))
PeriodicKernel() = PeriodicKernel(1.0, 1.0)

function ker(k::PeriodicKernel, x1::Real, x2::Real)
    exp(k.θ1 * cos(abs(x1 - x2) / k.θ2))
end

function ker(k::PeriodicKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T <: Real,S <: Real}
    exp(k.θ1 * cos(sum(abs.(x1 - x2)) / k.θ2))
end

function logderiv(k::PeriodicKernel, x1::Real, x2::Real)
    k_ker = ker(x1, x2)
    t = abs(x1 - x2) / k.θ2
    [k.θ1 * cos(t) * k_ker, k.θ1 * t * sin(t) * k_ker]
end

function logderiv(k::PeriodicKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T <: Real,S <: Real}
    k_ker = ker(x1, x2)
    t = sum(abs.(x1 - x2) / k.θ2)
    [k.θ1 * cos(t) * k_ker, k.θ1 * t * sin(t) * k_ker]
end

function update!(k::PeriodicKernel, θ1::Real, θ2::Real)
    k.θ1 = Float64(θ1)
    k.θ2 = Float64(θ2)
    k
end


"""
Matern kernel
https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
"""
mutable struct MaternKernel <: BaseKernel
    ν::Float64
    θ::Float64
    function MaternKernel(ν::Float64, θ::Float64)
        @assert θ != 0
        new(ν, θ)
    end
end

# Outer constructors
MaternKernel(ν::Real, θ::Real) = MaternKernel(Float64(ν), Float64(θ))

function ker(k::MaternKernel, x1::Real, x2::Real)
    if x1 == x2
        return 1.0
    end
    r = abs(x1 - x2)
    t = sqrt(2 * k.ν) * r / k.θ
    2^(1 - k.ν) / gamma(k.ν) * t^k.ν * besselk(k.ν, t)
end

function ker(k::MaternKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T <: Real,S <: Real}
    if x1 == x2
        return 1.0
    end
    r = sum(abs.(x1 - x2))
    t = sqrt(2 * k.ν) * r / k.θ
    2^(1 - k.ν) / gamma(k.ν) * t^k.ν * besselk(k.ν, t)
end

function logderiv(k::MaternKernel, x1::Real, x2::Real)
    throw("unimplemented")
end

function logderiv(k::MaternKernel, x1::Array{T}, x2::Array{S}) where {T <: Real,S <: Real}
    throw("unimplemented")
end

function update!(k::MaternKernel, ν::Real, θ::Real)
    k.ν = Float64(ν)
    k.θ = Float64(θ)
    k
end


"""
Constant kernel
"""
struct ConstantKernel <: BaseKernel end

ker(k::ConstantKernel, x1::Real, x2::Real) = 1.0

function ker(k::ConstantKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T <: Real,S <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    return 1.0
end

logderiv(k::ConstantKernel, x1::Real, x2::Real) = []

function logderiv(k::ConstantKernel, x1::AbstractVector{T}, x2::AbstractVector{S}) where {T <: Real,S <: Real}
    []
end

update!(k::ConstantKernel) = k


"""
Kernel product
"""
mutable struct KernelProduct <: Kernel
    coef::Float64
    kernel::Vector{T} where {T <: BaseKernel}
end

# Outer constructors
KernelProduct(k::BaseKernel) = KernelProduct(1.0, [k])
KernelProduct(coef::Real, k::BaseKernel) = KernelProduct(Float64(coef), [k])

function ker(k::KernelProduct, x1::S, x2::T) where {S,T}
    k.coef * prod([ker(kr, x1, x2) for kr in k.kernel])
end

function logderiv(k::KernelProduct, x1::S, x2::T) where {S,T}
    ks = [ker(kr, x1, x2) for kr in k.kernel]
    mat = repeat(ks, Base.length(ks))
    for i in 1:Base.length(k.kernel)
        mat[i, i] = 1
    end
    kerprod = prod(mat, dims = 1)[:]
    deriv = vcat([kerprod[i] .* logderiv(kr, x1, x2) for (i, kr) in enumerate(k.kernel)]...)
    [ker(k, x1, x2), deriv...]
end

Base.length(k::KernelProduct) = 1 + sum([Base.length(kr) for kr in k.kernel])

function update!(k::KernelProduct, params::Real...)
    @assert Base.length(params) == Base.length(k)
    k.coef = params[1]
    l = 2
    for kr in k.kernel
        l_kr = Base.length(kr)
        update!(kr, params[l:l + l_kr - 1]...)
        l += l_kr
    end
end

*(coef::Real, k::BaseKernel) =  KernelProduct(coef, k)
*(k::BaseKernel, coef::Real) = coef * k
*(coef::Real, k::KernelProduct) =  KernelProduct(Float64(coef) * k.coef, k.kernel)
*(k::KernelProduct, coef::Real) = coef * k

*(k1::BaseKernel, k2::BaseKernel) = KernelProduct(1.0, [k1, k2])
*(k1::KernelProduct, k2::BaseKernel) = KernelProduct(k1.coef, [k1.kernel..., k2])
*(k1::BaseKernel, k2::KernelProduct) = k2 * k1
*(k1::KernelProduct, k2::KernelProduct) = KernelProduct(k1.coef * k2.coef, [k1.kernel..., k2.kernel...])

-(k::BaseKernel) = -1.0 * k
-(k::KernelProduct) = -1.0 * k


"""
Kernel sum
"""
mutable struct KernelSum <: Kernel
    kernel::Vector{KernelProduct}
end

function ker(k::KernelSum, x1::S, x2::T) where {S,T}
    sum([ker(kr, x1, x2) for kr in k.kernel])
end

function logderiv(k::KernelSum, x1::S, x2::T) where {S,T}
    vcat([logderiv(kr, x1, x2) for kr in k.kernel]...)
end

Base.length(k::KernelSum) = sum([Base.length(kr) for kr in k.kernel])

function update!(k::KernelSum, params::Real...)
    @assert Base.length(params) == Base.length(k)
    l = 1
    for kr in k.kernel
        l_kr = Base.length(kr)
        update!(kr, params[l:l + l_kr - 1]...)
        l += l_kr
    end
end

*(coef::Real, k::KernelSum) =  KernelSum([coef * kr for kr in k.kernel])
*(k::KernelSum, coef::Real) = coef * k

+(k1::KernelProduct, k2::KernelProduct) = KernelSum([k1, k2])
+(k1::KernelSum, k2::KernelProduct) = KernelSum([k1.kernel..., k2])

+(k1::BaseKernel, k2::BaseKernel) = KernelProduct(k1) + KernelProduct(k2)
+(k1::Kernel, k2::BaseKernel) = k1 + KernelProduct(k2)
+(k1::BaseKernel, k2::Kernel) = k2 + k1

-(k1::Kernel, k2::Kernel) = k1 + (-k2)

