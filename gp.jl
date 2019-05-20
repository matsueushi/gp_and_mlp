import Base: +, -, *, rand

using Distributions
using LinearAlgebra
using SpecialFunctions


abstract type Kernel end
length(k::Kernel) = Base.length(fieldnames(typeof(k)))

"""
Gaussian kernel / radial basis function, RBF
"""
mutable struct GaussianKernel <: Kernel
    theta1::Float64
    theta2::Float64
    function GaussianKernel(theta1::T, theta2::T) where {T <: Real}
        @assert theta2 != 0
        new(Float64(theta1), Float64(theta2))
    end
end

function ker(k::GaussianKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    k.theta1 * exp(- sum(abs.(x1 - x2).^2) / k.theta2)
end

function update!(k::GaussianKernel, theta1::T, theta2::T) where {T <: Real}
    k.theta1 = Float64(theta1)
    k.theta2 = Float64(theta2)
    k
end


"""
Linear kernel
"""
struct LinearKernel <: Kernel end

function ker(k::LinearKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    1 + dot(x1, x2)
end

update!(k::LinearKernel) = k


"""
Exponential kernel, Ornstein-Uhlenbeck
"""
mutable struct ExponentialKernel <: Kernel
    theta::Float64
    function ExponentialKernel(theta::T) where {T <: Real}
        @assert theta != 0
        new(Float64(theta))
    end
end

function ker(k::ExponentialKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    exp(-sum(abs.(x1 - x2)) / k.theta)
end

function update!(k::ExponentialKernel, theta::T) where {T <: Real}
    k.theta = Float64(theta)
    k
end


"""
Periodic kernel
"""
mutable struct PeriodicKernel <: Kernel
    theta1::Float64
    theta2::Float64
    function PeriodicKernel(theta1::T, theta2::T) where {T <: Real}
        @assert theta2 != 0
        new(Float64(theta1), Float64(theta2))
    end
end

function ker(k::PeriodicKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    exp(k.theta1 * cos(sum(abs.(x1 - x2) / k.theta2)))
end

function update!(k::PeriodicKernel, theta1::T, theta2::T) where {T <: Real}
    k.theta1 = Float64(theta1)
    k.theta2 = Float64(theta2)
    k
end


"""
Matern kernel
https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
"""
mutable struct MaternKernel <: Kernel
    nu::Float64
    theta::Float64
    function MaternKernel(nu::T, theta::T) where {T <: Real}
        @assert theta != 0
        new(Float64(nu), Float64(theta))
    end
end

function ker(k::MaternKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    if x1 == x2
        return 1.0
    end
    r = sum(abs.(x1 - x2))
    t = sqrt(2 * k.nu) * r / k.theta
    2^(1 - k.nu) / gamma(k.nu) * t^k.nu * besselk(k.nu, t)
end

function update!(k::MaternKernel, nu::T, theta::T) where {T <: Real}
    k.nu = Float64(nu)
    k.theta = Float64(theta)
    k
end


"""
Composite kernel
"""
mutable struct CompositeKernel <: Kernel
    op::Symbol
    kernel1::Kernel
    kernel2::Kernel
end

function ker(k::CompositeKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    ker1 = ker(k.kernel1, x1, x2)
    ker2 = ker(k.kernel2, x1, x2)
    eval(Expr(:call, k.op, ker1, ker2))
end

length(k::CompositeKernel) = length(k.kernel1) + length(k.kernel2)

function update!(k::CompositeKernel, params::T...) where {T <: Real}
    @assert Base.length(params) == length(k) 
    l1 = length(k.kernel)
    update!(k.kernel1, params[1:l1]...)
    update!(k.kernel2, params[l1 + 1:end]...)
    return k
end

+(k1::Kernel, k2::Kernel) = CompositeKernel(:+, k1, k2)
-(k1::Kernel, k2::Kernel) = CompositeKernel(:-, k1, k2)
*(k1::Kernel, k2::Kernel) = CompositeKernel(:*, k1, k2)


"""
Kernel scalar product
"""
mutable struct KernelScalarProduct <: Kernel
    scale::Float64
    kernel::Kernel
end

length(k::KernelScalarProduct) = 1 + length(k.kernel)

function ker(k::KernelScalarProduct, x1::Array{T}, x2::Array{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    k.scale * ker(k.kernel, x1, x2)
end

function update!(k::KernelScalarProduct, params::T...) where {T <: Real}
    @assert Base.length(params) == length(k) 
    k.scale = params[1]
    update!(k.kernel, params[2:end]...)
    k 
end

function *(scale::T, k::Kernel) where {T <: Real}
    KernelScalarProduct(Float64(scale), k)
end

*(k::Kernel, scale::T) where {T <: Real} = scale * k


"""
Gaussian Process
"""
mutable struct GaussianProcess{K <: Kernel}
    kernel::K
    eta::Float64 # regularization parameter
    GaussianProcess(kernel::K) where {K <: Kernel} = new{K}(kernel, 1e-6)
    GaussianProcess(kernel::K, eta::T) where {K <: Kernel,T <: Real} = new{K}(kernel, Float64(eta))
end

function update!(gp::GaussianProcess{K}, params::T...) where {K <: Kernel,T <: Real}
    update!(gp.kernel, params[1:end - 1]...)
    gp.eta = params[end]
    gp
end

# covariance matrix
function cov(gp::GaussianProcess{K}, xs::Array{T}, ys::Array{T}) where {K <: Kernel,T}
    nx = size(xs, 1)
    ny = size(ys, 1)
    c = zeros(nx, ny)
    for i in 1:nx
        for j in 1:ny
            c[i, j] = ker(gp.kernel, xs[i, :], ys[j, :])
        end
    end
    c
end

function cov(gp::GaussianProcess{K}, xs::Array{T}, reg::Bool = true) where {K <: Kernel,T}
    c = cov(gp, xs, xs)
    # regularlize
    if reg == true
        n = size(xs, 1)
        c += gp.eta * Matrix{Float64}(I, n, n) 
    end
    c
end

function dist(gp::GaussianProcess{K}, xs::Array{T}) where {K <: Kernel,T}
    l = size(xs, 1)
    k = cov(gp, xs)
    MvNormal(zeros(l), k)
end

function gpr(gp::GaussianProcess{K}, xtest::Array{T},
            xtrain::Array{T}, ytrain::Array{T}) where {K <: Kernel,T}
    Base.length(xtrain) == Base.length(ytrain) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    k_star = cov(gp, xtrain, xtest)
    s = cov(gp, xtest)

    k_inv = inv(cov(gp, xtrain))
    k_star_inv = k_star' * k_inv
    mu = k_star_inv * ytrain
    sig = Symmetric(s - k_star_inv * k_star)
    MvNormal(mu, sig)
end