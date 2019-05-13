import Base: +, *, rand

using Distributions
using LinearAlgebra
using SpecialFunctions


ElementOrVector{T} = Union{T,Vector{T}}

abstract type Kernel end

"""
Gaussian kernel / radial basis function, RBF
"""
struct GaussianKernel{T <: Real} <: Kernel
    theta1::T
    theta2::T
    function GaussianKernel(theta1::T, theta2::T) where {T <: Real}
        theta2 == 0 ? throw(DomainError(theta2, "theta2 must not be zero")) : new{T}(theta1, theta2)
    end
end

function ker(k::GaussianKernel, x1::ElementOrVector{T}, x2::ElementOrVector{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    k.theta1 * exp(- sum(abs.(x1 - x2).^2) / k.theta2)
end


"""
Linear kernel
"""
struct LinearKernel <: Kernel end

function ker(k::LinearKernel, x1::ElementOrVector{T}, x2::ElementOrVector{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    1 + dot(x1, x2)
end


"""
Exponential kernel, Ornstein-Uhlenbeck
"""
struct ExponentialKernel{T <: Real} <: Kernel
    theta::T
    function ExponentialKernel(theta::T) where {T <: Real}
        theta == 0 ? throw(DomainError(theta, "theta must not be zero")) : new{T}(theta)
    end
end

function ker(k::ExponentialKernel, x1::ElementOrVector{T}, x2::ElementOrVector{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    return exp(- sum(abs.(x1 - x2)) / k.theta)
end


"""
Periodic kernel
"""
struct PeriodicKernel{T <: Real} <: Kernel
    theta1::T
    theta2::T
    function PeriodicKernel(theta1::T, theta2::T) where {T <: Real}
        theta2 == 0 ? throw(DomainError(theta2, "theta2 must not be zero")) : new{T}(theta1, theta2)
    end
end

function ker(k::PeriodicKernel, x1::ElementOrVector{T}, x2::ElementOrVector{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    exp(k.theta1 * cos(sum(abs.(x1 - x2) / k.theta2)))
end

"""
Matern kernel

https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
"""
struct MaternKernel{T <: Real} <: Kernel
    nu::T
    theta::T
    function MaternKernel(nu::T, theta::T) where {T <: Real}
        theta == 0 ? throw(DomainError(theta, "theta must not be zero")) : new{T}(nu, theta)
    end
end

function ker(k::MaternKernel, x1::ElementOrVector{T}, x2::ElementOrVector{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    if x1 == x2
        return 1.0
    end
    r = sum(abs.(x1 - x2))
    t = sqrt(2 * k.nu) * r / k.theta
    2^(1 - k.nu) / gamma(k.nu) * t^k.nu * besselk(k.nu, t)
end


"""
Kernel sum
"""
struct KernelSum <: Kernel
    kernel1::Kernel
    kernel2::Kernel
end

function ker(k::KernelSum, x1::ElementOrVector{T}, x2::ElementOrVector{T}) where {T <: Real}
    ker(k.kernel1, x1, x2) + ker(k.kernel2, x1, x2)
end


"""
Kernel product 
"""
struct KernelProduct <: Kernel
    kernel1::Kernel
    kernel2::Kernel
end

function ker(k::KernelProduct, x1::ElementOrVector{T}, x2::ElementOrVector{T}) where {T <: Real}
    ker(k.kernel1, x1, x2) * ker(k.kernel2, x1, x2)
end


"""
Kernel scalar product 
"""
struct KernelScalarProduct{T <: Real} <: Kernel
    scale::T
    kernel::Kernel
end

function ker(k::KernelScalarProduct, x1::ElementOrVector{T}, x2::ElementOrVector{T}) where {T <: Real}
    k.scale * ker(k.kernel, x1, x2)
end


function +(k1::Kernel, k2::Kernel)
    KernelSum(k1, k2)
end

function *(k1::Kernel, k2::Kernel)
    KernelProduct(k1, k2)
end

function *(k::Kernel, scale::T) where {T <: Real}
    KernelScalarProduct{T}(scale, k)
end

function *(scale::T, k::Kernel) where {T <: Real}
    KernelScalarProduct{T}(scale, k)
end


"""
Gaussian Process
"""
struct GaussianProcess{K <: Kernel}
    kernel::K
    eta::Float64 # regularization parameter
    GaussianProcess(kernel::K) where {K <: Kernel} = new{K}(kernel, 1e-6)
    GaussianProcess(kernel::K, eta::T) where {K <: Kernel,T <: Real} = new{K}(kernel, Float64(eta))
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
        c += gp.eta .* Matrix{Float64}(I, n, n) 
    end
    c
end

# sampling function
function rand(gp::GaussianProcess{K}, xs::Array{T}) where {K <: Kernel,T}
    l = size(xs, 1)
    k = cov(gp, xs)
    Base.rand(MvNormal(zeros(l), k))
end

function rand(gp::GaussianProcess{K}, xs::Array{T}, n::Int) where {K <: Kernel,T}
    l = size(xs, 1)
    k = cov(gp, xs)
    Base.rand(MvNormal(zeros(l), k), n)
end


function gpr(gp::GaussianProcess{K}, xtest::Array{T},
            xtrain::Array{T}, ytrain::Array{T}) where {K <: Kernel,T}
    Base.length(xtrain) == Base.length(ytrain) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    k = cov(gp, xtrain)
    k_star = cov(gp, xtrain, xtest)
    s = cov(gp, xtest)

    k_inv = inv(k)
    k_star_inv = k_star' * k_inv
    MvNormal(k_star_inv * ytrain, Symmetric(s - k_star_inv * k_star))
end
