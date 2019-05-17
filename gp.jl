import Base: +, *, rand

using Distributions
using LinearAlgebra
using SpecialFunctions


abstract type Kernel end
length(k::Kernel) = Base.length(fieldnames(typeof(k)))

"""
Gaussian kernel / radial basis function, RBF
"""
mutable struct GaussianKernel{T <: Real} <: Kernel
    theta1::T
    theta2::T
    function GaussianKernel(theta1::T, theta2::T) where {T <: Real}
        theta2 == 0 ? throw(DomainError(theta2, "theta2 must not be zero")) : new{T}(theta1, theta2)
    end
end

function ker(k::GaussianKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    k.theta1 * exp(- sum(abs.(x1 - x2).^2) / k.theta2)
end

function update(k::GaussianKernel, theta1::T, theta2::T) where {T <: Real}
    k.theta1 = theta1
    k.theta2 = theta2
    return k
end


"""
Linear kernel
"""
struct LinearKernel <: Kernel end

function ker(k::LinearKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    1 + dot(x1, x2)
end

update(k::LinearKernel) = k


"""
Exponential kernel, Ornstein-Uhlenbeck
"""
mutable struct ExponentialKernel{T <: Real} <: Kernel
    theta::T
    function ExponentialKernel(theta::T) where {T <: Real}
        theta == 0 ? throw(DomainError(theta, "theta must not be zero")) : new{T}(theta)
    end
end

function ker(k::ExponentialKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    return exp(- sum(abs.(x1 - x2)) / k.theta)
end

function update(k::ExponentialKernel, theta::T) where {T <: Real}
    k.theta = theta
    return k
end


"""
Periodic kernel
"""
mutable struct PeriodicKernel{T <: Real} <: Kernel
    theta1::T
    theta2::T
    function PeriodicKernel(theta1::T, theta2::T) where {T <: Real}
        theta2 == 0 ? throw(DomainError(theta2, "theta2 must not be zero")) : new{T}(theta1, theta2)
    end
end

function ker(k::PeriodicKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    exp(k.theta1 * cos(sum(abs.(x1 - x2) / k.theta2)))
end

function update(k::PeriodicKernel, theta1::T, theta2::T) where {T <: Real}
    k.theta1 = theta1
    k.theta2 = theta2
    return k
end


"""
Matern kernel

https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
"""
mutable struct MaternKernel{T <: Real} <: Kernel
    nu::T
    theta::T
    function MaternKernel(nu::T, theta::T) where {T <: Real}
        theta == 0 ? throw(DomainError(theta, "theta must not be zero")) : new{T}(nu, theta)
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

function update(k::MaternKernel, nu::T, theta::T) where {T <: Real}
    k.nu = nu
    k.theta = theta
    return k
end


"""
Kernel sum
"""
mutable struct KernelSum <: Kernel
    kernel1::Kernel
    kernel2::Kernel
end

function ker(k::KernelSum, x1::Array{T}, x2::Array{T}) where {T <: Real}
    ker(k.kernel1, x1, x2) + ker(k.kernel2, x1, x2)
end

length(k::KernelSum) = length(k.kernel1) + length(k.kernel2)

"""
Kernel product
"""
mutable struct KernelProduct <: Kernel
    kernel1::Kernel
    kernel2::Kernel
end


function ker(k::KernelProduct, x1::Array{T}, x2::Array{T}) where {T <: Real}
    ker(k.kernel1, x1, x2) * ker(k.kernel2, x1, x2)
end

length(k::KernelProduct) = length(k.kenrel1) + length(k.kernel2)

function update(k::Union{KernelSum,KernelProduct}, params::T...) where {T <: Real}
    l1 = length(k.kernel1)
    update(k.kernel1, params[1:l1]...)
    update(k.kernel2, params[l1 + 1:end]...)
    return k
end

"""
Kernel scalar product
"""
mutable struct KernelScalarProduct{T <: Real} <: Kernel
    scale::T
    kernel::Kernel
end

length(k::KernelScalarProduct) = 1 + length(k.kernel)

function ker(k::KernelScalarProduct, x1::Array{T}, x2::Array{T}) where {T <: Real}
    k.scale * ker(k.kernel, x1, x2)
end

function update(k::KernelScalarProduct, params::T...) where {T <: Real}
    k.scale = params[1]
    update(k.kernel, params[2:end]...)
    return k
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

function update(gp::GaussianProcess{K}, params::T...) where {K <: Kernel,T <: Real}
    update(gp.kernel, params...)
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

function dist(gp::GaussianProcess{K}, xs::Array{T}) where {K <: Kernel,T}
    l = size(xs, 1)
    k = cov(gp, xs)
    MvNormal(zeros(l), k)
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