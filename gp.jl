import Base: +, -, *

using Distributions
using LinearAlgebra
using SpecialFunctions

include("kernels.jl")


"""
Kernel with noise
"""
mutable struct GPKernel{K <: Kernel}
    kernel::K
    eta::Float64 # regularization parameter
end

function GPKernel(kernel::K, eta::T) where {K <: Kernel,T <: Real}
    new{K}(kernel, Float64(eta))
end

function update!(gpk::GPKernel, params::Real...)
    update!(gpk.kernel, params[1:end - 1]...)
    gpk.eta = params[end]
    gpk
end

Base.length(gpk::GPKernel) = Base.length(gpk.kernel) + 1


"""
Prediction Method
"""
abstract type PredictMethod end

"""
Standard method
"""
struct GPStandard <: PredictMethod end

cov(_::GPStandard, k::GPKernel, xs1::Array, xs2::Array) = cov(k.kernel, xs1, xs2)

function cov(_::GPStandard, k::GPKernel, xs::Array)
    # regularlize
    n = size(xs, 1)
    cov(k.kernel, xs, xs) + k.eta * Matrix{Float64}(I, n, n) 
end


"""
Gaussian Process
"""
mutable struct GaussianProcess{K <: Kernel}
    gpk::GPKernel{K}
    method::PredictMethod
    GaussianProcess(kernel::K) where {K <: Kernel} = new{K}(GPKernel{K}(kernel, 1e-6), GPStandard())
    function GaussianProcess(kernel::K, eta::Real) where {K <: Kernel}
        new{K}(GPKernel{K}(kernel, eta), GPStandard())
    end
end

function update!(gp::GaussianProcess, params::Real...)
    update!(gp.gpk, params...)
    gp
end

cov(gp::GaussianProcess, xs1::Array, xs2::Array) = cov(gp.method, gp.gpk, xs1, xs2)

cov(gp::GaussianProcess, xs::Array) = cov(gp.method, gp.gpk, xs)

function dist(gp::GaussianProcess, xs::Array)
    l = size(xs, 1)
    k = cov(gp, xs)
    MvNormal(zeros(l), k)
end

function predict(gp::GaussianProcess, xtest::Array, xtrain::Array, ytrain::Array{<: Real})
    Base.length(xtrain) == Base.length(ytrain) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    k_star = cov(gp, xtrain, xtest)
    s = cov(gp, xtest)

    k_inv = inv(cov(gp, xtrain))
    k_star_inv = k_star' * k_inv
    mu = k_star_inv * ytrain
    sig = Symmetric(s - k_star_inv * k_star)
    MvNormal(mu, sig)
end

function logp(gp::GaussianProcess, xs::Array, ys::Array)
    k = cov(gp, xs)
    k_inv = inv(k)
    -log(det(k)) - ys' * k_inv * ys
end

function fg!(gp::GaussianProcess, xs::Array, ys::Array, F, G, params)
    # -logp and gradient
    y = exp.(params)
    update!(gp, y...)
    k = cov(gp, xs)
    k_inv = inv(k)
    k_inv_y = k_inv * ys

    n = size(xs, 1)

    function deriv(d_mat::Matrix{<: Real})
        -(-tr(k_inv * d_mat) + k_inv_y' * d_mat * k_inv_y)
    end

    # gradient
    if G != nothing
        d_tensor = zeros(n, n, Base.length(gp.gpk))
        for i in 1:n
            for j in 1:n
                t = logderiv(gp.gpk.kernel, xs[i, :], xs[j, :])
                d_tensor[i, j, 1:end - 1] = t
            end
        end
        d_tensor[:, :, end] = y[end] .* Matrix{Float64}(I, n, n) # eta
        G .= mapslices(deriv, d_tensor, dims = [1, 2])[:]
    end

    # log likelihoood
    if F != nothing
        return -(-log(det(k)) - ys' * k_inv * ys)
    end
end

