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
    η::Float64 # regularization parameter
end

GPKernel(kernel::Kernel, η::Real) = GPKernel(kernel, Float64(η))

function update!(gpk::GPKernel, params::Real...)
    update!(gpk.kernel, params[1:end - 1]...)
    gpk.η = params[end]
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

function cov(_::GPStandard, gpk::GPKernel, xs::AbstractVector)
    # regularlize
    n = size(xs, 1)
    cov(gpk.kernel, xs, xs) + gpk.η * Matrix{Float64}(I, n, n) 
end

function _predict(gps::GPStandard, gpk::GPKernel,  xtrain::AbstractVector{S}, 
    ytrain::AbstractVector{T}, xtest::AbstractVector{R}) where {T <: Real,R,S}

    Base.length(xtrain) == Base.length(ytrain) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    k_star = cov(gpk.kernel, xtrain, xtest)
    s = cov(gps, gpk, xtest)

    k_inv = inv(cov(gps, gpk, xtrain))
    k_star_inv = k_star' * k_inv
    mu = k_star_inv * ytrain
    sig = Symmetric(s - k_star_inv * k_star)
    mu, sig
end


"""
Inducing variable method
"""
mutable struct IVM <: PredictMethod
    ind_xs::Array
end

function cov(ivm::IVM, gpk::GPKernel, xs::AbstractVector)
    # compute K_mm, K_mn
    K_mm = cov(gpk.kernel, ivm.ind_xs, ivm.ind_xs)
    K_mn = cov(gpk.kernel, ivm.ind_xs, xs)

    n = size(xs, 1)

    function lambda(x, k)
        ker(gpk.kernel, x, x) - k' * K_mm * k
    end

    Λ = Diagonal([lambda(xs[i, :], K_mn[i, :]) for i in 1:n])
end


"""
Gaussian Process
"""
mutable struct GaussianProcess
    gpk::GPKernel{K} where {K <: Kernel}
    method::PredictMethod
end

# Outer Constructors
function GaussianProcess(kernel::Kernel, η::Real) 
    GaussianProcess(GPKernel(kernel, η), GPStandard())
end

GaussianProcess(kernel::Kernel) = GaussianProcess(kernel, 1e-6)

function update!(gp::GaussianProcess, params::Real...)
    update!(gp.gpk, params...)
    gp
end

cov(gp::GaussianProcess, xs1::Array, xs2::Array) = cov(gp.gpk.kernel, xs1, xs2)

cov(gp::GaussianProcess, xs::Array) = cov(gp.method, gp.gpk, xs)

function dist(gp::GaussianProcess, xs::AbstractVector)
    l = size(xs, 1)
    k = cov(gp, xs)
    MvNormal(zeros(l), k)
end

function predict(gp::GaussianProcess, xtrain::AbstractVector{S}, 
    ytrain::AbstractVector{T}, xtest::R) where {T <: Real,R,S}

    mu, var = _predict(gp.method, gp.gpk, xtrain, ytrain, [xtest])
    Normal(mu[1], sqrt(var[1]))
end

function predict(gp::GaussianProcess, xtrain::AbstractVector{S},
    ytrain::AbstractVector{T}, xtest::AbstractVector{R}) where {T <: Real,R,S}

    mu, sig = _predict(gp.method, gp.gpk, xtrain, ytrain, xtest)
    MvNormal(mu, sig)
end


function logp(gp::GaussianProcess, xs::AbstractVector, ys::AbstractVector{T}) where {T <: Real}
    k = cov(gp, xs)
    k_inv = inv(k)
    -log(det(k)) - ys' * k_inv * ys
end

function fg!(gp::GaussianProcess, xs::AbstractVector, ys::AbstractVector{T}, F, G, params) where {T <: Real}
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
        d_tensor[:, :, end] = y[end] .* Matrix{Float64}(I, n, n) # η
        G .= mapslices(deriv, d_tensor, dims = [1, 2])[:]
    end

    # log likelihoood
    if F != nothing
        return -(-log(det(k)) - ys' * k_inv * ys)
    end
end

