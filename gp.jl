import Base: +, -, *, rand

using Distributions
using LinearAlgebra
using SpecialFunctions


"""
Abstract kernel
"""
abstract type Kernel end
length(k::Kernel) = Base.length(fieldnames(typeof(k)))

function cov(k::Kernel, xs::Array{T}, ys::Array{T}) where {T}
    # covariance matrix
    nx = size(xs, 1)
    ny = size(ys, 1)
    c = zeros(nx, ny)
    for i in 1:nx
        for j in 1:ny
            c[i, j] = ker(k, xs[i, :], ys[j, :])
        end
    end
    c
end

cov(k::Kernel, xs::Array{T}) where {T} = cov(k, xs, xs)


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
    k.theta1 * exp(- sum((x1 - x2).^2) / k.theta2)
end

function logderiv(k::GaussianKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    # derivative for parameter estimation
    # return ∂g/∂τ, where
    # g(τ) = ker(exp(τ)), exp(τ)=θ
    
    k_ker = ker(k, x1, x2)
    [k_ker, k_ker / k.theta2 * sum((x1 - x2).^2)]
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

function logderiv(k::LinearKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    []
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

function logderiv(k::ExponentialKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    [ker(k, x1, x2) / k.theta * sum(abs.(x1 - x2))]
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

function logderiv(k::PeriodicKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    k_ker = ker(x1, x2)
    t = sum(abs.(x1 - x2) / k.theta2)
    [k.theta1 * cos(t) * k_ker, k.theta1 * t * sin(t) * k_ker]
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

function logderiv(k::MaternKernel, x1::Array{T}, x2::Array{T}) where {T <: Real}
    throw("unimplemented")
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

cov(gp::GaussianProcess{K}, xs::Array{T}, ys::Array{T}) where {K <: Kernel,T} = cov(gp.kernel, xs, ys)

function cov(gp::GaussianProcess{K}, xs::Array{T}, reg::Bool = true) where {K <: Kernel,T}
    c = cov(gp.kernel, xs)
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


"""
Parameter calibrator
"""
mutable struct ParamCalibrator
    gp::GaussianProcess
    xs::Vector{Float64}
    ys::Vector{Float64}
    n_xs::Int64
    distance_matrix::Matrix{Float64}
    k::Matrix{Float64}
    k_inv::Matrix{Float64}
    k_inv_y::Vector{Float64}
end

function ParamCalibrator(gp::GaussianProcess, xs::Vector{Float64}, ys::Vector{Float64})
    n_xs = Base.length(xs)
    distance_matrix = zeros(n_xs, n_xs)
    for i in 1:n_xs
        for j in 1:n_xs
            distance_matrix[i, j] = (xs[i] - xs[j]).^2
        end
    end
    k = cov(gp, xs)
    k_inv = inv(k)
    k_inv_y = k_inv * ys
    ParamCalibrator(gp, xs, ys, n_xs, distance_matrix, k, k_inv, k_inv_y)
end

function update!(pc::ParamCalibrator, params)
    y = exp.(params)
    update!(pc.gp, y...)
    pc.k = cov(pc.gp, pc.xs)
    pc.k_inv = inv(pc.k)
    pc.k_inv_y = pc.k_inv * pc.ys
    pc
end

function logp(pc::ParamCalibrator)
    - log(det(pc.k)) - pc.ys' * pc.k_inv * pc.ys
end

function deriv(pc::ParamCalibrator, d_mat::Matrix{T}) where {T <: Real}
    -tr(pc.k_inv * d_mat) + pc.k_inv_y' * d_mat * pc.k_inv_y
end

function fg!(pc::ParamCalibrator, F, G, x)
    # -logp and gradient
    update!(pc, x)
    y = exp.(x)

    # gradient
    if G != nothing
        d_tensor = zeros(pc.n_xs, pc.n_xs, length(pc.gp.kernel) + 1)
        for i in 1:pc.n_xs
            for j in 1:pc.n_xs
                d_tensor[i, j, 1:end - 1] = logderiv(pc.gp.kernel, pc.xs[i, :], pc.xs[j, :])
            end
        end
        d_tensor[:, :, end] = y[end] .* Matrix{Float64}(I, pc.n_xs, pc.n_xs) # eta
        G .= mapslices(x->-deriv(pc, x), d_tensor, dims = [1, 2])[:]
    end

    # log likelihoood
    if F != nothing
        return -logp(pc)
    end
end