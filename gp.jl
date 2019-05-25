import Base: +, -, *

using Distributions
using LinearAlgebra
using SpecialFunctions


"""
Abstract kernel
"""
abstract type Kernel end
Base.length(k::Kernel) = Base.length(fieldnames(typeof(k)))

function cov(k::Kernel, xs::Array, ys::Array)
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

cov(k::Kernel, xs::Array) = cov(k, xs, xs)


abstract type BaseKernel <: Kernel end


"""
Gaussian kernel / radial basis function, RBF
"""
mutable struct GaussianKernel <: BaseKernel
    theta::Float64
    function GaussianKernel(theta::Real)
        @assert theta != 0
        new(Float64(theta))
    end
end

function ker(k::GaussianKernel, x1::Array{<: Real}, x2::Array{<: Real})
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    exp(- sum((x1 - x2).^2) / k.theta)
end

function logderiv(k::GaussianKernel, x1::Array{<: Real}, x2::Array{<: Real})
    # derivative for parameter estimation
    # return ∂g/∂τ, where
    # g(τ) = ker(exp(τ)), exp(τ)=θ
    [ker(k, x1, x2) / k.theta * sum((x1 - x2).^2)]
end

function update!(k::GaussianKernel, theta::Real)
    @assert theta != 0
    k.theta = Float64(theta)
    k
end


"""
Linear kernel
"""
struct LinearKernel <: BaseKernel end

function ker(k::LinearKernel, x1::Array{<: Real}, x2::Array{<: Real})
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    1 + dot(x1, x2)
end

function logderiv(k::LinearKernel, x1::Array{<: Real}, x2::Array{<: Real})
    []
end

update!(k::LinearKernel) = k


"""
Exponential kernel, Ornstein-Uhlenbeck
"""
mutable struct ExponentialKernel <: BaseKernel
    theta::Float64
    function ExponentialKernel(theta::Real)
        @assert theta != 0
        new(Float64(theta))
    end
end

function ker(k::ExponentialKernel, x1::Array{<: Real}, x2::Array{<: Real})
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    exp(-sum(abs.(x1 - x2)) / k.theta)
end

function logderiv(k::ExponentialKernel, x1::Array{<: Real}, x2::Array{<: Real})
    [ker(k, x1, x2) / k.theta * sum(abs.(x1 - x2))]
end

function update!(k::ExponentialKernel, theta::Real)
    k.theta = Float64(theta)
    k
end


"""
Periodic kernel
"""
mutable struct PeriodicKernel <: BaseKernel
    theta1::Float64
    theta2::Float64
    function PeriodicKernel(theta1::Real, theta2::Real)
        @assert theta2 != 0
        new(Float64(theta1), Float64(theta2))
    end
end

function ker(k::PeriodicKernel, x1::Array{<: Real}, x2::Array{<: Real})
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    exp(k.theta1 * cos(sum(abs.(x1 - x2) / k.theta2)))
end

function logderiv(k::PeriodicKernel, x1::Array{<: Real}, x2::Array{<: Real})
    k_ker = ker(x1, x2)
    t = sum(abs.(x1 - x2) / k.theta2)
    [k.theta1 * cos(t) * k_ker, k.theta1 * t * sin(t) * k_ker]
end

function update!(k::PeriodicKernel, theta1::Real, theta2::Real)
    k.theta1 = Float64(theta1)
    k.theta2 = Float64(theta2)
    k
end


"""
Matern kernel
https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
"""
mutable struct MaternKernel <: BaseKernel
    nu::Float64
    theta::Float64
    function MaternKernel(nu::Real, theta::Real)
        @assert theta != 0
        new(Float64(nu), Float64(theta))
    end
end

function ker(k::MaternKernel, x1::Array{<: Real}, x2::Array{<: Real})
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    if x1 == x2
        return 1.0
    end
    r = sum(abs.(x1 - x2))
    t = sqrt(2 * k.nu) * r / k.theta
    2^(1 - k.nu) / gamma(k.nu) * t^k.nu * besselk(k.nu, t)
end

function logderiv(k::MaternKernel, x1::Array{<: Real}, x2::Array{<: Real})
    throw("unimplemented")
end

function update!(k::MaternKernel, nu::Real, theta::Real)
    k.nu = Float64(nu)
    k.theta = Float64(theta)
    k
end


"""
Kernel product
"""
mutable struct KernelProduct <: Kernel
    coef::Float64
    kernel::Vector{<: BaseKernel}
end

function ker(k::KernelProduct, x1::Array, x2::Array)
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    k.coef * prod([ker(kr, x1, x2) for kr in k.kernel])
end

function logderiv(k::KernelProduct, x1::Array, x2::Array)
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


"""
Kernel sum
"""
mutable struct KernelSum <: Kernel
    kernel::Vector{<: KernelProduct}
end

function ker(k::KernelSum, x1::Array, x2::Array)
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    sum([ker(kr, x1, x2) for kr in k.kernel])
end

function logderiv(k::KernelSum, x1::Array, x2::Array)
    vcat([logderiv(kr, x1, x2) for kr in k.kernel]...)
end

Base.length(k::KernelSum) = sum([Base.length(kr) for kr in k.kernel])

function update!(k::KernelSum, params::Real...)
    @assert Base.length(params) == Base.length(k)
    l = 1
    for kr in k.kernel
        update!(kr, params[l, l + Base.length(kr) - 1]...)
        l += Base.length(kr)
    end
end


*(coef::Real, k::BaseKernel) =  KernelProduct(Float64(coef), [k])
*(k::BaseKernel, coef::Real) = *(coef::Real, k::BaseKernel) 
*(coef::Real, k::KernelProduct) =  KernelProduct(Float64(coef) * k.coef, k.kernel)
*(k::KernelProduct, coef::Real) = *(coef::Real, k::KernelProduct)
*(coef::Real, k::KernelSum) =  KernelSum([coef * kr for kr in k.kernel])
*(k::KernelSum, coef::Real) = *(coef::Real, k::KernelSum)


*(k1::BaseKernel, k2::BaseKernel) = KernelProduct(1.0, [k1, k2])
*(k1::KernelProduct, k2::BaseKernel) = KernelProduct(k1.coef, [k1.kernel..., k2])

+(k1::Kernel, k2::Kernel) = KernelSum([k1, k2])
-(k1::Kernel, k2::Kernel) = KernelSum([k1, -1.0 * k2])

+(k1::KernelSum, k2::Kernel) = KernelSum([k1.kernel..., k2])
-(k1::KernelSum, k2::Kernel) = KernelSum([k1.kernel..., -k2])


"""
Gaussian Process
"""
mutable struct GaussianProcess{K <: Kernel}
    kernel::K
    eta::Float64 # regularization parameter
    GaussianProcess(kernel::K) where {K <: Kernel} = new{K}(kernel, 1e-6)
    GaussianProcess(kernel::K, eta::Real) where {K <: Kernel} = new{K}(kernel, Float64(eta))
end

function update!(gp::GaussianProcess, params::Real...)
    update!(gp.kernel, params[1:end - 1]...)
    gp.eta = params[end]
    gp
end

Base.length(gp::GaussianProcess) = Base.length(gp.kernel) + 1

cov(gp::GaussianProcess, xs::Array, ys::Array{<: Real}) = cov(gp.kernel, xs, ys)

function cov(gp::GaussianProcess, xs::Array, reg::Bool = true)
    c = cov(gp.kernel, xs)
    # regularlize
    if reg == true
        n = size(xs, 1)
        c += gp.eta * Matrix{Float64}(I, n, n) 
    end
    c
end

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


"""
Parameter calibrator
"""
mutable struct ParamCalibrator
    gp::GaussianProcess
    xs::Vector{Float64}
    ys::Vector{Float64}
    k::Matrix{Float64}
    k_inv::Matrix{Float64}
    k_inv_y::Vector{Float64}
end

function ParamCalibrator(gp::GaussianProcess, xs::Vector{Float64}, ys::Vector{Float64})
    k = cov(gp, xs)
    k_inv = inv(k)
    k_inv_y = k_inv * ys
    ParamCalibrator(gp, xs, ys, k, k_inv, k_inv_y)
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

function deriv(pc::ParamCalibrator, d_mat::Matrix{<: Real})
    -tr(pc.k_inv * d_mat) + pc.k_inv_y' * d_mat * pc.k_inv_y
end

function fg!(pc::ParamCalibrator, F, G, x)
    # -logp and gradient
    update!(pc, x)
    y = exp.(x)
    n = size(pc.xs, 1)

    # gradient
    if G != nothing
        d_tensor = zeros(n, n, Base.length(pc.gp))
        for i in 1:n
            for j in 1:n
                t = logderiv(pc.gp.kernel, pc.xs[i, :], pc.xs[j, :])
                d_tensor[i, j, 1:end - 1] = t
            end
        end
        d_tensor[:, :, end] = y[end] .* Matrix{Float64}(I, n, n) # eta
        G .= mapslices(x->-deriv(pc, x), d_tensor, dims = [1, 2])[:]
    end

    # log likelihoood
    if F != nothing
        return -logp(pc)
    end
end