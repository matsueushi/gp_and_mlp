"""
Abstract kernel
"""
abstract type Kernel end
Base.length(k::Kernel) = Base.length(fieldnames(typeof(k)))

"""
Covariance matrix
"""
function cov(k::Kernel, xs1::Array, xs2::Array)
    n1 = size(xs1, 1)
    n2 = size(xs2, 1)
    c = zeros(n1, n2)
    for i in 1:n1
        for j in 1:n2
            c[i, j] = ker(k, xs1[i, :], xs2[j, :])
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
Constant kernel
"""
struct ConstantKernel <: BaseKernel end

function ker(k::ConstantKernel, x1::Array{<: Real}, x2::Array{<: Real})
    Base.length(x1) == Base.length(x2) || throw(DimensionMismatch("size of x1 not equal to size of x2"))
    return 1.0
end

function logderiv(k::ConstantKernel, x1::Array{<: Real}, x2::Array{<: Real})
    []
end

update!(k::ConstantKernel) = k

"""
Kernel product
"""
mutable struct KernelProduct <: Kernel
    coef::Float64
    kernel::Vector{<: BaseKernel}
    KernelProduct(k::BaseKernel) = new(1.0, [k])
    KernelProduct(coef::Real, k::BaseKernel) = new(Float64(coef), [k])
end

function ker(k::KernelProduct, x1::Array, x2::Array)
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

