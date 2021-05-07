"""
Constructs a Lagrangian Neural Network [1]. This neural network is useful for
learning symmetries and conservation laws by supervision on the gradients
of the trajectories. It takes as input a concatenated vector of length `2n`
containing the position (of size `n`) and velocity (of size `n`) of the
particles. It then returns the acceleration of the particle following from 
the Euler-Lagrange equations.
!!! note
    This doesn't solve the Lagrangian Problem. Use [`NeuralLagrangianDE`](@ref)
    for such applications.
!!! note
    This layer currently doesn't support GPU. The support will be added in future
    with some AD fixes.
To obtain the gradients to train this network, ReverseDiff.gradient is supposed to
be used. This prevents the usage of `DiffEqFlux.sciml_train` or `Flux.train`. Follow
this [tutorial](https://diffeqflux.sciml.ai/dev/examples/lagrangian_nn/) to see how
to define a training loop to circumvent this issue.
```julia
LagrangianNN(model; p = nothing)
LagrangianNN(model::FastChain; p = initial_params(model))
```
Arguments:
1. `model`: A Chain or FastChain neural network that returns the Lagrangian of the
            system.
2. `p`: The initial parameters of the neural network.
References:
[1] Cranmer, Miles, Samuel Greydanus, Stephan Hoyer, Peter Battaglia, David Spergel, Shirley Ho. 
"Lagrangian Neural Networks."  ICLR 2020 Deep Differential Equations Workshop. 2020.
"""
module LagrangianNeuralNetwork
using Flux, DiffEqFlux, DifferentialEquations
using DiffEqSensitivity, LinearAlgebra, ForwardDiff
struct LagrangianNN{M, R, P}
    model::M
    re::R
    p::P

    function LagrangianNN(model; p = nothing)
        _p, re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        return new{typeof(model), typeof(re), typeof(p)}(model, re, p)
    end

    function LagrangianNN(model::DiffEqFlux.FastChain; p = initial_params(model))
        re = nothing
        return new{typeof(model), typeof(re), typeof(p)}(model, re, p)
    end
end

Flux.trainable(lnn::LagrangianNN) = (lnn.p,)

function _lagrangian_forward(re, p, x)
    n = size(x, 1) ÷ 2
    gradient = Flux.gradient(x -> sum(re(p)(x)), x)[1]
    hessian = Flux.hessian(x -> sum(re(p)(x)), x)
    inverse = inv(hessian[(n+1):end, (n+1):end])
    offdiagonal = hessian[1:n, 1:n]
    return inverse .* ( gradient[1:n] .- offdiagonal .* x[(n+1):end] )
end

function _lagrangian_forward(m::DiffEqFlux.FastChain, p, x)
    n = size(x, 1) ÷ 2
    grad = ForwardDiff.gradient(x -> sum(m(x, p)), x)
    #println("grad: ", grad)
    hessian = ForwardDiff.hessian(x -> sum(m(x, p)), x)
    # println("hessian: ", hessian[(n+1):end, (n+1):end])
    result = nothing
    if det(hessian) == 0.0
        println("Not invertible!")
        result = zeros(n)
    else
        inverse = inv(hessian[(n+1):end, (n+1):end])
        #println("inverse: ", inverse)
        offdiagonal = hessian[1:n, (n+1):end]
        result = inverse .* ( grad[1:n] .- offdiagonal .* x[(n+1):end] )
    end
    return result
end

(lnn::LagrangianNN)(x, p = lnn.p) = _lagrangian_forward(lnn.re, p, x)

(lnn::LagrangianNN{M})(x, p = lnn.p) where {M<:DiffEqFlux.FastChain} =
    _lagrangian_forward(lnn.model, p, x)


"""
Contructs a Neural Lagrangian DE Layer for solving Lagrangian Problems
parameterized by a Neural Network [`LagrangianNN`](@ref).
```julia
NeuralLagrangianDE(model, tspan, args...; kwargs...)
```
Arguments:
- `model`: A Chain, FastChain or Lagrangian Neural Network that predicts the
           Lagrangian of the system.
- `tspan`: The timespan to be solved on.
- `kwargs`: Additional arguments splatted to the ODE solver. See the
            [Common Solver Arguments](https://diffeq.sciml.ai/dev/basics/common_solver_opts/)
            documentation for more details.
"""
struct NeuralLagrangianDE{M,P,RE,T,A,K} <: DiffEqFlux.NeuralDELayer
    lnn::LagrangianNN{M,RE,P}
    p::P
    tspan::T
    args::A
    kwargs::K

    function NeuralLagrangianDE(model, tspan, args...; p = nothing, kwargs...)
        lnn = LagrangianNN(model, p=p)
        new{typeof(lnn.model), typeof(lnn.p), typeof(lnn.re),
            typeof(tspan), typeof(args), typeof(kwargs)}(
            lnn, lnn.p, tspan, args, kwargs)
    end

    function NeuralLagrangianDE(lnn::LagrangianNN{M,RE,P}, tspan, args...;
                                 p = lnn.p, kwargs...) where {M,RE,P}
        new{M, P, RE, typeof(tspan), typeof(args),
            typeof(kwargs)}(lnn, lnn.p, tspan, args, kwargs)
    end
end

function (nlde::NeuralLagrangianDE)(x, p = nlde.p)
    function neural_lagrangian!(du, u, p, t)
        n = size(du, 1) ÷ 2
        x = u[1:n]
        dx = u[(n+1):end]

        du[1:n] = dx
        du[(n+1):end] = nlde.lnn(u, p)
        # du .= reshape(nlde.lnn(u, p), size(du))
    end
    prob = DifferentialEquations.ODEProblem(neural_lagrangian!, x, nlde.tspan, p)
    # NOTE: Nesting Zygote is an issue. So we can't use ZygoteVJP
    sense = DiffEqSensitivity.InterpolatingAdjoint(autojacvec = false)
    solve(prob, nlde.args...; sensealg = sense, nlde.kwargs...)
end
end
