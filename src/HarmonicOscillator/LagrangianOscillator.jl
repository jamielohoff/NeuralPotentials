using DifferentialEquations, Flux, DiffEqFlux, ReverseDiff, ForwardDiff
using Plots, LinearAlgebra, Statistics
include("../MyUtils.jl")
using .MyUtils

### Creation of synthethetic data---------------------- #
const omega = 1.0f0
const m = 1.0f0
tspan = (0.0f0, 10.0f0)
t = Array(range(tspan[1], tspan[2], length=100))
u0 = [1.0f0, 0.0f0] # contains x0 and dx0
true_p = [omega, m]

L0(x,p) = 0.5f0*(-(p[1]*x[1])^2 + x[2]^2/p[2])

function _lagrangian_forward(model, x, p) # ::DiffEqFlux.FastChain
    n = size(x, 1) ÷ 2
    @showgrad grad = ForwardDiff.gradient(x -> sum(model(x, p)), x)
    hessian = ForwardDiff.hessian(x -> sum(model(x, p)), x)
    result = nothing
    if det(hessian) == 0.0
        result = [1.0 0.0; 0.0 1.0]
    else
        inverse = inv(hessian[(n+1):end, (n+1):end])
        offdiagonal = hessian[1:n, (n+1):end]
        result = inverse .* ( grad[1:n] .- offdiagonal .* x[(n+1):end] )
    end
    return result
end

function oscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = _lagrangian_forward(L0, [x,dx], p)[1]
end
problem = ODEProblem(oscillator!, u0, tspan, true_p)
@time sol = solve(problem, Tsit5(), saveat=t)

## End ------------------------------------------------ #

# Defining the lagrange function
L = FastChain(
    FastDense(2, 16, relu), 
    FastDense(16, 1)
)

ps = vcat(initial_params(L))

function neuraloscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = _lagrangian_forward(L, [x,dx], p)[1]
end

prob = ODEProblem(neuraloscillator!, u0, tspan, ps)

function predict(params)
    return Array(solve(prob, Tsit5(), p=params, saveat=t))
end

function loss(params)
    pred = predict(params)
    return mean((pred .- sol[1:2,:]).^2), pred
end

cb = function(p,l,pred)
    display(l)
    traj_plot = plot(t, pred[1, :], ylim=(-4,4))
    traj_plot = scatter!(traj_plot, t, sol[1,:])

    # Plotting the potential
    # x0 = Array(range(-u0[1], u0[1], step=0.01))
    # y0 = map(x -> L(x, p)[1], x0)
    # z0 = map(x -> L0(x, true_p), x0)
    # pot_plot = plot(x0, y0)
    # pot_plot = plot!(pot_plot, x0, z0)

    display(plot(traj_plot, size=(1200, 800)))
    return l < 0.001
end
# @time result = DiffEqFlux.sciml_train(loss, params, opt, cb=cb, maxiters=2000)

opt = ADAM(1e-2)
epochs = 10
for epoch in 1:epochs
    println("epoch: ", epoch, " of ", epochs)
    _loss, _pred = loss(ps)
    gs = ForwardDiff.gradient(p -> loss(p)[1], ps)
    println(gs)
    Flux.update!(opt, ps, gs)
    breakCondition = cb(ps, _loss, _pred)
    if breakCondition
        break
    end
end

