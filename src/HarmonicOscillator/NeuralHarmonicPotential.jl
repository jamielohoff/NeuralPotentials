using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, LinearAlgebra, Statistics
include("../MyUtils.jl")
using .MyUtils

### Creation of synthethetic data---------------------- #
omega = 1.0f0
beta = 0.0f0
tspan = (0.0f0, 5.0f0)
t = Array(range(tspan[1], tspan[2], length=256))
u0 = [3.0f0, 0.0f0] # contains x0 and dx0
p = [omega, beta]

V0(x,p) = p[1]*x^2 # 
dV0(x,p) = Flux.gradient(x -> V0(x,p)[1], x)[1]

function oscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -dV0(x, p)
end
problem = ODEProblem(oscillator!, u0, tspan, p)
@time true_sol = solve(problem, Tsit5(), saveat=t)
noise =  0.2f0 * (2 * rand(Float32, size(true_sol)) .- 1.0f0)

data = true_sol[1:2,:] .+ noise
### End ------------------------------------------------ #

# Defining the gradient of the potential
dV = FastChain(
    FastDense(1, 16, relu), 
    FastDense(16, 1)
)

otherparams = rand(Float32, 2) .+ [1.5f0, 0.0f0] # contains u0, du0 and beta
params = vcat(otherparams, initial_params(dV))

function neuraloscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -dV(x,p)[1] # - p[1] * dx 
end

prob = ODEProblem(neuraloscillator!, u0, tspan, params[3:end])

function predict(params)
    return Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=t))
end

function loss(params)
    pred = predict(params)
    return mean((pred .- data).^2), pred
end

anim = Animation()

cb = function(p,l,pred)
    display(l)
    display(p[1:2])
    traj_plot = plot(t, pred[1, :], ylim=(-4,4))
    traj_plot = scatter!(traj_plot, t, data[1,:])

    # Plotting the potential
    x0 = Array(range(-u0[1], u0[1], step=0.01))
    y0 = map(x -> MyUtils.integrateNN(dV, x, p[3:end]), x0)
    z0 = map(x -> V0(x,[omega,beta]), x0)
    pot_plot = plot(x0, y0)
    pot_plot = plot!(pot_plot, x0, z0, ylims=(-0.25,3.5), xlims=(-u0[1],u0[1]))
    resultplot = plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 800))
    display(resultplot)
    frame(anim)
    return l < 0.0145
end

opt = ADAM(0.1)

@time result = DiffEqFlux.sciml_train(loss, params, opt, cb=cb, maxiters=2000)

gif(anim, string(@__DIR__) * "osci.gif", fps=60)

