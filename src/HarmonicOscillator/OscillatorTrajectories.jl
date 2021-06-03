using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, LinearAlgebra, Statistics, ProgressBars, Printf
include("../Qtils.jl")
using .Qtils

### Creation of synthethetic data---------------------- #
const omega = 1.0f0
const beta = 0.1f0
tspan = (0.0f0, 2.0f0)
t = Array(range(tspan[1], tspan[2], length=256))
true_u0 = [3.0f0, 0.0f0] # contains x0 and dx0
true_p = [omega, beta]

V0(x,p) = p[1]*x^4 # 1 - cos(x) # p[1]*x^2 # 
dV0(x,p) = Flux.gradient(x -> V0(x,p)[1], x)[1]

function oscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -dV0(x, p)
end

initialConditions = hcat(2pi .* (rand(Float32, (64, 1)) .- 0.5f0), zeros(64))
@time trajectories = Qtils.sampletrajectories(oscillator!, true_p, initialConditions, t)

### End ------------------------------------------------ #

# Defining the gradient of the potential
dV = FastChain(
    FastDense(1, 16, relu), 
    FastDense(16, 1)
)

ps = vcat(initial_params(dV))

function neuraloscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -dV(x,p)[1] 
end

prob = ODEProblem(neuraloscillator!, true_u0, tspan, params)

function predict(u0, params)
    return Array(solve(prob, Tsit5(), u0=u0, p=params, saveat=t))
end

function loss(groundtruth, params)
    u0 = [groundtruth[1,1], groundtruth[2,1]]
    pred = predict(u0, params)
    return mean((pred .- groundtruth).^2), pred
end

cb = function(p,l,pred,gt)
    display(l)
    traj_plot = plot(t, pred[1, :], ylim=(-4,4))
    traj_plot = scatter!(traj_plot, t, gt[1,:])

    # Plotting the potential
    x0 = Array(range(-3.25, 3.25, step=0.01))
    y0 = map(x -> MyUtils.integrateNN(dV, x, p), x0)
    z0 = map(x -> V0(x,[omega,beta]), x0)
    pot_plot = plot(x0, y0)
    pot_plot = plot!(pot_plot, x0, z0, ylims=(-0.25,3.5), xlims=(-3.25,3.25))

    display(plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 800)))
    return l < 0.01
end

# Now we tell Flux how to train the neural network
dataloader = Flux.Data.DataLoader(trajectories; batchsize=1, shuffle=true)
opt = ADAM(0.01, (0.85, 0.9))
epochs = 1000
for epoch in 1:epochs
    println("epoch: ", epoch, " of ", epochs)
    for batch in dataloader
        _loss, _pred = loss(batch[1], ps)
        gs = Flux.gradient(p -> loss(batch[1], p)[1], ps)[1]
        Flux.update!(opt, ps, gs)
        breakCondition = cb(ps, _loss, _pred, batch[1])
        if breakCondition
            break
        end
    end
end


