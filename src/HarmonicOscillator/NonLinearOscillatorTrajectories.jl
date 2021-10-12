using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, LinearAlgebra, Statistics, DataFrames, LaTeXStrings, Measures
include("../Qtils.jl")
include("../MechanicsDatasets.jl")
include("../AwesomeTheme.jl")
using .Qtils
using .MechanicsDatasets

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

### Creation of synthethetic data---------------------- #
tspan = (0.0, 5.0)
t = Array(range(tspan[1], tspan[2], length=128))
true_u0 = [3.1, 0.0] # contains x0 and dx0
true_p = [1.0]
stderr = 0.01

V0(x,p) = 0.25*x^4 - 2.0*x^2 + 4.0 # 4.5*(1 - cos(x)) # p[1]*x^2 # 
dV0(x,p) = Flux.gradient(x -> V0(x,p), x)[1]

samplesize = 1
initialConditions = hcat(π, zeros(samplesize,1)) # .*rand(samplesize,1).-π
parameters = hcat(rand(Float32, (samplesize, 1)), rand(Float32, (samplesize, 1)))
@time trajectories = MechanicsDatasets.sampletrajectories(V0, parameters, initialConditions, t, addnoise=true, σ=stderr)

### End ------------------------------------------------ #

# Defining the gradient of the potential
dV = FastChain(
    FastDense(1, 16, relu),
    FastDense(16, 16, relu), 
    FastDense(16, 1)
)

ps = vcat(initial_params(dV))

function neuraloscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -dV(x,p)[1]
    #du[1] = dx
    #du[2] = -dV(x,p)[1] 
end

prob = ODEProblem(neuraloscillator!, true_u0, tspan, true_p)

function predict(u0, params)
    return Array(solve(prob, Tsit5(), u0=u0, p=Zygote.@showgrad(params), saveat=t))
end

function loss(gt, params)
    u0 = [gt[2,1], gt[3,1]]
    pred = predict(u0, params)
    return sum(abs2, (pred[1,:] .- gt[2,:])), pred # /(size(gt, 2) - size(params, 1)), pred
end

epoch = 0

cb = function(p,l,pred,gt)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    if mod(epoch,1) == 0
        # Plotting fitting result
        traj_plot = scatter(gt[1,:], gt[2,:], 
                    label="Oscillator data",
                    markershape=:cross,
        )
        traj_plot = plot!(traj_plot, t, pred[1, :],
                    title="Trajectory",
                    label="Prediction using neural potential",
                    xlabel=L"\textrm{Time } t",
                    ylabel=L"\textrm{Displacement } x(t)",
                    ylim=(-7.5,5.0),
                    legend=:bottomright,
        )

        # Plotting the potential
        x0 = Array(range(-true_u0[1], true_u0[1], step=0.01))
        predicted_potential = [Qtils.integrateNN(dV, p, 0.0, x) for x ∈ x0]
        # predicted_potential = [dV(x, p)[1] for x ∈ x0] 
        true_potential = [V0(x,true_p) for x ∈ x0]

        pot_plot = plot(x0, true_potential, 
                    label="Potential used for data generation",
        )
        pot_plot = plot!(pot_plot, x0, predicted_potential,
                    title="Potential",
                    label="Prediction of the potential",
                    xlabel=L"\textrm{Displacement } x",
                    ylabel=L"\textrm{Potential } V(x)",
                    xlims=(-true_u0[1],true_u0[1]),
                    legend=:bottomright,
        )
        
        display(plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 1200)))
    end
    global epoch+=1
    return l < 0.2
end

# Now we tell Flux how to train the neural network
dataloader = Flux.Data.DataLoader(trajectories; batchsize=8, shuffle=true)
opt = NADAM(0.01)
epochs = 10000
for epoch in 1:epochs
    println("epoch: ", epoch, " of ", epochs)
    for batch in dataloader
        for gt in batch
            _loss, _pred = loss(gt, ps)
            gs = Flux.gradient(p -> loss(gt, p)[1], ps)[1]
            # if _loss < 400.0
            #     global opt = Nesterov(1e-5)
            # end
            Flux.update!(opt, ps, gs)
            breakCondition = cb(ps, _loss, _pred, gt)
            if breakCondition
                break
            end
        end
    end
end


