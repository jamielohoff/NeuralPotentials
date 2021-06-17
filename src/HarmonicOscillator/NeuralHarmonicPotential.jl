using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, LinearAlgebra, Statistics
include("../Qtils.jl")
include("../MechanicsDatasets.jl")
using .Qtils
using .MechanicsDatasets

### Creation of synthethetic data---------------------- #
tspan = (0.0, 5.0)
t0 = Array(range(tspan[1], tspan[2], length=256))
u0 = [3.0, 0.0] # contains x0 and dx0
p = [1.0]

# Define potential and get dataset
V0(x,p) = p[1]*x^2 
data = MechanicsDatasets.potentialproblem(V0, u0, p, t0, addnoise=true, σ=0.2)
### End ------------------------------------------------ #

# Defining the gradient of the potential
dV = FastChain(
    FastDense(1, 8, relu), 
    FastDense(8, 1)
)

otherparams = rand(Float32, 2) .+ [1.5, 0.0] # contains u0, du0 and beta
ps = vcat(otherparams, initial_params(dV))

function neuraloscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -dV(x,p)[1] 
end

prob = ODEProblem(neuraloscillator!, u0, tspan, ps[3:end])

function predict(params)
    return Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=t0))
end

function loss(params)
    pred = predict(params)
    return sum(abs2, (pred .- data[2:3,:])./0.2) / (size(data, 2) - size(params, 1)), pred
end

cb = function(p,l,pred)
    println("Loss: ", l)
    println("Parameters: ", p[1:2])
    return l < 0.05
end

opt = ADAM(0.1)

@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=500)

res = Array(solve(prob, Tsit5(), u0=result.minimizer[1:2], p=result.minimizer[3:end], saveat=t0))

traj_plot = plot(t0, res[1, :], ylim=(-4,4))
traj_plot = scatter!(traj_plot, t0, data[2,:])

# Plotting the potential
x0 = Array(range(-u0[1], u0[1], step=0.01))
predicted_potential = map(x -> Qtils.integrateNN(dV, x, result.minimizer[3:end]), x0)
true_potential = map(x -> V0(x, p), x0)
pot_plot = plot(x0, predicted_potential)
pot_plot = plot!(pot_plot, x0,true_potential, ylims=(-0.25,3.5), xlims=(-u0[1],u0[1]))
resultplot = plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 800))

