using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, LinearAlgebra, Statistics, LaTeXStrings, Measures
include("../lib/Qtils.jl")
include("../lib/MechanicsDatasets.jl")
include("../lib/AwesomeTheme.jl")
using .Qtils
using .MechanicsDatasets

# Load predefined plot theme and adjust font sizes
theme(:awesome)
resetfontsizes()
scalefontsizes(2)

### Creation of synthethetic data---------------------- #
tspan = (0.0, 5.0)
t0 = Array(range(tspan[1], tspan[2], length=256))
true_u0 = [3.0, 0.0] # contains x0 and dx0
true_p = [2.0]
stderr = 0.1

# Define potential and get dataset
V0(x,p) = 0.5*p[1]*x^2 
data = MechanicsDatasets.potentialproblem1D(V0, true_u0, true_p, t0, addnoise=true, σ=stderr)
### End ------------------------------------------------ #

# Defining the gradient of the potential
V = FastChain(
    FastDense(1, 8, relu), 
    FastDense(8, 8, relu),
    FastDense(8, 1)
)

# Initialize the parameters as well as the weights and biases of the neural network
ps = vcat(rand(Float32, 2) .+ [2.5, -0.5], initial_params(V))

dV(x, p) = Zygote.gradient(x -> V(x, p)[1], x)[1]

# Define ODE problem for our system
function neuraloscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -dV([x],p)[1] 
end

# Defining the problem and optimizer
prob = ODEProblem(neuraloscillator!, ps[1:2], tspan, ps[3:end])
opt = ADAM(0.1)

# Function that predicts the results for a given set of parameters by solving the ODE at the time-steps
function predict(params)
    return Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=t0))
end

# Function that calculates the loss with respect to the synthetic data
function loss(params)
    pred = predict(params)
    return sum(abs2, pred[1,:] .- data[2,:]), pred
end

epoch = 0
# Callback function that shows the loss for every iteration
cb = function(p,l,pred)
    println("Loss: ", l, " at epoch ", epoch)
    println("Parameters: ", p[1:2])
    global epoch += 1
    return false
end

# Now we tell Flux how to train the neural network
@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=1000)

res = Array(solve(prob, Tsit5(), u0=result.minimizer[1:2], p=result.minimizer[3:end], saveat=t0))

# Plotting fitting result
traj_plot = scatter(t0, data[2,:], 
            label="Harmonic oscillator data",
            markershape=:cross,
)
traj_plot = plot!(traj_plot, t0, res[1, :],
            title="Trajectory",
            label="Prediction using neural potential",
            xlabel=L"\textrm{Time } t",
            ylabel=L"\textrm{Displacement } x(t)",
            ylim=(-7.5,5.0),
            legend=:bottomright,
            linewidth=3
)


# Plotting the potential
x0 = Array(range(-true_u0[1], true_u0[1], step=0.01))
predicted_potential = [V(x, result.minimizer[3:end])[1] for x ∈ x0] # [Qtils.integrateNN(dV, result.minimizer[3:end], 0.0, x) for x ∈ x0]
true_potential = [V0(x, true_p) for x ∈ x0]

pot_plot = plot(x0, true_potential, 
            label="Potential used for data generation",
)
pot_plot = plot!(pot_plot, x0, predicted_potential,
            title="Potential",
            label="Prediction of the potential",
            xlabel=L"\textrm{Displacement } x",
            ylabel=L"\textrm{Potential } V(x)",
            xlims=(-true_u0[1],true_u0[1]),
            ylims=(-5.0, 10.0),
            legend=:bottomright,
)

resultplot = plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 1200))

# Save the figure
# savefig(resultplot, "harmonicoscillator_positiononly.pdf")

