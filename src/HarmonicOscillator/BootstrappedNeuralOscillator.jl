using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, LinearAlgebra, Statistics, DataFrames, LaTeXStrings, Measures, CSV
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
tspan = (0.0, 7.5)
t0 = Array(range(tspan[1], tspan[2], length=128))
true_u0 = [3.0, 0.0]
true_p = [2.0]
stderr = 0.1

# Define potential and get dataset
V0(x,p) = 0.5*p[1]*x^2 
dV0(x,p) = Flux.gradient(x -> V0(x,p), x)[1]
data = MechanicsDatasets.potentialproblem1D(dV0, true_u0, true_p, t0, addnoise=true, σ=stderr)

x0 = Array(range(-true_u0[1], true_u0[1], step=0.05))
true_potential = [V0(x, true_p) for x ∈ x0]

### Bootstrap Loop ----------------------------------------------------- #
repetitions = 1024
itmlist = DataFrame(params = Array[], trajectory = Array[], potential = Array[],)
println("Beginning Bootstrap...")
lk = ReentrantLock()
@time Threads.@threads for rep in 1:repetitions
    println("Starting repetition ", rep, " at thread ", Threads.threadid())

    # Defining the gradient of the potential
    dV = FastChain(
        FastDense(1, 8, relu), 
        FastDense(8, 8, relu),
        FastDense(8, 1)
    )

    # Initialize the parameters as well as the weights and biases of the neural network
    ps = vcat(rand(Float32, 2) .+ [2.5, 0.0], initial_params(dV))

    # Define ODE problem for our system
    function neuraloscillator!(du, u, p, t)
        x = u[1]
        dx = u[2]

        du[1] = dx
        du[2] = -dV(x, p)[1]
    end

    # Defining the problem and optimizer
    prob = ODEProblem(neuraloscillator!, ps[1:2], tspan, ps[3:end])
    opt = ADAM(0.2)

    # Function that predicts the results for a given set of parameters by solving the ODE at the timesteps
    function predict(params)
        return Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=t0))
    end
    
    # Function that calculates the loss with respect to the synthetic data
    function loss(params)
        pred = predict(params)
        return sum(abs2, (pred[1:2,:] .- data[2:3,:])./stderr), pred
    end
    
    # Callback function 
    cb = function(p,l,pred)
        return false
    end

    # Start the training of the model
    @time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=10000)

    # Use the best result, i.e. the one with the lowest loss and compute the potential etc. for it
    res = Array(solve(prob, Tsit5(), u0=result.minimizer[1:2], p=result.minimizer[3:end], saveat=t0))
    potential = [Qtils.integrateNN(dV, result.minimizer[3:end], 0.0, x) for x ∈ x0]

    # Push the result into the array
    lock(lk)
    push!(itmlist, [result.minimizer[1:2], res[1,:], potential])
    unlock(lk)
    println("Repetition ", rep, " is done!")
end
println("Bootstrap complete!")

# Calculate the mean, standard deviation and 95% confidence intervals for the quantities of interest
mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_traj, std_traj, CI_traj = Qtils.calculatestatistics(itmlist.trajectory)
mean_pot, std_pot, CI_pot = Qtils.calculatestatistics(itmlist.potential)

println("Initial conditions: ")
println("x(0) = ", mean_params[1], " ± ", std_params[1])
println("v(0) = ", mean_params[2]," ± ", std_params[2])

# Plot all the stuff into a 2x1 figure
traj_plot = scatter(data[1,:], data[2,:],
            title="Trajectory",
            label="Harmonic oscillator data",
            xlabel=L"\textrm{Time } t",
            ylabel=L"\textrm{Displacement } x(t)",
            ylim=(-5.0,5.0),
            legend=:bottomright,
            markershape=:cross,
)
traj_plot = plot!(traj_plot, t0, mean_traj, 
            ribbon=CI_traj,
            label="Prediction using neural potential",
)

pot_plot = plot(x0, true_potential,
            title="Potential",
            label="Potential used for data generation",
            xlabel=L"\textrm{Displacement } x",
            ylabel=L"\textrm{Potential } V(x)",
            xlims=(-true_u0[1],true_u0[1]),
            ylim=(-5.0,10.0),
            legend=:bottomright,
)
pot_plot = plot!(pot_plot, x0, mean_pot, 
            ribbon=CI_pot,
            label="Prediction of the potential",

)
resultplot = plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 1200))

# Save the figure
savefig(resultplot, "256_sampleSimpleHarmonicOscillator.pdf")

