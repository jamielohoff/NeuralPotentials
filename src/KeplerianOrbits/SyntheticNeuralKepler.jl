using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, Statistics, LinearAlgebra, LaTeXStrings, DataFrames, Distributions, Measures
include("../lib/SagittariusData.jl")
include("../lib/MechanicsDatasets.jl")
include("../lib/AwesomeTheme.jl")
include("../lib/Qtils.jl")
using .SagittariusData
using .MechanicsDatasets
using .Qtils

# Load predefined plot theme and adjust font sizes
theme(:awesome)
resetfontsizes()
scalefontsizes(2)

# Natural constant
const c = 306.4 # mpc/yr
const G = 4.49 # gravitational constant in new units : (milli parsec)^3 * yr^-2 * (10^6*M_solar)^-1

### Initialisation of the Sagittarius data ------------------------ #
ϕ0span = (0.01, 4π-0.01)
ϕ0 = Array(range(ϕ0span[1], ϕ0span[2], length=288))
r0 = 0.5 # distance in mpc from the periapsis
M = 4.35 # mass of the central SMBH
true_v0 = 1.2*sqrt(G*M/r0) # velocity in mpc/yr at the periapsis
true_u0 = [1.0/r0, 0.0, 0.0] 
true_p = [M, 1.0/(true_v0*r0)]
println(true_p)
# Definition of the potential and its gradient
V0(U,p) = G*p[1]*[p[2]^2*U - U^3/c^2]
dV0(U,p) = Zygote.gradient(x -> V0(x,p)[1], U)[1]
data = MechanicsDatasets.keplerproblem(dV0, true_u0, true_p, ϕ0, addnoise=false)

# Definition of the radial basisfunction
rbf(x) = sqrt(1.0 + 0.25*x^2)

# Definition of the gradient of the potential as a neural network and initializing parameters
dV = FastChain(
    FastDense(1, 8, celu),
    FastDense(8, 8, rbf),
    FastDense(8, 1)
)
ps = vcat(1.0 .+ rand(Float64, 1), rand(Float64, 1), 0.1 .+ 0.4*rand(Float64, 1), 2.0*rand(Float64, 2), initial_params(dV))
println(180.0*ps[2:4])

angles = [50.0, 100.0, 65.0].*π/180 # [132.0, 226.0, 65.0].*π/180
println(angles)
r = 0
ϕ = 0
# Implement the rotation
if angles[1] > π/2
    angles[1] = angles[1] - π/2
    r, ϕ = SagittariusData.transform(angles, data.r, ϕ0, false)
else
    r, ϕ = SagittariusData.transform(angles, data.r, ϕ0, true)
end

orbit = hcat(r, ϕ, data.t, data.x_err, data.y_err)
star = DataFrame(orbit, [:r, :ϕ, :t, :x_err, :y_err])
star = sort!(star, [:t])

# Check if the orbit is prograde or retrograde
prograde = SagittariusData.isprograde(star.ϕ)
println("isprograde: ", prograde)
phase = 0
if !prograde
    phase = 0.5
end

# Define ODE problem for our system
function neuralkepler!(du, u, p, ϕ)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = -G*dV(U, p)[1] - U
end

# Defining the problem and optimizer
ϕspan = (0.0, 10π)
problem = ODEProblem(neuralkepler!, ps[1:2], ϕspan, ps[6:end]) 
opt = ADAM(0.01)

# Function that predicts the results for a given set of parameters by solving the ODE at the given angles ϕ
function predict(params)
    # Rotate the trajectory in the observational plane
    s, θ = SagittariusData.inversetransform(params[3:5].*π, star.r, star.ϕ, prograde)
    # Solve the ODE in the observational plane using the θ's
    pred = Array(solve(problem, Tsit5(), u0=params[1:2], p=params[6:end], saveat=θ))
    # Rotate the trajectory back into the plane of motion and project it onto the observational plane
    r, ϕ = SagittariusData.transform(params[3:5].*π, 1.0./pred[1,:], θ, prograde)
    return vcat(reshape(r,1,:), reshape(ϕ,1,:))
end

# Function that calculates the loss with respect to the synthetic data
function loss(params) 
    pred = predict(vcat(params[1:4], Zygote.hook(x -> 1e12*x, params[5]), params[6:end])) # gradient is boosted by facto 1e12
    return sum(abs2, pred[1,:].-star.r), pred
end

# Callback function 
epoch = 0
cb = function(p,l,pred)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    println("Initial radius and velocity: ", p[1:2])
    println("Rotation angles: ", 180.0 * vcat(mod.(p[3], 0.5) + phase, mod.(p[4:5], 2)))
    println("Initial values: ", 180.0*ps[3:5])

    if epoch % 1 == 0
        # Plotting the prediction and ground truth of the orbit
        orbit_plot = scatter(star.r .* cos.(star.ϕ), star.r .* sin.(star.ϕ), label="Synthetic data")
        orbit_plot = plot!(orbit_plot, cos.(pred[2,:]) .* pred[1,:], sin.(pred[2,:]) .* pred[1,:],
                            label="Prediction using neural potential",
                            xlabel=L"x \textrm{ coordinate [mpc]}",
                            ylabel=L"y \textrm{ coordinate [mpc]}",
                            title="Position of the Star S2 and Gravitational Potential"
        )

        # Calculating the potential and plotting it with the true potential
        u0 = range(0.1, 2.1, step=0.01)
        true_potential = [V0(u, true_p)[1] for u ∈ u0]
        potential = -G.*[Qtils.integrateNN(dV, p[6:end], 0.0, u)[1] for u ∈ u0]

        pot_plot = plot(u0, true_potential, label="Potential used for data generation")
        pot_plot = plot!(pot_plot, u0, potential, 
                            label="Prediction of the neural network",
                            xlabel=L"u \textrm{ coordinate [mpc]}",
                            ylabel=L"\textrm{Potential } \frac{\mu}{L_z^2}V(1/u)"
        )

        result_plot = plot(orbit_plot, pot_plot, layout=(2,1), size=(1200, 1200), legend=:bottomright)
        display(plot(result_plot))
    end
    global epoch+=1
    return l < 0.025
    
end

# Start the training of the model
@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=5000)

# Preparing and plotting the result
s, θ = SagittariusData.inversetransform(result.minimizer[3:5].*π, star.r, star.ϕ, prograde)
res = solve(problem, Tsit5(), u0=result.minimizer[1:2], p=result.minimizer[6:end], saveat=θ)
r, ϕ = SagittariusData.transform(result.minimizer[3:5].*π, 1.0./res[1,:], θ, prograde)

# Plotting the best fit trjectory and ground truth of the orbit
orbit_plot = scatter(star.r .* cos.(star.ϕ), star.r .* sin.(star.ϕ), label="Synthetic data")
orbit_plot = plot!(orbit_plot, cos.(ϕ) .* r, sin.(ϕ) .* r,
                    label="Prediction of the trajectory",
                    xlabel=L"x \textrm{ coordinate [mpc]}",
                    ylabel=L"y \textrm{ coordinate [mpc]}",
                    title="Trajectory of a Star"
)

# Calculating the best-fit potential and plotting it with the true potential
u0 = range(0.1, 2.1, step=0.01)
true_potential = [V0(u, true_p)[1] for u ∈ u0]
potential = -G.*[Qtils.integrateNN(dV, result.minimizer[6:end], 0.0, u)[1] for u ∈ u0]

pot_plot = plot(u0, true_potential, label="Potential used for data generation")
pot_plot = plot!(pot_plot, u0, potential, 
                    label="Prediction of the neural network",
                    xlabel=L"u \textrm{ coordinate [} \textrm{mpc}^{-1} \textrm{]}",
                    ylabel=L"\textrm{Potential } \frac{\mu}{L_z^2}V(1/u)"
)

result_plot = plot(orbit_plot, pot_plot, layout=(2,1), size=(1200, 1200), legend=:bottomright)

# Save the figure
savefig(result_plot, "SyntheticNeuralFit.pdf")

