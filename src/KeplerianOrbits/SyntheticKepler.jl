using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, Statistics, LinearAlgebra, LaTeXStrings, DataFrames, Distributions, Measures
include("../lib/SagittariusData.jl")
include("../lib/MechanicsDatasets.jl")
include("../lib/AwesomeTheme.jl")
using .SagittariusData
using .MechanicsDatasets

# Load predefined plot theme and adjust font sizes
theme(:awesome)
resetfontsizes()
scalefontsizes(2)

# Natural constant
const c = 306.4 # mpc per yr
const G = 4.49 # gravitational constant in new units : (milli parsec)^3 * yr^-2 * (10^6*M_solar)^-1

### Initialisation of the Sagittarius data ------------------------ #
ϕ0span = (0.01, 4π-0.01)
ϕ0 = Array(range(ϕ0span[1], ϕ0span[2], length=288))
r0 = 0.5 # distance in mpc from the periapsis
M = 4.35 # mass of the central SMBH
true_v0 = 1.35*sqrt(G*M/r0) # velocity in mpc/yr at the periapsis
true_u0 = [1.0/r0, 0.0, 0.0] 
true_p = [M, 1.0/(true_v0*r0)]
println(true_p)
# Definition of the potential and its gradient and initializing parameters
dV(U,p) = G*p[1]*[p[2]^2 - 3.0*U^2/c^2]
ps = vcat(rand(Float64, 2), 0.1 .+ 0.5*rand(Float64, 1), 2.0*rand(Float64, 2), 14.0 .+ rand(Float64,1), rand(Float64,1))
println(180.0*ps[3:5])

data = MechanicsDatasets.keplerproblem(dV, true_u0, true_p, ϕ0)

# Define the angles which we use to rotate the traject
angles = [50.0, 100.0, 65.0].*π/180 # 
println(angles)
r = 0
ϕ = 0
x_err = ones(size(data.r))
y_err = ones(size(data.r))
# Implement the rotation
if angles[1] > π/2
    angles[1] = angles[1] - π/2
    r, ϕ = SagittariusData.transform(angles, data.r, ϕ0, false)
else
    r, ϕ = SagittariusData.transform(angles, data.r, ϕ0, true)
end

orbit = hcat(r, ϕ, data.t, x_err, y_err)
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
    du[2] = dV(U, p)[1] - U
end

# Defining the problem and optimizer
ϕspan = (0.0, 12π)
problem = ODEProblem(neuralkepler!, ps[1:2], ϕspan, ps[6:end]) 
opt = NADAM(0.001)

function predict(params)
    # Rotate the trajectory in the observational plane
    s, θ = SagittariusData.inversetransform(params[3:5].*π, star.r, star.ϕ, prograde)
    # Solve the ODE in the observational plane using the θ's
    pred = Array(solve(problem, Tsit5(), u0=params[1:2], p=params[6:end], saveat=θ))
    # Rotate the trajectory back into the plane of motion and project it onto the observational plane
    r, ϕ = SagittariusData.transform(params[3:5].*π, 1.0./pred[1,:], θ, prograde)
    return vcat(reshape(r,1,:), reshape(ϕ,1,:), reshape(s,1,:), reshape(θ,1,:))
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
    println("Initial velocity: ", p[1:2])
    println("Rotation angles: ", 180.0 * vcat(mod.(p[3], 0.5) + phase, mod.(p[4:5], 2)))
    println("Initial values: ", 180.0*ps[2:4])
    println("Parameters of the potential: ", p[6:end])

    if epoch % 10 == 0
        orbit_plot = scatter(star.r .* cos.(star.ϕ), star.r .* sin.(star.ϕ), label="Synthetic data")
        orbit_plot = plot!(orbit_plot, cos.(pred[2,:]) .* pred[1,:], sin.(pred[2,:]) .* pred[1,:],
                            label="Prediction of the trajectory",
                            xlabel=L"x \textrm{ coordinate [mpc]}",
                            ylabel=L"y \textrm{ coordinate [mpc]}",
                            title="Trajectory of a Star"
        )
        
        result_plot = plot(orbit_plot, layout=(1,1), size=(1200, 1200), legend=:topright)
        display(plot(result_plot))
    end
    global epoch+=1
    return l < 2e-4
    
end

# Start the training of the model
@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=50000)

# Preparing and plotting the result
s, θ = SagittariusData.inversetransform(result.minimizer[3:5].*π, star.r, star.ϕ, prograde)
θrange = Array(range(minimum(θ), maximum(θ), length=500))
res = solve(problem, Tsit5(), u0=result.minimizer[1:2], p=result.minimizer[6:end], saveat=θrange)
r, ϕ = SagittariusData.transform(result.minimizer[3:5].*π, 1.0./res[1,:], θrange, prograde)

orbit_plot = scatter(star.r .* cos.(star.ϕ), star.r .* sin.(star.ϕ), label="Synthetic data")
orbit_plot = plot!(orbit_plot, cos.(ϕ) .* r, sin.(ϕ) .* r,
                    label="Prediction of the trajectory",
                    xlabel=L"x \textrm{ coordinate [mpc]}",
                    ylabel=L"y \textrm{ coordinate [mpc]}",
                    title="Trajectory of a Star"
)

result_plot = plot(orbit_plot, size=(1200, 1200), legend=:topright)

# Save the figure
savefig(result_plot, "SyntheticKepler.pdf")

