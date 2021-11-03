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
const G = 4.49 # gravitational constant in new units: (mpc)^3 * yr^-2 * (10^6*M_solar)^-1
const D_Astar = 8.178 * 1e6 # distance of Sagittarius A* in mpc

### Initialisation of the Sagittarius data ------------------------ #
path = joinpath(@__DIR__, "SagittariusData.csv")
S2data = SagittariusData.loadstar(path, "S2", timestamps=true)
S2 = SagittariusData.orbit(S2data, D_Astar)
path = joinpath(@__DIR__, "SagittariusOrbitalElements.csv")
S2_orbitalelements = SagittariusData.loadorbitalelements(path, "S2")
star = SagittariusData.orderobservations(S2)
# Defining the gradient of the potential using Newton + relativistic correction and initializing parameters
dV(U,p) = G*p[1]*[p[2]^2 - 3.0*U^2/c^2]
ps = vcat(rand(Float64, 2), 0.5*rand(Float64, 1), 2.0*rand(Float64, 2), 4.0 .+ rand(Float64,1), rand(Float64,1))

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
ϕspan = (0.0, 10π)
problem = ODEProblem(neuralkepler!, ps[1:2], ϕspan, ps[6:end]) 
opt = NADAM(0.001)

# Function that predicts the results for a given set of parameters by solving the ODE at the given angles ϕ
function predict(params)
    # Rotate the trajectory in the observational plane
    s, θ = SagittariusData.inversetransform(params[3:5].*π, star.r, star.ϕ, prograde)
    # Solve the ODE in the observational plane using the θ's
    pred = Array(solve(problem, Tsit5(), u0=params[1:2], p=params[6:end], saveat=θ))
    # Rotate the trajectory back into the plane of motion and project it onto the observational plane
    r, ϕ = SagittariusData.transform(params[3:5].*π, 1.0./pred[1,:], θ, prograde)
    # Convert the physical distances into pairs of equatorial coordinates, i.e. right ascension and declination
    ra, dec = SagittariusData.converttoangles(r, ϕ, D_Astar)
    return vcat(reshape(r,1,:), reshape(ϕ,1,:), reshape(ra,1,:), reshape(dec,1,:))
end

# Function that calculates the loss with respect to the S2 data
function loss(params) # 1e9
    pred = predict(vcat(params[1:4], Zygote.hook(x -> 1e12*x, params[5]), params[6:end])) # gradient is boosted by facto 1e12
    return sum(abs2, pred[1,:].-star.r), pred
end

# Callback function 
epoch = 0
cb = function(p,l,pred)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    println("Initial conditions: ", p[1:2])
    println("Rotation angles: ", 180.0 * vcat(mod.(p[3], 0.5) + phase, mod.(p[4], 1), mod.(p[5], 2)))
    println("Parameters of the potential: ", p[6:end])

    if epoch % 10 == 0
        # orbit_plot = scatter(star.r .* cos.(star.ϕ), star.r .* sin.(star.ϕ), label="S2 data",  xerror = star.x_err, yerror=star.y_err)
        # orbit_plot = plot!(orbit_plot, cos.(pred[2,:]) .* pred[1,:], sin.(pred[2,:]) .* pred[1,:],
        #                     label="Prediction using neural potential",
        #                     xlabel="x coordinate [mpc]",
        #                     ylabel="y coordinate [mpc]",
        #                     title="Position of the Star S2"
        # )
        # orbit_plot = plot!(orbit_plot, pred[3,:] .* cos.(pred[4,:]), pred[3,:] .* sin.(pred[4,:]), label="Prediction of unrotated data")

        angular_plot = scatter(S2data.RA, S2data.DEC, xerror=S2data.RA_err, yerror=S2data.DEC_err)
        angular_plot = plot!(angular_plot, pred[3,:], pred[4,:], 
                            label="Prediction of the Trajectory",
                            xlabel=L"\textrm{Right ascension } \alpha \textrm{ [mas]}",
                            ylabel=L"\textrm{Declination } \delta \textrm{ [mas]}",
                            title="Trajectory of the Star S2")

        result_plot = plot(angular_plot, size=(1200, 1200), legend=:bottomleft)
        display(plot(result_plot))
    end
    global epoch+=1
    return l < 1.3
    
end

# Start the training of the model
@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=40000)

# Calculating the resulting trjectory
s, θ = SagittariusData.inversetransform(result.minimizer[3:5].*π, star.r, star.ϕ, prograde)
θrange = Array(range(minimum(θ), maximum(θ), length=300))
res = solve(problem, Tsit5(), u0=result.minimizer[1:2], p=result.minimizer[6:end], saveat=θrange)
r, ϕ = SagittariusData.transform(result.minimizer[3:5].*π, 1.0./res[1,:], θrange, prograde)
ra, dec = SagittariusData.converttoangles(r, ϕ, D_Astar)

angular_plot = scatter(S2data.RA, S2data.DEC, 
                        xerror=S2data.RA_err, yerror=S2data.DEC_err,
                        title="Angular Trajectory of the Star S2",
                        xlabel=L"\textrm{Right ascension } \alpha \textrm{ [mas]}",
                        ylabel=L"\textrm{Declination } \delta \textrm{ [mas]}",
                        label="S2 data"
)
angular_plot = plot!(angular_plot, ra, dec,label="Prediction of the trajectory")

result_plot = plot(angular_plot, size=(1200, 1200), legend=:bottomleft)

# Save the figure
savefig(result_plot, "SagittariusFit.pdf")

