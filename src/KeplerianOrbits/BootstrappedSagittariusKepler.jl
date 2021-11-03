using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Statistics, LinearAlgebra, DataFrames, Distributions
using Plots, Measures, CairoMakie, EarCut, GeometryTypes, LaTeXStrings
include("../lib/SagittariusData.jl")
include("../lib/AwesomeTheme.jl")
include("../lib/Qtils.jl")
using .SagittariusData
using .Qtils

# Load predefined plot theme and adjust font sizes
theme(:awesome)
resetfontsizes()
scalefontsizes(2)

# Natural constants
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

ϕspan = (0.0, 10π)

# Check if the orbit is prograde or retrograde
prograde = SagittariusData.isprograde(star.ϕ)
println("isprograde: ", prograde)
phase = 0
if !prograde
    phase = 0.5
end

### Bootstrap Loop ----------------------------------------------------- #
repetitions = 512
itmlist = DataFrame(params = Array[], r = Array[])
println("Beginning Bootstrap...")
lk = ReentrantLock()
@time Threads.@threads for rep in 1:repetitions
    println("Starting repetition ", rep, " at thread ID ", Threads.threadid())

    # Defining the gradient of the potential using Newton + relativistic correction and initializing parameters
    dV(U,p) = G*p[1]*[p[2]^2 - 3.0*U^2/c^2]
    ps = vcat(rand(Float64, 2), 0.5*rand(Float64, 1), 2.0*rand(Float64, 2), 4.0 .+ rand(Float64,1), rand(Float64,1))

    # Define ODE problem for our system
    function neuralkepler!(du, u, p, ϕ)
        U = u[1]
        dU = u[2]

        du[1] = dU
        du[2] = dV(U, p)[1] - U
    end

    # Defining the problem and optimizer
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
    function loss(params)
        pred = predict(vcat(params[1:4], Zygote.hook(x -> 1e12*x, params[5]), params[6:end])) # gradient is boosted by facto 1e12
        return sum(abs2, pred[1,:].-star.r), pred
    end

    # Callback function 
    cb = function(p,l,pred)
        return l < 1.3
    end

    # Start the training of the model
    @time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=25000)

    # Write result if loss is small enough
    if loss(result.minimizer)[1] < 2.5
        println("Writing results...")
        # Use the best result, i.e. the one with the lowest loss and compute the potential etc. for it
        s, θ = SagittariusData.inversetransform(result.minimizer[3:5].*π, star.r, star.ϕ, prograde)
        θrange = Array(range(minimum(θ), maximum(θ), length=300))
        res = solve(problem, Tsit5(), u0=result.minimizer[1:2], p=result.minimizer[6:end], saveat=θrange)
        r, ϕ = SagittariusData.transform(result.minimizer[3:5].*π, 1.0./res[1,:], θrange, prograde)
        ra, dec = SagittariusData.converttoangles(r, ϕ, D_Astar)
        
        R = sqrt.(ra.^2 .+ dec.^2)

        result.minimizer[3:5] = 180.0 * vcat(mod.(result.minimizer[3], 0.5) + phase, mod.(result.minimizer[4], 1), mod.(result.minimizer[5], 2))

        # Push the result into the array
        lock(lk)
        push!(itmlist, [result.minimizer, R])
        unlock(lk)
    end
    println("Repetition ", rep, " is done!")
end
println("Bootstrap complete!")

# Calculate the mean, standard deviation and 95% confidence intervals for the quantities of interest
mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_r, std_r, CI_r = Qtils.calculatestatistics(itmlist.r)

# Solve the problem again using the parameter mean to obtain values for ϕ
dV(U,p) = G*p[1]*[p[2]^2 - 3.0*U^2/c^2]
ps = vcat(rand(Float64, 2), 0.5*rand(Float64, 1), 2.0*rand(Float64, 2), 4.0 .+ rand(Float64,1), rand(Float64,1))

# Use the ODE again
function neuralkepler!(du, u, p, ϕ)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = dV(U, p)[1] - U
end

# Define the problem again
problem = ODEProblem(neuralkepler!, ps[1:2], ϕspan, ps[6:end]) 

# Use the ϕ computed before to create an interval over which we plot the best fit trajectory
s, θ = SagittariusData.inversetransform(mean_params[3:5].*π, star.r, star.ϕ, prograde)
θrange = Array(range(minimum(θ), maximum(θ), length=300))
res = solve(problem, Tsit5(), u0=mean_params[1:2], p=mean_params[6:end], saveat=θrange)
r, ϕ = SagittariusData.transform(mean_params[3:5].*π, 1.0./res[1,:], θrange, prograde)

println("Parameters: ")
println("Initial Conditions: ", mean_params[1:2], " ± ", std_params[1:2])
println("Angles: ", mean_params[3:5], " ± ", std_params[3:5])
println("Parameters of the potential: ", mean_params[6:end], " ± ", std_params[6:end])

# Create 2D confidence regions around the best fit trajectory
lowerbound = CairoMakie.Point2f0.([[x,y] for (x,y) ∈ zip((mean_r .+ CI_r[1]).*cos.(ϕ), (mean_r .+ CI_r[1]).*sin.(ϕ))])
upperbound = CairoMakie.Point2f0.([[x,y] for (x,y) ∈ zip((mean_r .+ CI_r[2]).*cos.(ϕ), (mean_r .+ CI_r[2]).*sin.(ϕ))])

polygon = [upperbound, lowerbound]
triangle_faces = EarCut.triangulate(polygon)

f = Figure(resolution = (1200, 1200), fontsize=40)
ax = Axis(f[1, 1], title="Angular Trajectory of the Star S2",
            xlabel="Right ascension [mas]",
            ylabel="Declination [mas]",
            bbox = BBox(0, 1200, 0, 1200)
)

CairoMakie.lines!(mean_r.*cos.(ϕ), mean_r.*sin.(ϕ), color=(colorant"#c67", 1.0), label="Prediction of the trajectory")
CairoMakie.errorbars!(S2data.RA, S2data.DEC, S2data.RA_err, direction=:x, whiskerwidth=5)
CairoMakie.errorbars!(S2data.RA, S2data.DEC, S2data.DEC_err, direction=:y, whiskerwidth=5)
CairoMakie.scatter!(S2data.RA, S2data.DEC, 
                    color=colorant"#328", 
                    label="S2 data", 
                    strokewidth=2
)
CairoMakie.mesh!(vcat(polygon...), triangle_faces, color=(colorant"#c67", 0.5), shading=false)
axislegend(ax, position=:lb)

# Save the figure
CairoMakie.save("BootstrappedSagittariusKepler.pdf", f)

