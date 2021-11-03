using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Statistics, LinearAlgebra, DataFrames, Distributions
using Plots, Measures, CairoMakie, EarCut, GeometryTypes, LaTeXStrings
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
const G = 4.49 # gravitational constant in new units : (mpc)^3 * yr^-2 * (10^6*M_solar)^-1

### Initialisation of the Synthetic data ------------------------ #
ϕ0span = (0.01, 2π-0.01)
ϕ0 = Array(range(ϕ0span[1], ϕ0span[2], length=144))
r0 = 0.5 # length of the periapsis in mpc 
M = 4.35 # mass of the central SMBH
true_v0 = 1.2*sqrt(G*M/r0) # velocity in mpc/yr at the periapsis
println(true_v0)
true_u0 = [1.0/r0, 0.0, 0.0] 
true_p = [M, 1.0/(true_v0*r0)]
println(true_p)
# Definition of the potential and its gradient
V0(U,p) = G*p[1]*[p[2]^2*U - U^3/c^2]
dV0(U,p) = Zygote.gradient(x -> V0(x,p)[1], U)[1]
data = MechanicsDatasets.keplerproblem(dV0, true_u0, true_p, ϕ0)

# Define the angles which we use to rotate the trajectory
angles = [50.0, 100.0, 65.0].*π/180 
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

### Bootstrap Loop ----------------------------------------------------- # 
repetitions = 1024
itmlist = DataFrame(params = Array[], r = Array[])
println("Beginning Bootstrap...")
lk = ReentrantLock()
@time Threads.@threads for rep in 1:repetitions
    println("Starting repetition ", rep, " at thread ID ", Threads.threadid())

    # Defining the gradient of the potential using Newton + relativistic correction and initializing parameters
    dV(U,p) = G*p[1]*[p[2]^2 + 3.0*U^2/c^2]
    ps = vcat(1.0 .+ rand(Float64, 1), 0.2.*rand(Float64, 1), 0.2 .+ 0.3*rand(Float64, 1), 0.1 .+ 1.9*rand(Float64, 2), 4.0 .+ rand(Float64,1), rand(Float64,1))

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
    opt = NADAM(0.003)

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
        return SagittariusData.χ2(pred[1,:], pred[2,:], star), pred
    end

    # Callback function 
    cb = function(p,l,pred)
        return l < 2e-5
    end

    # Start the training of the model
    @time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=20000)

    # Write result if loss is small enough
    if loss(result.minimizer)[1] < 0.1
        println("Writing results...")
        # Use the best result, i.e. the one with the lowest loss and compute the potential etc. for it
        s, θ = SagittariusData.inversetransform(result.minimizer[3:5].*π, star.r, star.ϕ, prograde)
        res = solve(problem, Tsit5(), u0=result.minimizer[1:2], p=result.minimizer[6:end], saveat=θ)
        r, ϕ = SagittariusData.transform(result.minimizer[3:5].*π, 1.0./res[1,:], θ, prograde)

        result.minimizer[3:5] = 180.0 * vcat(mod.(result.minimizer[3], 0.5) + phase, mod.(result.minimizer[4], 1), mod.(result.minimizer[5], 2))

        # Push the result into the array
        lock(lk)
        push!(itmlist, [result.minimizer, r])
        unlock(lk)
    end
    println("Repetition ", rep, " is done!")
end
println("Bootstrap complete!")

# Calculate the mean, standard deviation and 95% confidence intervals for the quantities of interest
mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_r, std_r, CI_r = Qtils.calculatestatistics(itmlist.r)

println("Parameters: ")
println("Initial Conditions: ", mean_params[1:2], " ± ", std_params[1:2])
println("Angles: ", mean_params[3:5], " ± ", std_params[3:5])
println("Parameters of the potential: ", mean_params[6:end], " ± ", std_params[6:end])

# Create 2D confidence regions around the best fit trajectory
lowerbound = CairoMakie.Point2f0.([[x,y] for (x,y) ∈ zip(cos.(ϕ) .* (mean_r .+ CI_r[1]), sin.(ϕ) .* (mean_r .+ CI_r[1]))])
upperbound = CairoMakie.Point2f0.([[x,y] for (x,y) ∈ zip(cos.(ϕ) .* (mean_r .+ CI_r[2]), sin.(ϕ) .* (mean_r .+ CI_r[2]))])

polygon = [upperbound, lowerbound]
triangle_faces = EarCut.triangulate(polygon)

f = Figure(resolution = (1200, 1200), fontsize=40)
ax = Axis(f[1, 1], title="Trajectory of a Star",
            xlabel="x coordinate [mpc]",
            ylabel="y coordinate [mpc]",
            bbox = BBox(0, 1200, 0, 1200)
)

CairoMakie.lines!(cos.(ϕ) .* mean_r, sin.(ϕ) .* mean_r, color=(colorant"#c67", 1.0), label="Prediction of the trajectory")
CairoMakie.scatter!(star.r .* cos.(star.ϕ), star.r .* sin.(star.ϕ), color=colorant"#328", label="Synthetic data", strokewidth=2)
CairoMakie.mesh!(vcat(polygon...), triangle_faces, color=(colorant"#c67", 0.5), shading=false)
axislegend(ax, position=:lb)

# Save the figure
CairoMakie.save("BootstrappedSyntheticKepler.pdf", f)

