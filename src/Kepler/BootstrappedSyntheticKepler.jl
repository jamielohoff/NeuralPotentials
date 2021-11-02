using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, Statistics, LinearAlgebra, LaTeXStrings, DataFrames, Distributions, Measures
include("SagittariusData.jl")
include("../MechanicsDatasets.jl")
include("../AwesomeTheme.jl")
include("../Qtils.jl")
using .SagittariusData
using .MechanicsDatasets
using .Qtils

Plots.theme(:awesome)
resetfontsizes()
scalefontsizes(2)

### Sagittarius A* System ###
const c = 306.4 # milliparsec per year
const G = 4.49 # gravitational constant in new units : (milliparsec)^3 * yr^-2 * (10^6*M_solar)^-1

### Initialisation of the Sagittarius data ------------------------ #

ϕ0span = (0.01, 2π-0.01)
ϕ0 = Array(range(ϕ0span[1], ϕ0span[2], length=144))
r0 = 0.5
M = 4.35
true_v0 = 1.2*sqrt(G*M/r0) # initial velocity # 0.01*c # 
println(true_v0)
true_u0 = [1.0/r0, 0.0, 0.0] 
true_p = [M, 1.0/(true_v0*r0)]
println(true_p)
V0(U,p) = G*p[1]*[p[2]^2*U - U^3/c^2]
dV0(U,p) = Zygote.gradient(x -> V0(x,p)[1], U)[1]
data = MechanicsDatasets.keplerproblem(dV0, true_u0, true_p, ϕ0)

angles = [50.0, 100.0, 65.0].*π/180 
println(angles)
r = 0
ϕ = 0
x_err = ones(size(data.r))
y_err = ones(size(data.r))
if angles[1] > π/2
    angles[1] = angles[1] - π/2
    r, ϕ = SagittariusData.transform(angles, data.r, ϕ0, false)
else
    r, ϕ = SagittariusData.transform(angles, data.r, ϕ0, true)
end

orbit = hcat(r, ϕ, data.t, x_err, y_err)
star = DataFrame(orbit, [:r, :ϕ, :t, :x_err, :y_err])
star = sort!(star, [:t])

prograde = SagittariusData.isprograde(star.ϕ)
println("isprograde: ", prograde)
phase = 0
if !prograde
    phase = 0.5
end

### End -------------------------------------------------- #

### Bootstrap Loop 
itmlist = DataFrame(params = Array[], r = Array[])
repetitions = 8

println("Beginning Bootstrap...")
lk = ReentrantLock()
@time Threads.@threads for rep in 1:repetitions
    println("Starting repetition ", rep, " at thread ID ", Threads.threadid())

    dV(U,p) = G*p[1]*[p[2]^2 + 3.0*U^2/c^2]
    ps = vcat(1.0 .+ rand(Float64, 1), 0.2.*rand(Float64, 1), 0.2 .+ 0.3*rand(Float64, 1), 0.1 .+ 1.9*rand(Float64, 2), 4.0 .+ rand(Float64,1), rand(Float64,1))

    function neuralkepler!(du, u, p, ϕ)
        U = u[1]
        dU = u[2]

        du[1] = dU
        du[2] = dV(U, p)[1] - U
    end

    u0 = ps[1:2]
    ϕspan = (0.0, 10π)
    problem = ODEProblem(neuralkepler!, u0, ϕspan, ps[6:end]) 

    function predict(params)
        s, θ = SagittariusData.inversetransform(params[3:5].*π, star.r, star.ϕ, prograde)
        pred = Array(solve(problem, Tsit5(), u0=params[1:2], p=params[6:end], saveat=θ))
        r, ϕ = SagittariusData.transform(params[3:5].*π, 1.0./pred[1,:], θ, prograde)
        return vcat(reshape(r,1,:), reshape(ϕ,1,:))
    end

    function χ2(r, ϕ)
        return sum(abs2, (r.*cos.(ϕ) .- star.r.*cos.(star.ϕ))./star.x_err) + sum(abs2, (r.*sin.(ϕ) .- star.r.*sin.(star.ϕ))/star.y_err)
    end

    function loss(params) 
        pred = predict(vcat(params[1:4], Zygote.hook(x -> 1e12*x, params[5]), params[6:end]))
        return χ2(pred[1,:], pred[2,:]), pred
    end

    opt = NADAM(0.003)

    cb = function(p,l,pred)
        # println("Loss: ", l)
        return l < 2e-5
    end

    @time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=20000)

    if loss(result.minimizer)[1] < 0.1
        println("Writing results...")
        s, θ = SagittariusData.inversetransform(result.minimizer[3:5].*π, star.r, star.ϕ, prograde)
        res = solve(problem, Tsit5(), u0=result.minimizer[1:2], p=result.minimizer[6:end], saveat=θ)
        r, ϕ = SagittariusData.transform(result.minimizer[3:5].*π, 1.0./res[1,:], θ, prograde)

        result.minimizer[3:5] = 180.0 * vcat(mod.(result.minimizer[3], 0.5) + phase, mod.(result.minimizer[4], 1), mod.(result.minimizer[5], 2))

        lock(lk)
        push!(itmlist, [result.minimizer, r])
        unlock(lk)
    end
    println("Repetition ", rep, " is done!")
end
println("Bootstrap complete!")

mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_r, std_r, CI_r = Qtils.calculatestatistics(itmlist.r)

println("Parameters: ")
println("Initial Conditions: ", mean_params[1:2], " ± ", std_params[1:2])
println("Angles: ", mean_params[3:5], " ± ", std_params[3:5])
println("Parameters of the potential: ", mean_params[6:end], " ± ", std_params[6:end])

using CairoMakie, EarCut, GeometryTypes

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
# display(f)
CairoMakie.save("BootstrappedSyntheticKepler.pdf", f)

