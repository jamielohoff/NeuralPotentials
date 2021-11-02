using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, Statistics, LinearAlgebra, LaTeXStrings, DataFrames, Distributions, Measures
include("SagittariusData.jl")
include("../AwesomeTheme.jl")
include("../Qtils.jl")
using .SagittariusData
using .Qtils

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

### Sagittarius A* System ###
const c = 306.4 # mpc per yr
const G = 4.49 # gravitational constant in new units: (mpc)^3 * yr^-2 * (10^6*M_solar)^-1
const D_Astar = 8.178 * 1e6 # distance of Sagittarius A* in mpc

### Initialisation of the Sagittarius data ------------------------ #

path = joinpath(@__DIR__, "SagittariusData.csv")
S2data = SagittariusData.loadstar(path, "S2", timestamps=true)
S2 = SagittariusData.orbit(S2data, D_Astar)
path = joinpath(@__DIR__, "SagittariusOrbitalElements.csv")
S2_orbitalelements = SagittariusData.loadorbitalelements(path, "S2")
star = sort(S2, [:t])

# Δϕ = star.ϕ[2:end] .- star.ϕ[1:end-1]
# idx = findall(x -> abs(x) > 4.5, Δϕ)[end]
# star1 = star[1:idx,:]
# star2 = star[idx+1:end,:]
# star1.ϕ = star1.ϕ .+ 2π
# sort!(star1, :ϕ, rev=true)
# sort!(star2, :ϕ, rev=true)


Δϕ = star.ϕ[2:end] .- star.ϕ[1:end-1]
idx = findall(x -> abs(x) > 4.5, Δϕ)[1]
star1 = star[1:idx,:]
star2 = star[idx+1:end,:]
star2.ϕ = star2.ϕ .+ 2π
sort!(star1, :ϕ)
sort!(star2, :ϕ)

star = outerjoin(star1,star2,on=[:r,:ϕ,:t,:x_err,:y_err])
unique!(star, [:ϕ])

println(star)

prograde = SagittariusData.isprograde(star.ϕ)
println("isprograde: ", prograde)
phase = 0
if !prograde
    phase = 0.5
end

### End -------------------------------------------------- #

itmlist = DataFrame(params = Array[], r = Array[])
repetitions = 64

println("Beginning Bootstrap...")
lk = ReentrantLock()
@time Threads.@threads for rep in 1:repetitions
    println("Starting repetition ", rep, " at thread ID ", Threads.threadid())
    dV(U,p) = G*p[1]*[p[2]^2 - 3.0*U^2/c^2]
    ps = vcat(rand(Float64, 2), 0.5*rand(Float64, 1), 2.0*rand(Float64, 2), 4.0 .+ rand(Float64,1), rand(Float64,1))

    function neuralkepler!(du, u, p, ϕ)
        U = u[1]
        dU = u[2]

        du[1] = dU
        du[2] = dV(U, p)[1] - U
    end

    ϕspan = (0.0, 10π)
    problem = ODEProblem(neuralkepler!, ps[1:2], ϕspan, ps[6:end]) 

    function predict(params)
        s, θ = SagittariusData.inversetransform(params[3:5].*π, star.r, star.ϕ, prograde)
        pred = Array(solve(problem, Tsit5(), u0=params[1:2], p=params[6:end], saveat=θ))
        r, ϕ = SagittariusData.transform(params[3:5].*π, 1.0./pred[1,:], θ, prograde)
        ra, dec = SagittariusData.converttoangles(r, ϕ, D_Astar)
        return vcat(reshape(r,1,:), reshape(ϕ,1,:), reshape(s,1,:), reshape(θ,1,:), reshape(ra,1,:), reshape(dec,1,:))
    end

    function χ2(r, ϕ)
        return sum(abs2, (r.*cos.(ϕ) .- star.r.*cos.(star.ϕ))./star.x_err) + sum(abs2, (r.*sin.(ϕ) .- star.r.*sin.(star.ϕ))./star.y_err)
    end

    function loss(params) # 1e9
        pred = predict(vcat(params[1:4], Zygote.hook(x -> 1e12*x, params[5]), params[6:end]))
        return sum(abs2, pred[1,:].-star.r), pred
        # return χ2(pred[1,:], pred[2,:]), pred
        # return geom(pred[1,:], pred[2,:]), pred
    end

    opt = NADAM(0.001)
    epoch = 0

    cb = function(p,l,pred)
        return l < 1.3
    end

    @time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=25000)

    if loss(result.minimizer)[1] < 2.5
        println("Writing results...")
        s, θ = SagittariusData.inversetransform(result.minimizer[3:5].*π, star.r, star.ϕ, prograde)
        θrange = Array(range(minimum(θ), maximum(θ), length=300))
        res = solve(problem, Tsit5(), u0=result.minimizer[1:2], p=result.minimizer[6:end], saveat=θrange)
        r, ϕ = SagittariusData.transform(result.minimizer[3:5].*π, 1.0./res[1,:], θrange, prograde)
        ra, dec = SagittariusData.converttoangles(r, ϕ, D_Astar)
        
        R = sqrt.(ra.^2 .+ dec.^2)

        result.minimizer[3:5] = 180.0 * vcat(mod.(result.minimizer[3], 0.5) + phase, mod.(result.minimizer[4], 1), mod.(result.minimizer[5], 2))

        lock(lk)
        push!(itmlist, [result.minimizer, R])
        unlock(lk)
    end
    println("Repetition ", rep, " is done!")
end
println("Bootstrap complete!")

mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_r, std_r, CI_r = Qtils.calculatestatistics(itmlist.r)

dV(U,p) = G*p[1]*[p[2]^2 - 3.0*U^2/c^2]
ps = vcat(rand(Float64, 2), 0.5*rand(Float64, 1), 2.0*rand(Float64, 2), 4.0 .+ rand(Float64,1), rand(Float64,1))

function neuralkepler!(du, u, p, ϕ)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = dV(U, p)[1] - U
end

ϕspan = (0.0, 10π)
problem = ODEProblem(neuralkepler!, ps[1:2], ϕspan, ps[6:end]) 

s, θ = SagittariusData.inversetransform(mean_params[3:5].*π, star.r, star.ϕ, prograde)
θrange = Array(range(minimum(θ), maximum(θ), length=300))
res = solve(problem, Tsit5(), u0=mean_params[1:2], p=mean_params[6:end], saveat=θrange)
r, ϕ = SagittariusData.transform(mean_params[3:5].*π, 1.0./res[1,:], θrange, prograde)

println("Parameters: ")
println("Initial Conditions: ", mean_params[1:2], " ± ", std_params[1:2])
println("Angles: ", mean_params[3:5], " ± ", std_params[3:5])
println("Parameters of the potential: ", mean_params[6:end], " ± ", std_params[6:end])

using CairoMakie, EarCut, GeometryTypes

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
# display(f)
CairoMakie.save("BootstrappedSagittariusKepler.pdf", f)

