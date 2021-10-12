using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, Statistics, LinearAlgebra, LaTeXStrings, DataFrames, Distributions, Measures
include("SagittariusData.jl")
include("../MechanicsDatasets.jl")
include("../AwesomeTheme.jl")
using .SagittariusData
using .MechanicsDatasets

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

### Sagittarius A* System ###
const c = 30.64 # centiparsec per year
const G = 4.49e-3 # gravitational constant in new units: (10^-2 parsec)^3 * yr^-2 * (10^6*M_solar)^-1
D_Astar = 8.178 # distance of Sagittarius A* in kpc
# const hunderedkmstocpcyear = 10.0*3.154e7/3.09e11

### Initialisation of the Sagittarius data ------------------------ #

path = joinpath(@__DIR__, "SagittariusData.csv")
S2data = SagittariusData.loadstar(path, "S2", timestamps=true)
S2 = SagittariusData.orbit(S2data, D_Astar)
path = joinpath(@__DIR__, "SagittariusOrbitalElements.csv")
S2_orbitalelements = SagittariusData.loadorbitalelements(path, "S2")

dV(U,p) = G*p[1]*[1/p[2]^2 + 3*U^2/c^2] # [p[1]] # 

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

ps = vcat(rand(Float64, 1), 0.5*rand(Float64, 1), 2.0*rand(Float64, 2), rand(Float64,1), 4.0 .+ rand(Float64,1), (0.0166*c * 5.946e-2))
println(180.0*ps[2:4])

function neuralkepler!(du, u, p, ϕ)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = dV(U, p)[1] - U
end

u0 = ps[1:2]
ϕspan = (0.0, 10π)
problem = ODEProblem(neuralkepler!, u0, ϕspan, ps[6:end]) 

function converttoposition(ra, dec, D)
    RA = SagittariusData.toradian(ra)
    DEC = SagittariusData.toradian(dec)
    x = D*tan.(RA)*1e5
    y = D*tan.(DEC)*1e5
    r = sqrt.(x.^2 + y.^2)
    ϕ = mod.(atan.(y,x), 2π)

    Δϕ = ϕ[2:end] .- ϕ[1:end-1]
    idx = findall(x -> abs(x) > 5.0, Δϕ)[1]
    ϕ1 = ϕ[1:idx]
    ϕ2 = ϕ[idx+1:end]
    ϕ2 = ϕ2 .+ 2π
    ϕ = vcat(ϕ1, ϕ2)

    arr = vcat(reshape(r,1,:), reshape(ϕ,1,:), reshape(RA,1,:), reshape(DEC,1,:))

    buf = Zygote.Buffer(arr)
    sortedϕ = sort(ϕ)
    for i ∈ 1:size(r,1)
        j = findall(x -> x==ϕ[i], sortedϕ)[1]
        buf[1,j] = r[i]
        buf[2,j] = ϕ[i]
        buf[3,j] = RA[i]
        buf[4,j] = DEC[i]
    end
    res = copy(buf)
    println(res)
    return res
end

function predict(params)
    s, θ = SagittariusData.inversetransform(params[2:4].*π, star.r, star.ϕ, prograde)
    pred = Array(solve(problem, Tsit5(), u0=vcat(params[5], params[1]), p=params[6:end], saveat=θ)) # 1.0/s[1]
    r, ϕ = SagittariusData.transform(params[2:4].*π, 1.0./pred[1,:], θ, prograde)
    ra, dec = SagittariusData.converttoangles(r, ϕ, D_Astar)
    return vcat(reshape(r,1,:), reshape(ϕ,1,:), reshape(s,1,:), reshape(θ,1,:), reshape(ra,1,:), reshape(dec,1,:))
end

function χ2loss(r, ϕ)
    return sum(abs2, (r.*cos.(ϕ) .- star.r.*cos.(star.ϕ))./star.x_err) + sum(abs2, (r.*sin.(ϕ) .- star.r.*sin.(star.ϕ))/star.y_err)
end

function loss(params) 
    pred = predict(vcat(params[1:3], Zygote.@showgrad(Zygote.hook(x -> 1e9*x, params[4])), params[5:end])) # Zygote.@showgrad(Zygote.hook(x -> x, params[4]))
    return χ2loss(pred[1,:], pred[2,:]), pred
end

opt = NADAM(0.01)
epoch = 0

cb = function(p,l,pred)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    println("Initial velocity: ", p[1])
    println("Rotation angles: ", 180.0 * vcat(mod.(p[2], 0.5) + phase, mod.(p[3:4], 2)))
    println("Initial values: ", 180.0*ps[2:4])
    println("Parameters of the potential: ", p[5:end])

    if epoch % 20 == 0
        orbit_plot = plot(cos.(pred[2,:]) .* pred[1,:], sin.(pred[2,:]) .* pred[1,:],
                            label="Prediction using neural potential",
                            xlabel=L"x\textrm{ coordinate in }10^{-2}\textrm{pc}",
                            ylabel=L"y\textrm{ coordinate in }10^{-2}\textrm{pc}",
                            title="Position of the Star S2 and Gravitational Potential"
        )
        orbit_plot = scatter!(orbit_plot, star.r .* cos.(star.ϕ), star.r .* sin.(star.ϕ), label="S2 data",  xerror = star.x_err, yerror=star.y_err)
        orbit_plot = plot!(orbit_plot, pred[3,:] .* cos.(pred[4,:]), pred[3,:] .* sin.(pred[4,:]), label="Prediction of unrotated data")

        angular_plot = scatter(pred[5,:], pred[6,:])
        angular_plot = scatter!(angular_plot, S2data.RA, S2data.DEC, xerror=S2data.RA_err, yerror=S2data.DEC_err)

        result_plot = plot(orbit_plot, angular_plot, layout=(2,1), size=(1200, 1200), legend=:bottomright)
        display(plot(result_plot))
    end
    global epoch+=1
    return l < 150.0
    
end

_p = copy(ps)
epochs = 300000
for epoch in 1:epochs
    _loss, _pred = loss(_p)
    _g = Flux.gradient(p -> loss(p)[1], _p)[1]
    if _loss < 300.0
        global opt = NADAM(1e-3)
    end
    if _loss < 150.0
        global opt = NADAM(1e-4)
    end
    Flux.update!(opt, _p, _g)
    breakCondition = cb(_p, _loss, _pred)
    if breakCondition
        break
    end
end

