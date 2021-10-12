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

ps = vcat(rand(Float64, 1), 0.5*rand(Float64, 1), 2.0*rand(Float64, 2), 8.0 .+ rand(Float64,1), 4.0 .+ rand(Float64,1), (0.0166*c * 5.946e-2))
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

function converttoangles(r, ϕ, D)
    # D in kpc
    ra = atan.(r.*cos.(ϕ), D*1e5)
    dec = atan.(r.*sin.(ϕ), D*1e5)
    RA = SagittariusData.tomas(ra)
    DEC = SagittariusData.tomas(dec)
    return RA, DEC
end

function converttoposition(ra, dec, ra_err, dec_err, D)
    RA = SagittariusData.toradian(ra)
    DEC = SagittariusData.toradian(dec)
    RA_err = SagittariusData.toradian(ra_err)
    DEC_err = SagittariusData.toradian(dec_err)

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

    arr = vcat(reshape(r,1,:), reshape(ϕ,1,:), reshape(ra,1,:), reshape(dec,1,:), reshape(ra_err,1,:), reshape(dec_err,1,:))

    buf = Zygote.Buffer(arr)
    sortedϕ = sort(ϕ)
    for i ∈ 1:size(r,1)
        j = findall(x -> x==ϕ[i], sortedϕ)[1]
        buf[1,j] = r[i]
        buf[2,j] = ϕ[i]
        buf[3,j] = ra[i]
        buf[4,j] = dec[i]
        buf[5,j] = ra_err[i]
        buf[6,j] = dec_err[i]
    end
    res = copy(buf)
    return res
end

function predict(params)
    star = converttoposition(S2data.RA, S2data.DEC, S2data.RA_err, S2data.DEC_err, 8.33)
    s, θ = SagittariusData.inversetransform(params[2:4].*π, star[1,:], star[2,:], prograde)
    pred = Array(solve(problem, Tsit5(), u0=vcat(1.0/s[1], params[1]), p=params[6:end], saveat=θ))
    r, ϕ = SagittariusData.transform(params[2:4].*π, 1.0./pred[1,:], θ, prograde)
    ra, dec = converttoangles(r, ϕ, 8.33)
    # return vcat(reshape(r,1,:), reshape(ϕ,1,:), reshape(s,1,:), reshape(θ,1,:))
    return vcat(reshape(r,1,:), reshape(ϕ,1,:), reshape(s,1,:), reshape(θ,1,:), 
            reshape(ra,1,:), reshape(dec,1,:), reshape(star[3,:],1,:), reshape(star[4,:],1,:), 
            reshape(star[5,:],1,:), reshape(star[6,:],1,:))
end

function angularχ2loss(ra, dec, RA, DEC, RA_err, DEC_err)
    return sum(abs2, (ra .- RA)) + sum(abs2, (dec .- DEC))
end

function loss(params) 
    pred = predict(Zygote.@showgrad(params)) # Zygote.@showgrad(Zygote.hook(x -> x, params[4]))
    # return χ2loss(pred[1,:], pred[2,:]), pred
    return angularχ2loss(pred[5,:], pred[6,:], pred[7,:], pred[8,:], pred[9,:], pred[10,:]), pred
    # return geometricloss(pred[1,:], pred[2,:]), pred
end

opt = NADAM(0.01) # Nesterov(1e-8) # 
epoch = 0

cb = function(p,l,pred)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    println("Initial velocity: ", p[1])
    println("Rotation angles: ", 180.0 * vcat(mod.(p[2], 0.5) + phase, mod.(p[3:4], 2)))
    println("Initial values: ", 180.0*ps[2:4])
    println("Parameters of the potential: ", p[5:end])

    if epoch % 20 == 0
        orbit_plot = plot(cos.(pred[2,:]) .* pred[1,:], sin.(pred[2,:]) .* pred[1,:], # xlims=(-0.5, 0.5), ylims=(-0.5, 0.5),
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
        # result_plot = plot(orbit_plot, layout=(1,1), size=(1200, 1200), legend=:bottomright)
        display(plot(result_plot))
    end
    global epoch+=1
    return l < 5e-3
    
end

_p = copy(ps)
epochs = 300000
for epoch in 1:epochs
    _loss, _pred = loss(_p)
    _g = Flux.gradient(p -> loss(p)[1], _p)[1]
    if _loss < 500.0
        global opt = NADAM(1e-4)
    end
    Flux.update!(opt, _p, _g)
    breakCondition = cb(_p, _loss, _pred)
    if breakCondition
        break
    end
end

