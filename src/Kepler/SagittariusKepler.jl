using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, Statistics, LinearAlgebra, LaTeXStrings, DataFrames, Distributions
include("SagittariusData.jl")
include("../MechanicsDatasets.jl")
using .SagittariusData
using .MechanicsDatasets

gr()

### Sagittarius A* System ###
const c = 30.64 # centiparsec per year
const G = 4.49e-3 # gravitational constant in new units : (10^-2 parsec)^3 * yr^-2 * (10^6*M_solar)^-1
const kmstocpcyear = 3.154e7/3.09e11

### Initialisation of the Sagittarius data ------------------------ #

path = joinpath(@__DIR__, "SagittariusData.csv")
S2data = SagittariusData.loadstar(path, "S2", timestamps=true)
S2 = SagittariusData.orbit(S2data)
path = joinpath(@__DIR__, "SagittariusOrbitalElements.csv")
S2_orbitalelements = SagittariusData.loadorbitalelements(path, "S2")

S2.r = 100.0 .* S2.r

ϕ0span = (0.01, 5π/2-0.01)
ϕ0 = Array(range(ϕ0span[1], ϕ0span[2], length=145))
r0 = 5.946e-2
true_v0 = sqrt(G*4.35/r0) # initial velocity
true_u0 = [1.0/r0, 0.0, 0.0] 
true_p = [1.0/(1.3*true_v0*r0), G*4.35/(1.3*true_v0*r0)^2]
dV(U,p) = [p[1]*(1.0 - U^2/c^2)]

data = MechanicsDatasets.keplerproblem(dV, true_u0, true_p, ϕ0)

angles = [34.0, 150.0, 65.0].*π/180 # vcat(S2_orbitalelements.i, S2_orbitalelements.Ω, S2_orbitalelements.ω).*π/180 # 
r = 0 
ϕ = 0
if angles[1] > π/2
    angles[1] = angles[1] - π/2
    r, ϕ = SagittariusData.transform(angles, 1.0./data[1,:], ϕ0, false)
else
    r, ϕ = SagittariusData.transform(angles, 1.0./data[1,:], ϕ0, true)
end
orbit = hcat(r, ϕ, data[3,:])
star = DataFrame(orbit, ["r", "ϕ", "t"])
star = sort!(star, [:t])

# star = sort(S2, [:t])

# Δϕ = star.ϕ[2:end] .- star.ϕ[1:end-1]
# idx = findall(x -> abs(x) > 5.0, Δϕ)

# star1 = star[1:idx[1],:]
# star2 = star[idx[1]:end,:]
# sort!(star1, :ϕ)
# sort!(star2, :ϕ)
# star = outerjoin(star1,star2,on=[:r,:ϕ,:t,:x_err,:y_err])

prograde = SagittariusData.isprograde(star.ϕ)
println("isprograde: ", prograde)

### End -------------------------------------------------- #

ps = vcat(rand(Float64, 1), 0.25*rand(Float64, 1), 1.0*rand(Float64, 2), 10.0 .+ rand(Float64,1))
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

function predict(params)
    s, θ = SagittariusData.inversetransform(vcat(params[2:3],Zygote.hook(x -> -x, params[4])).*π, star.r, star.ϕ, prograde) # Zygote.hook(x -> 1e10 * x, params[4])
    println("theta: ", θ)
    # println(sum(abs2, θ .- ϕ0))
    pred = Array(solve(problem, Tsit5(), u0=vcat(1.0/s[1], kmstocpcyear*params[1]), p=params[5:end], saveat=θ)) # params[1]
    r, ϕ = SagittariusData.transform(vcat(params[2:3],Zygote.hook(x -> x, params[4])).*π, 1.0./pred[1,:], θ, prograde) # Zygote.hook(+, params[4])
    # println("phi: ", ϕ)
    return vcat(reshape(r,1,:), reshape(ϕ,1,:), reshape(s,1,:), reshape(θ,1,:))
end

function loss(params) 
    pred = predict(Zygote.@showgrad(params))
    return sum(abs2, 1.0 ./ star.r .- 1.0 ./ pred[1,:]), pred
end

opt = RMSProp(1e-4) # ADAM(1e-3) # 
epoch = 0

cb = function(p,l,pred)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    println("Initial velocity: ", p[1])
    println("Rotation angles: ", 180.0 * vcat(mod.(p[2], 0.5), mod.(p[3:4], 2)))
    println("Inititalization values: ", 180.0*ps[2:4])
    println("Parameters of the potential: ", p[5:end])
    if epoch % 20 == 0
        orbit_plot = plot(cos.(pred[2,:]) .* pred[1,:], sin.(pred[2,:]) .* pred[1,:], # xlims=(-0.5, 0.5), ylims=(-0.5, 0.5),
                            label="fit using neural network",
                            xlabel=L"x\textrm{ coordinate in }10^{-2}pc",
                            ylabel=L"y\textrm{ coordinate in }10^{-2}pc",
                            title="position of the star S2 and gravitational potential"
        )
        orbit_plot = scatter!(orbit_plot, star.r .* cos.(star.ϕ), star.r .* sin.(star.ϕ), label="rotated data") # , xerror = S2.x_err, yerror=S2.y_err)
        orbit_plot = plot!(orbit_plot, pred[3,:] .* cos.(pred[4,:]), pred[3,:] .* sin.(pred[4,:]), label="rotated data")
        orbit_plot = scatter!(orbit_plot, cos.(ϕ0)./data[1,:], sin.(ϕ0)./data[1,:] , label="rotated data")

        # Plotting the potential
        R0 = Array(range(0.3, 11.5, length=100))
        dv = map(u -> dV(u, p[5:end])[1], 1 ./ R0)
        dv0 = map(u -> dV(u, true_p[2:end])[1], 1 ./ R0)
        pot_plot = plot(1 ./ R0, dv)
        pot_plot = plot!(pot_plot, 1 ./ R0, dv0, ylims=(-0.1, 12.0))

        result_plot = plot(orbit_plot, pot_plot, layout=(2,1), size=(1200, 800), legend=:topleft)
        display(plot(result_plot))
    end
    global epoch+=1
    return l < 1e-2
    
end

@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=150000)

