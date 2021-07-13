using Plots: length
using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, Statistics, LinearAlgebra, DataFrames, Distributions
include("../Qtils.jl")
include("../MechanicsDatasets.jl")
include("SagittariusData.jl")
using .Qtils
using .SagittariusData
using .MechanicsDatasets

### Sagittarius A* System ###

const c = 30.64 # centiparsec per year
const G = 4.49e-3 # gravitational constant in new units : (centi-parsec)^3 * yr^-2 * (10^6*M_solar)^-1

### Initialisation of the Sagittarius data ------------------------ #

path = joinpath(@__DIR__, "SagittariusData.csv")
S2data, velocitydata = SagittariusData.loadstar(path, "S2", timestamps=true, velocities=true)
S2 = SagittariusData.orbit(S2data)
S2 = unique!(SagittariusData.centerorbit(S2, sortby=:t), [:t])
tspan = (minimum(S2.t), maximum(S2.t))
trange = Array(range(tspan[1], tspan[end], step=0.1))
t = sort!(unique!(vcat(trange, S2.t, velocitydata.t)))

idx = []
for time in S2.t
    push!(idx, findall(x->x==time, t)[1]) 
end

vidx = []
for time in velocitydata.t
    push!(vidx, findall(x->x==time, t)[1]) 
end

dV0(x,p) = [G*p[1]/x^2]

dϕ0 = 1/0.5*sqrt(G*4.152/0.5)
ps = vcat(0.5, 0.0, -π/2, 1.1*dϕ0, 4.152)

function kepler!(du, u, p, t)
    r = u[1]
    dr = u[2]
    ϕ = u[3]
    dϕ = u[4]

    du[1] = dr
    du[2] = r*dϕ^2 - dV0(r,p)[1]
    du[3] = dϕ
    du[4] = -2*dr*dϕ/r
end

Tspan = (0.0, 23.0)
prob = ODEProblem(kepler!, ps[1:4], Tspan, ps[5:end])
T = Array(range(Tspan[1], Tspan[2], length=200))
data = Array(solve(prob, Tsit5(), u0=ps[1:4], p=ps[5:end], saveat=T))

sigma = 0.01
pdf = Normal(0.0, sigma)
noise =  rand(pdf, size(data))
data =  data .+ noise
# S2.r = 100.0 .* S2.r
# velocitydata.RV = 100.0 * velocitydata.RV

### End -------------------------------------------------- #

# dV = FastChain(
#     FastDense(1, 16, sigmoid),
#     FastDense(16, 8, sigmoid),
#     FastDense(8, 1, exp)
# )

# otherparams = vcat(rand(Float64, 4))
# ps = vcat(otherparams, initial_params(dV))
dV(x,p) = [G*p[1]/x^2]

ps = vcat(rand(Float64, 5))

function neuralkepler!(du, u, p, t)
    r = u[1]
    dr = u[2]
    ϕ = u[3]
    dϕ = u[4]

    du[1] = dr
    du[2] = r*dϕ^2 - dV(r,p)[1]
    du[3] = dϕ
    du[4] = -2*dr*dϕ/r
end

prob = ODEProblem(neuralkepler!, ps[1:4], Tspan, ps[5:end])

function predict(params)
    u0 = vcat(data[:,1])
    return Array(solve(prob, Tsit5(), u0=params[1:4], p=params[5:end], saveat=T))
end

function loss(params)
    pred = predict(params)
    return MechanicsDatasets.reducedχ2(pred, data, size(params,1), sigma), pred # mean( (pred .- data).^2 ), pred
    # return mean((pred[1,idx] .- S2.r).^2)  + mean((pred[2,vidx] .- velocitydata.RV).^2), pred
end

# Now we tell Flux how to train the neural network
opt = ADAM(1e-2, (0.85, 0.95))

cb = function(p,l,pred)
    println("Loss: ", l)
    println("Initial conditions: ", p[1:5])
    R = pred[1, :]
    ϕ = pred[3, :]
    # Plotting the prediction and ground truth
    orbit_plot = plot(R .* cos.(ϕ), R .* sin.(ϕ)) #, xlims=(-3.0, 3.0), ylims=(-3.0, 3.0))
    orbit_plot = scatter!(orbit_plot, data[1,:] .* cos.(data[3,:]), data[1,:] .* sin.(data[3,:]))

    # Plotting the potential
    R0 = Array(range(0.4, 2.0, length=100))
    # y0 = map(x -> MyUtils.integrateNN(dV, x, ps[3:end]), 1.0./u0)
    dv = map(x -> dV(x, p[5:end])[1], R0)
    dv0 = map(x -> dV0(x, [4.152])[1], R0)
    pot_plot = plot(R0, dv)
    pot_plot = plot!(pot_plot, R0, dv0)

    display(plot(orbit_plot, pot_plot, layout=(2,1), size=(1600, 1200)))
    return l < 1e-4
end

@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=3000)

