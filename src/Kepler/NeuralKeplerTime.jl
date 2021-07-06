using Plots: length
using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, Statistics, LinearAlgebra, DataFrames
include("../Qtils.jl")
include("SagittariusData.jl")
using .Qtils
using .SagittariusData

### Sagittarius A* System ###

const c = 0.3064 # parsec per year
const G = 4.49e-9 # gravitational constant in new units : parsec^3 * yr^-2 * (10^6*M_solar)^-1

### Initialisation of the Sagittarius data ------------------------ #

path = joinpath(@__DIR__, "SagittariusData.csv")
S2data, velocitydata = SagittariusData.loadstar(path, "S2", timestamps=true, velocities=true)
S2 = SagittariusData.starorbit(S2data)
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

#S2.r = 100.0 .* S2.r
#velocitydata.RV = 100.0 * velocitydata.RV

### End -------------------------------------------------- #

# dV = FastChain(
#     FastDense(1, 16, sigmoid),
#     FastDense(16, 4, sigmoid),
#     FastDense(4, 1)
# )

# otherparams = vcat(S2.r[1], S2.ϕ[1], rand(Float64, 2))
# ps = vcat(otherparams, initial_params(dV))
dV(x,p) = [G*p[1]/x^2]

ps = vcat(S2.r[1], rand(Float64, 3))

function neuralkepler!(du, u, p, t)
    r = u[1]
    dr = u[2]

    du[1] = dr
    du[2] = r*p[1]^2 - dV(r,p)[1]
end

prob = ODEProblem(neuralkepler!, ps[1:2], tspan, ps[3:end])

function predict(params)
    u0 = vcat(S2.r[1], params[2])
    return Array(solve(prob, Tsit5(), u0=u0, p=params[3:end], saveat=t))
end

function loss(params) 
    pred = predict(params)
    return mean((pred[1,idx] .- S2.r).^2)  + mean((pred[2,vidx] .- velocitydata.RV).^2), pred
end

# Now we tell Flux how to train the neural network
opt = ADAM(1e-2, (0.75, 0.9))

cb = function(p,l,pred)
    println("Loss: ", l)
    println("Initial conditions: ", p[1:4])
    R = pred[1, idx]
    # Plotting the prediction and ground truth
    orbit_plot = plot(R .* cos.(S2.ϕ), R .* sin.(S2.ϕ))#, xlims=(-3.0, 3.0), ylims=(-3.0, 3.0))
    orbit_plot = scatter!(orbit_plot, S2.r .* cos.(S2.ϕ), S2.r .* sin.(S2.ϕ))

    # Plotting the potential
    R0 = Array(range(0.01, 8.0, step=0.01))
    # y0 = map(x -> MyUtils.integrateNN(dV, x, ps[3:end]), 1.0./u0)
    dv = map(x -> dV(x, p[5:end])[1], R0)
    # dv0 = map(x -> dV0(x, true_p), R0)
    # z0 = G*M / (true_v0*r0)^2 * 1.0./u0
    pot_plot = plot(R0, dv)
    # pot_plot = plot!(pot_plot, R0, dv0)

    display(plot(orbit_plot, pot_plot, layout=(2,1), size=(1600, 1200)))
    return l < 1e-4
end

@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=3000)

