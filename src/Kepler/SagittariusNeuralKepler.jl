using DifferentialEquations, Flux, DiffEqFlux
using Plots, Statistics, LinearAlgebra, LaTeXStrings
include("../Qtils.jl")
include("SagittariusData.jl")
using .Qtils
using .SagittariusData

### Sagittarius A* System ###

const c = 0.3064 # parsec per year
const G = 4.49e-15 # gravitational constant in new units : parsec^3 * yr^-2 * M_solar^-1

### Initialisation of the Sagittarius data ------------------------ #

path = joinpath(@__DIR__, "SagittariusData.csv")
S2data = SagittariusData.loadstar(path, "S2", timestamps=true)
S2 = SagittariusData.starorbit(S2data)
S2 = unique!(SagittariusData.centerorbit(S2, sortby=:t), [:t])
t = S2.t
tspan = (minimum(t), maximum(t))
S2 = 100.0 .* S2

### End -------------------------------------------------- #

dV = FastChain(
    FastDense(2, 4, relu),
    FastDense(4, 4, relu),
    FastDense(4, 2)
)

otherparams = vcat(1.0-9.0 * rand(Float64, 1)[1], 0.05 - 0.1 * rand(Float64, 1)[1], 0.002 .* rand(Float64, 2))
ps = vcat(otherparams, initial_params(dV))

function neuralkepler!(du, u, p, t)
    x = u[1:2]
    dx = u[3:4]

    du[1:2] = dx
    du[3:4] = -dV(x, p)
end

prob = ODEProblem(neuralkepler!, ps[1:4], tspan, ps[5:end])

function predict(params)
    return Array(solve(prob, Tsit5(), u0=params[1:4], p=params[5:end], saveat=t))
end

function loss(params) 
    pred = predict(params)
    return SagittariusData.reducedχ2(pred, S2, size(params,1)), pred
end

opt = ADAM(1e-2, (0.9, 0.95))# RMSProp(1e-2)

u = Array(range(-10.0, 2.5, length=15))
v = Array(range(-0.5, 0.5, length=15))
x = vec([x for (x, y) = Iterators.product(u, v)])
y = vec([y for (x, y) = Iterators.product(u, v)])

mesh = []
for (X, Y) in zip(x, y)
    push!(mesh, [X, Y])
end
scale = 0.5
anim = Animation()

cb = function(p,l,pred)
    println("Loss: ", l)
    println("Initial conditions: ", p[1:4]) # convert into radii
    orbit_plot = plot(pred[1,:], pred[2,:], 
                        label="fit using neural network",
                        xlabel=L"x\textrm{ coordinate in }AU",
                        ylabel=L"y\textrm{ coordinate in }AU",
                        title="position of the test mass and potential"
    )
    orbit_plot = scatter!(orbit_plot, S2.x, S2.y, label="data", xerror=S2.x_err, yerror=S2.y_err)

    dVx = scale .* map(x -> dV(x, p[5:end])[1], mesh)
    dVy = scale .* map(x -> dV(x, p[5:end])[2], mesh)

    # dVx0 = scale .* map(x -> true_p[1]*x[1]/sqrt(x[1]^2 + x[2]^2)^3, mesh)
    # dVy0 = scale .* map(x -> true_p[1]*x[2]/sqrt(x[1]^2 + x[2]^2)^3, mesh)

    dV_plot = quiver(x, y, quiver=(dVx, dVy))
    # dV_plot = quiver!(dV_plot, x, y, quiver=(dVx0, dVy0))

    result_plot = plot(orbit_plot, dV_plot, layout=(2,1), size=(1600, 1200), legend=:bottomright)
    frame(anim)
    # display(plot(orbit_plot, dV_plot, layout=(2,1), size=(1600, 1200), legend=:bottomright))
    return false
end

@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=5000)

gif(anim, "S2.gif", fps=60)

