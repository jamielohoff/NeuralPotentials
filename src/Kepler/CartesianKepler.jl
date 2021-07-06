using DifferentialEquations, Flux, DiffEqFlux, ForwardDiff
using Plots, Statistics, LinearAlgebra, LaTeXStrings
include("../MechanicsDatasets.jl")
using .MechanicsDatasets

const c = 63241 # AU per year
const G = 39.478 # gravitational constant in new units : AU^3 * yr^-2 * M_solar^-1
const m = (5.9722 / 1.98847) * 10^(-6) # mass of earth in solar masses
const M = 1.0 # measure mass in multiples of central mass
const r0 = 1.0
const year = 1.0

### Creation of synthethetic data ------------------------ #

true_v0 = 2π*r0 / year # intial velocity
true_u0 =  [1.0, 0.0, 0.0, 1.0*true_v0] # 
true_p = [G*M]
tspan = (0.0, 1.0)
t = range(tspan[1], tspan[2], length=365)

dV0(x, p) = p[1] .* x ./ sqrt(x[1]^2 + x[2]^2).^3

data = MechanicsDatasets.potentialproblem2D(dV0, true_u0, true_p, t, addnoise=true)

### End -------------------------------------------------- #

dV = FastChain(
    FastDense(2, 4, relu),
    FastDense(4, 4, relu),
    FastDense(4, 4, relu),
    FastDense(4, 2)
)

otherparams = rand(Float64, 4)
ps = vcat(otherparams, initial_params(dV))

function neuralkepler!(du, u, p, t)
    x = u[1:2]
    dx = u[3:4]

    du[1:2] = dx
    du[3:4] = -G*dV(x, p)
end

prob = ODEProblem(neuralkepler!, ps[1:4], tspan, ps[5:end])

function predict(params)
    return Array(solve(prob, Tsit5(), u0=params[1:4], p=params[5:end], saveat=t))
end

function loss(params) 
    pred = predict(params)
    return mean((pred .- data).^2), pred
end

opt = ADAM(1e-2, (0.8, 0.95))

u = Array(range(-2.0, 2.0, length=15))
v = Array(range(-2.0, 2.0, length=15))
x = vec([x for (x, y) = Iterators.product(u, v)])
y = vec([y for (x, y) = Iterators.product(u, v)])

mesh = []
for (X, Y) in zip(x, y)
    push!(mesh, [X, Y])
end
scale = 0.05
anim = Animation()

i = 1

cb = function(p,l,pred)
    println("Iteration: ", i)
    println("Loss: ", l)
    println("Initial conditions: ", p[1:4]) # convert into radii
    orbit_plot = plot(pred[1,:], pred[2,:], 
                        label="fit using neural network",
                        xlabel=L"x\textrm{ coordinate in }AU",
                        ylabel=L"y\textrm{ coordinate in }AU",
                        title="position of the test mass and potential"
    )
    orbit_plot = scatter!(orbit_plot, data[2,:], data[3,:], label="data")

    dVx = scale .* map(x -> dV(x, p[5:end])[1], mesh)
    dVy = scale .* map(x -> dV(x, p[5:end])[2], mesh)

    dVx0 = scale .* map(x -> x[1]/sqrt(x[1]^2 + x[2]^2)^3, mesh)
    dVy0 = scale .* map(x -> x[2]/sqrt(x[1]^2 + x[2]^2)^3, mesh)

    dV_plot = quiver(x, y, quiver=(dVx, dVy))
    dV_plot = quiver!(dV_plot, x, y, quiver=(dVx0, dVy0))

    result_plot = plot(orbit_plot, dV_plot, layout=(2,1), size=(1600, 1200), legend=:bottomright)
    frame(anim)
    if mod(i, 10) == 0
        display(result_plot)
    end
    global i += 1
    return false
end

@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=10000)

gif(anim, "S2.gif", fps=60)

