using DifferentialEquations, Flux, DiffEqFlux
using Plots, Statistics, LinearAlgebra
include("../Qtils.jl")
using .Qtils

### Earth-Sun System ###

const c = 63241 # AU per year
const G = 39.478 # gravitational constant in new units : AU^3 * yr^-2 * M^-1
const M = 1 # measure mass in multiples of central mass

const m = (5.9722 / 1.98847) * 10^(-6) # 
const r0 = 1 # measure radius in multiples of r0, i.e. unit of length is AU
const year = 1 # measure time in years

### Creation of synthethetic data ------------------------ #

true_v0 = 2π*r0 / year # intial velocity
true_u0 =  [1/r0, 1.0] # initial distance, make 2nd argument 0.0 to get circular orbit
true_p = [G*M / (true_v0*r0)^2]
ϕspan = (0.0, 2π)
ϕ = range(ϕspan[1], ϕspan[2], length=100)

V0(x,p) = -p[1]*x 
dV0(x,p) = Flux.gradient(x -> V0(x,p)[1], x)[1]

# Works! :)
function kepler!(du, u, p, ϕ)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = -dV0(U,p) - U
end

problem = ODEProblem(kepler!, true_u0, ϕspan, true_p)

@time sol = solve(problem, Tsit5(), saveat=ϕ)

R_gt = 1 ./ sol[1,:] # convert into Radii
noise = 0.02 * (2 * rand(Float64, size(R_gt)) .- 1)
R_gt = R_gt .+ noise

### End -------------------------------------------------- #

dV = FastChain(
    FastDense(1, 8, relu),
    FastDense(8, 8, relu),
    FastDense(8, 1)
)

otherparams = rand(Float64, 2) # otherparams = [u0, p]
ps = vcat(otherparams, initial_params(dV))

function neuralkepler!(du, u, p, t)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = -dV(U, p)[1] - U
end

prob = ODEProblem(neuralkepler!, ps[1:2], ϕspan, ps[3:end])

function predict(params)
    return Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=ϕ))
end

function loss(params) 
    pred = predict(params)
    return mean((pred .- sol[1:2, :]).^2), pred
end

# Now we tell Flux how to train the neural network
opt = ADAM(1e-2)

cb = function(p,l,pred)
    display(l)
    display(p[1:2])
    R = 1 ./ pred[1, :] # convert into Radii
    # Plotting the prediction and ground truth
    orbit_plot = plot(R .* cos.(ϕ), R .* sin.(ϕ), xlims=(-3.0, 3.0), ylims=(-3.0, 3.0))
    orbit_plot = scatter!(orbit_plot, R_gt .* cos.(ϕ), R_gt .* sin.(ϕ))

    # Plotting the potential
    u0 = Array(range(0.01, 1.0, step = 0.01))
    # y0 = map(x -> MyUtils.integrateNN(dV, x, ps[3:end]), 1.0./u0)
    y0 = map(x -> dV(x, p[3:end])[1], u0)
    z0 = map(x -> dV0(x, true_p), u0)
    # z0 = G*M / (true_v0*r0)^2 * 1.0./u0
    pot_plot = plot(u0, y0)
    pot_plot = plot!(pot_plot, u0, z0, ylims=(-1.5, 0.0))

    display(plot(orbit_plot, pot_plot, layout=(2,1), size=(1200, 800)))
    return l < 1e-4
end

@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=1000)

