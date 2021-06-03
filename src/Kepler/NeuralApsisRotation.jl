using DifferentialEquations, Flux, DiffEqFlux
using Plots, Statistics, LinearAlgebra

### Stellar Black Hole System ###

const c = 9.454e9 # 63241 # megameters per year
const G = 1.32e17# 39.478 # gravitational constant in new units : AU^3 * yr^-2 * M^-1
const M = 4 # measure mass in multiples of solar mass
const rs = 2*G*M/c^2 # Schwarzschild radius in AU
const year = 1 # measure time in years

### Creation of synthethetic data --------------------------------- #

r0 = 0.75 # measure radius in mega meters
true_v0 = 0.6 * sqrt(G*M/r0)/sqrt(1-rs/r0)
true_u0 =  [1/r0, 0] # 

true_p = [G*M/(true_v0*r0)^2, G*M/c^2]
phispan = (0.0, 10*pi)
phi = range(phispan[1], phispan[2], length=512)

V0(x,p) = -p[1]*x - p[2]*x^3
dV0(x,p) = Flux.gradient(x -> V0(x,p)[1], x)[1]

# Works! :)
function apsisrotation!(du, u, p, t)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = -dV0(U,p) - U
end

problem = ODEProblem(apsisrotation!, true_u0, phispan, true_p)

@time sol = solve(problem, Tsit5(), saveat=phi)

R_gt = 1 ./ sol[1,:] # convert into Radii
noise = 0.01 * (2 * rand(Float64, size(R_gt)) .- 1)
R_gt = R_gt .+ noise

### End ----------------------------------------------------------- #

dV = FastChain(
    FastDense(1, 16, relu),
    FastDense(16, 1)
)

otherparams = rand(Float64, 2) # otherparams = [u0, p]
ps = vcat(otherparams, initial_params(dV))

function neuralkepler!(du, u, p, t)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = -dV([U], p)[1] - U
end

prob = ODEProblem(neuralkepler!, ps[1:2], phispan, ps[3:end])

function predict(params)
    return Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=phi))
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
    # R_gt = 1 ./ gt[1,:]
    traj_plot = plot(R .* cos.(phi), R .* sin.(phi), xlims=(-1.0, 1.0), ylims=(-1.0, 1.0))
    traj_plot = scatter!(traj_plot, R_gt .* cos.(phi), R_gt .* sin.(phi))

    # Plotting the potential
    u0 = Array(range(0.01, 1.0, step=0.01))
    # y0 = map(x -> Qtils.integrateNN(dV, x, ps[3:end]), 1.0./u0)
    y0 = map(x -> dV(x, p[3:end])[1], u0)
    z0 = map(x -> dV0(x, true_p), u0)
    # z0 = G*M / (true_v0*r0)^2 * 1.0./u0
    pot_plot = plot(u0, y0)
    pot_plot = plot!(pot_plot, u0, z0)

    display(plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 800)))
    return false
end

@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=1000)

