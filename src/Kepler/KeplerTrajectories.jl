using DifferentialEquations, DiffEqFlux, Flux, Zygote
using Plots, Statistics, LinearAlgebra
include("../Qtils.jl")
using .Qtils

### Earth-Sun System ###

const c = 63241 # AU per year
const G = 39.478 # gravitational constant in new units : AU^3 * yr^-2 * M^-1
const M = 1 # measure mass in multiples of central mass, i.e. sun

const m = (5.9722 / 1.98847) * 10^(-6) # 
const r0 = 1 # measure radius in multiples of r0, i.e. unit of length is AU
const year = 1 # measure time in years

### Creation of synthethetic data ------------------------ #

true_v0 = 2*pi*r0 / year # intial velocity
true_u0 =  [1/r0, 0] # 
true_p = [G*M / (true_v0*r0)^2]
phispan = (0.0, 2*pi)
phi = Array(range(phispan[1], phispan[2], length=128))

V0(x,p) = -p[1]*x 
dV0(x,p) = Flux.gradient(x -> V0(x,p)[1], x)[1]

# Works! :)
function kepler!(du, u, p, t)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = -dV0(U,p) - U
end

initialConditions = rand(Float32, (64, 2))
@time trajectories = Qtils.sampletrajectories(kepler!, true_p, initialConditions, phi)

### End -------------------------------------------------- #

dV = FastChain(
    FastDense(1, 16, relu),
    FastDense(16, 1)
)

ps = vcat(initial_params(dV))

function neuralkepler!(du, u, p, t)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = -dV([U], p)[1] - U
end

prob = ODEProblem(neuralkepler!, true_u0, phispan, ps)

function predict(u0, params)
    return Array(solve(prob, Tsit5(), u0=u0, p=params, saveat=phi))
end

function loss(groundtruth, params) 
    u0 = [groundtruth[1,1], groundtruth[2,1]]
    pred = predict(u0, params)
    return mean((pred .- groundtruth).^2), pred
end

cb = function(p,l,pred,gt)
    display(l)
    R_gt = 1 ./ gt[1,:] # convert into Radii
    R = 1 ./ pred[1, :] # convert into Radii

    # Plotting the prediction and ground truth
    traj_plot = plot(R .* cos.(phi), R .* sin.(phi), xlims=(-10.0, 10.0), ylims=(-10.0, 10.0))
    traj_plot = scatter!(traj_plot, R_gt .* cos.(phi), R_gt .* sin.(phi))

    # Plotting the potential
    u0 = Array(range(0.01, 1.0, step = 0.01))
    # y0 = map(x -> MyUtils.integrateNN(dV, x, ps[3:end]), 1.0./u0)
    y0 = map(x -> dV(x, p)[1], u0)
    z0 = map(x -> dV0(x, true_p), u0)
    # z0 = G*M / (true_v0*r0)^2 * 1.0./u0
    pot_plot = plot(u0, y0)
    pot_plot = plot!(pot_plot, u0, z0, ylims=(-1.5, 0.0))

    display(plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 800)))
    return l < 0.01
end

# Now we tell Flux how to train the neural network
dataloader = Flux.Data.DataLoader(trajectories; batchsize=1, shuffle=true)
opt = ADAM(1e-2)
epochs = 1000
for epoch in 1:epochs
    println("epoch: ", epoch, " of ", epochs)
    for batch in dataloader
        _loss, _pred = loss(batch[1], ps)
        gs = Flux.gradient(p -> loss(batch[1], p)[1], ps)[1]
        Flux.update!(opt, ps, gs)
        breakCondition = cb(ps, _loss, _pred, batch[1])
        if breakCondition
            break
        end
    end
end

