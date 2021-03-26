using DifferentialEquations, Flux, DiffEqFlux, Plots
include("../PotPlot.jl")
using .PotPlot

c = 63241 # AU per year
G = 39.478 # gravitational constant in new units : AU^3 * yr^-2 * M^-1


m = (5.9722 / 1.98847) * 10^(-6) # 
M = 1 # measure mass in multiples of central mass
r0 = 1 # measure radius in multiples of r0, i.e. unit of length is AU
year = 1 # measure time in years

### Creation of synthethetic data ------------------------ #

true_v0 = 2*pi*r0 / year # intial velocity
true_u0 =  [1/r0, 0] # 
true_p = [M / (true_v0*r0)^2]
tspan = (0.0, 2*pi)
t = range(tspan[1], tspan[2], length=100)

# Works! :)
function kepler!(du, u, p, t)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = G * p[1] - U # - G/c^2 * p[1] * U^2
end

problem = ODEProblem(kepler!, true_u0, tspan, true_p)

@time sol = solve(problem, Tsit5(), saveat=t)

angle_gt = sol.t 
R_gt = 1 ./ sol[1,:] # convert into Radii

noise = 0.03 * (2 * rand(Float64, size(R_gt)) .- 1)
R_gt = R_gt .+ noise

### End -------------------------------------------------- #

dV = FastChain(
    FastDense(1, 25, sigmoid),
    FastDense(25, 1)
)

otherparams = rand(Float64, 3) # otherparams = [u0, p]
ps = vcat(otherparams, initial_params(dV))

function neuralkepler!(du, u, p, t)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = G * p[1] - dV([U], p[2:end])[1]
end

prob = ODEProblem(neuralkepler!, ps[1:2], tspan, ps[3:end])

function predict(params)
    solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=t)
end

function loss(params) 
    pred = predict(params)
    sum(abs2, x-y for (x,y) in zip(pred[1,:], sol[1, :])), pred
end

# Now we tell Flux how to train the neural network
opt = ADAM(0.1, (0.85, 0.95))

cb = function(p,l,pred)
    display(l)
    display(p[1:3])
    R = 1 ./ pred[1, :] # convert into Radii
    # Plotting the potential
    x0 = Array(range(0.01, 3.0, step = 0.01))
    y0 = PotPlot.calculatepotential(1.0 ./ x0, dV, ps[4:end])
    pot_plot = plot(x0, y0)
    # Plotting the prediction and ground truth
    pred_plot = plot(R .* cos.(pred.t), R .* sin.(pred.t), xlims=(-1.5, 1.5), ylims=(-1.5, 1.5))
    traj_plot = scatter!(pred_plot, R_gt .* cos.(angle_gt), R_gt .* sin.(angle_gt))
    display(plot(traj_plot, pot_plot,layout=(2,1)))
    return false
end

@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=500)

