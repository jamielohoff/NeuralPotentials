using DifferentialEquations, Plots, Flux, DiffEqFlux

c = 1 # AU per year
G = 0.5 # gravitational constant in new units : AU^3 * yr^-2 * M^-1
M = 1 # measure mass in multiples of central mass
rs = 1 # Schwarzschild radius in AU

### Creation of synthethetic data --------------------------------- #

r0 = 4*rs # measure radius in multiples of r0, i.e. unit of length is AU
year = 1 # measure time in years
v0 = 0.6 * sqrt(G*M/r0)/sqrt(1-rs/r0)# 1.26
u0 =  [1/r0, 0] # 

p = [M / (v0*r0)^2]
tspan = (0.0, 10*pi)
t = range(tspan[1], tspan[2], length=500)

# Works! :)
function apsisrotation!(du, u, p, t)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = G * p[1] * (1 - U^2) - U
end

problem = ODEProblem(apsisrotation!, u0, tspan, p)

@time sol = solve(problem, Tsit5(), saveat=t)

angle_gt = sol.t 
R_gt = 1 ./ sol[1,:] # convert into Radii

noise = 0.02 * (2 * rand(Float64, size(R_gt)) .- 1)
R_gt = R_gt .+ noise

plot(R_gt .* cos.(angle_gt), R_gt .* sin.(angle_gt))

### End ----------------------------------------------------------- #

ps = rand(Float64, 3) # otherparams = [u0, p]

prob = ODEProblem(apsisrotation!, ps[1:2], tspan, ps[3:end])

function predict(params)
    solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=t)
end

function loss(params) 
    pred = predict(params)
    sum(abs2, x-y for (x,y) in zip(pred[1,:], sol[1, :])), pred
end

# Now we tell Flux how to train the neural network
opt = ADAM(0.2, (0.85, 0.95))

cb = function(p,l,pred)
    display(l)
    display(p[1:3])
    R = 1 ./ pred[1, :] # convert into Radii
    display(plot(R .* cos.(pred.t), R .* sin.(pred.t), xlims=(-4.0, 4.0), ylims=(-4.0, 4.0)))
    display(scatter!(R_gt .* cos.(angle_gt), R_gt .* sin.(angle_gt)))
    l < 0.01
end

@time res = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=1000)

