using DifferentialEquations, Plots, Flux, DiffEqFlux

c = 63241 # AU per year
G = 39.478 # gravitational constant in new units : AU^3 * yr^-2 * M^-1


m = (5.9722 / 1.98847) * 10^(-6) # 
M = 1 # measure mass in multiples of central mass
r0 = 1 # measure radius in multiples of r0, i.e. unit of length is AU
year = 1 # measure time in years

### Creation of synthethetic data ------------------------ #

true_v0 = 2*pi*r0 / year # intial velocity
true_u0 =  [1/r0, 0.0] # 
true_p = [M / (true_v0*r0)^2]
ϕspan = (0.0, 2π)
ϕ = range(ϕspan[1], ϕspan[2], length=100)

# Works! :)
function kepler!(du, u, p, ϕ)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = G*p[1] - U # - G/c^2 * p[1] * U^2
end

problem = ODEProblem(kepler!, true_u0, ϕspan, true_p)

@time data = solve(problem, Tsit5(), saveat=ϕ)

R_gt = 1 ./ data[1,:] # convert into Radii

# noise = 0.02 * (2 * rand(Float64, size(R_gt)) .- 1)
# R_gt = R_gt .+ noise

### End -------------------------------------------------- #

ps = rand(Float64, 3) # otherparams = [u0, p]

prob = ODEProblem(kepler!, ps[1:2], ϕspan, ps[3:end])

function predict(params)
    return Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=ϕ))
end

function loss(params) 
    pred = predict(params)
    return sum((pred .- data).^2), pred
end

# Now we tell Flux how to train the neural network
opt = ADAM(0.2)

cb = function(p,l,pred)
    println("Loss:", l)
    println("Parameters", p[1:3])
    R = 1 ./ pred[1, :] # convert into Radii
    orbit_plot = plot(R .* cos.(ϕ), R .* sin.(ϕ), xlims=(-1.5, 1.5), ylims=(-1.5, 1.5))
    orbit_plot = scatter!(orbit_plot, R_gt .* cos.(ϕ), R_gt .* sin.(angle_gt))
    display(plot(orbit_plot))
    return l < 0.01
end

@time res = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=1000)

