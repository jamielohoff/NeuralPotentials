using DifferentialEquations
using Plots
using Flux
using DiffEqFlux

c = 63241 # AU per year
G = 39.478 # gravitational constant in new units : AU^3 * yr^-2 * M^-1


m = (5.9722 / 1.98847) * 10^(-6) # 
M = 1 # measure mass in multiples of central mass
r0 = 1 # measure radius in multiples of r0, i.e. unit of length is AU
year = 1 # measure time in years
v0 = 2*pi*r0 / year # intial velocity
u0 =  [1/r0, 0] # 

p = [M / (v0*r0)^2]
tspan = (0.0, 2*pi)

# Works! :)
function angularkepler!(du, u, p, t)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = G * p[1] - U # - G/c^2 * p[1] * U^2
end



problem = ODEProblem(angularkepler!, u0, tspan, p)

@time sol = solve(problem, Tsit5(), saveat=0.1, reltol=1e-8, abstol=1e-8)

angle = sol.t 
R_gt = 1 ./ sol[1,:] # convert into Radii

noise = 0.02 * (2 * rand(Float64, size(R_gt)) .- 1)
R_gt = R_gt .+ noise

plot(R_gt .* cos.(angle), R_gt .* sin.(angle))

u0 = rand(Float64, 2)
p = rand(Float64, 1)
params = Flux.params(u0, p)

function predict()
     solve(problem, Tsit5(), u0=u0, p=p, saveat=0.1)[1, :]
end

function loss() 
    sum(abs2, x-y for (x,y) in zip(predict(), sol[1, :]))
end

# Now we tell Flux how to train the neural network
data = Iterators.repeated((), 200)
opt = ADAM(0.2)

cb = function()
    display(loss())
    display(params)
    s = solve(remake(problem, u0=u0,p=p), Tsit5(), saveat=0.1, reltol=1e-8, abstol=1e-8)
    R = 1 ./ s[1, :] # convert into Radii
    display(plot(R .* cos.(s.t), R .* sin.(s.t), xlims=(-1.5, 1.5), ylims=(-1.5, 1.5)))
    display(scatter!(R_gt .* cos.(angle), R_gt .* sin.(angle)))
end

cb()
@time Flux.train!(loss, params, data, opt, cb=cb)

println(params)

