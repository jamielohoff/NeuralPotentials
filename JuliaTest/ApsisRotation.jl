using DifferentialEquations
using Plots
using Flux
using DiffEqFlux

c = 1 # AU per year
G = 0.5 # gravitational constant in new units : AU^3 * yr^-2 * M^-1
M = 1 # measure mass in multiples of central mass
rs = 1 # Schwarzschild radius in AU

r0 = 4*rs # measure radius in multiples of r0, i.e. unit of length is AU
year = 1 # measure time in years
v0 = 0.6 * sqrt(G*M/r0)/sqrt(1-rs/r0)# 1.26
u0 =  [1/r0, 0] # 

p = [M / (v0*r0)^2]
tspan = (0.0, 10*pi)

# Works! :)
function apsisrotation!(du, u, p, t)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = G * p[1] * (1 - U^2) - U
end

problem = ODEProblem(apsisrotation!, u0, tspan, p)

@time sol = solve(problem, Tsit5(), saveat=0.1, reltol=1e-8, abstol=1e-8)

angle_gt = sol.t 
R_gt = 1 ./ sol[1,:] # convert into Radii

noise = 0.02 * (2 * rand(Float64, size(R_gt)) .- 1)
R_gt = R_gt .+ noise

plot(R_gt .* cos.(angle_gt), R_gt .* sin.(angle_gt))

u0 = rand(Float64, 2)
p = rand(Float64, 1)
params = Flux.params(u0, p)

function predict()
     solve(problem, Tsit5(), u0=u0, p=p, saveat=0.1, reltol=1e-8, abstol=1e-8)[1, :]
end

function loss() 
    sum(abs2, x-y for (x,y) in zip(predict(), sol[1, :]))
end
# Now we tell Flux how to train the neural network
data = Iterators.repeated((), 200)
opt = ADAM(0.05)

cb = function()
    display(loss())
    display(params)
    s = solve(remake(problem, u0=u0, p=p), Tsit5(), saveat=0.1, reltol=1e-8, abstol=1e-8)
    R = 1 ./ s[1, :] # convert into Radii
    display(plot(R .* cos.(s.t), R .* sin.(s.t), xlims=(-4,4), ylims=(-4, 4)))
    display(scatter!(R_gt .* cos.(angle_gt), R_gt .* sin.(angle_gt)))
end

cb()
@time Flux.train!(loss, params, data, opt, cb=cb)

println(params)
println("Ground truth: ", [1/r0, 0, M / (v0*r0)^2])

