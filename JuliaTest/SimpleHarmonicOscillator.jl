using DifferentialEquations
using Plots
using Flux
using DiffEqFlux

omega = 1.0
beta = 0.1
tspan = (0.0, 20.0)
u0 = [0, 1.0]

p = [omega, beta]

# inplace version of harmonical oscillator
function simpleoscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]
    du[1] = dx
    du[2] = -p[1] * x - p[2] * dx
end

problem = ODEProblem(simpleoscillator!, u0, tspan, p)

sol = solve(problem, Tsit5(), saveat=0.1, reltol=1e-8, abstol=1e-8)

data_batch = sol[1,:]
noise = 0.1 * (2 * rand(Float64, size(data_batch)) .- 1)
data_batch = data_batch .+ noise
t = sol.t

u0 = rand(Float64, 2)
p = rand(Float64, 2)
params = Flux.params(u0, p)

function predict()
     solve(problem, Tsit5(), u0=u0, p=p, saveat=0.1, reltol=1e-8, abstol=1e-8)[1, :]
end

function loss() 
    sum(abs2, x-y for (x,y) in zip(predict(), data_batch))
end

# Now we tell Flux how to train the neural network
data = Iterators.repeated((), 200)
opt = ADAM(0.2)
cb = function ()
    display(loss())
    s = solve(remake(problem, u0=u0, p=p), Tsit5(), saveat=0.1, reltol=1e-8, abstol=1e-8)
    display(plot(s.t, s[1, :], ylim=(-1,1)))
    display(scatter!(t, data_batch, ylim=(-1,1)))
end


cb()
@time Flux.train!(loss, params, data, opt, cb=cb)

println(Flux.params(u0, p))

