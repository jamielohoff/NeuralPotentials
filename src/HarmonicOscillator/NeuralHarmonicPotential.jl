using DifferentialEquations, Flux, DiffEqFlux
using Plots
include("../MyUtils.jl")
using .MyUtils

### Creation of synthethetic data---------------------- #
omega = 1.0f0
beta = 0.0f0
tspan = (0.0f0, 5.0f0)
t = Array{Float32}(range(tspan[1], tspan[2], length=50))
u0 = [0.0f0, 6.0f0] # contains x0 and dx0
p = [omega]

V0(x,p) = p[1]*x^4
dV0(x,p) = Flux.gradient(y -> V0(y,p)[1], x)[1]

function oscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -dV0(x, p[1:end])[1] # -p[1] * dx
end
problem = ODEProblem(oscillator!, u0, tspan, p)
@time true_sol = solve(problem, Tsit5(), saveat=t)

data_batch = true_sol[1,:]
noise = 0.0f0 # 0.1 * (2 * rand(Float32, size(data_batch)) .- 1)
data_batch = Float32.(data_batch .+ noise)
data_t = Float32.(true_sol.t)

### End ------------------------------------------------ #

# Defining the gradient of the potential
V = FastChain(
    FastDense(1, 10, sigmoid), 
    FastDense(10, 1)
)
dV(x, p) = Flux.gradient(y -> V(y,p)[1], x)[1]

otherparams = rand(Float32, 2) .+ [1.5f0, 0.0f0] # contains u0, du0 and beta
p = vcat(otherparams, initial_params(V))

function neuraloscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -dV(x,p[1:end])[1] # - p[1] * dx 
end

prob = ODEProblem(neuraloscillator!, u0, tspan, p[3:end])


function predict(params)
    return Array(solve(prob, Tsit5(), p=params[3:end], saveat=t))
end

function loss(params)
    pred = predict(params)
    return sum(abs2, data_batch .- pred[1, :]), pred
end

cb = function(p,l,pred)
    display(l)
    display(p[1:2])
    display(Plots.plot(data_t, pred[1, :], ylim=(-4,4)))
    display(Plots.scatter!(data_t, data_batch))
    return l < 0.01
end

@time result = DiffEqFlux.sciml_train(loss, p, ADAM(0.1), cb=cb, maxiters=5000)
println("Best result: ", result.minimizer)

x0 = Array(range(-5.0f0, 5.0f0, step = 0.1f0))
y0 = map(x -> V(x, result.minimizer[3:end])[1], x0)
z0 = map(x -> V0(x, [omega]), x0)

Plots.plot(x0, y0)
Plots.plot!(x0, z0)

