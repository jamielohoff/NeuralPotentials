using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, ProgressBars, Printf
include("../MyUtils.jl")
using .MyUtils

### Creation of synthethetic data---------------------- #
const g = 9.81
l = 2.0
beta = 0.0
tspan = (0.0, 10.0)
t = range(tspan[1], tspan[2], length=500)
u0 = [-3.1, 0.0]
p = vcat(u0, [beta, l])

function pendulum!(du, u, p, t)
    phi = u[1]
    dphi = u[2]

    du[1] = dphi
    du[2] = -p[1] * dphi - g/p[2] * sin(phi) # phi^4, polynome fitten
end
problem = ODEProblem(pendulum!, p[1:2], tspan, p[3:end])
true_sol = solve(problem, Tsit5(), saveat=t)

data_batch = true_sol[1,:] 
noise = 0.0 # 0.1 * (2 * rand(Float64, size(data_batch)) .- 1)
data_batch = Float32.(data_batch .+ noise)
data_t = Float32.(true_sol.t)

### End ------------------------------------------------ #

# Defining the gradient of the potential
dV = FastChain(
    FastDense(1, 20, tanh), 
    FastDense(20, 1)
)

boundaryconditions = rand(Float64, 2) # contains u0, du0 and beta
p = vcat(boundaryconditions, initial_params(dV))

function neuralpendulum!(du, u, p, t)
    phi = u[1]
    dphi = u[2]

    du[1] = dphi
    du[2] = -dV([phi], p)[1]
end

problem = ODEProblem(neuralpendulum!, p[1:2], tspan, p[3:end])

function loss(params)
    dbatch = nothing
    tbatch = nothing
    Zygote.ignore() do 
        idx = unique(sort(rand(1:length(t), 50)))
        tbatch = map(i -> t[i], idx)
        dbatch = map(i -> data_batch[i], idx)
    end
    pred = solve(problem, Tsit5(), u0=params[1:2], p=params[3:end], saveat=tbatch)
    sum(abs2, dbatch .- pred[1, :])
end

# Callback function
cb = function(loss, params, ProgBar)
    # display(params[1:2])
    set_description(ProgBar, string(@sprintf("Loss: %.2f", loss)))
end

function stochastic_train!(lossfunction, params, maxiters, batchsize, opt; cb)
    # implementation of stochstic gradient descent
    steps = ProgressBar(1:maxiters)
    for step in steps
        loss = 0.0
        gradslist = zeros(length(params))
        # Calculate gradients for minibatch
        for batch in 1:batchsize
            _loss = lossfunction(params)
            grads = Zygote.gradient(ps -> lossfunction(ps), params)[1]
            _loss, back = Zygote.pullback(ps -> lossfunction(ps), params)
            println(back(one(_loss))[1])
            println(grads)
            loss += _loss
            gradslist .+= grads
        end

        # Create statistical average
        _grads = gradslist ./ batchsize
        avggrads = Dict(params => _grads)
        # Update parameters with averaged gradient
        Flux.update!(opt, Flux.params(params), avggrads)
        cb(loss / batchsize, params, steps)
    end
end

@time stochastic_train!(loss, p, 1000, 5, ADAM(0.1); cb)

x0 = Array(range(-3.5, 3.5, step=0.1))
y0 = MyUtils.calculatepotential(x0, dV, p)
z0 = g/l*(1 .- cos.(x0))

Plots.plot(x0, y0)
Plots.plot!(x0, z0)

