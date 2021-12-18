using DifferentialEquations, Flux, DiffEqFlux
using Plots, ProgressBars, Printf
include("../lib/Qtils.jl")
using .Qtils
using SymbolicUtils
using SymbolicRegression

### Creation of synthethetic data----------------------- #
tspan = (-2π, 2π)
data_t = range(tspan[1], tspan[2], length=100)
noise = 0.2 * (2 * rand(Float64, size(data_t)) .- 1)
data_batch = cos.(2 .* data_t) .+ data_t + noise
### End ------------------------------------------------ #

# Defining the neural network model
model = Chain(
    Dense(1, 8, tanh),
    Dense(8, 8, tanh), 
    Dense(8, 1)
)

# Defining the loss function
function loss()
    idx = rand(1:length(data_batch))[1]
    pred = model([data_t[idx]])[1]
    return (data_batch[idx] - pred)^2
end

ps = Flux.params(model)

# Callback function
cb = function(loss, ProgBar)
    set_description(ProgBar, string(@sprintf("Loss: %.2f", loss)))
end

function stochastic_train!(lossfunction, params, maxiters, batchsize, opt; cb)
    # implementation of stochstic gradient descent
    steps = ProgressBar(1:maxiters)
    for step in steps
        loss = 0.0
        gradslist = []
        # Calculate gradients for minibatch
        for batch in 1:batchsize
            grads = Flux.gradient(params) do
                _loss = lossfunction()
                loss += _loss
                return _loss
            end
            push!(gradslist, grads)
        end

        # Create statistical average
        avggrads = Dict()
        for p in params
            gs = gradslist[1][p]
            for i in 2:batchsize
                gs .+= gradslist[i][p]
            end
            avggrads[p] = gs ./ batchsize 
        end
        # Update parameters with averaged gradient
        Flux.update!(opt, params, avggrads)
        cb(loss, steps)
    end
end

@time stochastic_train!(loss, ps, 5000, 25, ADAM(0.1); cb)

x0 = Array(data_t)
X0 = reshape(x0, 1, :)

@time y0 = Array{Float64}(MyUtils.predictarray(x0, model))

options = SymbolicRegression.Options(
    binary_operators=(+, *, /, -),
    unary_operators=(cos, sin),
    npopulations=20
)

hallOfFame = EquationSearch(X0, y0, niterations=5, options=options, numprocs=4)
dominating = calculateParetoFrontier(X0, y0, hallOfFame, options)
eqn = node_to_symbolic(dominating[end].tree, options)
println("Resulting equation: ", simplify(eqn))

Plots.plot(x0, y0)
Plots.scatter!(x0, data_batch)

