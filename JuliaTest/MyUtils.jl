module MyUtils

using Flux, DiffEqFlux, Plots, QuadGK

# A package to be able to plot neural networks with one input neuron
# and one output neuron. Usually used to plot gradients and potentials
# that are approximated through neural networks

function predictarray(range::Array, neuralnetwork::FastChain, params::Array)
    return map(x -> neuralnetwork([x], params)[1], range)
end

function predictarray(range::Array, neuralnetwork::Chain)
    return map(x -> neuralnetwork([x])[1], range)
end

function integrateNN(upperbound::Real, neuralnetwork::FastChain, params::Array)
    integral, err = quadgk(x -> neuralnetwork([x], params), 0.0, upperbound, rtol=1e-8)
    return integral[1]
end

function integrateNN(upperbound::Real, neuralnetwork::Chain)
    integral, err = quadgk(x -> neuralnetwork([x]), 0.0, upperbound, rtol=1e-8)
    return integral[1]
end

function calculatepotential(range::Array, neuralnetwork::FastChain, params::Array)
    return map(x -> integrateNN(x, neuralnetwork, params), range)
end

function calculatepotential(range::Array, neuralnetwork::Chain)
    return map(x -> integrateNN(x, neuralnetwork), range)
end

end

