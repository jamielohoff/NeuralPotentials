module PotPlot

using Flux, DiffEqFlux, Plots, QuadGK

# A package to be able to plot neural networks with one input neuron
# and one output neuron. Usually used to plot gradients and potentials
# that are approximated through neural networks

dV = FastChain(
    FastDense(1, 10, sigmoid), 
    FastDense(10, 1),
)

p = initial_params(dV)

function calculategradient(range::Array, neuralnetwork::FastChain, p::Array)
    y = zeros(0)
    for x in range
        val = neuralnetwork([x], p)[1]
        push!(y, val)
    end
    return y
end

function integrateNN(upperbound::Real, neuralnetwork::FastChain, p::Array)
    # time is measured in Gyr
    integral, err = quadgk(x -> neuralnetwork([x], p), 0.0, upperbound, rtol=1e-8)
    return integral[1]
end

function calculatepotential(range::Array, neuralnetwork::FastChain, p::Array)
    y = zeros(0)
    for x in range
        val = integrateNN(x, neuralnetwork, p)
        push!(y, val)
    end
    return y
end

end

