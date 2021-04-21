module MyUtils
# A package to be able to plot neural networks with one input neuron
# and one output neuron.
# Also contains other helper functions for the Master thesis project.

using Flux, DiffEqFlux, QuadGK
using DataFrames, CSV

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
    
    function loaddata(dir, snfile, grbfile)
        sndatapath = joinpath(dir, snfile)
        grbdatapath = joinpath(dir, grbfile)

        sndata = CSV.read(sndatapath, delim=' ', DataFrame) # supernova data
        grbdata = CSV.read(grbdatapath, delim=' ', DataFrame) # gamma-ray bursts

        data = outerjoin(sndata,grbdata,on=[:z,:my,:me])
        uniquez = unique(data.z)

        return data, uniquez
    end
    
end

