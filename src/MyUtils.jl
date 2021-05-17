module MyUtils
# A package to be able to plot neural networks with one input neuron
# and one output neuron.
# Also contains other helper functions for the Master thesis project.

using Flux, DiffEqFlux, QuadGK, DifferentialEquations
using DataFrames, CSV

    function predictarray(range::Array, neuralnetwork::FastChain, params::Array)
        return map(x -> neuralnetwork([x], params)[1], range)
    end

    function predictarray(range::Array, neuralnetwork::Chain)
        return map(x -> neuralnetwork([x])[1], range)
    end

    function integrateNN(neuralnetwork::FastChain, upperbound::Real, params::Array)
        integral, err = quadgk(x -> neuralnetwork([x], params), 0.0, upperbound, rtol=1e-8)
        return integral[1]
    end

    function integrateNN(neuralnetwork::Chain, upperbound::Real)
        integral, err = quadgk(x -> neuralnetwork([x]), 0.0, upperbound, rtol=1e-8)
        return integral[1]
    end
    
    function loaddata(dir::AbstractString, snfile::AbstractString, grbfile::AbstractString)
        sndatapath = joinpath(dir, snfile)
        grbdatapath = joinpath(dir, grbfile)

        sndata = CSV.read(sndatapath, delim=' ', DataFrame) # supernova data
        grbdata = CSV.read(grbdatapath, delim=' ', DataFrame) # gamma-ray bursts

        data = outerjoin(sndata,grbdata,on=[:z,:my,:me])
        uniquez = unique(data.z)

        return data, uniquez
    end

    function preparedata(data::DataFrame, uniquez::AbstractArray)
        averagedata = DataFrame(mu = Real[], me = Real[])
        for z in uniquez
            idx = findall(x -> x==z, data.z)
            avg_mu = sum([data.my[i] for i in idx]) / length(idx)
            avg_me = sum([data.me[i] for i in idx]) / length(idx)
            push!(averagedata, [avg_mu, avg_me])
        end
        return averagedata
    end

    function sampletrajectories(ODE, params::Any, initialConditions::AbstractArray, t::AbstractArray)
        tspan = (t[1], t[end])
        trajectoryList = []
        for i in 1:size(initialConditions)[1]
            u0 = initialConditions[i,:]
            problem = ODEProblem(ODE, u0, tspan, params)
            sol = solve(problem, Tsit5(), saveat=t)
            push!(trajectoryList, Array(sol))
        end
        return trajectoryList
    end

    # function to calculate the reduced χ² statistical measure 
    # for a given model prediction, groundtruth and variance/error 
    function reducedchisquared(model::AbstractArray, data::DataFrame)
        return sum(abs2, (model .- data.mu) ./ data.me)
    end

    # function to calculate the equation of state depending on the redshift
    # of a quintessence model with one scalar field
    function calculateEOS(pot::AbstractArray, dphi::AbstractArray)
        return (dphi.^2 .- 2 .* pot) ./ (dphi.^2 .+ 2 .* pot)
    end
    
end

