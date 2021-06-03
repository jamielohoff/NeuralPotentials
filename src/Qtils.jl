module Qtils
# A package to be able to plot neural networks with one input neuron
# and one output neuron.
# Also contains other helper functions for the Master thesis project.

using Flux, DiffEqFlux, QuadGK, DifferentialEquations
using DataFrames, CSV, Statistics, LinearAlgebra

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

    function calculateslowroll(NN::FastChain, p::AbstractArray, range::AbstractArray, G::Real)
        κ_squared = 8*pi*G
        dV = map(x -> NN(x, p)[1], range)
        dNN(x) = Flux.gradient(x -> NN(x,p)[1], x)[1]
        ddV = map(x -> dNN(x)[1], range)
        V = map(x -> Qtils.integrateNN(NN, x, p), range)

        ϵ = 1/(2*κ_squared) .* (dV./V).^2
        η = 1/κ_squared .* (ddV ./ V)

        return ϵ, η
    end

    """
    A function that randomly samples (ratio*dataset length) many observations
    from a sample and returns a ordered dataframe.
    """
    function randomsample(df::DataFrame, ratio::Real)
        len = trunc(Int, nrow(df)*(1-ratio))
        for _ in 1:len
            idx = rand(1:nrow(df))
            delete!(df, idx)
        end
        return df
    end

    """
    A function that randomly samples (ratio*dataset length) many observations
    from a sample and returns a ordered dataframe.
    """
    function randomsample(df::AbstractDataFrame, ratio::Real)
        len = trunc(Int, nrow(df)*(1-ratio))
        for _ in 1:len
            idx = rand(1:nrow(df))
            delete!(df, idx)
        end
        return df
    end

    """
    A function that randomly samples (ratio*dataset length) many observations
    from a sample and returns a ordered array.
    """
    function randomsample(arr::AbstractArray, ratio::Real)
        len = trunc(Int, size(arr,2)*(1-ratio))
        for _ in 1:len
            idx = rand(1:size(arr,2))
            arr = arr[:, 1:end .!= idx]
        end
        return arr
    end

    """
    Function to plot rows from a 2dim array for a given range
    """
    function multiplot(plotobject, range::AbstractArray, list::AbstractArray)
        for entry in list
            plot!(plotobject, range, entry)
        end
        return plotobject
    end
end
