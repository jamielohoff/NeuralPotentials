module Qtils
using Zygote: @adjoint
"""
A package containing utilities that are used to analyze cosmological data in 
the context of quintessence and supernova data.
Also contains other helper functions for the Master thesis project.
"""

using Flux, DiffEqFlux, QuadGK, DifferentialEquations, Zygote
using DataFrames, CSV, Statistics, LinearAlgebra, Random, Distributions

"""
    Function that can be used to integrate a neural network with one input and output node.
    A neural network like this is effectively a one-dimensional function and thus can be integrated.
    The lower bound is 0 while the upper bounds can be specified,

    Arguments:
    1. `NN`: The neural network as a FastChain object.
    2. `params`: Parameters, i.e. weights and biases of the neural network.
    3. `a`: Lower bound for the integration.
    4. `b`: Upper bound for the integration.
    """
    function integrateNN(NN::FastChain, params::AbstractArray, a, b)
        integral, err = quadgk(x -> NN(x, params), a, b, rtol=1e-8)
        return integral[1]
    end
    
    @adjoint integrateNN(NN, params, a, b) = Qintegrate(NN, params, a, b), c -> (
        0.0, 
        c * Qintegrate((x,p) -> Zygote.gradient(p -> NN(x,p)[1], p)[1], params, a, b), 
        0.0, 
        0.0
    )
    
    """
    Function to load the supernova data from a given directory.
    The function also returns the a list of unique redshifts, 
    as some redshifts occur multiple times for different values of the distance modulus.

    Arguments:
    1. `dir`: The directory where the supernova data and gamma ray burst data is stored.
    2. `snfile`: File name of the supernova data.
    3. `grbfile`: File name of the gamma ray burst data.
    """
    function loaddata(dir::AbstractString, snfile::AbstractString, grbfile::AbstractString)
        sndatapath = joinpath(dir, snfile)
        grbdatapath = joinpath(dir, grbfile)

        sndata = CSV.read(sndatapath, delim=' ', DataFrame) # supernova data
        grbdata = CSV.read(grbdatapath, delim=' ', DataFrame) # gamma-ray bursts

        data = outerjoin(sndata,grbdata,on=[:z,:my,:me])
        uniquez = unique(data.z)

        rename!(data,:my => :mu)

        return data, uniquez
    end

    """
    Function to prepare the supernova and Gamma-ray burst data for further processing.
    Next to other things, it also averages over the different distance moduli that sometimes occur
    for the same redshift.

    Arguments:
    1. `data`: The dataframe that contains the supernova and gamma ray burst data.
    2. `uniquez`: The unique instanced of redshift in the dataset.
    """
    function preparedata(data::AbstractDataFrame, uniquez::AbstractArray)
        averagedata = DataFrame(mu = Real[], me = Real[])
        for z in uniquez
            idx = findall(x -> x==z, data.z)
            avg_mu = sum([data.mu[i] for i in idx]) / length(idx)
            avg_me = sum([data.me[i] for i in idx]) / length(idx)
            push!(averagedata, [avg_mu, avg_me])
        end
        return averagedata
    end

    """
    Function to calculate the reduced χ²-statistical measure 
    for a given model prediction, groundtruth and variance/error.
    The dataframe which contains the experimental data has to have two keys, :mu and :me,
    which contain the measurements and their respective standard errors.
    
    Arguments:
    1. `model`: Array that contains the model predictions.
    2. `data`: Dataframe which contains the exprimental data.
    3. `nparams`: The number of parameters of the model.
    """
    function reducedχ2(model::AbstractArray, data::DataFrame, nparams::Int)
        n = nrow(data)
        return sum(abs2, (model .- data.mu) ./ (data.me)) ./ (n - nparams)
    end

    """
    Function to calculate the χ²-statistical measure 
    for a given model prediction, groundtruth and variance/error.
    The dataframe which contains the experimental data has to have two keys, :mu and :me,
    which contain the measurements and their respective standard errors.
    
    Arguments:
    1. `model`: Array that contains the model predictions.
    2. `data`: Dataframe which contains the exprimental data.
    3. `nparams`: The number of parameters of the model.
    """
    function χ2(model::AbstractArray, data::DataFrame)
        n = nrow(data)
        return sum(abs2, (model .- data.mu) ./ (data.me))
    end

    """
    Function to calculate the equation of state depending on the redshift
    of a quintessence model with one dimensionsless scalar field Q.
    All quantities have to be dimensionless!

    Arguments:
    1. `V`: Array that contains the values of the potential for different values of the field Q at redshift z.
    2. `dQ`: Array that contains the first derivative of the scalar field ϕ with respect to the REDSHIFT z.
    3. `E`: Array that contains the values of the dimensionless Hubble expansion rate at the  given redshift z.
    4. `z`: Array that contains the redshifts z.
    """
    function calculateEOS(V::AbstractArray, dQ::AbstractArray, E::AbstractArray, z::AbstractArray)
        dq = -E.*(1 .+ z).*dQ
        return (dq.^2 .- 2 .* V) ./ (dq.^2 .+ 2 .* V)
    end

    """
    Function to check if the slow roll conditions are satisfied for a given quintessence potential.
    
    Arguments: 
    1. `NN`: Neural network which resembles the potential as a FastChain object. 
    2. `params`: Parameters of the neural network, i.e. weights and biases.
    3. `range`: The range of field values Q, where we want to test the validity of the slow roll conditions.
    4. `threshold`: The threshold below which we assume the slow roll conditions to be true.
    5. `verbose`: If set to true, the function will print and array which contains the slow roll conditions at every point defined in range.
    """
    function slowrollsatisfied(NN::FastChain, params::AbstractArray, range::AbstractArray; threshold=0.01, verbose=false)
        dNN(Q, P) = Zygote.gradient(q -> NN(q, P)[1], Q)[1]
        ddNN(Q, P) = Zygote.hessian(q -> NN(q, P)[1], Q)[1]
        V = map(q -> NN(q, params)[1], range)
        dV = map(q -> dNN(q, params)[1], range)
        ddV = map(q -> ddNN(q, params)[1], range)

        # Calculation of the slow roll parameters
        ϵ = 1/(16π) .* (dV./V).^2
        η = 1/(8π) .* (ddV ./ V)

        if verbose
            println(all(x -> x < threshold, ϵ), all(x -> x < threshold, η))
        end

        return ϵ, η # Check if there are any values above the threshold
    end

    """
    Function to check if the slow roll conditions are satisfied for a given quintessence potential.
    
    Arguments: 
    1. `F`: Function which represents the potential.
    2. `dF`: Function which represents the gradient of the potential
    3. `params`: Parameters of potential.
    4. `range`: The range of field values Q, where we want to test the validity of the slow roll conditions.
    5. `threshold`: The threshold below which we assume the slow roll conditions to be true.
    6. `verbose`: If set to true, the function will print and array which contains the slow roll conditions at every point defined in range.
    """
    function slowrollsatisfied(F, params::AbstractArray, range::AbstractArray; threshold=0.01, verbose=false)
        dF(x, p) = Zygote.gradient(x -> F(x, p), x)[1]
        ddF(x, p) = Zygote.hessian(x -> F(x, p), x)[1,1]
        V = map(x -> F(x, params), range)
        dV = map(x -> dF(x, params), range)
        ddV = map(x -> ddF(x, params), range)

        # Calculation of the slow roll parameters
        ϵ = 1/(16π) .* (dV./V).^2
        η = 1/(8π) .* (ddV./V)

        if verbose
            println(all(x -> x < threshold, ϵ), all(x -> x < threshold, abs.(η)))
        end

        return ϵ, η # Check if there are any values above the threshold
    end

    """
    Function to check if the slow roll conditions are satisfied for a given quintessence potential.
    
    Arguments: 
    1. `V`: Array which represents the potential.
    2. `NN`: Neural network which represents the gradient of the potential.
    3. `params`: Parameters of potential.
    4. `range`: The range of field values Q, where we want to test the validity of the slow roll conditions.
    5. `threshold`: The threshold below which we assume the slow roll conditions to be true.
    6. `verbose`: If set to true, the function will print and array which contains the slow roll conditions at every point defined in range.
    """
    function slowrollsatisfied(V::AbstractArray, NN, params::AbstractArray, range::AbstractArray; threshold=0.01, verbose=false)
        dV = map(x -> NN(x, params)[1], range)
        dNN(x) = Flux.gradient(x -> NN(x,params)[1], x)[1]
        ddV = map(x -> dNN(x)[1], range)

        # Calculation of the slow roll parameters
        ϵ = 1/(16pi) .* (dV./V).^2
        η = 1/(8pi) .* (ddV./V)

        if verbose
            println(ϵ, η)
        end

        return all(x -> x < threshold, ϵ), all(x -> x < threshold, η) # Check if there are any values above the threshold
    end

    """
    A function that randomly samples (ratio*dataset length) many observations
    from a sample and returns a redshift-ordered dataframe.
    
    Arguments:
    1. `df`: Dataframe containing the observations.
    2. `ratio`: Relative size of the sample compared to the whole population.
    """
    function sample(df::AbstractDataFrame, ratio::Real)
        len = trunc(Int, nrow(df)*ratio)
        return df[sort(shuffle(1:nrow(df))[1:len]), :]
    end

    """
    A function that randomly samples (ratio*dataset length) many observations
    from a sample and returns a ordered array.

    Arguments:
    1. `arr`: Array containing the observations.
    2. `ration`: Relative size of the sample compared to the whole population.
    """
    function sample(arr::AbstractArray, ratio::Real)
        len = trunc(Int, size(arr,2)*ratio)
        return arr[:,sort(shuffle(1:size(arr,2))[1:len])]
    end

    """
    Function to sample from the supernova data in such a way that for each redshift we only have
    exactly one value for the distance modulus when it is drawn.

    Arguments:
    1. `data`: Dataframe containing the supernove observations.
    2. `ratio`: Relative size of the sample compared to the whole population.
    3. `uniquez`: The unique redshifts in the data.
    """
    function elaboratesample(data::AbstractDataFrame, uniquez::AbstractArray, ratio::Real)
        # randomly choose one of the values for multiple occurences of the same redshift
        population = deepcopy(data)
        idx_list = Array([])
        for z in uniquez
            occurences = findall(x -> x == z, population.z)
            if size(occurences, 1) > 1
                idx_list = vcat(idx_list, shuffle(occurences)[2:end])
            end
        end
        delete!(population, sort(idx_list))
        return sample(population, ratio)
    end

    """
    Function to resample data with the given standard deviation, 
    assuming a gaussian distribution of each datapoint.

    Arguments:
    1. `df`: Dataframe with two fields, named :mu and :me giving the datapoint and its standard error σ.
    """
    function resample(df::AbstractDataFrame)
        resampled_df = DataFrame(z = Real[], mu = Real[], me = Real[])
        for idx in 1:nrow(df)
            z = df[idx, :z]
            σ = df[idx, :me]
            pdf = Normal(df[idx, :mu], σ)
            μ =  rand(pdf, 1)[1]
            push!(resampled_df, [z, μ, σ])
        end
        return resampled_df
    end

    """
    Function that calculates the per-column quantiles of a stacked dataframe of 
    different bootstrap experiments.

    Arguments: 
    1. `data`: Dataframe of stacked arrays.
    2. `bounds`: Array containing all the quantiles we want to know.
    """
    function quantiles(data; bounds=[0.025, 0.975])
        itm = data[1]
        for i in 2:size(data,1)
            itm = hcat(itm, data[i])
        end
        qtl = Float64[0,0]
        for j in 1:size(itm, 1)
            qtl = hcat(qtl, quantile(itm[j,:], bounds))
        end
        return qtl[:,2:end]
    end

    """
    Function to calculate the column-like statistics of a field of a dataframe.
    It returns the rowlike mean, standard deviation and 5% confidence intervals.

    Arguments:
    1. `input`: The column of the dataframe from which we want to know the statistics.
    """
    function calculatestatistics(input)
        μ = mean(input)
        σ = stdm(input, μ)

        qtls = quantiles(input)

        lower_CI = qtls[1,:] - μ
        upper_CI = qtls[2,:] - μ

        return μ, σ, (-lower_CI, upper_CI)
    end

    """
    TODO Docstring
    """
    function calculate_cov_map(covm::AbstractArray, std::AbstractArray, scale::Real=1.0)
        map = zeros(size(covm))
        for i ∈ 1:length(std)
            for j ∈ 1:length(std)
                if std[i]*std[j] == 0
                    map[i,j] = 0.0
                else
                    map[i,j] = asinh(covm[i,j]/scale) # /(std_V[i]*std_V[j])
                end
            end
        end
        return map
    end
end

