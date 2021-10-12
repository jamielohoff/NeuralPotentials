module MechanicsDatasets
using DifferentialEquations, Flux, DiffEqFlux, Distributions, Zygote
using Statistics, DataFrames
    """
    Function that solves the 1-dimensional 2nd order ODE for a given potential V
    This function is also able to add gaussian noise to the data, 
    but all columns have the same variance.

    Arguments: 
    1. `V`: Potential of the problem.
    2. `u0`: Initial conditions of the ODE.
    3. `p`: Array of parameters for the gradient
    4. `t`: Array of timepoints t where we want to know the solution of the potential problem
    5. `addnoise`: Boolean variable to quantify whether we want to add gaussian noise or not.
    6. `σ`: Standard deviation of the gaussian noise.
    """
    function potentialproblem1D(V::Function, 
                                u0::AbstractArray, 
                                p::AbstractArray, 
                                t::AbstractArray; 
                                addnoise=false, 
                                σ=0.01)

        dV(x,p) = Zygote.gradient(x -> V(x,p)[1], x)[1]

        function potentialODE!(du, u, p, t)
            x = u[1]
            dx = u[2]

            du[1] = dx
            du[2] = -dV(x, p)
        end

        problem = ODEProblem(potentialODE!, u0, (t[1],t[end]), p)
        solution = solve(problem, Tsit5(), saveat=t)

        # add normally distributed noise
        if addnoise
            pdf = Normal(0.0, σ)
            noise =  rand(pdf, size(solution))
            solution = solution .+ noise
        end
        return vcat(reshape(t,1,:), solution)
    end

    """
    Function that solves the kepler problem for a given potential.
    This function is also able to add gaussian noise to the radial coordinate.

    Arguments: 
    1. `dV`: Gradient of the potential.
    2. `u0`: Initial conditions of the ODE.
    3. `p`: Array of parameters for the gradient of the potential.
    4. `ϕ`: Array containing the polar angles where we want to know the solution of the Kepler problem.
    5. `addnoise`: Boolean variable to quantify whether we want to 
                    add gaussian noise to the radial coordinate or not.
    6. `σ`: Variance of the gaussian noise.
    """
    function keplerproblem(dV::Function, 
                            u0::AbstractArray, 
                            p::AbstractArray, 
                            ϕ::AbstractArray; 
                            addnoise=false, 
                            σ=0.01)

        function kepler!(du, u, p, ϕ)
            U = u[1]
            dU = u[2]
            t = u[3]

            du[1] = dU
            du[2] = dV(U, p)[1] - U
            du[3] = p[1]/(p[2]U^2)
        end

        ϕspan = (0.0, 10π) # (ϕ[1], ϕ[end])
        problem = ODEProblem(kepler!, u0, ϕspan, p)

        @time solution = Array(solve(problem, Tsit5(), u0=u0, p=p, saveat=ϕ))

        if addnoise
            pdf = Normal(0.0, σ)
            noise = rand(pdf, size(solution[1,:]))
            x = cos.(ϕ)./solution[1,:] .+ noise
            y = sin.(ϕ)./solution[2,:] .+ noise
            r = sqrt.(x.^2 .+ y.^2)
            ϕ = mod.(atan.(x, y), 2π)
            t = solution[3,:]
            x_err = σ .* ones(size(r))
            y_err = σ .* ones(size(r))
            return DataFrame(hcat(r,ϕ,t,x_err, y_err), [:r, :ϕ, :t, :x_err, :y_err])
        else
            r = 1.0./solution[1,:]
            t = solution[3,:]
            return DataFrame(hcat(r,ϕ,t), [:r, :ϕ, :t])
        end
    end

    """
    Function designed to sample trajectories from the same potential with 
    different initial conditions and parameters.
    
    """
    function sampletrajectories(V::Function, 
                                params::AbstractArray, 
                                initialConditions::AbstractArray, 
                                t::AbstractArray; 
                                addnoise=true, 
                                σ=0.01)
        trajectoryList = []
        for i in 1:size(initialConditions)[1]
            u0 = initialConditions[i,:]
            p = params[i,:]
            sol = potentialproblem1D(V, u0, p, t, addnoise=addnoise, σ=σ)
            push!(trajectoryList, Array(sol))
        end
        return trajectoryList
    end

    """
    Function to calculate the reduced χ² statistical measure 
    for a given model prediction, groundtruth and a fixed standard error/deviation.
    The array which contains the experimental data can have multiple rows, 
    but of course the number of rows of the data should equal the number of 
    rows of the model output.
    Also, the data in all rows should have the same variance σ.
    
    Arguments:
    1. `model`: Array that contains the model predictions.
    2. `data`: Dataframe which contains the exprimental data.
    3. `nparams`: The number of parameters of the model.
    4. `σ`: Standard error/deviation of the datapoints.
    """
    function reducedχ2(model::AbstractArray, data::AbstractArray, nparams::Number, σ::Number)
        n = size(data,2)
        return sum(abs2, (model .- data) ./ σ) ./ (n - nparams)
    end
end