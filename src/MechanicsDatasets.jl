module MechanicsDatasets

    using Flux: Zygote
using DifferentialEquations, Flux, DiffEqFlux, Distributions, Zygote
    """
    Function that solves the 1-dimensional 2nd order ODE for a given potential V
    """
    function potentialproblem(V::Function, 
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
            data = solution[1:2,:] .+ noise
        end
        return vcat(reshape(t,1,:),data)
    end

    """
    Function designed to sample 
    """
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
end