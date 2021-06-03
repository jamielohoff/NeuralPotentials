module MechanicsDatasets

    using DifferentialEquations, Flux, DiffEqFlux
    """
    Function that solves the 1-dimensional 2nd order ODE for a given potential V
    """
    function potentialproblem(V::Function, 
        u0::AbstractArray, 
        p::AbstractArray, 
        t::Array; 
        addnoise=false, 
        noisescale=0.01f0)
        dV(x,p) = Flux.gradient(x -> V(x,p)[1], x)[1]

        function potentialODE!(du, u, p, t)
            x = u[1]
            dx = u[2]

            du[1] = dx
            du[2] = -dV(x, p)
        end

        problem = ODEProblem(potentialODE!, u0, (t[1],t[end]), p)
        solution = solve(problem, Tsit5(), saveat=t)

        if addnoise
            noise =  noisescale * randn(Float32, size(solution)) # add normally distributed noise
            data = solution[1:2,:] .+ noise
        end
        return vcat(reshape(t,1,:),data)
    end
    

    """
    Function that plots the groundtruth potential and the predicted potential together with the trajectory.
    """
    function potentialplotter(prediction::AbstractArray, 
        goundtruth::AbstractArray, 
        dNN::FastChain, 
        potential::Function,
        dimensions::Array)

    end
end