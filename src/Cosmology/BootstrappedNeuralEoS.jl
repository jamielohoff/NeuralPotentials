using DifferentialEquations: tanh
using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics, Measures, LaTeXStrings
include("../lib/Qtils.jl")
include("../lib/AwesomeTheme.jl")
using .Qtils

# Load predefined plot theme and adjust font sizes
theme(:awesome)
resetfontsizes()
scalefontsizes(2)

# Load supernova Ia
data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")

# Natural constants
const H0 = 0.06766 # in 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.475e-53 # in Mpc^3 / (Gyr^2 * planck mass)

zspan = (0.0, 7.0)

# Priors from PLANCK collaboration1
Ω_m_0 = 0.3111

mu(z, χ) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* χ)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

# Definition of our custom activation function
boundrelu(x) = min.(max.(x, 0.0), 1.0)

# Plot supvernova Ia data
μ_plot = scatter(
            data.z, data.mu, 
            title="Redshift-Luminosity Data",
            xlabel=L"\textrm{Redshift } z",
            ylabel=L"\textrm{Distance modulus } \mu",
            yerror=data.me,
            label="Supernova and Gamma-ray burst data",
            legend=:bottomright,
)

### Bootstrap Loop
repetitions = 1024
itmlist = DataFrame(params = Array[], Ω = Array[], μ = Array[], EoS = Array[])
println("Beginning Bootstrap...")
lk = ReentrantLock()
@time Threads.@threads for rep in 1:repetitions
    println("Starting repetition ", rep, " at thread ID ", Threads.threadid())

    # Sample supernova Ia data from the dataset
    df = Qtils.elaboratesample(data, uniquez, 1.0)
    sampledata = Qtils.resample(df)

    # Defining the time-dependent equation of state as neural network
    w_DE = FastChain(
        FastDense(1, 8, boundrelu),
        FastDense(8, 8, boundrelu),
        FastDense(8, 1) 
    )

    # Initialize the parameters as well as the weights and biases of the neural network
    ps = vcat(0.15 .+ 0.85 .* rand(Float32, 1), initial_params(w_DE))
    # ps = vcat(Ω_m_0, initial_params(w_DE))

    # 1st order ODE for Friedmann equation in terms of z
    function friedmann!(du,u,p,z)
        Ω_m = u[1]
        E = u[2]
        χ = u[3]

        Ω_DE = 1.0 - Ω_m 
        dE = 1.5*E/(1+z) * (Ω_m + (1.0 + w_DE(z, p)[1])*Ω_DE)
        
        du[1] = (3.0/(1+z) - 2.0*dE/E) * Ω_m
        du[2] = dE
        du[3] = 1.0/E
    end

    # Defining the problem and optimizer
    u0 = [Ω_m_0, 1.0, 0.0]
    problem = ODEProblem(friedmann!, u0, zspan, ps)
    opt = NADAM(1e-3)

    # Function that predicts the results for a given set of parameters by solving the ODE at the given redshifts
    function predict(params)
        u0 = [params[1], 1.0, 0.0]
        # u0 = [Ω_m_0, 1.0, 0.0]
        return Array(solve(problem, Tsit5(), u0=u0, p=params[2:end], saveat=sampledata.z))
    end
    
    # Function that calculates the loss with respect to the observed data
    function loss(params)
        pred = predict(params)
        µ = Qtils.mu(sampledata.z, pred[end,:])
        return Qtils.χ2(μ, sampledata), pred
    end
    
    # Callback function 
    cb = function(p, l, pred)
        return false
    end

    # Start the training of the model
    @time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=1000)

    # Use the best result, i.e. the one with the lowest loss and compute the equation of state etc. for it
    u0 = [result.minimizer[1], 1.0, 0.0]
    res = Array(solve(problem, Tsit5(), u0=u0, p=result.minimizer[2:end], saveat=uniquez))
    EoS = [w_DE(z, result.minimizer[2:end])[1] for z ∈ uniquez]

    # Push the result into the array
    lock(lk)
    push!(itmlist, [[result.minimizer[1]], res[1,:], Qtils.mu(uniquez,res[end,:]), EoS])
    unlock(lk)
    println("Repetition ", rep, " is done!")
end
println("Bootstrap complete!")

# Calculate the mean, standard deviation and 95% confidence intervals for the quantities of interest
mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_μ, std_μ, CI_μ = Qtils.calculatestatistics(itmlist.μ)
mean_EoS, std_EoS, CI_EoS = Qtils.calculatestatistics(itmlist.EoS)
mean_Ω, std_Ω, CI_Ω = Qtils.calculatestatistics(itmlist.Ω)

println("Cosmological parameters: ")
println("Mass parameter Ω_m = ", mean_params[1], " ± ", std_params[1])

# Plot all the stuff into a 3x1 figure
μ_plot = plot!(μ_plot, uniquez, mean_μ, ribbon=CI_μ, 
                label="Prediction of the neural equation of state"
)
EoS_plot = plot(uniquez, mean_EoS, ribbon=CI_EoS, 
                title="Equation of State", 
                xlabel=L"\textrm{Redshift } z", 
                ylabel=L"\textrm{Equation of state } w",
                label="Prediction of the equation of state",
                legend=:topright,
                color=colorant"#c67", # rose
)
Ω_plot = plot(uniquez, mean_Ω, ribbon=CI_Ω, 
                title="Density Evolution", 
                xlabel=L"\textrm{Redshift } z", 
                ylabel=L"\textrm{Density parameter } \Omega", 
                label=L"\textrm{Prediction of the dark energy density } \Omega_\textrm{DE}",
                legend=:bottomright,
                ylims=(-0.7, 1.3)
)
Ω_plot = plot!(Ω_plot, uniquez, 1.0 .- mean_Ω, ribbon=CI_Ω,
                label=L"\textrm{Prediction of the matter energy density } \Omega_m",
)
result_plot = plot(μ_plot, EoS_plot, Ω_plot, layout=(3,1), size=(1200, 1600))

# Save the figure
savefig(result_plot, "1024_sample_NeuralEoS.pdf")

