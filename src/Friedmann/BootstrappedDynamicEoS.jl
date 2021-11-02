using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics, Measures, LaTeXStrings
include("../lib/Qtils.jl")
include("../lib/AwesomeTheme.jl")
using .Qtils

# Load predefined plot theme and adjust font sizes
theme(:awesome)
resetfontsizes()
scalefontsizes(2)

# Load supernova Ia data
data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")

# Natural constants
const H0 = 0.06766 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.475e-53 # in Mpc^3 / (Gyr^2 * planck mass)

# Priors from PLANCK Collaboration for CPL parametrization
Ω_m_0 = 0.3111
w_0 = -0.76
w_a = -0.72
ps0 = [Ω_m_0 , w_0, w_a]
prior_μ = ps0
prior_σ = [0.0056, 0.20, 0.62]
ps0 = vcat(0.25 .+  0.75 .* rand(Float32, 1), -rand(Float32,2))

# Priors from PLANCK Collaboration for wCDM model
# Ω_m_0 = 0.3111
# w_0 = -1.028
# ps0 = [Ω_m_0, w_0]
# prior_μ = ps0
# prior_σ = [0.0056, 0.031]

tspan = (0.0, 7.0)
u0 = [ps0[1], 1.0, 0.0]

# Defining the time-dependent equation of state
w_DE(z, p) = p[1] + p[2] * z/(1+z)

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Ω_m = u[1]
    E = u[2]
    χ = u[3]

    Ω_DE = 1.0 - Ω_m
    dE = 1.5*E/(1+z) * (Ω_m + (1 + w_DE(z, p))*Ω_DE)
    
    du[1] = (3.0/(1+z) - 2.0*dE/E) * Ω_m
    du[2] = dE
    du[3] = 1.0/E
end

problem = ODEProblem(friedmann!, u0, tspan, ps0[2:end])

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
    sampledata = Qtils.elaboratesample(data, uniquez, 1.0)

    # Function that predicts the results for a given set of parameters by solving the ODE at the given redshifts
    function predict(params)
        u0 = [params[1], 1.0, 0.0] # params[1]
        return Array(solve(problem, Tsit5(), u0=u0, p=params[2:end], saveat=sampledata.z))
    end
    
    # Function that calculates the loss with respect to the observed data
    function loss(params)
        pred = predict(params)
        µ = Qtils.mu(sampledata.z, pred[end,:])
        return Qtils.reducedχ2(μ, sampledata, size(params,1)), pred
    end
    
    # Callback function 
    cb = function(p, l, pred)
        return l < 1.0075
    end

    # Defining the optimitzer and parameter initializations
    opt = NADAM(0.01)
    # ps = vcat(0.15 .+  0.85 .* rand(Float32, 1), -0.15 .- 2.0.*rand(Float32,2))
    ps = vcat(0.3111, -0.15 .- 2.0.*rand(Float32,2))

    # Start the training of the model
    @time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=5000)

    # Use the best result, i.e. the one with the lowest loss and compute the equation of state etc. for it
    u0 = [result.minimizer[1], 1.0, 0.0]
    res = Array(solve(problem, Tsit5(), u0=u0, p=result.minimizer[2:end], saveat=uniquez))
    EoS = [w_DE(z, result.minimizer[2:end]) for z ∈ uniquez]

    # Push the result into the array
    lock(lk)
    push!(itmlist, [result.minimizer, res[1,:], Qtils.mu(uniquez,res[end,:]), EoS])
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
println("CPL parameters w_0, w_a = ", mean_params[2:end]," ± ", std_params[2:end])

# Plot all the stuff into a 3x1 figure
μ_plot = plot!(μ_plot, uniquez, mean_μ, ribbon=CI_μ, 
                label="Prediction of the neural equation of state", 
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
result_plot = plot(μ_plot, EoS_plot, Ω_plot, layout=(3,1), size=(1200,1600))

# Save the figure
savefig(result_plot, "1024_sample_CPL.pdf")

