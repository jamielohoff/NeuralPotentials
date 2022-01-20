using Flux, DiffEqFlux, DifferentialEquations, Zygote, ForwardDiff
using DataFrames, Plots, LinearAlgebra, Statistics, Measures, LaTeXStrings
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

# Conversion factors to electron volts
const eVMpc = 1.564e29
const eVmpl = 1.2209e28
const eVGyr = 4.791e31

const ϕfactor = c/sqrt(G) * sqrt(eVmpl*eVMpc)/eVGyr
const Vfactor = H0^2/G * eVmpl/(eVMpc*eVGyr^2)

zspan = (0.0, 7.0)
zrange = Array(range(uniquez[1], uniquez[end], step=0.01))
zrange = sort(unique(vcat(zrange, uniquez)))
indexes = findall(z -> z ∈ uniquez, zrange)

# Function to calculate the distance modulus
# We have a +25 instead of -5 because we measure distances in Mpc
mu(z, χ) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* χ)) .+ 25.0 
# Function to calculate the density paramter of quintessence
Ω_ϕ(dQ, E, V, z) = 8π/3 .* (0.5 .* ((1 .+ z).*dQ).^2 .+ V./ E.^2)

# Plot supvernova Ia data
μ_plot = scatter(data.z, data.mu, 
            title="Redshift-Luminosity Data",
            xlabel=L"\textrm{Redshift } z",
            ylabel=L"\textrm{Distance modulus } \mu",
            yerror=data.me,
            label="Supernova and Gamma-ray burst data",
            legend=false,
)

### Bootstrap Loop 
repetitions = 128
itmlist = DataFrame(params = Array[], Ω_ϕ = Array[], μ = Array[], EoS = Array[], potential = Array[], Q = Array[], dQ = Array[])
println("Beginning Bootstrap...")
lk = ReentrantLock()
@time Threads.@threads for rep in 1:repetitions
    println("Starting repetition ", rep, " at thread ID ", Threads.threadid())

    # Sample supernova Ia data from the dataset
    sampledata = Qtils.elaboratesample(data, uniquez, 1.0)

    # Defining the potential and its gradient in the form of a neural network
    V = FastChain(
        FastDense(1, 4, relu),
        FastDense(4, 4, relu),
        FastDense(4, 4, relu),
        FastDense(4, 1, relu) # maybe choose exp as output function to enforce positive potentials only
    )
    dV(x, p) = ForwardDiff.gradient(x -> V(x, p)[1], x)[1]

    # Initialize parameters of the model with uniform distributions
    p = [3.0, 0.1].*rand(Float64, 2)
    u0 = vcat(p, [1.0, 0.0])
    ps = vcat(p, initial_params(V))

    # 1st order ODE for Friedmann equation in terms of z
    function friedmann!(du,u,p,z)
        Q = u[1]
        dQ = u[2]
        E = u[3]
        χ = u[4]
        
        Ω_m = 1.0 - Ω_ϕ(dQ, E, V([Q], p)[1], z)
        dE = 1.5*(E/(1+z)) * (Ω_m + 8π/3 * ((1+z)*dQ)^2)

        du[1] = dQ
        du[2] = (2.0/(1+z) - dE/E) * dQ - dV([Q], p)/(E*(1+z))^2
        du[3] = dE
        du[4] = 1.0/E
    end

    # Define ODE problem and Optimizer
    problem = ODEProblem(friedmann!, u0, zspan, ps)
    opt = ADAM(1e-3)

    # Function that predicts the results for a given set of parameters by solving the ODE at the given redshifts
    function predict(params)
        return Array(solve(problem, Tsit5(), u0=vcat(params[1:2],[1.0, 0.0]), p=params[3:end], saveat=zrange))
    end

    # Function that calculates the loss with respect to the observed data
    function loss(params)
        pred = predict(params)
        µ = mu(sampledata.z, pred[end,indexes])
        return Qtils.χ2(μ, sampledata), pred
    end

    # Callback function
    cb = function(p, l, pred)
        return false
    end

    try
        # Start the training of the model
        @time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=650)

        # Use the best result, i.e. the one with the lowest loss and compute the potential etc. for it
        u0 = vcat(result.minimizer[1:2], [1.0, 0.0])
        res = Array(solve(problem, Tsit5(), u0=u0, p=result.minimizer[3:end], saveat=zrange))

        potential = [V(q,result.minimizer[3:end])[1] for q ∈ res[1,:]]
        EoS = Qtils.calculateEOS(potential, res[2,:], res[3,:], zrange)
        density_ϕ = Ω_ϕ(res[2,:], res[3,:], potential, zrange)

        # Push the result into the array
        println("Writing results...")
        lock(lk)
        push!(itmlist, [ϕfactor.*result.minimizer[1:2]./eVmpl, 
                        density_ϕ, 
                        mu(zrange,res[end,:]), 
                        EoS, 
                        Vfactor.*potential./1e-16, 
                        ϕfactor.*res[1,:]./(1e-3*eVmpl), 
                        ϕfactor.*res[2,:]./(1e-3*eVmpl), 
        ])
        unlock(lk)
    catch
        println("Annoying error in repetion ", rep, "!")
    finally
        println("Repetition ", rep, " is done!")
    end
end
println("Bootstrap complete!")

# Calculate the mean, standard deviation and 95% confidence intervals for the quantities of interest
println(itmlist.params)
mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_μ, std_μ, CI_μ = Qtils.calculatestatistics(itmlist.μ)
mean_EoS, std_EoS, CI_EoS = Qtils.calculatestatistics(itmlist.EoS)
mean_Ω_ϕ, std_Ω_ϕ, CI_Ω_ϕ = Qtils.calculatestatistics(itmlist.Ω_ϕ)
mean_V, std_V, CI_V = Qtils.calculatestatistics(itmlist.potential)

println("Cosmological parameters: ")
println("Density parameter Ω_ϕ = ", mean_Ω_ϕ[1], "±", std_Ω_ϕ[1])
println("Initial conditions of scalar field in 1e-3 Planck masses: ")
println("Q_0 = ", mean_params[1], "±", std_params[1])
println("dQ_0 = ", mean_params[2], "±", std_params[2])

# Plot all the stuff into a 3x2 figure
μ_plot = plot!(μ_plot, zrange, mean_μ, ribbon=CI_μ, 
                label="Prediction from the neural network"
)
EoS_plot = plot(zrange, mean_EoS, ribbon=CI_EoS,
                title="Equation of State",
                xlabel=L"\textrm{Redshift } \; z", 
                ylabel=L"\textrm{Equation of state }\; w",
                legend=false
)
Ω_plot = plot(zrange, mean_Ω_ϕ, ribbon=CI_Ω_ϕ, 
                title="Density Evolution",
                xlabel=L"\textrm{Redshift }\; z", 
                ylabel=L"\textrm{Density parameter }\; \Omega", 
                label=L"\textrm{Prediction of the dark energy density } \; \Omega_\textrm{DE}",
                ylims=(-0.7, 1.3),
                legend=false
)
Ω_plot = plot!(Ω_plot, zrange, 1.0 .- mean_Ω_ϕ, ribbon=CI_Ω_ϕ, 
                label=L"\textrm{Prediction of the dark energy density }\; \Omega_m"
)
V_plot = plot(zrange, mean_V, ribbon=CI_V, 
                title="Potential",
                xlabel=L"\textrm{Redshift }\; z", 
                ylabel=L"\textrm{Potential } \; \frac{V(\phi (z))}{10^{-16}} \textrm{ } [\textrm{eV}^4]", 
                legend=false
)

result_plot = plot(μ_plot, Ω_plot, V_plot, EoS_plot,
                    layout=(2,2), 
                    size=(1200, 1200), 
                    margin=12mm
)
# Save the figure
savefig(result_plot, "Application.pdf")

