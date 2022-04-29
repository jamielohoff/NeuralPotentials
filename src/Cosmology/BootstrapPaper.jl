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
# Gauss function
gauss(x, σ, μ) = 1.0/sqrt(2π*σ^2) .* exp(-(x .- μ).^2 ./ σ^2)

### Bootstrap Loop 
repetitions = 4
itmlist = DataFrame(params = Array[], 
                        Ωϕ = Array[], 
                        V = Array[], 
                        ϕ = Array[], 
                        dϕ = Array[], 
                        H = Array[],
                        EoS = Array[])
println("Beginning Bootstrap...")
lk = ReentrantLock()
@time Threads.@threads for rep ∈ 1:repetitions
    @label start
    println("Starting repetition ", rep, " at thread ID ", Threads.threadid())

    # Sample supernova Ia data from the dataset
    sampledata = Qtils.elaboratesample(data, uniquez, 1.0)

    # Defining the potential and its gradient in the form of a neural network
    U = FastChain(
        FastDense(1, 4, tanh),
        FastDense(4, 8, tanh),
        FastDense(8, 4, tanh),
        FastDense(4, 1, x -> gauss(x, 3.0, 0.0)) # enforce positive potentials only
    )
    dU(x, p) = ForwardDiff.gradient(x -> U(x, p)[1], x)[1]

    # Initialize parameters of the model with uniform distributions
    p = [0.3, 0.1].*rand(Float64, 2) .+ [0.4, -0.05]
    u0 = vcat(p, [1.0, 0.0])
    ps = vcat(p, initial_params(U))

    # 1st order ODE for Friedmann equation in terms of z
    function friedmann!(du,u,p,z)
        Q = u[1]
        dQ = u[2]
        E = u[3]
        χ = u[4]
        
        Ω_m = 1.0 - Ω_ϕ(dQ, E, U([Q], p)[1], z)
        dE = 1.5*(E/(1+z)) * (Ω_m + 8π/3 * ((1+z)*dQ)^2)

        du[1] = dQ
        du[2] = (2.0/(1+z) - dE/E) * dQ - dU([Q], p)/(E*(1+z))^2
        du[3] = dE
        du[4] = 1.0/E
    end

    # Define ODE problem and Optimizer
    problem = ODEProblem(friedmann!, u0, zspan, ps)
    opt = ADAM(1e-2)

    # Function that predicts the results for a given set of parameters by solving the ODE at the given redshifts
    function predict(params)
        return Array(solve(problem, Tsit5(), u0=vcat(params[1:2],[1.0, 0.0]), p=params[3:end], saveat=zrange))
    end

    # Function that calculates the loss with respect to the observed data
    function loss(params)
        pred = predict(params)
        µ = mu(sampledata.z, pred[end,indexes])
        potential = U(reshape(pred[1,:],1,:), params[3:end])
        Ωϕ = Ω_ϕ(pred[2,indexes], pred[3,indexes], potential[indexes], uniquez)
        return Qtils.χ2(μ, sampledata) + 250.0*abs(Ωϕ[1] - 0.7) + 125.0*abs(Ωϕ[end]), pred
    end

    # Callback function
    cb = function(p, l, pred)
        return false
    end

    try
        # Start the training of the model
        @time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=60) # 6000 is good value

        # Use the best result, i.e. the one with the lowest loss and compute the potential etc. for it
        u0 = vcat(result.minimizer[1:2], [1.0, 0.0])
        res = Array(solve(problem, Tsit5(), u0=u0, p=result.minimizer[3:end], saveat=zrange))

        Q = res[1,:]
        dQ = res[2,:]
        E = res[3,:]
        H = H0*E*1e3
        potential = [U(q,result.minimizer[3:end])[1] for q ∈ Q]
        Ωϕ = Ω_ϕ(dQ, E, potential, zrange)

        # Calculate the effective equation of state
        EoS = Qtils.calculateEOS(potential, dQ, E, zrange)

        # Push the result into the array
        println("Writing results...")
        lock(lk)
        push!(itmlist, [ϕfactor.*result.minimizer[1:2]./(1e-3*eVmpl), Ωϕ, 
                        Vfactor.*potential./1e-16, 
                        ϕfactor.*Q./(1e-3*eVmpl), 
                        ϕfactor.*dQ./(1e-3*eVmpl),
                        H./(1.0 .+ zrange).^3,
                        EoS 
        ])
        unlock(lk)
    catch
        println("Solver crashed in repetion ", rep, "!")
        @goto start
    finally
        println("Repetition ", rep, " is done!")
    end
end
println("Bootstrap complete: ", size(itmlist.params)[1],"/", repetitions)

# Calculate the mean, standard deviation and 95% confidence intervals for the quantities of interest
mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_Ωϕ, std_Ωϕ, CI_Ωϕ = Qtils.calculatestatistics(itmlist.Ωϕ)
mean_V, std_V, CI_V = Qtils.calculatestatistics(itmlist.V)
mean_ϕ, std_ϕ, CI_ϕ = Qtils.calculatestatistics(itmlist.ϕ)
mean_dϕ, std_dϕ, CI_dϕ = Qtils.calculatestatistics(itmlist.dϕ)
mean_H, std_H, CI_H = Qtils.calculatestatistics(itmlist.H)
mean_EoS, std_EoS, CI_EoS = Qtils.calculatestatistics(itmlist.EoS)

ϕ0 = [arr[1] for arr ∈ itmlist.params]
dϕ0 = [arr[2] for arr ∈ itmlist.params]

println("Cosmological parameters: ")
println("Density parameter Ω_ϕ = ", mean_Ωϕ[1], "±", std_Ωϕ[1])
println("Initial conditions of scalar field in 1e-3 Planck masses: ")
println("ϕ0 = ", mean_params[1], "±", std_params[1])
println("dϕ0 = ", mean_params[2], "±", std_params[2])

# Plot all the stuff into a 4x2 figure
H_plot = plot(zrange, mean_H, ribbon=CI_H, 
                title="expansion rate",
                xlabel=L"\textrm{redshift} \; z", 
                ylabel=L"Ha^{(3 \backslash 2)} \; [\textrm{km}\textrm{s}^{-1}\textrm{Mpc}^{-1}]",
                legend=false
)

ϕ_plot = plot(mean_ϕ, mean_dϕ, ribbon=CI_dϕ,
                title="phase-space trajectory",
                xlabel=L"\phi \; [10^{-3}\textrm{m}_P]", 
                ylabel=L"\dfrac{\mathrm{d}\phi}{\mathrm{d}z} \; [10^{-3}\textrm{m}_P]",
                legend=false
)

Ω_plot = plot(zrange, mean_Ωϕ, ribbon=CI_Ωϕ, 
                title="density evolution",
                xlabel=L"\textrm{redshift}\; z", 
                ylabel=L"\textrm{density} \; \textrm{parameter}\; \Omega", 
                label=L"\Omega_\phi",
                ylims=(0.0, 1.0),
                legend=:right
)

Ω_plot = plot!(Ω_plot, zrange, 1.0 .- mean_Ωϕ, ribbon=CI_Ωϕ, label=L"\Omega_m")

V_plot = plot(mean_ϕ, mean_V, ribbon=CI_V, 
                title="potential",
                xlabel=L"\phi \; [10^{-3}\textrm{m}_P]", 
                ylabel=L"\frac{V(\phi)}{10^{-16}} \; [\textrm{eV}^4]", 
                legend=false
)

EoS_plot = plot(zrange, mean_EoS, ribbon=CI_EoS, 
                title="equation of state",
                xlabel=L"\textrm{redshift}\; z", 
                ylabel=L"\textrm{equation}\;\textrm{of}\;\textrm{state}\; w", 
                legend=false
)

nbins = 30 # :scott

ϕ0_hist = plot(title="initial distribution",
                xlabel=L"\phi(z=0) \; [10^{-3}\textrm{m}_P]", 
                ylabel=L"\# \; \textrm{of} \; \textrm{occurences}",
                legend=false
)
ϕ0_hist = histogram!(ϕ0_hist, ϕ0, bins=nbins)

dϕ0_hist = plot(title="initial distribution",
                xlabel=L"\dfrac{\mathrm{d}\phi}{\mathrm{d}z}(z=0) \; [10^{-3}\textrm{m}_P]", 
                ylabel=L"\# \; \textrm{of} \; \textrm{occurences}",
                legend=false
)
dϕ0_hist = histogram!(dϕ0_hist, dϕ0, bins=nbins)

println("Plotting results...")
result_plot = plot(H_plot, Ω_plot, EoS_plot, ϕ0_hist,
                    V_plot, ϕ_plot, EoS_plot, dϕ0_hist,
                    layout=(2,4), 
                    size=(2400, 1200), 
                    margin=12mm, 
)

# Save the figure
savefig(result_plot, "Plots.pdf")

# Covariance matrix
covm = cov(itmlist.V)
map = zeros(size(covm))

scale = 1.0
for i ∈ 1:length(std_V)
    for j ∈ 1:length(std_V)
        if std_V[i]*std_V[j] == 0
            map[i,j] = 0.0
        else
            map[i,j] = asinh(covm[i,j]/scale) # /(std_V[i]*std_V[j])
        end
    end
end

ticks = 200:200:1000
labels = round.([mean_V[i] for i ∈ ticks]; digits=3)
heat_plot = heatmap(map, size=(1200,1200), c=:plasma,
                    title="covariance matrix",
                    xlabel=L"\textrm{potential} \; \frac{V(\phi)}{10^{-16}} \; [\textrm{eV}^4]",
                    ylabel=L"\textrm{potential} \; \frac{V(\phi)}{10^{-16}} \; [\textrm{eV}^4]",
                    xticks=(ticks, labels),
                    yticks=(ticks, labels),
)

# Save the figure
savefig(heat_plot, "CovarianceMatrix.pdf")

