using Flux, DiffEqFlux, DifferentialEquations, Zygote
using DataFrames, Plots, LinearAlgebra, Statistics, Measures, LaTeXStrings
include("../Qtils.jl")
include("../AwesomeTheme.jl")
using .Qtils

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.06766 # in 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.475e-53 # in Mpc^3 / (Gyr^2 * planck mass)

eVMpc = 1.564e29
eVmpl = 1.2209e28
eVGyr = 4.791e31

ϕfactor = c/sqrt(G) * sqrt(eVmpl*eVMpc)/eVGyr
Vfactor = H0^2/G * eVmpl/(eVMpc*eVGyr^2)

zspan = (0.0, 7.0)
# Function to calculate the distance modulus
# we have a +25 instead of -5 because we measure distances in Mpc
mu(z, χ) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* χ)) .+ 25.0 
# Function to calculate the density paramter of quintessence
Ω_ϕ(dQ, E, V, z) = 8π/3 .* (0.5 .* ((1.0 .+ z).*dQ).^2 .+ V./ E.^2)

### Bootstrap Loop 
itmlist = DataFrame(params = Array[], Ω_ϕ = Array[], μ = Array[], EoS = Array[], potential = Array[], Q = Array[], dQ = Array[], ϵ = Array[], η = Array[])
repetitions = 64

μ_plot = scatter(data.z, data.mu, 
            title="Redshift-Luminosity Data",
            xlabel=L"\textrm{Redshift } z",
            ylabel=L"\textrm{Distance modulus } \mu",
            yerror=data.me,
            label="Supernova and Gamma-ray burst data",
            legend=false,
)

println("Beginning Bootstrap...")
lk = ReentrantLock()
@time Threads.@threads for rep in 1:repetitions
    println("Starting repetition ", rep, " at thread ID ", Threads.threadid())
    sampledata = Qtils.elaboratesample(data, uniquez, 1.0)

    V = FastChain(
        FastDense(1, 8, tanh),
        FastDense(8, 8, tanh),
        FastDense(8, 1) # maybe choose exp as output function to enforce positive potentials only
    )
    dV(x, p) = Zygote.gradient(x -> V(x, p)[1], x)[1]

    p = vcat(2.0.*rand(Float64, 1) .- 1.0, rand(Float64,1) .- 0.5)
    u0 = vcat(p, [1.0, 0.0])
    ps = vcat(p, initial_params(V))

    # 1st order ODE for Friedmann equation in terms of z
    function friedmann!(du,u,p,z)
        Q = u[1]
        dQ = u[2]
        E = u[3]
        χ = u[4]
        
        Ω_m = 1.0 - Ω_ϕ(dQ, E, V(Q, p)[1], z)
        dE = 1.5*(E/(1+z)) * (Ω_m + 8π/3 * ((1+z)*dQ)^2)

        du[1] = dQ
        du[2] = (2.0/(1+z) - dE/E) * dQ - dV(Q, p)[1]/(E*(1+z))^2
        du[3] = dE
        du[4] = 1.0/E
    end

    problem = ODEProblem(friedmann!, u0, zspan, ps)
    opt = ADAM(1e-2)

    function predict(params)
        return Array(solve(problem, Tsit5(), u0=vcat(params[1:2],[1.0, 0.0]), p=params[3:end], saveat=sampledata.z))
    end

    function loss(params)
        pred = predict(params)
        µ = mu(sampledata.z, pred[end,:])
        return Qtils.χ2(μ, averagedata), pred
    end

    cb = function(p, l, pred)
        println("Loss: ", l)
        return false
    end

    try
        @time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=500)
        u0 = vcat(result.minimizer[1:2], [1.0, 0.0])
        res = Array(solve(problem, Tsit5(), u0=u0, p=result.minimizer[3:end], saveat=uniquez))

        potential = [V(q,result.minimizer[3:end])[1] for q ∈ res[1,:]]

        EoS = Qtils.calculateEOS(potential, res[2,:], res[3,:], uniquez)
        density_ϕ = Ω_ϕ(res[2,:], res[3,:], potential, uniquez)

        ϵ, η = Qtils.slowrollsatisfied(V, result.minimizer[3:end], res[1,:], verbose=false)

        println("Writing results...")
        lock(lk)
        push!(itmlist, [ϕfactor.*result.minimizer[1:2]./eVmpl, 
                        density_ϕ, 
                        mu(uniquez,res[end,:]), 
                        EoS, 
                        Vfactor.*potential./1e-16, 
                        ϕfactor.*res[1,:]./(1e-3*eVmpl), 
                        ϕfactor.*res[2,:]./(1e-3*eVmpl), 
                        ϵ, η]
        )
        unlock(lk)
    catch
        println("Annoying error in repetion ", rep, "!")
    finally
        println("Repetition ", rep, " is done!")
    end
end
println("Bootstrap complete!")

mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_μ, std_μ, CI_μ = Qtils.calculatestatistics(itmlist.μ)
mean_EoS, std_EoS, CI_EoS = Qtils.calculatestatistics(itmlist.EoS)
mean_Ω_ϕ, std_Ω_ϕ, CI_Ω_ϕ = Qtils.calculatestatistics(itmlist.Ω_ϕ)
mean_V, std_V, CI_V = Qtils.calculatestatistics(itmlist.potential)
mean_Q, std_Q, CI_Q = Qtils.calculatestatistics(itmlist.Q)
mean_dQ, std_dQ, CI_dQ = Qtils.calculatestatistics(itmlist.dQ)
mean_ϵ, std_ϵ, CI_ϵ = Qtils.calculatestatistics(itmlist.ϵ)
mean_η, std_η, CI_η = Qtils.calculatestatistics(itmlist.η)


println("Cosmological parameters: ")
println("Density parameter Ω_ϕ = ", mean_Ω_ϕ[1], "±", std_Ω_ϕ[1])
println("Initial conditions of scalar field in 1e-3 Planck masses: ")
println("Q_0 = ", mean_params[1], "±", std_params[1])
println("dQ_0 = ", mean_params[2], "±", std_params[2])


μ_plot = plot!(μ_plot, uniquez, mean_μ, ribbon=CI_μ, 
                label="Prediction from the neural network"
)
EoS_plot = plot(uniquez, mean_EoS, ribbon=CI_EoS,
                title="Equation of State",
                xlabel=L"\textrm{Redshift } z", 
                ylabel=L"\textrm{Equation of state } w",
                legend=false
)
Ω_plot = plot(uniquez, mean_Ω_ϕ, ribbon=CI_Ω_ϕ, 
                title="Density Evolution",
                xlabel=L"\textrm{Redshift } z", 
                ylabel=L"\textrm{Density parameter } \Omega", 
                label=L"\textrm{Prediction of the dark energy density } \Omega_\textrm{DE}",
                ylims=(-0.7, 1.3),
                legend=false
)
Ω_plot = plot!(Ω_plot, uniquez, 1.0 .- mean_Ω_ϕ, ribbon=CI_Ω_ϕ, 
                label=L"\textrm{Prediction of the dark energy density } \Omega_m"
)
V_plot = plot(mean_Q, mean_V, ribbon=CI_V, 
                title="Potential",
                xlabel=L"\textrm{Field Amplitude } \phi \textrm{ } [10^{-3}\textrm{m}_P]", 
                ylabel=L"\textrm{Potential } \frac{V(\phi (z))}{10^{-16}} \textrm{ } [\textrm{eV}^4]", 
                legend=false
)

Q_plot = plot(uniquez, mean_Q, ribbon=CI_Q, 
                title="Field Evolution",
                xlabel=L"\textrm{Redshift } z", 
                ylabel=L"\textrm{Field Amplitude } \phi \textrm{ } [10^{-3}\textrm{m}_P]", 
                legend=false
)

dQ_plot = plot(uniquez, mean_dQ, ribbon=CI_dQ, 
                title="Field Evolution",
                xlabel=L"\textrm{Redshift } z", 
                ylabel=L"\textrm{Derivative } \frac{\mathrm{d}\phi}{\mathrm{d}z} \textrm{ } [10^{-3}\textrm{m}_P]", 
                legend=false
)
result_plot = plot(μ_plot, Ω_plot, V_plot, EoS_plot, Q_plot, dQ_plot, 
                    layout=(3,2), 
                    size=(1600, 2100), 
                    margin=12mm
)
savefig(result_plot, "64_sample_NeuralQuintessence.pdf")

slowroll_plot = plot(uniquez, mean_ϵ, ribbon=[CI_ϵ[1], CI_ϵ[2]],
                    title="Slowroll Parameters for Neural Quintessence", 
                    size=(1600, 1200), label="Slowroll parameter ϵ")
slowroll_plot = plot!(slowroll_plot, uniquez, mean_η, ribbon=[CI_η[1], CI_η[2]], label="Slowroll parameter η")

savefig(slowroll_plot, "64_sample_QuintessenceSlowroll.pdf")

