using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics, Measures, LaTeXStrings
include("../Qtils.jl")
include("../AwesomeTheme.jl")
using .Qtils

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.475e-53 # in Mpc^3 / (Gyr^2 * planck mass)

# # Priors from Abbott et al., ApJ Letters 872
# Ω_m_0 = 0.332
# w_0 = -0.714
# w_a = -0.714
# ps = [Ω_m_0 , w_0, w_a] # 0.25 .+  0.75 .* rand(Float32, 1)
# ps = vcat(0.25 .+  0.75 .* rand(Float32, 1), -rand(Float32,2))

# Priors from Abbott et al., ApJ Letters 872
Ω_m_0 = 0.310
w_0 = -0.988
# ps = vcat(rand(Float32,1), -0.334 .- rand(Float32,1)) # [Ω_m_0 , w_0] 

tspan = (0.0, 7.0)
u0 = [ps[1], 1.0, 0.0]

# we have a +25 instead of -5 because we measure distances in Mpc
mu(z, χ) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* χ)) .+ 25.0 

# Defining the time-dependent equation of state
w_DE(z, p) = p[1] # + p[2] * z/(1+z)

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Ω_m = u[1]
    E = u[2]
    χ = u[3]

    Ω_DE = 1 - Ω_m
    dE = 1.5*E/(1+z) * (Ω_m + (1 + w_DE(z, p))*Ω_DE)
    
    du[1] = (3/(1+z) - 2*dE/E) * Ω_m
    du[2] = dE
    du[3] = 1/E
end

problem = ODEProblem(friedmann!, u0, tspan, ps[2:end])

### Bootstrap Loop 
itmlist = DataFrame(params = Array[], Ω = Array[], μ = Array[], EoS = Array[])
repetitions = 64

μ_plot = scatter(
            data.z, data.mu, 
            title="Redshift-Luminosity Data",
            xlabel=L"\textrm{Redshift } z",
            ylabel=L"\textrm{Distance modulus } \mu",
            yerror=data.me,
            label="Supernova and gamma-ray burst data",
            legend=:bottomright,
)

println("Beginning Bootstrap...")
lk = ReentrantLock()
@time Threads.@threads for rep in 1:repetitions
    println("Starting repetition ", rep, " at thread ID ", Threads.threadid())
    sampledata = Qtils.elaboratesample(data, uniquez, 1.0)

    function predict(params)
        u0 = [params[1], 1.0, 0.0]
        return Array(solve(problem, Tsit5(), u0=u0, p=params[2:end], saveat=sampledata.z))
    end
    
    function loss(params)
        pred = predict(params)
        µ = mu(sampledata.z, pred[end,:])
        return Qtils.reducedχ2(μ, sampledata, size(params,1)), pred
    end
    
    cb = function(p, l, pred)
        # println("Loss at thread ", Threads.threadid(), " : ", l)
        # println("Params at thread ", Threads.threadid(), " : ", p[1:3])
        return false
    end

    opt = ADAM(1e-3)
    @time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=4000)

    u0 = [result.minimizer[1], 1.0, 0.0]
    res = Array(solve(problem, Tsit5(), u0=u0, p=result.minimizer[2:end], saveat=uniquez))
    EoS = map(z -> w_DE(z, result.minimizer[2:end]), uniquez)
    lock(lk)
    push!(itmlist, [result.minimizer, res[1,:], mu(uniquez,res[end,:]), EoS])
    unlock(lk)
    println("Repetition ", rep, " is done!")
end
println("Bootstrap complete!")

# CSV.write(joinpath(@__DIR__, "Bootstrap.CSV"), itmlist)

mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_μ, std_μ, CI_μ = Qtils.calculatestatistics(itmlist.μ)
mean_EoS, std_EoS, CI_EoS = Qtils.calculatestatistics(itmlist.EoS)
mean_Ω, std_Ω, CI_Ω = Qtils.calculatestatistics(itmlist.Ω)

println("Cosmological parameters: ")
println("Mass parameter Ω_m = ", mean_params[1], " ± ", std_params[1])
println("CPL parameters w_0, w_a = ", mean_params[2:end]," ± ", std_params[2:end])

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
Ω_plot = plot!(Ω_plot, uniquez, 1 .- mean_Ω, ribbon=CI_Ω,
                label=L"\textrm{Prediction of the matter energy density } \Omega_m",
)
result_plot = plot(μ_plot, EoS_plot, Ω_plot, layout=(3,1), size=(1200,1600))
savefig(result_plot, "64_sample_w_free_omega_m_prior.pdf")

