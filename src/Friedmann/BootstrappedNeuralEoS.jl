using DifferentialEquations: tanh
using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics
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

tspan = (0.0, 7.0)
# Priors from Abbott et al., ApJ Letters 872
Ω_m_0 = 0.332
u0 = [Ω_m_0 , 1.0, 0.0]

mu(z, χ) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* χ)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

# Defining the time-dependent equation of state
w_DE = FastChain(
    FastDense(1, 8, relu),
    FastDense(8, 8, relu),
    FastDense(8, 1, relu) # choose output function such that -1 < w < 1, constraint from experiments (TODO: find some papers to verify this)
)

ps = vcat(rand(Float32, 1), initial_params(w_DE))

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Ω_m = u[1]
    E = u[2]
    χ = u[3]

    Ω_DE = 1 - Ω_m 
    dE = 1.5*E/(1+z) * (Ω_m + (1 + w_DE(z, p)[1])*Ω_DE)
    
    du[1] = (3/(1+z) - 2*dE/E) * Ω_m
    du[2] = dE
    du[3] = 1/E
end

problem = ODEProblem(friedmann!, u0, tspan, ps)

### Bootstrap Loop
itmlist = DataFrame(params = Array[], Ω = Array[], μ = Array[], EoS = Array[])
repetitions = 16

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
    df = Qtils.elaboratesample(data, uniquez, 1.0)
    sampledata = Qtils.resample(df)

    function predict(params)
        u0 = [Ω_m_0 , 1.0, 0.0]
        return Array(solve(problem, Tsit5(), u0=u0, p=params[2:end], saveat=sampledata.z))
    end
    
    function loss(params)
        pred = predict(params)
        µ = mu(sampledata.z, pred[end,:])
        return Qtils.reducedχ2(μ, sampledata, size(params,1)), pred
    end
    
    cb = function(p, l, pred)
        # println("Loss at thread ", Threads.threadid(), " : ", l)
        # println("Params at thread ", Threads.threadid(), " : ", p[1])
        return l < 1.0
    end

    p = 0.25 .+  0.75 .* rand(Float32, 1) # Ω_m_0 #
    ps = vcat(p, initial_params(w_DE))
    opt = ADAM(1e-2)
    @time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=300)

    u0 = [Ω_m_0, 1.0, 0.0]
    res = Array(solve(problem, Tsit5(), u0=u0, p=result.minimizer[2:end], saveat=uniquez))
    EoS = map(z -> w_DE(z, result.minimizer[2:end])[1], uniquez)

    lock(lk)
    push!(itmlist, [[Ω_m_0], res[1,:], mu(uniquez,res[end,:]), EoS])
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
Ω_plot = plot!(Ω_plot, uniquez, 1 .- mean_Ω, ribbon=CI_Ω,
                label=L"\textrm{Prediction of the matter energy density } \Omega_m",
)
result_plot = plot(μ_plot, EoS_plot, Ω_plot, layout=(3,1), size=(1600, 1200))
savefig(result_plot, "64_sample_NeuralEoS_fixed_omega_m_random.png")

