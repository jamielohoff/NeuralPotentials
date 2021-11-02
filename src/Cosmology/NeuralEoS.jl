using Flux, DiffEqFlux, DifferentialEquations, Zygote
using DataFrames, CSV, Plots, Statistics, Measures, LaTeXStrings
include("../Qtils.jl")
include("../AwesomeTheme.jl")
using .Qtils

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.06766 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr

Ω_m_0 = 0.25 .+  0.75 .* rand(Float32, 1)
u0 = [Ω_m_0, 1.0, 0.0]
zspan = (uniquez[1] - 0.01, uniquez[end] + 0.01)

mu(z, χ) = 5.0 .* log10.((c/H0)*abs.((1.0 .+ z) .* χ)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

boundrelu(x) = min.(max.(x, 0.0), 1.0)

# Defining the time-dependent equation of state
w_DE = FastChain(
    FastDense(1, 8, boundrelu),
    FastDense(8, 8, boundrelu),
    FastDense(8, 1) # choose output function such that -1 < w < 1
)

ps = vcat(Ω_m_0, initial_params(w_DE))

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

problem = ODEProblem(friedmann!, u0, zspan, ps)

function predict(params)
    u0 = [params[1], 1.0, 0.0]
    return Array(solve(problem, Tsit5(), u0=u0, p=Zygote.@showgrad(params[2:end]), saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,:])
    return Qtils.χ2(μ, averagedata), pred
end

epoch = 0
cb = function(p, l, pred)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    println("Parameters: ", p[1])
    w = [ w_DE(z, p[2:end])[1] for z ∈ uniquez ]
    display(plot(uniquez, w))
    global epoch += 1
    return false
end

opt = NADAM(1e-2)
@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=5000)

u0 = [result.minimizer[1], 1.0, 0.0]
res = solve(problem, Tsit5(), u0=u0, p=result.minimizer[2:end], saveat=uniquez)

μ_plot = scatter(
            data.z, data.mu, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

μ_plot = plot!(μ_plot, uniquez, mu(uniquez, res[end,:]), label="fit")
EoS = [w_DE(z, result.minimizer[2:end])[1] for z ∈ uniquez]
EoS_plot = plot(uniquez, EoS, title="Equation of State w", xlabel="redshift z", ylabel="equation of state w")
Ω_plot = plot(uniquez, res[1,:], title="Density evolution", xlabel="redshift z", ylabel="density parameter Ω", label="Ω_m")
Ω_plot = plot!(Ω_plot, uniquez, 1.0 .- res[1,:], label="Ω_DE")

println("Cosmological parameters: ")
println("Mass parameter Ω_m = ", result.minimizer[1])

plot(μ_plot, EoS_plot, Ω_plot, layout=(3, 1), legend=:bottomright, size=(1600, 1200))

