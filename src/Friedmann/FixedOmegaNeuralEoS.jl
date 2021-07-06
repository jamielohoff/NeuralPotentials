using DifferentialEquations: tanh
using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.475e-53 # in Mpc^3 / (Gyr^2 * planck mass)

Ω_m_0 = 0.311 # From Planck + WMAP + BAO, so statistically independent from SNIa
u0 = [Ω_m_0, 1.0, 0.0]
zspan = (0.0, 7.0)

mu(z, χ) = 5.0 .* log10.((c/H0)*abs.((1.0 .+ z) .* χ)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

# Defining the time-dependent equation of state
w_DE = FastChain(
    FastDense(1, 8, relu),
    FastDense(8, 8, relu),
    FastDense(8, 1, tanh) # choose output function such that -1 < w < 1
)

ps = initial_params(w_DE)

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

problem = ODEProblem(friedmann!, u0, zspan, p)

function predict(params)
    return Array(solve(problem, Tsit5(), p=params, saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,:])
    return Qtils.reducedχ2(μ, averagedata, size(params,1)), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    return false # l < 1.05
end

opt = ADAM(1e-2)
@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=500)
res = solve(problem, Tsit5(), p=result.minimizer, saveat=uniquez)

plot1 = Plots.scatter(
            data.z, data.mu, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

plot1 = Plots.plot!(plot1, uniquez, mu(uniquez, res[end,:]), label="fit")
EoS = map(z -> w_DE(z, result.minimizer)[1], uniquez)
plot2 = Plots.plot(uniquez, EoS, title="Equation of State w", xlabel="redshift z", ylabel="equation of state w")
plot3 = Plots.plot(uniquez, res[1,:], title="Density evolution", xlabel="redshift z", ylabel="density parameter Ω", label="Ω_m")
plot3 = Plots.plot!(plot3, uniquez, 1 .- res[1,:], label="Ω_DE")

println("Cosmological parameters: ")
println("Mass parameter Ω_m = ", Ω_m_0)

plot(plot1, plot2, plot3, layout=(3, 1), legend=:bottomright, size=(1200, 800))

