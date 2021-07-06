using Base: Float32
using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data,uniquez)

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.475e-53 # in Mpc^3 / (Gyr^2 * planck mass)

p = 0.311 # 0.25 .+  0.75 .* rand(Float32, 1)
ps = vcat(p, rand(Float32, 2) .- 1.0)
tspan = (0.0, 7.0)
u0 = [p[1], 1.0, 0.0]

# we have a +25 instead of -5 because we measure distances in Mpc
mu(z, χ) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* χ)) .+ 25.0 

# Defining the time-dependent equation of state
w_DE(z, p) = p[1] + p[2] * z/(1+z)

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

problem = ODEProblem(friedmann!, u0, tspan, p)
opt = ADAM(1e-3)

function predict(params)
    u0 = [p, 1.0, 0.0]
    return Array(solve(problem, Tsit5(), u0=u0, p=params[2:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,:])
    # return mean((μ - averagedata.mu).^2), pred
    return Qtils.reducedχ2(μ, averagedata, size(params,1)), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    println("Parameters: ", p)
    return false
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=10000)

u0 = [p, 1.0, 0.0]
res = solve(problem, Tsit5(), u0=u0, p=result.minimizer[2:end], saveat=uniquez)

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
EoS = map(z -> w_DE(z, result.minimizer[2:end]), uniquez)

plot2 = Plots.plot(uniquez, EoS, title="Equation of State w")
plot3 = Plots.plot(uniquez, res[1,:], title="Density evolution", xlabel="redshift z", ylabel="density parameter Ω", label="Ω_DE")
plot3 = Plots.plot!(plot3, uniquez, 1 .- res[1,:], label="Ω_m")

println("Cosmological parameters: ")
println("Mass parameter  Ω_DE = ", p)
println("Parameters of the equation of state w_0, w_a: ", result.minimizer[2:end])
println("EoS limit w_a + w_0 = ", sum(result.minimizer[2:end]))

plot(plot1, plot2, plot3, layout=(3, 1), legend=:bottomright, size=(1200, 800))

