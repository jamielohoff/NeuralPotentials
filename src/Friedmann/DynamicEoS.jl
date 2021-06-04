using Base: Float32
using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data,uniquez)

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 1.0 # in Mpc^3 / (Gy^2 * eV)

p =  2 .* rand(Float32, 2) .- 1
Ω_0 = rand(Float32,1)[1]
u0 = [Ω_0, 1-Ω_0, H0, 0.0]
tspan = (0.0, 7.0)

ps = vcat([Ω_0],p)
mu(z, d_L) = 5.0 .* log10.(abs.((1.0 .+ z) .* d_L)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

# Defining the time-dependent equation of state
w_DE(z, p) = p[1] + p[2] * z/(1+z)

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Ω_m = u[1]
    Ω_DE = u[2]
    H = u[3]
    d_L = u[4]
    
    du[1] = 3.0/(1+z) * Ω_m
    du[2] = 3.0 * (1 + w_DE(z,p))/(1+z) * Ω_DE
    du[3] = 1.5*H0^2/(H*(1+z)) * (Ω_m + (1 + w_DE(z, p))*Ω_DE)
    du[4] = c/H
end

problem = ODEProblem(friedmann!, u0, tspan, p)
opt = ADAM(1e-2)

function predict(params)
    Ω_0 = params[1]
    u0 = [Ω_0, 1-Ω_0, H0, 0.0]
    return Array(solve(problem, Tsit5(), u0=u0, p=params[2:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[4,:])
    return sum(abs2, µ .- averagedata.mu), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    println("Parameters: ", p)
    return false
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=3000)

Ω_0_min = result.minimizer[1]
u0_min = [Ω_0_min, 1-Ω_0_min, H0, 0.0]
res = solve(problem, Tsit5(), u0=u0_min, p=result.minimizer[2:end], saveat=uniquez)
println("Best result: ", result.minimizer)

plot1 = Plots.scatter(
            data.z, data.my, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

plot1 = Plots.plot!(plot1, uniquez, mu(uniquez, res[4,:]), label="fit")
EoS = map(z -> w_DE(z, result.minimizer[2:end]), uniquez)
plot2 = Plots.plot(uniquez, EoS, title="Equation of State w")

println("Cosmological parameters: ")
println("Mass parameter omega_m = ", result.minimizer[1])
println("Parameters of the equation of state w: ", result.minimizer[2:end])
println("Average equation of state w = ", mean(EoS))

plot(plot1, plot2, layout=(2, 1), legend=:bottomright, size=(1200, 800))

