using Base: Float32
using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data,uniquez)

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr

p =  rand(Float32, 3)
tspan = (0.0, 7.0)

ps = vcat(p)
mu(z, d_L) = 5.0 .* log10.(abs.((1.0 .+ z) .* d_L)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

# Defining the time-dependent equation of state
w_DE(z, p) = p[1] + p[2] * z/(1+z)

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Ω_m = u[1]
    H = u[2]
    d_L = u[3]

    Ω_DE = 1 - Ω_m 
    dH = 1.5*H/(1+z) * (Ω_m + (1 + w_DE(z, p))*Ω_DE)
    
    du[1] = (3/(1+z) - 2*dH/H) * Ω_m
    du[2] = dH
    du[3] = c/H
end

problem = ODEProblem(friedmann!, u0, tspan, p)
opt = ADAM(1e-2)

function predict(params)
    u0 = [params[1], H0, 0.0]
    return Array(solve(problem, Tsit5(), u0=u0, p=params[2:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,:])
    return sum(abs2, µ .- averagedata.mu), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    println("Parameters: ", p)
    return l < 47.0
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=1000)

u0_min = [result.minimizer[1], H0, 0.0]
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

plot1 = Plots.plot!(plot1, uniquez, mu(uniquez, res[3,:]), label="fit")
EoS = map(z -> w_DE(z, result.minimizer[2:end]), uniquez)
plot2 = Plots.plot(uniquez, EoS, title="Equation of State w")

plot3 = Plots.plot(uniquez, res[1,:], title="Density evolution", xlabel="redshift z", ylabel="density parameter Ω", label="Ω_m")
plot3 = Plots.plot!(plot3, uniquez, 1 .- res[1,:], label="Ω_Λ")

println("Cosmological parameters: ")
println("Mass parameter omega_m = ", result.minimizer[1])
println("Parameters of the equation of state w: ", result.minimizer[2:end])
println("Average equation of state w = ", mean(EoS))

plot(plot1, plot2, plot3, layout=(3, 1), legend=:bottomright, size=(1200, 800))

