using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data,uniquez)
const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 1.0 # in Mpc^3 / (Gy^2 * eV)

p = rand(Float32, 1)
u0 = vcat(p, [H0, 0.0])
tspan = (0.0, 7.0)

mu(z, d_L) = 5.0 .* log10.(abs.((1.0 .+ z) .* d_L)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

# Defining the time-dependent equation of state
w = FastChain(
    FastDense(1, 16, relu),
    FastDense(16, 1, sigmoid) # choose output function such that 0 < |w| < 1
)

ps = vcat(p, initial_params(w))

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Ω = u[1]
    H = u[2]
    d_L = u[3]
    
    du[1] = (1 - w(z, p)[1])/(1+z) * Ω
    du[2] = 1.5*H0^2/(H*(1+z)) * (1 - w(z, p)[1])*Ω
    du[3] = c/H
end

problem = ODEProblem(friedmann!, u0, tspan, p)

function predict(params)
    return Array(solve(problem, Tsit5(), u0=[params[1],H0,0.0], p=params[2:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[3,:])
    return sum(abs2, µ .- averagedata.mu), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    println("Parameters: ", p[1])
    return l < 47.0
end

opt = ADAM(1e-2)
@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=500)

res = solve(problem, Tsit5(), u0=u0, p=result.minimizer, saveat=uniquez)

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
EoS = map(z -> w(z, result.minimizer[2:end])[1], uniquez)
plot2 = Plots.plot(uniquez, EoS, title="Equation of State w", xlabel="redshift z", ylabel="equation of state w")

println("Cosmological parameters: ")
println("Initial Ω = ", result.minimizer[1])
println("Average equation of state w = ", mean(EoS))

plot(plot1, plot2, layout=(2, 1), legend=:bottomright, size=(1200, 800))

