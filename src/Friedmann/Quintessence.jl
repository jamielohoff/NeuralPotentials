using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, LinearAlgebra, Statistics
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 1.0 # in Mpc^3 / (Gy^2 * eV)
const ρ_c = 3*H0^2/(8pi*G) # Definition of the critical density

p = rand(Float32, 3) 
modelparams = rand(Float32, 1)
u0 = vcat(p, [H0, 0.0]) # [H, phi, dphi, d_L]
tspan = (0.0, 7.0)

ps = vcat(p, modelparams)
mu(z, d_L) = 5.0 .* log10.(abs.((1.0 .+ z) .* d_L)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

V(ϕ, p) = p[1]*(exp(1/ϕ) - 1)
dV(ϕ, p) = Flux.gradient(ϕ -> V(ϕ,p), ϕ)[1]

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    ϕ = u[1]
    dϕ = u[2]
    H = u[3]
    d_L = u[4]
    
    dH = 1.5*H0^2/(H*(1+z)) * (Ω_m + H^2*(1+z)^2*dϕ^2/ρ_c)
    
    Ω_m = p[1]*(1+z)^3
    du[1] = dϕ
    du[2] = -dϕ*((1+z)*dH - 2*H + dH/H + 1/(1+z)) - 1/(H*(1+z))*dV(ϕ, p)
    du[3] = dH
    du[4] = c/H
end

problem = ODEProblem(friedmann!, u0, tspan, p)
opt = ADAM(1e-2)

function predict(params)
    return Array(solve(problem, Tsit5(), u0=vcat(params[1:2], [H0, 0.0]), p=params[3:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[5,:])
    return sum(abs2, µ .- averagedata.mu), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    println("Parameters: ", p)
    return false
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=1000)

res = solve(problem, Tsit5(), u0=vcat(result.minimizer[1:2],[H0, 0.0]), p=result.minimizer[3:end], saveat=uniquez)
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
pot = map(x -> V(x, result.minimizer[4:end])[1], res[1,:])
EoS = Qtils.calculateEOS(pot, res[2,:])
plot2 = Plots.plot(uniquez, EoS, title="Equation of State")
plot3 = Plots.plot(res[1,:], pot, title="Potential")

plot(plot1, plot2, plot3, layout=(3, 1), legend=:bottomright)

