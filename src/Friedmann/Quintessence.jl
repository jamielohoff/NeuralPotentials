using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, LinearAlgebra, Statistics
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 1.0 # in Mpc^3 / (Gy^2 * eV)

# we cannot vary the initial conditions much, otherwise we get inconsistent results!!!
p = vcat([1.0f0, 0.0f0], rand(Float32, 1))
modelparams = rand(Float32, 1)
u0 = vcat(p, [H0, 0.0]) # [H, phi, dphi, d_L]
tspan = (0.0, 7.0)

ps = vcat(p, modelparams)
mu(z, d_L) = 5.0 .* log10.(abs.((1.0 .+ z) .* d_L)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc
Ω_ϕ(dϕ, H, V) = 8pi*G/3 .* (0.5 .* dϕ.^2 .+ V)./ H.^2

V(ϕ, p) = p[1] * (exp(1/ϕ) - 1)
dV(ϕ, p) = Flux.gradient(ϕ -> V(ϕ,p), ϕ)[1]

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    ϕ = u[1]
    dϕ = u[2]
    Ω_m = u[3]
    H = u[4]
    d_L = u[5]
    
    dH = 1.5*H/(1+z)*(Ω_m + 8pi*G/3 * ((1+z)*dϕ)^2)
    
    du[1] = dϕ
    du[2] = -dϕ*((1+z)*dH - 2*H + dH/H + 1/(1+z)) - 1/(H*(1+z))*dV(ϕ, p)
    du[3] = (3/(1+z) - 2*dH/H) * Ω_m
    du[4] = dH
    du[5] = c/H
end

problem = ODEProblem(friedmann!, u0, tspan, p)
opt = ADAM(1e-2)

function predict(params)
    return Array(solve(problem, Tsit5(), u0=vcat(params[1:3], [H0, 0.0]), p=params[4:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[5,:])
    return sum(abs2, µ .- averagedata.mu), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    println("Parameters: ", p)
    return l < 47.0
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=1000)

res = solve(problem, Tsit5(), u0=vcat(result.minimizer[1:3],[H0, 0.0]), p=result.minimizer[4:end], saveat=uniquez)
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

plot1 = Plots.plot!(plot1, uniquez, mu(uniquez, res[5,:]), label="fit")
pot = map(x -> V(x, result.minimizer[4:end])[1], res[1,:])
EoS = Qtils.calculateEOS(pot, res[2,:])
slowroll = Qtils.slowrollsatisfied(V, dV, result.minimizer[4:end], res[1,:], G, threshold=0.1, verbose=true)
println("Slowroll conditions satisfied: ", slowroll)
plot2 = Plots.plot(uniquez, EoS, title="Equation of State", xlabel="redshift z", ylabel="equation of state w")
plot3 = Plots.plot(res[1,:], pot, title="Potential", xlabel="quintessence field ϕ", ylabel="V(ϕ)")
plot4 = Plots.plot(uniquez, res[3,:], title="Density Evolution", xlabel="redshift z", ylabel="density parameter Ω")
plot4 = Plots.plot!(plot4, uniquez, Ω_ϕ(res[2,:], res[4,:], pot), legend=:topright)

println("Cosmological parameters: ")
println("Mass parameter Ω_m = ", result.minimizer[3])
println("Average equation of state w = ", mean(EoS))
println("Initial conditions for quintessence field = ", result.minimizer[1:2])

#m_ϕ = Flux.gradient(phi -> dV(phi, result.minimizer[4:end])[1], 0)[1][1]
#println("Mass of the scalar field = ", sqrt(abs(m_ϕ)))

resultplot = Plots.plot(plot1, plot2, plot3, plot4, layout=(2, 2), legend=:bottomright, size=(1200, 800))
Plots.plot(resultplot)

