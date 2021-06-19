using Flux, DiffEqFlux, DifferentialEquations, Zygote
using DataFrames, Plots, LinearAlgebra, Statistics
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.07 # in 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.475e-53 # in Mpc^3 / (Gyr^2 * planck mass)

# we cannot vary the initial conditions much, otherwise we get inconsistent results!!!
p = [3.0, 0.0]
u0 = vcat(p[1:2], [1.0, 0.0])
zspan = (0.0, 7.0)
# Function to calculate the distance modulus
# we have a +25 instead of -5 because we measure distances in Mpc
mu(z, d_L) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* d_L)) .+ 25.0 
# Function to calculate the density paramter of quintessence
Ω_ϕ(dQ, E, V) = 8pi/3 .* (0.5 .* dQ.^2 .+ V)./ E.^2

V = FastChain(
    FastDense(1, 8, relu),
    FastDense(8, 8, relu),
    FastDense(8, 1) # maybe choose exp as output function to enforce positive potentials only
)

dV(Q, p) = Flux.gradient(q -> V(q, p)[1], Q)[1]

ps = vcat(p, initial_params(V))

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Q = u[1]
    dQ = u[2]
    E = u[3]
    d_L = u[4]
    
    Ω_m = 1 - 8pi/3 * (0.5*dQ^2 .+ V(Q, p)[1])/E^2
    dE = 1.5*E/(1+z)*(Ω_m + 8pi/3 * ((1+z)*dQ/E)^2)

    du[1] = dQ
    du[2] = (2/(1+z) - dE/E) * dQ - dV(Q, p)[1]/(E*(1+z))^2
    du[3] = dE
    du[4] = 1/E
end

problem = ODEProblem(friedmann!, u0, zspan, p)
opt = ADAM(1e-2)

function predict(params)
    return Array(solve(problem, Tsit5(), u0=vcat(params[1:2],[1.0, 0.0]), p=params[3:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,:])
    return Qtils.reducedchisquared(μ, averagedata, size(params,1)), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    println("Parameters: ", p[1:2])
    return l < 1.10
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=500)

res = solve(problem, Tsit5(), u0=vcat(result.minimizer[1:2],[1.0, 0.0]), p=result.minimizer[3:end], saveat=uniquez)

plot1 = Plots.scatter(
            data.z, data.my, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

potential = map(q -> V(q, result.minimizer[3:end])[1], res[1,:])
EoS = Qtils.calculateEOS(potential, res[2,:], res[3,:], uniquez)
slowroll = Qtils.slowrollsatisfied(V, result.minimizer[3:end], res[1,:], verbose=true)

plot1 = Plots.plot!(plot1, uniquez, mu(uniquez, res[end,:]), label="fit")
plot2 = Plots.plot(uniquez, EoS, title="Equation of State", xlabel="redshift z", ylabel="equation of state w", legend=:topright)
plot3 = Plots.plot(res[1,:], potential, title="Potential", xlabel="quintessence field ϕ", ylabel="V(ϕ)", legend=:bottomright)
plot4 = Plots.plot(uniquez, 1 .- Ω_ϕ(res[2,:], res[3,:], potential), title="Density Evolution", xlabel="redshift z", ylabel="density parameter Ω", label="Ω_m")
plot4 = Plots.plot!(plot4, uniquez, Ω_ϕ(res[2,:], res[3,:], potential), legend=:topright, label="Ω_ϕ")

println("Cosmological parameters: ")
println("Dark matter density parameter Ω_m = ", 1 - Ω_ϕ(res[2,:], res[3,:], potential)[1])
println("Initial conditions for quintessence field = ", result.minimizer[1:2])
println("Slowroll satisfied for ϵ and η: ", slowroll)

# m_ϕ =  Flux.gradient(Q -> dV(Q, result.minimizer[3:end])[1], 0)[1]
# println("Mass of the scalar field = ", m_ϕ)

resultplot = Plots.plot(plot1, plot2, plot3, plot4, layout=(2, 2), size=(1200, 800))
Plots.plot(resultplot)

