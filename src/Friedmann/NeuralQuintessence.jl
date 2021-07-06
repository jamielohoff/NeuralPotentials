using DifferentialEquations: tanh
using Flux: Zygote
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
Ω_m_0 = 0.311 # From Planck + WMAP + BAO, so statistically independent from SNIa
p = [3.0, 0.0, 0.0]
u0 = vcat(p[1:3], [Ω_m_0, 1.0, 0.0])
zspan = (0.0, 7.0)
# Function to calculate the distance modulus
# we have a +25 instead of -5 because we measure distances in Mpc
mu(z, χ) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* χ)) .+ 25.0 
# Function to calculate the density paramter of quintessence
Ω_ϕ(dQ, E, V, z) = 8π/3 .* (0.5 .* ((1 .+ z).*dQ).^2 .+ V./ E.^2)

V = FastChain(
    FastDense(1, 4, relu),
    FastDense(4, 4, relu),
    FastDense(4, 1) # maybe choose exp as output function to enforce positive potentials only
)
dV(x, p) = Zygote.gradient(x -> V(x, p)[1], x)[1]

ps = vcat(p, initial_params(V))

function slowrollregulatisation(dQ, V, E, z)
    return sum(((1 .+ z).*E.*dQ).^2 ./ V) / size(z, 1)
end

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Q = u[1]
    dQ = u[2]
    E = u[3]
    χ = u[4]
    
    Ω_m = 1 - Ω_ϕ(dQ, E, V(Q, p)[1], z)
    dE = 1.5*(E/(1+z)) * (Ω_m + 8π/3 * ((1+z)*dQ)^2)

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
    println(V(pred[1,:], params[3:end]))
    # v = map(x -> V(x, params[3:end])[1], pred[1,:])
    return Qtils.reducedχ2(μ, averagedata, size(params,1)) + abs(Ω_ϕ(pred[2,:], pred[3,:], v, uniquez)[1] + Ω_m_0 - 1.0), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    println("Parameters: ", p[1:2])
    v = map(x-> V(x, params[3:end])[1], pred[1,:])
    println("Density parameter Ω_m = ", 1 - Ω_ϕ(pred[2,:], pred[3,:], v, uniquez)[1])
    return l < 1.05
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=300)

res = solve(problem, Tsit5(), u0=vcat(result.minimizer[1:2],[1.0, 0.0]), p=result.minimizer[3:end], saveat=uniquez)

μ_plot = Plots.scatter(
            data.z, data.mu, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

potential = map(x -> V(x,result.minimizer[3:end])[1], res[1,:])
EoS = Qtils.calculateEOS(potential, res[2,:], res[3,:], uniquez)
slowroll = Qtils.slowrollsatisfied(V, result.minimizer[3:end], res[1,:], verbose=true)

μ_plot = plot!(μ_plot, uniquez, mu(uniquez, res[end,:]), label="fit")
EoS_plot = plot(uniquez, EoS, title="Equation of State", xlabel="redshift z", ylabel="equation of state w", legend=:topright)
V_plot = plot(uniquez, potential, title="Potential", xlabel="reshift z", ylabel="V", legend=:bottomright)
Ω_plot = plot(uniquez, 1 .- Ω_ϕ(res[2,:], res[3,:], potential, uniquez), title="Density Evolution", xlabel="redshift z", ylabel="density parameter Ω", label="Ω_m")
Ω_plot = plot!(Ω_plot, uniquez, Ω_ϕ(res[2,:], res[3,:], potential, uniquez), legend=:topright, label="Ω_ϕ")

println("Cosmological parameters: ")
println("Dark matter density parameter Ω_m = ", res[3,1])
println("Initial conditions for quintessence field = ", result.minimizer[1:2])
println("Slowroll satisfied for ϵ and η: ", slowroll)

resultplot = plot(μ_plot, EoS_plot, V_plot, Ω_plot, layout=(2, 2), size=(1600, 1200))

