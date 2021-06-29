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
p = [3.0, 0.0, 0.0]
u0 = vcat(p[1:3], [1.0, 0.0,])
zspan = (0.0, 7.0)
# Function to calculate the distance modulus
# we have a +25 instead of -5 because we measure distances in Mpc
mu(z, d) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* d)) .+ 25.0 
# Function to calculate the density paramter of quintessence
Ω_ϕ(dQ, E, V, z) = 8pi/3 .* (0.5 .* ((1 .+ z).*dQ).^2 .+ V./ E.^2)

dV = FastChain(
    FastDense(1, 8, relu),
    FastDense(8, 8, relu),
    FastDense(8, 1)
)

ps = vcat(p, initial_params(dV))

function slowrollregulatisation(dQ, V, E, z)
    return sum(((1 .+ z).*E.*dQ).^2 ./ V) / size(z, 1)
end

# function slowrollregulatisation(Q, V, params)
#     gradV = reshape(dV(reshape(Q, 1, :), params), :)
#     ϵ = 1/(16pi) * (gradV./V).^2
#     return sum(ϵ) / size(V, 1)
# end

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Q = u[1]
    dQ = u[2]
    V = u[3]
    E = u[4]
    d = u[5]
    
    Ω_m = 1 - Ω_ϕ(dQ, E, V, z)
    dE = 1.5*(E/(1+z)) * (Ω_m + 8pi/3 * ((1+z)*dQ)^2)

    du[1] = dQ
    du[2] = (2/(1+z) - dE/E) * dQ - dV(Q, p)[1]/(E*(1+z))^2
    du[3] = abs(dV(Q, p)[1] * dQ)
    du[4] = dE
    du[5] = 1/E
end

problem = ODEProblem(friedmann!, u0, zspan, p)
opt = ADAM(1e-2)

function predict(params)
    return Array(solve(problem, Tsit5(), u0=vcat(params[1:3],[1.0, 0.0]), p=params[4:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,:])
    return Qtils.reducedchisquared(μ, averagedata, size(params,1)) + slowrollregulatisation(pred[2,:], pred[3,:], pred[4,:], uniquez), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    println("Parameters: ", p[1:3])
    println("Dark matter density parameter: ", 1 - Ω_ϕ(pred[2,:], pred[4,:], pred[3,:], uniquez)[1])
    return l < 0.05
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=500)

res = solve(problem, Tsit5(), u0=vcat(result.minimizer[1:3],[1.0, 0.0]), p=result.minimizer[4:end], saveat=uniquez)

plot1 = Plots.scatter(
            data.z, data.mu, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

potential = res[3,:]
EoS = Qtils.calculateEOS(potential, res[2,:], res[4,:], uniquez)
slowroll = Qtils.slowrollsatisfied(potential, dV, result.minimizer[4:end], res[1,:], verbose=true)
plot1 = Plots.plot!(plot1, uniquez, mu(uniquez, res[end,:]), label="fit")
plot2 = Plots.plot(uniquez, EoS, title="Equation of State", xlabel="redshift z", ylabel="equation of state w", legend=:topright)
plot3 = Plots.plot(uniquez, potential, title="Potential", xlabel="reshift z", ylabel="V(ϕ)", legend=:bottomright)
plot4 = Plots.plot(uniquez, 1 .- Ω_ϕ(res[2,:], res[4,:], potential, uniquez), title="Density Evolution", xlabel="redshift z", ylabel="density parameter Ω", label="Ω_m")
plot4 = Plots.plot!(plot4, uniquez, Ω_ϕ(res[2,:], res[4,:], potential, uniquez), legend=:topright, label="Ω_ϕ")

println("Cosmological parameters: ")
println("Dark matter density parameter Ω_m = ", 1 - Ω_ϕ(res[2,:], res[4,:], potential, uniquez)[1])
println("Initial conditions for quintessence field = ", result.minimizer[1:3])
println("Slowroll satisfied for ϵ and η: ", slowroll)

m_ϕ =  Zygote.gradient(Q -> dV(Q, result.minimizer[4:end])[1], res[1,1])[1][1]
println("Mass of the scalar field = ", sqrt(abs(m_ϕ)))

resultplot = Plots.plot(plot1, plot2, plot3, plot4, layout=(2, 2), size=(1200, 800))

