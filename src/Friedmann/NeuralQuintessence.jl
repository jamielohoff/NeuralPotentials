using Flux, DiffEqFlux, DifferentialEquations, Zygote
using DataFrames, Plots, LinearAlgebra, Statistics, Measures
include("../Qtils.jl")
include("../AwesomeTheme.jl")
using .Qtils

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.06766 # in 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.491e-53 # in Mpc^3 / (Gyr^2 * planck mass)

# we cannot vary the initial conditions much, otherwise we get inconsistent results!!!
Ω_m_0 = 0.3111 # From Planck + WMAP + BAO, so statistically independent from SNIa
p = [1.0, 0.0, 0.0]
u0 = vcat(p[1:3], [1.0, 0.0])
zspan = (0.0, 7.0)
# Function to calculate the distance modulus
# we have a +25 instead of -5 because we measure distances in Mpc
mu(z, χ) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* χ)) .+ 25.0 

# Function to calculate the density paramter of quintessence
Ω_ϕ(dQ, E, V, z) = 8π/3 .* (0.5 .* ((1 .+ z).*dQ).^2 .+ V./ E.^2)

function slowrollregulatisation(dQ, V, E, z)
    return sum(((1 .+ z).*E.*dQ).^2 ./ V) / size(z, 1)
end

function bound(x)
    b = 0.0
    if x < 0.0
        b = exp(-100.0*x) - 1.0
    end
    return b
end

dV = FastChain(
    FastDense(1, 16, relu),
    FastDense(16, 16, relu),
    FastDense(16, 1)
)

ps = vcat(p, initial_params(dV))

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Q = u[1]
    dQ = u[2]
    V = u[3]
    E = u[4]
    χ = u[5]
    
    Ω_m = 1.0 - Ω_ϕ(dQ, E, V, z) # 0.3111/E^2 * (1+z)^3 # 
    dE = 1.5*(E/(1+z)) * (Ω_m + 8π/3 * ((1+z)*dQ)^2)

    du[1] = dQ
    du[2] = (2.0/(1+z) - dE/E) * dQ - dV(Q, p)[1]/(E*(1+z))^2
    du[3] = dV(Q, p)[1] * dQ
    du[4] = dE
    du[5] = 1.0/E
end

problem = ODEProblem(friedmann!, u0, zspan, p)
opt = ADAM(1e-2)

function predict(params)
    return Array(solve(problem, Tsit5(), u0=vcat(params[1:3],[1.0, 0.0]), p=params[4:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,:])
    # Ω_m = 0.3111./pred[4,:].^2 .* (1.0.+uniquez).^3
    D = pred[3,:] .- pred[2,:].^2 ./ c^2
    # return Qtils.χ2(μ, averagedata) + 100.0 * sum(abs2, Ω_m .+ Ω_ϕ(pred[2,:], pred[4,:], pred[3,:], uniquez) .- 1.0) + 100.0 * sum(abs2, bound.(pred[3,:])), pred
    return Qtils.χ2(μ, averagedata) + 100.0 * sum(abs2, bound.(D)), pred
end

epoch = 0
cb = function(p, l, pred)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    println("Parameters: ", p[1:3])
    println("Density parameter Ω_ϕ = ", Ω_ϕ(pred[2,:], pred[4,:], pred[3,:], uniquez)[1])
    if mod(epoch, 5) == 0
        display(plot(uniquez, pred[3,:]))
    end
    global epoch += 1
    return false
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=500)

res = solve(problem, Tsit5(), u0=vcat(result.minimizer[1:3],[1.0, 0.0]), p=result.minimizer[4:end], saveat=uniquez)

μ_plot = Plots.scatter(
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
# slowroll = Qtils.slowrollsatisfied(V, result.minimizer[3:end], res[1,:], verbose=true)

μ_plot = plot!(μ_plot, uniquez, mu(uniquez, res[end,:]), label="fit")
EoS_plot = plot(uniquez, EoS, title="Equation of State", xlabel="redshift z", ylabel="equation of state w", legend=:topright)
V_plot = plot(uniquez, potential, title="Potential", xlabel="reshift z", ylabel="V", legend=:bottomright)
Ω_plot = plot(uniquez, 1.0 .- Ω_ϕ(res[2,:], res[4,:], potential, uniquez), title="Density Evolution", xlabel="redshift z", ylabel="density parameter Ω", label="Ω_m")
Ω_plot = plot!(Ω_plot, uniquez, Ω_ϕ(res[2,:], res[4,:], potential, uniquez), legend=:topright, label="Ω_ϕ")

println("Cosmological parameters: ")
println("Matter density parameter Ω_DE = ", Ω_ϕ(res[2,:], res[4,:], potential, uniquez)[1])
println("Initial conditions for quintessence field = ", result.minimizer[1:2])
# println("Slowroll satisfied for ϵ and η: ", slowroll)

resultplot = plot(μ_plot, EoS_plot, V_plot, Ω_plot, layout=(2, 2), size=(1200, 1600))
savefig(resultplot, "NeuralQuintessencePlot.pdf")

