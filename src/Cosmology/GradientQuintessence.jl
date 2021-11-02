using Flux, DiffEqFlux, DifferentialEquations, Zygote
using DataFrames, Plots, LinearAlgebra, Statistics, Measures, LaTeXStrings
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
p = [1.0, 0.0]
u0 = vcat(p[1:2], [Ω_m_0, 1.0, 0.0])
zspan = (0.0, 7.0)
# Function to calculate the distance modulus
# we have a +25 instead of -5 because we measure distances in Mpc
mu(z, χ) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* χ)) .+ 25.0 
# Function to calculate the density paramter of quintessence
Ω_ϕ(dQ, E, V, z) = 8π/3 .* (0.5 .* ((1 .+ z).*dQ).^2 .+ V./ E.^2)

V(x, p) = p[1] * (1.0 + cos(x/p[2])) # p[1]*exp(-p[2]*x)
dV(x, p) = Zygote.gradient(x -> V(x, p), x)[1]

ps = vcat(p, 0.05.*rand(Float64,1), rand(Float64,1))

function slowrollregulatisation(dQ, V, E, z)
    return sum(((1 .+ z).*E.*dQ).^2 ./ V) / size(z, 1)
end

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Q = u[1]
    dQ = u[2]
    E = u[3]
    χ = u[4]
    
    Ω_m = 1.0 - Ω_ϕ(dQ, E, V(Q, p), z)
    dE = 1.5*(E/(1+z)) * (Ω_m + 8π/3 * ((1+z)*dQ)^2)

    du[1] = dQ
    du[2] = (2.0/(1+z) - dE/E) * dQ - dV(Q, p)[1]/(E*(1+z))^2
    du[3] = dE
    du[4] = 1.0/E
end

problem = ODEProblem(friedmann!, u0, zspan, p)
opt = ADAM(1e-2)

function predict(params)
    return Array(solve(problem, Tsit5(), u0=vcat(params[1:2],[1.0, 0.0]), p=params[3:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,:])
    return Qtils.χ2(μ, averagedata), pred
end

epoch = 0

cb = function(p, l, pred)
    # println("Epoch: ", epoch)
    # println("Loss: ", l)
    # println("Initial conditions: ", p[1:2])
    # println("Parameters: ", p[3:end])
    # potential = [V(q, p[3:end]) for q ∈ pred[1,:]]
    # # display(plot(uniquez, v))
    # println("Density parameter Ω_ϕ = ", Ω_ϕ(pred[2,:], pred[3,:], potential, uniquez)[1])
    global epoch += 1
    return l < 532.0
end

println("Starting training...")
@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=15000)

res = solve(problem, Tsit5(), u0=vcat(result.minimizer[1:2],[1.0, 0.0]), p=result.minimizer[3:end], saveat=uniquez)

μ_plot = scatter(data.z, data.mu, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

potential = [V(q,result.minimizer[3:end]) for q ∈ res[1,:]]
EoS = Qtils.calculateEOS(potential, res[2,:], res[3,:], uniquez)
# slowroll = Qtils.slowrollsatisfied(V, result.minimizer[3:end], res[1,:], verbose=true)

μ_plot = plot!(μ_plot, uniquez, mu(uniquez, res[end,:]), label="fit")
EoS_plot = plot(uniquez, EoS, title="Equation of State", xlabel="redshift z", ylabel="equation of state w", legend=false)
V_plot = plot(res[1,:], potential, title="Potential", xlabel="field amplitude ϕ", ylabel="V", legend=false)
Ω_plot = plot(uniquez, 1.0 .- Ω_ϕ(res[2,:], res[3,:], potential, uniquez), title="Density Evolution", xlabel="redshift z", ylabel="density parameter Ω", label="Ω_m")
Ω_plot = plot!(Ω_plot, uniquez, Ω_ϕ(res[2,:], res[3,:], potential, uniquez), legend=:topright, label="Ω_ϕ")
Q_plot = plot(uniquez, res[1,:], xlabel="redshift z", ylabel="Field Amplitude Q", legend=false)
dQ_plot = plot(uniquez, res[2,:], xlabel="redshift z", ylabel="dQ/dz", legend=false)
println("Cosmological parameters: ")
println("Dark matter density parameter Ω_m = ", 1.0 - Ω_ϕ(res[2,:], res[3,:], potential, uniquez)[1])
println("Initial conditions for quintessence field = ", result.minimizer[1:2])
# println("Slowroll satisfied for ϵ and η: ", slowroll)

resultplot = plot(μ_plot, Ω_plot, V_plot, EoS_plot, Q_plot, dQ_plot, layout=(3, 2), size=(1200, 1600))
savefig(resultplot, "GradientQuintessence.pdf")

