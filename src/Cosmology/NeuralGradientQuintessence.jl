using Flux, DiffEqFlux, DifferentialEquations, Zygote, ForwardDiff
using DataFrames, Plots, LinearAlgebra, Statistics, Measures, LaTeXStrings, CSV
include("../lib/Qtils.jl")
include("../lib/AwesomeTheme.jl")
using .Qtils

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.06766 # in 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.475e-53 # in Mpc^3 / (Gyr^2 * planck mass)

eVMpc = 1.564e29
eVmpl = 1.2209e28
eVGyr = 4.791e31

ϕfactor = c/sqrt(G) * sqrt(eVmpl*eVMpc)/eVGyr
Vfactor = H0^2/G * eVmpl/(eVMpc*eVGyr^2)

# We cannot vary the initial conditions much, otherwise we get inconsistent results!!!
Ω_m_0 = 0.3111 # From Planck + WMAP + BAO, so statistically independent from SNIa
p = [3.0, 0.1].*rand(Float64, 2)
u0 = vcat(p[1:2], [1.0, 0.0])
zspan = (0.0, 7.0)

# Function to calculate the distance modulus
# We have a +25 instead of -5 because we measure distances in Mpc
mu(z, χ) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* χ)) .+ 25.0 
# Function to calculate the density paramter of quintessence
Ω_ϕ(dQ, E, V, z) = 8π/3 .* (0.5 .* ((1 .+ z).*dQ).^2 .+ V./ E.^2)

V = FastChain(
    FastDense(1, 4, tanh),
    FastDense(4, 4, tanh),
    FastDense(4, 4, tanh),
    FastDense(4, 1, relu) # maybe choose exp as output function to enforce positive potentials only
)
dV(x, p) = ForwardDiff.gradient(x -> V(x, p)[1], x)[1]

ps = vcat(p, initial_params(V))
 
# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Q = u[1]
    dQ = u[2]
    E = u[3]
    χ = u[4]
    
    Ω_m = 1.0 - Ω_ϕ(dQ, E, V([Q], p)[1], z)
    dE = 1.5*(E/(1+z)) * (Ω_m + 8π/3 * ((1+z)*dQ)^2)

    du[1] = dQ
    du[2] = (2.0/(1+z) - dE/E) * dQ - dV([Q], p)/(E*(1+z))^2
    du[3] = dE
    du[4] = 1.0/E
end

problem = ODEProblem(friedmann!, u0, zspan, p)
opt = ADAM(1e-3)

function predict(params)
    return Array(solve(problem, Tsit5(), u0=vcat(params[1:2],[1.0, 0.0]), p=params[3:end], saveat=zrange))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,indexes])
    return Qtils.χ2(μ, averagedata), pred
end

epoch = 0

cb = function(p, l, pred)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    println("Parameters: ", p[1:2])
    potential = [V(q, p[3:end])[1] for q ∈ pred[1,:]]
    display(plot(zrange, potential))
    Ω_m = 1.0 .- Ω_ϕ(pred[2,indexes], pred[3,indexes], potential[indexes], uniquez)[1]
    println("Density parameter Ω_m = ", Ω_m)
    global epoch += 1
    return false # Ω_m < 0.3111 
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=650)

res = solve(problem, Tsit5(), u0=vcat(result.minimizer[1:2],[1.0, 0.0]), p=result.minimizer[3:end], saveat=zrange)

μ_plot = Plots.scatter(
            data.z, data.mu, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

potential = [V(q,result.minimizer[3:end])[1] for q ∈ res[1,:]]
EoS = Qtils.calculateEOS(potential, res[2,:], res[3,:], zrange)

μ_plot = plot!(μ_plot, zrange, mu(zrange, res[end,:]), label="fit")
EoS_plot = plot(zrange, EoS, title="Equation of State", xlabel="redshift z", ylabel="equation of state w", legend=:topright)
V_plot = plot(zrange, Vfactor.*potential./1e-16, title="Potential", xlabel="Redshift z", ylabel="V/1e-16 in eV4", legend=:bottomright)
Ω_plot = plot(zrange, 1.0 .- Ω_ϕ(res[2,:], res[3,:], potential, zrange), title="Density Evolution", xlabel="Redshift z", ylabel="density parameter Ω", label="Ω_m")
Ω_plot = plot!(Ω_plot, zrange, Ω_ϕ(res[2,:], res[3,:], potential, zrange), legend=:topright, label="Ω_ϕ")

println("Cosmological parameters: ")
println("Dark matter density parameter Ω_m = ", 1.0 .- Ω_ϕ(res[2,:], res[3,:], potential, zrange)[1])
println("Initial conditions for quintessence field = ", ϕfactor.*result.minimizer[1:2]./eVmpl)

result_plot = plot(μ_plot, EoS_plot, V_plot, Ω_plot, layout=(2, 2), size=(1200, 1200))
savefig(result_plot, "NeuralGradientQuintessence.pdf")

