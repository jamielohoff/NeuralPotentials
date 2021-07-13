using Zygote : @adjoint
using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics, SpecialFunctions, QuadGK
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.475e-53 # in Mpc^3 / (Gyr^2 * planck mass)

z0 = 0.64
β = 2.0
q0 = β/(z0 * gamma(3/β))
χ_H = c/H0

Ω_m_0 = 0.25 .+  0.75 .* rand(Float32, 1)
u0 = [Ω_m_0, 1.0, 1.0, 0.0, 0.0]
zspan = (0.0, 7.0)

mu(z, χ) = 5.0 .* log10.((c/H0)*abs.((1.0 .+ z) .* χ)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc
q(z) = q0*(z/z0)^2*exp(-(z/z0)^β)
T2(q) = log(1+2.34*q)^2/((2.34*q)^2 * sqrt(1 + 3.89*q + (16.1*q)^2 + (5.46*q)^3 + (6.71*q)^4))
P(k, ns) = k^ns * T2(k)


function G_(z::Real, χ::Real)
    integral, err = quadgk(x -> q(z)*(x - χ)/x, χ, χ_H, rtol=1e-8)
    return integral
end

function W_(χ::Real, z::Real, D::Real, Ω_m::Real)   
    return 1.5*(Ω_m*D*(1+z)*χ)/χ_H^2 * G_(χ, z)
end

# Defining the time-dependent equation of state
w_DE = FastChain(
    FastDense(1, 4, relu),
    FastDense(4, 4, relu),
    FastDense(4, 1, tanh) # choose output function such that -1 < w < 1
)

ps = vcat(Ω_m_0, initial_params(w_DE))

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Ω_m = u[1]
    E = u[2]
    D = u[3]
    dD = u[4]
    χ = u[5]

    println(W_(χ,z,D,Ω_m))

    println(D, dD)

    Ω_DE = 1 - Ω_m
    dE = 1.5*E/(1+z) * (Ω_m + (1 + w_DE(z, p)[1])*Ω_DE)
    
    du[1] = (3/(1+z) - 2*dE/E) * Ω_m
    du[2] = dE
    du[3] = dD
    du[4] = 1.5 * Ω_m*D/(1+z)^2 - (dE/((1+z)^2*E) - 1/(1+z))*dD
    du[5] = 1/E
end

problem = ODEProblem(friedmann!, u0, zspan, ps)

function predict(params)
    u0 = [params[1], 1.0, 1.0, 0.0, 0.0]
    return Array(solve(problem, Tsit5(), u0=u0, p=params[2:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,:])
    return Qtils.reducedχ2(μ, averagedata, size(params,1)), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    println("Parameters: ", p[1])
    return l < 1.0
end

opt = ADAM(1e-2)
@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=300)

u0 = [result.minimizer[1], 1.0, 1.0, 0.0, 0.0]
res = solve(problem, Tsit5(), u0=u0, p=result.minimizer[2:end], saveat=uniquez)

μ_plot = scatter(
            data.z, data.mu, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

μ_plot = plot!(μ_plot, uniquez, mu(uniquez, res[end,:]), label="fit")
EoS = map(z -> w_DE(z, result.minimizer[2:end])[1], uniquez)
EoS_plot = plot(uniquez, EoS, title="Equation of State w", xlabel="redshift z", ylabel="equation of state w")
Ω_plot = plot(uniquez, res[1,:], title="Density evolution", xlabel="redshift z", ylabel="density parameter Ω", label="Ω_m")
Ω_plot = plot!(Ω_plot, uniquez, 1 .- res[1,:], label="Ω_DE")

println("Cosmological parameters: ")
println("Mass parameter Ω_m = ", result.minimizer[1])

plot(μ_plot, EoS_plot, Ω_plot, layout=(3, 1), legend=:bottomright, size=(1600, 1200))

