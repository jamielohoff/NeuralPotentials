using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics
include("../Qtils.jl")
using .MyUtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 1.0 # in Mpc^3 / (Gy^2 * eV)
const rho_c_0 = 3*H0^2/(8pi*G) # Definition of the critical density

p = 0.25 .+  0.75 .* rand(Float32, 1)
u0 = [H0, 0.0]
tspan = (0.0, 7.0)

mu(z, d_L) = 5.0 .* log10.(abs.((1.0 .+ z) .* d_L)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

# Defining the time-dependent equation of state
w_DE = FastChain(
    FastDense(1, 16, relu),
    FastDense(16, 1) # choose output function such that 0 < |w| < 1
)

ps = vcat(p, initial_params(w_DE))

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    H = u[1]
    d_L = u[2]
    
    # p[1] = omega_m_0, p[2] = w
    w = w_DE(z, p[2:end])[1]
    omega_m = p[1]*(1+z)^3
    omega_DE = (1-p[1])*(1+z)^(3 + 3*w)
    du[1] = 1.5*H0^2/(H*(1+z)) * (omega_m + (1 + w)*omega_DE)
    du[2] = c/H
end

problem = ODEProblem(friedmann!, u0, tspan, p)

function predict(params)
    return Array(solve(problem, Tsit5(), p=params, saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[2,:])
    return sum(abs2, µ .- averagedata.mu), pred
    # return MyUtils.reducedchisquared(µ, averagedata), pred
end

cb = function(p, l, pred)
    display(l)
    display(p[1])
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

plot1 = Plots.plot!(plot1, uniquez, mu(uniquez, res[2,:]), label="fit")
EoS = map(z -> w_DE(z, result.minimizer[2:end])[1], uniquez)
plot2 = Plots.plot(uniquez, EoS, title="Equation of State w", xlabel="redshift z", ylabel="equation of state w")

println("Cosmological parameters: ")
println("Mass parameter Ω_m = ", result.minimizer[1])
println("Average equation of state w = ", mean(EoS))

plot(plot1, plot2, layout=(2, 1), legend=:bottomright, size=(1200, 800))

