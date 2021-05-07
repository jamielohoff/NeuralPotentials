using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics
include("../MyUtils.jl")
using .MyUtils

data, uniquez = MyUtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")

const H0 = 0.074 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 1.0 # in Mpc^3 / (Gy^2 * eV)
const rho_c_0 = 3*H0^2/(8pi*G) # Definition of the critical density

p = 0.25 .+  0.75 .* rand(Float32, 3) .- [0, 1, 1]
u0 = [H0, 0.0]
tspan = (0.0, 7.0)

ps = vcat(p)
mu(z, d_L) = 5.0 .* log10.(abs.((1.0 .+ z) .* d_L)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

function preparedata(data)
    averagedata = []
    for z in uniquez
        idx = findall(x -> x==z, data.z)
        avg = sum([data.my[i] for i in idx]) / length(idx)
        push!(averagedata, avg)
    end
    return averagedata
end

averagemu = preparedata(data)

# Defining the time-dependent equation of state
w_DE(z, p) = p[1] + p[2] * log(1+z)

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    H = u[1]
    d_L = u[2]
    
    # p[1] = omega_m_0, p[2] = w
    omega_m = p[1]*(1+z)^3
    omega_DE = (1-p[1])*(1+z)^(3 + 3*w_DE(z, p[2:end]))
    du[1] = 1.5*H0^2/(H*(1+z)) * (omega_m + (1 + w_DE(z, p[2:end]))*omega_DE)
    du[2] = c/H
end

problem = ODEProblem(friedmann!, u0, tspan, p)
opt = ADAM(1e-2)

function predict(params)
    return Array(solve(problem, Tsit5(), p=params, saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[2,:])
    return sum(abs2, µ .- averagemu), pred
end

cb = function(p, l, pred)
    display(l)
    display(p)
    return false
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=1000)

res = solve(problem, Tsit5(), u0=u0, p=result.minimizer, saveat=uniquez)
println("Best result: ", result.minimizer)

plot1 = Plots.scatter(
            data.z, data.my, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="apparent magnitude μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

plot1 = Plots.plot!(plot1, uniquez, mu(uniquez, res[2,:]), label="fit")
w = map(z -> w_DE(z, result.minimizer[2:end]), uniquez)
plot2 = Plots.plot(uniquez, w, title="Equation of State w")

println("Cosmological parameters: ")
println("Mass parameter omega_m = ", result.minimizer[1])
println("Parameters of the equation of state w: ", result.minimizer[2:end])
println("Average equation of state w = ", mean(w))

plot(plot1, plot2, layout=(2, 1), legend=:bottomright, size=(1200, 800))

