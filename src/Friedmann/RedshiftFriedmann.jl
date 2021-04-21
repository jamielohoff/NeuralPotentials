using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots

data, uniquez = MyUtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")

const H0 = 0.069 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 1.0 # in Mpc^3 / (Gy^2 * eV)
const rho_c_0 = 3*H0^2/(8pi*G) # Definition of the critical density

u0 = [H0, 0.0]
tspan = (0.0, 7.0)

ps = vcat(p)
mu(z, d_L) = 5.0 .* log10.((1.0 .+ z) .* d_L) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

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

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    H = u[1]
    d_L = u[2]
    
    # p[1] = omega_m_0, p[2] = w
    # H = H0*sqrt((1-p[1])*(1+z)^(3-3*p[2]) + p[1]*(1+z)^3)
    omega_m = p[1]*(1+z)^3
    omega_DE = (1-p[1])*(1+z)^(3-3*p[2])
    du[1] = 1.5*H0^2/(H*(1+z)) * (omega_m + (1-p[2])*omega_DE)
    du[2] = c/H
end

problem = ODEProblem(friedmann!, u0, tspan, p)
opt = ADAM(1e-2, (0.85, 0.9))

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

res = solve(problem, Tsit5(), p=result.minimizer, saveat=uniquez)
println("Best result: ", result.minimizer)

Plots.scatter(
            data.z, data.my, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="apparent magnitude μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

Plots.plot!(uniquez, mu(uniquez, res[2,:]), label="fit")

