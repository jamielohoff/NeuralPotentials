using Flux, DiffEqFlux, DifferentialEquations, Zygote
using DataFrames, CSV, Plots, Statistics, Measures, LaTeXStrings
include("../Qtils.jl")
include("../AwesomeTheme.jl")
using .Qtils

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

# sndatapath = joinpath(@__DIR__, "supernovae.csv")
# sndata = CSV.read(sndatapath, delim=' ', DataFrame) 
# rename!(sndata,:my => :mu)
# uniquez = unique(sndata.z)
# averagedata = Qtils.preparedata(sndata,uniquez)

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data,uniquez)

const H0 = 0.06766 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr

Ω_m_0 = 0.25 .+  0.75 .* rand(Float32, 1) # 0.3111 # 
ps = vcat(Ω_m_0, -1.0.*rand(Float32, 2))
zspan = (0.0, 7.0)
u0 = [Ω_m_0, 1.0, 0.0]

# we have a +25 instead of -5 because we measure distances in Mpc
mu(z, χ) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* χ)) .+ 25.0 

# Defining the time-dependent equation of state
w_DE(z, p) = p[1] + p[2] * z/(1+z)

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Ω_m = u[1]
    E = u[2]
    χ = u[3]

    Ω_DE = 1 - Ω_m
    dE = 1.5*E/(1+z) * (Ω_m + (1.0 + w_DE(z, p))*Ω_DE)
    
    du[1] = (3/(1+z) - 2*dE/E) * Ω_m
    du[2] = dE
    du[3] = 1/E
end

problem = ODEProblem(friedmann!, u0, zspan, ps[2:end])
opt = ADAM(1e-3)

function predict(params)
    u0 = [params[1], 1.0, 0.0]
    return Array(solve(problem, Tsit5(), u0=u0, p=Zygote.@showgrad(params[2:end]), saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,:])
    return Qtils.reducedχ2(μ, averagedata, size(params,1)), pred
end

epoch = 0

cb = function(p, l, pred)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    println("Parameters: ", p)
    global epoch += 1
    return l < 1.0075
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=20000)

u0 = [result.minimizer[1], 1.0, 0.0]
res = solve(problem, Tsit5(), u0=u0, p=result.minimizer[2:end], saveat=uniquez)

plot1 = Plots.scatter(
            data.z, data.mu, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

plot1 = Plots.plot!(plot1, uniquez, mu(uniquez, res[end,:]), label="fit")
EoS = [w_DE(z, result.minimizer[2:end]) for z ∈ uniquez]

plot2 = Plots.plot(uniquez, EoS, title="Equation of State w")
plot3 = Plots.plot(uniquez, res[1,:], title="Density evolution", xlabel="redshift z", ylabel="density parameter Ω", label="Ω_DE")
plot3 = Plots.plot!(plot3, uniquez, 1 .- res[1,:], label="Ω_m")

println("Cosmological parameters: ")
println("Mass parameter  Ω_m = ", result.minimizer[1])
println("Parameters of the equation of state w_0, w_a: ", result.minimizer[2:end])

plot(plot1, plot2, plot3, layout=(3, 1), legend=:bottomright, size=(1600, 1200))

