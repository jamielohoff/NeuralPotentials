using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 1.0 # in Mpc^3 / (Gy^2 * eV)

tspan = (0.0, 7.0)
u0 = [0.3, H0, 0.0]

mu(z, d_L) = 5.0 .* log10.(abs.((1.0 .+ z) .* d_L)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

# Defining the time-dependent equation of state
w_DE = FastChain(
    FastDense(1, 16, relu),
    FastDense(16, 1, tanh) # choose output function such that -1 < w < 1
)

p = vcat(rand(Float32, 1), initial_params(w_DE))

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Ω_m = u[1]
    H = u[2]
    d_L = u[3]

    Ω_DE = 1 - Ω_m 
    dH = 1.5*H/(1+z) * (Ω_m + (1 + w_DE(z, p)[1])*Ω_DE)
    
    du[1] = (3/(1+z) - 2*dH/H) * Ω_m
    du[2] = dH
    du[3] = c/H
end

problem = ODEProblem(friedmann!, u0, tspan, p)

function predict(params)
    u0 = [params[1], H0, 0.0]
    return Array(solve(problem, Tsit5(), u0=u0, p=params[2:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[3,:])
    return sum(abs2, µ .- averagedata.mu), pred
    # return Qtils.reducedchisquared(µ, averagedata), pred
end

cb = function(p, l, pred)
    # display(l)
    # display(p[1])
    return l < 47.0
end

### Bootstrap Loop
itmlist = DataFrame(params = Array[], Ω = Array[], d_L = Array[], EoS = Array[])
repetitions = 64

dplot = scatter(
            data.z, data.my, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

println("Beginning Bootstrap...")
lk = ReentrantLock()
@time Threads.@threads for rep in 1:repetitions
    sampledata = Qtils.sample(averagedata, 0.5)

    p = 0.25 .+  0.75 .* rand(Float32, 1)
    params = vcat(p, initial_params(w_DE))
    opt = ADAM(1e-2)
    @time result =  DiffEqFlux.sciml_train(loss, params, opt, cb=cb, maxiters=5)

    u0 = [result.minimizer[1], H0, 0.0]
    res = solve(problem, Tsit5(), u0=u0, p=result.minimizer[2:end], saveat=uniquez)

    EoS = map(z -> w_DE(z, result.minimizer[2:end])[1], uniquez)

    lock(lk)
    push!(itmlist, [[result.minimizer[1]], res[1,:], mu(uniquez,res[3,:]), EoS])
    unlock(lk)
    println("Done!")
end
println("Bootstrap complete!")

# CSV.write(joinpath(@__DIR__, "Bootstrap.CSV"), itmlist)

mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_d_L, std_d_L, CI_d_L = Qtils.calculatestatistics(itmlist.d_L)
mean_EoS, std_EoS, CI_EoS = Qtils.calculatestatistics(itmlist.EoS)
mean_Ω, std_Ω, CI_Ω = Qtils.calculatestatistics(itmlist.Ω)

println("Cosmological parameters: ")
println("Mass parameter Ω_m = ", mean_params[1], "±", std_params[1])

dplot = plot!(dplot, uniquez, mean_d_L, ribbon=CI_d_L, label="fit")
EoS_plot = plot(uniquez, mean_EoS, ribbon=CI_EoS, 
                title="Equation of State w", 
                xlabel="redshift z", 
                ylabel="equation of state w",
                label="w",
                legend=:topright
)
Ω_plot = plot(uniquez, mean_Ω, ribbon=CI_Ω, 
                title="Density evolution", 
                xlabel="redshift z", 
                ylabel="density parameter Ω", 
                label="Ω_m"
)
plot(dplot, EoS_plot, Ω_plot, layout=(3,1), size=(1200, 800))

