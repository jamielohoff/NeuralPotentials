using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.475e-53 # in Mpc^3 / (Gyr^2 * planck mass)

u0 = [1.0, 0.0]
p = vcat([3.0f0, 0.0f0])
tspan = (0.0, 7.0)

mu(z, d_L) = 5.0 .* log10.((c/H0)*abs.((1.0 .+ z) .* d_L)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc
Ω_ϕ(dQ, E, V) = 8pi/3 .* (0.5 .* dQ.^2 .+ V)./ E.^2

V = FastChain(
    FastDense(1, 8, relu),
    # FastDense(8, 8, relu),
    FastDense(8, 1, exp) # maybe choose sigmoid as output function to enforce positive potentials only
)

dV(Q, p) = Flux.gradient(q -> V(q, p)[1], Q)[1]

ps = vcat(p, initial_params(V))

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Q = u[1]
    dQ = u[2]
    E = u[3]
    d_L = u[4]
    
    Ω_m = 1 - 8pi/3 * (0.5*dQ^2 .+ V(Q, p)[1])/E^2
    dE = 1.5*E/(1+z)*(Ω_m + 8pi/3 * ((1+z)*dQ/E)^2)

    du[1] = dQ
    du[2] = (2/(1+z) - dE/E) * dQ - dV(Q, p)[1]/(E*(1+z))^2
    du[3] = dE
    du[4] = 1/E
end

problem = ODEProblem(friedmann!, u0, tspan, p)

function predict(params)
    u0 = vcat(params[1:2],[1.0, 0.0])
    return Array(solve(problem, Tsit5(), u0=u0, p=params[3:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,:])
    return Qtils.reducedchisquared(μ, averagedata, size(params,1)), pred
end

cb = function(p, l, pred)
    # println("Loss: ", l)
    # println("Parameters: ", p[1:2])
    return l < 1.20
end

### Bootstrap Loop
itmlist = DataFrame(params = Array[], Q = Array[], Ω_ϕ = Array[], d_L = Array[], EoS = Array[], potential = Array[])
repetitions = 8

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

    p = [3.0f0, 0.0f0]
    ps = vcat(p, initial_params(V))
    opt = ADAM(1e-2)
    @time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=500)

    u0 = vcat(result.minimizer[1:2], [1.0, 0.0])
    res = solve(problem, Tsit5(), u0=u0, p=result.minimizer[3:end], saveat=uniquez)

    potential = map(q -> V(q, result.minimizer[3:end])[1], res[1,:])
    EoS = Qtils.calculateEOS(potential, res[2,:])
    density_ϕ = Ω_ϕ(res[2,:], res[3,:], potential)

    lock(lk)
    push!(itmlist, [[result.minimizer[1:2]], res[1,:], density_ϕ, mu(uniquez,res[end,:]), EoS, potential])
    unlock(lk)
    println("Repetition ", rep, " is done!")
end
println("Bootstrap complete!")

# CSV.write(joinpath(@__DIR__, "Bootstrap.CSV"), itmlist)

# mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_d_L, std_d_L, CI_d_L = Qtils.calculatestatistics(itmlist.d_L)
mean_EoS, std_EoS, CI_EoS = Qtils.calculatestatistics(itmlist.EoS)
mean_Ω_ϕ, std_Ω_ϕ, CI_Ω_ϕ = Qtils.calculatestatistics(itmlist.Ω_ϕ)
# mean_V, std_V, CI_V = Qtils.calculatestatistics(itmlist.potential)


println("Cosmological parameters: ")
# println("Mass parameter Ω_m = ", mean_params[3], "±", std_params[3])
println("Initial conditions of scalar field: ", )
#println("Q_0: = ", mean_params[1], "±", std_params[1])
#println("dQ_0: = ", mean_params[2], "±", std_params[2])


dplot = plot!(dplot, uniquez, mean_d_L, ribbon=CI_d_L, label="fit")
EoS_plot = plot(uniquez, mean_EoS, ribbon=CI_EoS, 
                title="Equation of State w", 
                xlabel="redshift z", 
                ylabel="equation of state w",
                label="w",
                legend=:topright
)
Ω_plot = plot(uniquez, mean_Ω_ϕ, ribbon=CI_Ω_ϕ, 
                title="Density evolution", 
                xlabel="redshift z", 
                ylabel="density parameter Ω", 
                label="Ω_ϕ"
)
# V_plot = plot(uniquez, mean_V, ribbon=CI_V, 
#                 title="Potential", 
#                 xlabel="redshift z", 
#                 ylabel="potential", 
#                 label="Ω_ϕ"
# )
plot(dplot, EoS_plot, Ω_plot, layout=(3,1), size=(1200, 800))

