using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.475e-53 # in Mpc^3 / (Gyr^2 * planck mass)

u0 = [1.0, 0.0]
p = vcat([3.0, 0.0, 0.0])
tspan = (0.0, 7.0)

mu(z, χ) = 5.0 .* log10.((c/H0)*abs.((1.0 .+ z) .* χ)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc
Ω_ϕ(dQ, E, V, z) = 8π/3 .* (0.5 .* ((1 .+ z).*E.*dQ).^2 .+ V)./ E.^2

dV = FastChain(
    FastDense(1, 4, relu),
    FastDense(4, 4, relu),
    FastDense(4, 4, relu),
    FastDense(4, 1) 
)

ps = vcat(p, initial_params(dV))

function slowrollregulatisation(dQ, V, E, z)
    return sum(((1 .+ z).*E.*dQ).^2 ./ V) / size(z, 1)
end

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Q = u[1]
    dQ = u[2]
    V = u[3]
    E = u[4]
    χ = u[5]
    
    Ω_m = 0 # 1 - Ω_ϕ(dQ, E, V, z)
    dE = 1.5*(E/(1+z)) * (Ω_m + 8π/3 * ((1+z)*dQ)^2)

    du[1] = dQ
    du[2] = (2/(1+z) - dE/E) * dQ - dV(Q, p)[1]/(E*(1+z))^2
    du[3] = dV(Q, p)[1] * dQ
    du[4] = dE
    du[5] = 1/E
end

problem = ODEProblem(friedmann!, u0, tspan, p)

### Bootstrap Loop
itmlist = DataFrame(params = Array[], Q = Array[], Ω_ϕ = Array[], μ = Array[], EoS = Array[], potential = Array[])
repetitions = 16

μ_plot = scatter(
            data.z, data.mu, 
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
    println("Starting repetition ", rep, " at thread ID ", Threads.threadid())
    df = Qtils.elaboratesample(data, uniquez, 1.0)
    sampledata = Qtils.resample(df)

    function predict(params)
        return Array(solve(problem, Tsit5(), u0=vcat(params[1:3],[1.0, 0.0]), p=params[4:end], saveat=sampledata.z))
    end
    
    function loss(params)
        pred = predict(params)
        µ = mu(sampledata.z, pred[end,:])
        # + slowrollregulatisation(pred[2,:], pred[3,:], pred[4,:], sampledata.z)
        # + abs(Ω_ϕ(pred[2,:], pred[4,:], pred[3,:], sampledata.z)[1] - 1 + 0.311)
        return Qtils.reducedχ2(μ, sampledata, size(params,1)), pred
    end
    
    cb = function(p, l, pred)
        # println("Loss: ", l)
        # println("Parameters: ", p[1:3])
        # println("Dark matter density parameter: ", 1 - Ω_ϕ(pred[2,:], pred[4,:], pred[3,:], sampledata.z)[1])
        return l < 1.05
    end

    p = [3.0, 0.0, 0.0]
    ps = vcat(p, initial_params(dV))
    opt = ADAM(1e-2)
    @time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=500)

    u0 = vcat(result.minimizer[1:3], [1.0, 0.0])
    res = Array(solve(problem, Tsit5(), u0=u0, p=result.minimizer[4:end], saveat=uniquez))

    EoS = Qtils.calculateEOS(res[3,:], res[2,:], res[4,:], uniquez)
    density_ϕ = Ω_ϕ(res[2,:], res[4,:], res[3,:], uniquez)

    lock(lk)
    push!(itmlist, [result.minimizer[1:2], res[1,:], density_ϕ, mu(uniquez,res[end,:]), EoS, res[3,:]])
    unlock(lk)
    println("Repetition ", rep, " is done!")
end
println("Bootstrap complete!")

CSV.write(joinpath(@__DIR__, "Bootstrap.CSV"), itmlist)

mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_μ, std_μ, CI_μ = Qtils.calculatestatistics(itmlist.μ)
mean_EoS, std_EoS, CI_EoS = Qtils.calculatestatistics(itmlist.EoS)
mean_Ω_ϕ, std_Ω_ϕ, CI_Ω_ϕ = Qtils.calculatestatistics(itmlist.Ω_ϕ)
mean_V, std_V, CI_V = Qtils.calculatestatistics(itmlist.potential)


println("Cosmological parameters: ")
println("Density parameter Ω_ϕ = ", mean_Ω_ϕ[1], "±", std_Ω_ϕ[1])
println("Initial conditions of scalar field: ", )
println("Q_0 = ", mean_params[1], "±", std_params[1])
println("dQ_0 = ", mean_params[2], "±", std_params[2])


μ_plot = plot!(μ_plot, uniquez, mean_μ, ribbon=CI_μ, label="fit")
EoS_plot = plot(uniquez, mean_EoS, ribbon=CI_EoS, 
                title="Equation of State w", 
                xlabel="redshift z", 
                ylabel="equation of state w",
                label="w",
                legend=:topright
)
Ω_plot = plot(uniquez, zeros(541), ribbon=CI_Ω_ϕ, 
                title="Density evolution", 
                xlabel="redshift z", 
                ylabel="density parameter Ω", 
                label="Ω_m"
)
Ω_plot = plot!(Ω_plot, uniquez, mean_Ω_ϕ, ribbon=CI_Ω_ϕ, label="Ω_ϕ")
V_plot = plot(uniquez, mean_V, ribbon=CI_V, 
                title="Potential", 
                xlabel="redshift z", 
                ylabel="potential V", 
                label="V"
)
result_plot = plot(μ_plot, EoS_plot, Ω_plot, V_plot, layout=(2,2), size=(1600, 1200))
savefig(result_plot, "64_sample_Quintessence_0_omega_m.png")

