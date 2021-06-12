using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 1e-4 # in Mpc^3 / (Gy^2 * eV)

u0 = [0.3, H0, 0.0]
p = vcat([10.0f0, 0.0f0],rand(Float32, 1))
tspan = (0.0, 7.0)

mu(z, d_L) = 5.0 .* log10.(abs.((1.0 .+ z) .* d_L)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc
Ω_ϕ(dϕ, H, V) = 8pi*G/3 .* (0.5 .* dϕ.^2 .+ V)./ H.^2

dV = FastChain(
    FastDense(1, 16, relu),
    FastDense(16, 1, exp) # maybe choose sigmoid as output function to enforce positive potentials only
)

params = vcat(p, initial_params(dV))

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    ϕ = u[1]
    dϕ = u[2]
    Ω_m = u[3]
    H = u[4]
    d_L = u[5]
    
    dH = 1.5*H/(1+z)*(Ω_m + 8pi*G/3 * ((1+z)*dϕ)^2)

    du[1] = dϕ
    du[2] = -dϕ*((1+z)*dH - 2*H + dH/H + 1/(1+z)) - 1/(H*(1+z))*dV(ϕ, p)[1]
    du[3] = (3/(1+z) - 2*dH/H) * Ω_m
    du[4] = dH
    du[5] = c/H
end

problem = ODEProblem(friedmann!, u0, tspan, p)

function predict(params)
    u0 = vcat(params[1:3],[H0, 0.0])
    return Array(solve(problem, Tsit5(), u0=u0, p=params[4:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[3,:])
    return sum(abs2, µ .- averagedata.mu), pred
end

cb = function(p, l, pred)
    # display(l)
    # display(p[1])
    return l < 47.0
end

### Bootstrap Loop
itmlist = DataFrame(params = Array[], ϕ = Array[], Ω_m = Array[], Ω_ϕ = Array[], d_L = Array[], EoS = Array[], potential = Array[])
repetitions = 6

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

    p = vcat([10.0f0, 0.0f0],rand(Float32, 1))
    params = vcat(p, initial_params(dV))
    opt = ADAM(1e-2)
    @time result =  DiffEqFlux.sciml_train(loss, params, opt, cb=cb, maxiters=5)

    u0 = vcat(result.minimizer[1:3], [H0, 0.0])
    res = solve(problem, Tsit5(), u0=u0, p=result.minimizer[4:end], saveat=uniquez)

    potential = map(x -> Qtils.integrateNN(dV, x, result.minimizer[4:end]), res[1,:])
    EoS = Qtils.calculateEOS(potential, res[2,:])
    density_ϕ = Ω_ϕ(res[2,:], res[4,:], potential)

    lock(lk)
    push!(itmlist, [[result.minimizer[1:3]], res[1,:], res[3,:], density_ϕ, mu(uniquez,res[5,:]), EoS, potential])
    unlock(lk)
    println("Done!")
end
println("Bootstrap complete!")

# CSV.write(joinpath(@__DIR__, "Bootstrap.CSV"), itmlist)

#mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_d_L, std_d_L, CI_d_L = Qtils.calculatestatistics(itmlist.d_L)
mean_EoS, std_EoS, CI_EoS = Qtils.calculatestatistics(itmlist.EoS)
mean_Ω_m, std_Ω_m, CI_Ω_m = Qtils.calculatestatistics(itmlist.Ω_m)
mean_Ω_ϕ, std_Ω_ϕ, CI_Ω_ϕ = Qtils.calculatestatistics(itmlist.Ω_ϕ)


println("Cosmological parameters: ")
# println("Mass parameter Ω_m = ", mean_params[3], "±", std_params[3])
# println("Initial conditions of scalar field: ", )
# println("ϕ_0: = ", mean_params[1], "±", std_params[1])
# println("dϕ_0: = ", mean_params[2], "±", std_params[2])


dplot = plot!(dplot, uniquez, mean_d_L, ribbon=CI_d_L, label="fit")
EoS_plot = plot(uniquez, mean_EoS, ribbon=CI_EoS, 
                title="Equation of State w", 
                xlabel="redshift z", 
                ylabel="equation of state w",
                label="w",
                legend=:topright
)
Ω_plot = plot(uniquez, mean_Ω_m, ribbon=CI_Ω_m, 
                title="Density evolution", 
                xlabel="redshift z", 
                ylabel="density parameter Ω", 
                label="Ω_m"
)
Ω_plot = plot!(Ω_plot, uniquez, mean_Ω_ϕ, ribbon=CI_Ω_ϕ, 
                label="Ω_ϕ"
)
plot(dplot, EoS_plot, Ω_plot, layout=(3,1), size=(1200, 800))

