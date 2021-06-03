using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, Plots, LinearAlgebra, Statistics
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

const H0 = 0.07 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 1.0 # in Mpc^3 / (Gy^2 * whatever) 4.43e44 kg = 2.4e80 eV
const rho_c_0 = 3*H0^2/(8pi*G) # Definition of the critical density

p = rand(Float32, 3)
u0 = [H0, p[1], p[2], 0.0] # [H, phi, dphi, d_L]
tspan = (0.0, 7.0)

# function to calculate the distance modulus
# we have a +25 instead of -5 because we measure distances in Mpc
mu(z, d_L) = 5.0 .* log10.(abs.((1.0 .+ z) .* d_L)) .+ 25.0 

dV = FastChain(
    FastDense(1, 16, relu),
    FastDense(16, 1, sigmoid) # maybe choose sigmoid as output function to enforce positive potentials only
)

ps = vcat(p, initial_params(dV))

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    H = u[1]
    phi = u[2]
    dphi = u[3]
    d_L = u[4]
    
    # p[1] = omega_m_0
    omega_m = p[1]*(1+z)^3
    dH = 1.5*H0^2/(H*(1+z))*(omega_m + (H*(1+z)*dphi)^2/rho_c_0)
    du[1] = dH
    du[2] = dphi
    du[3] = -dphi*((1+z)*dH - 2*H + dH/H + 1/(1+z)) - 1/(H*(1+z))*dV(phi, p[2:end])[1]
    du[4] = c/H
end

problem = ODEProblem(friedmann!, u0, tspan, p)
opt = ADAM(1e-2)

function predict(params)
    return Array(solve(problem, Tsit5(), u0=vcat([H0],params[1:2],[0.0]), p=params[3:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[4,:])
    return sum(abs2, µ .- averagedata.mu), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    println("Parameters: ", p[1:3])
    return l < 47.0
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=500 )

res = solve(problem, Tsit5(), u0=vcat([H0],result.minimizer[1:2],[0.0]), p=result.minimizer[3:end], saveat=uniquez)

plot1 = Plots.scatter(
            data.z, data.my, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="distance modulus μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

plot1 = Plots.plot!(plot1, uniquez, mu(uniquez, res[4,:]), label="fit")
pot = map(x -> Qtils.integrateNN(dV, x, result.minimizer[4:end]), res[2,:])
EoS = Qtils.calculateEOS(pot, res[3,:])
slowroll = Qtils.calculateslowroll(dV, result.minimizer[4:end], res[2,:], G)
println("slowroll: ", slowroll)
plot2 = Plots.plot(uniquez, EoS, title="Equation of State w", xlabel="redshift z", ylabel="equation of state w")
plot3 = Plots.plot(res[2,:], pot, title="Potential", xlabel="quintessence field ϕ", ylabel="V(ϕ)")

println("Cosmological parameters: ")
println("Mass parameter Ω_m = ", result.minimizer[3])
println("Average equation of state w = ", mean(EoS))
println("Initial conditions for quintessence field = ", result.minimizer[1:2])

m_phi = Flux.gradient(phi -> dV(phi, result.minimizer[4:end])[1], 0)[1][1]
println("Mass of the scalar field = ", sqrt(abs(m_phi)))

resultplot = Plots.plot(plot1, plot2, plot3, layout=(1, 3), legend=:bottomright, size=(1600, 600))
Plots.plot(resultplot)
# savefig(resultplot, string(@__DIR__) * "NeuralQuintessence.png")

