using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots

data, uniquez = MyUtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = preparedata(data)

const H0 = 0.069 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 1.0 # in Mpc^3 / (Gy^2 * eV)
const rho_c_0 = 3*H0^2/(8pi*G) # Definition of the critical density

p = 0.25 .+  0.75 .* rand(Float32, 3) # [0.3, 1.0] # 
u0 = [H0, 1.0, 0.0, 0.0] # [H, phi, dphi, d_L]
tspan = (0.0, 7.0)

ps = vcat(u0, p)
mu(z, d_L) = 5.0 .* log10.(abs.((1.0 .+ z) .* d_L)) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

V(phi, p) = p[1]*exp(-p[2]*phi)*phi^2
dV(phi, p) = Flux.gradient(x -> V(x,p), phi)[1]

# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    H = u[1]
    phi = u[2]
    dphi = u[3]
    d_L = u[4]
    
    # p[1] = omega_m_0
    omega_m = p[1]*(1+z)^3
    dH = 1.5*H0^2/(H*(1+z)) * (omega_m + H^2*(1+z)^2*dphi^2/rho_c_0)
    du[1] = dH
    du[2] = dphi
    du[3] = -dphi*((1+z)*dH - 2*H + dH/H + 1/(1+z)) - 1/(H*(1+z))*dV(phi, p[2:end])
    du[4] = c/H
end

problem = ODEProblem(friedmann!, u0, tspan, p)
opt = ADAM(1e-2, (0.85, 0.9))

function predict(params)
    return Array(solve(problem, Tsit5(), u0=[H0, params[2], params[3], 0.0], p=params[5:end], saveat=uniquez))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[4,:])
    return sum(abs2, µ .- averagedata.mu), pred
end

cb = function(p, l, pred)
    println("Loss: ", l)
    println("Parameters: ", p)
    return false
end

@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=300)

res = solve(problem, Tsit5(), u0=result.minimizer[1:4], p=result.minimizer[5:end], saveat=uniquez)
println("Best result: ", result.minimizer)

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
pot = map(x -> V(x, result.minimizer[6:end])[1], res[2,:])
EoS = MyUtils.calculateEOS(pot, res[3,:])
plot2 = Plots.plot(uniquez, EoS, title="Equation of State")
plot3 = Plots.plot(res[2,:], pot, title="Potential")

plot(plot1, plot2, plot3, layout=(3, 1), legend=:bottomright)

