using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots

sndata = CSV.read(raw"D:\Masterthesis\JuliaTest\Friedmann\supernovae.data", delim=' ', DataFrame) # supernova data
grbdata = CSV.read(raw"D:\Masterthesis\JuliaTest\Friedmann\grbs.data", delim=' ', DataFrame) # gamma-ray bursts

data = outerjoin(sndata,grbdata,on=[:z,:my,:me])

H0 = 0.069 # 1 / Gyr
c = 306.4 # in Mpc / Gyr
ps = 0.5 .* rand(Float32, 4) .+ 0.25 # [0.3, 0.7, H0, 1.0] #  
omega_m0 = ps[1]
phi0 = ps[2]
dphi0 = ps[3]
u0 = [omega_m0, phi0, dphi0, H0, 0.0]
tspan = (0.0, 7.5)

dV = FastChain(
    FastDense(1, 25, sigmoid),
    FastDense(25, 1)
)

ps = vcat(ps, initial_params(dV))

# 1st order ODE for Friedmann equation in terms of z
# All parameters should be of the same order
function Friedmann!(du,u,p,z)
    # p = [w_DE]
    omega_m = u[1]
    phi = u[2]
    dphi = u[3]
    H = u[4]
    d_L = u[5]

    dH = (H/(1+z)) * ( 0.5 * (omega_m + 2.0 * (dphi^2 + dV([phi], p)[1])) + 1.0 )

    # Mass fractions omega_m and omega_DE
    du[1] = 3*omega_m/(1+z)
    du[2] = dphi
    du[3] = (2/(1+z) + dH/H) * dphi - dV([phi], p)[1] / ((1+z)^2*H^2)
    # Hubble function
    du[4] = dH
    # Luminosity distance d_L
    du[5] = c/H 
end

problem = ODEProblem(Friedmann!, u0, tspan, ps)

function mu(z, d_L)
     5.0 .* log10.((1 .+ z) .* d_L) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc
end

function loss(params)
    pred = solve(problem, Tsit5(), u0 =[params[1], params[2], params[3], params[4], 0.0], p=params[5:end], saveat=data.z)
    loss = sum(abs2, mu(data.z, pred[4,:]) .- data.my)
    return loss, pred
end

# Now we tell Flux how to train the neural network
opt = ADAM(1e-3, (0.8, 0.999))

cb = function(p, l, pred)
    display(l)
    display(p)
    display(plot(data.z, mu(data.z , pred[4,:])))
    return false
end

@time result_ode = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=5000)

println("Initial Condition: ", ps)
println("Best result: ", result_ode.minimizer)

remade_solution = solve(remake(problem, u0=[ps[1], ps[2], ps[3], 0.0], p=ps[4:end]), Tsit5(), saveat=data.z)

scatter!(
    sndata.z, sndata.my, 
    title="Supernova Data",
    xlabel="redshift z",
    ylabel="apparent magnitude μ",
    yerror=sndata.me,
    label="supernovae data",
    legend=:bottomright
)

scatter!(
    grbdata.z, grbdata.my, 
    yerror=grbdata.me,
    label="gamma-ray bursts"
)

plot!(data.z,mu(data.z, remade_solution[4,:]), label="fit")

