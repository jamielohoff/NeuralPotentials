using Plots
using Flux
using DiffEqFlux
using DifferentialEquations
using DataFrames
using CSV
using Dierckx

sndata = CSV.read(raw"D:\Masterthesis\JuliaTest\Friedmann\supernovae.data", delim=' ', DataFrame) # supernova data
grbdata = CSV.read(raw"D:\Masterthesis\JuliaTest\Friedmann\grbs.data", delim=' ', DataFrame) # gamma-ray bursts

data = outerjoin(sndata,grbdata,on=[:z,:my,:me])

H0 = 0.069 # 1 / Gyr
c = 306.4 # in Mpc / Gyr
p = [0.3, 0.1, 1.0, 10*H0]

# 1st order ODE for Friedmann equation in terms of z
# all parameters should be of the same order
omega_DM_0 = 1 - p[1] - 1e-3*p[2]
function Friedmann!(du,u,p,z)
    omega_DM = u[1]
    H = u[2]
    d_L = u[3]

    du[1] = 3*(1-p[3])*omega_DM/(1+z)
    du[2] = (0.1*p[4])^2/(H*(1+z)) * (3*p[1]*(1+z)^3 + 1e-3*5*p[2]*(1+z)^4 + 3*(1-2*p[3])*omega_DM)
    du[3] = c/H
end

u0 = [omega_DM_0, 0.1*p[4], 0.0]
tspan = (0.0,7.5)
problem = ODEProblem(Friedmann!, u0, tspan, p)

function mu(z, d_L)
     5 .* log10.((1 .+ z) .* d_L) .+ 25 # we have a + 25 instead of -5 because we measure distances in Mpc
end

function loss(params)
    pred = solve(problem, Tsit5(), u0=[1 - params[1] - 1e-3*params[2], 0.1*params[4], 0.0], p=params, saveat=data.z)
    loss = sum(abs2, mu(data.z,pred[3,:]) .- data.my)
    return loss, pred
end

# Now we tell Flux how to train the neural network
opt = ADAM(1e-4)

cb = function(p, l, pred)
    display(l)
    display(p)
    # display(plot(sndata.zsn, mu(sndata.zsn , pred[2,:])))
    return false
end

@time result_ode = DiffEqFlux.sciml_train(loss, p, opt, cb=cb, maxiters=5000)

println("Initial Condition: ", p)
println("Best result: ", result_ode.minimizer)

remade_solution = solve(remake(problem, p=result_ode.minimizer), Tsit5(), saveat=data.z)

scatter(
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

plot!(data.z,mu(data.z,remade_solution[3,:]))

