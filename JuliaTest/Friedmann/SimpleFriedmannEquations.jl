using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots

sndata = CSV.read(raw"D:\Masterthesis\JuliaTest\Friedmann\supernovae.data", delim=' ', DataFrame) # supernova data
grbdata = CSV.read(raw"D:\Masterthesis\JuliaTest\Friedmann\grbs.data", delim=' ', DataFrame) # gamma-ray bursts

data = outerjoin(sndata,grbdata,on=[:z,:my,:me])

H0 = 0.069 # 1 / Gyr
c = 306.4 # in Mpc / Gyr
ps = 0.5 .* rand(Float32, 3) .+ 0.25 # [0.7, H0, 0.0, 1.0] 
omega_DM_0 = ps[1]
u0 = [omega_DM_0, 0.0]
tspan = (0.0, 7.5)

# 1st order ODE for Friedmann equation in terms of z
# all parameters should be of the same order

function Friedmann!(du,u,p,z)
    omega_m = u[1]
    omega_DM = u[2]
    d_L = u[3]
    # mass fractions omega_m and omega_DM
    du[1] = 3*omega_m/(1+z)
    du[2] = 3*(1-p[2])*omega_DM/(1+z)
    # Hubble function
    H = p[1] * sqrt(omega_DM + omega_m)
    # luminosity distance d_L
    du[3] = c/H 
end

problem = ODEProblem(Friedmann!, u0, tspan, ps)

function mu(z, d_L)
     5.0 .* log10.((1 .+ z) .* d_L) .+ 25.0 # we have a + 25 instead of -5 because we measure distances in Mpc
end

function loss(params)
    pred = solve(problem, Tsit5(), u0 =[1 - params[1], params[1], 0.0], p=params[2:3], saveat=data.z)
    loss = sum(abs2, mu(data.z,pred[3,:]) .- data.my)
    return loss, pred
end

# Now we tell Flux how to train the neural network
opt = ADAM(1e-3)

cb = function(p, l, pred)
    display(l)
    display(p)
    # display(plot(sndata.zsn, mu(sndata.zsn , pred[3,:])))
    return false
end

@time result_ode = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=5000)

println("Initial Condition: ", ps)
println("Best result: ", result_ode.minimizer)

remade_solution = solve(remake(problem, u0=[1-result_ode.minimizer[1], result_ode.minimizer[1], 0.0], p=result_ode.minimizer[2:3]), Tsit5(), saveat=data.z)

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

plot!(data.z,mu(data.z, remade_solution[3,:]), label="fit")

