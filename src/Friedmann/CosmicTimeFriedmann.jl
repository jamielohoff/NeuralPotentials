using Flux, DiffEqFlux, DifferentialEquations, QuadGK
using DataFrames, CSV, Plots
include("../Qtils.jl")
using .Qtils

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")

const H0 = 0.069 # 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 1.0 # in Mpc^3 / (Gy^2 * eV)
const rho_c_0 = 3*H0^2/(8pi*G) # Definition of the critical density

p = 0.25 .+  0.75 .* rand(Float32, 2) # [0.3, 1.0] # 
u0 = [1.0, 0.0]
tspan = (0.0, 13.4)

params = Flux.params(p)

dadt(a, p) = H0*sqrt( p[1] / a + (1.0-p[1]) * a^(3.0*p[2]-1.0) )
mu(z, d_L) = 5.0 .* log10.((1.0 .+ z) .* d_L) .+ 25.0 # we have a +25 instead of -5 because we measure distances in Mpc

# 1st order ODE for Friedmann equation in terms of a
function friedmann!(du,u,p,t)
    a = u[1]
    d_L = u[2]
    
    # p[1] = omega_DE_0, p[2] = w
    du[1] = -dadt(a, p)
    du[2] = c/a
end

problem = ODEProblem(friedmann!, u0, tspan, p)

function cosmologicaltime(z, p)
    # time is measured in Gyr
    integral, err = quadgk(a -> 1.0/dadt(a, p), 0.0, 1.0/(1.0+z), rtol=1e-8)
    return integral
end

function timelist(zlist, p)
    tlist = []
    cosmologicalage = cosmologicaltime(0.0, p)
    for z in zlist
        tstep = cosmologicalage - cosmologicaltime(z, p)
        push!(tlist, tstep)
    end
    return Array(tlist)
end

# Now we tell Flux how to train the neural network
iterator = Iterators.repeated((), 500)
opt = ADAM(1e-3, (0.85, 0.9))

cb = function()
    display(p)
end

function custom_train!(params, iterator, opt; cb)
    z = 0.0
    µ = 0.0
    for iter in iterator
        tlist = unique(timelist(data.z, params[1]))
        grads = Flux.gradient(params) do
            pred = solve(remake(problem,tspan=(0.0,tlist[end])), Tsit5(), p=params[1], saveat=tlist)
            z = 1.0 ./ pred[1,:] .- 1.0
            d_L = pred[2,:]
            µ = mu(z, d_L)
            loss = -1.0 * sum(abs2, µ .- data.my)
            println(loss)
            return loss
        end
        cb()
        Flux.update!(opt, params, grads)
    end
    return z, µ
end

@time z, µ = custom_train!(params, iterator, opt; cb)

plot1 = scatter(
            data.z, data.mu, 
            title="Redshift-Magnitude Data",
            xlabel="redshift z",
            ylabel="apparent magnitude μ",
            yerror=data.me,
            label="data",
            legend=:bottomright
)

plot2 = plot(z, µ, label="fit")

plot(plot1, plot2, layout = (2, 1), legend = false)

