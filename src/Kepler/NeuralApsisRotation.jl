using Flux: Zygote
using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, Statistics, LinearAlgebra, LaTeXStrings
include("../Qtils.jl")
include("../MechanicsDatasets.jl")
using .Qtils
using .MechanicsDatasets

### Stellar Black Hole System ###

const c = 9.454e9 # 63241 # megameters per year
const G = 1.32e17# gravitational constant: Mm^3 * yr^-2 * M_sol^-1
const M = 8.0 # measure mass in multiples of solar mass
const m = 0.5 # measure mass in multiples of solar mass
const rs = 2*G*M/c^2 # Schwarzschild radius in Mm
μ = m*M/(m + M)
println(rs)

### Creation of synthethetic data --------------------------------- #

r0 = 2.0 # measure radius in mega meters
true_v0 = 0.6 * sqrt(G*M/r0)/sqrt(1-rs/r0)
true_u0 =  [1/r0, 0] # 

true_p = [G*M/(true_v0*r0)^2, 3*G*(M+m)*m^2/(c^2*μ)]
ϕspan = (0.0, 7.5*pi)
ϕ = Array(range(ϕspan[1], ϕspan[2], length=256))
stderror = 0.01
V0(x,p) = -p[1]*x - p[2]*x^3 + 0.5*x^2
dV0(x,p) = Zygote.gradient(x -> V0(x,p)[1], x)[1]
data = MechanicsDatasets.potentialproblem(V0, true_u0, true_p, ϕ, addnoise=true, σ=stderror)
R_gt = 1 ./ data[2,:] # convert into Radii

### End ----------------------------------------------------------- #

dV = FastChain(
    FastDense(1, 4, relu),
    FastDense(4, 4, relu),
    FastDense(4, 4, relu),
    FastDense(4, 1)
)

otherparams = rand(Float64, 2) # otherparams = [u0, p]
ps = vcat(otherparams, initial_params(dV))

function neuralkepler!(du, u, p, ϕ)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = -dV(U, p)[1] - U
end

prob = ODEProblem(neuralkepler!, ps[1:2], ϕspan, ps[3:end])

function predict(params)
    return Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=ϕ))
end

function loss(params) 
    pred = predict(params)
    # return Qtils.reducedchisquared(pred, data[2:3,:], size(params, 1), stderror), pred
    return mean((pred .- data[2:3, :]).^2), pred
end

opt = RMSProp(1e-2)

cb = function(p,l,pred)
    display(l)
    display(p[1:2])
    R = 1 ./ pred[1, :] # convert into Radii
    traj_plot = plot(R .* cos.(ϕ), R .* sin.(ϕ), 
                    xlims=(-r0, r0), 
                    ylims=(-r0, r0),
                    xlabel=L"x\textrm{ coordinate in }10^6 m",
                    ylabel=L"y\textrm{ coordinate in }10^6 m",
                    title="position of the test mass and potential")
    traj_plot = scatter!(traj_plot, R_gt .* cos.(ϕ), R_gt .* sin.(ϕ))

    # Plotting the potential
    u0 = Array(range(0.1, r0, step=0.001))
    y0 = map(x -> Qtils.integrateNN(dV, p[3:end], 0.0, x) + 0.5*x^2, 1 ./ u0)
    z0 = map(x -> V0(x, true_p), 1 ./ u0)
    #y0 = map(x -> dV(x, p[3:end])[1], u0)
    #z0 = map(x -> dV0(x, true_p), u0)

    pot_plot = plot(u0, y0, ylims=(-10.0, 20.0),
                    xlabel=L"\textrm{radius } R \textrm{ in } 10^6 m",
                    ylabel=L"\textrm{potential } V \textrm{ in } ")
    pot_plot = plot!(pot_plot, u0, z0)

    u_plot = plot(ϕ, pred[1,:],
                    xlabel=L"\textrm{angle } \phi",
                    ylabel=L"\textrm{inverse radius} \dfrac{1}{r} \textrm{ in } 10^{-6}m^{-1}")
    u_plot = scatter!(ϕ, data[2,:])

    du_plot = plot(ϕ, pred[2,:])
    du_plot = scatter!(ϕ, data[3,:],
                        xlabel=L"\textrm{angle } \phi",
                        ylabel=L"\textrm{inverse radius} \dfrac{1}{r} \textrm{ in } 10^{-6}m^{-1}")

    display(plot(traj_plot, pot_plot, u_plot, du_plot, layout=(2,2), size=(1200, 800), legend=:bottomright))
    return l < 0.001
end

@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=2000)

