using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, Statistics, LinearAlgebra, LaTeXStrings, DataFrames, Distributions, Measures, CSV
include("SagittariusData.jl")
include("../MechanicsDatasets.jl")
include("../AwesomeTheme.jl")
include("../Qtils.jl")
using .SagittariusData
using .MechanicsDatasets
using .Qtils

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

## Sagittarius A* System ###
const c = 306.4 # milliparsec per year
const G = 4.49 # gravitational constant in new units: (milliparsec)^3 * yr^-2 * (10^6*M_solar)^-1
const D_Astar = 8.178 * 1e6 # distance of Sagittarius A* in mpc

### Initialisation of the Sagittarius data ------------------------ #

path = joinpath(@__DIR__, "SagittariusData.csv")
S2data = SagittariusData.loadstar(path, "S2", timestamps=true)
S2 = SagittariusData.orbit(S2data, D_Astar)
path = joinpath(@__DIR__, "SagittariusOrbitalElements.csv")
S2_orbitalelements = SagittariusData.loadorbitalelements(path, "S2")

rbf(x) = sqrt(1.0 + 0.25*x^2)

dV = FastChain(
    FastDense(1, 8, celu), # celu
    FastDense(8, 8, rbf), # rbf
    FastDense(8, 1)
)

star = sort(S2, [:t])

Δϕ = star.ϕ[2:end] .- star.ϕ[1:end-1]
idx = findall(x -> abs(x) > 4.5, Δϕ)[1]
star1 = star[1:idx,:]
star2 = star[idx+1:end,:]
star2.ϕ = star2.ϕ .+ 2π
sort!(star1, :ϕ)
sort!(star2, :ϕ)

star = outerjoin(star1,star2,on=[:r,:ϕ,:t,:x_err,:y_err])
unique!(star, [:ϕ])

println(star)

prograde = SagittariusData.isprograde(star.ϕ)
println("isprograde: ", prograde)
phase = 0
if !prograde
    phase = 0.5
end

### End -------------------------------------------------- #
ps = vcat(0.5.*rand(Float64, 2), 0.5*rand(Float64, 1), 2.0*rand(Float64, 2), initial_params(dV))
println(180.0*ps[3:5])

function neuralkepler!(du, u, p, ϕ)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = -G*dV(U, p)[1] - U
end

u0 = ps[1:2]
ϕspan = (0.0, 10π)
problem = ODEProblem(neuralkepler!, u0, ϕspan, ps[6:end]) 

function predict(params)
    s, θ = SagittariusData.inversetransform(params[3:5].*π, star.r, star.ϕ, prograde)
    pred = Array(solve(problem, Tsit5(), u0=params[1:2], p=params[6:end], saveat=θ))
    r, ϕ = SagittariusData.transform(params[3:5].*π, 1.0./pred[1,:], θ, prograde)
    ra, dec = SagittariusData.converttoangles(r, ϕ, D_Astar)
    return vcat(reshape(r,1,:), reshape(ϕ,1,:), reshape(ra,1,:), reshape(dec,1,:))
end

function χ2(r, ϕ)
    return sum(abs2, (r.*cos.(ϕ) .- star.r.*cos.(star.ϕ))./star.x_err) + sum(abs2, (r.*sin.(ϕ) .- star.r.*sin.(star.ϕ))/star.y_err)
end

function loss(params) 
    pred = predict(vcat(params[1:4], Zygote.hook(x -> 1e12*x, params[5]), params[6:end]))
    # return χ2(pred[1,:], pred[2,:]), pred
    return sum(abs2, pred[1,:].-star.r), pred
end

opt = NADAM(0.01)
epoch = 0

cb = function(p,l,pred)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    println("Initial conditions: ", p[1:2])
    println("Rotation angles: ", 180.0 * vcat(mod.(p[3], 0.5) + phase, mod.(p[4], 1), mod.(p[5], 2)))
    println("Initial values: ", 180.0*ps[3:5])
    # println("Parameters of the potential: ", p[6:end])

    if epoch % 1 == 0
        angular_plot = scatter(S2data.RA, S2data.DEC, xerror=S2data.RA_err, yerror=S2data.DEC_err, label="S2 data")
        angular_plot = plot!(angular_plot, pred[3,:], pred[4,:], 
                            label="Prediction of the Trajectory",
                            xlabel=L"\textrm{Right ascension } \alpha \textrm{ [mas]}",
                            ylabel=L"\textrm{Declination } \delta \textrm{ [mas]}",
                            title="Trajectory of the Star S2",
                            legend=:bottomleft
        )

        u0 = range(0.1, 2.3, step=0.01)
        potential = -G*[Qtils.integrateNN(dV,p[6:end],0.0,u)[1] for u ∈ u0]
        v0 = 7.481 # from GRAVITY Collaboration
        r0 = 0.5915
        true_potential = G.*4.08.*(1.0/(r0*v0)^2 .* u0 .- u0.^3 ./ c^2)

        pot_plot = plot(u0, true_potential, label="Expected potential from GR")
        pot_plot = plot!(pot_plot, u0, potential, 
                            label="Prediction of the neural network",
                            xlabel=L"u \textrm{ coordinate [} \textrm{mpc}^{-1} \textrm{]}",
                            ylabel=L"\textrm{Potential } \frac{\mu}{L_z^2}V(1/u)",
                            color=colorant"#c67",
                            margin=12mm,
                            legend=:bottomright
        )
        result_plot = plot(angular_plot, pot_plot, layout=(1,2), size=(2100, 1200), margin=12mm)
        display(plot(result_plot))
    end
    global epoch+=1
    return l < 3.0
    
end

@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=3000)

s, θ = SagittariusData.inversetransform(result.minimizer[3:5].*π, star.r, star.ϕ, prograde)
θrange = Array(range(minimum(θ), maximum(θ), length=300))
res = solve(problem, Tsit5(), u0=result.minimizer[1:2], p=result.minimizer[6:end], saveat=θrange)
r, ϕ = SagittariusData.transform(result.minimizer[3:5].*π, 1.0./res[1,:], θrange, prograde)
ra, dec = SagittariusData.converttoangles(r, ϕ, D_Astar)

angular_plot = scatter(S2data.RA, S2data.DEC, 
                        xerror=S2data.RA_err, yerror=S2data.DEC_err,
                        label="S2 data",
                        title="Angular Trajectory of the Star S2",
                        xlabel=L"\textrm{Right ascension } \alpha \textrm{ [mas]}",
                        ylabel=L"\textrm{Declination } \delta \textrm{ [mas]}",
                        legend=:bottomleft
)
angular_plot = plot!(angular_plot, ra, dec, label="Prediction using neural potential")

u0 = range(0.1, 2.3, step=0.01)

v0 = 7.481 # from GRAVITY Collaboration
r0 = 0.5915
true_potential = G.*4.08.*(1.0/(r0*v0)^2 .* u0 .- u0.^3 ./ c^2)
potential = -G*[Qtils.integrateNN(dV,result.minimizer[6:end],0.0,u)[1] for u ∈ u0]

pot_plot = plot(u0, true_potential, label="Expected potential from GR")
pot_plot = plot!(pot_plot, u0, potential, 
                    label="Prediction of the neural network",
                    xlabel=L"u \textrm{ coordinate [} \textrm{mpc}^{-1} \textrm{]}",
                    ylabel=L"\textrm{Potential } \frac{\mu}{L_z^2}V(1/u)",
                    color=colorant"#c67",
                    legend=:bottomright,
                    margin=12mm,
)

result_plot = plot(angular_plot, pot_plot, layout=(1,2), size=(2100, 1200), margin=12mm)
savefig(result_plot, "SagittariusNeuralFit.pdf")

