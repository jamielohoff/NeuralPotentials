using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, Statistics, LinearAlgebra, LaTeXStrings, DataFrames, Distributions, Measures
include("SagittariusData.jl")
include("../MechanicsDatasets.jl")
include("../AwesomeTheme.jl")
using .SagittariusData
using .MechanicsDatasets

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

### Sagittarius A* System ###
const c = 30.64 # centiparsec per year
const G = 4.49e-3 # gravitational constant in new units : (10^-2 parsec)^3 * yr^-2 * (10^6*M_solar)^-1

### Initialisation of the Sagittarius data ------------------------ #

ϕ0span = (0.01, 5π/2-0.01)
ϕ0 = Array(range(ϕ0span[1], ϕ0span[2], length=144))
r0 = 5.946e-2
true_v0 = 0.01666*c # sqrt(G*4.35/r0) # initial velocity
true_u0 = [1.0/r0, 0.0, 0.0] 
λ = 1.0 # controls the eccentricity of the orbit
M = 4.35
true_p = [M, (true_v0*r0)]
println(true_p)
dV(U,p) = G*p[1]*[1/p[2]^2 + 3*U^2/c^2]
data = MechanicsDatasets.keplerproblem(dV, true_u0, true_p, ϕ0)

angles = [132.0, 226.0, 65.0].*π/180 # 
println(angles)
r = 0
ϕ = 0
x_err = ones(size(data.r))
y_err = ones(size(data.r))
if angles[1] > π/2
    angles[1] = angles[1] - π/2
    r, ϕ = SagittariusData.transform(angles, data.r, ϕ0, false)
else
    r, ϕ = SagittariusData.transform(angles, data.r, ϕ0, true)
end

orbit = hcat(r, ϕ, data.t, x_err, y_err)
star = DataFrame(orbit, [:r, :ϕ, :t, :x_err, :y_err])
star = sort!(star, [:t])

prograde = SagittariusData.isprograde(star.ϕ)
println("isprograde: ", prograde)
phase = 0
if !prograde
    phase = 0.5
end

### End -------------------------------------------------- #

ps = vcat(rand(Float64, 1), 0.5*rand(Float64, 1), 2.0*rand(Float64, 2), 4.0 .+ rand(Float64,1), 0.01*rand(Float64,1))
println(180.0*ps[2:4])

function neuralkepler!(du, u, p, ϕ)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = dV(U, p)[1] - U
end

u0 = ps[1:2]
ϕspan = (0.0, 10π)
problem = ODEProblem(neuralkepler!, u0, ϕspan, ps[6:end]) 

function predict(params)
    s, θ = SagittariusData.inversetransform(params[2:4].*π, star.r, star.ϕ, prograde)
    pred = Array(solve(problem, Tsit5(), u0=vcat(1.0/s[1], params[1]), p=params[5:end], saveat=θ))
    r, ϕ = SagittariusData.transform(params[2:4].*π, 1.0./pred[1,:], θ, prograde)
    return vcat(reshape(r,1,:), reshape(ϕ,1,:), reshape(s,1,:), reshape(θ,1,:))
end

function χ2loss(r, ϕ)
    return sum(abs2, (r.*cos.(ϕ) .- star.r.*cos.(star.ϕ))./star.x_err) + sum(abs2, (r.*sin.(ϕ) .- star.r.*sin.(star.ϕ))/star.y_err)
end

function loss(params) 
    pred = predict(vcat(params[1:3], Zygote.@showgrad(Zygote.hook(x -> 1e10*x, params[4])), params[5:end]))
    return χ2loss(pred[1,:], pred[2,:]), pred
end

opt = NADAM(0.01) # Nesterov(1e-8) # 
epoch = 0

cb = function(p,l,pred)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    println("Initial velocity: ", p[1])
    println("Rotation angles: ", 180.0 * vcat(mod.(p[2], 0.5) + phase, mod.(p[3:4], 2)))
    println("Initial values: ", 180.0*ps[2:4])
    println("Parameters of the potential: ", p[5:end])

    if epoch % 10 == 0
        orbit_plot = plot(cos.(pred[2,:]) .* pred[1,:], sin.(pred[2,:]) .* pred[1,:],
                            label="Prediction using neural potential",
                            xlabel=L"x\textrm{ coordinate in }10^{-2}\textrm{pc}",
                            ylabel=L"y\textrm{ coordinate in }10^{-2}\textrm{pc}",
                            title="Position of the Star S2 and Gravitational Potential"
        )
        orbit_plot = scatter!(orbit_plot, star.r .* cos.(star.ϕ), star.r .* sin.(star.ϕ), label="Rotated data")
        orbit_plot = plot!(orbit_plot, pred[3,:] .* cos.(pred[4,:]), pred[3,:] .* sin.(pred[4,:]), label="Prediction of unrotated data")
        orbit_plot = scatter!(orbit_plot, data.r.*cos.(ϕ0), data.r.*sin.(ϕ0), label="Unrotated data")

        result_plot = plot(orbit_plot, layout=(1,1), size=(1200, 1200), legend=:bottomright)
        display(plot(result_plot))
    end
    global epoch+=1
    return l < 5e-6
    
end

_p = copy(ps)
epochs = 300000
for epoch in 1:epochs
    _loss, _pred = loss(_p)
    _g = Flux.gradient(p -> loss(p)[1], _p)[1]
    # if _loss < 300.0
    #     global opt = NADAM(1e-4) # NADAM(1e-4) # 1e-7 
    # end
    Flux.update!(opt, _p, _g)
    breakCondition = cb(_p, _loss, _pred)
    if breakCondition
        break
    end
end

