using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, Statistics, LinearAlgebra, LaTeXStrings, DataFrames, Distributions
include("SagittariusData.jl")
using .SagittariusData

### Sagittarius A* System ###
const c = 30.64 # centiparsec per year
const G = 4.49e-3 # gravitational constant in new units : (centi-parsec)^3 * yr^-2 * (10^6*M_solar)^-1
### Initialisation of the Sagittarius data ------------------------ #
path = joinpath(@__DIR__, "SagittariusData.csv")
S2data = SagittariusData.loadstar(path, "S2", timestamps=true)
S2 = SagittariusData.orbit(S2data)
S2 = unique!(SagittariusData.centerorbit(S2, sortby=:ϕ), [:ϕ])

S2.r = 100.0 .* S2.r
S2.x = 100.0 .* S2.x
S2.x_err = 100.0 .* S2.x_err
S2.y = 100.0 .* S2.y
S2.y_err = 100.0 .* S2.y_err

# S2max = S2[findall(x -> x > -π/6, S2.ϕ), :]
# S2min = S2[findall(x -> x ≤ -π/6, S2.ϕ), :]

# S2min.ϕ = S2min.ϕ .+ 2π


# S2 = outerjoin(S2max,S2min,on=[:r,:ϕ,:x,:x_err,:y,:y_err,:t])
# ϕ = S2.ϕ
# ϕspan = (minimum(ϕ), maximum(ϕ))
# ϕ0 = Array(range(ϕspan[1], ϕspan[2], length=200))

# ϕ = unique!(sort!(vcat(ϕ, ϕ0)))

# idx = []
# for φ in S2.ϕ
#     push!(idx, findall(x->x==φ, ϕ)[1]) 
# end

ϕ0span = (0.01, 2π-0.01)
ϕ0 = Array(range(ϕ0span[1], ϕ0span[2], length=144))
r0 = 2.0
true_v0 = sqrt(G*4.35/r0) # initial velocity
true_u0 = [1.0/r0, 0.0] 
true_p = [G*4.35/(1.3*true_v0*r0)^2]

function kepler!(du, u, p, ϕ)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = p[1]-U # - G/c^2 * p[1] * U^2
end

problem = ODEProblem(kepler!, true_u0, ϕ0span, true_p)

@time data = Array(solve(problem, Tsit5(), saveat=ϕ0))

function resort(s, θ)
    A = zeros((size(θ,1),2))
    buf = Zygote.Buffer(A)
    arr = hcat(s, θ)
    arr = arr[sortperm(arr[:,2]), :]
    for j in 1:size(θ,1)
        buf[j,:] = arr[j,:]
    end
    res = copy(buf)
    return res[:,1], res[:,2]
end

function transform(angles, r, φ)
    ι = angles[1]
    Ω = mod(angles[2],2π)
    ω = mod(angles[3],2π)

    δ = Ω - ω

    x = r.*cos.(φ)
    y = r.*sin.(φ)

    X = ( cos(Ω)^2*(1-cos(ι)) + cos(ι) ) * x .+ cos(Ω)*sin(Ω)*(1-cos(ι)) * y
    Y = cos(Ω)*sin(Ω)*(1-cos(ι)) * x .+ ( sin(Ω)^2*(1-cos(ι)) + cos(ι) ) * y
    Z = -sin(Ω)*sin(ι) * x .+ cos(Ω)*sin(ι) * y

    ϕ = mod.(atan.(Y,X) .+ δ, 2π)
    R = sqrt.(X.^2 + Y.^2)
    return R, ϕ
end

function inversetransform(angles, r, φ)
    ι = -angles[1]
    Ω = mod(angles[2],2π)
    ω = mod(angles[3],2π)

    δ = Ω - ω

    x = r.*cos.(φ.- δ)
    y = r.*sin.(φ.- δ)
    z = r.*sin.((Ω-π) .- (φ .- δ)).*tan(-ι)

    X = ( cos(Ω)^2*(1-cos(ι)) + cos(ι) ) * x .+ cos(Ω)*sin(Ω)*(1-cos(ι)) * y .+ sin(Ω)*sin(ι) * z
    Y = cos(Ω)*sin(Ω)*(1-cos(ι)) * x .+ ( sin(Ω)^2*(1-cos(ι)) + cos(ι) ) * y .- cos(Ω)*sin(ι) * z

    ϕ = mod.(atan.(Y,X), 2π)
    R = sqrt.(X.^2 + Y.^2)
    return R, ϕ
end

angles = [85/180, 1/3, 0.0]
R, ϕ = transform(angles.*π, 1.0 ./data[1,:], ϕ0)

pdf = Normal(0.0, 0.05)
noise =  rand(pdf, size(R))
R = R .+ noise

# pdf = Normal(0.0, 0.02)
# noise =  rand(pdf, size(ϕ))
# ϕ = ϕ .+ noise

###

orbit = hcat(R, ϕ)
star = DataFrame(orbit, ["r", "ϕ"])

# starmax = star[findall(x -> x > 5π/4, star.ϕ), :]
# starmin = star[findall(x -> x ≤ 5π/4, star.ϕ), :]
# star = outerjoin(starmax,starmin,on=[:r,:ϕ])

### End -------------------------------------------------- #
dV(U,p) = [p[1]] # Removed the gravitational constant G
ps = vcat(rand(Float32, 5), 0.50*rand(Float32, 1) .+ 0.25 ,)
# ps = vcat(0.0, angles, true_p)

function neuralkepler!(du, u, p, ϕ)
    U = u[1]
    dU = u[2]

    du[1] = dU
    du[2] = dV(U, p)[1]-U
end

u0 = vcat(1.0/r0, ps[1])
ϕspan = (0.0, 2π)
prob = ODEProblem(neuralkepler!, u0, ϕspan, ps[6:end])

function predict(params)
    _s, _θ = inversetransform(params[3:5].*π, star.r, star.ϕ)
    s, θ = resort(_s, _θ)
    pred = solve(prob, Tsit5(), u0=params[1:2], p=params[6:end], saveat=θ)
    _R, _ϕ = transform(params[3:5].*π, 1 ./ pred[1,:], _θ)

    Zygote.ignore() do 
        # println("s[1]: ", 1.0/params[1])
        # println("θ - ϕ0: ", θ .- ϕ0)
        # println(" ")
        # println("pred - data:", pred[1,:] .- data[1,:])
    end

    return vcat(reshape(_R,1,:), reshape(_ϕ,1,:))
end

function trigonometricloss(R, ϕ)
    return sum(abs2, R.*cos.(ϕ) .- star.r.*cos.(star.ϕ)) + sum(abs2, R.*sin.(ϕ) .- star.r.*sin.(star.ϕ))
end

function loss(params) 
    pred = predict(params)
    # return sum(abs2, pred[1,:] .- star.r), pred
    # return sum(abs2, 1.0 ./ pred[1,:] .- 1.0 ./ star.r), pred
    return trigonometricloss(pred[1,:], pred[2,:]), pred
end

opt = Flux.Optimiser(ClipValue(1e-5), AMSGrad(1e-3))
# opt = Flux.Optimiser(ClipValue(1e-5), ADAM(1e-3))
# opt = Flux.Optimiser(ClipValue(1e-5), NADAM(0.001))

i = 0

cb = function(p,l,pred)
    println("Epoch: ", i)
    println("Loss: ", l)
    println("Initial conditions: ", p[1:2])
    println("Rotation angles: ", p[3:5], " true value: ", angles)
    println("Parameters of the potential: ", p[6:end], " true value: ", true_p)
    println("Angular fit: ", sum(abs2, star.ϕ .- pred[2,:]))
    if i % 100 == 0
        orbit_plot = plot(cos.(pred[2,:]) .* pred[1,:], sin.(pred[2,:]) .* pred[1,:], xlims=(-7.5, 7.5), ylims=(-7.5, 7.5),
                            label="fit using neural network",
                            xlabel=L"x\textrm{ coordinate in }10^{-2}pc",
                            ylabel=L"y\textrm{ coordinate in }10^{-2}pc",
                            title="position of the test mass and potential"
        )
        orbit_plot = scatter!(orbit_plot, star.r .* cos.(star.ϕ), star.r .* sin.(star.ϕ), label="rotated data")
        orbit_plot = scatter!(orbit_plot, cos.(ϕ0)./data[1,:], sin.(ϕ0)./data[1,:], label="original data")
        orbit_plot = scatter!(orbit_plot, [cos.(star.ϕ[1]).*star.r[1]], [sin.(star.ϕ[1]).*star.r[1]], label="initial point")

        # Plotting the potential
        R0 = Array(range(0.3, 11.5, length=100))
        dv = map(u -> dV(u, p[6:end])[1], 1 ./ R0)
        dv0 = map(u -> dV(u, true_p)[1], 1 ./ R0)
        pot_plot = plot(1 ./ R0, dv, ylims=(-0.1, 0.8))
        pot_plot = plot!(pot_plot, 1 ./ R0, dv0)

        result_plot = plot(orbit_plot, pot_plot, layout=(2,1), size=(1200, 800), legend=:bottomright)
        display(plot(result_plot))
    end
    global i+=1
    return false
    
end

@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=50000)

