using Zygote: Array, Threads
using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, LinearAlgebra, Statistics, DataFrames, CSV
include("../Qtils.jl")
include("../MechanicsDatasets.jl")
using .Qtils
using .MechanicsDatasets

### Creation of synthethetic data---------------------- #
tspan = (0.0, 5.0)
t0 = Array(range(tspan[1], tspan[2], length=256))
u0 = [3.0, 0.0] # contains x0 and dx0
p = [1.0, 0.1]

# Define potential and get dataset
V0(x,p) = p[1]*x^2 
fulldata = MechanicsDatasets.potentialproblem(V0, u0, p, t0, addnoise=true, σ=0.2)
### End ------------------------------------------------ #

# Defining the gradient of the potential
dV = FastChain(
    FastDense(1, 8, relu), 
    FastDense(8, 1)
)

# Define ODE for our system
function neuraloscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -dV(x, p)[1]
end

prob = ODEProblem(neuraloscillator!, u0, tspan, p)

function predict(params, t)
    return Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=t))
end

function loss(params, sampledata)
    pred = predict(params, sampledata[1,:])
    return sum(abs2, (pred .- sampledata[2:3,:])./0.2) / (size(sampledata, 2) - size(params, 1)), pred
end

cb = function(p,l,pred)
    println("Loss at thread ", Threads.threadid(), " : ", l)
    println("Params at thread ", Threads.threadid(), " : ", p[1:2])
    # display(scatter(pred[1,:]))
    return l < 1.1
end

function mytrain!(sampledata, loss, params, opt, cb; epochs=1000)
    for epoch in 1:epochs
        println("epoch: ", epoch, " of ", epochs)
        _loss, _pred = loss(params, sampledata)
        grads = Flux.gradient(p -> loss(p, sampledata)[1], params)[1]
        Flux.update!(opt, params, grads)

        if cb(params, _loss, _pred)
            break
        end
    end
end

sampledata = Qtils.sample(fulldata, 0.5)
plot(sampledata[1,:], sampledata[2,:])
otherparams = rand(Float32, 2) .+ [1.5, 0.0] # contains u0, du0
ps = vcat(otherparams, initial_params(dV))

opt = ADAM(0.1)
@time mytrain!(sampledata, loss, ps, opt, cb, epochs=300)

### Bootstrap Loop ----------------------------------------------------- #
# itmlist = DataFrame(params = Array[], potential = Array[], trajectories = Array[])
# repetitions = 1

# x0 = Array(range(-u0[1], u0[1], step=0.05))
# true_potential = map(x -> V0(x, p), x0)
# traj_plot = scatter(t0, fulldata[2,:])
# pot_plot = plot(x0, true_potential)

# println("Beginning Bootstrap...")
# lk = ReentrantLock()
# opt = ADAM(0.1)
# @time Threads.@threads for rep in 1:repetitions
#     sampledata = Qtils.sample(fulldata, 0.9)

#     display(scatter(sampledata[1,:], sampledata[2,:]))

#     otherparams = rand(Float32, 2) .+ [1.5, 0.0] # contains u0, du0
#     ps = vcat(otherparams, initial_params(dV))

#     @time mytrain!(sampledata, loss, ps, opt, cb, epochs=300)

#     result = Array(solve(prob, Tsit5(), u0=ps[1:2], p=ps[3:end], saveat=t0))
#     potential = map(x -> Qtils.integrateNN(dV, x, ps[3:end]), x0)

#     lock(lk)
#     push!(itmlist, [ps[1:2], potential, result[1,:]])
#     unlock(lk)
#     println("Repetition ", rep, " is done!")
# end
# println("Bootstrap complete!")

# # CSV.write(joinpath(@__DIR__, "Bootstrap.CSV"), itmlist)

# mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
# mean_traj, std_traj, CI_traj = Qtils.calculatestatistics(itmlist.trajectories)
# mean_pot, std_pot, CI_pot = Qtils.calculatestatistics(itmlist.potential)

# println("parameter mean: ", mean_params, "±", std_params)

# traj_plot = plot!(traj_plot, t0, mean_traj, ribbon=CI_traj)
# pot_plot = plot!(pot_plot, x0, mean_pot, ribbon=CI_pot)
# resultplot = plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 800))

