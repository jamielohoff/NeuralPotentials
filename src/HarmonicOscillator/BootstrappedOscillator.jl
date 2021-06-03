using Zygote: Array
using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, LinearAlgebra, Statistics, DataFrames, CSV
include("../Qtils.jl")
include("../MechanicsDatasets.jl")
using .Qtils
using .MechanicsDatasets

### Creation of synthethetic data---------------------- #
tspan = (0.0f0, 5.0f0)
t0 = Array(range(tspan[1], tspan[2], length=256))
u0 = [3.0f0, 0.0f0] # contains x0 and dx0
p = [1.0f0, 0.1f0]

# Define potential and get dataset
V0(x,p) = p[1]*x^2 
fulldata = MechanicsDatasets.potentialproblem(V0, u0, p, t0, addnoise=true, noisescale=0.2f0)
### End ------------------------------------------------ #

# Defining the gradient of the potential
dV = FastChain(
    FastDense(1, 16, relu), 
    FastDense(16, 1)
)

# Define ODE for our system
function neuraloscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -dV(x,p)[1]
end

prob = ODEProblem(neuraloscillator!, u0, tspan, p)

function predict(params, t)
    return Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=t))
end

function loss(params, sampledata)
    pred = predict(params, sampledata[1,:])
    return mean((pred .- sampledata[2:3,:]).^2), pred
end

cb = function(p,l,pred)
    println("Loss at thread ", Threads.threadid(), " : ", l)
    return l < 0.05
end

opt = ADAM(0.1)

function mytrain!(sampledata, loss, params, opt, cb; epochs=1000)
    for epoch in 1:epochs
        # println("epoch: ", epoch, " of ", epochs)
        _loss, _pred = loss(params, sampledata)
        grads = Flux.gradient(p -> loss(p, sampledata)[1], params)[1]
        Flux.update!(opt, params, grads)

        breakCondition = cb(params, _loss, _pred)
        if breakCondition
            break
        end
    end
end

### Bootstrap Loop
itmlist = DataFrame(params = Array[], potential = Array[], trajectories = Array[])
repetitions = 4

x0 = Array(range(-u0[1], u0[1], step=0.05))
y0 = map(x -> V0(x,p), x0)
traj_plot = scatter(t0, fulldata[2,:])
pot_plot = plot(x0, y0)

println("Beginning Bootstrap...")
lk = ReentrantLock()
@time Threads.@threads for rep in 1:repetitions
    sampledata = Qtils.sample(fulldata, 0.5)
    println(size(sampledata))

    otherparams = rand(Float32, 2) .+ [1.5, 0.0] # contains u0, du0
    params = vcat(otherparams, initial_params(dV))

    @time mytrain!(sampledata, loss, params, opt, cb, epochs=5)

    result = Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=t0))[1,:]
    potential = map(x -> Qtils.integrateNN(dV, x, params[3:end]), x0)

    lock(lk)
    push!(itmlist, [params[1:2], potential, result])
    unlock(lk)
end
println("Bootstrap complete!")

# CSV.write(joinpath(@__DIR__, "Bootstrap.CSV"), itmlist)

mean_params = mean(itmlist.params)
std_params = stdm(itmlist.params, mean_params)
println("parameter mean: ", mean_params, "±", std_params)

mean_traj = mean(itmlist.trajectories)
mean_pot = mean(itmlist.potential)

std_traj = stdm(itmlist.trajectories, mean_traj)
std_pot = stdm(itmlist.potential, mean_pot)

function quantiles(arr)
    catet = arr[1]
    for i in 2:size(arr,1)
        catet = hcat(catet, arr[i])
    end
    quant = Float64[0,0]
    for j in 1:size(catet, 1)
        quant = hcat(quant, quantile(catet[j,:], [0.025, 0.975]))
    end
    return quant[:,2:end]
end
quantile_traj = quantiles(itmlist.trajectories)
quantile_pot = quantiles(itmlist.potential)

lower_CI_traj = quantile_traj[1,:] - mean_traj
upper_CI_traj = quantile_traj[2,:] - mean_traj
lower_CI_pot = quantile_pot[1,:] - mean_pot
upper_CI_pot = quantile_pot[2,:] - mean_pot

traj_plot = plot!(traj_plot, t0, mean_traj, ribbon=[lower_CI_traj, upper_CI_traj]) # std_traj)
pot_plot = plot!(pot_plot, x0, mean_pot, ribbon=[lower_CI_pot, upper_CI_pot])# std_pot)
resultplot = plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 800))
display(plot(resultplot))

