using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, LinearAlgebra, Statistics, DataFrames, LaTeXStrings, Measures, CSV
include("../Qtils.jl")
include("../MechanicsDatasets.jl")
using .Qtils
using .MechanicsDatasets

theme(:mute)
resetfontsizes()
scalefontsizes(2)

### Creation of synthethetic data---------------------- #
fulltspan = (0.0, 8.5)
tspan = (0.0, 5.0)
t0 = Array(range(fulltspan[1], fulltspan[2], length=256))
t = Array(range(tspan[1], tspan[2], length=256))
true_u0 = [3.0, 0.0] # contains x0 and dx0
true_p = [2.0]
stderr = 0.2

# Define potential and get dataset
V0(x,p) = 0.5*p[1]*x^2 
fulldata = MechanicsDatasets.potentialproblem1D(V0, true_u0, true_p, t0, addnoise=true, σ=stderr)
### End ------------------------------------------------ #

# Defining the gradient of the potential
dV = FastChain(
    FastDense(1, 8, relu), 
    FastDense(8, 8, relu),
    FastDense(8, 1)
)

# Define ODE for our system
function neuraloscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -dV(x, p)[1]
end

prob = ODEProblem(neuraloscillator!, true_u0, tspan, true_p)

### Bootstrap Loop ----------------------------------------------------- #
itmlist = DataFrame(params = Array[], potential = Array[], trajectories = Array[])
repetitions = 64

x0 = Array(range(-true_u0[1], true_u0[1], step=0.05))
true_potential = map(x -> V0(x, true_p), x0)

println("Beginning Bootstrap...")
lk = ReentrantLock()
sampledata = fulldata[:,findall(x -> x < 5.0, fulldata[1,:])]
@time Threads.@threads for rep in 1:repetitions
    println("Starting repetition ", rep, " at thread ", Threads.threadid())
    function predict(params)
        return Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=sampledata[1,:]))
    end
    
    function loss(params)
        pred = predict(params)
        return sum(abs2, (pred .- sampledata[2:3,:])./stderr) / (size(sampledata, 2) - size(params, 1)), pred
    end
    
    cb = function(p,l,pred)
        # println("Loss at thread ", Threads.threadid(), " : ", l)
        # println("Params at thread ", Threads.threadid(), " : ", p[1:2])
        return l < 2.5
    end

    otherparams = rand(Float32, 2) .+ [2.5, 0.0] # contains u0, du0
    ps = vcat(otherparams, initial_params(dV))
    opt = ADAM(0.2)

    @time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=1500)

    prediction_problem = ODEProblem(neuraloscillator!, true_u0, fulltspan, true_p)
    res = Array(solve(prediction_problem, Tsit5(), u0=result.minimizer[1:2], p=result.minimizer[3:end], saveat=t0))
    println(size(res))
    potential = map(x -> Qtils.integrateNN(dV, result.minimizer[3:end], 0.0, x), x0)

    lock(lk)
    push!(itmlist, [result.minimizer[1:2], potential, res[1,:]])
    unlock(lk)
    println("Repetition ", rep, " is done!")
end
println("Bootstrap complete!")

CSV.write("harmonicoscillatorpredictions.csv", itmlist)

mean_params, std_params, CI_params = Qtils.calculatestatistics(itmlist.params)
mean_traj, std_traj, CI_traj = Qtils.calculatestatistics(itmlist.trajectories)
mean_pot, std_pot, CI_pot = Qtils.calculatestatistics(itmlist.potential)

println("parameter mean: ", mean_params, "±", std_params)

traj_plot = scatter(fulldata[1,:], fulldata[2,:],
            title="Trajectory",
            label="Harmonic oscillator data",
            xlabel=L"\textrm{Time } t",
            ylabel=L"\textrm{Displacement } x(t)",
            ylim=(-7.5,5.0),
            legend=:bottomright,
            margin=8mm,
            markersize=6,
            markerstrokewidth=0,
            markershape=:cross,
            foreground_color_minor_grid = "white",
            framestyle=:box,
            color=colorant"#328" # indigo
)
traj_plot = plot!(traj_plot, t0, mean_traj, 
            ribbon=CI_traj,
            label="Prediction using neural potential",
            linewidth=2,
            color=colorant"#c67" # rose
)


pot_plot = plot(x0, true_potential,
            title="Potential",
            label="Potential used for data generation",
            xlabel=L"\textrm{Displacement } x",
            ylabel=L"\textrm{Potential } V(x)",
            xlims=(-true_u0[1],true_u0[1]),
            ylim=(-5.0,10.0),
            legend=:bottomright,
            margin=8mm,
            foreground_color_minor_grid = "white",
            framestyle=:box,
            color=colorant"#328" # indigo
)
pot_plot = plot!(pot_plot, x0, mean_pot, 
            ribbon=CI_pot,
            label="Prediction of the potential",
            color=colorant"#c67" # rose
)


resultplot = plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 1200))
savefig(resultplot, "harmonicoscillatorprediction.pdf")

