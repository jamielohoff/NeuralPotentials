using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, LinearAlgebra, Statistics, LaTeXStrings, Measures
include("../Qtils.jl")
include("../MechanicsDatasets.jl")
using .Qtils
using .MechanicsDatasets

theme(:mute)
resetfontsizes()
scalefontsizes(2)

### Creation of synthethetic data---------------------- #
tspan = (0.0, 5.0)
t0 = Array(range(tspan[1], tspan[2], length=256))
true_u0 = [3.0, 0.0] # contains x0 and dx0
true_p = [2.0]
stderr = 0.1

# Define potential and get dataset
V0(x,p) = 0.5*p[1]*x^2 
data = MechanicsDatasets.potentialproblem1D(V0, true_u0, true_p, t0, addnoise=true, σ=stderr)
### End ------------------------------------------------ #

# Defining the gradient of the potential
dV = FastChain(
    FastDense(1, 8, relu), 
    FastDense(8, 8, relu),
    FastDense(8, 1)
)

otherparams = rand(Float32, 2) .+ [2.5, 0.0]
ps = vcat(otherparams, initial_params(dV))

function neuraloscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -dV(x,p)[1] 
end

prob = ODEProblem(neuraloscillator!, ps[1:2], tspan, ps[3:end])

function predict(params)
    return Array(solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=t0))
end

function reducedχ2(pred, params)
    return sum(abs2, (pred[1,:] .- data[2,:])./stderr) / (size(data, 2) - size(params, 1))
end

function loss(params)
    pred = predict(params)
    return reducedχ2(pred, params), pred
end

i = 0
cb = function(p,l,pred)
    println("Loss: ", l, " at epoch ", i)
    println("Parameters: ", p[1:2])

    # # Plotting fitting result
    # traj_plot = scatter(t0, data[2,:], 
    #             label="Harmonic oscillator data",
    #             markersize=6,
    #             markerstrokewidth=0,
    #             markershape=:cross,
    #             color=colorant"#328" # indigo
    # )
    # traj_plot = plot!(traj_plot, t0, pred[1, :],
    #             title="Trajectory",
    #             label="Prediction using neural potential",
    #             xlabel=L"\textrm{Time } t",
    #             ylabel=L"\textrm{Displacement } x(t)",
    #             ylim=(-7.5,5.0),
    #             legend=:bottomright,
    #             margin=8mm,
    #             foreground_color_minor_grid = "white",
    #             framestyle=:box,
    #             linewidth=3,
    #             color=colorant"#c67" # rose
    # )

    # # Plotting the potential
    # x0 = Array(range(-true_u0[1], true_u0[1], step=0.01))
    # predicted_potential = map(x -> Qtils.integrateNN(dV, p[3:end], 0.0, x), x0)
    # true_potential = map(x -> V0(x, true_p), x0)

    # pot_plot = plot(x0, true_potential, 
    #             label="Potential used for data generation",
    #             color=colorant"#328" # indigo
    # )
    # pot_plot = plot!(pot_plot, x0, predicted_potential,
    #             title="Potential",
    #             label="Prediction of the potential",
    #             xlabel=L"\textrm{Displacement } x",
    #             ylabel=L"\textrm{Potential } V(x)",
    #             xlims=(-true_u0[1],true_u0[1]),
    #             ylims=(-5.0, 10.0),
    #             legend=:bottomright,
    #             margin=8mm,
    #             foreground_color_minor_grid = "white",
    #             framestyle=:box,
    #             color=colorant"#c67" # rose
    # )
    
    # display(plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 1200)))

    global i += 1
    return l < 1.05
end

opt = ADAM(0.2)

@time result = DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=1500)

res = Array(solve(prob, Tsit5(), u0=result.minimizer[1:2], p=result.minimizer[3:end], saveat=t0))

# Plotting fitting result
traj_plot = scatter(t0, data[2,:], 
            label="Harmonic oscillator data",
            markersize=6,
            markerstrokewidth=0,
            markershape=:cross,
            color=colorant"#328" # indigo
)
traj_plot = plot!(traj_plot, t0, res[1, :],
            title="Trajectory",
            label="Prediction using neural potential",
            xlabel=L"\textrm{Time } t",
            ylabel=L"\textrm{Displacement } x(t)",
            ylim=(-7.5,5.0),
            legend=:bottomright,
            margin=8mm,
            foreground_color_minor_grid = "white",
            framestyle=:box,
            linewidth=3,
            color=colorant"#c67" # rose
)


# Plotting the potential
x0 = Array(range(-true_u0[1], true_u0[1], step=0.01))
predicted_potential = map(x -> Qtils.integrateNN(dV, result.minimizer[3:end], 0.0, x), x0)
true_potential = map(x -> V0(x, true_p), x0)

pot_plot = plot(x0, true_potential, 
            label="Potential used for data generation",
            color=colorant"#328" # indigo
)
pot_plot = plot!(pot_plot, x0, predicted_potential,
            title="Potential",
            label="Prediction of the potential",
            xlabel=L"\textrm{Displacement } x",
            ylabel=L"\textrm{Potential } V(x)",
            xlims=(-true_u0[1],true_u0[1]),
            ylims=(-5.0, 10.0),
            legend=:bottomright,
            margin=8mm,
            foreground_color_minor_grid = "white",
            framestyle=:box,
            color=colorant"#c67" # rose
)

resultplot = plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 1200))
savefig(resultplot, "harmonicoscillator_positiononly.pdf")

