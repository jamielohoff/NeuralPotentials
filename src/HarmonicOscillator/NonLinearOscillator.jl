using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, LinearAlgebra, Statistics, DataFrames, LaTeXStrings, Measures
include("../Qtils.jl")
include("../MechanicsDatasets.jl")
include("../AwesomeTheme.jl")
using .Qtils
using .MechanicsDatasets

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

### Creation of synthethetic data---------------------- #
tspan = (0.0, 5.0)
t0 = Array(range(tspan[1], tspan[2], length=128))
true_u0 = [3.1, 0.0] # contains x0 and dx0
true_p = [0.25]
stderr = 0.01

V0(x,p) = 0.25*x^4 - 2.0*x^2 # 4.5*(1 - cos(x)) # p[1]*x^2 # 
dV0(x,p) = Flux.gradient(x -> V0(x,p), x)[1]
@time data = MechanicsDatasets.potentialproblem1D(V0, true_u0, true_p, t0, addnoise=true, σ=stderr)

### End ------------------------------------------------ #

# Defining the gradient of the potential
dV = FastChain(
    FastDense(1, 16, relu),
    FastDense(16, 16, relu), 
    FastDense(16, 1)
)

ps = vcat(rand(Float64,2) .+ [3.0, 0.0], initial_params(dV))

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

function loss(params)
    pred = predict(params)
    return sum(abs2, pred[1,:] .- data[2,:]), pred # /(size(gt, 2) - size(params, 1)), pred
end

epoch = 0

cb = function(p,l,pred)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    if mod(epoch,1) == 0
        # Plotting fitting result
        traj_plot = scatter(data[1,:], data[2,:], 
                    label="Oscillator data",
                    markershape=:cross,
        )
        traj_plot = plot!(traj_plot, t0, pred[1, :],
                    title="Trajectory",
                    label="Prediction using neural potential",
                    xlabel=L"\textrm{Time } t",
                    ylabel=L"\textrm{Displacement } x(t)",
                    ylim=(-7.5,5.0),
                    legend=:bottomright,
        )

        # Plotting the potential
        x0 = Array(range(-true_u0[1], true_u0[1], step=0.01))
        # predicted_potential = map(x -> Qtils.integrateNN(dV, p, 0.0, x), x0)
        predicted_potential = [dV(x, p)[1] for x ∈ x0] 
        true_potential = [dV0(x, true_p) for x ∈ x0]

        pot_plot = plot(x0, true_potential, 
                    label="Potential used for data generation",
        )
        pot_plot = plot!(pot_plot, x0, predicted_potential,
                    title="Potential",
                    label="Prediction of the potential",
                    xlabel=L"\textrm{Displacement } x",
                    ylabel=L"\textrm{Potential } V(x)",
                    xlims=(-true_u0[1],true_u0[1]),
                    ylims=(-6.5, 8.5),
                    legend=:bottomright,
        )
        
        display(plot(traj_plot, pot_plot, layout=(2,1), size=(1200, 1200)))
    end
    global epoch+=1
    return l < 0.3
end

# Now we tell Flux how to train the neural network
opt = NADAM(0.01)
epochs = 20000
for epoch in 1:epochs
    _loss, _pred = loss(ps)
    gs = Flux.gradient(p -> loss(p)[1], ps)[1]
    if _loss < 2.0
        global opt = NADAM(1e-3)
    end
    Flux.update!(opt, ps, gs)
    breakCondition = cb(ps, _loss, _pred)
    if breakCondition
        break
    end
end

res = predict(ps)

# Plotting fitting result
traj_plot = scatter(data[1,:], data[2,:], 
            label="Oscillator data",
            markershape=:cross,
)
traj_plot = plot!(traj_plot, t0, res[1, :],
            title="Trajectory",
            label="Prediction using neural potential",
            xlabel=L"\textrm{Time } t",
            ylabel=L"\textrm{Displacement } x(t)",
            ylim=(-7.5,5.0),
            legend=:bottomright,
)

# Plotting the potential
x0 = Array(range(-true_u0[1], true_u0[1], step=0.01))
# predicted_potential = map(x -> Qtils.integrateNN(dV, p, 0.0, x), x0)
predicted_potential = [Qtils.integrateNN(dV, p, 0.0, x) for x ∈ x0] 
true_potential = [V0(x, true_p) for x ∈ x0]

pot_plot = plot(x0, true_potential, 
            label="Potential used for data generation",
)
pot_plot = plot!(pot_plot, x0, predicted_potential,
            title="Potential",
            label="Prediction of the potential",
            xlabel=L"\textrm{Displacement } x",
            ylabel=L"\textrm{Potential } V(x)",
            xlims=(-true_u0[1],true_u0[1]),
            ylims=(-6.5, 8.5),
            legend=:bottomright,
)

result_plot = plot(traj_plot, pot_plot, layout=(2,1), size=(1200,1200))
savefig(result_plot, "NonLinearOscillator.pdf")

