using DifferentialEquations, Flux, DiffEqFlux, Plots

### Creation of synthethetic data---------------------- #
omega = 1.0
beta = 0.1
tspan = (0.0, 20.0)
t = range(tspan[1], tspan[2], length=100)
u0 = [0.0]
du0 = [1.0]
p = [omega, beta]

true_ff(du,u,p,t) = -p[1]*u - p[2]*du
problem = SecondOrderODEProblem{false}(true_ff, du0, u0, tspan, p)
true_sol = solve(problem, Tsit5(), saveat=t)

data_batch = true_sol[2,:]
noise = 0.1 * (2 * rand(Float64, size(data_batch)) .- 1)
data_batch = Float32.(data_batch .+ noise)
data_t = Float32.(true_sol.t)

### End ------------------------------------------------ #

# Defining the gradient of the potential
dV = FastChain(
    FastDense(1, 10, sigmoid), 
    FastDense(10, 1)
)
otherparams = rand(Float64, 3) # contains u0, du0 and beta
p = vcat(otherparams, initial_params(dV))

function neuraloscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = -p[1] * dx - dV([x],p[2:end])[1]
end

prob = ODEProblem{true}(neuraloscillator!, p[1:2], tspan, p[3:end])

function predict(params)
    solve(prob, Tsit5(), u0=params[1:2], p=params[3:end], saveat=t)
end

function loss_n_ode(params)
    pred = predict(params)
    sum(abs2, data_batch .- pred[1, :]), pred
end

opt = ADAM(0.01, (0.7, 0.9))

cb = function(p,l,pred)
    println(l)
    display(plot(data_t, pred[1, :], ylim=(-1,1)))
    display(scatter!(data_t, data_batch, ylim=(-1,1)))
    l < 0.01
end

result = DiffEqFlux.sciml_train(loss_n_ode, p, opt, cb=cb, maxiters=1000)
println("Best result: ", result.minimizer)

