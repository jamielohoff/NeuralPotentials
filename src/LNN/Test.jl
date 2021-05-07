using DiffEqFlux, Flux, OrdinaryDiffEq, ReverseDiff, ForwardDiff
using Statistics, Plots
include("lnn.jl")
using .LagrangianNeuralNetwork

beta = 0.1f0
w = 1.0f0
true_p = [w, beta]
true_u0 = [1.0f0, 0.0f0]

tspan = (0.0f0, 10.0f0)
t = range(tspan[1], tspan[2], length=1024)

f(dx, x, p, t) = -p[1]^2*x - p[2]*dx

function oscillator!(du, u, p, t)
    x = u[1]
    dx = u[2]

    du[1] = dx
    du[2] = f(dx, x, p, t)
end

problem = ODEProblem(oscillator!, true_u0, tspan, true_p)

@time true_sol = solve(problem, Tsit5(), saveat=t)

X = true_sol[1,:]
dX = true_sol[2,:]
ddX = f(dX, X, true_p, t)

data = []
for i in 1:length(X)
    d = [X[i], dX[i]]
    push!(data, d)
end
target = reshape(ddX, 1, :)
dataloader = Flux.Data.DataLoader((data, target); batchsize=256, shuffle=true)

lnn = LagrangianNeuralNetwork.LagrangianNN(
    FastChain(
        FastDense(2, 64, tanh), 
        FastDense(64, 1))
)

params = lnn.p
opt = ADAM(0.1, (0.8, 0.9))

function loss(x, y, p)
    batch = map(x -> lnn(x, p)[1], x)
    return mean((batch .- y).^2)
end

callback = function () 
    println("Loss Neural Lagrangian DE: $(loss(data, target, params))")
end

@time for epoch in 1:500
    println("Epoch: ", epoch)
    for (x, y) in dataloader
        gs = ForwardDiff.gradient(p -> loss(x, y, p), params)
        Flux.update!(opt, params, gs)
    end
    if epoch % 10 == 1
        callback()
    end
end
callback()

model = LagrangianNeuralNetwork.NeuralLagrangianDE(
    lnn, tspan,
    Tsit5(), save_everystep=false,
    save_start=true, saveat=t
)

pred = Array(model(true_u0))
plot(t, X, lw=2, label="Original")
plot!(t, pred[1, :], lw=2, label="Predicted", ylims=(-1.5, 1.5))
xlabel!("time")
ylabel!("acceleration")

