using DiffEqFlux, Flux, OrdinaryDiffEq, ReverseDiff
using Statistics, Plots

m = 1.0
w = 1.0
true_p = [m, w]
true_u0 = [1.0, 0.0]

tspan = (0.0f0, 10.0f0)
t = range(tspan[1], tspan[2], length=1024)

dQdt(Q, P, p, t) = P/p[1]
dPdt(Q, P, p, t) = -p[2]^2*Q

function oscillator!(du, u, p, t)
    Q = u[1]
    P = u[2]

    du[1] = dQdt(Q, P, p, t)
    du[2] = dPdt(Q, P, p, t)
end

problem = ODEProblem(oscillator!, true_u0, tspan, true_p)

@time true_sol = solve(problem, Tsit5(), saveat=t)

plot(true_sol.t, true_sol[1,:])
plot!(true_sol.t, true_sol[2,:])

Qdot = reshape(dQdt(true_sol[1,:], true_sol[2,:], true_p, t), 1, :)
Pdot = reshape(dPdt(true_sol[1,:], true_sol[2,:], true_p, t), 1, :)

Q_t = reshape(true_sol[1,:], 1, :)
P_t = reshape(true_sol[2,:], 1, :)

data = cat(Q_t, P_t, dims=1)
target = cat(Qdot, Pdot, dims=1)
dataloader = Flux.Data.DataLoader((data, target); batchsize=256, shuffle=true)

hnn = HamiltonianNN(
    Chain(Dense(2, 64, relu), Dense(64, 1))
)

params = hnn.p
opt = ADAM(0.01)

loss(x, y, p) = sum(abs2, hnn(x, p) .- y)

callback() = println("Loss Neural Hamiltonian DE = $(loss(data, target, params))")

epochs = 1000
for epoch in 1:epochs
    for (x, y) in dataloader
        gs = ReverseDiff.gradient(p -> loss(x, y, p), params)
        Flux.update!(opt, params, gs)
    end
    if epoch % 100 == 1
        callback()
    end
end
callback()

model = NeuralHamiltonianDE(
    hnn, (0.0f0, 10.0f0),
    Tsit5(), save_everystep=false,
    save_start=true, saveat=t
)

println(data[:, 1])
pred = Array(model(data[:, 1]))
plot(data[1, :], data[2, :], lw=4, label="Original")
plot!(pred[1, :], pred[2, :], lw=4, label="Predicted")
xlabel!("Position (q)")
ylabel!("Momentum (p)")