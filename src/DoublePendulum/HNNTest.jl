using DiffEqFlux, Flux, OrdinaryDiffEq, ReverseDiff
using Statistics, Plots

t = range(0.0f0, 1.0f0, length=1024)
π_32 = Float32(π)
q_t = reshape(sin.(2π_32 * t), 1, :)
p_t = reshape(cos.(2π_32 * t), 1, :)
dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t

data = cat(q_t, p_t, dims=1)
target = cat(dqdt, dpdt, dims=1)
dataloader = Flux.Data.DataLoader((data, target); batchsize=256, shuffle=true)

hnn = HamiltonianNN(
    Chain(Dense(2, 64, relu), Dense(64, 1))
)

params = hnn.p
opt = ADAM(0.01)

loss(x, y, p) = mean((hnn(x, p) .- y) .^ 2)

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
    hnn, (0.0f0, 1.0f0),
    Tsit5(), save_everystep = false,
    save_start = true, saveat = t
)

pred = Array(model(data[:, 1]))
plot(data[1, :], data[2, :], lw=4, label="Original")
plot!(pred[1, :], pred[2, :], lw=4, label="Predicted")
xlabel!("Position (q)")
ylabel!("Momentum (p)")