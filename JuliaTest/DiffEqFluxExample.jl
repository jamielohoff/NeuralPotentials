using DifferentialEquations
using Plots
using Flux
using DiffEqFlux

function lotka_volterra(du,u,p,t)
  x, y = u
  du[1] = dx = p[1]*x - p[2]*x*y
  du[2] = dy = -p[3]*y + p[4]*x*y
end
u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p)

sol = solve(prob, Tsit5(), saveat=0.1)
A = sol[1, :]

#plot(sol)

#t = 0:0.1:10.0
#scatter!(t, A)

p = [2.2, 1.0, 2.0, 0.4]
params = Flux.params()

function predict_rd()
    solve(prob, Tsit5(), p=p, saveat=0.1)[1, :]
end

loss_rd() = sum(abs2, x-1 for x in predict_rd())

# No we tell Flux how to train the neural network
data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
    display(loss_rd())
    display(plot(solve(remake(prob, p=p), Tsit5(), saveat=0.1), ylim=(0,6)))
end


cb()

Flux.train!(loss_rd, params, data, opt, cb=cb)

gui()

