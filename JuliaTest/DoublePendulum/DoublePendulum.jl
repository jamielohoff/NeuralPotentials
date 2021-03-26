using DifferentialEquations
using Plots
using Flux
using DiffEqFlux

m1 = 0.5
m2 = 1.0
l1 = 2.0 # 2.0
l2 = 1.0
const g = 9.81

tspan = (0.0, 20.0)
u0 = [50 * pi/180, -120 * pi/180, 0, 0] # 50, -120

params = [m1, m2, l1, l2]

############################
# Define Functions for ODE #
############################
f1(p1, p2, q1, q2, p) = (p[2]*p[4]^2*q1 - p[2]*p[3]*p[4]*cos(p1-p2)*q2) / (p[2]*(p[1]+p[2])*(p[3]*p[4])^2 - (p[2]*p[3]*p[4]*cos(p1-p2))^2)

f2(p1, p2, q1, q2, p) = ((p[1]+p[2])*p[3]^2*q2 - p[2]*p[3]*p[4]*cos(p1-p2)*q1) / (p[2]*(p[1]+p[2])*(p[3]*p[4])^2 - (p[2]*p[3]*p[4]*cos(p1-p2))^2)

f3(p1, p2, q1, q2, p) = -p[2]*p[3]*p[4]*f1(p1, p2, q1, q2, p)*f2(p1, p2, q1, q2, p)*sin(p1-p2) - g*(p[1] + p[2])*p[3]*sin(p1)

f4(p1, p2, q1, q2, p) = p[2]*p[3]*p[4]*f1(p1, p2, q1, q2, p)*f2(p1, p2, q1, q2, p)*sin(p1-p2) - g*p[2]*p[4]*sin(p2)

# inplace version of double pendulum
function doublependulum!(du, u, p, t)
    p1 = u[1]
    p2 = u[2]
    q1 = u[3] 
    q2 = u[4]

    du[1] = f1(p1, p2, q1, q2, p) # dp1
    du[2] = f2(p1, p2, q1, q2, p) # dp2
    du[3] = f3(p1, p2, q1, q2, p) # dq1
    du[4] = f4(p1, p2, q1, q2, p) # dq2
end

problem = ODEProblem(doublependulum!, u0, tspan, params)

@time sol = solve(problem, Tsit5(), saveat=0.1)

t = sol.t
angles = sol[1:2,:]
plot(l1*sin.(angles[1,:]), -l1*cos.(angles[1,:]))
plot!(l1*sin.(angles[1,:]) + l2*sin.(angles[2, :]), -l1*cos.(angles[1, :]) - l2*cos.(angles[2, :]))

# noise = 0.01 * (2 * rand(Float64, size(angles)[2]) .- 1)
# angles[1,:] = angles[1,:] .+ noise
# angles[2,:] = angles[2,:] .+ noise

p = rand(Float64, 4)
params = Flux.params(p)

function predict()
    solve(problem, Tsit5(), p=p, saveat=0.1)[1:2, :]
end

function loss() 
    sum(sum(abs2, phi-phi_gt) for (phi, phi_gt) in zip(predict(), angles))
end

# Now we tell Flux how to train the neural network
data = Iterators.repeated((), 3000)
opt = ADAM(0.01)
cb = function()
    display(Flux.params(p))
    s = solve(remake(problem, p=p), Tsit5(), saveat=0.1)
    display(plot(s.t, s[1, :], ylim=(-2,2)))
    display(scatter!(t, angles[1,:], ylim=(-2,2)))
end

function custom_train!(params, data, opt; cb)
    for iter in data
        println("Training loop 1")
        grads = Flux.gradient(params) do
            predict = solve(remake(problem, tspan=(0.0,1.0)), Tsit5(), p=p, saveat=0.1)[1:2, :]
            loss = sum(sum(abs2, phi-phi_gt) for (phi, phi_gt) in zip(predict, angles[:, 1:11]))
            println(loss)
            return loss
        end
        Flux.update!(opt, params, grads)
        cb()
    end

    opt = ADAM(1e-3)
    for iter in data
        println("Training loop 2")
        grads = Flux.gradient(params) do
            predict = solve(remake(problem, tspan=(0.0,3.0)), Tsit5(), p=p, saveat=0.1)[1:2, :]
            loss = sum(sum(abs2, phi-phi_gt) for (phi, phi_gt) in zip(predict, angles[:, 1:51]))
            println(loss)
            return loss
        end
        Flux.update!(opt, params, grads)
        cb()
    end

    opt = ADAM(1e-3)
    for iter in data
        println("Training loop 3")
        grads = Flux.gradient(params) do
            predict = solve(remake(problem, tspan=(0.0,5.0)), Tsit5(), p=p, saveat=0.1)[1:2, :]
            loss = sum(sum(abs2, phi-phi_gt) for (phi, phi_gt) in zip(predict, angles[:, 1:101]))
            println(loss)
            return loss
        end
        Flux.update!(opt, params, grads)
        cb()
    end

    opt = ADAM(1e-4)
    for iter in data
        println("Training loop 4")
        grads = Flux.gradient(params) do
            predict = solve(remake(problem, tspan=(0.0,7.0)), Tsit5(), p=p, saveat=0.1)[1:2, :]
            loss = sum(sum(abs2, phi-phi_gt) for (phi, phi_gt) in zip(predict, angles[:, 1:101]))
            println(loss)
            return loss
        end
        Flux.update!(opt, params, grads)
        cb()
    end

    opt = ADAM(1e-4)
    for iter in data
        println("Training loop 5")
        grads = Flux.gradient(params) do
            predict = solve(remake(problem, tspan=(0.0,10.0)), Tsit5(), p=p, saveat=0.1)[1:2, :]
            loss = sum(sum(abs2, phi-phi_gt) for (phi, phi_gt) in zip(predict, angles[:, 1:101]))
            println(loss)
            return loss
        end
        Flux.update!(opt, params, grads)
        cb()
    end
end

@time custom_train!(params, data, opt, cb=cb)

println(Flux.params(p))

