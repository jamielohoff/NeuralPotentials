using DifferentialEquations
using Plots
using Flux
using DiffEqFlux

m1 = 0.5
m2 = 1.0
l1 = 2.0
l2 = 1.0
const g = 9.81

tspan = (0.0, 3.0)
u0 = [50 * pi/180, -12 * pi/180, 0, 0] # 50, -120

params = [m1, m2, l1, l2]

############################
# Define Functions for ODE #
############################

square(x) = x * x

f1(p1, p2, q1, q2, p) = (p[2]*square(p[4])*q1 - p[2]*p[3]*p[4]*cos(p1-p2)*q2) / (p[2]*(p[1]+p[2])*square(p[3]*p[4]) - square(p[2]*p[3]*p[4]*cos(p1-p2)))

f2(p1, p2, q1, q2, p) = ((p[1]+p[2])*square(p[3])*q2 - p[2]*p[3]*p[4]*cos(p1-p2)*q1) / (p[2]*(p[1]+p[2])*square(p[3]*p[4]) - square(p[2]*p[3]*p[4]*cos(p1-p2)))

f3(p1, p2, q1, q2, p) = -p[2]*p[3]*p[4]*f1(p1, p2, q1, q2, p)*f2(p1, p2, q1, q2, p)*sin(p1-p2) - g*(p[1] + p[2])*p[3]*sin(p1)

f4(p1, p2, q1, q2, p) = p[2]*p[3]*p[4]*f1(p1, p2, q1, q2, p)*f2(p1, p2, q1, q2, p)sin(p1-p2) - g*p[2]*p[4]*sin(p2)

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

@time sol = solve(problem, Tsit5(), saveat=0.1, reltol=1e-8, abstol=1e-8)

t = sol.t
angles = sol[1:2,:]
plot(l1*sin.(angles[1,:]), -l1*cos.(angles[1,:]))
plot!(l1*sin.(angles[1,:]) + l2*sin.(angles[2, :]), -l1*cos.(angles[1, :]) - l2*cos.(angles[2, :]))

#data_batch = sol[1,:]
#noise = 0.1 * (2 * rand(Float64, size(data_batch)) .- 1)
#data_batch = data_batch .+ noise

p = rand(Float64, 4)
params = Flux.params(p)

function predict()
    solve(problem, Tsit5(), p=p, saveat=0.1, reltol=1e-8, abstol=1e-8)[1:2, :]
end

function loss() 
    # println("lengths: ", length(predict()[1,:]), " ", length(angles[1,:]))
    sum(abs2, (phi-phi_gt)*exp(-0.7*time) for (phi, phi_gt, time) in zip(predict()[1,:], angles[1,:], t)) + sum(abs2, (phi-phi_gt)*exp(-0.7*time) for (phi, phi_gt, time) in zip(predict()[2,:], angles[2,:], t))
end

# Now we tell Flux how to train the neural network
data = Iterators.repeated((), 5000)
opt = ADAM(0.01)
cb = function ()
    display(loss())
    display(Flux.params(p))
    s = solve(remake(problem, p=p), Tsit5(), saveat=0.1, reltol=1e-8, abstol=1e-8)
    display(plot(s.t, s[1, :], ylim=(-2,2)))
    display(scatter!(t, angles[1,:], ylim=(-2,2)))
end

cb()
@time Flux.train!(loss, params, data, opt, cb=cb)

println(Flux.params(p))

