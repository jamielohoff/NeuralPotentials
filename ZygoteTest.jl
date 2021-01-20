using Zygote: @adjoint, gradient
using Dierckx
using DifferentialEquations
using Plots

H0 = 0.069 # 1 / Gyr
c = 306.4 # in Mpc / Gyr

p = [0.3, 1e-4, 0.7, 0] # p[1]=omega_m, p[2]=omega_r, p[3]=omega_l, p[4]=omega_k

# 2nd order ODE for Friedmann equation 
f(u, p, z) = 0.5*H0^2*(3*p[1]*(1.0+z)^4 + 3*p[2]*(1.0+z)^3 - 2*p[3]) / ((1+z)*u) # (p[1]*(1.0+z)^4 + p[2]*(1.0+z)^3 + p[3])

u0 = H0
tspan = (0.0, 7.5)
problem = ODEProblem(f, u0, tspan, p)
solution = solve(problem, Tsit5(), saveat=0.01)

H_predict = Spline1D(sol.t, 1.0 ./ sol[1,:])

function u_integrate(H_itp, z)
    integrate(H_itp, 0, z)
end

@adjoint u_integrate(H_itp, z) = u_integrate(H_itp, z), c -> begin
    b = H_itp(z)
    println(b)
    (nothing, c*b)
end

function luminosity_distance(H_itp, z)
    c/H0 * (1.0 + z) * u_integrate(H_itp,z)
end

gradient(5) do x
    x    
end

