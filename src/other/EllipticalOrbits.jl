using DifferentialEquations, Flux, DiffEqFlux, Zygote
using Plots, Statistics, LinearAlgebra, LaTeXStrings, DataFrames, Distributions, Measures
include("../lib/MechanicsDatasets.jl")
include("../lib/AwesomeTheme.jl")
using .MechanicsDatasets

# Load predefined plot theme and adjust font sizes
theme(:awesome)
resetfontsizes()
scalefontsizes(2)

# Natural constants
const c = 306.4 # mpc/yr
const G = 4.49 # gravitational constant in new units : (milliparsec)^3 * yr^-2 * (10^6*M_solar)^-1

ϕ0span = (0.0, 4π-0.01)
ϕ0 = Array(range(ϕ0span[1], ϕ0span[2], length=1000))
ϕ1span = (0.0, 3π/4)
ϕ1 = Array(range(ϕ1span[1], ϕ1span[2], length=1000))
r0 = 0.1 # length of the periapsis in mpc 
M = 4.35 # mass of the central SMBH
true_v0 = sqrt(G*M/r0) # velocity for circular orbit in mpc/yr

dV(U,p) = G*p[1]*[p[2]^2 + 3.0*U^2/c^2]
dV0(U, p) = [G*p[1]*p[2]^2]

# circular orbit 
true_u0 = [1.0/r0, 0.0, 0.0] 
true_p = [M, 1.0/(1.0*true_v0*r0)]
circle = MechanicsDatasets.keplerproblem(dV0, true_u0, true_p, ϕ0)

# elliptic orbit 
true_u0 = [1.0/r0, 0.0, 0.0] 
true_p = [M, 1.0/(1.3*true_v0*r0)]
ellipse = MechanicsDatasets.keplerproblem(dV0, true_u0, true_p, ϕ0)

# parabolic orbit 
true_u0 = [1.0/r0, 0.0, 0.0] 
true_p = [M, 1.0/(sqrt(2*G*M/r0)*r0)]
parabola = MechanicsDatasets.keplerproblem(dV0, true_u0, true_p, ϕ1)

# hyperbolic orbit 
true_u0 = [1.0/r0, 0.0, 0.0] 
true_p = [M, 1.0/(1.45*true_v0*r0)]
hyperbola = MechanicsDatasets.keplerproblem(dV0, true_u0, true_p, ϕ1)

# orbit with periapsis shift
true_u0 = [1.0/r0, 0.0, 0.0] 
true_p = [M, 1.0/(1.3*true_v0*r0)]
rela= MechanicsDatasets.keplerproblem(dV, true_u0, true_p, ϕ0)

orbit_plot = plot(circle.r .* cos.(circle.ϕ), circle.r .* sin.(circle.ϕ), label="Closed circular orbit")
orbit_plot = plot!(orbit_plot, ellipse.r.*cos.(ellipse.ϕ), ellipse.r.*sin.(ellipse.ϕ),
                    label="Closed elliptical orbit",
                    xlabel=L"x \textrm{ coordinate [mpc]}",
                    ylabel=L"y \textrm{ coordinate [mpc]}",
                    title="Newtonian and Relativistic Orbits",
                    size=(1200, 1200)
)
orbit_plot = plot!(orbit_plot, parabola.r .* cos.(parabola.ϕ), parabola.r .* sin.(parabola.ϕ), label="Parabolic orbit")
orbit_plot = plot!(orbit_plot, hyperbola.r .* cos.(hyperbola.ϕ), hyperbola.r .* sin.(hyperbola.ϕ), label="Hyperbolic orbit")
orbit_plot = plot!(orbit_plot, rela.r .* cos.(rela.ϕ), rela.r .* sin.(rela.ϕ), label="Orbit with periapsis shift")

savefig(orbit_plot, "DifferentOrbits.pdf")

