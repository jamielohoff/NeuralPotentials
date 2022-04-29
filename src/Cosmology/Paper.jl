using Flux, DiffEqFlux, DifferentialEquations, Zygote, ForwardDiff
using DataFrames, Plots, LinearAlgebra, Statistics, Measures, LaTeXStrings, CSV
include("../lib/Qtils.jl")
include("../lib/AwesomeTheme.jl")
using .Qtils

theme(:awesome)
resetfontsizes()
scalefontsizes(2)

data, uniquez = Qtils.loaddata(@__DIR__, "supernovae.csv", "grbs.csv")
averagedata = Qtils.preparedata(data, uniquez)

zrange = Array(range(uniquez[1], uniquez[end], step=0.01))
zrange = sort(unique(vcat(zrange, uniquez)))
indexes = findall(z -> z ∈ uniquez, zrange)

const H0 = 0.06766 # in 1 / Gyr
const c = 306.4 # in Mpc / Gyr
const G = 4.475e-53 # in Mpc^3 / (Gyr^2 * planck mass)

eVMpc = 1.564e29 # Conversion factor Mpc -> eV
eVmpl = 1.2209e28 # Conversion factor Planck mass -> eV
eVGyr = 4.791e31 # Conversion factor Gyr -> eV

ϕfactor = c/sqrt(G) * sqrt(eVmpl*eVMpc)/eVGyr # Conversion factor for quintessence field
Vfactor = H0^2/G * eVmpl/(eVMpc*eVGyr^2) # Conversion factor for potential

# We cannot vary the initial conditions much, otherwise we get inconsistent results!!!
Ω_m_0 = 0.3 # From Planck + WMAP + BAO, so statistically independent from SNIa
p = [0.3, 0.1].*rand(Float64, 2) .+ [0.4, -0.05] # Random initial conditions for quintessence field

@info "Initial conditions for quintessence field: " p

u0 = vcat(p[1:2], [1.0, 0.0])
zspan = (0.0, 7.0)

# Function to calculate the distance modulus
# We have a +25 instead of -5 because we measure distances in Mpc
mu(z, χ) = 5.0 .* log10.((c/H0) * abs.((1.0 .+ z) .* χ)) .+ 25.0 
# Function to calculate the density paramter of quintessence
Ω_ϕ(dQ, E, U, z) = 8π/3 .* (0.5 .* ((1 .+ z).*dQ).^2 .+ U./E.^2)
# Gauss function
gauss(x, σ, μ) = 1.0/sqrt(2π*σ^2) .* exp(-(x .- μ).^2 ./ σ^2)

@info "Initializing neural network..."

U = FastChain(
    FastDense(1, 4, tanh),
    FastDense(4, 8, tanh),
    FastDense(8, 4, tanh),
    FastDense(4, 1, x -> gauss(x, 3.0, 0.0)) # enforce positive potentials only # x -> gauss(x, 3.0, 0.0)
)
dU(x, p) = ForwardDiff.gradient(x -> U(x, p)[1], x)[1]

ps = vcat(p, initial_params(U))
 
# 1st order ODE for Friedmann equation in terms of z
function friedmann!(du,u,p,z)
    Q = u[1]
    dQ = u[2]
    E = u[3]
    χ = u[4]
    
    Ω_m = 1.0 - Ω_ϕ(dQ, E, U([Q], p)[1], z)
    dE = 1.5*(E/(1+z)) * (Ω_m + 8π/3*((1+z)*dQ)^2)

    du[1] = dQ
    du[2] = (2/(1+z) - dE/E) * dQ - dU([Q], p)/(E*(1+z))^2
    du[3] = dE
    du[4] = 1.0/E
end

problem = ODEProblem(friedmann!, u0, zspan, p)
opt = ADAM(1e-2)

function predict(params)
    return Array(solve(problem, Tsit5(), u0=vcat(params[1:2],[1.0, 0.0]), p=params[3:end], saveat=zrange))
end

function loss(params)
    pred = predict(params)
    µ = mu(uniquez, pred[end,indexes])
    potential = U(reshape(pred[1,:],1,:), params[3:end])
    Ωϕ = Ω_ϕ(pred[2,indexes], pred[3,indexes], potential[indexes], uniquez)
    # Added two additional loss terms for regularization:
    # 1.)
    # 2.)
    return Qtils.χ2(μ, averagedata) + 250.0*abs(Ωϕ[1] - 0.7) + 125.0*abs(Ωϕ[end]), pred
end

epoch = 0

cb = function(p, l, pred)
    println("Epoch: ", epoch)
    println("Loss: ", l)
    println("Parameters: ", p[1:2])
    potential = [U(q, p[3:end])[1] for q ∈ pred[1,:]]
    plt1 = plot(pred[1,:], potential)
    Ωϕ = Ω_ϕ(pred[2,indexes], pred[3,indexes], potential[indexes], uniquez)
    plt2 = plot(uniquez, 1 .- Ωϕ, label="Ω_m", ylims=(0.0,1.0))
    plt2 = plot(plt2, uniquez, Ωϕ, label="Ω_ϕ")
    display(plot(plt1, plt2, size=(800, 400)))
    println("Density parameter Ω_ϕ = ", Ωϕ[1])
    println(Ω_m_0 + Ωϕ[1])
    global epoch += 1
    return false
end

@info "Beginning Training..."
@time result =  DiffEqFlux.sciml_train(loss, ps, opt, cb=cb, maxiters=10000)

res = solve(problem, Tsit5(), u0=vcat(result.minimizer[1:2],[1.0, 0.0]), p=result.minimizer[3:end], saveat=zrange)

Q = res[1,:]
dQ = res[2,:]
E = res[3,:]
H = H0*E*1e3 # turn into kilometers per second per megaparsec
potential = [U(q,result.minimizer[3:end])[1] for q ∈ Q]

H_plot = plot(zrange, H./(1.0 .+ zrange).^3, 
                title="Expansion rate", 
                xlabel=L"\textrm{redshift} \; z", 
                ylabel=L"Ha^{(3 \backslash 2)} \; [\textrm{km}\textrm{s}^{-1}\textrm{Mpc}^{-1}]",
                legend=false)
ϕ_plot = plot(ϕfactor.*Q./eVmpl, ϕfactor.*dQ./eVmpl, 
                title="Phase-space trajectory", 
                xlabel=L"\phi \; [10^{-3}\textrm{m}_P]", 
                ylabel=L"\dfrac{\mathrm{d}\phi}{\mathrm{d}z} \; [10^{-3}\textrm{m}_P]",
                legend=false)
U_plot = plot(ϕfactor.*Q./eVmpl, Vfactor.*potential./1e-16, 
                title="Potential", 
                xlabel=L"\phi \; [10^{-3}\textrm{m}_P]", 
                ylabel=L"\dfrac{U(\phi)}{10^{-16}} \; [\textrm{eV}^4]", 
                legend=false)
Ω_plot = plot(zrange, 1 .- Ω_ϕ(dQ, E, potential, zrange), 
                title="Density evolution", 
                xlabel=L"\textrm{redshift} \; z", 
                ylabel=L"\textrm{density} \; \textrm{parameter} \; \Omega", label=L"\Omega_m")
Ω_plot = plot!(Ω_plot, zrange, Ω_ϕ(dQ, E, potential, zrange), legend=:right, label=L"\Omega_\phi")

println("Cosmological parameters: ")
println("Dark energy density parameter Ω_ϕ = ", Ω_ϕ(dQ, E, potential, zrange)[1])
println("Initial conditions for quintessence field = ", ϕfactor.*result.minimizer[1:2]./eVmpl)

result_plot = plot(H_plot, ϕ_plot, U_plot, Ω_plot, layout=(2, 2), size=(1200, 1200))
# savefig(result_plot, "NeuralGradientQuintessence.pdf")

