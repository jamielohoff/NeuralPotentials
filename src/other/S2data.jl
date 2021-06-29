using Flux, DiffEqFlux, DifferentialEquations, Zygote
using DataFrames, Plots, LinearAlgebra, Statistics, QuadGK, CSV

path = joinpath(@__DIR__, "S2.csv")
s2data = CSV.read(path, delim=' ', DataFrame)
scatter(s2data.dRA, s2data.dDEC)

