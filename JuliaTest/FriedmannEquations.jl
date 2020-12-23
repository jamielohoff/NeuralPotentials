using Plots
using Flux
using DiffEqFlux
using DifferentialEquations
using DataFrames
using CSV

df = readtable("./data/supernovae.data")


