using Flux, DiffEqFlux, DifferentialEquations
using DataFrames, CSV, Plots, Statistics, LaTeXStrings, Measures
include("../Qtils.jl")
using .Qtils

theme(:mute)
Plots.resetfontsizes()
Plots.scalefontsizes(2)

sndatapath = joinpath(@__DIR__, "supernovae.csv")
grbdatapath = joinpath(@__DIR__, "grbs.csv")

sndata = CSV.read(sndatapath, delim=' ', DataFrame) # supernova data
grbdata = CSV.read(grbdatapath, delim=' ', DataFrame) # gamma-ray bursts

μ_plot = scatter(sndata.z, sndata.my, yerror=sndata.me,
        title="Redshift-Luminosity Data",
        xlabel=L"\textrm{Redshift } z",
        ylabel=L"\textrm{Distance modulus } \mu",
        label="Supernova data",
        legend=:bottomright,
        size=(1000, 800),
        margin=5mm,
        markersize=6,
        markershape=:cross,
        foreground_color_minor_grid = "white",
        framestyle=:box,
        color=colorant"#328" # indigo
)
μ_plot = scatter!(μ_plot, grbdata.z, grbdata.my, yerror=grbdata.me,
        label="Gamma-ray burst data", 
        markersize=6, 
        markershape=:cross,
        color=colorant"#c67" # rose
)



plot(μ_plot)
savefig(μ_plot, "redshiftluminositydata.pdf")

