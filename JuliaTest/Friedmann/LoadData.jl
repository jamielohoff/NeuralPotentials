using Plots
using DataFrames
using CSV

sndata = CSV.File("supernovae.data", delim=' ') # supernova data
grbdata = CSV.File("grbs.data", delim=' ') # gamma-ray bursts

scatter(
    sndata.zsn, sndata.my, 
    title="Supernova Data",
    xlabel="redshift z",
    ylabel="apperant magnitude m",
    yerror=sndata.me,
    label="supernovae data"
)

scatter!(
    grbdata.zgrb, grbdata.mygrb, 
    yerror=grbdata.megrb,
    label="gamma-ray bursts"
)

