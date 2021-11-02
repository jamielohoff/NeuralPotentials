using PlotThemes, PlotUtils, Requires

"""
To use this plot theme, put it into the right directory and 
make it available by using 

    include("/path/to/this/file/AwesomeTheme.jl")

and then using it by adding the command 

    theme(:awesome)

after importing it.
"""

# Define shape of the plot theme
sheet_args = (
    fglegend = plot_color(colorant"#225", 0.1),
    bglegend = plot_color(:white, 0.9),
    gridcolor = colorant"#225",
    minorgridcolor = colorant"#225",
    framestyle = :box,
    minorgrid = false,
    linewidth = 2,
    markersize = 6,
    margin=8mm,
)

# Color palette of the plot theme
awesome_palette = [ # bright
    colorant"#328", # indigo
    colorant"#c67", # rose
    colorant"#dc7", # sand
    colorant"#173", # green
    colorant"#8ce", # cyan
    colorant"#825", # wine
    colorant"#4a9", # teal
    colorant"#993", # olive
    colorant"#a49", # purple
    colorant"#ddd", # grey
]

# Make the plot theme available to Plots.jl
PlotThemes._themes[:awesome] = PlotThemes.PlotTheme(;
    palette = awesome_palette,
    colorgradient = reverse(PlotThemes.ylorbr_gradient),
    sheet_args...
)