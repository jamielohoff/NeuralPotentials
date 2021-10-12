using PlotThemes, PlotUtils, Requires

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


PlotThemes._themes[:awesome] = PlotThemes.PlotTheme(;
    palette = awesome_palette,
    colorgradient = reverse(PlotThemes.ylorbr_gradient),
    sheet_args...
)