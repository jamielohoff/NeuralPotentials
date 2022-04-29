import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_data(
    zrange, 
    mean_V, CI_V,
    mean_omega, CI_omega,
    mean_H, CI_H,
    mean_phi, CI_phi,
    mean_dphi, CI_dphi,
    mean_EoS, CI_EoS,
    phi0, dphi0,
    bins=30
):
    mpl.style.use("classic")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Helvetica"
    figure, axis = plt.subplots(2, 4)

    # Plot all the stuff into a 4x2 figure
    axis[0,0].plot(zrange, mean_H)
    axis[0,0].fill_between(zrange, CI_H[1], CI_H[2])
    axis[0,0].set_title("expansion rate")
                #ylabel=L"Ha^{(3 \backslash 2)} \; [\textrm{km}\textrm{s}^{-1}\textrm{Mpc}^{-1}]",

    axis[0,1].errorbar(mean_phi, mean_dphi, CI_dphi)
    axis[0,1].set_title("phase-space trajectory")
    # axis[0,1].set_xlabel("phi")
    # axis[0,1].set_ylabel("dphi")
                #title="phase-space trajectory",
                #xlabel=L"\phi \; [10^{-3}\textrm{m}_P]", 
                #ylabel=L"\dfrac{\mathrm{d}\phi}{\mathrm{d}z} \; [10^{-3}\textrm{m}_P]",


    axis[0,2].errorbar(zrange, mean_omega, CI_omega)
    axis[0,2].errorbar(zrange, 1.0 - mean_omega, CI_omega)
    axis[0,2].set_title("density evolution")
                #xlabel=L"\textrm{redshift}\; z", 
                #ylabel=L"\textrm{density} \; \textrm{parameter}\; \Omega", 
                #label=L"\Omega_\phi",
                #ylims=(0.0, 1.0),


    axis[1,0].errorbar(mean_phi, mean_V, CI_V) 
    axis[1,0].set_title("potential")
                    #title="potential",
                    #xlabel=L"\phi \; [10^{-3}\textrm{m}_P]", 
                    #ylabel=L"\frac{V(\phi)}{10^{-16}} \; [\textrm{eV}^4]", 
                    #legend=false

    axis[1,1].errorbar(zrange, mean_EoS, CI_EoS)
    axis[1,1].set_title("equation of state")
                    #title="equation of state",
                    #xlabel=L"\textrm{redshift}\; z", 
                    #ylabel=L"\textrm{equation}\;\textrm{of}\;\textrm{state}\; w", 
                    #legend=false

    # ϕ0_hist = plot(title="initial distribution",
    #                 xlabel=L"\phi(z=0) \; [10^{-3}\textrm{m}_P]", 
    #                 ylabel=L"\# \; \textrm{of} \; \textrm{occurences}",
    #                 legend=false
    # )
    axis[1,2].hist(phi0, bins=bins)
    axis[1,2].set_title("distribution of initial conditions")

    # dϕ0_hist = plot(title="initial distribution",
    #                 xlabel=L"\dfrac{\mathrm{d}\phi}{\mathrm{d}z}(z=0) \; [10^{-3}\textrm{m}_P]", 
    #                 ylabel=L"\# \; \textrm{of} \; \textrm{occurences}",
    #                 legend=false
    # )
    axis[1,2].hist(dphi0, bins=bins)
    plt.show()

    # Save the figure
    plt.savefig("Plots.pdf")

