import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('classic')

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] =  r'\usepackage{siunitx}'

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 18

def plot_data(
    zrange, 
    mean_V, CI_V,
    mean_omega, CI_omega,
    mean_H, CI_H,
    mean_phi, CI_phi,
    mean_dphi, CI_dphi,
    mean_EoS, CI_EoS,
    phi0, dphi0,
    bins=30, save=False, 
    **kwargs
):
    # Plot all the stuff into a 4x2 figure
    figure, axis = plt.subplots(2, 4)
    figure.tight_layout()
    axisFontSize = 22

    # Plot expansion rate
    axis[0,0].plot(zrange, mean_H)
    axis[0,0].fill_between(zrange, CI_H[0], CI_H[1], **kwargs)

    axis[0,0].set_title(r'$\mathrm{expansion}\;\mathrm{rate}$')
    axis[0,0].set_xlabel(r'$z$', fontsize=axisFontSize)
    axis[0,0].set_ylabel(r'$Ha^{(3 \backslash 2)} \; [\mathrm{km}\mathrm{s}^{-1}\mathrm{Mpc}^{-1}]$', fontsize=axisFontSize)

    # Plot phase-space trajectory
    axis[0,1].plot(mean_phi, mean_dphi)
    axis[0,1].fill_between(mean_phi, CI_dphi[0], CI_dphi[1], **kwargs)

    axis[0,1].set_title(r'$\mathrm{phasespace}\;\mathrm{trajectory}$')
    axis[0,1].set_xlabel(r"$\phi \; [10^{-3}\mathrm{m}_\mathrm{P}]$", fontsize=axisFontSize)
    axis[0,1].set_ylabel(r"$\frac{\mathrm{d}\phi}{\mathrm{d}z} \; [10^{-3}\mathrm{m}_\mathrm{P}]$", fontsize=axisFontSize)

    # Plot evolution of density parameter
    axis[0,2].plot(zrange, mean_omega)
    axis[0,2].fill_between(zrange, CI_omega[0], CI_omega[1], **kwargs)

    axis[0,2].plot(zrange, 1.0 - mean_omega, color='green')
    axis[0,2].fill_between(zrange, 1.0 - CI_omega[0], 1.0 - CI_omega[1], facecolor='green', **kwargs)

    axis[0,2].set_title(r'$\mathrm{density}\;\mathrm{evolution}$')
    axis[0,2].set_xlabel(r'$z$', fontsize=axisFontSize)
    axis[0,2].set_ylabel(r'$\mathrm{density} \; \mathrm{parameter}\; \Omega$', fontsize=axisFontSize)
    axis[0,2].set_ylim([0.0, 1.0])

    # Plot the potential
    axis[1,0].plot(mean_phi, mean_V)
    axis[1,0].fill_between(mean_phi, CI_V[0], CI_V[1], **kwargs) 

    axis[1,0].set_title(r'$\mathrm{potential}$')
    axis[1,0].set_xlabel(r'$\phi \; [10^{-3}\mathrm{m}_\mathrm{P}]$', fontsize=axisFontSize) 
    axis[1,0].set_ylabel(r'$\frac{V(\phi)}{10^{-16}} \; [\mathrm{eV}^4]$', fontsize=axisFontSize) 

    # Plot the evolution of the equation of state
    axis[1,1].plot(zrange, mean_EoS)
    axis[1,1].fill_between(zrange, CI_EoS[0], CI_EoS[1], **kwargs)
    axis[1,1].set_title(r'$\mathrm{equation}\;\mathrm{of}\;\mathrm{state}$')
    axis[1,1].set_xlabel(r'$z$', fontsize=axisFontSize)
    axis[1,1].set_ylabel(r'$\mathrm{equation}\;\mathrm{of}\;\mathrm{state}\; w$', fontsize=axisFontSize) 

    # Plot distribution of phi-values
    axis[0,3].hist(phi0, bins=bins)
    axis[0,3].set_title(r'$\mathrm{distribution}\;\mathrm{of}\;\mathrm{initial}\;\mathrm{conditions}$')
    axis[0,3].set_xlabel(r'$\phi(z=0) \; [10^{-3}\mathrm{m}_\mathrm{P}]$', fontsize=axisFontSize)
    axis[0,3].set_ylabel(r'$\# \; \mathrm{of} \; \mathrm{occurences}$', fontsize=axisFontSize) 

    # Plot distribution of dphi-values
    axis[1,3].hist(dphi0, bins=bins)
    axis[1,3].set_title(r'$\mathrm{distribution}\;\mathrm{of}\;\mathrm{initial}\;\mathrm{conditions}$')
    axis[1,3].set_xlabel(r'$\frac{\mathrm{d}\phi}{\mathrm{d}z}(z=0) \; [10^{-3}\mathrm{m}_\mathrm{P}]$', fontsize=axisFontSize)
    axis[1,3].set_ylabel(r'$\# \; \mathrm{of} \; \mathrm{occurences}$', fontsize=axisFontSize) 

    plt.show()

    # Save the figure
    if save:
        plt.savefig("Plots.pdf")

def plot_heatmap(map, save=False):
    # ticks = 200:200:1000
    # labels = round.([mean_V[i] for i ∈ ticks]; digits=3)
    # heat_plot = heatmap(map, size=(1200,1200), c=:plasma,
    #                     title="covariance matrix",
    #                     xlabel=L"\textrm{potential} \; \frac{V(\phi)}{10^{-16}} \; [\textrm{eV}^4]",
    #                     ylabel=L"\textrm{potential} \; \frac{V(\phi)}{10^{-16}} \; [\textrm{eV}^4]",
    #                     xticks=(ticks, labels),
    #                     yticks=(ticks, labels),
    # )
    figure, axis = plt.subplots()
    figure.tight_layout()
    axisFontSize = 22

    axis.set_title(r'$\mathrm{covariance}\;\mathrm{matrix}$')
    axis.set_xlabel(r'$\mathrm{potential} \; \frac{V(\phi)}{10^{-16}} \; [\mathrm{eV}^4]$', fontsize=axisFontSize)
    axis.set_ylabel(r'$\mathrm{potential} \; \frac{V(\phi)}{10^{-16}} \; [\mathrm{eV}^4]$', fontsize=axisFontSize)

    axis.imshow(map)
    plt.show()

    # Save the figure
    if save:
        plt.savefig("CovarianceMatrix.pdf")


