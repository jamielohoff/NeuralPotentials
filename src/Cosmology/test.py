from classicplot import plot_data, plot_heatmap
import numpy as np

SEED = 42
np.random.seed(SEED)
N = 10

zrange = np.linspace(0.0, 7.0, num=N)
mean_V = np.random.rand(N)
CI_V = [mean_V + 0.2, mean_V - 0.2]
mean_omega = np.random.rand(N)
CI_omega = [mean_omega + 0.2, mean_omega - 0.2]
mean_H = np.random.rand(N)
CI_H =[mean_H + 0.2, mean_H - 0.2]
mean_phi = np.random.rand(N)
CI_phi = [mean_phi + 0.2, mean_phi - 0.2]
mean_dphi = np.random.rand(N)
CI_dphi = [mean_dphi + 0.2, mean_dphi - 0.2]
mean_EoS = np.random.rand(N)
CI_EoS = [mean_EoS + 0.2, mean_EoS - 0.2]
phi0 = np.random.randn(N)
dphi0 = np.random.randn(N)

# plot_data(zrange, 
#     mean_V, CI_V,
#     mean_omega, CI_omega,
#     mean_H, CI_H,
#     mean_phi, CI_phi,
#     mean_dphi, CI_dphi,
#     mean_EoS, CI_EoS,
#     phi0, dphi0, 
#     save=False, alpha=0.5
# )

map = np.random.rand(100,100)
plot_heatmap(map)

