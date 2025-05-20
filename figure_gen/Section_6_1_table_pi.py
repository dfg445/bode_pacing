import matplotlib.pyplot as plt
import numpy as np
import math
import control
import pandas as pd
from sim_utils import (PlantSimulator, PIcompensator, Metrics, ZeroOrderHold,
                       ModeledAdsPlant, FirstOrderLowPassFilter, TrafficFetcher)

def exp_taylor(x, order=10):
  e = 0
  for i in range(0, order):
    e += (x ** i) / math.factorial(i)
  return e

"""
max $ / second * lambda 13.521057805716055 
min $ / second * lambda  1.7066297676378885 
"""
Wn_max = 13.52
Wn_min = 1.71


Tps = 10
Tf_cut = 10
Tf = Tf_cut / (2 * np.pi)
s = control.TransferFunction.s
order = 10


def system_function(Kp, Ki, freq_q=0, Wn=13.52):
    Gs = Wn * (1 - exp_taylor(-s * Tps, order)) / s
    Hs = 1 / (1 + s * Tf)
    Gc = Kp + Ki / s

    G_open_loop = Gc * Gs * Hs
    G_closed_loop = Gc * Gs / (1 + Gc * Gs * Hs)

    gm, pm, fcg, fcp = control.margin(G_open_loop)

    return gm, pm, abs(G_closed_loop(1j * 2 * np.pi * freq_q))

num_points = 10

Kp_list = np.logspace(-2, -5, num=num_points) 
Ki_list = np.logspace(-2, -5, num=num_points)
Kp_grid, Ki_grid = np.meshgrid(Kp_list, Ki_list)


gm_grid_wn_max = np.zeros([num_points, num_points])
pm_grid_wn_max = np.zeros([num_points, num_points])
gm_grid_wn_min = np.zeros([num_points, num_points])
pm_grid_wn_min = np.zeros([num_points, num_points])


for i in range(num_points):
    for j in range(num_points):
        gm, pm, _ = system_function(Kp_grid[i][j], Ki_grid[i][j], Wn=Wn_max)
        gm_grid_wn_max[i][j] = 20 * np.log10(gm)
        pm_grid_wn_max[i][j] = pm
        
        gm, pm, _ = system_function(Kp_grid[i][j], Ki_grid[i][j], Wn=Wn_min)
        gm_grid_wn_min[i][j] = 20 * np.log10(gm)
        pm_grid_wn_min[i][j] = pm


plt.close("all")
ticks = np.linspace(1e-2, 1e-5, num=3)

fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'})


surf = axs[0][0].plot_surface(Ki_grid, Kp_grid, gm_grid_wn_max, edgecolor='royalblue', lw=1.0, alpha=0.5)
axs[0][0].set(
    xlim=(0.011, -0.001), ylim=(0.011, -0.001), zlim=(-50, 60), 
    xticks=ticks, yticks=ticks,
    xlabel='Ki', ylabel='Kp', zlabel='Max Wn GM (dB)'
)
axs[0][0].contourf(Ki_grid, Kp_grid, gm_grid_wn_max, zdir='z', offset=-40, cmap='Spectral', alpha=0.7)
axs[0][0].contourf(Ki_grid, Kp_grid, gm_grid_wn_max, zdir='x', offset=-0.001, cmap='Spectral', alpha=0.7)
axs[0][0].contourf(Ki_grid, Kp_grid, gm_grid_wn_max, zdir='y', offset=0.011, cmap='Spectral', alpha=0.7)


surf = axs[0][1].plot_surface(Ki_grid, Kp_grid, pm_grid_wn_max, edgecolor='royalblue', lw=1.0, alpha=0.5)
axs[0][1].set(
    xlim=(0.011, -0.001), ylim=(0.011, -0.001), zlim=(-100, 150),
    xticks=ticks, yticks=ticks,
    xlabel='Ki', ylabel='Kp', zlabel='Max Wn PM (degree)'
)
axs[0][1].contourf(Ki_grid, Kp_grid, pm_grid_wn_max, zdir='z', offset=-90, cmap='Spectral', alpha=0.7)
axs[0][1].contourf(Ki_grid, Kp_grid, pm_grid_wn_max, zdir='x', offset=-0.001, cmap='Spectral', alpha=0.7)
axs[0][1].contourf(Ki_grid, Kp_grid, pm_grid_wn_max, zdir='y', offset=0.011, cmap='Spectral', alpha=0.7)


surf = axs[1][0].plot_surface(Ki_grid, Kp_grid, gm_grid_wn_min, edgecolor='royalblue', lw=1.0, alpha=0.5)
axs[1][0].set(
    xlim=(0.011, -0.001),ylim=(0.011, -0.001),zlim=(-20, 60),
    xticks=ticks, yticks=ticks,
    xlabel='Ki', ylabel='Kp', zlabel='Min Wn GM (dB)'
)
axs[1][0].contourf(Ki_grid, Kp_grid, gm_grid_wn_min, zdir='z', offset=-10, cmap='Spectral', alpha=0.7)
axs[1][0].contourf(Ki_grid, Kp_grid, gm_grid_wn_min, zdir='x', offset=-0.001, cmap='Spectral', alpha=0.7)
axs[1][0].contourf(Ki_grid, Kp_grid, gm_grid_wn_min, zdir='y', offset=0.011, cmap='Spectral', alpha=0.7)


surf = axs[1][1].plot_surface(Ki_grid, Kp_grid, pm_grid_wn_min, edgecolor='royalblue', lw=1.0, alpha=0.5)
axs[1][1].set(
    xlim=(0.011, -0.001), ylim=(0.011, -0.001), zlim=(20, 100),
    xticks=ticks, yticks=ticks,
    xlabel='Ki', ylabel='Kp', zlabel='Min Wn PM (degree)'
)
# axs[1][1].set_title('Min Wn Phase Margin')
axs[1][1].contourf(Ki_grid, Kp_grid, pm_grid_wn_min, zdir='z', offset=23, cmap='Spectral', alpha=0.7)
axs[1][1].contourf(Ki_grid, Kp_grid, pm_grid_wn_min, zdir='x', offset=-0.001, cmap='Spectral', alpha=0.7)
axs[1][1].contourf(Ki_grid, Kp_grid, pm_grid_wn_min, zdir='y', offset=0.011, cmap='Spectral', alpha=0.7)

"""
This is to display data in tables
"""
Kp_list = [5e-2, 5e-3, 5e-4]
Ki_list = [5e-3, 5e-4, 5e-5]


result_wn_max_min = []
print("\nKp,", "Ki,", "GM_Wn_Max,", "PM_Wn_Max,", "gain_Wn_Max,", "GM_Wn_Min,", "PM_Wn_Min", "gain_Wn_Min")
for j in range(len(Ki_list)):
    for i in range(len(Kp_list)):
        Kp = Kp_list[i]
        Ki = Ki_list[j]
        gm1, pm1, gain1 = system_function(Kp, Ki, freq_q=9.30265e-05, Wn=Wn_max)
        gm2, pm2, gain2 = system_function(Kp, Ki, freq_q=9.30265e-05, Wn=Wn_min)
        element = [
            Kp, 
            Ki, 
            round(20 * np.log10(gm1), 2), 
            round(pm1, 2), 
            round(20 * np.log10(gain1), 4), 
            round(20 * np.log10(gm2), 4), 
            round(pm2, 2), 
            round(20 * np.log10(gain2), 4),
        ]
        
        result_wn_max_min.append(element)
        print(element)



df_pacing = pd.read_pickle("pacing_line_a783f80d-8926-4a41-b318-1beddca022de.pkl")
ts_list = []

for ts in range(int(df_pacing.loc[0, "ts"].timestamp()), int(df_pacing.loc[len(df_pacing) - 1, "ts"].timestamp()), 1):
    ts_list.append(ts)


result = []

Wn = 13.52
print("\nWn:", Wn)
print("Kp,", "Ki,", "PE")
for j in range(len(Ki_list)):
    for i in range(len(Kp_list)):
        Kp = Kp_list[i]
        Ki = Ki_list[j]
    
        simulator = PlantSimulator(
            compensator=PIcompensator(Kp=Kp, Ki=Ki, integral_init=0.0 / Ki, integral_limits=(0, 0.1)),
            zoh=ZeroOrderHold(hold_interval=10),
            plant=ModeledAdsPlant(Wn=Wn, noise_var_pect=0.05),
            lpf=FirstOrderLowPassFilter(Tcut=10, sampling_time=1),
            traffic_fetcher=TrafficFetcher(
                ts_list=df_pacing["ts"].map(lambda x: x.timestamp()).tolist(),
                traffic_curve=df_pacing["forecasted_impressions_per_min"].tolist()
            ),
            daily_budget = 387.5
        )
    
        desired_velo_list, observed_velo_list, error_list, raw_lambda_list, hold_lambda_list,  \
            actual_velo_list, traffic_ratio_list, spend_list, integral_list = simulator.run_sim(ts_list)
        element = [Kp, Ki, Metrics.signal_line_pacing_error(desired_velo_list, actual_velo_list)]
        result.append(element)
        print(element)


freq_q = 9.302650092445086e-05


f = np.logspace(-6, -1, 10000)
omega = 2 * np.pi * f


def cut_off_bandwidth(Wn, omega):
    for j in range(len(Ki_list)):
        for i in range(len(Kp_list)):
            Kp = Kp_list[i]
            Ki = Ki_list[j]
            
            Gs = Wn * (1 - exp_taylor(-s * Tps, order)) / s
            Hs = 1 / (1 + s * Tf)
            Gc = Kp + Ki / s
            G_closed_loop = Gc * Gs / (1 + Gc * Gs * Hs)
            
            mag, phase, omega = control.bode(G_closed_loop, omega=omega, plot=False) 
            mag_db = 20 * np.log10(mag)
            low_freq_gain_db = mag_db[0]  # Gain at low frequency
            target_gain_db = low_freq_gain_db - 3  # -3 dB from initial gain

            # Locate the first frequency where the gain drops below the target
            bandwidth_idx = np.where(mag_db <= target_gain_db)[0][0]
            bandwidth = omega[bandwidth_idx]  # Bandwidth in rad/s
            
            cut_off_gain = 20 * np.log10(abs(G_closed_loop(1j * 2 * np.pi * freq_q)))
            
            print("Kp:", Kp, "Ki:", Ki, f"Cut-off gain: {cut_off_gain:.3e} dB, Closed-loop bandwidth: {bandwidth / (2 * np.pi):.2e} Hz")
   


print("\nWn Max")
cut_off_bandwidth(Wn_max, omega)
print("\nWn Min")
cut_off_bandwidth(Wn_min, omega)
