import matplotlib.pyplot as plt
import numpy as np
import math
import control
import pandas as pd
from sim_utils import (PlantSimulator, ZPcompensator, Metrics, ZeroOrderHold,
                       ModeledAdsPlant, FirstOrderLowPassFilter, TrafficFetcher)


def bode_transfer_function(G, omega, title=""):
    plt.figure()
    control.bode(G, omega=omega, dB=True, Hz=True)


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

f = np.logspace(-5, -1, 1000)
omega = 2 * np.pi * f

Tps = 10
Tf_cut = 10

Tf = Tf_cut / (2 * np.pi)
s = control.TransferFunction.s
order = 10


def system_function(zeros, poles, Kc=1.0, Wn=13.52):
    Gs = Wn * (1 - exp_taylor(-s * Tps, order)) / s
    Hs = 1 / (1 + s * Tf)

    Gc = Kc

    for zero in zeros:
        Gc *= (s / zero + 1)
    for pole in poles:
        Gc /= (s / pole + 1)

    gm, pm, fcg, fcp = control.margin(Gc * Gs * Hs)
    return gm, pm


zero_list = [(1e-1,), (1e-1,), (1e-1,)]
pole_list = [(1e-4, 1e-3), (1e-4,), (1e-3,)]


result_wn_max_min = []
print("GM_Wn_Max,", "PM_Wn_Max,", "GM_Wn_Min,", "PM_Wn_Min")

for zeros, poles in zip(zero_list, pole_list):
    gm1, pm1 = system_function(zeros, poles, Wn=Wn_max)
    gm2, pm2 = system_function(zeros, poles, Wn=Wn_min)
    element = [round(20 * np.log10(gm1), 2), round(pm1, 2), round(20 * np.log10(gm2), 2), round(pm2, 2)]

    result_wn_max_min.append(element)
    print(element)


df_pacing = pd.read_pickle("pacing_line_a783f80d-8926-4a41-b318-1beddca022de.pkl")
ts_list = []

for ts in range(int(df_pacing.loc[0, "ts"].timestamp()), int(df_pacing.loc[len(df_pacing) - 1, "ts"].timestamp()), 1):
    ts_list.append(ts)


result = []

for zeros, poles in zip(zero_list, pole_list):
    simulator = PlantSimulator(
        compensator=ZPcompensator(Kc=1.0, zeros=zeros, poles=poles, Tas=1.0),
        zoh=ZeroOrderHold(hold_interval=10),
        plant=ModeledAdsPlant(Wn=13.52, noise_var_pect=0.05),
        lpf=FirstOrderLowPassFilter(Tcut=10, sampling_time=1),
        traffic_fetcher=TrafficFetcher(
            ts_list=df_pacing["ts"].map(lambda x: x.timestamp()).tolist(),
            traffic_curve=df_pacing["forecasted_impressions_per_min"].tolist()
        ),
        daily_budget = 387.5
    )

    desired_velo_list, observed_velo_list, error_list, raw_lambda_list, hold_lambda_list,  \
        actual_velo_list, traffic_ratio_list, spend_list, integral_list = simulator.run_sim(ts_list)
    element = [Metrics.signal_line_pacing_error(desired_velo_list, actual_velo_list)]
    result.append(element)
    print("PE:", element)


freq_q = 9.302650092445086e-05

f = np.logspace(-6, -1, 10000)
omega = 2 * np.pi * f

def cut_off_bandwidth(Wn, omega):
    for zeros, poles in zip(zero_list, pole_list):
        Gs = Wn * (1 - exp_taylor(-s * Tps, order)) / s
        Hs = 1 / (1 + s * Tf)
        Gc = 1.0
    
        for zero in zeros:
            Gc *= (s / zero + 1)
        for pole in poles:
            Gc /= (s / pole + 1)
    
        G_closed_loop = Gc * Gs / (1 + Gc * Gs * Hs)
        
        mag, phase, omega = control.bode(G_closed_loop, omega=omega, plot=False) 
        mag_db = 20 * np.log10(mag)
        low_freq_gain_db = mag_db[0]  # Gain at low frequency
        target_gain_db = low_freq_gain_db - 3  # -3 dB from initial gain

        # Locate the first frequency where the gain drops below the target
        bandwidth_idx = np.where(mag_db <= target_gain_db)[0][0]
        bandwidth = omega[bandwidth_idx]  # Bandwidth in rad/s
        
        cut_off_gain = 20 * np.log10(abs(G_closed_loop(1j * 2 * np.pi * freq_q)))
        
        print("Zeros:", zeros, "Poles:", poles, f"Cut-off gain: {cut_off_gain:.3e} dB, Closed-loop bandwidth: {bandwidth / (2 * np.pi):.2e} Hz")


            

print("\nWn Max")
cut_off_bandwidth(Wn_max, omega)
print("\nWn Min")
cut_off_bandwidth(Wn_min, omega)
