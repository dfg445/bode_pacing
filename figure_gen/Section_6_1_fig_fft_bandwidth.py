import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import pandas as pd
from sim_utils import TrafficFetcher
import math
import control


df_pacing = pd.read_pickle("pacing_line_a783f80d-8926-4a41-b318-1beddca022de.pkl")
ts_list = []

for ts in range(int(df_pacing.loc[0, "ts"].timestamp()), int(df_pacing.loc[len(df_pacing) - 1, "ts"].timestamp()), 1):
    ts_list.append(ts)

traffic_fetcher = TrafficFetcher(
    ts_list=df_pacing["ts"].map(lambda x: x.timestamp()).tolist(),
    traffic_curve=df_pacing["forecasted_impressions_per_min"].tolist()
)

traffic_curve = []
for ts in ts_list:
    traffic_curve.append(traffic_fetcher.get_traffic(ts))

x = [x[0] for x in traffic_curve]

# %%

X = fft(x)
N = len(X)
n = np.arange(N)

T = N / 1
freq = n / T
period_log10 = np.log10(1 / freq)

plt.close("all")
plt.figure()

plt.stem(freq[2:200], np.abs(X)[2:200], 'b', markerfmt=".")
plt.xlabel('Freq (Hz)')
plt.ylabel('Traffic Curve FFT Amplitude')
plt.xscale('log', base=10)
plt.grid()


print("Max freqneucy:", freq[8])

def exp_taylor(x, order=10):
  e = 0
  for i in range(0, order):
    e += (x ** i) / math.factorial(i)
  return e


s = control.TransferFunction.s

Wn_max = 13.52
Wn_min = 1.03


def cut_off_sim(Wn):
    Tps = 10
    Tf_cut = 10
    Tf = Tf_cut / (2 * np.pi)
    order = 10
    
    freq_q = 9.302650092445086e-05
    
    
    Kp_list = [5e-2, 5e-3, 5e-4]
    Ki_list = [5e-3, 5e-4, 5e-5]
    
    for j in range(len(Ki_list)):
        for i in range(len(Kp_list)):
            Kp = Kp_list[i]
            Ki = Ki_list[j]
            
            Gs = Wn * (1 - exp_taylor(-s * Tps, order)) / s
            Hs = 1 / (1 + s * Tf)
            Gc = Kp + Ki / s
            G_closed_loop = Gc * Gs / (1 + Gc * Gs * Hs)
            
            print("Kp:", Kp, "Ki:", Ki, "Cut-off gain:", 20 * np.log10(abs(G_closed_loop(1j * 2 * np.pi * freq_q))))
    
    
    
    zero_list = [(1e-1,), (1e-1,), (2e-3,)]
    pole_list = [(1e-4, 1e-3), (1e-4,), (1e-4,)]
    index = 0
    
    for zeros, poles in zip(zero_list, pole_list):
        index += 1
        Gs = Wn * (1 - exp_taylor(-s * Tps, order)) / s
        Hs = 1 / (1 + s * Tf)
        Gc = 1.0
    
        for zero in zeros:
            Gc *= (s / zero + 1)
        for pole in poles:
            Gc /= (s / pole + 1)
    
        G_closed_loop = Gc * Gs / (1 + Gc * Gs * Hs)
        print("Index:", index, "Cut-off gain:", 20 * np.log10(abs(G_closed_loop(1j * 2 * np.pi * freq_q))))

print("Wn Max")
cut_off_sim(Wn_max)

print("\nWn Min")
cut_off_sim(Wn_min)

