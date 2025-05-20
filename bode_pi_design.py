import matplotlib.pyplot as plt
import numpy as np
import math
import control

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
Wn = 1.71

f = np.logspace(-6, -1, 10000)
omega = 2 * np.pi * f

Tps = 10
Tf_cut = 10

Tf = Tf_cut / (2 * np.pi)
s = control.TransferFunction.s
order = 10

Kp = 0.05
Ki = 0.00005

Gs = Wn * (1 - exp_taylor(-s * Tps, order)) / s
Hs = 1 / (1 + s * Tf)

Gc = Kp + Ki / s
G_closed_loop = Gc * Gs / (1 + Gc * Gs * Hs)


gm, pm, fcg, fcp = control.margin(Gs * Hs)
print("\nNon-Compensated system")
print("gain margin (dB):", 20 * np.log10(gm))
print("phase margin (degree):", pm)
print("gm frequency (Hz):", fcg / 2 / np.pi)
print("pm frequency (Hz)", fcp / 2 / np.pi)


print("\nCompensated system")
gm, pm, fcg, fcp = control.margin(Gc * Gs * Hs)
print("gain margin (dB):", 20 * np.log10(gm))
print("phase margin (degree):", pm)
print("gm frequency (Hz):", fcg / 2 / np.pi)
print("pm frequency (Hz)", fcp / 2 / np.pi)



mag, phase, omega = control.bode(G_closed_loop, omega=omega, Plot=False) 


mag_db = 20 * np.log10(mag)

# Step 4: Find the -3 dB Bandwidth
low_freq_gain_db = mag_db[0]  # Gain at low frequency
target_gain_db = low_freq_gain_db - 3  # -3 dB from initial gain

# Locate the first frequency where the gain drops below the target
bandwidth_idx = np.where(mag_db <= target_gain_db)[0][0]
bandwidth = omega[bandwidth_idx]  # Bandwidth in rad/s

print(f"Closed-Loop Bandwidth: {bandwidth:.4e} rad/s, {bandwidth / (2 * np.pi):.4e} Hz")


plt.close("all")

bode_transfer_function(Gs * Hs, omega)
bode_transfer_function(Gc, omega)
bode_transfer_function(Gc * Gs * Hs, omega)
bode_transfer_function(G_closed_loop, omega)
