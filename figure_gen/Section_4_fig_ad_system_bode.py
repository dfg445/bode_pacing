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
Wn = 13.52

f = np.logspace(-5, -1, 1000)
omega = 2 * np.pi * f

Tps = 10
Tf_cut = 10

Tf = Tf_cut / (2 * np.pi)
s = control.TransferFunction.s
order = 10


Gs = Wn * (1 - exp_taylor(-s * Tps, order)) / s
Hs = 1 / (1 + s * Tf)


gm, pm, fcg, fcp = control.margin(Gs * Hs)
# gm, pm, fcg, fcp = control.margin(Gs * Hs)
print("gain margin (dB):", 20 * np.log10(gm))
print("phase margin (degree):", pm)
print("gm frequency (Hz):", fcg / 2 / np.pi)
print("pm frequency (Hz)", fcp / 2 / np.pi)


plt.close("all")
bode_transfer_function(Gs * Hs, omega)