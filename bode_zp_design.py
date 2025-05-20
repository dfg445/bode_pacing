import matplotlib.pyplot as plt
import numpy as np
import math
import control
import pandas as pd

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
Wn = 1.7066297676378885

f = np.logspace(-5, -1, 1000)
omega = 2 * np.pi * f

Tps = 10
Tf_cut = 10

Tf = Tf_cut / (2 * np.pi)
s = control.TransferFunction.s
order = 10


Gs = Wn * (1 - exp_taylor(-s * Tps, order)) / s
Hs = 1 / (1 + s * Tf)

Kc = 1.0


# zeros = [1e-1]
# poles = [1e-4]

zeros = [1e-1]
poles = [1e-4, 1e-2]
Gc = Kc

for zero in zeros:
    Gc *= (s / zero + 1)
for pole in poles:
    Gc /= (s / pole + 1)

G_closed_loop = Gc * Gs / (1 + Gc * Gs * Hs)

gm, pm, fcg, fcp = control.margin(Gc * Gs * Hs)
# gm, pm, fcg, fcp = control.margin(Gs * Hs)
print("gain margin (dB):", 20 * np.log10(gm))
print("phase margin (degree):", pm)
print("gm frequency (Hz):", fcg / 2 / np.pi)
print("pm frequency (Hz)", fcp / 2 / np.pi)


Ts = np.median(
    np.diff(
        pd.read_pickle(
            "auction_line_a783f80d-8926-4a41-b318-1beddca022de.pkl"
            )["ts"].map(lambda x: x.timestamp()
        )
    )
)
Gcz = control.c2d(Gc, Ts=Ts, method='tustin')
num, den = Gcz.num[0][0], Gcz.den[0][0]

print("Difference Equation:")
print("y[n] =", end=" ")
for i in range(1, len(den)):
    print(f"- ({den[i]:.3e}) * y[n-{i}]", end=" ")
print("+", end=" ")
for i in range(len(num)):
    print(f"({num[i]:.3e}) * x[n-{i}]", end=" ")


plt.close("all")

# bode_transfer_function(Gs * Hs, omega)
bode_transfer_function(Gc, omega)
bode_transfer_function(Gc * Gs * Hs, omega)
bode_transfer_function(G_closed_loop, omega)
