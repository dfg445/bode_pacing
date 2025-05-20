import matplotlib.pyplot as plt
import numpy as np
import math
import control

def bode_transfer_function(G, omega, title=""):
    plt.figure()
    return control.bode(G, omega=omega, dB=True, Hz=True)



f = np.logspace(-1, 3, 1000)
omega = 2 * np.pi * f

s = control.TransferFunction.s

Kc = 20
z1 = 10 * 2 * np.pi
p1 = 10 * 2 * np.pi

Gc = Kc * (s / z1 + 1)



plt.close("all")
bode_transfer_function(Gc, omega)



Gc = Kc / (s / p1 + 1)
bode_transfer_function(Gc, omega)


