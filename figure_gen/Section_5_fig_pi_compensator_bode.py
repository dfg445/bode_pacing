import matplotlib.pyplot as plt
import numpy as np
import control

def bode_transfer_function(G, omega, title=""):
    plt.figure()
    control.bode(G, omega=omega, dB=True, Hz=True)


f = np.logspace(-5, -1, 1000)
omega = 2 * np.pi * f

Tf_cut = 10
Tf = Tf_cut / (2 * np.pi)
s = control.TransferFunction.s


Kp = 0.00005
Ki = 0.00001

Gc = Kp + Ki / s


plt.close("all")
bode_transfer_function(Gc, omega)


