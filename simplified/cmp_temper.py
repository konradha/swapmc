import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import sys

fname1 = str(sys.argv[1])
fname2 = str(sys.argv[2])

def fun(x):
    rho1 = 3556 / 20 ** 3
    rho2 = 2370 / 20 ** 3
    rho = rho1 + rho2
    c0 = rho **2 * (rho1 + rho2)
    N = rho * 20 **3
    return (1./N * x - c0) / (1. -c0)


for i, f in enumerate([fname1, fname2]):
    data = np.loadtxt(f, delimiter=',')
    autocorr = data.T[1]
    overlap  = data.T[2] 
    autos = np.array([fun(l) for l in autocorr])
    overl = overlap / (3556 + 2370)

    if i == 0:
        plt.plot(data.T[0]+1, autos, linestyle='--',label=f"autocorrelation", color="red")
        plt.plot(data.T[0]+1, overl, linestyle='--',label=f"overlap", color="blue")
    else:
        plt.plot(data.T[0]+1, autos, linestyle='dotted',color="red")
        plt.plot(data.T[0]+1, overl, linestyle='dotted',color="blue")


    

plt.legend()
plt.xscale("log")
plt.title("dashed = first; dotted = second")
plt.show()
