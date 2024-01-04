import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import sys

bs = [".25",".5",".75","1.","2.","3.","3.5","3.75","4."]

fnames = [f"olap_{b}.txt" for b in bs]

def fun(x):
    rho1 = 3556 / 20 ** 3
    rho2 = 2370 / 20 ** 3
    rho = rho1 + rho2
    c0 = rho * (rho1**2 + rho2**2)
    N = rho * 20 **3
    return (1./N * x - c0) / (1. -c0)



for fname in fnames:
    data = np.loadtxt(fname, delimiter=',')
    autocorr = data.T[1]
    overlap  = data.T[2] 
    autos = np.array([fun(l) for l in autocorr])
    overl = 1 - overlap / (3556 + 2370)
    
    plt.plot(data.T[0]+1, autos, label=f"autocorrelation", color="red")
    plt.plot(data.T[0]+1, overl, label=f"overlap", color="blue")
    

plt.legend()
plt.xscale("log")

plt.show()
