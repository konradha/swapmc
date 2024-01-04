import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import sys

bs = [".25",".5",".75","1.","2.","3.","3.5","3.75","4."]

fnames = [f"olap_{b}.txt" for b in bs]
fnames_little = [f"olap_littleswap_{b}.txt" for b in bs]
fnames_none = [f"olap_noswap_{b}.txt" for b in bs]


def fun(x):
    rho1 = 3556 / 20 ** 3
    rho2 = 2370 / 20 ** 3
    rho = rho1 + rho2
    c0 = rho * (rho1**2 + rho2**2)
    N = rho * 20 **3
    return (1./N * x - c0) / (1. -c0)

fig, axs = plt.subplots(2)

for b, fname in enumerate(fnames):
    data = np.loadtxt(fname, delimiter=',')
    autocorr = data.T[1]
    overlap  = data.T[2] 
    autos = np.array([fun(l) for l in autocorr])
    overl = 1 - overlap / (3556 + 2370)
    
    axs[0].plot(data.T[0]+1, autos, label=f"beta={bs[b]}", color=plt.cm.RdYlBu(b/len(fnames)))
    axs[1].plot(data.T[0]+1, overl,  color=plt.cm.RdYlBu(b/len(fnames)))

for b, fname in enumerate(fnames_none):
    data = np.loadtxt(fname, delimiter=',')
    autocorr = data.T[1]
    overlap  = data.T[2] 
    autos = np.array([fun(l) for l in autocorr])
    overl = 1 - overlap / (3556 + 2370)
    
    axs[0].plot(data.T[0]+1, autos, linestyle='-.', color=plt.cm.RdYlBu(b/len(fnames)))
    axs[1].plot(data.T[0]+1, overl, linestyle='-.', color=plt.cm.RdYlBu(b/len(fnames)))
    

for i in range(2):
    axs[i].set_xscale("log")

axs[0].legend()

axs[0].set_title("autocorrelation")
axs[1].set_title("overlap")

fig.suptitle("straight = swap at every sweep; dashed = no swap moves")

plt.show()
