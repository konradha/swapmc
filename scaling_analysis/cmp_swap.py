import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import sys


Ls = [5, 10, 15, 20]
#bs = [f"{i}.{d}" for i in range(0,3) for d in range(0,10)]
bs = [f"{i}.{d}" for i in range(10, 40+1, 10) for d in [0]]


fnames = [f"run_L_{L}_beta_{b}_sliced_0.txt" for b in bs for L in Ls]
fnames_swap = [f"run_L_{L}_beta_{b}_sliced_1.txt" for b in bs for L in Ls]


def fun(x, L):
    rho = .75
    rho1 = .6
    rho2 = .4
    N = rho * L**3
    phi = rho
    q0 = (rho1 ** 2 + rho2 ** 2) * rho
    return (1/(1-q0)) * (1/N*x-q0)

def relax(autos):
    f = False
    for i, a in enumerate(autos):
        if f: autos[i] = np.nan
        if autos[i] + autos[i-1] <= .01: f = True
    return autos

        

fig, axs = plt.subplots(len(Ls))
for lidx, L in enumerate(Ls):
    for b,beta in enumerate(bs):
        fname = f"run_L_{L}_beta_{beta}_sliced_0.txt" 
        data = np.loadtxt(fname, delimiter=',')
        autocorr = data.T[1]  
        autos = np.array([fun(l, L) for l in autocorr])
        autos = relax(autos) 

            

        axs[lidx].plot(data.T[0]+1, autos, linestyle='-', label=beta, color=plt.cm.RdYlBu(b/len(bs)))

for lidx, L in enumerate(Ls):
    for b,beta in enumerate(bs):
        fname = f"run_L_{L}_beta_{beta}_sliced_1.txt" 
        data = np.loadtxt(fname, delimiter=',')
        autocorr = data.T[1] 
        #autos = autocorr 
        autos = np.array([fun(l, L) for l in autocorr])
        autos = relax(autos)
        axs[lidx].plot(data.T[0]+1, autos, linestyle='-.', color=plt.cm.RdYlBu(b/len(bs)))



for i in range(len(Ls)):
    #axs[i].hlines(.2,xmin=1,xmax=1<<17+1,linestyle='-.',color="grey")
    axs[i].set_xscale("log")
    axs[i].set_title(f"q(t), L={Ls[i]}")
    #axs[i].legend()
    #axs[i].set_ylim(-.05, 1.)

axs[0].legend(loc='upper right', bbox_to_anchor=(1.05, .55))

fig.suptitle("full line = no swap; dashed = swap at every sweep")
plt.show()
