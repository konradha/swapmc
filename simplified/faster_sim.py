import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

bs = [".25",    ".5", 
    "1.", 
    "1.5",
    "2.", 
    "2.5",
    "3.",  
    "3.5",
    "4."
    ]

fnames = [f"fastcheck_{b}.txt" for b in bs]
fnames_none = [f"fastcheck_noswap_{b}.txt" for b in bs]

def fun(x):
    rho1 = 3556 / 20 ** 3
    rho2 = 2370 / 20 ** 3
    rho = rho1 + rho2
    c0 = rho * (rho1**2 + rho2**2)
    N = rho * 20 **3
    return (1./N * x - c0) / (1. -c0)

for b, f in enumerate(fnames):
    data = np.loadtxt(f, delimiter=',')
    ls = data.T[1]
    #if ls[0]/5926 != 1.: print(ls) 
    ls = np.array([fun(l) for l in ls])
    if ls[0] != 1.: print(ls)
    plt.plot(data.T[0]+1, ls, label=f"b={bs[b]} with swap", color=plt.cm.RdYlBu(b/len(fnames)))
    

for b, f in enumerate(fnames_none):
    data = np.loadtxt(f, delimiter=',')
    ls = data.T[1]
    #if ls[0]/5926 != 1.: print(ls)
    ls = np.array([fun(l) for l in ls])
    if ls[0] != 1.: print(ls)
    plt.plot(data.T[0]+1, ls, label=f"b={bs[b]} no swap", linestyle='-.', color=plt.cm.RdYlBu(b/len(fnames)))
   
plt.ylim(ymin=.2, ymax=1.1)

plt.hlines(-.001, xmin=min(data.T[0]), xmax=max(data.T[0]), color="grey", linestyle='-.', linewidth=1)
plt.legend()
plt.xscale("log")

plt.show()
