import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

bs = [".4", "3.1"
    ]

fnames = [f"fastcheck_t_{t}_{b}.txt" for b in bs for t in range(1,6)]


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
    plt.plot(data.T[0]+1, ls, label=f"{f}", color=plt.cm.RdYlBu(b%5 + .3))
    
plt.ylim(ymin=-.001, ymax=1.1)

plt.legend()
plt.xscale("log")

plt.show()
