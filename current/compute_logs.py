import numpy as np
import matplotlib.pyplot as plt

#bs = [".01",".1", ".2", ".3", ".4", ".5", "1.", "1.1", "1.5", "3.", "3.5", "4.", "4.5", "5."]
bs = [".01",
    ".1",   
    ".2",
    ".3",
    ".4",
    ".5",
    ".6",
    ".7",
    ".8",
    ".9",
    "1.",
    "1.2",
    "1.4",
    "1.6",
    "1.8",
    "2.",
    "2.2",
    "2.4",
    "2.6",
    "2.8",
    "3.",
    "3.2",
    "3.4",
    "3.6"
    ]
fnames = [f"smallcheck_new_warmed_{b}.txt" for b in bs]

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
    lss = ls / ls[0]
    ls = np.array([fun(l) for l in ls])
    plt.plot(data.T[0], ls, label=f"b={bs[b]}", color=plt.cm.RdYlBu(b/len(bs)))
    #plt.plot(data.T[0], lss, label=f"b={bs[b]}", color=plt.cm.RdYlBu(b/len(fnames)))
    

plt.hlines(0, xmin=min(data.T[0]), xmax=max(data.T[0]), color="grey", linestyle='-.')
plt.legend()
plt.xscale("log")
plt.show()


