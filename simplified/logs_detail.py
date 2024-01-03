import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

#bs = [".01",".1", ".2", ".3", ".4", ".5", "1.", "1.1", "1.5", "3.", "3.5", "4.", "4.5", "5."]
bs = [    ".5", 
    "1.", 
    "1.5",
    "2.", 
    "2.5",
    "3.",  
    "3.5" 
    ]


fnames = [f"check_withswap_{b}.txt" for b in bs]
fnames_none = [f"check_noswap_{b}.txt" for b in bs]


def fun(x):
    rho1 = 3556 / 20 ** 3
    rho2 = 2370 / 20 ** 3
    rho = rho1 + rho2
    c0 = rho * (rho1**2 + rho2**2)
    N = rho * 20 **3
    return (1./N * x - c0) / (1. -c0)

def smooth(data):
    cumsum = np.cumsum(data)  # Compute the cumulative sum of data
    smoothed_data = np.zeros_like(data)
    for i in range(len(data)):
        if i < 10:
            window_size = i + 1
        elif i < 100:
            window_size = 10
        else:
            window_size = i // 10

        if i - window_size >= 0:
            window_sum = cumsum[i] - cumsum[i - window_size]
        else:
            window_sum = cumsum[i]
        smoothed_data[i] = window_sum / window_size
    return smoothed_data

for b, f in enumerate(fnames):
    data = np.loadtxt(f, delimiter=',')
    ls = data.T[1]
    lss = ls / ls[0]
    ls = np.array([fun(l) for l in ls])

    #plt.plot(data.T[0][4:], ls[4:], label=f"b={bs[b]}", color=plt.cm.RdYlBu(b/len(bs)))
    #plt.plot(data.T[0], savgol_filter(ls, 10, 3), label=f"b={bs[b]}", color=plt.cm.RdYlBu(b/len(bs)))
    plt.plot(data.T[0], lss, label=f"b={bs[b]} with swap", color=plt.cm.RdYlBu(b/len(fnames)))

for b, f in enumerate(fnames_none):
    data = np.loadtxt(f, delimiter=',')
    ls = data.T[1]
    lss = ls / ls[0]
    ls = np.array([fun(l) for l in ls])

    plt.plot(data.T[0], lss, label=f"b={bs[b]} no swap", linestyle='-.', color=plt.cm.RdYlBu(b/len(fnames)))
    

plt.hlines(0, xmin=min(data.T[0]), xmax=max(data.T[0]), color="grey", linestyle='-.')
plt.legend()
plt.xscale("log")
#plt.savefig("naive_fig2a_1e7.png", dpi=300)
plt.show()
