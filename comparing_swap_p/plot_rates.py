import numpy as np
import matplotlib.pyplot as plt


ps = [".6", ".7", ".8", ".9", ".95", ".99"]
bs = ["1.","2.","3."]

fname = "swap_proba_cmp_${b}_${p}.txt"

fnames = []


for p in ps:
    d = []
    di = []
    for b in bs:
        fname = f"swap_proba_cmp_{b}_{p}.txt"
        data = np.loadtxt(fname, delimiter=',',skiprows=1)


        d.append(1. - np.sum(data.T[2] == -1.) /  len(data.T[2]))
        di.append(1. - np.sum(data.T[3] == -1.) / len(data.T[3]))


    plt.plot(bs, d, label=f"mc frequency @ {p}", marker='x', linewidth=.1)
    plt.plot(bs, di, label=f"swap frequency @ {p}", linestyle='-.', marker='o', linewidth=.1)

plt.legend()
plt.yscale("log")
plt.xlabel("beta")
plt.ylabel("frequency / [1]")
plt.title("1e6 steps")
plt.show()

