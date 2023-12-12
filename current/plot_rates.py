import numpy as np
import matplotlib.pyplot as plt


ps = [".6", ".7", ".8", ".9", ".95", ".99"]
bs = ["1.","2.","3."]

fname = "swap_proba_cmp_${b}_${p}.txt"

fnames = []
split = True

for p in ps:
    d = []
    di = []
    for b in bs:
        fname = f"swap_proba_cmp_{b}_{p}.txt"
        data = np.loadtxt(fname, delimiter=',',skiprows=1)

        #if split:
        #    split1 = np.array_split(data.T[2], 10, axis=0)
        #    split2 = np.array_split(data.T[3], 10, axis=0)
        #    for s in split1:
        #        d.append(1. - np.sum(s == -1.) / len(s))
        #    for s in split2:
        #        di.append(1. - np.sum(s == -1.) / len(s))
        #    break
                


        d.append(1. - np.sum(data.T[2] == -1.) /  len(data.T[2]))
        di.append(1. - np.sum(data.T[3] == -1.) / len(data.T[3]))

    #print(d)
    #print(di)
    plt.plot(bs, d, label=f"mc frequency @ {p}", marker='x', linewidth=.1)
    plt.plot(bs, di, label=f"swap frequency @ {p}", linestyle='-.', marker='o', linewidth=.1)

plt.legend()
plt.yscale("log")
plt.xlabel("beta")
plt.ylabel("frequency / [1]")
plt.title("1e6 steps")
plt.show()

