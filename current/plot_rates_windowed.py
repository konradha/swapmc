import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib


ps = [".6", ".7", ".8",]# ".9", ".95", ".99"]
bs = ["1.","2.","3."]

fname = "swap_proba_cmp_${b}_${p}.txt"

fnames = []
k = 10

fig, axs = plt.subplots(2, len(ps))
dss = []
diss = []

for j, p in enumerate(ps):
    ds = [] 
    dis = []
    for b in bs:
        d = []
        di = []
        fname = f"swap_proba_cmp_{b}_{p}_longer.txt"
        data = np.loadtxt(fname, delimiter=',',skiprows=1)
        split1 = np.array_split(data.T[2], k, axis=0)
        split2 = np.array_split(data.T[3], k, axis=0)
        for s in split1: 
            d.append(1. - np.sum(s == -1.) / len(s))
        for s in split2:
            di.append(1. - np.sum(s == -1.) / len(s))

        #plt.plot(range(1, k+1), d, label=f"mc frequency {b=} @ {p=}", marker='x')
        #plt.plot(range(1, k+1), di, linestyle='-.', label=f"swap frequency {b=} @ {p=}", marker='o')
        ds.append(d)
        dis.append(di)
    dss.append(np.array(ds))        
    diss.append(np.array(dis))

    U = axs[0][j].contourf(range(1, k+1), bs, dss[-1], cmap=cm.PuBu_r, )



    #plt.plot(bs, d, label=f"mc frequency @ {p}", marker='x', linewidth=.1)
    #plt.plot(bs, di, label=f"swap frequency @ {p}", linestyle='-.', marker='o', linewidth=.1)

#for i, _ in enumerate(dss):
#    C = axs[1][i].contourf(range(1, k+1), bs, np.abs(dss[i] - dss[i-1]), cmap=cm.PuBu_r, norm=matplotlib.colors.LogNorm())

axs[0][0].set_xlabel("window of size 1e6")
axs[0][0].set_ylabel("beta")

fig.colorbar(U, ax=axs[0].ravel().tolist())
#fig.colorbar(C, ax=axs[1].ravel().tolist())

plt.legend()
plt.show()

