import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from tqdm import tqdm


ps = [".1", ".25", ".3", ".45", ".6", ".7", ".8", ".9", ".95", ".99"]
bs = ["1.","2.","3."]

fname = "swap_proba_cmp_${b}_${p}.txt"

fnames = []
k = 10

fig, axs = plt.subplots(2, len(ps))
dss = []
diss = []

def array_split(array: np.array, sizes: list):
    u = []    
    for i, s in enumerate(sizes):
        u.append( array[:s] )
    return u

splits = [int(10 ** i) for i in range(1, 7)]
k = len(splits)
for j, p in tqdm(enumerate(ps)):
    ds = [] 
    dis = []
    for b in bs:
        d = []
        di = []
        fname = f"swap_proba_cmp_{b}_{p}_longer.txt"
        data = np.loadtxt(fname, delimiter=',',skiprows=1)
        #split1 = np.array_split(data.T[2], k, axis=0)
        #split2 = np.array_split(data.T[3], k, axis=0)
        split1 = array_split(data.T[2], splits)
        split2 = array_split(data.T[3], splits)
        for s in split1: 
            d.append(1. - np.sum(s == -1.) / len(s))
        for s in split2:
            di.append(1. - np.sum(s == -1.) / len(s))

        ds.append(d)
        dis.append(di)
    dss.append(np.array(ds))        
    diss.append(np.array(dis))

    U = axs[0][j].contourf(splits, bs, dss[-1], cmap=cm.PuBu_r, )
    axs[0][j].set_xscale("log")

for i, _ in enumerate(dss):
    C = axs[1][i].contourf(splits, bs, np.abs(dss[i] - dss[i-1]), cmap=cm.PuBu_r, norm=matplotlib.colors.LogNorm())
    axs[1][i].set_xscale("log")

axs[0][0].set_xlabel("window of size 1e6")
axs[0][0].set_ylabel("beta")

fig.colorbar(U, ax=axs[0].ravel().tolist())
fig.colorbar(C, ax=axs[1].ravel().tolist())

plt.show()

