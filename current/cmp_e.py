import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

fn = str(sys.argv[1])
fnames = ["swap_proba_cmp_2._.1_max.txt", "swap_proba_cmp_3._.1_max.txt", "swap_proba_cmp_4._.1_max.txt"]
windows = [10 **i for i in range(6)]

for i, fstart in tqdm(enumerate(fnames)):
    ds = []
    for w in windows:
        fname = f"{fstart}_corr_{w}.txt" 
        data  = np.loadtxt(fname, delimiter=',')
        ds.append(np.mean(data[w:]))
    plt.plot(windows, ds, marker='x', linewidth=.3, label=f"beta={i+1}")

plt.xscale("log")


plt.legend()
plt.show()
