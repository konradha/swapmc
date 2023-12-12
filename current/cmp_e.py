import numpy as np
import sys
import matplotlib.pyplot as plt

# fname = swap_proba_cmp_1._.1_longer.txt_cum.txt
# bash command:
# for p in .05 .1 .15 .25; do for b in 1. 2. 3.; do python proto_read.py swap_proba_cmp_${b}_${p}_longer.txt


for b in ["1.", "3.", "2."]:# [".1", "1.", "2.", "3.", "9."]:
    for p in [".1"]:# [".05", ".1", ".15", ".25"]:        
        ws = []
        for w in ["1", "5", "7", "10", "50", "75", "100", "1000"]:
            fname = f"swap_proba_cmp_{b}_{p}_longer.txt_cum_{w}.txt" 
            data  = np.loadtxt(fname, delimiter=',')
            plt.plot(data, label=f"{b=} {w=}")

plt.xscale("log")


plt.legend()
plt.show()
