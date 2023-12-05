import numpy as np
import matplotlib.pyplot as plt

arr = [f"{i}." for i in range(1,5)]
ei = []
betas = []
colors = ["black", "red", "blue", "orange"]
for i,e in enumerate(arr):
    fname = f"ei_{e}.txt"
    fswap = f"ei_{e}_noswap.txt"
    fswaphigher= f"ei_{e}_swapthird.txt"
    swap  = np.loadtxt(fname,delimiter=',',skiprows=1)
    none  = np.loadtxt(fswap,delimiter=',',skiprows=1)
    third = np.loadtxt(fswaphigher,delimiter=',',skiprows=1)

    epp = swap.T[2]
    epp_other = none.T[2]
    epp_higher = third.T[2]
    
    plt.plot(epp, label=f"T={e}, swap", linestyle='-.', color=colors[i], linewidth=3)
    plt.plot(epp_other,label=f"T={e}, no swap", color=colors[i], linewidth=.5)
    plt.plot(epp_higher,label=f"T={e}, swap++", color=colors[i],linestyle=':', linewidth=3)

    break

plt.legend()
plt.show()
