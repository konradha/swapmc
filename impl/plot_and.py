"""
    DATA GENERATED WITH and_cmp.cpp

    AND

    for num in 100 1000 10000 100000 1000000; do for align in 32 64 128 256 512; do echo $align; ./cmp_and $num $align > data_and_${num}_${align}.txt & done; wait; done
"""
import numpy as np
import matplotlib.pyplot as plt

nums = [10 ** i for i in range(2, 7)]
align = [32, 64, 128, 256, 512] 
n_align = len(align)
fig, ax = plt.subplots(n_align)
for i, a in enumerate(align):
    d = []
    for n in nums:
        fname = f"data_and_{n}_{a}.txt"
        data = np.loadtxt(fname, delimiter=',',skiprows=1)
        slow, fast = data.T[0] / n, data.T[1] / n
        slow, fast = slow[10:len(slow)-1], fast[10:len(fast)-1]
        diff = slow > fast
        

        N = len(slow)
        xn = np.linspace(0,N+1,N) 
        ax[i].scatter(xn, fast, label=f"intrin {n=} {a=}", color="black")
        ax[i].scatter(xn, slow, label=f"normal {n=} {a=}", color="red", marker='x')
        #ax[i].vlines(diff, ymin=min(fast), ymax=max(slow), label="normal better", linewidth=.3, color="black")
        #ax[i].hlines(np.mean(fast), xmin=0, xmax=N, label=f"avg {n=}")
        ax[i].set_title(f"{a=}")
        ax[i].set_aspect("auto")        
        ax[i].set_yscale("log")

        # TODO
        # d.append(fast)
        #ax[i].violinplot((a, fast))
        absdiff = abs(slow-fast)
        meanabsdiff = np.mean(absdiff) 
        mean_fast = np.mean(fast)
        mean_slow = np.mean(slow)
        print(f"""
    {n=}
    {meanabsdiff=} {mean_fast=} {mean_slow=}""")

#plt.legend()
#plt.yscale("log")
plt.show()

