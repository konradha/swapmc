import numpy as np
import matplotlib.pyplot as plt

arr = []
arr += [f"0.{i}" for i in range(1, 10)]
arr += ["1."]
arr += [f"1.{i}" for i in range(1, 10)]
arr += ["2.1", "2.3" ,"2.5", "2.7", "2.9", "3.", "3.1", "3.3", "3.5", "3.7"]
arr += ["3.8", "3.9", "4.", "4.1"]
print(arr)
ei = []
betas = []
for i,e in enumerate(arr):
    fname = f"ei_{e}.txt"
    data  = np.loadtxt(fname,delimiter=',',skiprows=1)
    epp = data.T[2]
    plt.plot(np.linspace(0,epp.size-1,epp.size), epp, label=f"{e}")
    if i%8==0:
        plt.legend()
        plt.show()


    
    betas.append(float(e))  
    ei.append(np.mean(epp[-10:-1])) 

plt.plot(betas, ei)
plt.show()
