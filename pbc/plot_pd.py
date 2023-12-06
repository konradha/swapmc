import numpy as np
import matplotlib.pyplot as plt

beta =  [f".{i}" for i in range(1,10)]
beta += [f"1.{i}" for i in range(1,10)]
beta += [f"2.{i}" for i in range(1,10)]
beta += [f"3.{i}" for i in range(1,10)]
beta += [f"4.{i}" for i in range(1,10)]
beta += [f"6.{i}" for i in range(1,10)]
rho  = [".7", ".75", ".8"]

#fname = "beta=.1_rho=.8.txt" 

epp = []
for r in rho:
    for b in beta:
        fname = f"data/beta={b}_rho={r}.txt"
        data  = np.loadtxt(fname, delimiter=',', skiprows=2)
        epp.append(data.T[2][-1]) 
    plt.plot(beta, epp, label=f"rho={r}")
    epp = []
plt.legend()
plt.title("1e5 steps defnitely not sufficient to thermalize the _interesting_ parts of the phase diagram")
plt.show()
