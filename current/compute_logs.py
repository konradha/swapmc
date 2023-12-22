import numpy as np
import matplotlib.pyplot as plt

bs = [".5", "1.", "1.1", "1.5", "3."]
fnames = [f"smallcheck_{b}.txt" for b in bs]

def fun(x):
    rho1 = .45
    rho2 = .3
    rho = rho1 + rho2
    c0 = rho * (rho1**2 + rho2**2)
    N = rho * 20 **3
    return (1./N * x - c0) / (1. -c0)

for b, f in enumerate(fnames):
    data = np.loadtxt(f, delimiter=',')
    ls = data.T[1]
    lss = ls / ls[0]
    ls = np.array([fun(l) for l in ls])
    plt.plot([10 ** i for i in range(7)], ls, label=f"b={bs[b]}")
    #plt.plot([10 ** i for i in range(7)], lss, label=f"b={bs[b]}", linestyle='-.')


plt.legend()
plt.xscale("log")
plt.show()


