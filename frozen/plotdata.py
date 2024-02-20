import numpy as np
import matplotlib.pyplot as plt

# 0 .2 .4 .6 .8 1. 2.; do echo "beta=${b}"; ./checkerboard_sliced $b 5 12 0 > warmedup_${b}.txt

betas = ["0", ".2", ".4", ".6", ".8", "1.", "1.2", "1.4", "5."]
def q(x):
    L = 12
    rho = .75
    rho1 = .6
    rho2 = .4
    
    c0 = rho * (rho1 ** 2 + rho2 ** 2)
    N = rho * L ** 3
    return (x/N - c0) / (1-c0)




for b in betas:
    fname = f"warmedup_{b}.txt"
    data = np.loadtxt(fname, delimiter=',')
    plt.plot(.1 + data.T[0], q(data.T[1]), label=b)
plt.legend()
plt.xscale("log")
plt.show()
