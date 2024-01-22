import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == '__main__':
    L = 12
    fname = str(sys.argv[1])
    d = np.loadtxt(fname)
    u,c = np.unique(d, return_counts=True)
    mask = u <= 0
    nomask = u > 0

    print("immediate accepts", np.sum(c[mask] / np.sum(c)))


    plt.bar(u[mask], c[mask] / L**3, color="red", label="dE<=0")
    plt.bar(u[nomask], c[nomask] / L ** 3, color="blue", label="dE>0")

    plt.legend()


    plt.yscale("log")
    plt.show()
