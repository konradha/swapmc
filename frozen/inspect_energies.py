# RUN AS
# python inspect_energies.py configs.txt 12

import numpy as np
import matplotlib.pyplot as plt
import sys

fname = str(sys.argv[1])
L = int(sys.argv[2])

betas = np.linspace(0.1, 4., 8)
data = np.loadtxt(fname)
nepochs = int(data.shape[0] / len(betas))

def get_nn(i, j, k, L):
    iprev, inext = (i - 1) % L, (i + 1) % L 
    jprev, jnext = (j - 1) % L, (j + 1) % L
    kprev, knext = (k - 1) % L, (k + 1) % L

    return (
            k + L * (j + iprev * L), k + L * (j + inext * L), 
            k + L * (jprev + i * L), k + L * (jnext + i * L),
            kprev + L * (j + i * L), knext + L * (j + i * L)
            )

    
def local_energy(lattice, i, j, k, L):
    site = k + L * (j + i * L)
    nn = get_nn(i, j, k, L)
    e = 0.
    ty = lattice[site]
    if ty == 0: return 0
    pref = 3 if ty == 1 else 5
    for n in nn:
        e += int(lattice[n] > 0)
    return (pref - e) ** 2



def energy(lattice):
    e = 0.
    L = int(lattice.shape[0] ** .33333)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                e += local_energy(lattice, i, j, k, L)
    return e

def logand(lattice1, lattice2):
    assert  lattice1.shape == lattice2.shape
    L = int(lattice1.shape[0] ** .33333)
    q = np.sum(np.logical_and(lattice1, lattice2))
    return q



for i, b in enumerate(betas[2:7]):
    energies = []
    for epoch in range(nepochs): 
        e = energy(data[epoch * len(betas) + i]) 
        energies.append(e)
    plt.plot([2 ** j for j in range(nepochs)], energies, label=f"beta={b:.2f}")

plt.legend()
plt.xscale("log")
plt.show()


# 8 ~ 10 as done in the paper

ws = [3, 7, 12]
colors = ["black", "grey", "red"]
for wi, w in enumerate(ws):
    for i, b in enumerate(betas):
        q = []
        starting_lattice = data[i]
        for next_epoch in range(w, nepochs):
            next_lattice = data[next_epoch * len(betas) + i]
            q.append(logand(starting_lattice, next_lattice))
            starting_lattice = data[(next_epoch - w) * len(betas) + i] 
        plt.plot([2 ** j for j in range(w, nepochs)], q, label=f"beta={b:.2f}", color=colors[wi])

plt.legend()
plt.xscale("log")
plt.show()

