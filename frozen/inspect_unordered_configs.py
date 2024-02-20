import numpy as np
import matplotlib.pyplot as plt
from sys import argv

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
    if ty > 2: raise Exception("OH NO, bad particle type")

    pref = 3 if ty == 1 else 5
    for n in nn:
        e += int(lattice[n] > 0)
    return (pref - e) ** 2



def energy(lattice, L):
    e = 0.
    for i in range(L):
        for j in range(L):
            for k in range(L):
                e += local_energy(lattice, i, j, k, L)
    return e

def eq(lat1, lat2):
    assert lat1.shape == lat2.shape
    q = 0.
    N = int(lat1.shape[0])
    for i in range(N):
        q += int(lat1[i] == lat2[i] and lat1[i] > 0)
    return q



configs_name = str(argv[1])
ordering_name = str(argv[2])
L = int(argv[3])

configs = np.loadtxt(configs_name,)
beta_orderings = np.loadtxt(ordering_name,)


ordered_configs = []
for i, ordering in enumerate(beta_orderings):
    num_betas = len(ordering)
    current_configs = configs[i * num_betas : (i + 1) * num_betas]
    argsort = ordering.argsort() 
    ordered_configs.append(current_configs[argsort]) 


ordered_configs = np.array(ordered_configs, dtype=np.int8)
num_epochs = ordered_configs.shape[0]


for i, b in enumerate(beta_orderings[0]):
    es = []
    for epoch in range(num_epochs):  
        current_config = ordered_configs[epoch][i] 
        es.append(energy(current_config, L)) 
    plt.plot([2 ** i for i in range(num_epochs)], es, label=f"beta={b:.2f}")

plt.xscale("log")
plt.legend()
plt.show()
