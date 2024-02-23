import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

def get_nn(i, j, k, L):
    iprev, inext = (i - 1) % L, (i + 1) % L
    jprev, jnext = (j - 1) % L, (j + 1) % L
    kprev, knext = (k - 1) % L, (k + 1) % L

    return [
            (iprev, j, k), (inext, j, k), 
            (i, jprev, k), (i, jnext, k),
            (i, j, kprev), (i, j, knext),
           ]

def energy(lattice, L):
    e = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                e += local_e(lattice, i, j, k, L)
    return int(e)

def local_e(lattice, i, j, k, L):
    nn = get_nn(i, j, k, L)
    e = 0.
    ty = lattice[i, j, k]
    if ty == 0: return 0
    if ty > 2: raise Exception("OH NO, bad particle type")
    pref = 3 if ty == 1 else 5
    for (ni, nj, nk) in nn:
        e += int(lattice[ni, nj, nk] > 0)
    return (pref - e) ** 2

def nn_e(lattice, i, j, k, L):
    nn = get_nn(i, j, k, L)
    e = local_e(lattice, i, j, k, L)
    for (ni, nj, nk) in nn:
        e += local_e(lattice, ni, nj, nk, L)
    return e

rng = np.random.default_rng()
L = 20
rho = .75
rho1 = .45
rho2 = .3

N = int(rho * L ** 3)
N1 = int(rho1 * L ** 3)
N2 = N - N1

g = np.zeros((L, L, L))
nr, nb = 0, 0
while nr < N1:
    i, j, k = rng.integers(0, L, 3) % L
    if (i + j + k) % 2 == 0 and g[i, j, k] == 0:
        g[i, j, k] = 1
        nr += 1

while nb < N2:
    i, j, k = rng.integers(0, L, 3) % L
    if (i + j + k) % 2 == 1 and g[i, j, k] == 0:
        g[i, j, k] = 2
        nb += 1

lattice = g
print(np.sum(lattice == 1), np.sum(lattice == 2))


def sweep(lattice, L, beta):
    for _ in range(L ** 3):
        i, j, k = rng.integers(0, L, 3) % L 
        ni, nj, nk = rng.integers(0, L, 3) % L
        site_ty = lattice[i, j, k]
        mv_ty = lattice[ni, nj, nk]
        if site_ty == mv_ty: continue
        E1 = nn_e(lattice, i, j, k, L) + nn_e(lattice, ni, nj, nk, L) 
        lattice[i, j, k], lattice[ni, nj, nk] = lattice[ni, nj, nk], lattice[i, j, k] 
        E2 = nn_e(lattice, i, j, k, L) + nn_e(lattice, ni, nj, nk, L)
        dE = E2 - E1
        if np.random.random() < np.exp(-beta * dE): continue 
        lattice[i, j, k], lattice[ni, nj, nk] = lattice[ni, nj, nk], lattice[i, j, k]
    assert(np.sum(lattice == 1) == N1 and np.sum(lattice == 2) == N2)
    return energy(lattice, L)

es = []
nsweeps = 100
for _ in tqdm(range(nsweeps)):
    es.append(sweep(lattice, L, 4.))
plt.plot(range(nsweeps), es)
plt.show()

def get_random_neighbor(i, j, k, L):
    nn = get_nn(i, j, k, L)
    mv = rng.integers(0, len(nn), 1)
    return nn[mv]

def local_sweep(lattice, L, beta):
    for _ in range(L ** 3):
        i, j, k = rng.integers(0, L, 3) % L
        ni, nj, nk = get_random_neighbor(i, j, k, L)
        site_ty = lattice[i, j, k]
        mv_ty = lattice[ni, nj, nk]
        if site_ty == mv_ty: continue
        E1 = nn_e(lattice, i, j, k, L) + nn_e(lattice, ni, nj, nk, L) 
        lattice[i, j, k], lattice[ni, nj, nk] = lattice[ni, nj, nk], lattice[i, j, k] 
        E2 = nn_e(lattice, i, j, k, L) + nn_e(lattice, ni, nj, nk, L)
        dE = E2 - E1
        if np.random.random() < np.exp(-beta * dE): continue 
        lattice[i, j, k], lattice[ni, nj, nk] = lattice[ni, nj, nk], lattice[i, j, k]
    assert(np.sum(lattice == 1) == N1 and np.sum(lattice == 2) == N2)
    return energy(lattice, L)

lattice_cpy = deepcopy(lattice)

betas = [.5, 4.]
lattices = [lattice, lattice_cpy]
nsweeps = 50

configs = [[deepcopy(lattice)], [deepcopy(lattice_cpy)]]
for b, beta in enumerate(betas):
    es = []
    lattice = lattices[b]
    for _ in tqdm(range(nsweeps)):
        es.append(sweep(lattice, L, beta))
        configs[b].append(lattice)
    plt.plot(range(nsweeps), es, label=f"beta={beta:.2f})")
plt.legend()
plt.xlabel("MCS / [1]")
plt.ylabel("E / [1]")
plt.show()


def q(x, rho, rho1, rho2, L):
    N = int(rho * L ** 3)
    N1 = int(rho1 * L ** 3)
    N2 = N - N1
    c0 = rho * (rho1 ** 2 + rho2 ** 2) 
    return ((1/N) * x - c0) / (1 - c0)

for b, beta in enumerate(betas):
    beg_lattice = configs[b][0]
    cs = [1.]
    for i in tqdm(range(1, nsweeps)):
        curr_config = configs[b][i]
        lat1_s1 = beg_lattice == 1
        lat2_s1 = curr_config == 1
        lat1_s2 = beg_lattice == 2
        lat2_s2 = curr_config == 2
        s = np.sum(np.logical_and(lat1_s1, lat2_s1) + np.logical_and(lat1_s2, lat2_s2))
        c = q(s, rho, rho1, rho2, L)
        cs.append(c)
    plt.plot(.1 + np.array(range(nsweeps)), cs, label=f"beta={beta:.2f})")

plt.legend()
plt.xscale("log")
plt.xlabel("MCS / [1]")
plt.ylabel("C_t / [1]")
plt.ylim(-.1, 1.1)
plt.show()
