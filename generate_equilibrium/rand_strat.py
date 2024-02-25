import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from sys import argv

def get_nn(i, j, k, L):
    iprev, inext = (i - 1) % L, (i + 1) % L
    jprev, jnext = (j - 1) % L, (j + 1) % L
    kprev, knext = (k - 1) % L, (k + 1) % L

    return [
            (iprev, j, k), (inext, j, k), 
            (i, jprev, k), (i, jnext, k),
            (i, j, kprev), (i, j, knext),
           ]

def generate_nn_list(L):
    d = {}
    for i in range(L):
        d[i] = {}
        for j in range(L):
            d[i][j] = {}
            for k in range(L):
                d[i][j][k] = get_nn(i, j, k, L)
    return d

rng = np.random.default_rng()
L = 12
nn_list = generate_nn_list(12) 

def energy(lattice, L):
    e = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                e += local_e(lattice, i, j, k, L)
    return int(e)

def local_e(lattice, i, j, k, L):
    #nn = get_nn(i, j, k, L)
    nn = nn_list[i][j][k]
    e = 0.
    ty = lattice[i, j, k]
    if ty == 0: return 0
    if ty > 2: raise Exception("OH NO, bad particle type")
    pref = 3 if ty == 1 else 5
    for (ni, nj, nk) in nn:
        e += int(lattice[ni, nj, nk] > 0)
    return (pref - e) ** 2

def nn_e(lattice, i, j, k, L):
    #nn = get_nn(i, j, k, L)
    nn = nn_list[i][j][k]
    e = local_e(lattice, i, j, k, L)
    for (ni, nj, nk) in nn:
        e += local_e(lattice, ni, nj, nk, L)
    return e


rho = .75
rho1 = .45
rho2 = .3

N = int(rho * L ** 3)
N1 = int(rho1 * L ** 3)
N2 = N - N1

g = np.zeros((L, L, L)).astype(int)
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
print(N1, N2)
print(np.sum(lattice == 1), np.sum(lattice == 2))
print(lattice.shape)
pass


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
        if dE <= 0 or np.random.random() < np.exp(-beta * dE): continue 
        lattice[i, j, k], lattice[ni, nj, nk] = lattice[ni, nj, nk], lattice[i, j, k]
    #assert(np.sum(lattice == 1) == N1 and np.sum(lattice == 2) == N2)
    return energy(lattice, L)

es = [energy(lattice, L)]
nsweeps = 100
for _ in tqdm(range(nsweeps)):
    es.append(sweep(lattice, L, 5.))
print(es)
plt.plot([0.1, *np.array(range(1,nsweeps+1))], es)
plt.show()


des = []
for i in range(L):
    for j in range(L):
        for k in range(L):
            if lattice[i, j, k] == 0: continue
            #nn = get_nn(i, j, k, L)
            nn = nn_list[i][j][k]
            nn_b = nn_e(lattice, i, j, k, L)
            for (ni, nj, nk) in nn: 
                if lattice[i, j, k] == lattice[ni, nj, nk]: continue 
                nn_a = nn_e(lattice, ni, nj, nk, L)
                des.append(nn_a - nn_b)
des = np.array(des)
u, c = np.unique(des, return_counts=True)
c = c / np.sum(c)
plt.bar(u, c)
plt.vlines(np.sum(u * c), ymin=0, ymax=max(c))
plt.show()



def get_random_neighbor(i, j, k, L):
    #nn = get_nn(i, j, k, L)
    nn = nn_list[i][j][k]
    mv = rng.integers(0, len(nn), 1)[0]
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
        if dE <= 0 or np.random.random() < np.exp(-beta * dE): continue 
        lattice[i, j, k], lattice[ni, nj, nk] = lattice[ni, nj, nk], lattice[i, j, k]
    #assert(np.sum(lattice == 1) == N1 and np.sum(lattice == 2) == N2)
    return energy(lattice, L)

lattice_cpy = deepcopy(lattice)

betas = [2.5, 4.]
lattices = [lattice, lattice_cpy]
nsweeps = 10000

configs = [[deepcopy(lattice)], [deepcopy(lattice_cpy)]]
for b, beta in enumerate(betas):
    es = []
    lattice = lattices[b]
    for _ in tqdm(range(nsweeps)):
        es.append(local_sweep(lattice, L, beta))
        configs[b].append(lattice)
    plt.plot(range(nsweeps), es, label=f"beta={beta:.2f})")
plt.legend()
plt.xlabel("MCS / [1]")
plt.ylabel("E / [1]")
plt.show()


def q(x, rho, rho1, rho2, L):
    N = int(rho * L ** 3)
    N1 = int(rho1 * rho * L ** 3)
    N2 = N - N1
    c0 = rho * (rho1 ** 2 + rho2 ** 2) 
    return ((1/N) * x - c0) / (1 - c0)

for b, beta in enumerate(betas):
    beg_lattice = configs[b][0]
    cs = [1.]
    for i in tqdm(range(1, nsweeps)):
        curr_config = configs[b][i]
        s = np.sum(np.logical_and(beg_lattice == curr_config, beg_lattice > 0))
        c = q(s, .75, .6, .4, L)
        cs.append(c)
    plt.plot(.1 + np.array(range(nsweeps)), cs, label=f"beta={beta:.2f})")

plt.legend()
plt.xscale("log")
plt.xlabel("MCS / [1]")
plt.ylabel("C_t / [1]")
plt.show()

for b, beta in enumerate(betas):
    with open(f"config-equilibrium-{beta:.2f}.txt", 'w') as f:
        f.write(' '.join(map(str, list(configs[b][-1].reshape(L*L*L)))))
