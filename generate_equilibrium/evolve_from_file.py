import numpy as np
from sys import argv
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
import numba

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
    arr = np.zeros((L, L, L, 6, 3), dtype=int)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                nn = get_nn(i, j, k, L)
                for n, nb in enumerate(nn):
                    arr[i, j, k, n, :] = nb
    d = {}
    for i in range(L):
        d[i] = {}
        for j in range(L):
            d[i][j] = {}
            for k in range(L):
                d[i][j][k] = get_nn(i, j, k, L)
    return d, arr

rng = np.random.default_rng()
L = 12
nn_list, nn_tensor = generate_nn_list(12)

rho = .75
rho1 = .45
rho2 = .3

N = int(rho * L ** 3)
N1 = int(rho1 * L ** 3)
N2 = N - N1

def local_e(lattice, i, j, k, L):
    nn = nn_list[i][j][k]  
    e = 0
    ty = lattice[i, j, k]
    if ty == 0: return 0
    if ty > 2: raise Exception("OH NO, bad particle type")
    pref = 3 if ty == 1 else 5 
    for (ni, nj, nk) in nn:
        e += int(lattice[ni, nj, nk] > 0)
    return (pref - e) ** 2

def nn_e(lattice, i, j, k, L):
    nn = nn_list[i][j][k]
    e = local_e(lattice, i, j, k, L) 
    for (ni, nj, nk) in nn:
        e += local_e(lattice, ni, nj, nk, L)
    return e 

def energy(lattice, L):
    e = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                e += local_e(lattice, i, j, k, L)
    return int(e)


def local_sweep(lattice, L, beta):
    locs = rng.integers(0, L, (L ** 3, 3)) % L
    nbs  = rng.integers(0, 6, (L ** 3, 1)) % L
    # TODO: not correct yet, may yield some speedup
    #mvs  = nn_tensor[locs[:]][nbs[:][0]] 
    rngu = rng.uniform(0, 1, L ** 3)
    for s in range(L ** 3):
        i, j, k = locs[s]
        ni, nj, nk = nn_list[i][j][k][nbs[s][0]]
        site_ty, mv_ty = lattice[i, j, k], lattice[ni, nj, nk]
        if site_ty == mv_ty: continue

        E1 = nn_e(lattice, i, j, k, L) + nn_e(lattice, ni, nj, nk, L) 
        lattice[i, j, k], lattice[ni, nj, nk] = lattice[ni, nj, nk], lattice[i, j, k] 
        E2 = nn_e(lattice, i, j, k, L) + nn_e(lattice, ni, nj, nk, L)
        dE = E2 - E1
        if dE <= 0 or rngu[s] < np.exp(-beta * dE): continue 
        lattice[i, j, k], lattice[ni, nj, nk] = lattice[ni, nj, nk], lattice[i, j, k]
    assert(np.sum(lattice == 1) == N1 and np.sum(lattice == 2) == N2)
    return energy(lattice, L)


fname = str(argv[1])
config = np.loadtxt(fname).reshape((L,L,L))
print(energy(config, L))
betas = np.geomspace(1., 2.2, 8)
nsweeps = 10

cpy_config = deepcopy(config)

for b, beta in enumerate(betas):
    print(f"running for {beta=:.2f}")
    conf = deepcopy(cpy_config)
    es = []
    for _ in tqdm(range(nsweeps)):
        e = local_sweep(conf, L, beta)
        es.append(e)
    plt.plot(
            es,
            label=f"{beta=:.2f}",
            color=plt.cm.RdYlBu(b/(len(betas)))
            )
plt.legend()
plt.xlabel("MCS / [1]")
plt.ylabel("E / [1]")
plt.show()


nsweeps = 1000

for b, beta in enumerate(betas):
    print(f"running for {beta=:.2f}")
    conf = deepcopy(cpy_config)
    starting_conf = deepcopy(conf)
    corr = [N]
    for _ in tqdm(range(nsweeps)):
        local_sweep(conf, L, beta)
        similar = conf == starting_conf
        nonzero = starting_conf > 0
        s = np.sum(np.logical_and(similar, nonzero))
        corr.append(s)
    plt.plot(corr, label=f"{beta=:.2f}", color=plt.cm.RdYlBu(b/(len(betas))))

plt.legend()
plt.xlabel("MCS / [1]")
plt.ylabel("delta / [1]")
plt.show()

#def get_random_neighbor(i, j, k, L):
#    #nn = get_nn(i, j, k, L)
#    nn = nn_list[i][j][k]
#    mv = rng.integers(0, len(nn), 1)[0]
#    return nn[mv]
