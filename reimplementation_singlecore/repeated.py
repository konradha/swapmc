import numpy as np
import matplotlib.pyplot as plt


def build_lattice(L, num_red, num_blue):
    grid = np.zeros(L**3, dtype=int)

    rng = np.random.default_rng()
    red_sites = rng.choice(L**3, size=num_red, replace=False)
    blue_sites = rng.choice(L**3, size=num_blue, replace=False)

    grid[red_sites] = 1
    grid[blue_sites] = 2

    return grid

def get_neighbors(i, j, k, N):
    ii = (i-1) % N
    jj = (j-1) % N
    kk = (k-1) % N

    f = k + N * (j + ii*N)
    b = k + N * (j + (i+1)%N*N)
    u = k + N * (jj + i*N)
    d = k + N * ((j+1)%N + i*N)
    l = kk + N*(j + i*N)
    r = (k+1)%N + N*(j + i*N)

    return np.array([u, d, l, r, f, b], dtype=int)

def get_nn_list(N):
    neighbor_map = {}
    for i in range(N):
        for j in range(N):
            for k in range(N):
                site = k + N * (j + i*N)
                neighbor_map[site] = get_neighbors(i, j, k, N)
    return neighbor_map

def local_e(lattice, pos, nn_list):
     
    if lattice[pos] == 0:
        return 0.

    count = np.sum(lattice[nn_list[pos]] > 0)
    if lattice[pos] == 1:
        pref = 3
    elif lattice[pos] == 2:
        pref = 5
    e = (count - pref)**2 
    return e

def nn_e(lattice, pos, nn_list):
    curr = 0.
    if lattice[pos] != 0: curr +=  local_e(lattice, pos, nn_list)

    nn = nn_list[pos]
    for n in nn:
        curr += local_e(lattice, n, nn_list)
    return curr

def exchange(lattice, site, mv):
    lattice[site], lattice[mv] = lattice[mv], lattice[site]


def dE(lattice, site, mv, nn_list):
    E1 = nn_e(lattice, site, nn_list) + nn_e(lattice, mv, nn_list) 
    exchange(lattice, site, mv)
    E2 = nn_e(lattice, site, nn_list) + nn_e(lattice, mv, nn_list)
    exchange(lattice, site, mv)
    return E2 - E1
    


def main():
    L = 12
    rho = 0.75
    rho1 = 0.6 * rho
    beta = 5.
    N = int(np.round(L**3 * rho)) 
    r = int(np.round(L**3 * rho1))
    b = N - r
    grid = build_lattice(L, r, b)
    nns  = get_nn_list(L)
    des = []
    accept = []
    reject = []
    for i in range(L):
        for j in range(L):
            for k in range(L):
                site = k + L * (j + i * L)
                if grid[site] == 0: continue # only move particles
                for n in nns[site]:
                    if grid[n] != 0: continue # only move to empty slots
                    de = dE(grid, site, n, nns)
                    if np.random.random() < np.exp( -beta * de):
                        accept.append((site, n))
                    else:
                        reject.append((site, n))

                    des.append(de)
    
    u, c = np.unique(des, return_counts=1)
    mask = u <= 0
    print("immediate accepts", np.sum(c[mask]) / np.sum(c))
    plt.bar(u, c / L ** 3)
    plt.yscale("log")
    plt.show()

    plt.clf()

    dA = []
    for s, m in accept:
        dA.append(abs(s-m))
    dA = np.array(dA)
    dR = []
    for s, m in reject:
        dR.append(abs(s-m))
    dR = np.array(dR)
    ua, ca = np.unique(dA, return_counts=True)
    ur, cr = np.unique(dR, return_counts=True)

    plt.bar(ua-5., ca, width=10., color='red', label="accepted distances")
    plt.bar(ur+5., cr, width=10., color='blue', label="rejected distances")

    plt.xticks(ur)
    plt.title("investigating correlation in neighbor moves")

    plt.yscale("log")
    plt.legend()
    plt.show()






    




if __name__ == '__main__': 
    main()
