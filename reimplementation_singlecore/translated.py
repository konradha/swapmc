import numpy as np
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

rng = np.random.default_rng()

def get_nn(i, j, k, L):
    l = i, j, (k - 1) % L
    r = i, j, (k + 1) % L

    u = i, (j - 1) % L, k
    d = i, (j + 1) % L, k

    f = (i - 1) % L, j, k
    b = (i + 1) % L, j, k
    return l, r, u, d, f, b


def build_nn(L):
    nearest_neighbors = L * L * L * [0]
    for i in range(L):
        for j in range(L):
            for k in range(L):
                nearest_neighbors[k + L * (j + i * L)] = get_nn(i, j, k, L)
    return nearest_neighbors


def fill_lattice(num_red, num_blue, N):
    grid = np.zeros((N, N, N), dtype=np.int8)
    numbers = np.array([1] * num_red + [2] * num_blue, dtype=np.int8)
    indices = np.random.choice(
        np.prod(
            grid.shape),
        num_red +
        num_blue,
        replace=False)
    x, y, z = np.unravel_index(indices, grid.shape)
    grid[x, y, z] = numbers
    return grid


def local_energy(grid, i, j, k, nn_list):
    L = grid.shape[0]
    site_ty = grid[i, j, k]
    if site_ty == 0: 0
    conn = 3. if site_ty == 1 else 5.
    nn = nn_list[k + L * (j + i * L)]
    curr = np.sum(np.array([grid[nn[0]], grid[nn[1]], grid[nn[2]], grid[nn[3]], grid[nn[4]], grid[nn[5]]])) 
    curr = curr - conn
    return curr * curr

def get_nn_energy(grid, i, j, k, nn_list):
    L = grid.shape[0]
    site_ty = grid[i, j, k]
    if site_ty == 0: 0
    curr = local_energy(grid, i, j, k, nn_list)
    nn = nn_list[k + L * (j + i * L)] 
    for ii, jj, kk in nn:
        curr += local_energy(grid, ii, jj, kk, nn_list)
    return curr


def swap(grid, i, j, k, ii, jj, kk):
    tmp = grid[i, j, k]
    grid[i, j, k] = grid[ii, jj, kk]
    grid[ii, jj, kk] = tmp

def step(grid, i, j, k, beta, nn, nn_list, distances = None, energies = None):
    distances.append(np.nan)
    L = grid.shape[0]
    site_ty = grid[i, j, k]
    if site_ty == 0:
        return
    nn_mv = rng.integers(low=0, high=5, size=1)[0]
    nn_i, nn_j, nn_k = nn[nn_mv]
    nn_ty = grid[nn_i, nn_j, nn_k]
    if nn_ty == site_ty:
        return
    dist = (nn_k + L * (nn_j + L * nn_i)) - (k + L * ( j + L * i))
    distances[-1] = dist

    E1 = get_nn_energy(grid, i, j, k, nn_list) + get_nn_energy(grid, nn_i, nn_j, nn_k, nn_list)
    swap(grid, i, j, k, nn_i, nn_j, nn_k)
    E2 = get_nn_energy(grid, i, j, k, nn_list) + get_nn_energy(grid, nn_i, nn_j, nn_k, nn_list)
    dE = E2 - E1 #abs(E2 - E1) # need absolute value here?
    energies.append(E2 - E1) 
    if rng.random() < np.exp(-beta * dE):
        return
    swap(grid, i, j, k, nn_i, nn_j, nn_k)


def sweep(grid, beta, nn, d = None, e = None):
    
    for offset in range(3):
        distribute_sweep(grid, offset, beta, nn, d, e)


def distribute_sweep(grid, offset, beta, nn, d = None, e = None):
    L = grid.shape[0]
    for ii in range((L - offset + 2) // 3):
        i = offset + ii * 3
        indices_j = np.random.choice(L, L)
        indices_k = np.random.choice(L, L)
        inner_sweep(grid, i, beta, nn, indices_j, indices_k, d, e)


def inner_sweep(grid, i, beta, nn, indices_j, indices_k, d = None, e = None):
    distances = []
    L = grid.shape[0]
    for j in indices_j:
        for k in indices_k:
            step(grid, i, j, k, beta, nn[k + L * (j + i * L)], nn, distances, e)
    d.append(distances)
    


def get_overlap(lattice1, lattice2):
    L = lattice1.shape[0]
    assert L == lattice.shape[0]
    l1 = lattice1 == lattice2
    l2 = lattice1 > 0
    return np.sum(np.logical_and(l1, l2))


def q(x, L):
    rho = .75
    rho1 = .6
    rho2 = .4

    N = rho * L**3
    phi = rho
    q0 = (rho1 ** 2 + rho2 ** 2) * rho
    return (1 / (1 - q0)) * (1 / N * x - q0)


if __name__ == '__main__':
    rho = .75
    rho1 = .6 * rho
    rho2 = .4 * rho
    L = int(sys.argv[1])
    nn = build_nn(L)

    betas = [.5, 2.9, 5.6]
    r = int(rho1 * L ** 3)
    b = int(rho * L ** 3) - r
    lattice = fill_lattice(r, b, L)

    
    reds = np.sum(lattice == 1)
    blues = np.sum(lattice == 2)
    print("red: ", r, reds)
    print("blu: ", b, blues)


    initial = deepcopy(lattice)

    datas = []; overlaps = []
    nsweeps = 1 << 8

    curr = 1
    distances = []
    energies = []
    for i, beta in tqdm(enumerate(betas)):
        data = [[0, 1.]]
        over = [[0, int(rho * L*L*L)]]
        for i in tqdm(range(nsweeps + 1)):
            sweep(lattice, beta, nn, distances, energies)
            if i % curr == 0:
                data.append([curr, q(get_overlap(lattice, initial), L)])
                over.append([curr, get_overlap(lattice, initial)])
                curr = curr * 2
                

        curr = 1
        datas.append(data)
        overlaps.append(over)
        lattice = deepcopy(initial)

        

        energies = np.array(energies)
        #print(energies)
        neg = energies <= 0
        pos = energies > 0
        un, cn = np.unique(energies[neg], return_counts = True) 
        up, cp = np.unique(energies[pos], return_counts = True)
        plt.bar(un-.1, cn, color='red')
        plt.bar(up+.1, cp, color='blue')
        plt.title("dE")
        plt.show()
        plt.clf()
        

        u, c = np.unique(energies, return_counts = True)
        #print(c)
        plt.bar(u, c, width=20, label=f"{beta=}")
        plt.legend()
        plt.title("dE, no 0 comparison")
        plt.show()
        energies = []



        distances = np.array(distances)
        u, c = np.unique(distances, return_counts = True)
        #print(c)
        plt.bar(u, c, width=20, label=f"{beta=}")
        plt.legend()
        plt.title("distances of moves")
        plt.show()

        distances = []




    datas = np.array(datas)
    overs = np.array(overlaps)
    for i, d in enumerate(datas):
        xn = d.T[0]
        data = d.T[1]
        
        plt.plot(
            xn + .1,
            data,
            label=f"{betas[i]:.2f}",
            color=plt.cm.RdYlBu(
                i / len(data)))
    plt.legend()
    plt.xscale("log")
    plt.show()


    for i, d in enumerate(overs):
        xn = d.T[0]
        data = d.T[1]
        
        plt.plot(
            xn + .1,
            data,
            label=f"{betas[i]:.2f}",
            color=plt.cm.RdYlBu(
                i / len(data)))

    plt.legend()
    plt.xscale("log")
    plt.show()
