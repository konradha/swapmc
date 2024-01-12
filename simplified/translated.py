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
    conn = 3. if site_ty == 1 else 5.
    nn = nn_list[k + L * (j + i * L)]
    curr = 0.
    for n in nn:
        curr += grid[n]
    curr = curr - conn
    return curr * curr


def swap(grid, i, j, k, ii, jj, kk):
    tmp = grid[i, j, k]
    grid[i, j, k] = grid[ii, jj, kk]
    grid[ii, jj, kk] = tmp


def step(grid, i, j, k, beta, nn, nn_list):
    L = grid.shape[0]
    site_ty = grid[i, j, k]
    if site_ty == 0:
        return
    #nn_mv = np.random.randint(0, 6)
    nn_mv = rng.integers(low=0, high=5, size=1)[0]
    nn_i, nn_j, nn_k = nn[nn_mv]
    nn_ty = grid[nn_i, nn_j, nn_k]
    if nn_ty == site_ty:
        return
    E1 = local_energy(grid, i, j, k, nn_list)
    swap(grid, i, j, k, nn_i, nn_j, nn_k)
    E2 = local_energy(grid, i, j, k, nn_list)
    dE = np.abs(E1 - E2)
    #if np.random.random() < np.exp(-beta * dE):
    if rng.random() < np.exp(-beta * dE):
        return
    swap(grid, i, j, k, nn_i, nn_j, nn_k)


def sweep(grid, beta, nn):
    for offset in range(3):
        distribute_sweep(grid, offset, beta, nn)


def distribute_sweep(grid, offset, beta, nn):
    L = grid.shape[0]
    for ii in range((L - offset + 2) // 3):
        i = offset + ii * 3
        inner_sweep(grid, i, beta, nn)


def inner_sweep(grid, i, beta, nn):
    L = grid.shape[0]
    for j in range(L):
        for k in range(L):
            step(grid, i, j, k, beta, nn[k + L * (j + i * L)], nn)


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

    betas = [.2, 5.6, 1000.]
    r = int(rho1 * L ** 3)
    b = int(rho * L ** 3) - r
    lattice = fill_lattice(r, b, L)
    initial = deepcopy(lattice)

    datas = []; overlaps = []
    nsweeps = 1 << 5

    curr = 1
    for i, beta in tqdm(enumerate(betas)):
        data = [[0, 1.]]
        over = [[0, int(rho * L*L*L)]]
        for i in tqdm(range(nsweeps + 1)):
            sweep(lattice, beta, nn)
            #print(i,curr)
            #print(f"{i%curr=}")
            #print(f"{i//curr=}")
            #if i % curr == 0:
            #    data.append([curr, q(get_overlap(lattice, initial), L)])

            #if i // curr == 10:
            #    curr = curr * 10
            if i % curr == 0:
                data.append([curr, q(get_overlap(lattice, initial), L)])
                over.append([curr, get_overlap(lattice, initial)])
                curr = curr * 2
                

        curr = 1
        datas.append(data)
        overlaps.append(over)
        lattice = deepcopy(initial)

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
