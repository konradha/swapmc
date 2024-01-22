import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    grid = torch.zeros((N, N, N), dtype=torch.int8) 
    numbers = torch.tensor([1] * num_red + [2] * num_blue, dtype=torch.int8)
    indices = torch.randperm(grid.numel())[:num_red + num_blue]
    x = indices % N
    y = (indices / N).long() % N 
    z = (indices / (N*N)).long()
    grid[x.long(), y, z] = numbers
    return grid

def local_energy(grid, i, j, k, nn_list):
    L = grid.shape[0]
    site_ty = grid[i, j, k]
    if site_ty == 0:
        return 0
    conn = 3. if site_ty == 1 else 5.
    nn = nn_list[k + L * (j + i * L)]
    curr = grid[nn[0]] + grid[nn[1]] + grid[nn[2]] + grid[nn[3]] + grid[nn[4]] + grid[nn[5]]
    curr = curr - conn 
    return curr * curr
    
def get_nn_energy(grid, i, j, k, nn_list):
    L = grid.shape[0]
    site_ty = grid[i, j, k]
    if site_ty == 0: 
        return 0
    curr = local_energy(grid, i, j, k, nn_list)
    nn = nn_list[k + L * (j + i * L)]
    for ii, jj, kk in nn:
        curr += local_energy(grid, ii, jj, kk, nn_list)
    return curr


def swap(grid, i, j, k, ii, jj, kk):
    grid[i, j, k], grid[ii, jj, kk] = grid[ii, jj, kk], grid[i, j, k]
    

def step(grid, i, j, k, beta, nn, nn_list, distances = None, energies = None):
    distances.append(torch.nan)
    L = grid.shape[0]
    site_ty = grid[i, j, k]
    if site_ty == 0:
        return
    nn_mv = torch.randint(0,5,(1,))
    nn_i, nn_j, nn_k = nn[nn_mv]  
    nn_ty = grid[nn_i, nn_j, nn_k]
    if nn_ty == site_ty:
        return
        
    dist = (nn_k + L * (nn_j + L * nn_i)) - (k + L * ( j + L * i)) 
    distances[-1] = dist

    E1 = get_nn_energy(grid, i, j, k, nn_list) + get_nn_energy(grid, nn_i, nn_j, nn_k, nn_list) 
    swap(grid, i, j, k, nn_i, nn_j, nn_k)
    E2 = get_nn_energy(grid, i, j, k, nn_list) + get_nn_energy(grid, nn_i, nn_j, nn_k, nn_list)
    dE = (E2 - E1)  
    energies.append(E2 - E1)
    if torch.rand(1) < torch.exp(-beta * dE):
        return
    swap(grid, i, j, k, nn_i, nn_j, nn_k)


def sweep(grid, beta, nn, d = None, e = None): 
    L = grid.shape[0]
    for i in range(L):  
        indices_j = torch.randperm(L) 
        indices_k = torch.randperm(L)
        inner_sweep(grid, i, beta, nn, indices_j, indices_k, d, e)

def inner_sweep(grid, i, beta, nn, indices_j, indices_k, d = None, e = None):
    distances = []
    L = grid.shape[0]
    for j in indices_j:
        for k in indices_k:
            step(grid, i, j, k, beta, nn[k + L * (j + i * L)], nn, distances, e)
    if d is not None:
        d.append(distances) 

def get_overlap(lattice1, lattice2):
    L = lattice1.shape[0]
    assert L == lattice2.shape[0]
    l1 = lattice1 == lattice2
    l2 = lattice1 > 0
    return torch.sum(l1 & l2)

def q(x, L):
    rho = .75
    rho1 = .6
    rho2 = .4

    N = rho * L**3
    phi = rho
    q0 = (rho1 ** 2 + rho2 ** 2) * rho
    return (1 / (1 - q0)) * (1 / N * x - q0)



def main():
    rho = .75
    rho1 = .6 * rho
    rho2 = .4 * rho
    L = int(sys.argv[1])
    beta = float(sys.argv[2])
    nn = build_nn(L)

    betas = [.5, 2.9, 5.6]
    r = int(rho1 * L ** 3)
    b = int(rho * L ** 3) - r
    lattice = fill_lattice(r, b, L)


    reds = torch.sum(lattice == 1).item()
    blues = torch.sum(lattice == 2).item()
    
    print("red: ", r, reds)
    print("blu: ", b, blues)


    initial = deepcopy(lattice)

    datas = []; overlaps = []
    nsweeps = 1 << 8
    
    data = []
    over = []
    distances = []
    energies = []
    curr = 1
    data.append([0, 1.])
    over.append([0, reds + blues])
    for i in tqdm(range(nsweeps + 1)):
        sweep(lattice, beta, nn, distances, energies)
        if i % curr == 0:
            data.append([curr, q(get_overlap(lattice, initial), L)])            
            over.append([curr, get_overlap(lattice, initial)])
            curr = curr * 2

    u, c = np.unique(energies, return_counts=True)
    plt.bar(u, c)
    plt.show()
    plt.clf()

    reds = torch.sum(lattice == 1).item()
    blues = torch.sum(lattice == 2).item()
    
    print("red: ", r, reds)
    print("blu: ", b, blues)



    o = np.array(over)
    plt.plot(o.T[0]+.01, o.T[1])
    plt.xscale("log")
    plt.show()

    

if __name__ == '__main__':
    main()

