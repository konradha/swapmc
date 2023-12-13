import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from copy import deepcopy
import sys

def reverse_idx(idx, N=30):
    k = idx % N
    j = ((idx - k) // N) % N
    i = (idx - k - j*N) // N ** 2
    return i, j, k
      
def update_grid(grid, from_mc, to_mc, from_swap, to_swap, N):
    def swap(arr, from_idx, to_idx): 
        i_f, j_f, k_f = reverse_idx(from_idx) 
        i_t, j_t, k_t = reverse_idx(to_idx)
        arr[i_f, j_f, k_f], arr[i_t, j_t, k_t] = arr[i_t, j_t, k_t], arr[i_f, j_f, k_f]

    if from_mc != -1:
        swap(grid, from_mc, to_mc)
       
    if from_swap != -1:
        swap(grid, from_swap, to_swap) 

def measure(grid_back, grid_front, N=30, pref=[0,3,5], rho=.75, rho1=.4, rho2=.35):
    n = rho * N**3
    C0 = rho * ( rho1 ** 2 + rho2 ** 2)
    v = np.sum(np.logical_and(grid_back, grid_front))
    return (v/n - C0) / (1. - C0)

def energy(grid, N, pref=[0, 3, 5],):
    e = np.zeros_like(grid)
    from itertools import product
    for i, j, k in product(range(N), range(N), range(N)):
        pi = grid[i, j, k] * np.ones(6).astype(int)
        neighbors = np.array([
            # this should be a parallel stencil ...
            grid[(i+1) % N, j, k], grid[(i-1) % N, j, k],
            grid[i, (j+1) % N, k], grid[i, (j-1) % N, k],
            grid[i, j, (k+1) % N], grid[i, j, (k-1) % N],

                ], dtype=int)
        count = 0.
        for n in neighbors:
            if n == 0: continue
            count += 1. 
        count = pref[pi[0]] - count 
        e[i, j, k] = count ** 2 
    return e


if __name__ == '__main__':
    fname = str(sys.argv[1])

    N = 30
    pref = [0, 3, 5] # neighbor preferences
    grid_raw = None 
    with open(fname, 'r') as f:
        grid_raw = np.fromstring(f.readline(), dtype=int, sep=' ',).astype(np.int8) 
    grid = grid_raw.reshape((N, N, N),) 
    n3 = N ** 3
    n2 = N ** 2

    
    
    data = np.loadtxt(fname, delimiter=',', skiprows=1)
    
    # e.g.: 51,16825,16824,-1,-1
    epochs   = data.T[0].astype(int)
    from_mcs = data.T[1].astype(int) 
    to_mcs   = data.T[2].astype(int)
    from_swp = data.T[3].astype(int) 
    to_swp   = data.T[4].astype(int)



    starting_grid = deepcopy(grid) 
    #walking_grid = deepcopy(grid)
    
    for window in tqdm([10 ** i for i in range(8)]):
        if window > len(epochs) // 2: break

        d = []
        grid = deepcopy(starting_grid)
        walking_grid = deepcopy(starting_grid)

        for i, e in enumerate(epochs):
            from_mc, to_mc = from_mcs[i], to_mcs[i]
            from_sw, to_sw = from_swp[i], to_swp[i]
            update_grid(grid, from_mc, to_mc, from_sw, to_sw, N) 
            if i > window:
                from_mc_l, to_mc_l = from_mcs[i-window], to_mcs[i-window]  
                from_sw_l, to_sw_l = from_swp[i-window], to_swp[i-window]
                update_grid(walking_grid, from_mc_l, to_mc_l, from_sw_l, to_sw_l, N)
                d.append(measure(walking_grid, grid))
            else:
                d.append(measure(starting_grid, grid,))
        d = np.array(d)
        save_f = f"{fname}_corr_{window}.txt"
        np.savetxt(save_f, d, delimiter=',')

        
