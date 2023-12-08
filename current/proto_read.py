import numpy as np
from copy import deepcopy
import sys

def reverse_idx(idx, N=30):
    k = from_mc % N
    j = ((from_mc - k) // N) % N
    i = (from_mc - k - j*N) // N ** 2
    return i, j, k
      
def update_grid(grid, from_mc, to_mc, from_swap, to_swap, N):
    # NOT the swap MC move, but swapping states in the grid though
    def swap(arr, from_idx, to_idx): 
        i_f, j_f, k_f = reverse_idx(from_idx) 
        i_t, j_t, k_t = reverse_idx(to_idx)
        arr[i_f, j_f, k_f], arr[i_t, j_t, k_t] = arr[i_t, j_t, k_t], arr[i_f, j_f, k_f]

    if from_mc != -1:
        swap(grid, from_mc, to_mc)
       
    if from_swap != -1:
        swap(grid, from_swap, to_swap)
       
        

def calculate_energies(grid, N=30, pref=[0,3,5]):
    # in parallel (numba.jit?)
    pass

def create_front(grid, window, N=30, pref=[0,3,5]):
    pass

def measure(grid_back, grid_front, window, N=30, pref=[0,3,5]):
    # autocorrelations
    pass

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
    #print(grid[0])

    data = np.loadtxt(fname, delimiter=',', skiprows=1)
    from_mc_arr   = data.T[3].astype(np.int8)
    to_mc_arr     = data.T[4].astype(np.int8)
    from_swap_arr = data.T[5].astype(np.int8)
    to_swap_arr   = data.T[6].astype(np.int8)
    starting_grid = deepcopy(grid) 

    for step in range(len(from_mc_arr)):
        from_mc, to_mc = from_mc_arr[step], to_mc_arr[step] 
        from_swap, to_swap = from_swap_arr[step], to_swap_arr[step]
        update_grid(grid, from_mc, to_mc, from_swap, to_swap, N)
        
    print(grid[0] - starting_grid[0]) 
