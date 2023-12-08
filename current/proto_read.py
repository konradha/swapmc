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
    # TODO -- see neighbor calculation below
    """
    static inline void fill_neighbors(lattice_entry *e, int i, int j, int k,
                    int nx, int ny, int nz)
    {
        // wrap around, PBC
        if(i==0)    e->neighbors[0] = k + ny * (j + (nx-1) * nx);
        if(i==nx-1) e->neighbors[3] = k + ny * (j + 0 * nx);
        if(j==0)    e->neighbors[1] = k + ny * ((ny-1) + i * nx);
        if(j==ny-1) e->neighbors[4] = k + ny * (0 + i * nx);
        if(k==0)    e->neighbors[2] = (nz-1) + ny * (j + i * nx);
        if(k==nz-1) e->neighbors[5] = 0 + ny * (j + i * nx);
    
        // all internal neighbors
        if(i>0)     e->neighbors[0] = k + ny * (j + (i-1) * nx);
        if(j>0)     e->neighbors[1] = k + ny * (j-1 + i * nx);
        if(k>0)     e->neighbors[2] = k-1 + ny * (j + i * nx);
        if(i<nx-1)  e->neighbors[3] = k + ny * (j + (i+1) * nx);
        if(j<ny-1)  e->neighbors[4] = k + ny * (j+1 + i * nx);
        if(k<nz-1)  e->neighbors[5] = k+1 + ny * (j + i * nx);
    }
    """
    energies = np.zeros((N, N, N), dtype=float)
    return energies

def create_front(grid, window, N=30, pref=[0,3,5]):
    pass

def measure(grid_back, grid_front, window, N=30, pref=[0,3,5], rho=.75, rho1=.4, rho2=.35):
    n = rho * N**3
    C0 = rho * ( rho1 ** 2 + rho2 ** 2)
    v = np.sum(np.logical_and(grid_back, grid_front))
    return (v/n - C0) / (1. - C0)

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
    
    from_mc_arr   = data.T[3].astype(np.int8)
    to_mc_arr     = data.T[4].astype(np.int8)
    from_swap_arr = data.T[5].astype(np.int8)
    to_swap_arr   = data.T[6].astype(np.int8)

    windows = [10, 100, 1000, 10000, 100000]

    starting_grid = deepcopy(grid) 
    walking_grid = deepcopy(grid)


    anded = np.zeros_like(to_mc_arr, dtype=float)
    autos = np.zeros_like(to_mc_arr, dtype=float)
    cs = []
    for window in windows:    
        starting_grid = walking_grid
        for step in tqdm(range(len(from_mc_arr))):
            from_mc, to_mc = from_mc_arr[step], to_mc_arr[step] 
            from_swap, to_swap = from_swap_arr[step], to_swap_arr[step]
            update_grid(grid, from_mc, to_mc, from_swap, to_swap, N)
            energies = calculate_energies(grid)
            if step >= window:
                wstep = step - window 
                wfrom_mc, wto_mc = from_mc_arr[wstep], to_mc_arr[wstep] 
                wfrom_swap, wto_swap = from_swap_arr[wstep], to_swap_arr[wstep]
                update_grid(starting_grid, wfrom_mc, wto_mc, wfrom_swap, wto_swap, N)
                #anded[step] = np.sum(np.logical_and(starting_grid, grid)) / N**3 
                autos[step] = measure(starting_grid, grid, window,) 
        cs.append(np.mean(autos[window:]))
        #print(autos[window:]) 
        #plt.plot(anded, label=f"{window=}")
        #plt.plot(autos, label=f"{window=}")
        anded = np.zeros_like(to_mc_arr, dtype=float)
        autos = np.zeros_like(to_mc_arr, dtype=float)

    print(fname, cs)
    #plt.plot(windows, cs, marker='o')
    #plt.xscale("log")
    #plt.show()

    #plt.xscale("log")
    ##plt.yscale("log")
    #plt.legend()
    #plt.show()
            
