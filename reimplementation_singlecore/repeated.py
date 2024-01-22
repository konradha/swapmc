import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys


def build_lattice(L, num_red, num_blue):
    grid = np.zeros(L**3, dtype=int)

    rng = np.random.default_rng()
    red_sites = rng.choice(L**3, size=num_red, replace=False)
    blue_sites = rng.choice(L**3, size=num_blue, replace=False)

    grid[red_sites] = 1
    grid[blue_sites] = 2
    for _ in range(10):
        np.random.shuffle(grid)

    return grid

def get_neighbors(i, j, k, N):
    iprev = (i-1) % N
    jprev = (j-1) % N
    kprev = (k-1) % N

    inext = (i+1) % N
    jnext = (j+1) % N
    knext = (k+1) % N

    f = k + N * (j + iprev * N)
    b = k + N * (j + inext * N)

    u = k + N * (jprev + i * N)
    d = k + N * (jnext + i * N)

    l = kprev + N * (j + i * N)
    r = knext + N * (j + i * N)

    #print(f,b,u,d,l,r)

    for i, n1 in enumerate([f, b, u, d, l, r]):
        for j, n2 in enumerate([f, b, u, d, l, r]): 
            if i != j:
                assert(n1 != n2)

    return np.array([u, d, l, r, f, b], dtype=np.int32)

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


def dE(lattice, site, mv, nn_list, kinetic=None):
    L = lattice.shape[0]
    E1 = nn_e(lattice, site, nn_list) + nn_e(lattice, mv, nn_list) 
    exchange(lattice, site, mv)
    E2 = nn_e(lattice, site, nn_list) + nn_e(lattice, mv, nn_list)
    exchange(lattice, site, mv)
    
    #kinetic = .6
    if kinetic:
        return kinetic + (E2 - E1) / 75 # |75| is maximum diff between energies
    else:
        return E2 - E1
    

def visualize_grid(L, grid, nns):
    if L < 5: raise Exception("Grid needs to have at least L=5 for 6-regularity to be workable")
    if L > 8: raise Exception("Grid too big to draw, choose different L") 

    g = nx.Graph() 
    for p in nns.keys():
        g.add_node(p, attribute=grid[p])

    for p, nn in nns.items(): 
        for n in nn:
            g.add_edge(p, n,)
    print("lattice is connected, not an issue with indices:", nx.is_connected(g))
    print("lattice is 6-regular:",nx.is_k_regular(g, k=6))
    nx.draw(g, with_labels=True)
    plt.show()

def compare_energies(L, grid, nns, beta):
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
    print("min/max of energy diff E2-E1:  ", np.min(u), np.max(u))

    mask = u <= 0
    print("immediate accepts", np.sum(c[mask]) / np.sum(c))# / L ** 3)
    plt.bar(u, c)
    plt.yscale("log")
    plt.show()

    plt.clf()

    dA = []
    for s, m in accept:
        dA.append((s-m))
    dA = np.array(dA)
    dR = []
    for s, m in reject:
        dR.append((s-m))
    dR = np.array(dR)
    ua, ca = np.unique(dA, return_counts=True)
    ur, cr = np.unique(dR, return_counts=True)

   

    plt.bar(ua-2, ca, width=5, color='red', label="accepted distances")
    plt.bar(ur+2, cr, width=5, color='blue', label="rejected distances")

    for d in [-1, 1, -(L-1), L-1, -(L**2-1), L**2-1]:
        plt.vlines(d, ymin=0, ymax=np.max(ca), linestyle='-.', alpha=.2, color="black")

      
    
    plt.title("investigating if bias exists in neighbor moves")

    plt.yscale("log")
    plt.legend()
    plt.show()

    plt.clf()

    plt.bar(np.linspace(0,len(ua),len(ua))-.1, ca, color='green')
    plt.bar(np.linspace(0,len(ur),len(ur))+.1, cr, color='red')
    plt.title("without actual differences; same plot as before: edge cases about 9 times less often (due to PBC)")

    plt.gca().set_xticks(np.linspace(0,len(ua),len(ua)))
    plt.gca().set_xticklabels(ua)
    plt.show()



def main():
    L = 12
    rho = 0.75
    rho1 = 0.6 * rho
    beta = 5.
    N = int(np.round(L**3 * rho)) 
    r = int(np.round(L**3 * rho1))
    b = N - r
    print(N, r, b)
    print(L, L**3)


    grid = build_lattice(L, r, b)
    nns  = get_nn_list(L)



    if len(sys.argv) > 1:
        visualize_grid(L, grid, nns)    
    else:
        compare_energies(L, grid, nns, beta)

    grid = grid.reshape((L, L, L))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = np.empty(grid.shape + (4,))
    
    colors[grid == 1] = [1, 0, 0, 1] # red
    colors[grid == 2] = [0, 0, 1, 1] # blue
    ax.voxels(grid, facecolors=colors, edgecolor='k')
    plt.show() 


if __name__ == '__main__': 
    main()
