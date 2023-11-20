import graphblas as g
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy
import networkx as nx


def rect_lattice(m: int, n: int) -> g.Matrix:
    G = g.Matrix(g.dtypes.BOOL, m*n, m*n)
    for i in range(m):
        for j in range(n):
            if j < n-1:
                G[i*n+j, i*n+j+1] = 1
                G[i*n+j+1, i*n+j] = 1
            if i < m-1:
                G[i*n+j, (i+1)*n+j] = 1
                G[(i+1)*n+j, i*n+j] = 1
    return G


def hex_lattice(m: int, n: int) -> g.Matrix:
    G = g.Matrix(g.dtypes.BOOL, m*n, m*n)
    for i in range(m):
        for j in range(n):
            if j > 0 and i < m-1:
                G[i*n+j, (i+1)*n+j-1] = 1
                G[(i+1)*n+j-1, i*n+j] = 1

            if j < n-1:
                G[i*n+j, i*n+j+1] = 1
                G[i*n+j+1, i*n+j] = 1

            if i < m-1:
                G[i*n+j, (i+1)*n+j] = 1
                G[(i+1)*n+j, i*n+j] = 1
    return G


def cubic_lattice(m: int, n: int, p: int) -> g.Matrix:
    G = g.Matrix(g.dtypes.BOOL, m*n*p, m*n*p)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                if k < p-1:
                    G[i*n*m+j*p+k, i*n*m+j*p+k+1] = 1
                    G[i*n*m+j*p+k+1, i*n*m+j*p+k] = 1
                if j < n-1:
                    G[i*n*m+j*p+k, i*n*m+(j+1)*p+k] = 1
                    G[i*n*m+(j+1)*p+k, i*n*m+j*p+k] = 1
                if i < m-1:
                    G[i*n*m+j*p+k, (i+1)*n*m+j*p+k] = 1
                    G[(i+1)*n*m+j*p+k, i*n*m+j*p+k] = 1
    return G


class Lattice:
    def __init__(self, beta: float, n: float, N: float, G: g.Matrix, preferred_connections: List = None):
        if preferred_connections is None:
            self.preferred_neighbors = [0, 3, 5]
        else:
            self.preferred_neighbors = preferred_connections

        self.beta = beta
        self.n = n
        self.n1 = N
        self.G = G

        self.N = G.ncols
        self.num_particles = int(self.N * self.n)

        self.num_p1 = int(self.n1 * self.N)
        self.num_p2 = self.num_particles - self.num_p1

        self.vacant_sites = np.ones(self.N, dtype=bool)
        self.first_sites  = np.logical_not(np.ones(self.N, dtype=bool))
        self.second_sites = np.logical_not(np.ones(self.N, dtype=bool))


        # set up more data
        self._fill_lattice(self.vacant_sites, self.first_sites, self.second_sites)

    def __local_energy(self, site: int):
        if not self.occupied[site]:
            return 0
        #print(f"{site=}")
        #print(f"{np.array(G[site, :].new().to_coo())=}")
        #print(f"{self.occupied=}")

        #print(f"{self.occupied.shape=}")
        #import pdb; pdb.set_trace()
        connection = self.preferred_neighbors[1] if self.first_sites[site] else self.preferred_neighbors[2]
        # self.occupied[np.array(G[site, :].new().to_coo()[0])] <- is this expression correct?
        curr_connections = np.sum(self.occupied[np.array(self.G[site, :].new().to_coo()[0])])
        #curr_connections = np.sum(np.array(np.logical_and(
        #    np.array(G[site, :].new().to_coo()[0]), self.occupied), dtype=int))  # mask `sites` array
        return (curr_connections - connection) ** 2

    def _local_energy(self, occupied: np.array, site: int):
        if not occupied[site]:
            return 0
        
        connection = self.preferred_neighbors[1] if self.first_sites[site] else self.preferred_neighbors[2]
        #curr_connections = np.sum(np.array(np.logical_and(
        #    np.array(G[site, :].new().to_coo()[0]), occupied), dtype=int))  # mask `sites` array
        curr_connections = np.sum(occupied[np.array(self.G[site, :].new().to_coo()[0])])

        return (curr_connections - connection) ** 2

    def _fill_lattice(self, vacant_sites: np.array, sites_p1: np.array, sites_p2: np.array):
        aux1 = 0
        aux2 = 0
        while aux1 < self.num_p1:
            site = np.random.choice(np.where(vacant_sites)[0])
            vacant_sites[site] = False
            sites_p1[site] = True
            aux1 += 1
        while aux2 < self.num_p2:
            site = np.random.choice(np.where(vacant_sites)[0])
            vacant_sites[site] = False
            sites_p2[site] = True
            aux2 += 1

        self.occupied = np.logical_or(self.first_sites, self.second_sites)
        self.sites_energies = np.zeros(self.N, dtype=int)
        
        for i in range(self.N):
            self.sites_energies[i] = self.__local_energy(i)

    def mc_step(self):

        # sites_1 = np.where(self.first_sites ) # get indices for type 1
        # sites_2 = np.where(self.second_sites) # get indices for type 2

        vac_cpy = deepcopy(self.vacant_sites)
        fst_cpy = deepcopy(self.first_sites)
        snd_cpy = deepcopy(self.second_sites)
        occ_cpy = deepcopy(self.occupied)
        eny_cpy = deepcopy(self.sites_energies)

        
        mv    = np.random.choice(np.where(self.occupied)[0])
        mv_to = np.random.choice(np.where(np.logical_not(self.occupied))[0]) 
        if fst_cpy[mv]:
            fst_cpy[mv] = False
            fst_cpy[mv_to] = True
        else:
            snd_cpy[mv] = False
            snd_cpy[mv_to] = True

        occ_cpy[mv] = False
        occ_cpy[mv_to] = True
        vac_cpy = np.logical_not(occ_cpy)

        eny_cpy[mv] = 0
        neighbors = np.where(
            np.array(self.G[mv, :].new().to_coo())[0])[0]  # get indices
        #print(f"neighbors from: {(np.array(self.G[mv, :].new().to_coo()))=}")
        new_nn_e1 = 0
        for n in neighbors:
            #print(f"{neighbors=}")
            e = self._local_energy(occ_cpy, n)
            eny_cpy[n] = e
            new_nn_e1 += e

        new_neighbors = np.where(
            np.array(self.G[mv_to, :].new().to_coo()[0]))[0]  # get indices
        new_nn_e2 = 0
        for n in new_neighbors:
            e = self._local_energy(occ_cpy, n)
            eny_cpy[n] = e
            new_nn_e2 += e

        # neighboring energies
        E1 = self.sites_energies[mv] + \
            np.sum(
                self.occupied[np.array(self.G[mv, :].new().to_coo()[0])])
        E2 = eny_cpy[mv_to] + new_nn_e1 + new_nn_e2
        dE = E2 - E1

        if np.random.random() < np.exp(- self.beta * dE):
            self.vacant_sites = vac_cpy
            self.first_sites = fst_cpy
            self.second_sites = snd_cpy
            self.occupied = occ_cpy
            self.sites_energies = eny_cpy

    def swap_step(self):
        s1 = np.random.choice(np.where(self.first_sites)[0])
        s2 = np.random.choice(np.where(self.second_sites)[0])

        vac_cpy = deepcopy(self.vacant_sites)
        fst_cpy = deepcopy(self.first_sites)
        snd_cpy = deepcopy(self.second_sites)
        occ_cpy = deepcopy(self.occupied)
        eny_cpy = deepcopy(self.sites_energies)

        fst_cpy[s1] = False
        fst_cpy[s2] = True
        snd_cpy[s1] = True
        snd_cpy[s2] = False
        occ_cpy = np.logical_or(fst_cpy, snd_cpy)  # not strictly necessary

        eny_cpy[s1] = self._local_energy(occ_cpy, s1)
        eny_cpy[s2] = self._local_energy(occ_cpy, s2)

        neighbors = np.array(self.G[s1, :].new().to_coo()[0])  # get indices
        #import pdb; pdb.set_trace()
        new_nn_e1 = 0
        for n in neighbors:
            e = self._local_energy(occ_cpy, n)
            eny_cpy[n] = e
            new_nn_e1 += e

        new_neighbors = np.array(self.G[s2, :].new().to_coo()[0])  # get indices
        new_nn_e2 = 0
        for n in new_neighbors:
            e = self._local_energy(occ_cpy, n)
            eny_cpy[n] = e
            new_nn_e2 += e

        E1 = self.sites_energies[s1] + \
            np.sum(
                self.occupied[np.array(self.G[s2, :].new().to_coo()[0])])
        E2 = new_nn_e1 + new_nn_e2 + eny_cpy[s1] + eny_cpy[s2]
        dE = E2 - E1
        if np.random.random() < np.exp(- self.beta * dE):
            self.vacant_sites = vac_cpy
            self.first_sites = fst_cpy
            self.second_sites = snd_cpy
            self.occupied = occ_cpy
            self.sites_energies = eny_cpy

    def plot_config(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        g.viz.draw(G)

        node_colors = []
        for i in range(self.N):
            if self.first_sites[i]: node_colors.append(100)
            elif self.second_sites[i]: node_colors.append(300) 
            else: node_colors.append(0)

        #pos = dict(zip(self.G.cols(), np.array(list(self.G.nodes()))))

        #nx.draw(self.G, pos, node_color=node_colors,
        #cmap=custom_cmap, node_size=75, vmin=0, vmax=2, ax=ax)

        nx.draw(self.G,  node_color=node_colors,
        node_size=75, vmin=0, vmax=2, ax=ax)

        p1_patch = mpatches.Patch(color=c1, label=f"{self.pref_conn[0]}")
        p2_patch = mpatches.Patch(color=c2, label=f"{self.pref_conn[1]}")


        ax.legend(handles=[p1_patch, p2_patch], title="preferred valence",
                  loc='upper center', bbox_to_anchor=(0.5, -0.02),
                  fancybox=True, shadow=True, ncol=len(self.pref_conn))

        return ax


def run_with_profiler():
    beta = 2.0
    n = 0.45
    n_1 = 0.8 * n
    l = 5

    import cProfile
    import pstats
    import re
    import pandas as pd


    G = cubic_lattice(l*l, l*l, l*l)

    L = Lattice(beta, n, n_1, G, [0,0,1])

    nsteps = 10

    

    profiler = cProfile.Profile()
    profiler.enable()

    energies = list()
    for _ in range(nsteps):
        L.mc_step()
        if np.random.rand() <= .3: L.swap_step()
        energies.append(
                    np.sum(L.sites_energies)
                    )

    #plt.plot(range(nsteps), energies)
    #plt.show()

    profiler.disable()
    #profiler.print_stats(sort='time')

    

    stats = pstats.Stats(profiler)
    stats.sort_stats('time')
    
    local = ['_local_energy', '__local_energy', 'swap_step', 'mc_step']    
    names = []
    times = []
    loc   = []
    for item in stats.stats.items():
        if item[0][2] in local: loc.append(True) # skip self-defined functions
        else: loc.append(False)
        names.append(item[0][2]) 
        times.append(item[1][3])

    colors = {True: 'green', False: 'blue'}
    timing_df = pd.DataFrame({"fn_name": names, "rt": times, "usr": loc})
   
    plt.barh(timing_df['fn_name'], timing_df['rt'], color=timing_df['usr'].map(colors))
    plt.xlabel('Running Time [s]')
    plt.show()



def run():
    beta = 2.0
    n = 0.45
    n_1 = 0.8 * n
    l = 5


    G = cubic_lattice(l*l, l*l, l*l)

    L = Lattice(beta, n, n_1, G, [0,0,1])

    nsteps = 100


    energies = list()
    for _ in range(nsteps):
        L.mc_step()
        if np.random.rand() <= .3: L.swap_step()
        energies.append(
                    np.sum(L.sites_energies)
                    )


if __name__ == '__main__':
    run()






# print(G)
#g.viz.draw(G)
#plt.show()
