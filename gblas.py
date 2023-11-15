import graphblas as g
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy


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

    def _local_energy(self, site: int):
        if not self.occupied[site]:
            return 0

        connection = self.preferred_neighbors[1] if self.first_sites[site] else self.preferred_neighbors[2]
        curr_connections = np.sum(np.array(np.logical_and(
            (G[site, :].new().to_coo()), self.occupied), dtype=int))  # mask `sites` array
        return (curr_connections - connection) ** 2

    def _local_energy(self, occupied: np.array, site: int):
        if not occupied[site]:
            return 0

        connection = self.preferred_neighbors[1] if self.first_sites[site] else self.preferred_neighbors[2]
        curr_connections = np.sum(np.array(np.logical_and(
            (G[site, :].new().to_coo()), occupied), dtype=int))  # mask `sites` array
        return (curr_connections - connection) ** 2

    def _fill_lattice(self, vacant_sites: np.array, sites_p1: np.array, sites_p2: np.array):
        aux1 = 0
        aux2 = 0
        print(f"{aux1=} and {self.num_p1=}")
        while aux1 < self.num_p1:
            print(np.where(vacant_sites)[0], "\n\n\n")
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
        self.sites_energies = np.array(self.N, dtype=int)

        for i in range(self.N):
            self.sites_energies[i] = self._local_energy(i)

    def mc_step(self):

        # sites_1 = np.where(self.first_sites ) # get indices for type 1
        # sites_2 = np.where(self.second_sites) # get indices for type 2

        vac_cpy = deepcopy(self.vacant_sites)
        fst_cpy = deepcopy(self.first_sites)
        snd_cpy = deepcopy(self.second_sites)
        occ_cpy = deepcopy(self.occupied)
        eny_cpy = deepcopy(self.energies)

        mv = np.random.choice(np.where(self.occupied))
        mv_to = np.random.choice(np.where(np.logical_not(self.occupied)))
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
            np.array(self.G[mv, :].new().to_coo()))  # get indices
        new_nn_e1 = 0
        for n in neighbors:
            e = self._local_energy(occ_cpy, n)
            eny_cpy[n] = e
            new_nn_e1 += e

        new_neighbors = np.where(
            np.array(self.G[mv_to, :].new().to_coo()))  # get indices
        for n in new_neighbors:
            e = self._local_energy(occ_cpy, n)
            eny_cpy[n] = e
            new_nn_e2 += e

        # neighboring energies
        E1 = self.energies[mv] + \
            np.sum(
                np.array(self.occupied[self.G[mv, :].new().to_coo()], dtype=int))
        E2 = eny_cpy[mv_to] + new_nn_e1 + new_nn_e2
        dE = E2 - E1

        if np.random.random() < np.exp(- self.beta * dE):
            self.vacant_sites = vac_cpy
            self.first_sites = fst_cpy
            self.second_sites = snd_cpy
            self.occupied = occ_cpy
            self.energies = eny_cpy

    def swap_step(self):
        s1 = np.random.choice(np.where(self.first_sites))
        s2 = np.random.choice(np.where(self.second_sites))

        vac_cpy = deepcopy(self.vacant_sites)
        fst_cpy = deepcopy(self.first_sites)
        snd_cpy = deepcopy(self.second_sites)
        occ_cpy = deepcopy(self.occupied)
        eny_cpy = deepcopy(self.energies)

        fst_cpy[s1] = False
        fst_cpy[s2] = True
        snd_cpy[s1] = True
        snd_cpy[s2] = False
        occ_cpy = np.logical_or(fst_cpy, snd_cpy)  # not strictly necessary

        eny_cpy[s1] = self._local_energy(occ_cpy, s1)
        eny_cpy[s2] = self._local_energy(occ_cpy, s2)

        neighbors = np.where(
            np.array(self.G[s1, :].new().to_coo()))  # get indices
        new_nn_e1 = 0
        for n in neighbors:
            e = self._local_energy(occ_cpy, n)
            eny_cpy[n] = e
            new_nn_e1 += e

        new_neighbors = np.where(
            np.array(self.G[s2, :].new().to_coo()))  # get indices
        for n in new_neighbors:
            e = self._local_energy(occ_cpy, n)
            eny_cpy[n] = e
            new_nn_e2 += e

        E1 = self.energies[s1] + \
            np.sum(
                np.array(self.occupied[self.G[mv, :].new().to_coo()], dtype=int))
        E2 = new_nn_e1 + new_nn_e2 + eny_cpy[s1] + eny_cpy[s2]

        if np.random.random() < np.exp(- self.beta * dE):
            self.vacant_sites = vac_cpy
            self.first_sites = fst_cpy
            self.second_sites = snd_cpy
            self.occupied = occ_cpy
            self.energies = eny_cpy

    def plot_config(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        g.viz.draw(G)

        node_colors = []
        for i in range(self.N):
            if self.first_sites[i]: node_colors.append(100)
            elif self.second_sites[i]: node_colors.append(300) 
            else: node_colors.append(0)

        pos = dict(zip(self.G.cols(), np.array(list(self.G.nodes()))))

        nx.draw(self.G, pos, node_color=node_colors,
        cmap=custom_cmap, node_size=75, vmin=0, vmax=2, ax=ax)

        p1_patch = mpatches.Patch(color=c1, label=f"{self.pref_conn[0]}")
        p2_patch = mpatches.Patch(color=c2, label=f"{self.pref_conn[1]}")


        ax.legend(handles=[p1_patch, p2_patch], title="preferred valence",
                  loc='upper center', bbox_to_anchor=(0.5, -0.02),
                  fancybox=True, shadow=True, ncol=len(self.pref_conn))

        return ax



G = rect_lattice(5, 5)

L = Lattice(.33, 5, 5, G)

# print(G)
#g.viz.draw(G)
#plt.show()
