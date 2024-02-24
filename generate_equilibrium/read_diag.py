import numpy as np
import matplotlib.pyplot as plt
from sys import argv

rng = np.random.default_rng()
L = 12
rho = .75
rho1 = .45
rho2 = .3

N = int(rho * L ** 3)
N1 = int(rho1 * L ** 3)
N2 = N - N1


fname = str(argv[1])
g = np.loadtxt(fname).reshape((L, L, L))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
grid = g
colors = np.empty(grid.shape + (4,))
colors[grid == 1] = [1, 0, 0, 1] # red
colors[grid == 2] = [0, 0, 1, 1] # blue
ax.voxels(grid, facecolors=colors, edgecolor='k')
plt.show()


for i in range(L):
    plt.imshow(g[i])
    plt.show()
