import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()
L = 12
rho = .75
rho1 = .45
rho2 = .3

N = int(rho * L ** 3)
N1 = int(rho1 * L ** 3)
N2 = N - N1

g = np.zeros((L, L, L)).astype(int)
nr, nb = 0, 0
while nr < N1:
    i, j, k = rng.integers(0, L, 3) % L
    if (i + j + k) % 2 == 0 and g[i, j, k] == 0:
        g[i, j, k] = 1
        nr += 1

while nb < N2:
    i, j, k = rng.integers(0, L, 3) % L
    if (i + j + k) % 2 == 1 and g[i, j, k] == 0:
        g[i, j, k] = 2
        nb += 1

assert np.sum(g == 1) == N1
assert np.sum(g == 2) == N2


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
