import numpy as np
import sys
import matplotlib.pyplot as plt

L = 12

fname = str(sys.argv[1])
grid = np.loadtxt(fname, delimiter=',')
grid = grid.reshape((L,L,L))
fix, axs = plt.subplots(L//2, 2)
for i, l in enumerate(range(L//2)):
    for j in range(2):
        axs[i][j].imshow(grid[i])

plt.show()
