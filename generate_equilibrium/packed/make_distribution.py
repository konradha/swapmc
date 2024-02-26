import numpy as np
import matplotlib.pyplot as plt
from sys import argv
fname = str(argv[1])
data = np.loadtxt(fname)
u, c = np.unique(data, return_counts=True)
plt.bar(u, c/np.sum(c))
plt.show()

