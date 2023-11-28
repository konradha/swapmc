import numpy as np
import matplotlib.pyplot as plt

fname = "energies.txt"
data  = np.loadtxt(fname, delimiter=',')
xn, yn = data.T[0], data.T[1]
plt.scatter(xn, yn)
plt.show()

