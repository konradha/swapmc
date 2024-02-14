import numpy as np
import matplotlib.pyplot as plt
import sys

fname = str(sys.argv[1])
num_ranks = int(sys.argv[2])

data = np.loadtxt(fname, delimiter=',')
epochs = data.T[0]
data = data[:, 1:num_ranks+1]

beta_min = .1
beta_max = 4.
db = (beta_max - beta_min) / (num_ranks - 1)

for i in range(num_ranks):
    plt.plot(epochs, data.T[i], label=f"beta={(beta_min + i * db):.2f}")
plt.xscale("log")
plt.legend()
plt.show()
