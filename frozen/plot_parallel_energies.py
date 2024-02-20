import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib import cm

fname = str(sys.argv[1])
num_ranks = int(sys.argv[2])

data = np.loadtxt(fname, delimiter=',')
epochs = data.T[0]
data = data[:, 1:num_ranks+1]

betas = []
beta_min = .1
beta_max = 4.
t = beta_min
while t <= beta_max:
    betas.append(t)
    t *= 1.65

cmap = cm.PuBu_r
for i in range(num_ranks):
    plt.plot(epochs, data.T[i], label=f"beta={(betas[i]):.2f}", color=plt.cm.RdYlBu(i/num_ranks))

#start = 1000
#window = 1000
#t = start
#while t <= max(epochs):
#    plt.vlines(t, ymin=790, ymax=2300)
#    t += window
    
plt.xscale("log")
plt.legend()
plt.show()
