import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

import sys

fname = str(sys.argv[1])
df = pd.read_csv(fname)
cols = df.columns
data = df.iloc[:,-1].to_numpy()
data = data[0:len(data)-10]

N = len(data)
var = np.var(data)
m_t = fft(data)
G_t = ifft(data * data.conj())
G_t = G_t / var / 30 ** 3
mask = G_t < 1e-6
G_t[mask] = np.nan

plt.plot(G_t[0:1000], marker='x')
plt.yscale("log")
plt.show()
