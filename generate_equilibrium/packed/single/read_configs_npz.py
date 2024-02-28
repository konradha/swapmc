from sys import argv
import numpy as np

fname = str(argv[1])
data = np.load(fname)["arr_0"]
num_configs = data.shape[0] // 30 ** 3
data = data.reshape((num_configs, 30, 30, 30)).astype(np.int8)
print(data.shape)
