import numpy as np
import binascii
import sys

fname = str(sys.argv[1])

# our simulation configuration
N = 30
n3 = N**3
packed_array = np.fromfile(fname, dtype=np.uint8)
unpacked_arr = np.unpackbits(packed_array)
boolean_arr  = unpacked_arr[:np.prod(n3)].reshape(N, N, N).astype(bool)
print(boolean_arr.shape)
