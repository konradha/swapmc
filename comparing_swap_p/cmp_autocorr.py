import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft


from scipy import signal

def autocorrelate(time_series):
    n = len(time_series)
    time_series = time_series - np.mean(time_series)
    result = signal.correlate(time_series, time_series, mode='full')
    result = result[n-1:] / np.arange(n, 0, -1)
    return result / result[0]


def autocorr_func_1d(x,):
    def next_pow_two(n):
        i = 1
        while i < n:
            i = i << 1
        return i

    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n
    acf /= acf[0]
    return acf

def auto_corr_fast(M):   
    kappa = 100000
    M = M - np.mean(M)
    N = len(M)
    fvi = np.fft.fft(M, n=2*N)

    G = np.real( np.fft.ifft( fvi * np.conjugate(fvi) )[:N] )
    G /= N - np.arange(N); G /= G[0]
    G = G[:kappa]
    return G

#data_glass, data_liq = np.loadtxt("quickrun.txt", delimiter=',',), np.loadtxt("quickrun_faster.txt", delimiter=',')
#d_glass, d_liq = data_glass.T[2], data_liq.T[2]
#a_glass, a_liq = autocorrelate(d_glass), autocorrelate(d_liq) 
#
#
#plt.plot(d_glass, label="glassy?")
#plt.plot(d_liq, label="liquid?")
#
#plt.plot(a_glass, label="glassy?")
#plt.plot(a_liq, label="liquid?")
#
#plt.xscale("log")
#plt.yscale("log")
#plt.legend()
#plt.show()


#betas = ["1.", "2.5", "2.", "3.5", "3.", ".5"]
#betas += [".1", ".4", "2.1", "3.9", "3.1", "3.2"]
#betas += ["4.", "4.5", "5.", "5.5"]
#betas += ["9.", "10."]
#betas = sorted(betas)
betas = [".1", ".4", "1.", "3.",]# "9."]

fnames = [f"data_beta_{b}.txt" for b in betas]
fnames_noswap = [f"data_beta_{b}_noswap.txt" for b in betas]
data = [np.loadtxt(fname, delimiter=',') for fname in fnames]
data_noswap = [np.loadtxt(fname, delimiter=',') for fname in fnames_noswap] 

fig, ax = plt.subplots(3)
for i, b in enumerate(betas):
    ax[0].plot(data[i].T[2], label=f"{b=}")
    ax[0].plot(data_noswap[i].T[2], label=f"{b=}")
    #autocorr = autocorr_gw2010(data[i].T[1])
    autocorr_diff = autocorr_func_1d(data[i].T[1])
    autocorr = auto_corr_fast(data[i].T[1])

    autocorr_diff_noswap = autocorr_func_1d(data_noswap[i].T[1])
    autocorr_noswap = auto_corr_fast(data_noswap[i].T[1])
   
    # CLAMPING to 0.
    l = len(autocorr) // 2
    mask = autocorr < 0.
    #autocorr[mask] = 0.
    mask_diff = autocorr_diff < 0.
    #autocorr_diff[mask_diff] = 0.
    ax[1].plot(autocorr[:l], label=f"{b=}") 
    ax[2].plot(autocorr_diff[:l], label=f"{b=}")

    mask = autocorr_noswap < 0.
    mask_diff = autocorr_diff_noswap < 0.
    #autocorr_noswap[mask] = 0.
    #autocorr_diff_noswap[mask_diff] = 0.
    ax[1].plot(autocorr_noswap[:l], label=f"{b=} no swap", linestyle='-.')
    ax[2].plot(autocorr_diff_noswap[:l], label=f"{b=} no swap", linestyle='-.')



ax[0].legend()
ax[0].set_xscale("log")
ax[1].legend()
ax[1].set_xscale("log")
ax[2].set_xscale("log")
plt.show()
