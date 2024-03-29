import subprocess
import numpy as np
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_nn(i, j, k, L):
    iprev, inext = (i - 1) % L, (i + 1) % L
    jprev, jnext = (j - 1) % L, (j + 1) % L
    kprev, knext = (k - 1) % L, (k + 1) % L

    return [
            (iprev, j, k), (inext, j, k),
            (i, jprev, k), (i, jnext, k),
            (i, j, kprev), (i, j, knext),
           ]

def generate_nn_list(L):
    d = {}
    for i in range(L):
        d[i] = {}
        for j in range(L):
            d[i][j] = {}
            for k in range(L):
                d[i][j][k] = get_nn(i, j, k, L)
    return d



betas = np.geomspace(.5, 2.49, 1 * 8) # factors for hyperthreading
L = 30


def local_e(config, i, j, k, nn_list):
    ty = config[i, j, k]
    if ty == 0: return 0
    pref = 3 if ty == 1 else 5 
    nn = nn_list[i][j][k]
    nb = 0
    for ni, nj, nk in nn:
        if config[ni][nj][nk] > 0: nb += 1 
    return  (pref - nb)**2

def energy(config, nn_list):
    L = config.shape[0]
    e = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                e += local_e(config, i, j, k, nn_list)
    return e  

def energy2(config, nn_list):
    L = config.shape[0]
    e = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                e += local_e(config, i, j, k, nn_list) ** 2
    return e



def q(x):
    rho = .75
    rho1 = .6
    rho2 = .4
    c0 = rho * (rho1 ** 2 + rho2 ** 2)
    N = rho * L ** 3
    return ((1/N) * x - c0) / (1 - c0)



bin_name =  "to_sim"
def run_proc(beta: float):
    start = time.time()
    p = subprocess.Popen([f"./{bin_name}", f"{beta:.2f}"], stdout=subprocess.PIPE)
    out, err = p.communicate()
    p.wait()
    end = time.time()
    duration = end - start 
    return out, duration


with Pool() as p:
    args = [[betas[b]] for b in
            range(len(betas))] 
    ret = p.starmap(run_proc, args)
    


nn_list = generate_nn_list(L)
all_e = []
for thread_num, (configs, duration) in tqdm(enumerate(ret)):
    es = []
    data = np.fromstring(configs, dtype=np.int8, sep=' ') 
    np.savez_compressed(f"configs_{betas[thread_num]:.2f}_test.npz", data)
    num_configs = data.shape[0] // L ** 3 
    data = data.reshape((num_configs, L, L, L))  
    beg_conf = data[0]
    # ATTENTION
    range_test = np.array([0.1, *[2 ** i - 1 for i in range(1, 12 + 1)]])
    corr = []
    for r in range_test:
        r = int(r)
        n1 = np.sum(np.logical_and(beg_conf == 1, data[r] == 1))
        n2 = np.sum(np.logical_and(beg_conf == 2, data[r] == 2))
        corr.append(q(n1 + n2))
        es.append(energy(data[r], nn_list))
    all_e.append(es)
    plt.plot(range_test, corr, label=f"{1/betas[thread_num]:.2f}", color=plt.cm.RdYlBu(thread_num/(len(betas))))
plt.xscale("log")
plt.legend()
plt.xlabel("MCS / [1]")
plt.ylabel("C(t;t_w) / [1]")
plt.show()


for i, e in enumerate(all_e):
    plt.plot(range_test, e, label=f"{1/betas[i]:.2f}", color=plt.cm.RdYlBu(i/(len(betas))))
plt.xscale("log")
plt.legend()
plt.xlabel("MCS / [1]")
plt.ylabel("E / [1]")
plt.show()
