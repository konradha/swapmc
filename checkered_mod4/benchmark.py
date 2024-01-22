import subprocess
import timeit
import matplotlib.pyplot as plt
import numpy as np
import os

def run_process(cmd):
    args = cmd.split()
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL)
    p.wait()

def benchmark_runtimes(sliced, beta, Ls, ts, swap_enabled=0):
    runtimes = np.zeros((len(Ls), len(ts)))
    binary = None
    if sliced: binary = "checkerboard_sliced"
    else: binary = "checkerboard"
    for j, L in enumerate(Ls):
        for i, t in enumerate(ts):
            # set environment variable for best thread alloc
            affinity = "0,2,4,6,8,10,12,14"
            env = os.environ.copy()
            env["GOMP_CPU_AFFINITY"] = affinity
            cmd = f"./{binary} {beta} {t} {L} {swap_enabled}"
            def inner():
                run_process(cmd)
            timed = timeit.timeit(inner, number= 5)     
            runtimes[j][i] = 1000 * timed # get ms
    return runtimes

if __name__ == '__main__':
    beta = 5.
    Ls = [8, 12, 16, 20]
    ts = np.array([range(2, 9)], dtype=int).reshape(-1)
 
    num_colors = 7
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))


    runtimes_checkerboard = benchmark_runtimes(False, beta, Ls, ts, 0)
    for i, t in enumerate(ts):
        plt.plot(Ls, runtimes_checkerboard.T[i], color=colors[i], label=f"{t} threads")

    runtimes_sliced = benchmark_runtimes(True, beta, Ls, ts, 0)
    for i, t in enumerate(ts):
        plt.plot(Ls, runtimes_sliced.T[i], linestyle='-.', color=colors[i])


    plt.xticks(Ls)
    plt.xlabel("system size L / [1]")
    plt.ylabel("time / [ms]")
    plt.title(f"16385 sweeps, beta={beta}, checkerboard sweep, dashed line sliced checkerboard sweeps")
    plt.legend()
    plt.show()
