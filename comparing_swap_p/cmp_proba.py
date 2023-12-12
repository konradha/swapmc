import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


betas  = ["1.", "2.", "3.", "4."]
probas = [".1", ".2", ".3", ".4", ".5"]
fnames = [f"swap_proba_cmp_{b}_{p}.txt" for b in betas for p in probas]


#for f in fnames:
#    data = np.loadtxt(f, delimiter=',', skiprows=1)
#    print(f, data[-1])


probs = []
for b in tqdm(betas):
    di = []
    for p in probas:
        fname = f"swap_proba_cmp_{b}_{p}.txt" 
        data = np.loadtxt(fname, delimiter=',', skiprows=1)
        di.append(np.mean(data.T[2]))
    probs.append(di)

print(f"{len(probs)=}")
print(f"{len(betas)=}")
print(f"{len(probas)=}")

print("probs", probs)
print("betas", betas)
print("probas", probas)

for i, p in enumerate(probs):
    plt.plot(probas, p, label=f"{betas[i]}")

plt.legend()
plt.show()
