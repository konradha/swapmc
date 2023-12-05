import numpy as np
import matplotlib.pyplot as plt

d = []
for i in range(20):
    fname = f"energies{i+1}.txt"
    data  = np.loadtxt(fname, delimiter=',')
    #print(data.shape)
    xn, yn, ynp = data.T[0], data.T[1], data.T[2]

    d.append(ynp) 
    #plt.scatter(xn, yn, label="E")
    #if i == 1:
    #    plt.scatter(xn, ynp, label="E/particle")
    #else:
    #    plt.scatter(xn, ynp,)

d = np.array(d)
mean = np.array([np.mean(d[:, i]) for i in range(d.shape[1])])
stdd = np.std(d, axis=1)

for i in range(20):
    if i == 0: plt.scatter(xn, d[i],s=.1,
            label="energy per particle",marker='.',color='grey')
    else: plt.scatter(xn, d[i],s=.1,marker='.',color='grey')


plt.plot(xn, mean, label="mean", color="red", linewidth=3)
plt.legend()
plt.show()




#d = np.array(d)
#mean = 
#
#
#plt.yscale("log")
#plt.legend()
#plt.show()

