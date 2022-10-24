import json
import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt
from sympy.utilities import lambdify

from curved import dotG, eperp, epll, magG

location = "/home/gsalinas/GitHub/angular/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransAngular as PyT

nF, nP = PyT.nF(), PyT.nP()
with open("./output/setup/params.json", "r") as file:
    params = json.loads(file.readline())
pval = np.array(list(params.values()))
back = np.load("./output/background/background.npy")
Ns, phis, phidots =  back.T[0], back.T[1:nF+1], back.T[nF+1:]


Hs = np.array([PyT.H(elem, pval) for elem in back[:, 1:]])
np.save("./output/background/Hs", Hs)
plt.plot(Ns, Hs, c="k", linewidth=2)
plt.title('Hubble parameter')
plt.xlabel(r'$N$', fontsize=16)
plt.ylabel(r'$H$', fontsize=16)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/background/hubble.png")
plt.clf()

dN = Ns[1] - Ns[0]
epsilon = -np.gradient(Hs, dN)/Hs
np.save("./output/background/epsilon", epsilon)
plt.plot(Ns, epsilon, c="k", linewidth=2)
plt.title('Epsilon parameter')
plt.xlabel(r'$N$', fontsize=16)
plt.ylabel(r'$\epsilon$', fontsize=16)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/background/epsilon.png")
plt.clf()

with open("./output/setup/G.txt", "rb") as file:
    G = pickle.load(file)
Gamma = PyTransSetup.fieldmetric(G, nF, nP)[1]

params_subs = {'p_'+str(ii): pval[ii] for ii in range(len(pval))}
Gparams = G.subs(params_subs)
Glbd = lambdify(['f_'+str(ii) for ii in range(nF)], Gparams)

Gmatrices = np.array([Glbd(phi[0], phi[1]) for phi in phis.T])
