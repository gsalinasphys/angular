import json
import pickle
import sys
from itertools import combinations_with_replacement, product

import numpy as np
from matplotlib import pyplot as plt
from sympy.utilities import lambdify

from curved import dotG, eperp2d, epll, magG

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

phiprimes = phidots / Hs
np.save("./output/background/phiprimes", phiprimes)
phidoubleprimes = np.gradient(phiprimes, dN, axis=1)
np.save("./output/background/phidoubleprimes", phidoubleprimes)

with open("./output/setup/G.txt", "rb") as file:
    G = pickle.load(file)
Gamma = PyTransSetup.fieldmetric(G, nF, nP)[1]

params_subs = {'p_'+str(ii): pval[ii] for ii in range(len(pval))}
Glbd = lambdify(['f_'+str(ii) for ii in range(nF)], G.subs(params_subs))

Gammalbd = np.empty((nF, nF, nF)).tolist()
for aa, (bb, cc) in product(range(1, nF+1), combinations_with_replacement(range(1, nF+1), 2)):
    Gammalbd[aa-1][bb-1][cc-1] = lambdify(['f_'+str(ii) for ii in range(nF)], Gamma(-aa, bb, cc).subs(params_subs))

Gmatrices = np.array([Glbd(phi[0], phi[1]) for phi in phis.T])
Gammamatrices = np.empty((len(Ns), nF, nF, nF))
for ii in range(len(Ns)):
    for aa, (bb, cc) in product(range(nF), combinations_with_replacement(range(nF), 2)):
        Gammamatrices[ii, aa, bb, cc] = Gammalbd[aa][bb][cc](phis.T[ii][0], phis.T[ii][1])

etas = phidoubleprimes
for ii in range(len(Ns)):
    for aa in range(nF):
        etas[aa][ii] += sum([Gammamatrices[ii, aa, bb, cc] * phiprimes[bb, ii] * phiprimes[cc, ii]
                            for bb, cc in product(range(nF), repeat=2)])
np.save("./output/background/etas", etas)

eplls = np.array([epll(Gmatrices[ii], phidots.T[ii]) for ii in range(len(Ns))])
eperps = np.array([eperp2d(Gmatrices[ii], phidots.T[ii], etas.T[ii]) for ii in range(len(Ns))])
np.save("./output/background/eplls", eplls.T)
np.save("./output/background/eperps", eperps.T)

etaplls = np.array([dotG(Gmatrices[ii], etas.T[ii], eplls[ii]) for ii in range(len(Ns))])
np.save("./output/background/etaplls", etaplls)
plt.plot(Ns, etaplls, c="k", linewidth=2)
plt.title('Eta parallel')
plt.xlabel(r'$N$', fontsize=16)
plt.ylabel(r'$\eta_\parallel$', fontsize=16)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/background/etaplls.png")
plt.clf()

etaperps = np.array([dotG(Gmatrices[ii], etas.T[ii], eperps[ii]) for ii in range(len(Ns))])
np.save("./output/background/etaperps", etaperps)
plt.plot(Ns, etaperps, c="k", linewidth=2)
plt.title('Eta perpendicular')
plt.xlabel(r'$N$', fontsize=16)
plt.ylabel(r'$\eta_\perp$', fontsize=16)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/background/etaperps.png")
plt.clf()

omegas = etaperps / np.sqrt(2*epsilon)
np.save("./output/background/omegas", omegas)
plt.plot(Ns, omegas, c="k", linewidth=2)
plt.title('Turn rate')
plt.xlabel(r'$N$', fontsize=16)
plt.ylabel(r'$\omega$', fontsize=16)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/background/omegas.png")
plt.clf()
