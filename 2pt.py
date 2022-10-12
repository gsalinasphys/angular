import json
import pickle
import sys
from math import cos, pi, sin, sqrt

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sympy.utilities import lambdify

from curved import dotG, eperp, epll, magG

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi'] = 600

location = "/home/gsalinas/GitHub/angular/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransAngular as PyT
import PyTransScripts as PyS

nF, nP = PyT.nF(), PyT.nP()
with open("./output/params.json", "r") as file:
    params = json.loads(file.readline())
pval = np.array(list(params.values()))
back = np.load("./output/background.npy")
epsilon = np.load("./output/epsilon.npy")
Ns =  back.T[0]

Nend = Ns[-1]
Nexit = Nend - 55
iexit = np.argmin(np.abs(Ns - Nexit))
k = PyS.kexitN(Nexit, back, pval, PyT) 

print(f'Horizon exit at N = {Nexit:.3} with k = {k:.3}')

NB = 8.0
Nstart, backExitMinus = PyS.ICsBE(NB, k, back, pval, PyT)

print(f"2-pt calculation starts at: {Nstart:.3} e-folds")

Nev = Ns[Ns > Nstart]
back = back[Ns > Nstart]
epsilon = epsilon[Ns > Nstart]
phis, phidots = back.T[1:nF+1], back.T[nF+1:]

tols = np.array([10**-8, 10**-8])
twoPt = PyT.sigEvolve(Nev, k, backExitMinus, pval, tols, True)
Nsig = twoPt[:, 0]
Pzeta = twoPt[:, 1]
sigma = twoPt[:, 1+1+2*nF:].reshape(len(Nsig), 4, 4)
Pphi = sigma[:, :2, :2]

Pzeta_nodim = Pzeta * k**3 / 2 / np.pi**2

k_deformed = k + 0.01*k
twoPt_deformed = PyT.sigEvolve(Nev, k_deformed, backExitMinus, pval, tols, True)
Pzeta_deformed = twoPt_deformed[:,1]
n_s = (np.log(Pzeta_deformed[-1])-np.log(Pzeta[-1])) / (np.log(k_deformed)-np.log(k)) + 4.0
print(f'n_s: {n_s:.3f}')

plt.plot(Nsig, Pphi[:, 0, 0], label=r"$P^{11}$")
plt.plot(Nsig, np.abs(Pphi[:, 0, 1]), label=r"$\vert P^{12} \vert$")
plt.plot(Nsig, Pphi[:, 1, 1], label=r"$P^{22}$")
plt.title(r'$\Sigma$ evolution',fontsize=16)
plt.legend(fontsize=16)
plt.ylabel(r'Aboslute 2pt field correlations', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.axvline(Nexit, c='k', linestyle='--')
plt.tight_layout()
plt.savefig("./output/2pt.png")
plt.clf()

print(f'k: {k:.3}')
print(f'Power spectrum at the end of inflation: {Pzeta_nodim[-1]:.3}')

iexit = np.argmin(np.abs(Nsig - Nexit))
print(f'Power spectrum at horizon crossing: {Pzeta_nodim[iexit]:.3}')

with open("./output/G.txt", "rb") as file:
    G = pickle.load(file)

params_subs = {'p_'+str(ii): pval[ii] for ii in range(len(pval))}
Gparams = G.subs(params_subs)
Glbd = lambdify(['f_'+str(ii) for ii in range(nF)], Gparams)

Gmatrices = np.array([Glbd(phi[0], phi[1]) for phi in phis.T])

PR = np.array([epll(Gmatrices[ii], phidots.T[ii]) @
            Gmatrices[ii] @
            Pphi[ii] @
            Gmatrices[ii] @
            epll(Gmatrices[ii], phidots.T[ii])
            for ii in range(len(Nsig))]) / 2 / epsilon
PR_nodim = PR * k**3 / 2 / np.pi**2
np.save("./output/PR", PR_nodim)
plt.plot(Nsig, PR_nodim, c='k')
plt.plot(Nsig, Pzeta_nodim, c='k', linestyle='--')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$P_R$ evolution',fontsize=16);
plt.ylabel(r'$P_R$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/PR.png")
plt.clf()

CRS = np.array([epll(Gmatrices[ii], phidots.T[ii]) @
            Gmatrices[ii] @
            Pphi[ii] @
            Gmatrices[ii] @
            eperp(Gmatrices[ii], phidots.T[ii])
            for ii in range(len(Nsig))]) / 2 / epsilon
CRS_nodim = CRS * k**3 / 2 / np.pi**2
np.save("./output/CRS", CRS_nodim)
plt.plot(Nsig, np.abs(CRS_nodim), c='k')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$C_{RS}$ evolution',fontsize=16);
plt.ylabel(r'$\vert C_{RS} \vert$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/CRS.png")
plt.clf()

PS = np.array([eperp(Gmatrices[ii], phidots.T[ii]) @
            Gmatrices[ii] @
            Pphi[ii] @
            Gmatrices[ii] @
            eperp(Gmatrices[ii], phidots.T[ii])
            for ii in range(len(Nsig))]) / 2 / epsilon
PS_nodim = PS * k**3 / 2 / np.pi**2
np.save("./output/PS", PS_nodim)
plt.plot(Nsig, PS_nodim, c='k')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$P_S$ evolution',fontsize=16);
plt.ylabel(r'$P_S$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/PS.png")
plt.clf()

