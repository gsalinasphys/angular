import json
import pickle
import sys
from math import sqrt

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from sympy.utilities import lambdify

from curved import eperp2d, epll

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi'] = 600

location = "/home/gsalinas/GitHub/angular/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransAngular as PyT
import PyTransScripts as PyS


def beta(PR: np.ndarray, CRS: np.ndarray, PS: np.ndarray) -> np.ndarray:
    return CRS / np.sqrt(PR * PS)

def TRS(PR: np.ndarray, CRS: np.ndarray, PS: np.ndarray, iexit: int) -> np.ndarray:
    return (-CRS[iexit] + np.sqrt(CRS[iexit]**2 + PR*PS[iexit] - PR[iexit]*PS[iexit])) / PS[iexit]

def TSS(PS: np.ndarray, iexit: int) -> np.ndarray:
    return np.sqrt(PS / PS[iexit])

def Is(PR: np.ndarray, CRS: np.ndarray, PS: np.ndarray, iexit: int):
    beta_end = beta(PR, CRS, PS)[-1]
    TRS_end = TRS(PR, CRS, PS, iexit)[-1]

    I1 = beta_end * sqrt(PS[iexit]/PR[-1]) * (1 - TRS_end*beta_end*sqrt(PS[iexit]/PR[-1]))
    I2 = beta_end**2 * PS[iexit]/PR[-1]
    I3 = 1 - TRS_end*beta_end*sqrt(PS[iexit]/PR[-1])
    I4 = TRS_end * (1 - 2*TRS_end*beta_end*sqrt(PS[iexit]/PR[-1]) + (1+TRS_end**2)*beta_end**2*PS[iexit]/PR[-1])
    I5 = beta_end * sqrt(PS[iexit]/PR[-1]) * (TRS_end - (1+TRS_end**2)*beta_end*sqrt(PS[iexit]/PR[-1]))

    return I1, I2, I3, I4, I5

nF, nP = PyT.nF(), PyT.nP()
with open("./output/setup/params.json", "r") as file:
    params = json.loads(file.readline())
pval = np.array(list(params.values()))
back = np.load("./output/background/background.npy")
epsilon = np.load("./output/background/epsilon.npy")
etas = np.load("./output/background/etas.npy")
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
np.save("./output/2pt/Nsig", Nsig)
Pzeta = twoPt[:, 1]
sigma = twoPt[:, 1+1+2*nF:].reshape(len(Nsig), 2*nF, 2*nF)
Pphi = sigma[:, :nF, :nF]

Pzeta_nodim = Pzeta * k**3 / 2 / np.pi**2

k_deformed = k + 0.01*k
twoPt_deformed = PyT.sigEvolve(Nev, k_deformed, backExitMinus, pval, tols, True)
Pzeta_deformed = twoPt_deformed[:,1]
n_s = (np.log(Pzeta_deformed[-1])-np.log(Pzeta[-1])) / (np.log(k_deformed)-np.log(k)) + 4.0
print(f'n_s: {n_s:.3f}')

plt.plot(Nsig, Pphi[:, 0, 0], label=r"$P^{11}_\phi$")
plt.plot(Nsig, np.abs(Pphi[:, 0, 1]), label=r"$\vert P^{12}_\phi \vert$")
plt.plot(Nsig, Pphi[:, 1, 1], label=r"$P^{22}_\phi$")
plt.title(r'$P_\phi$ evolution',fontsize=16)
plt.legend(fontsize=16)
plt.ylabel(r'Absolute 2pt field correlations', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.axvline(Nexit, c='k', linestyle='--')
plt.tight_layout()
plt.savefig("./output/2pt/2pt.png")
plt.clf()

print(f'k: {k:.3}')
print(f'Power spectrum at the end of inflation: {Pzeta_nodim[-1]:.3}')

iexit = np.argmin(np.abs(Nsig - Nexit))
print(f'Power spectrum at horizon crossing: {Pzeta_nodim[iexit]:.3}')

with open("./output/setup/G.txt", "rb") as file:
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
np.save("./output/2pt/PR", PR_nodim)
plt.plot(Nsig, PR_nodim, c='k')
plt.plot(Nsig, Pzeta_nodim, c='k', linestyle='--')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$P_R$ evolution',fontsize=16);
plt.ylabel(r'$P_R$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/2pt/PR.png")
plt.clf()

CRS = np.array([epll(Gmatrices[ii], phidots.T[ii]) @
            Gmatrices[ii] @
            Pphi[ii] @
            Gmatrices[ii] @
            eperp2d(Gmatrices[ii], phidots.T[ii], etas.T[ii])
            for ii in range(len(Nsig))]) / 2 / epsilon
CRS_nodim = CRS * k**3 / 2 / np.pi**2
np.save("./output/2pt/CRS", CRS_nodim)
plt.plot(Nsig, np.abs(CRS_nodim), c='k')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$C_{RS}$ evolution',fontsize=16);
plt.ylabel(r'$\vert C_{RS} \vert$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/2pt/CRS.png")
plt.clf()

PS = np.array([eperp2d(Gmatrices[ii], phidots.T[ii], etas.T[ii]) @
            Gmatrices[ii] @
            Pphi[ii] @
            Gmatrices[ii] @
            eperp2d(Gmatrices[ii], phidots.T[ii], etas.T[ii])
            for ii in range(len(Nsig))]) / 2 / epsilon
PS_nodim = PS * k**3 / 2 / np.pi**2
np.save("./output/2pt/PS", PS_nodim)
plt.plot(Nsig, PS_nodim, c='k')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$P_S$ evolution',fontsize=16);
plt.ylabel(r'$P_S$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/2pt/PS.png")
plt.clf()

betaa = beta(PR_nodim, CRS_nodim, PS_nodim)
np.save("./output/2pt/beta", betaa)
plt.plot(Nsig, betaa, c='k')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$\tilde{\beta}$ evolution',fontsize=16);
plt.ylabel(r'$\tilde{\beta}$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.tight_layout()
plt.savefig("./output/2pt/beta.png")
plt.clf()

TRSa = TRS(PR_nodim, CRS_nodim, PS_nodim, iexit)
np.save("./output/2pt/TRS", TRSa)
plt.plot(Nsig, TRSa, c='k')
plt.xlim([Nsig[iexit], Nsig[-1]])
plt.title(r'$T_{RS}$ evolution',fontsize=16);
plt.ylabel(r'$T_{RS}$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/2pt/TRS.png")
plt.clf()

TSSa = TSS(PS_nodim, iexit)
np.save("./output/2pt/TSS", TSSa)
plt.plot(Nsig, TSSa, c='k')
plt.xlim([Nsig[iexit], Nsig[-1]])
plt.title(r'$T_{SS}$ evolution',fontsize=16);
plt.ylabel(r'$T_{SS}$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/2pt/TSS.png")
plt.clf()

I1 = Is(PR_nodim, CRS_nodim, PS_nodim, iexit)[0]
I2 = Is(PR_nodim, CRS_nodim, PS_nodim, iexit)[1]
I3 = Is(PR_nodim, CRS_nodim, PS_nodim, iexit)[2]
I4 = Is(PR_nodim, CRS_nodim, PS_nodim, iexit)[3]
I5 = Is(PR_nodim, CRS_nodim, PS_nodim, iexit)[4]
with open("./output/2pt/Is.txt", "w") as f:
    f.write("I1 = " + str(I1) + "\n")
    f.write("I2 = " + str(I2) + "\n")
    f.write("I3 = " + str(I3) + "\n")
    f.write("I4 = " + str(I4) + "\n")
    f.write("I5 = " + str(I5) + "\n")
