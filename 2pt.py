import json
import sys
from math import sqrt

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from curved import eperp2d, epll
from slowroll import get_epsilons, get_etas, get_metrics

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi'] = 600

location = "/home/gsalinas/GitHub/angular/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransAngular as PyT
import PyTransScripts as PyS


def get_2pt_initial(back: np.ndarray, params: dict, efolds_before: float, NB: float = 8.):
    pval = np.array(list(params.values()))
    
    Ns = back[:, 0]
    Nexit = Ns[-1] - efolds_before

    k = PyS.kexitN(Nexit, back, pval, PyT)
    Nstart, backExitMinus = PyS.ICsBE(NB, k, back, pval, PyT)

    return k, Nstart, backExitMinus

def get_2pts(back: np.ndarray, params: dict, efolds_before: float, NB: float = 8., tol: float = 1e-16) -> tuple:
    nF = PyT.nF()
    pval = np.array(list(params.values()))
    
    k, Nstart, backExitMinus = get_2pt_initial(back, params, efolds_before, NB)

    Ns = back[:, 0]
    Nev = Ns[Ns >= Nstart]
    tols = np.array([tol, tol])
    twoPt = PyT.sigEvolve(Nev, k, backExitMinus, pval, tols, True)

    Nsig = twoPt[:, 0]
    Pzeta = twoPt[:, 1]
    sigma = twoPt[:, 1+1+2*nF:].reshape(len(Nsig), 2*nF, 2*nF)
    Pphi = sigma[:, :nF, :nF]

    k_deformed = k + 0.01*k
    twoPt_deformed = PyT.sigEvolve(Nev, k_deformed, backExitMinus, pval, tols, True)
    Pzeta_deformed = twoPt_deformed[:, 1]
    ns = (np.log(Pzeta_deformed[-1])-np.log(Pzeta[-1])) / (np.log(k_deformed)-np.log(k)) + 4.0

    Pzeta_nodim = Pzeta * k**3 / 2 / np.pi**2

    return Nsig, Pphi, Pzeta_nodim, k, ns

def get_PR_PS_CRS(back: np.ndarray, params: dict, efolds_before: float, NB: float = 8., tol: float = 1e-8):
    nF = PyT.nF()
    Nsig, Pphi, _, k, _ = get_2pts(back, params, efolds_before, NB, tol)

    back = back[Ns >= Nsig[0]]
    phidots = back[:, nF+1:]

    epsilons = get_epsilons(back, params)[:, 1]
    etas = get_etas(back, params)[:, 1:]

    Gs = get_metrics(back, params)
    PR = np.array([epll(Gs[ii], phidots[ii]) @ Gs[ii] @ Pphi[ii] @ Gs[ii] @ epll(Gs[ii], phidots[ii])
                for ii in range(len(Nsig))]) / 2 / epsilons
    PR_nodim = PR * k**3 / 2 / np.pi**2

    CRS = np.array([epll(Gs[ii], phidots[ii]) @ Gs[ii] @ Pphi[ii] @ Gs[ii] @ eperp2d(Gs[ii], phidots[ii], etas[ii])
                for ii in range(len(Nsig))]) / 2 / epsilons
    CRS_nodim = CRS * k**3 / 2 / np.pi**2

    PS = np.array([eperp2d(Gs[ii], phidots[ii], etas[ii]) @ Gs[ii] @ Pphi[ii] @ Gs[ii] @ eperp2d(Gs[ii], phidots[ii], etas[ii])
                for ii in range(len(Nsig))]) / 2 / epsilons
    PS_nodim = PS * k**3 / 2 / np.pi**2

    return PR_nodim, CRS_nodim, PS_nodim

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
back = np.load("./output/background/background.npy")
Ns =  back[:, 0]
efolds_before = 55.
Nexit = back[-1, 0] - efolds_before

Nsig, Pphi, Pzeta, k, ns = get_2pts(back, params, efolds_before)
print("ns: ", ns)
PR, CRS, PS = get_PR_PS_CRS(back, params, efolds_before)
np.save("./output/2pt/PR", PR)
np.save("./output/2pt/CRS", CRS)
np.save("./output/2pt/PS", PS)

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

print(f'Power spectrum at the end of inflation: {Pzeta[-1]:.3}')

iexit = np.argmin(np.abs(Nsig - Nexit))
print(f'Power spectrum at horizon crossing: {Pzeta[iexit]:.3}')

plt.plot(Nsig, PR, c='k')
plt.plot(Nsig, Pzeta, c='b', linestyle='--')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$P_R$ evolution',fontsize=16);
plt.ylabel(r'$P_R$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/2pt/PR.png")
plt.clf()

plt.plot(Nsig, np.abs(CRS), c='k')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$C_{RS}$ evolution',fontsize=16);
plt.ylabel(r'$\vert C_{RS} \vert$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/2pt/CRS.png")
plt.clf()

plt.plot(Nsig, PS, c='k')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$P_S$ evolution',fontsize=16);
plt.ylabel(r'$P_S$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/2pt/PS.png")
plt.clf()

betaa = beta(PR, CRS, PS)
np.save("./output/2pt/beta", betaa)
plt.plot(Nsig, betaa, c='k')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$\tilde{\beta}$ evolution',fontsize=16);
plt.ylabel(r'$\tilde{\beta}$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.tight_layout()
plt.savefig("./output/2pt/beta.png")
plt.clf()

TRSa = TRS(PR, CRS, PS, iexit)
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

TSSa = TSS(PS, iexit)
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

I1 = Is(PR, CRS, PS, iexit)[0]
I2 = Is(PR, CRS, PS, iexit)[1]
I3 = Is(PR, CRS, PS, iexit)[2]
I4 = Is(PR, CRS, PS, iexit)[3]
I5 = Is(PR, CRS, PS, iexit)[4]
with open("./output/2pt/Is.txt", "w") as f:
    f.write("I1 = " + str(I1) + "\n")
    f.write("I2 = " + str(I2) + "\n")
    f.write("I3 = " + str(I3) + "\n")
    f.write("I4 = " + str(I4) + "\n")
    f.write("I5 = " + str(I5) + "\n")
