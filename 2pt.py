import json
import sys
from itertools import product
from math import sqrt
from typing import Callable

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from background import get_background_func
from curved import eperp2d, epll
from slowroll import (get_epsilon_func, get_epsilons, get_eta_func, get_etas,
                      get_metric_func, get_metrics)

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

    k_deformed = k + 0.001*k
    twoPt_deformed = PyT.sigEvolve(Nev, k_deformed, backExitMinus, pval, tols, True)
    Pzeta_deformed = twoPt_deformed[:, 1]
    ns = (np.log(Pzeta_deformed[-1])-np.log(Pzeta[-1])) / (np.log(k_deformed)-np.log(k)) + 4.0

    Pzeta_nodim = Pzeta * k**3 / 2 / np.pi**2

    Pphi_func = lambda N: np.array([np.interp(N, Nsig, Pphi[:, aa, bb]) for aa, bb in product(range(nF), repeat=2)]).reshape(nF, nF)

    return Nsig, Pphi_func, lambda N: np.interp(N, Nsig, Pzeta_nodim), k, ns

def get_PR_PS_CRS(back: np.ndarray, params: dict, efolds_before: float, NB: float = 8., tol: float = 1e-8) -> tuple:
    _, _, phidotx, phidoty = get_background_func(back)
    _, Pphi, _, k, _ = get_2pts(back, params, efolds_before, NB, tol)

    epsilon = get_epsilon_func(back, params)
    eta = get_eta_func(back, params)
    G = get_metric_func(back, params)

    PR = lambda N: epll(G(N), np.array([phidotx(N), phidoty(N)])) @ G(N) @ Pphi(N) @ G(N) @ \
        epll(G(N), np.array([phidotx(N), phidoty(N)])) * k**3 / 4 / np.pi**2 / epsilon(N)
    CRS = lambda N: epll(G(N), np.array([phidotx(N), phidoty(N)])) @ G(N) @ Pphi(N) @ G(N) @ \
        eperp2d(G(N), np.array([phidotx(N), phidoty(N)]), eta(N)) * k**3 / 4 / np.pi**2 / epsilon(N)
    PS = lambda N: eperp2d(G(N), np.array([phidotx(N), phidoty(N)]), eta(N)) @ G(N) @ Pphi(N) @ G(N) @ \
        eperp2d(G(N), np.array([phidotx(N), phidoty(N)]), eta(N)) * k**3 / 4 / np.pi**2 / epsilon(N)

    return PR, CRS, PS

def alpha(PR: Callable, PS: Callable) -> Callable:
    return lambda N: PS(N) / PR(N)

def beta(PR: Callable, CRS: Callable) -> Callable:
    return lambda N: CRS(N) / PR(N)

def TRS(PR: Callable, CRS: Callable, PS: Callable, Nexit: float) -> Callable:
    return lambda N:(-CRS(Nexit) + np.sqrt(CRS(Nexit)**2 + PR(N)*PS(Nexit) - PR(Nexit)*PS(Nexit))) / PS(Nexit)

def TSS(PS: Callable, Nexit: float) -> Callable:
    return lambda N: np.sqrt(PS(N) / PS(Nexit))

def Is(PR: np.ndarray, CRS: np.ndarray, PS: np.ndarray, iexit: int):
    alpha_exit = alpha(PR, PS)[iexit]
    beta_exit = beta(PR, CRS)[iexit]
    TRS_end = TRS(PR, CRS, PS, iexit)[-1]

    I1 = (beta_exit+TRS_end*alpha_exit) * (1+TRS_end*beta_exit) / (1+2*TRS_end*beta_exit+TRS_end**2*alpha_exit)**2
    I2 = (beta_exit+TRS_end*alpha_exit)**2 / (1+2*TRS_end*beta_exit+TRS_end**2*alpha_exit)**2
    I3 = -(1+TRS_end*beta_exit) / (1+2*TRS_end*beta_exit+TRS_end**2*alpha_exit)
    I4 = (1+TRS_end*beta_exit) * (beta_exit+(alpha_exit-1)*TRS_end-beta_exit*TRS_end**2) / (1+2*TRS_end*beta_exit+TRS_end**2*alpha_exit)**2
    I5 = (beta_exit+TRS_end*alpha_exit) / (1+2*TRS_end*beta_exit+TRS_end**2*alpha_exit)
    I6 = (beta_exit+TRS_end*alpha_exit) * (beta_exit+(alpha_exit-1)*TRS_end-beta_exit*TRS_end**2) / (1+2*TRS_end*beta_exit+TRS_end**2*alpha_exit)**2

    return I1, I2, I3, I4, I5, I6

nF, nP = PyT.nF(), PyT.nP()
with open("./output/setup/params.json", "r") as file:
    params = json.loads(file.readline())
back = np.load("./output/background/background.npy")

Ns =  back[:, 0]
efolds_before = 55.
Nexit = back[-1, 0] - efolds_before

Nsig, Pphi, Pzeta, k, ns = get_2pts(back, params, efolds_before)
print("ns: ", ns)

num_points = 1_000
Nplot = np.linspace(Nsig[0], Nsig[-1], num_points)
Pphis = np.array([Pphi(_) for _ in Nplot])

plt.plot(Nplot, Pphis[:, 0, 0], label=r"$P^{11}_\phi$")
plt.plot(Nplot, np.abs(Pphis[:, 0, 1]), label=r"$\vert P^{12}_\phi \vert$")
plt.plot(Nplot, Pphis[:, 1, 1], label=r"$P^{22}_\phi$")
plt.title(r'$P_\phi$ evolution',fontsize=16)
plt.legend(fontsize=16)
plt.ylabel(r'Absolute 2pt field correlations', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.axvline(Nexit, c='k', linestyle='--')
plt.tight_layout()
plt.savefig("./output/2pt/2pt.png")
plt.clf()

PR, CRS, PS = get_PR_PS_CRS(back, params, efolds_before)
print(f'Power spectrum at the end of inflation: {Pzeta(Nsig[-1]):.3}')

print(f'Power spectrum at horizon crossing: {Pzeta(Nexit):.3}')
print(f'Power spectrum at horizon crossing 2: {PR(Nexit):.3}')

plt.plot(Nplot, [PR(_) for _ in Nplot], c='k')
plt.plot(Nplot, [Pzeta(_) for _ in Nplot], c='b', linestyle='--')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$P_R$ evolution',fontsize=16);
plt.ylabel(r'$P_R$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/2pt/PR.png")
plt.clf()

plt.plot(Nplot, np.abs([CRS(_) for _ in Nplot]), c='k')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$C_{RS}$ evolution',fontsize=16);
plt.ylabel(r'$\vert C_{RS} \vert$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/2pt/CRS.png")
plt.clf()

plt.plot(Nplot, [PS(_) for _ in Nplot], c='k')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$P_S$ evolution',fontsize=16);
plt.ylabel(r'$P_S$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/2pt/PS.png")
plt.clf()

betaa = [beta(PR, CRS)(_) for _ in Nplot]
np.save("./output/2pt/beta", betaa)
plt.plot(Nplot, betaa, c='k')
plt.axvline(Nexit, c='gray', linestyle='--')
plt.title(r'$\tilde{\beta}$ evolution',fontsize=16);
plt.ylabel(r'$\tilde{\beta}$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.tight_layout()
plt.savefig("./output/2pt/beta.png")
plt.clf()

TRSf = TRS(PR, CRS, PS, Nexit)
plt.plot(Nplot, [TRSf(_) for _ in Nplot], c='k')
plt.xlim([Nexit, Nsig[-1]])
plt.title(r'$T_{RS}$ evolution',fontsize=16);
plt.ylabel(r'$T_{RS}$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/2pt/TRS.png")
plt.clf()

TSSf = TSS(PS, Nexit)
plt.plot(Nplot, [TSSf(_) for _ in Nplot], c='k')
plt.xlim([Nexit, Nsig[-1]])
plt.title(r'$T_{SS}$ evolution',fontsize=16);
plt.ylabel(r'$T_{SS}$', fontsize=20) 
plt.xlabel(r'$N$', fontsize=20)
plt.yscale('log')
plt.tight_layout()
plt.savefig("./output/2pt/TSS.png")
plt.clf()

# I1 = Is(PR, CRS, PS, iexit)[0]
# I2 = Is(PR, CRS, PS, iexit)[1]
# I3 = Is(PR, CRS, PS, iexit)[2]
# I4 = Is(PR, CRS, PS, iexit)[3]
# I5 = Is(PR, CRS, PS, iexit)[4]
# I6 = Is(PR, CRS, PS, iexit)[5]
# with open("./output/2pt/Is.txt", "w") as f:
#     f.write("I1 = " + str(I1) + "\n")
#     f.write("I2 = " + str(I2) + "\n")
#     f.write("I3 = " + str(I3) + "\n")
#     f.write("I4 = " + str(I4) + "\n")
#     f.write("I5 = " + str(I5) + "\n")
#     f.write("I6 = " + str(I6) + "\n")
