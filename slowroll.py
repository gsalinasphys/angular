import json
import pickle
import sys
from functools import partial
from itertools import combinations_with_replacement, product
from math import cos, pi, sin
from typing import Callable, List

import numdifftools as nd
import numpy as np
from matplotlib import pyplot as plt
from sympy.utilities import lambdify

from background import get_background
from curved import dotG, eperp2d, epll, magG

location = "/home/gsalinas/GitHub/angular/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransAngular as PyT


def get_Hs(back: np.ndarray, params: dict) -> tuple:
    pval = np.array(list(params.values()))
    Hs = np.array([PyT.H(elem, pval) for elem in back[:, 1:]])
    return np.hstack((back[:, 0].reshape(-1,1), Hs.reshape(-1,1)))

def get_epsilons(back: np.ndarray, params: dict) -> tuple:
    dN = back[1, 0] - back[0, 0]
    Hs = get_Hs(back, params)[:, 1]
    epsilons = -np.gradient(Hs, dN)/Hs
    return np.hstack((back[:, 0].reshape(-1,1), epsilons.reshape(-1,1)))

def get_phi_primes(back: np.ndarray, params: dict) -> tuple:
    nF = PyT.nF()
    Hs = get_Hs(back, params)[:, 1]
    phi_primes = (back[:, nF+1:].T / Hs).T
    return np.hstack((back[:, 0].reshape(-1,1), phi_primes))

def get_phi_double_primes(back: np.ndarray, params: dict) -> tuple:
    dN = back[1, 0] - back[0, 0]
    phi_primes = get_phi_primes(back, params)[:, 1:]
    phi_double_primes = np.gradient(phi_primes.T, dN, axis=1).T
    return np.hstack((back[:, 0].reshape(-1,1), phi_double_primes))

def get_metric_sympy():
    with open("./output/setup/G.txt", "rb") as file:
        G = pickle.load(file)

    return G

def get_metric_func(params: dict) -> Callable:
    nF = PyT.nF()
    G = get_metric_sympy()
    pval = list(params.values())
    params_subs = {'p_'+str(ii): pval[ii] for ii in range(len(pval))}

    return lambdify(['f_'+str(ii) for ii in range(nF)], G.subs(params_subs))

def get_metrics(back: np.ndarray, params: dict) -> np.ndarray:
    nF = PyT.nF()
    phis = back[:, 1:nF+1]    
    G = get_metric_func(params)
    return np.array([G(phi[0], phi[1]) for phi in phis])

def get_christoffel_func(params: dict) -> List[Callable]:
    nF, nP = PyT.nF(), PyT.nP()
    Gamma_sympy = PyTransSetup.fieldmetric(get_metric_sympy(), nF, nP)[1]

    Gamma_func = np.empty((nF, nF, nF)).tolist()
    pval = list(params.values())
    params_subs = {'p_'+str(ii): pval[ii] for ii in range(len(pval))}
    for aa, (bb, cc) in product(range(1, nF+1), combinations_with_replacement(range(1, nF+1), 2)):
        Gamma_func[aa-1][bb-1][cc-1] = lambdify(['f_'+str(ii) for ii in range(nF)], Gamma_sympy(-aa, bb, cc).subs(params_subs))
        if bb != cc:
            Gamma_func[aa-1][cc-1][bb-1] = Gamma_func[aa-1][bb-1][cc-1]

    return Gamma_func

def get_christoffels(back: np.ndarray, params: dict) -> np.ndarray:
    nF = PyT.nF()
    Ns, phis = back[:, 0], back[:, 1:nF+1]
    Gamma_func = get_christoffel_func(params)
    Gammas = np.empty((len(Ns), nF, nF, nF))
    for ii in range(len(Ns)):
        for aa, (bb, cc) in product(range(nF), combinations_with_replacement(range(nF), 2)):
            Gammas[ii, aa, bb, cc] = Gamma_func[aa][bb][cc](phis[ii][0], phis[ii][1])
            if bb != cc:
                Gammas[ii, aa, cc, bb] = Gammas[ii, aa, bb, cc]

    return Gammas

def get_etas(back: np.ndarray, params: dict) -> np.ndarray:
    nF = PyT.nF()
    Gammas = get_christoffels(back, params)
    phi_primes = get_phi_primes(back, params)[:, 1:]
    etas = get_phi_double_primes(back, params)[:, 1:]
    for ii in range(len(back[:, 0])):
        for aa in range(nF):
            etas[ii, aa] += sum([Gammas[ii, aa, bb, cc] * phi_primes[ii, bb] * phi_primes[ii, cc]
                                for bb, cc in product(range(nF), repeat=2)])

    return np.hstack((back[:, 0].reshape(-1, 1), etas))

def get_kin_basis(back: np.ndarray, params: dict) -> np.ndarray:
    nF = PyT.nF()
    Gs = get_metrics(back, params)
    etas = get_etas(back, params)[:, 1:]
    eplls = np.array([epll(Gs[ii], back[ii, nF+1:]) for ii in range(len(back[:, 0]))])
    eperps = np.array([eperp2d(Gs[ii], back[ii, nF+1:], etas[ii]) for ii in range(len(back[:, 0]))])
    return np.hstack((back[:, 0].reshape(-1,1), eplls, eperps))

def get_eta_parallel_perp(back: np.ndarray, params: dict) -> np.ndarray:
    nF = PyT.nF()
    Gs = get_metrics(back, params)
    etas = get_etas(back, params)[:, 1:]
    eplls, eperps = get_kin_basis(back, params)[:, 1:nF+1], get_kin_basis(back, params)[:, nF+1:]
    etaplls = np.array([dotG(Gs[ii], etas[ii], eplls[ii]) for ii in range(len(back[:, 0]))])
    etaperps = np.array([dotG(Gs[ii], etas[ii], eperps[ii]) for ii in range(len(back[:, 0]))])

    return np.hstack((back[:, 0].reshape(-1,1), etaplls.reshape(-1,1), etaperps.reshape(-1,1)))

if __name__ == '__main__':
    nF, nP = PyT.nF(), PyT.nP()
    with open("./output/setup/params.json", "r") as file:
        params = json.loads(file.readline())
    back = np.load("./output/background/background.npy")

    with open("./output/setup/G.txt", "rb") as file:
        G = pickle.load(file)

    Hs = get_Hs(back, params)
    np.save("./output/background/Hs", Hs)
    plt.plot(Hs[:, 0], Hs[:, 1], c="k", linewidth=2)
    plt.title('Hubble parameter')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$H$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/Hs.png")
    plt.clf()

    epsilons = get_epsilons(back, params)
    np.save("./output/background/epsilons", epsilons)
    plt.plot(epsilons[:, 0], epsilons[:, 1], c="k", linewidth=2)
    plt.title('Epsilon parameter')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$\epsilon$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/epsilons.png")
    plt.clf()

    phiprimes = get_phi_primes(back, params)
    np.save("./output/background/phiprimes", phiprimes)
    phidoubleprimes = get_phi_double_primes(back, params)
    np.save("./output/background/phidoubleprimes", phidoubleprimes)

    etaskin = get_eta_parallel_perp(back, params)
    etaplls, etaperps = etaskin[:, 1], etaskin[:, 2]
    np.save("./output/background/etaplls", etaplls)
    plt.plot(back[:, 0], np.abs(etaplls), c="k", linewidth=2)
    plt.title('Eta parallel')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$\vert \eta_\parallel \vert$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/etaplls.png")
    plt.clf()

    np.save("./output/background/etaperps", etaperps)
    plt.plot(back[:, 0], etaperps, c="k", linewidth=2)
    plt.title('Eta perpendicular')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$\eta_\perp$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/etaperps.png")
    plt.clf()

    omegas = etaperps / np.sqrt(2*epsilons[:, 1])
    np.save("./output/background/omegas", omegas)
    plt.plot(back[:, 0], omegas, c="k", linewidth=2)
    plt.title('Turn rate')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$\omega$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/omegas.png")
    plt.clf()
