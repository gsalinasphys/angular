import json
import pickle
import sys
from math import atan, cos, pi, sin, sqrt

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from background import get_background
from curved import dotG, epll, magG
from slowroll import (get_christoffels, get_epsilons, get_eta_parallel_perp,
                      get_etas, get_Hs, get_kin_basis, get_metrics,
                      get_phi_primes)

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi'] = 600
location = "/home/gsalinas/GitHub/angular/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransAngular as PyT


def get_mass_matrices(back: np.ndarray, params: dict) -> np.ndarray:
    nF = PyT.nF()
    pval = np.array(list(params.values()))
    phis = back[:, 1:nF+1]

    return [PyT.ddV(phi, pval) / PyT.V(phi, pval) - np.outer(PyT.dV(phi, pval), PyT.dV(phi, pval)) / PyT.V(phi, pval)**2
        for phi in phis]

def deform_background(back: np.ndarray, params: dict, efolds_from_end: float = 55.,
                    epsilon: float = 1e-5, Nrange: tuple = (0, 100, 500_000), tol: float = 1e-12):
    Nexit = back[-1, 0] - efolds_from_end
    iexit = np.argmin(np.abs(back[:, 0] - Nexit))
    phis_phidots_exit = back[iexit, 1:]

    initials = [phis_phidots_exit,
                phis_phidots_exit*np.array([1.+epsilon, 1., 1., 1.]),
                phis_phidots_exit*np.array([1., 1.+epsilon, 1., 1.])]

    Hs = get_Hs(back, params)
    mphi, R =  params['mphi'], params['R']
    deformed_rs = [np.linalg.norm(initials[ii][:2]) for ii in range(1, 3)]
    deformed_thetas = [atan(initials[ii][1]/initials[ii][0]) for ii in range(1, 3)]
    deformed_thetadots = [-1 / 18 / Hs[iexit, 1] * mphi**2 * (R-1) * (1-deformed_rs[ii]**2)**2 * \
                        initials[ii+1][0] * initials[ii+1][1] / deformed_rs[ii]**2 for ii in range(2)]
    deformed_phidots = [deformed_rs[ii] * deformed_thetadots[ii] * np.array([-sin(deformed_thetas[ii]), cos(deformed_thetas[ii])]) for ii in range(2)]

    initials = [phis_phidots_exit,
                phis_phidots_exit*np.array([1.+epsilon, 1., 0., 0.]) + np.concatenate(([0.,0.], deformed_phidots[0])),
                phis_phidots_exit*np.array([1., 1.+epsilon, 0., 0.]) + np.concatenate(([0.,0.], deformed_phidots[1]))]

    all_backs = [get_background(initial, params, Nrange=Nrange, tol=tol) for initial in initials]
    min_length = min([len(backgnd) for backgnd in all_backs])
    all_backs = [backgnd[:min_length] for backgnd in all_backs]

    return np.hstack((all_backs[0], all_backs[1][:, 1:], all_backs[2][:, 1:]))

def deformed_epsilons(deformed_back: np.ndarray, params: dict):
    all_backs = np.array([np.hstack((deformed_back[:, 0].reshape(-1, 1), deformed_back[:, 4*ii+1:4*ii+5])) for ii in range(3)])
    all_epsilons = []
    for solution in all_backs:
        all_epsilons.append(get_epsilons(solution, params)[:, 1:])    

    return np.hstack((deformed_back[:, 0].reshape(-1, 1), all_epsilons[0], all_epsilons[1], all_epsilons[2]))

def grad_epsilon_exit(deformed_back: np.ndarray, params: dict):
    dphis = np.array([deformed_back[0, 5]-deformed_back[0, 1], deformed_back[0, 10]-deformed_back[0, 2]])
    epsilons = deformed_epsilons(deformed_back, params)[0, 1:]
    return np.array([(epsilons[1]-epsilons[0])/dphis[0], (epsilons[2]-epsilons[0])/dphis[1]])

def deformed_etas(deformed_back: np.ndarray, params: dict):
    all_backs = np.array([np.hstack((deformed_back[:, 0].reshape(-1, 1), deformed_back[:, 4*ii+1:4*ii+5])) for ii in range(3)])
    all_etas = []
    for solution in all_backs:
        all_etas.append(get_etas(solution, params)[:, 1:])    

    return np.hstack((deformed_back[:, 0].reshape(-1, 1), all_etas[0], all_etas[1], all_etas[2]))

def grad_etas_exit(deformed_back: np.ndarray, params: dict):
    dphis = np.array([deformed_back[0, 5]-deformed_back[0, 1], deformed_back[0, 10]-deformed_back[0, 2]])
    etas = deformed_etas(deformed_back, params)[0]
    Gammas = get_christoffels(deformed_back[:, :5], params)

    grad = np.array([[(etas[3] - etas[1]) / dphis[0], (etas[4] - etas[2]) / dphis[0]],
                    [(etas[5] - etas[1]) / dphis[1], (etas[6] - etas[2]) / dphis[1]]])

    for aa in range(2):
        for bb in range(2):
            grad[aa, bb] += sum(Gammas[0, bb, aa, cc]*etas[cc+1] for cc in range(2))

    return grad

def get_mass_matrices_exit(deformed_back: np.ndarray, params: dict) -> np.ndarray:
    nF = PyT.nF()
    pval = np.array(list(params.values()))
    phi = deformed_back[0, 1:nF+1]

    return PyT.ddV(phi, pval) / PyT.V(phi, pval) - np.outer(PyT.dV(phi, pval), PyT.dV(phi, pval)) / PyT.V(phi, pval)**2

def get_tildeM_exit(deformed_back: np.ndarray, params: dict) -> np.ndarray:
    G = get_metrics(deformed_back[:, :5], params)[0]
    epsilon = get_epsilons(deformed_back[:, :5], params)[0, 1:]
    tildeM = get_mass_matrices_exit(deformed_back[:, :5], params) + grad_etas_exit(deformed_back, params) @ G / (3-epsilon)
    
    etapll = get_eta_parallel_perp(deformed_back[:, :5], params)[0, 1]
    phi_prime = get_phi_primes(deformed_back[:, :5], params)[0, 1:]
    denominator = 1 + etapll*magG(G, phi_prime)/(3-epsilon)**2
    tildeM /= denominator

    return tildeM

if __name__ == '__main__':
    nF, nP = PyT.nF(), PyT.nP()
    with open("./output/setup/params.json", "r") as file:
        params = json.loads(file.readline())
    back = np.load("./output/background/background.npy")
    
    deformed_back = deform_background(back, params)

    labels = ["Original", fr"Perturbed in $\phi$", fr"Perturbed in $\chi$"]
    for ii in range(3):
        plt.plot(deformed_back[:, 4*ii+1], deformed_back[:, 4*ii+2], label=labels[ii])
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\chi$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("./output/perturbed/from_exit.png")
    plt.clf()

    # print(get_mass_matrices_exit(deformed_back, params))

    epsilon_exit = get_epsilons(deformed_back[:, :5], params)[0, 1]
    print(epsilon_exit)

    kin_basis = get_kin_basis(deformed_back[:, :5], params)
    epll_exit, eperp_exit = kin_basis[0, 1:3], kin_basis[0, 3:5]
    # print(epll_exit, eperp_exit)

    eta_pll_exit = get_eta_parallel_perp(deformed_back[:, :5], params)[0, 1]
    eta_perp_exit = get_eta_parallel_perp(deformed_back[:, :5], params)[0, 2]
    print(eta_perp_exit)

    # grad_epsilon = grad_epsilon_exit(deformed_back, params)
    # print(grad_epsilon)
    # grad_etas = grad_etas_exit(deformed_back, params)
    # print(grad_etas)
    # print(deformed_back[0])
    # print(get_mass_matrices_exit(deformed_back, params))
    tildeM = get_tildeM_exit(deformed_back, params)
    # print(tildeM)

    # phi_prime_exit = get_phi_primes(deformed_back[:, :5], params)[0, 1:]

    # print(tildeM @ phi_prime_exit)

    # print(epll_exit @ grad_epsilon)
    # print(-np.sqrt(2*epsilon_exit) * epll_exit @ tildeM @ epll_exit)

    # print(eperp_exit @ grad_epsilon)
    # print(-np.sqrt(2*epsilon_exit) * eperp_exit @ tildeM @ epll_exit)

    print(eperp_exit @ tildeM @ epll_exit)
    print(epll_exit @ tildeM @ eperp_exit)

    I1 = 0.03769592688493446
    I2 = 0.012327765535058
    I3 = 0.33950974428255654
    I4 = 0.7590259701237988
    I5 = 0.21191511228469154

    print(eperp_exit @ tildeM @ eperp_exit * I5)
    print(eta_perp_exit**2 / (3-epsilon_exit)**2 * I5)
    print(-eta_perp_exit / sqrt(2*epsilon_exit) * I4)
    print(-eta_pll_exit / sqrt(2*epsilon_exit) * I3)
