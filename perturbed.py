import json
import sys
from math import cos, pi, sin

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from background import get_background
from curved import dotG, magG
from slowroll import get_etas, get_metrics, get_phi_primes

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi'] = 600
location = "/home/gsalinas/GitHub/angular/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransAngular as PyT


def deform_background(back: np.ndarray, params: dict, efolds_from_end: float = 55.,
                    epsilon: float = 1e-5, Nrange: tuple = (0, 100, 500_000), tol: float = 1e-300):
    Nexit = back[-1, 0] - efolds_from_end
    iexit = np.argmin(np.abs(back[:, 0] - Nexit))
    phis_phidots_exit = back[iexit, 1:]

    initials = [phis_phidots_exit,
                phis_phidots_exit*np.array([1.+epsilon, 1., 1., 1.]),
                phis_phidots_exit*np.array([1., 1.+epsilon, 1., 1.])]

    all_backs = [get_background(initial, params, Nrange=Nrange, tol=tol) for initial in initials]
    min_length = min([len(backgnd) for backgnd in all_backs])
    all_backs = [backgnd[:min_length] for backgnd in all_backs]

    return np.hstack((all_backs[0], all_backs[1][:, 1:], all_backs[2][:, 1:]))

def deformed_etas(deformed_back: np.ndarray, params: dict):
    all_backs = np.array([np.hstack((deformed_back[:, 0].reshape(-1, 1), deformed_back[:, 4*ii+1:4*ii+5])) for ii in range(3)])
    all_etas = []
    for solution in all_backs:
        all_etas.append(get_etas(solution, params)[:, 1:])    

    return np.hstack((deformed_back[:, 0].reshape(-1, 1), all_etas[0], all_etas[1], all_etas[2]))

# def deformed_gammas(deformed_back: np.ndarray, params: dict):
#     all_backs = np.array([np.hstack((deformed_back[:, 0].reshape(-1, 1), deformed_back[:, 4*ii+1:4*ii+5])) for ii in range(3)])
#     all_gammas = []
#     for solution in all_backs:
#         all_gammas.append(get_christoffels(solution, params))

#     return all_gammas

# def grad_etas(deformed_back: np.ndarray, params: dict):
#     dphis = np.array([deformed_back[0, 5]-deformed_back[0, 1], deformed_back[0, 10]-deformed_back[0, 2]])
#     all_etas = deformed_etas(deformed_back, params)
#     Gammas = get_christoffels(deformed_back[:, :5], params)

#     grads = []
#     for elem in all_etas:
#         grad = np.array([[(elem[3] - elem[1]) / dphis[0], (elem[5] - elem[1]) / dphis[0]],
#                          [(elem[4] - elem[2]) / dphis[1], (elem[6] - elem[2]) / dphis[1]]])

#         for aa in range(2):
#             for bb in range(2):
#                 grad[aa, bb] += sum(Gammas[bb, aa, cc]*elem[cc+1] for cc in range(2))

#         grads.append(grad)

#     return np.array(grads)

if __name__ == '__main__':
    with open("./output/setup/params.json", "r") as file:
        params = json.loads(file.readline())

    r0, theta0 = 0.99, pi/4
    phi0 = r0 * np.array([cos(theta0), sin(theta0)])
    phidot0 = np.zeros(2)
    initial = np.concatenate((phi0, phidot0))

    back = get_background(initial, params)
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

    etas_exit = deformed_etas(deformed_back, params)[1]

    partial1_eta1 = (etas_exit[3] - etas_exit[1]) / 1e-5 / deformed_back[1, 1]
    partial1_eta2 = (etas_exit[4] - etas_exit[2]) / 1e-5 / deformed_back[1, 1]
    partial2_eta1 = (etas_exit[5] - etas_exit[1]) / 1e-5 / deformed_back[1, 2]
    partial2_eta2 = (etas_exit[6] - etas_exit[2]) / 1e-5 / deformed_back[1, 2]
    print(partial1_eta1)
    print(partial1_eta2)
    print(partial2_eta1)
    print(partial2_eta2)

    # etas_normal = deformed_etas(deformed_back, params)[:, 1:3]
    # print(etas_normal)
    # dN = back[1, 0] - back[0, 0]
    # detas = np.gradient(etas_normal.T, dN, axis=1).T
    # phi_primes = get_phi_primes(deformed_back[:, :5], params)[:, 1:]

    # print(detas[1, 0])
    # print(detas[1, 1])
    # print(phi_primes[1])

    # Gs = get_metrics(deformed_back[:, :5], params)
    # print(Gs[1])
    # print(detas[1, 0] / magG(Gs[1], phi_primes[1]))
    # print(detas[1, 1] / magG(Gs[1], phi_primes[1]))

    # print(-np.dot(np.array([partial1_eta1, partial2_eta1]), phi_primes[1]))
    # print(-np.dot(np.array([partial1_eta2, partial2_eta2]), phi_primes[1]))

    # print(grad_etas(deformed_back, params))
