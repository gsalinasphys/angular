import json
import sys
from math import cos, pi, sin, sqrt
from typing import Callable

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi'] = 600
location = "/home/gsalinas/GitHub/angular/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransAngular as PyT


def get_background(initial: np.ndarray, params: dict, Nrange: tuple, tol: float = 1e-30) -> np.ndarray:
    Ns = np.linspace(Nrange[0], Nrange[1], Nrange[2], endpoint=True)
    pval = np.array(list(params.values()))
    tols = np.array([tol, tol])
    back = PyT.backEvolve(Ns, initial, pval, tols, True)
    return back

def get_background_func(back: np.ndarray) -> Callable:
    Ns, phixs, phiys, phidotxs, phidotys = back[:, 0], back[:, 1], back[:, 2], back[:, 3], back[:, 4]

    return lambda N: np.interp(N, Ns, phixs), lambda N: np.interp(N, Ns, phiys), \
        lambda N: np.interp(N, Ns, phidotxs), lambda N: np.interp(N, Ns, phidotys),

if __name__ == '__main__':
    params = {'alpha': 1/600, 'R': 9, 'mphi': 2.e-5}
    with open("./output/setup/params.json", "w") as file:
        json.dump(params, file)
        
    r0, theta0 = 0.99, pi/4
    phi0 = r0 * np.array([cos(theta0), sin(theta0)])
    phidot0 = np.zeros(2)
    initial = np.concatenate((phi0, phidot0))

    Nrange = (0, 100, 100_000)
    back = get_background(initial, params, Nrange)
    np.save("output/background/background.npy", back)
    phix, phiy, phidotx, phidoty = get_background_func(back)

    # r = lambda N: np.sqrt(phi(N).T[0]**2 + phi(N).T[1]**2)
    # theta = lambda N: np.arctan(phi(N).T[1]/phi(N).T[0])
    # psi = lambda N: sqrt(6*params['alpha']) * np.arctanh(r(N))

    Nini, Nend = back[0, 0], back[-1, 0]
    print(f'Number of e-folds: {Nend:.3}')
    Nexit = Nend - 55

    num_points = 1_000
    Nplot = np.linspace(Nini, Nend, num_points)
    palette = sns.color_palette("crest", as_cmap=True)
    sns.scatterplot(x=phix(Nplot),
                    y=phiy(Nplot),
                    hue=Nplot,
                    s=5,
                    palette=palette)
    plt.scatter(phix(Nexit), phiy(Nexit), c="k")
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\chi$')
    plt.tight_layout()
    plt.savefig("./output/background/background.png")
    plt.clf()

    # sns.scatterplot(x=r(Nplot),
    #                 y=theta(Nplot),
    #                 hue=Nplot,
    #                 s=5,
    #                 palette=palette)
    # plt.scatter(r(Nexit), theta(Nexit), c="k")
    # plt.xlabel(r'$r$')
    # plt.ylabel(r'$\theta$')
    # plt.tight_layout()
    # plt.savefig("./output/background/background-polar.png")
    # plt.clf()

    # sns.scatterplot(x=(psi(Nplot)*np.cos(theta(Nplot))),
    #                 y=(psi(Nplot)*np.sin(theta(Nplot))),
    #                 hue=Nplot,
    #                 s=5,
    #                 palette=palette)
    # plt.scatter((psi(Nexit)*np.cos(theta(Nexit))), (psi(Nexit)*np.sin(theta(Nexit))), c="k")
    # plt.xlabel(r'$\psi \cos(\theta)$')
    # plt.ylabel(r'$\psi \sin(\theta)$')
    # plt.tight_layout()
    # plt.savefig("./output/background/background-canonical.png")
    # plt.clf()