import sys
from math import cos, pi, sin, sqrt

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


def get_background(initial: np.ndarray, params: dict, Nrange: tuple = (0, 100, 500_000), tol: float = 1e-12) -> np.ndarray:
    Ns = np.linspace(Nrange[0], Nrange[1], Nrange[2])
    pval = np.array(list(params.values()))
    tols = np.array([tol, tol])
    back = PyT.backEvolve(Ns, initial, pval, tols, True)
    return back

if __name__ == '__main__':
    params = {'alpha': 1/600, 'R': 9, 'mphi': 2.e-5}
    r0, theta0 = 0.99, pi/4
    phi0 = r0 * np.array([cos(theta0), sin(theta0)])
    phidot0 = np.zeros(2)
    initial = np.concatenate((phi0, phidot0))

    back = get_background(initial, params)
    nF = PyT.nF()
    Ns, phis, phidots = back[:, 0], back[:, 1:nF+1], back[:, nF+1:]
    rs = np.sqrt(phis.T[0]**2 + phis.T[1]**2)
    thetas = np.arctan(phis.T[1]/phis.T[0])
    psis = sqrt(6*params['alpha']) * np.arctanh(rs)

    Nend = Ns[-1]
    print(f'Number of e-folds: {Nend:.3}')
    Nexit = Nend - 55
    iexit = np.argmin(np.abs(Ns - Nexit))
    np.save("./output/background/exit", np.concatenate((phis[iexit], phidots[iexit])))

    palette = sns.color_palette("crest", as_cmap=True)
    num_points = 500
    sns.scatterplot(x=phis.T[0][::len(Ns)//num_points],
                    y=phis.T[1][::len(Ns)//num_points],
                    hue=Ns[::len(Ns)//num_points],
                    s=5,
                    palette=palette)
    plt.scatter(phis[iexit][0], phis[iexit][1], c="k")
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\chi$')
    plt.tight_layout()
    plt.savefig("./output/background/background.png")
    plt.clf()

    sns.scatterplot(x=rs[::len(Ns)//num_points],
                    y=thetas[::len(Ns)//num_points],
                    hue=Ns[::len(Ns)//num_points],
                    s=5,
                    palette=palette)
    plt.scatter(rs[iexit], thetas[iexit], c="k")
    plt.xlabel(r'$r$')
    plt.ylabel(r'$\theta$')
    plt.tight_layout()
    plt.savefig("./output/background/background-polar.png")
    plt.clf()

    sns.scatterplot(x=(psis*np.cos(thetas))[::len(Ns)//num_points],
                    y=(psis*np.sin(thetas))[::len(Ns)//num_points],
                    hue=Ns[::len(Ns)//num_points],
                    s=5,
                    palette=palette)
    plt.scatter((psis*np.cos(thetas))[iexit], (psis*np.sin(thetas))[iexit], c="k")
    plt.xlabel(r'$\psi \cos(\theta)$')
    plt.ylabel(r'$\psi \sin(\theta)$')
    plt.tight_layout()
    plt.savefig("./output/background/background-canonical.png")
    plt.clf()

    np.save("./output/background/background", np.hstack((Ns.reshape(-1, 1), phis, phidots)))
    np.savetxt("./output/background/background.txt", np.hstack((Ns.reshape(-1, 1), phis, phidots)))
