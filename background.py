import sys
from math import cos, pi, sin, sqrt

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

location = "/home/gsalinas/GitHub/angular/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransAngular as PyT
import PyTransScripts as PyS

nF, nP = PyT.nF(), PyT.nP()
pval = np.array([1/600, 9, 2e-5])   # Parameters [alpha, R, mphi]

r0, theta0 = 0.99, pi/4
phi0 = r0 * np.array([cos(theta0), sin(theta0)])
phidot0 = np.zeros(2)
initial = np.concatenate((phi0, phidot0))

Nstart, Nend = 0., 100
Nsteps = 1_000_000
Nrange = np.linspace(Nstart, Nend, Nsteps)

tols = np.array([10**-12, 10**-12])
back = PyT.backEvolve(Nrange, initial, pval, tols, True)
Ns, phis, phidots =  back.T[0], back.T[1:nF+1], back.T[nF+1:]
rs = np.sqrt(phis[0]**2 + phis[1]**2)
thetas = np.arctan(phis[1]/phis[0])
psis = sqrt(6*pval[0]) * np.arctanh(rs)

Nend = Ns[-1]
Nexit = Nend - 55
iexit = np.argmin(np.abs(Ns - Nexit))
print(f'Number of e-folds: {Nend:.3}')

palette = sns.color_palette("crest", as_cmap=True)
num_points = 10_000
sns.scatterplot(x=phis[0][::Nsteps//num_points],
                y=phis[1][::Nsteps//num_points],
                hue=Ns[::Nsteps//num_points],
                s=5,
                palette=palette)
plt.scatter(phis[0][iexit], phis[1][iexit], c="k")
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\chi$')
plt.tight_layout()
plt.savefig("./output/background.png")
plt.clf()

sns.scatterplot(x=rs[::Nsteps//num_points],
                y=thetas[::Nsteps//num_points],
                hue=Ns[::Nsteps//num_points],
                s=5,
                palette=palette)
plt.scatter(rs[iexit], thetas[iexit], c="k")
plt.xlabel(r'$\psi \cos(\theta)$')
plt.ylabel(r'$\psi \sin(\theta)$')
plt.tight_layout()
plt.savefig("./output/background-polar.png")
plt.clf()

sns.scatterplot(x=(psis*np.cos(thetas))[::Nsteps//num_points],
                y=(psis*np.sin(thetas))[::Nsteps//num_points],
                hue=Ns[::Nsteps//num_points],
                s=5,
                palette=palette)
plt.scatter((psis*np.cos(thetas))[iexit], (psis*np.sin(thetas))[iexit], c="k")
plt.xlabel(r'$\psi \cos(\theta)$')
plt.ylabel(r'$\psi \sin(\theta)$')
plt.tight_layout()
plt.savefig("./output/background-canonical.png")
plt.clf()

np.save("./output/background", back)
