import json
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

nF, nP = PyT.nF(), PyT.nP()
with open("./output/setup/params.json", "r") as file:
    params = json.loads(file.readline())
pval = np.array(list(params.values()))

phisexit = np.load("./output/background/exit.npy")
print(phisexit)
initials = [phisexit, phisexit + np.array([1.01*phisexit[0], 0., 0., 0.], phisexit + np.array([0., 1.01*phisexit[1], 0., 0.]))]

Nstart, Nend = 0., 100
Nsteps = 500_000
Nrange = np.linspace(Nstart, Nend, Nsteps)

tols = np.array([10**-12, 10**-12])
all_Ns, all_phis, all_phidots = [], [], []
labels = ["Original", r"Perturbed in $\phi$ (10%)", r"Perturbed in $\chi$ (10%)"]
for initial in initials:
    back = PyT.backEvolve(Nrange, initial, pval, tols, True)
    Ns, phis, phidots =  back.T[0], back.T[1:nF+1], back.T[nF+1:]

    all_Ns.append(Ns)
    all_phis.append(phis)
    all_phidots.append(phidots)

palette = sns.color_palette("crest", as_cmap=True)
num_points = 500
for phis in all_phis:
    sns.scatterplot(x=phis[0][::Nsteps//num_points],
                    y=phis[1][::Nsteps//num_points],
                    hue=Ns[::Nsteps//num_points],
                    s=5,
                    palette=palette)
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\chi$')
plt.tight_layout()
plt.savefig("./output/perturbed/from_exit.png")
plt.clf()
