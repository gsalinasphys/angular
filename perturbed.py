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
params = {'alpha': 1/600, 'R': 9, 'mphi': 2.e-5}
with open("./output/setup/params.json", "w") as file:
    file.write(json.dumps(params))
pval = np.array(list(params.values()))   # Parameters [alpha, R, mphi]

r0, theta0 = 0.99, pi/4
phi0 = r0 * np.array([cos(theta0), sin(theta0)])
phidot0 = np.zeros(2)
initial = np.concatenate((phi0, phidot0))

Nstart, Nend = 0., 100
Nsteps = 500_000
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
print(phis.T[iexit], phidots.T[iexit])
