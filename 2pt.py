import json
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
with open("./output/params.json", "r") as file:
    params = json.loads(file.readline())
pval = np.array(list(params.values()))
back = np.load("./output/background.npy")
Ns, phis, phidots =  back.T[0], back.T[1:nF+1], back.T[nF+1:]

Nend = Ns[-1]
Nexit = Nend - 55
iexit = np.argmin(np.abs(Ns - Nexit))
k = PyS.kexitN(Nexit, back, pval, PyT) 

print(f'Horizon exit at N = {Nexit:.3} with k = {k:.3}')
