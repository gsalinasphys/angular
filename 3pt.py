# Importing packages
import json
import sys

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi'] = 600

location = "/home/gsalinas/GitHub/angular/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransAngular as PyT
import PyTransScripts as PyS

######################################################################

nF, nP = PyT.nF(), PyT.nP() # Loading the number of fields and parameters
with open("./output/setup/params.json", "r") as file:
    params = json.loads(file.readline())    # Loading the values of the parameters
pval = np.array(list(params.values()))  # From dictionary to array
Npoints = 1_000 # Number of total integration steps
back = np.load("./output/background/background.npy")    # Loading background trajectory
back = back[::len(back)//Npoints]   # Reducing the number of total points in 'back' to reduce integration time
Ns =  back.T[0] # Array containing the number of e-folds

Nend = Ns[-1]
Nexit = Nend - 55
k = PyS.kexitN(Nexit, back, pval, PyT)  # Find the value of k which exits the horizon at Nexit

# Calculating the three values of momenta (k1, k2, k3) used in the calculation of the bispectrum
alpha = 0.
beta = 1/3.

k1 = k/2 - beta*k/2.
k2 = k/4*(1+alpha+beta)
k3 = k/4*(1-alpha+beta)
kM = np.min(np.array([k1, k2, k3])) # Smallest k of the three, needed as input for the function 'PyS.ICsBM'

NB = 6.0    # How many e-folds before Nexit to start integration
Nstart, backExitMinus = PyS.ICsBM(NB, kM, back, pval, PyT)  # Finds initial condition for the correlators at Nstart, needed for the integration step later
print(f"3-pt calculation starts at: {Nstart} e-folds")

Nev = Ns[Ns >= Nstart]  # Restricting to Ns after Nstart
tols = np.array([10**-8, 10**-8])   # Integration tolerances, try not to play too much with this
threePt = PyT.alphaEvolve(Nev, k1, k2, k3, backExitMinus, pval, tols, True) # Three-point function integration
Nalpha = threePt[:, 0]  # Array containing the number of e-folds for each resulting point
Pzetas, B = threePt[:, 1:4], threePt[:, 4]  # Curvature Power Spectrum and Bispectrum

fNL = 5.0/6.0*B/(Pzetas[:, 1]*Pzetas[:, 2]  + Pzetas[:, 0]*Pzetas[:, 1] + Pzetas[:, 0]*Pzetas[:, 2])
print(f"fNL at the end of inflation: {fNL[-1]}")

# Plotting fNL 
plt.plot(Nalpha, -fNL, label=f"$N_B = {NB}$", c='k')
plt.ylabel(r'$f_{\rm NL}$', fontsize=24)
plt.xlabel(r'$N$', fontsize=24)
plt.axvline(Nexit, c='gray', linestyle='--')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("./output/3pt/fNL.png")

# Save values of fNL for each N to a file
np.save("output/3pt/fNL", np.vstack([Nalpha, fNL]))