import pickle

from sympy import Identity, symarray

from PyTransport import PyTransSetup

nF, nP = 2, 3  # Number of fields and parameters
f, p = symarray('f', nF), symarray('p', nP)

V = p[0]/2 * p[2]**2 * (f[0]**2 + p[1]*f[1]**2)
G = 6 * p[0] / (1-f[0]**2-f[1]**2)**2 * Identity(2)
with open("./output/setup/G.txt", "wb") as file:
    pickle.dump(G, file)

PyTransSetup.potential(V, nF, nP, False, G)

PyTransSetup.compileName3("Angular", True)
