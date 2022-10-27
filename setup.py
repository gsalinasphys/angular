import pickle

from sympy import Identity, diff, log, symarray

from PyTransport import PyTransSetup

nF, nP = 2, 3  # Number of fields and parameters
f, p = symarray('f', nF), symarray('p', nP)

V = p[0]/2 * p[2]**2 * (f[0]**2 + p[1]*f[1]**2)
G = 6 * p[0] / (1-f[0]**2-f[1]**2)**2 * Identity(2)

# d1logV = diff(log(V), 'f_0')
# d2logV = diff(log(V), 'f_1')
# d11logV = diff(log(V), 'f_0', 2)
# d12logV = diff(diff(log(V), 'f_0'), 'f_1')
# d22logV = diff(log(V), 'f_1', 2)

# with open("./output/setup/G.txt", "wb") as file:
#     pickle.dump(G, file)

# with open("./output/setup/d1logV.txt", "wb") as file:
#     pickle.dump(d1logV, file)

# with open("./output/setup/d2logV.txt", "wb") as file:
#     pickle.dump(d2logV, file)

# with open("./output/setup/d11logV.txt", "wb") as file:
#     pickle.dump(d11logV, file)

# with open("./output/setup/d12logV.txt", "wb") as file:
#     pickle.dump(d12logV, file)

# with open("./output/setup/d22logV.txt", "wb") as file:
#     pickle.dump(d22logV, file)


PyTransSetup.potential(V, nF, nP, False, G)

PyTransSetup.compileName3("Angular", True)
