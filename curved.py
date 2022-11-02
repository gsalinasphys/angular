from itertools import product

import numpy as np


def dotG(G, v1, v2):
    return np.matmul(v1, np.matmul(G, v2))

def magG(G, v):
    return np.sqrt(dotG(G, v, v))

def epll(G, phidot):
    return phidot / magG(G, phidot)

def eperp2d(G, phidot, eta):
    epll_vec = epll(G, phidot)
    eperp_notnorm = np.matmul(np.identity(2) - np.outer(epll_vec, np.matmul(G, epll_vec)), eta)
    return eperp_notnorm / magG(G, eperp_notnorm)

# def eperp2d(G, phidot, eta):
#     epllv = epll(G, phidot)
#     return np.array([-epllv[1], epllv[0]])

# def eperp2d_2(Gmatrix, Gammamatrix, phidot, phiprimes, eta, dN):
#     epll_vec = epll(Gmatrix, phidot)
#     eperp_notnorm = np.gradient(epll_vec, dN) + sum([Gammamatrix[:, bb, cc] * phiprimes[bb] * epll_vec[cc] for bb, cc in product(range(2), repeat=2)])
#     return eperp_notnorm / magG(Gmatrix, eperp_notnorm)
