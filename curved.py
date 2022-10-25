from itertools import product

import numpy as np


def dotG(Gmatrix, v1, v2):
    return np.matmul(v1, np.matmul(Gmatrix, v2))

def magG(Gmatrix, v):
    return np.sqrt(dotG(Gmatrix, v, v))

def epll(Gmatrix, phidot):
    return phidot / magG(Gmatrix, phidot)

def eperp2d(Gmatrix, phidot, eta):
    epll_vec = epll(Gmatrix, phidot)
    eperp_notnorm = np.matmul(np.identity(2) - np.outer(epll_vec, np.matmul(Gmatrix, epll_vec)), eta)
    return eperp_notnorm / magG(Gmatrix, eperp_notnorm)

# def eperp2d_2(Gmatrix, Gammamatrix, phidot, phiprimes, eta, dN):
#     epll_vec = epll(Gmatrix, phidot)
#     eperp_notnorm = np.gradient(epll_vec, dN) + sum([Gammamatrix[:, bb, cc] * phiprimes[bb] * epll_vec[cc] for bb, cc in product(range(2), repeat=2)])
#     return eperp_notnorm / magG(Gmatrix, eperp_notnorm)
