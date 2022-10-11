import numpy as np


def dotG(Gmatrix, v1, v2):
    return np.matmul(v1, np.matmul(Gmatrix, v2))

def magG(Gmatrix, v):
    return np.sqrt(dotG(Gmatrix, v, v))

def epll(Gmatrix, phidot):
    return phidot / magG(Gmatrix, phidot)

def eperp(Gmatrix, phidot):
    epll_vec = epll(Gmatrix, phidot)
    eperp_notnorm = np.array([epll_vec[1], -epll_vec[0]])
    return eperp_notnorm / magG(Gmatrix, eperp_notnorm)