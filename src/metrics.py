import numpy as np
from scipy.linalg import sqrtm

def fidelity(rho, sigma):

    sr = sqrtm(rho)

    middle = sr @ sigma @ sr

    sm = sqrtm(middle)

    return np.real(np.trace(sm))**2


def purity(rho):

    return np.real(np.trace(rho @ rho))
