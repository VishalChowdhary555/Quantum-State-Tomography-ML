import numpy as np

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = {"X": X, "Y": Y, "Z": Z}

def bloch_to_density(rx, ry, rz):
    return 0.5 * (I2 + rx * X + ry * Y + rz * Z)

def density_to_bloch(rho):
    rx = np.real(np.trace(rho @ X))
    ry = np.real(np.trace(rho @ Y))
    rz = np.real(np.trace(rho @ Z))
    return np.array([rx, ry, rz])

def random_pure_state():
    u = np.random.rand()
    v = np.random.rand()

    theta = np.arccos(1 - 2*u)
    phi = 2*np.pi*v

    psi = np.array([
        np.cos(theta/2),
        np.exp(1j*phi)*np.sin(theta/2)
    ])

    return psi

def statevector_to_density(psi):
    return np.outer(psi, np.conj(psi))
