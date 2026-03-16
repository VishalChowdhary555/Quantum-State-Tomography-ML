import numpy as np
from quantum_utils import PAULIS, density_to_bloch, random_pure_state, statevector_to_density

def projective_measurement_probs(rho, basis):

    exp_val = np.real(np.trace(rho @ PAULIS[basis]))

    p_plus = (1 + exp_val) / 2
    p_minus = 1 - p_plus

    return p_plus, p_minus


def simulate_measurements(rho, shots=1000):

    results = {}
    features = []

    for basis in ["X","Y","Z"]:

        p_plus, p_minus = projective_measurement_probs(rho,basis)

        counts = np.random.multinomial(shots,[p_plus,p_minus])

        n_plus,n_minus = counts

        exp_est = (n_plus-n_minus)/shots

        results[basis] = {

            "n_plus":n_plus,
            "n_minus":n_minus,
            "exp_est":exp_est

        }

        features.extend([n_plus/shots,n_minus/shots,exp_est])

    return results, np.array(features)


def generate_dataset(n_samples=2000):

    X_data = []
    y_data = []

    for i in range(n_samples):

        psi = random_pure_state()
        rho = statevector_to_density(psi)

        _,features = simulate_measurements(rho)

        X_data.append(features)

        y_data.append(density_to_bloch(rho))

    return np.array(X_data), np.array(y_data)
