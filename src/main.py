from simulation import generate_dataset, simulate_measurements
from quantum_utils import random_pure_state, statevector_to_density, density_to_bloch
from models import train_nn_model, nn_reconstruct_density
from metrics import fidelity
from visualization import plot_density_matrix

from sklearn.metrics import mean_squared_error


def main():

    X_data,y_data = generate_dataset()

    model,X_test,y_test = train_nn_model(X_data,y_data)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test,y_pred)

    print("Test MSE:",mse)

    psi = random_pure_state()

    rho_true = statevector_to_density(psi)

    _,features = simulate_measurements(rho_true)

    rho_pred,bloch_pred = nn_reconstruct_density(model,features)

    print("True Bloch:",density_to_bloch(rho_true))

    print("Predicted Bloch:",bloch_pred)

    print("Fidelity:",fidelity(rho_true,rho_pred))

    plot_density_matrix(rho_true,"True Density Matrix")

    plot_density_matrix(rho_pred,"Predicted Density Matrix")


if __name__ == "__main__":

    main()
