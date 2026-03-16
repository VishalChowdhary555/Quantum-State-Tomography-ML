import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from quantum_utils import bloch_to_density


def train_nn_model(X_data,y_data):

    X_train,X_test,y_train,y_test = train_test_split(
        X_data,y_data,test_size=0.2
    )

    model = MLPRegressor(

        hidden_layer_sizes=(128,128),

        activation="relu",

        max_iter=500

    )

    model.fit(X_train,y_train)

    return model,X_test,y_test


def nn_reconstruct_density(model,features):

    bloch = model.predict(features.reshape(1,-1))[0]

    norm = np.linalg.norm(bloch)

    if norm > 1:
        bloch = bloch/norm

    rho = bloch_to_density(*bloch)

    return rho,bloch
