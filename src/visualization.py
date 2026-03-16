import matplotlib.pyplot as plt
import numpy as np

def plot_density_matrix(rho,title="Density Matrix"):

    fig,axes = plt.subplots(1,2,figsize=(8,4))

    axes[0].imshow(np.real(rho),cmap="coolwarm")
    axes[0].set_title(title+" (Real)")

    axes[1].imshow(np.imag(rho),cmap="coolwarm")
    axes[1].set_title(title+" (Imag)")

    plt.show()
