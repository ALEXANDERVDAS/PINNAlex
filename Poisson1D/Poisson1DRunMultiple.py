import tensorflow.compat.v1 as tf
import numpy as np
import timeit
from scipy.interpolate import griddata
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

from VisualizePoisson import *
from PoissonPINN import PoissonPINN

tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.FATAL)

def u(x, a):
  return np.sin(np.pi * a * x)

def u_xx(x, a):
  return -(np.pi * a)**2 * np.sin(np.pi * a * x)



#Constants

bc1_coords = np.array([[0.0],
                       [0.0]])

bc2_coords = np.array([[1.0],
                       [1.0]])

dom_coords = np.array([[0.0],
                       [1.0]])

frequencies = np.linspace(0, 40, 21)  # Frequencies to test (0, 40, 21)
layers = [1, 512, 1]  # Network architecture
nn = 100  # Number of training points
runs_per_frequency = 3  # Runs per frequency

results = {"Frequency (a)": [], "Average L2 Loss u": [], "Average L2 Loss r": []}

# Run the experiment
for a in frequencies:
    error_u_list = []
    error_r_list = []

    for run in range(runs_per_frequency):
        # Generate training data
        X_bc1 = dom_coords[0, 0] * np.ones((nn // 2, 1))
        X_bc2 = dom_coords[1, 0] * np.ones((nn // 2, 1))
        X_u = np.vstack([X_bc1, X_bc2])
        Y_u = u(X_u, a)

        X_r = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
        Y_r = u_xx(X_r, a)

        # Train PINN
        model = PoissonPINN(layers, X_u, Y_u, X_r, Y_r)
        print(f"Running experiment for a={a}, run={run}")
        model.train(iter=100000) #100000 Normal
        # model.
        # Evaluate L2 losses
        u_pred = model.predict_u(X_r)
        r_pred = model.predict_r(X_r)
        np.nan_to_num(u_pred, copy=False, nan=1.0e8, posinf=1.0e9, neginf=0.0)
        np.nan_to_num(r_pred, copy=False, nan=1.0e8, posinf=1.0e9, neginf=0.0)

        u_star = u(X_r, a)
        r_star = u_xx(X_r, a)

        error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
        error_r = np.linalg.norm(r_star - r_pred, 2) / np.linalg.norm(r_star, 2)

        error_u_list.append(error_u)
        error_r_list.append(error_r)
        tf.keras.backend.clear_session()
        # plot_loss_iterations(model)
        # plot_actual_and_loss(model)

    # Store average errors for this frequency
    results["Frequency (a)"].append(a)
    results["Average L2 Loss u"].append(np.mean(error_u_list))
    results["Average L2 Loss r"].append(np.mean(error_r_list))

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("Other_results_pinn_frequency_Adagrad.csv", index=False)

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(results["Frequency (a)"], results["Average L2 Loss u"], label='L2 Loss u', marker='o')
plt.plot(results["Frequency (a)"], results["Average L2 Loss r"], label='L2 Loss r', marker='x')
plt.yscale('log')
plt.xlabel('Frequency (a)')
plt.ylabel('L2 Loss (log scale)')
plt.legend()
plt.title('PINN L2 Loss vs Frequency of PDE')
plt.grid(True)
plt.tight_layout()
plt.savefig("pinn_l2_loss_vs_frequency.png")
plt.show()
