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



# Function to run the model for different values of a and batch size (nn)

def run_experiments():
    results = []  # To store results of experiments

    for a in range(1, 9, 1):  # Varying a from 2 to 22 with steps of 4  23
        for nn in range(4, 45, 8):  # Varying nn (batch size) from 10 to 210 with steps of 40



            # Generate training data for this experiment
            X_bc1 = dom_coords[0, 0] * np.ones((nn // 2, 1))
            X_bc2 = dom_coords[1, 0] * np.ones((nn // 2, 1))
            X_u = np.vstack([X_bc1, X_bc2])
            Y_u = u(X_u, a)

            X_r = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
            Y_r = u_xx(X_r, a)

            # Define model
            layers = [1, 512, 1]
            model = PoissonPINN(layers, X_u, Y_u, X_r, Y_r)

            print(f"Running experiment for a={a}, nn={nn}")

            # Train model
            model.train(iter=40001)

            # Log results
            u_pred = model.predict_u(X_r)
            r_pred = model.predict_r(X_r)
            np.nan_to_num(u_pred, copy=False, nan=1.0e8, posinf=1.0e9, neginf=0.0)
            np.nan_to_num(r_pred, copy=False, nan=1.0e8, posinf=1.0e9, neginf=0.0)

            # Calculate errors
            error_u = np.linalg.norm(u(X_r, a) - u_pred, 2) / np.linalg.norm(u(X_r, a), 2)
            error_r = np.linalg.norm(u_xx(X_r, a) - r_pred, 2) / np.linalg.norm(u_xx(X_r, a), 2)

            results.append({
                'a': a,
                'nn': nn,
                'error_u': error_u,
                'error_r': error_r
            })

            print(f"Results for a={a}, nn={nn}: error_u={error_u:.3e}, error_r={error_r:.3e}")

    # Save results to a DataFrame for analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv('experimental_results_batch_vs_error.csv', index=False)
    print("Experiments completed. Results saved to 'experimental_results_batch_vs_error.csv'.")

    return results_df


# Run the experiments
results_df = run_experiments()

# Visualize results
sns.lineplot(data=results_df, x='nn', y='error_u', hue='a', marker='o')
plt.title("Error_u vs Batch Size for Different Frequencies (a)")
plt.xlabel("Batch Size (nn)")
plt.ylabel("Relative L2 Error in u")
plt.yscale('log')
plt.legend(title="Frequency (a)")
plt.show()

sns.lineplot(data=results_df, x='nn', y='error_r', hue='a', marker='o')
plt.title("Error_r vs Batch Size for Different Frequencies (a)")
plt.xlabel("Batch Size (nn)")
plt.ylabel("Relative L2 Error in r")
plt.yscale('log')
plt.legend(title="Frequency (a)")
plt.show()
