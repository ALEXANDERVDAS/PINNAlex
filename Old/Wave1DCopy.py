#%%
from typing import Tuple

from Old.Models import create_dense_model, WavePinn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


c = 1.0
length = 1.0
n_samples = 512

def f_u(tx):
    t = tx[:, 0:1]
    x = tx[:, 1:2]
    return tf.sin(5 * np.pi * x) * tf.cos(5 * c * np.pi * t) + \
        2*tf.sin(7 * np.pi * x) * tf.cos(7 * c * np.pi * t)

def f_u_init(tx):
    x = tx[:, 1:2]
    return tf.sin(5 * np.pi * x) + 2*tf.sin(7 * np.pi * x)

def f_du_dt(tx):
    return tf.zeros_like(tx[:, 0:1])

def f_u_bnd(tx):
    return tf.zeros_like(tx[:, 1:2])


def simulate_wave(n_samples, phi_function, psi_function, boundary_function, x_start=0.0, length=1.0, time=1.0,
                  n_init=None, n_bndry=None,
                  random_seed=42, dtype=tf.float32) -> Tuple[
    Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    Tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the wave equation in 1D or 2D with a given initial condition and Dirichlet boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        dimension (int): dimension of the wave equation. Either 1 or 2.
        phi_function (function): Function that returns the initial condition of the wave equation on u.
        psi_function (function): Function that returns the initial condition of the wave equation on u_t.
        boundary_function_start (function): Function that returns the boundary condition of the wave equation on u at the start of the domain.
        boundary_function_end (function): Function that returns the boundary condition of the wave equation on u at the end of the domain.
        x_start (float, optional): Start of the domain. Defaults to 0.
        length (float, optional): Length of the domain. Defaults to 1.
        time (float, optional): Time frame of the simulation. Defaults to 1.
        n_init (int, optional): number of initial samples to generate. If None, then n_init = n_samples. Defaults to None.
        n_bndry (int, optional): number of boundary samples to generate. If None, then n_bndry = n_samples. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]: Samples of the wave equation. \
            Returns a tuple of tensors (equation_samples, initial_samples, boundary_samples).

    """
    if n_init is None:
        n_init = n_samples
    if n_bndry is None:
        n_bndry = n_samples

    assert n_bndry % 2 == 0, "n_bndry must be even"

    t = tf.random.uniform((n_samples, 1), 0, time, dtype=dtype, seed=random_seed)
    x = tf.random.uniform((n_samples, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    tx_eqn = tf.concat((t, x), axis=1)
    y_eqn = tf.zeros((n_samples, 1), dtype=dtype)

    t_init = tf.zeros((n_init, 1))
    x_init = tf.random.uniform((n_init, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    tx_init = tf.concat((t_init, x_init), axis=1)

    t_boundary = tf.random.uniform((n_bndry, 1), 0, time, dtype=dtype, seed=random_seed)
    x_boundary = tf.ones((n_bndry // 2, 1), dtype=dtype) * (x_start + length)
    x_boundary = tf.concat([x_boundary, tf.ones((n_bndry // 2, 1), dtype=dtype) * x_start], axis=0)
    x_boundary = tf.random.shuffle(x_boundary, seed=random_seed)
    tx_boundary = tf.concat([t_boundary, x_boundary], axis=1)

    y_phi = phi_function(tx_init)
    y_psi = psi_function(tx_init)
    y_boundary = boundary_function(tx_boundary)

    return (tx_eqn, y_eqn), (tx_init, y_phi, y_psi), (tx_boundary, y_boundary)


# (tx_samples, residual), (tx_init, u_init, du_dt_init), (tx_bndry, u_bndry) = simulate_wave(n_samples, f_u_init, f_du_dt, f_u_bnd, n_bndry=256, n_init=256)
(tx_samples, residual), (tx_init, u_init, du_dt_init), (tx_bndry, u_bndry) = simulate_wave(n_samples, f_u_init, f_du_dt, f_u_bnd)



inputs = [tx_samples, tx_init, tx_bndry]
outputs = [f_u(tx_samples), residual, u_init, du_dt_init, u_bndry]

backbone = create_dense_model([512]*3, 'tanh', 'GlorotNormal', n_inputs=2, n_outputs=1)
pinn = WavePinn(backbone, c)
scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 1000, 0.90)
optimizer = tf.keras.optimizers.Adam(scheduler)
pinn.compile(optimizer=optimizer)

history = pinn.fit(inputs, outputs, epochs=10000, batch_size=128, verbose=2)
# history = pinn.fit_custom(inputs, outputs, epochs=2000)



def plot_heatmap(model, title=""):
    n_points = 100
    t_vals = np.linspace(0, 1, n_points)
    x_vals = np.linspace(0, length, n_points)
    t_grid, x_grid = np.meshgrid(t_vals, x_vals)
    t_flat = t_grid.flatten()
    x_flat = x_grid.flatten()
    tx_flat = np.stack([t_flat, x_flat], axis=1)

    u_pred = model.predict(tx_flat, batch_size=512).reshape(n_points, n_points)
    u_actual = f_u(tf.convert_to_tensor(tx_flat)).numpy().reshape(n_points, n_points)

    plt.figure(figsize=(12, 5))

    # Plot Actual Solution
    plt.subplot(1, 2, 1)
    plt.contourf(t_vals, x_vals, u_actual, cmap="viridis")
    plt.colorbar(label="u(t, x)")
    plt.title(f"Actual Wave Function {title}")
    plt.xlabel("Time (t)")
    plt.ylabel("Space (x)")

    # Plot Predicted Solution
    plt.subplot(1, 2, 2)
    plt.contourf(t_vals, x_vals, u_pred, cmap="viridis")
    plt.colorbar(label="u(t, x)")
    plt.title(f"Predicted Wave Function {title}")
    plt.xlabel("Time (t)")
    plt.ylabel("Space (x)")

    plt.show()


# Training Data Points Visualization
def plot_data_points(tx_samples, tx_init, tx_bndry):
    plt.figure(figsize=(8, 6))
    plt.scatter(tx_samples[:, 0], tx_samples[:, 1], s=10, label="Equation Samples")
    plt.scatter(tx_init[:, 0], tx_init[:, 1], s=20, label="Initial Condition Samples")
    plt.scatter(tx_bndry[:, 0], tx_bndry[:, 1], s=20, label="Boundary Condition Samples")
    plt.xlabel("Time (t)")
    plt.ylabel("Space (x)")
    plt.title("Data Points Used for Training")
    plt.legend()
    plt.show()


# # Loss Evolution Plot
# def plot_loss(history):
#     plt.figure(figsize=(8, 6))
#     plt.plot(history.history['mean_absolute_error'], label="Total Loss")
#     plt.yscale("log")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss (log scale)")
#     plt.title("Loss Evolution During Training")
#     plt.legend()
#     plt.show()


def plot_loss(history):
    """
    Plots the loss evolution during training, including all loss components and mean absolute error.
    """
    plt.figure(figsize=(10, 6))

    # Plot residual loss
    plt.plot(history.history['loss_residual'], label="Residual Loss", linestyle="--")

    # Plot initial condition loss
    plt.plot(history.history['loss_initial'], label="Initial Condition Loss", linestyle="-.")

    # Plot boundary condition loss
    plt.plot(history.history['loss_boundary'], label="Boundary Condition Loss", linestyle=":")

    # Plot mean absolute error
    plt.plot(history.history['mean_absolute_error'], label="Total Loss", linestyle="-")

    # Set log scale for better visualization
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.title("Loss Evolution During Training")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

# Actual vs Predicted for Fixed Time
def plot_fixed_time(model, t_fixed):
    n_points = 100
    x_vals = np.linspace(0, length, n_points)
    tx_fixed = np.stack([np.full_like(x_vals, t_fixed), x_vals], axis=1)

    u_actual = f_u(tf.convert_to_tensor(tx_fixed)).numpy()
    u_pred = model.predict(tx_fixed, batch_size=512)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, u_actual, label="Actual", color="blue")
    plt.plot(x_vals, u_pred, label="Predicted", color="orange")
    plt.title(f"Wave Function at t={t_fixed}")
    plt.xlabel("Space (x)")
    plt.ylabel("u(t, x)")
    plt.legend()
    plt.show()


# Visualizations
plot_heatmap(pinn.backbone, title="After Training")
plot_data_points(tx_samples.numpy(), tx_init.numpy(), tx_bndry.numpy())
plot_loss(history)
plot_fixed_time(pinn.backbone, t_fixed=0.0)
plot_fixed_time(pinn.backbone, t_fixed=0.3)
plot_fixed_time(pinn.backbone, t_fixed=0.6)



