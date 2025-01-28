import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider


c = 1.0


# Define the neural network
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(50, activation='tanh') for _ in range(5)]
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


# Compute the PDE residual
def wave_pde(x, t, model):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        u = model(tf.concat([x, t], axis=1))
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)
        u_tt = tape.gradient(u_t, t)
    del tape
    return u_tt - c ** 2 * u_xx


# Loss function
def loss_fn(model, X_train, boundary_condition, pde_residual, X_data, u_data):
    u_pred = model(X_train)  # Predicted solution
    u_bc, u_bc_pred = boundary_condition  # Boundary condition and predicted BC values
    residual = pde_residual  # PDE residual

    # Data loss: difference between predicted and actual data
    u_data_pred = model(X_data)
    loss_data = tf.reduce_mean((u_data - u_data_pred) ** 2)

    # Loss as the mean squared error of boundary conditions and PDE residual
    loss_bc = tf.reduce_mean((u_bc - u_bc_pred) ** 2)
    loss_pde = tf.reduce_mean(residual ** 2)
    return loss_bc, loss_pde, loss_data


# Generate data points for the analytical solution
def generate_data(n_points, t_val, model=None):
    x_vals = np.linspace(0, 1, n_points)
    x_flat = x_vals.flatten()
    t_flat = np.full_like(x_flat, t_val)

    X_plot = np.stack([x_flat, t_flat], axis=1)

    if model:
        u_pred_flat = model(tf.convert_to_tensor(X_plot, dtype=tf.float32)).numpy().flatten()
        u_pred = u_pred_flat.reshape(n_points)
    else:
        u_pred = None

    # Ground truth wave function (analytical solution)
    u_actual = np.sin(5 * np.pi * x_flat) * np.cos(5 * c * np.pi * t_val) +  2 * np.sin(7 * np.pi * x_flat) * np.cos(7 * c * np.pi * t_val)

    return x_vals, u_actual, u_pred


# Plot and visualize
def visualize_wave(t_val):
    n_points = 100
    x_vals, u_actual, u_pred = generate_data(n_points, t_val, model)

    plt.figure(figsize=(12, 5))

    # Plot actual solution
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, u_actual, label="Actual", color="blue")
    plt.title(f"Actual Wave Function (t={t_val:.2f})")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.legend()

    # Plot predicted solution
    plt.subplot(1, 2, 2)
    if u_pred is not None:
        plt.plot(x_vals, u_pred, label="Predicted", color="orange")
        plt.title(f"Predicted Wave Function (t={t_val:.2f})")
        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Model not trained", ha="center", va="center", fontsize=12)
        plt.title(f"Predicted Wave Function (t={t_val:.2f})")

    plt.show()


# Slider to change t
t_slider = FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description='t')

# Training data
X_train = tf.convert_to_tensor(np.random.rand(1000, 2), dtype=tf.float32)  # Random points for (x, t)
u_bc = tf.convert_to_tensor(np.zeros((1000, 1)), dtype=tf.float32)  # Example BC values

# Example data points (ground truth)
n_data_points = 1000
X_data = np.random.rand(n_data_points, 2)  # Randomly sampled data points
u_data = np.sin(5 * np.pi * X_data[:, 0:1]) * np.cos(5 * c * np.pi * X_data[:, 1:2]) + 2 * np.sin(7 * np.pi * X_data[:, 0:1]) * np.cos(7 * c * np.pi * X_data[:, 1:2])  # Analytical solution

X_data = tf.convert_to_tensor(X_data, dtype=tf.float32)
u_data = tf.convert_to_tensor(u_data, dtype=tf.float32)

# Define model and optimizer
model = PINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Training loop with loss tracking
loss_bc_values = []
loss_pde_values = []
loss_data_values = []
total_loss_values = []

epochs = 1000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        boundary_condition = (u_bc, model(X_train))
        pde_residual = wave_pde(X_train[:, 0:1], X_train[:, 1:2], model)
        loss_bc, loss_pde, loss_data = loss_fn(model, X_train, boundary_condition, pde_residual, X_data, u_data)
        total_loss = loss_bc + loss_pde + loss_data
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_bc_values.append(loss_bc.numpy())
    loss_pde_values.append(loss_pde.numpy())
    loss_data_values.append(loss_data.numpy())
    total_loss_values.append(total_loss.numpy())

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}, Loss BC: {loss_bc.numpy()}, Loss PDE: {loss_pde.numpy()}, Loss Data: {loss_data.numpy()}, Total Loss: {total_loss.numpy()}")

print("Training completed.")

plt.figure(figsize=(8, 5))
plt.plot(loss_bc_values, label="Boundary Condition Loss")
plt.plot(loss_pde_values, label="PDE Residual Loss")
plt.plot(loss_data_values, label="Data Loss")
plt.plot(total_loss_values, label="Total Loss")
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Loss (log scale)")
plt.legend()
plt.title("Loss Evolution During Training")
plt.show()

interact(visualize_wave, t_val=t_slider)


def generate_heatmap_data(n_points, model=None):
    x_vals = np.linspace(0, 1, n_points)
    t_vals = np.linspace(0, 1, n_points)
    x_grid, t_grid = np.meshgrid(x_vals, t_vals)
    x_flat = x_grid.flatten()
    t_flat = t_grid.flatten()

    X_plot = np.stack([x_flat, t_flat], axis=1)

    if model:
        u_pred_flat = model(tf.convert_to_tensor(X_plot, dtype=tf.float32)).numpy().flatten()
        u_pred = u_pred_flat.reshape(n_points, n_points)
    else:
        u_pred = None

    u_actual_flat = np.sin(5 * np.pi * x_flat) * np.cos(5 * c * np.pi * t_flat) + 2 * np.sin(7 * np.pi * x_flat) * np.cos(7 * c * np.pi * t_flat)
    u_actual = u_actual_flat.reshape(n_points, n_points)

    return t_vals, x_vals, u_actual, u_pred


# Heatmap visualization function
def visualize_heatmap():
    n_points = 100
    t_vals, x_vals, u_actual, u_pred = generate_heatmap_data(n_points, model)

    plt.figure(figsize=(12, 5))

    # Plot actual solution
    plt.subplot(1, 2, 1)
    plt.contourf(t_vals, x_vals, u_actual, cmap="viridis")
    plt.colorbar(label="u(x, t)")
    plt.title("Actual Wave Function")
    plt.xlabel("t")
    plt.ylabel("x")

    # Plot predicted solution
    plt.subplot(1, 2, 2)
    if u_pred is not None:
        plt.contourf(t_vals, x_vals, u_pred, cmap="viridis")
        plt.colorbar(label="u(x, t)")
        plt.title("Predicted Wave Function")
        plt.xlabel("t")
        plt.ylabel("x")
    else:
        plt.text(0.5, 0.5, "Model not trained", ha="center", va="center", fontsize=12)
        plt.title("Predicted Wave Function")

    plt.show()


visualize_heatmap()