import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the wave speed
c = 1.0


# Define the neural network
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(32, activation='tanh') for _ in range(4)]
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


# Compute the PDE residual
def wave_pde(x, y, t, model):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, y, t])
        u = model(tf.concat([x, y, t], axis=1))
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        u_tt = tape.gradient(u_t, t)
    del tape
    return u_tt - c ** 2 * (u_xx + u_yy)


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


# Example training data
X_train = tf.convert_to_tensor(np.random.rand(1000, 3), dtype=tf.float32)  # Random points for (x, y, t)
u_bc = tf.convert_to_tensor(np.zeros((1000, 1)), dtype=tf.float32)  # Example BC values

# Example data points (ground truth)
n_data_points = 5000
X_data = np.random.rand(n_data_points, 3)  # Randomly sampled data points
u_data = np.sin(np.pi * X_data[:, 0:1]) * np.sin(np.pi * X_data[:, 1:2]) * np.cos(
    np.pi * X_data[:, 2:3])  # Analytical solution

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
        pde_residual = wave_pde(X_train[:, 0:1], X_train[:, 1:2], X_train[:, 2:3], model)
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

# Generate predictions for visualization
n_points = 100
x_vals = np.linspace(0, 1, n_points)
y_vals = np.linspace(0, 1, n_points)

# t_val = 0.2  # Fixed time slice for visualization
for t in range (11):
    t_val = t/10.0

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    t_flat = np.full_like(x_flat, t_val)



    X_plot = np.stack([x_flat, y_flat, t_flat], axis=1)
    u_pred_flat = model(tf.convert_to_tensor(X_plot, dtype=tf.float32)).numpy().flatten()

    # Ground truth wave function (example sine wave solution)
    u_actual_flat = np.sin(np.pi * x_flat) * np.sin(np.pi * y_flat) * np.cos(np.pi * t_val)

    # Reshape for plotting
    u_pred = u_pred_flat.reshape(n_points, n_points)
    u_actual = u_actual_flat.reshape(n_points, n_points)

    # Plot actual vs predicted wave function
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.contourf(x_grid, y_grid, u_actual, cmap="viridis")
    plt.colorbar()
    plt.title("Actual Wave Function")

    plt.subplot(1, 2, 2)
    plt.contourf(x_grid, y_grid, u_pred, cmap="viridis")
    plt.colorbar()
    plt.title("Predicted Wave Function")

    plt.show()

# Plot loss evolution
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

# Plot data points used in training
plt.figure(figsize=(8, 6))
plt.scatter(X_data[:, 0], X_data[:, 1], c=X_data[:, 2], cmap="coolwarm", s=30, edgecolor="k", label="Data Points")
plt.colorbar(label="Time (t)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Data Points Used for Training")
plt.legend()
plt.show()
