import math

import tensorflow as tf
# from Compute_Jacobian import jacobian # Please download 'Compute_Jacobian.py' in the repository
import numpy as np

from matplotlib import colors
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from Wave1D.WavePINN import Sampler
from BurgersPINN import BurgersPINN
import scipy.io


# Define the exact solution and its derivatives
# def u(x, a, v): # Real solution not known for Burgers.
#     """
#     :param x: x = (t, x)
#     """
#     t = x[:,0:1]
#     x = x[:,1:2]
#     return 2 / (1 + (np.exp((x - t) / v)))

def ics(x, a, v):
    t = x[:,0:1]
    x = x[:,1:2]
    return -np.sin(np.pi * x)

def bnd(x, a, v):
    return x[:, 1:2] * 0

def r(x, a, c):
    return x[:, 1:2] * 0

def operator(u, t, x, c, sigma_t=1.0, sigma_x=1.0):
    u_t = tf.gradients(u, t)[0] / sigma_t
    u_x = tf.gradients(u, x)[0] / sigma_x
    u_xx = tf.gradients(u_x, x)[0] / sigma_x
    residual = u_t + u * u_x - c * u_xx # Burgers!!!
    return residual


tf.compat.v1.disable_eager_execution()

ics_coords = np.array([[0.0, -1.0],
                        [0.0, 1.0]])
bc1_coords = np.array([[0.0, -1.0],
                        [1.0, -1.0]])
bc2_coords = np.array([[0.0, 1.0],
                        [1.0, 1.0]])
dom_coords = np.array([[0.0, -1.0],
                        [1.0, 1.0]])

# Create initial conditions samplers
ics_sampler = Sampler(2, ics_coords, lambda x: ics(x, a, v), name='Initial Condition 1')

# Create boundary conditions samplers
bc1 = Sampler(2, bc1_coords, lambda x: bnd(x, a, v), name='Dirichlet BC1')
bc2 = Sampler(2, bc2_coords, lambda x: bnd(x, a, v), name='Dirichlet BC2')
bcs_sampler = [bc1, bc2]

# Create residual sampler
res_sampler = Sampler(2, dom_coords, lambda x: r(x, a, v), name='Forcing')

# Define PINN model
a = 2
v = 0.01/math.pi

layers = [2, 500, 500, 500, 1] # Usually 3 layers of 500
kernel_size = 300
model = BurgersPINN(layers, operator, ics_sampler, bcs_sampler, res_sampler, v, kernel_size)
# Train model
itertaions = 1


model.train(nIter=itertaions, batch_size=512) # Maybe a little lower... (Normal is 512) (4 batch size 10x faster so do 10x as many epochs (500000))

# data = scipy.io.loadmat("BurgersData")
# x = data['x']
# t = data['t']
# # create a meshgrid
# X, T = np.meshgrid(x, t)
# # flatten the meshgrid
# tx_train = np.hstack((T.flatten()[:,None], X.flatten()[:,None]))
# u_train = data['usol'].T.flatten()[:,None]
# convert to tf
# tx_train = tf.convert_to_tensor(tx_train, dtype=tf.float32)
# u_train = tf.convert_to_tensor(u_train, dtype=tf.float32)

#Visualize


# for layer in model.layers:
#     weights, biases = layer.get_weights()  # Get weights and biases
#     print(f"Layer: {layer.name}")
#     print(f"Weights:\n{weights}")
#     print(f"Biases:\n{biases}\n")


def plot_fixed_time(model, t_fixed):
    n_points = 256
    # print("size = "+str(u_actual.size))

    data = scipy.io.loadmat("BurgersData")
    u_actual = data['usol'].T[int(t_fixed*100)]


    x_vals = np.linspace(-1, 1, n_points)
    tx_fixed = np.stack([np.full_like(x_vals, t_fixed), x_vals], axis=1)

    # X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))
    # u_actual = u(tx_fixed, a, v)
    u_pred = model.predict_u(tx_fixed)



    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, u_actual, label="Actual", color="blue")
    plt.plot(x_vals, u_pred, label="Predicted", color="orange")
    plt.title(f"Wave Function at t={t_fixed}")
    plt.xlabel("Space (x)")
    plt.ylabel("u(t, x)")
    plt.legend()
    plt.show()


def plot_loss_iterations(model):
    loss_res = model.loss_res_log
    loss_bcs = model.loss_bcs_log
    # loss_u_t_ics = model.loss_ut_ics_log
    loss_ics = model.loss_ics_log
    total_loss = model.total_loss_log

    fig = plt.figure(figsize=(6, 5))
    plt.plot(total_loss, label='$\mathcal{L}_{total}$')
    plt.plot(loss_res, label='$\mathcal{L}_{residual}$')
    plt.plot(loss_bcs, label='$\mathcal{L}_{boundary}$')
    # plt.plot(loss_u_t_ics, label='$\mathcal{L}_{u_t}$')
    plt.plot(loss_ics, label='$\mathcal{L}_{initial}$')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_squared_mean(model):
    squared = model.squared_mean_height_solution
    means = []
    for list in squared:
        templist = []
        for val in list:
            templist.append(val[0])

            # print(val)
        means.append(sum(templist) / len(templist))


    fig = plt.figure(figsize=(6, 5))
    plt.plot(means, label='average height')
    plt.title('Average Height of the prediction. (Red Line is real height)')
    plt.axhline(y=1.25, color='r', linestyle='-') #This complexity for c = 0.5 (See CalcMeanHeight.py)
    plt.xlabel('iterations x100')
    plt.ylabel('Height (squared mean of predicted graph)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_heat_map(model):
    # nn = 200
    # t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    # x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    # t, x = np.meshgrid(t, x)
    data = scipy.io.loadmat("BurgersData")
    X = data['x']
    T = data['t']
    # create a meshgrid
    t, x = np.meshgrid(T, X)
    X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

    # u_star = u(X_star, a, v)
    u_star = data['usol'].flatten()[:,None]
    R_star = r(X_star, a, v)

    # Predictions
    u_pred = model.predict_u(X_star)
    r_pred = model.predict_r(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

    print('Relative L2 error_u: %e' % (error_u))

    U_star = griddata(X_star, u_star.flatten(), (t, x), method='cubic')
    R_star = griddata(X_star, R_star.flatten(), (t, x), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')
    R_pred = griddata(X_star, r_pred.flatten(), (t, x), method='cubic')



    plt.figure(figsize=(18, 9))
    plt.subplot(2, 3, 1)
    u_divnorm = colors.TwoSlopeNorm(vcenter=0.)
    plt.pcolor(t, x, U_star, cmap='jet', norm=u_divnorm)


    plt.colorbar()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Exact u(t, x)')
    plt.tight_layout()

    plt.subplot(2, 3, 2)
    plt.pcolor(t, x, U_pred, cmap='jet', norm=u_divnorm)
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Predicted u(t, x)')
    plt.tight_layout()

    plt.subplot(2, 3, 3)
    plt.pcolor(t, x, np.abs(U_star - U_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Absolute error')
    plt.tight_layout()

    plt.subplot(2, 3, 4)
    plt.pcolor(t, x, R_star, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Exact r(t, x)')
    plt.tight_layout()

    plt.subplot(2, 3, 5)
    r_divnorm = colors.TwoSlopeNorm(vcenter=0.)
    plt.pcolor(t, x, R_pred, cmap='jet', norm=r_divnorm)
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Predicted r(t, x)')
    plt.tight_layout()

    plt.subplot(2, 3, 6)
    plt.pcolor(t, x, np.abs(R_star - R_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.show()

plot_squared_mean(model)
plot_fixed_time(model, 0)
plot_fixed_time(model, 0.3)
plot_fixed_time(model, 0.6)
plot_fixed_time(model, 0.9)
plot_loss_iterations(model)
plot_heat_map(model)


