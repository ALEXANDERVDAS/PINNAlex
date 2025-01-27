import numpy as np

from matplotlib import colors
from scipy.interpolate import griddata

import matplotlib.pyplot as plt



def plot_weights(model):
    for layer in model.layers:
        weights, biases = layer.get_weights()  # Get weights and biases
        print(f"Layer: {layer.name}")
        print(f"Weights:\n{weights}")
        print(f"Biases:\n{biases}\n")


def plot_fixed_time(model, t_fixed, u, a, c):
    n_points = 100
    x_vals = np.linspace(0, 1, n_points)
    tx_fixed = np.stack([np.full_like(x_vals, t_fixed), x_vals], axis=1)

    # X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))
    u_actual = u(tx_fixed, a, c)
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
    loss_u_t_ics = model.loss_ut_ics_log
    loss_ics = model.loss_ics_log
    total_loss = model.total_loss_log

    fig = plt.figure(figsize=(6, 5))
    plt.plot(total_loss, label='$\mathcal{L}_{total}$')
    plt.plot(loss_res, label='$\mathcal{L}_{residual}$')
    plt.plot(loss_bcs, label='$\mathcal{L}_{boundary}$')
    plt.plot(loss_u_t_ics, label='$\mathcal{L}_{u_t}$')
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

def plot_heat_map(model, a, c, dom_coords, u, r):
    nn = 200
    t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
    t, x = np.meshgrid(t, x)
    X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

    u_star = u(X_star, a, c)
    R_star = r(X_star, a, c)

    u_pred = model.predict_u(X_star)
    r_pred = model.predict_r(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)


    U_star = griddata(X_star, u_star.flatten(), (t, x), method='cubic')
    R_star = griddata(X_star, R_star.flatten(), (t, x), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')
    R_pred = griddata(X_star, r_pred.flatten(), (t, x), method='cubic')

    print('Relative L2 error_u: %e' % (error_u))

    plt.figure(figsize=(18, 9))
    plt.subplot(2, 3, 1)
    u_divnorm = colors.TwoSlopeNorm(vcenter=0.)
    plt.pcolor(t, x, U_star, cmap='jet', norm=u_divnorm)
    plt.colorbar().ax.tick_params(labelsize=24)
    plt.xlabel('$t$', fontsize=24)
    plt.ylabel('$x$', fontsize=24)
    # plt.title('Exact u(t, x)')
    plt.tight_layout()
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

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
