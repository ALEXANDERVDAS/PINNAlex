
import numpy as np
import matplotlib.pyplot as plt


def plot_actual_and_loss(model, u, u_xx, dom_coords, a):
    nn = 1000
    X_star = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
    u_star = u(X_star, a)
    r_star = u_xx(X_star, a)

    # Predictions
    u_pred = model.predict_u(X_star)
    r_pred = model.predict_r(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_r = np.linalg.norm(r_star - r_pred, 2) / np.linalg.norm(r_star, 2)

    print('Relative L2 error_u: {:.2e}'.format(error_u))
    print('Relative L2 error_r: {:.2e}'.format(error_r))

    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X_star, u_star, label='Exact')
    plt.plot(X_star, u_pred, '--', label='Predicted')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(X_star, np.abs(u_star - u_pred), label='Error')
    plt.yscale('log')
    plt.xlabel('$x$')
    plt.ylabel('Point-wise error')
    plt.tight_layout()
    plt.show()

def plot_loss_iterations(model):
    loss_res = model.loss_res_log
    loss_bcs = model.loss_bcs_log
    # loss_u_t_ics = model.loss_ut_ics_log
    # loss_ics = model.loss_ics_log
    # total_loss = model.total_loss_log

    fig = plt.figure(figsize=(6, 5))
    # plt.plot(total_loss, label='$\mathcal{L}_{total}$')
    plt.plot(loss_res, label='$\mathcal{L}_{residual}$')
    plt.plot(loss_bcs, label='$\mathcal{L}_{boundary}$')
    # plt.plot(loss_u_t_ics, label='$\mathcal{L}_{u_t}$')
    # plt.plot(loss_ics, label='$\mathcal{L}_{initial}$')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
