import tensorflow as tf
import numpy as np
import timeit

from matplotlib import colors
from scipy.interpolate import griddata
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from Sample import Sample
from WavePINN import WavePINN
from VisualizeWave import *


def u(x, a, c):
    """
    :param x: x = (t, x)
    """
    t = x[:, 0:1]
    x = x[:, 1:2]
    return np.sin(5 * np.pi * x) * np.cos(5 * c * np.pi * t) + \
        a * np.sin(7 * np.pi * x) * np.cos(7 * c * np.pi * t)

# def u_t(x,a, c):
#     t = x[:,0:1]
#     x = x[:,1:2]
#     u_t = -  c * np.pi * np.sin(np.pi * x) * np.sin(c * np.pi * t) - \
#             a * 4 * c * np.pi * np.sin(2 * c * np.pi* x) * np.sin(4 * c * np.pi * t)
#     return u_t

# def u_tt(x, a, c):
#     t = x[:,0:1]
#     x = x[:,1:2]
#     u_tt = -(c * np.pi)**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) - \
#             a * (4 * c * np.pi)**2 *  np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
#     return u_tt
#
# def u_xx(x, a, c):
#     t = x[:,0:1]
#     x = x[:,1:2]
#     u_xx = - np.pi**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) - \
#               a * (2 * c * np.pi)** 2 * np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
#     return u_xx


def r(x, a, c):
    return x[:, 1:2] * 0


def operator(u, t, x, c, sigma_t=1.0, sigma_x=1.0):
    u_t = tf.gradients(u, t)[0] / sigma_t
    u_x = tf.gradients(u, x)[0] / sigma_x
    u_tt = tf.gradients(u_t, t)[0] / sigma_t
    u_xx = tf.gradients(u_x, x)[0] / sigma_x
    residual = u_tt - c**2 * u_xx
    return residual


tf.compat.v1.disable_eager_execution()

# Constants

ics_coords = np.array([[0.0, 0.0],
                        [0.0, 1.0]])
bc1_coords = np.array([[0.0, 0.0],
                        [1.0, 0.0]])
bc2_coords = np.array([[0.0, 1.0],
                        [1.0, 1.0]])
dom_coords = np.array([[0.0, 0.0],
                        [1.0, 1.0]])

#All samples
ics_sample = Sample(2, ics_coords, lambda x: u(x, a, c), name='Initial Condition 1')
bc1 = Sample(2, bc1_coords, lambda x: u(x, a, c), name='Dirichlet BC1')
bc2 = Sample(2, bc2_coords, lambda x: u(x, a, c), name='Dirichlet BC2')
bcs_sample = [bc1, bc2]
res_sample = Sample(2, dom_coords, lambda x: r(x, a, c), name='Forcing')
samples = [ics_sample, bcs_sample, res_sample]

layers = [2, 500, 500, 500, 1]
kernel_size = 300

a = 2
c = 1 # Vary for frequency



#Run model

model = WavePINN(layers, operator, samples, c)
#Train wave model

iter = 1
model.train(iter=iter, batchsize=512) # Maybe a little lower... (Normal is 512) (4 batch size 10x faster so do 10x as many epochs (500000))

#Visualize

# plot_squared_mean(model)
# plot_fixed_time(model, 0, u, a, c)
# plot_fixed_time(model, 0.3, u, a, c)
# plot_fixed_time(model, 0.6, u, a, c)
# plot_fixed_time(model, 0.9, u, a, c)
# plot_loss_iterations(model)
plot_heat_map(model, a, c, dom_coords, u, r)


