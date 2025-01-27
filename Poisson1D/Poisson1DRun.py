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


# Constants

bc1_coords = np.array([[0.0],
                       [0.0]])

bc2_coords = np.array([[1.0],
                       [1.0]])

dom = np.array([[0.0],
                [1.0]])

tf.logging.set_verbosity(tf.logging.ERROR)

# Temp
nn = 100

a = 20

X_bc1 = 0.0 * np.ones((nn // 2, 1))
X_bc2 = 1.0 * np.ones((nn // 2, 1))
X_u = np.vstack([X_bc1, X_bc2])
Y_u = u(X_u, a)

X_r = np.linspace(0.0,
                  1.0, nn)[:, None]
Y_r = u_xx(X_r, a)



# Run Model

layers = [1, 512, 1]
# layers = [1, 512, 512, 512, 1]

model = PoissonPINN(layers, X_u, Y_u, X_r, Y_r)
iter = 1 #ITERATIONS
# Train model
model.train(iter=iter)

# Visualize
plot_loss_iterations(model)
plot_actual_and_loss(model, u, u_xx, dom, a)
