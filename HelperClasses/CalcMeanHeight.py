import numpy as np
import tensorflow as tf

dom_coords = np.array([[0.0, 0.0],
                       [1.0, 1.0]])

def u(x, a, c):
    """
    :param x: x = (t, x)
    """
    t = x[:, 0:1]
    x = x[:, 1:2]
    return np.sin(5 * np.pi * x) * np.cos(5 * c * np.pi * t) + \
        a * np.sin(7 * np.pi * x) * np.cos(7 * c * np.pi * t)

def check(x, a ,c):
    return np.full(x.shape, 0)

nn = 1000
t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
t, x = np.meshgrid(t, x)
X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

# Predictions
u_pred = u(X_star, 1, 0)
squared = u_pred ** 2
mean = tf.math.reduce_mean(squared)

print(mean)
