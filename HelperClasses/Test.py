import numpy as np


def u_xx(x, a):
  return -(np.pi * a)**2 * np.sin(np.pi * a * x)

dom_coords = np.array([[0.0],
                       [1.0]])
X_r = np.linspace(dom_coords[0, 0], dom_coords[1, 0], 100)[:, None]
a = np.linalg.norm(u_xx(X_r, 20), 2)

print(a)
