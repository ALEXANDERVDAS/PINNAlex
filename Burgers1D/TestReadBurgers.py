import scipy.io
import numpy as np
import tensorflow as tf


data = scipy.io.loadmat("BurgersData")
x = data['x']
t = data['t']
# create a meshgrid
X, T = np.meshgrid(x, t)
# flatten the meshgrid
tx_train = np.hstack((T.flatten()[:,None], X.flatten()[:,None]))
u_train = data['usol'].T.flatten()[:,None]
# convert to tf
# tx_train = tf.convert_to_tensor(tx_train, dtype=tf.float32)
# u_train = tf.convert_to_tensor(u_train, dtype=tf.float32)
print(u_train)