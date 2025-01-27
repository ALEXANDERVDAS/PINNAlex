import tensorflow.compat.v1 as tf
import timeit
import numpy as np


class PoissonPINN:
    def __init__(self, layers, X_u, Y_u, X_r, Y_r):
        self.mu_X, self.sigma_X = X_r.mean(0), X_r.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]

        # Normalize
        self.X_u = (X_u - self.mu_X) / self.sigma_X
        self.Y_u = Y_u
        self.X_r = (X_r - self.mu_X) / self.sigma_X
        self.Y_r = Y_r

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # Define the size of the Kernel
        self.kernel_size = X_u.shape[0]

        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Define placeholders and computational graph
        self.x_u_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_bc_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.u_bc_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_r_tf = tf.placeholder(tf.float32, shape=(None, 1))
        self.r_tf = tf.placeholder(tf.float32, shape=(None, 1))

        self.x_u_ntk_tf = tf.placeholder(tf.float32, shape=(self.kernel_size, 1))
        self.x_r_ntk_tf = tf.placeholder(tf.float32, shape=(self.kernel_size, 1))

        # Evaluate predictions
        self.u_bc_pred = self.net_u(self.x_bc_tf)

        self.u_pred = self.net_u(self.x_u_tf)
        self.r_pred = self.net_r(self.x_r_tf)

        self.u_ntk_pred = self.net_u(self.x_u_ntk_tf)
        self.r_ntk_pred = self.net_r(self.x_r_ntk_tf)

        # Boundary loss
        self.loss_bcs = 100 * tf.reduce_mean(tf.square(self.u_bc_pred - self.u_bc_tf))
        # self.loss_bcs = np.nan_to_num(temp, nan=1.0e8, posinf=1.0e9, neginf=0.0)

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_tf - self.r_pred))

        # Total loss
        self.loss = self.loss_res + self.loss_bcs

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-6 # WAS 1e-2 (1e-3 better for ADAM)   tested on highfreq20 (SGDM 4e-7 (step 5000)) (Nesterov 1e-7) (SGD 1e-6) (Adagrad 1e-2) (ADAM 5e-4) (Adam can maybe be optimized more (check loss))
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                        5000, 0.9, staircase=False)
        # Passing global_step to minimize() will increment it at each step.
        # To compute NTK, it is better to use SGD optimizer9.21e-03 3
        # since the corresponding gradient flow is not exactly same. 5
        # self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss, global_step=self.global_step) #GradientDescentOptimizer / AdamOptimizer / AdagradOptimizer / MomentumOptimizer(, 0.9)
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1e10, 1e10), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        # optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        # self.train_op = optimizer.apply_gradients([(tf.clip_by_value(grad, -1e10, 1e10), var) for grad, var in optimizer.compute_gradients(self.loss)], global_step=self.global_step)


        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver()

        # Logger
        # Loss logger
        self.loss_bcs_log = []
        self.loss_res_log = []

        # NTK logger
        self.K_uu_log = []
        self.K_rr_log = []
        self.K_ur_log = []

        # Weights logger
        self.weights_log = []
        self.biases_log = []

    # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)

    # NTK initialization
    def NTK_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        std = 1. / np.sqrt(in_dim)
        return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32) * std,
                           dtype=tf.float32)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]]) # WHY NOT XAVIER?
            b = tf.Variable(tf.random.normal([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # Evaluates the forward pass
    def forward_pass(self, H):
        num_layers = len(self.layers)
        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    # Evaluates the PDE solution
    def net_u(self, x):
        u = self.forward_pass(x)
        return u

    # Forward pass for the residual
    def net_r(self, x):
        u = self.net_u(x)

        u_x = tf.gradients(u, x)[0] / self.sigma_x
        u_xx = tf.gradients(u_x, x)[0] / self.sigma_x

        return u_xx

    #Trains the model (self)
    def train(self, iter=1):

        start_time = timeit.default_timer()
        for it in range(iter):
            # Fetch boundary mini-batches
            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x_bc_tf: self.X_u, self.u_bc_tf: self.Y_u,
                       self.x_u_tf: self.X_u, self.x_r_tf: self.X_r,
                       self.r_tf: self.Y_r
                       }

            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bcs_value, loss_res_value = self.sess.run([self.loss_bcs, self.loss_res], tf_dict)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)

                print('It: %d, Loss: %.3e, Loss_bcs: %.3e, Loss_res: %.3e ,Time: %.2f' %
                      (it, loss_value, loss_bcs_value, loss_res_value, elapsed))
                # print("Learning rate: "+str(self.learning_rate))
                # print('Learning rate: %f' % (self.sess.run(self.opt ._lr))) #ADAM
                # print('Learning rate: %f' % (self.sess.run(self.opt ._learning_rate))) #SGD
                start_time = timeit.default_timer()
            # if it % 5000 == 0:
                # plot_actual_and_loss(self)

    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_u_tf: X_star}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

    # Evaluates predictions at test points
    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.x_r_tf: X_star}
        r_star = self.sess.run(self.r_pred, tf_dict)
        return r_star
