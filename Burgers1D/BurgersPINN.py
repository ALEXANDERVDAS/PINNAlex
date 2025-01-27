import tensorflow as tf
# from Compute_Jacobian import jacobian # Please download 'Compute_Jacobian.py' in the repository
import numpy as np
import timeit
from scipy.interpolate import griddata
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os



class BurgersPINN:
    # Initialize the class
    def __init__(self, layers, operator, ics_sampler, bcs_sampler, res_sampler, v, kernel_size):
        # Normalization
        X, _ = res_sampler.sample(np.int32(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_t, self.sigma_t = self.mu_X[0], self.sigma_X[0]
        self.mu_x, self.sigma_x = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.operator = operator
        self.ics_sampler = ics_sampler
        self.bcs_sampler = bcs_sampler
        self.res_sampler = res_sampler

        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # weights
        self.lam_bound_val = np.array(1.0) # 5 tried
        self.lam_ics_val = np.array(1.0) # 9 tried
        # self.lam_ut_val = np.array(1.0)
        self.lam_r_val = np.array(1.0)

        # Wave constant
        self.v = tf.constant(v, dtype=tf.float32)

        self.kernel_size = kernel_size  # Size of the NTK matrix

        # Define Tensorflow session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        #
        # # Define placeholders and computational graph
        self.t_u_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        self.x_u_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

        self.t_ics_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        self.x_ics_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        self.u_ics_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        self.u_bc1_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        self.u_bc2_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

        self.t_bc1_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        self.x_bc1_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

        self.t_bc2_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        self.x_bc2_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

        self.t_r_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        self.x_r_tf = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))

        self.lam_bound_tf = tf.compat.v1.placeholder(tf.float32, shape=self.lam_bound_val.shape)
        self.lam_ics_tf = tf.compat.v1.placeholder(tf.float32, shape=self.lam_bound_val.shape)
        # self.lam_ut_tf = tf.compat.v1.placeholder(tf.float32, shape=self.lam_bound_val.shape)
        self.lam_r_tf = tf.compat.v1.placeholder(tf.float32, shape=self.lam_bound_val.shape)


        # Evaluate predictions
        self.u_ics_pred = self.net_u(self.t_ics_tf, self.x_ics_tf)
        # self.u_t_ics_pred = self.net_u_t(self.t_ics_tf, self.x_ics_tf) # Only used for wave function in NTK paper
        self.u_bc1_pred = self.net_u(self.t_bc1_tf, self.x_bc1_tf)
        self.u_bc2_pred = self.net_u(self.t_bc2_tf, self.x_bc2_tf)

        self.u_pred = self.net_u(self.t_u_tf, self.x_u_tf)
        self.r_pred = self.net_r(self.t_r_tf, self.x_r_tf)

        # # Define predictions for NTK computation
        # self.u_ntk_pred = self.net_u(self.t_u_ntk_tf, self.x_u_ntk_tf)
        # self.ut_ntk_pred = self.net_u_t(self.t_ut_ntk_tf, self.x_ut_ntk_tf)
        # self.r_ntk_pred = self.net_r(self.t_r_ntk_tf, self.x_r_ntk_tf)

        # Boundary loss and Initial loss
        self.loss_ics_u = tf.reduce_mean(tf.square(self.u_ics_tf - self.u_ics_pred))
        # self.loss_ics_u_t = tf.reduce_mean(tf.square(self.u_t_ics_pred))
        self.loss_bc1 = tf.reduce_mean(tf.square(self.u_bc1_tf - self.u_bc1_pred))
        self.loss_bc2 = tf.reduce_mean(tf.square(self.u_bc2_tf - self.u_bc2_pred))

        self.loss_bcs = self.loss_bc1 + self.loss_bc2

        # Residual loss
        self.loss_res = tf.reduce_mean(tf.square(self.r_pred))

        # Total loss
        self.loss = self.lam_r_tf * self.loss_res + self.lam_bound_tf * self.loss_bcs + self.lam_ics_tf * self.loss_ics_u

        # Define optimizer with learning rate schedule
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-3 # Should be 3 for ADAM
        self.learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, self.global_step,1000, 0.9, staircase=False) #Maybe decrease starter learning rate because alot of epochs
        # Passing global_step to minimize() will increment it at each step.
        # Using this dont forget to change Adam print. (Ctrl f Adam)
        self.opt = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate) #GradientDescentOptimizer / AdamOptimizer / AdagradOptimizer / MomentumOptimizer(, 0.9) / RMSPropOptimizer
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)
        # self.opt = tf.compat.v1.train.GradientDescentOptimizer(self.learning_rate)
        # self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

        # Logger
        self.loss_bcs_log = []
        self.loss_ics_log = []
        self.loss_ut_ics_log = []
        self.loss_res_log = []
        self.total_loss_log = []
        self.saver = tf.compat.v1.train.Saver()

        self.squared_mean_height_solution = []


        # NTK logger
        self.K_u_log = []
        self.K_ut_log = []
        self.K_r_log = []

        # weights logger
        self.lam_u_log = []
        self.lam_ut_log = []
        self.lam_r_log = []

        # Initialize Tensorflow variables
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                               dtype=tf.float32)

        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    # Evaluates the forward pass
    def forward_pass(self, H, layers, weights, biases):
        num_layers = len(layers)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    # Forward pass for u
    def net_u(self, t, x):
        u = self.forward_pass(tf.concat([t, x], 1),
                              self.layers,
                              self.weights,
                              self.biases)
        return u

    # Forward pass for du/dt r
    # def net_u_t(self, t, x):
    #     u_t = tf.gradients(self.net_u(t, x), t)[0] / self.sigma_t
    #     return u_t

    # Forward pass for the residual
    def net_r(self, t, x):
        u = self.net_u(t, x)
        residual = self.operator(u, t, x,
                                 self.v,
                                 self.sigma_t,
                                 self.sigma_x)
        return residual


    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

        # Trains the model by minimizing the MSE loss

    def train(self, nIter=10000, batch_size=128, log_NTK=False, update_lam=False):

        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary mini-batches
            X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size // 3)
            X_bc1_batch, u_bc1_batch = self.fetch_minibatch(self.bcs_sampler[0], batch_size // 3)
            X_bc2_batch, u_bc2_batch = self.fetch_minibatch(self.bcs_sampler[1], batch_size // 3)

            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.t_ics_tf: X_ics_batch[:, 0:1], self.x_ics_tf: X_ics_batch[:, 1:2],
                       self.u_ics_tf: u_ics_batch,
                       self.u_bc1_tf: u_bc1_batch,
                       self.u_bc2_tf: u_bc2_batch,
                       self.t_bc1_tf: X_bc1_batch[:, 0:1], self.x_bc1_tf: X_bc1_batch[:, 1:2],
                       self.t_bc2_tf: X_bc2_batch[:, 0:1], self.x_bc2_tf: X_bc2_batch[:, 1:2],
                       self.t_r_tf: X_res_batch[:, 0:1], self.x_r_tf: X_res_batch[:, 1:2],
                       self.lam_bound_tf: self.lam_bound_val,
                       self.lam_ics_tf: self.lam_ics_val,
                       # self.lam_ut_tf: self.lam_ut_val,
                       self.lam_r_tf: self.lam_r_val}

            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)

            if it & 100 == 0:

                #check squared mean heigh of graph. (To check if it is converging to 0 too much..)
                dom_coords = np.array([[0.0, 0.0],
                                       [1.0, 1.0]])
                nn = 10
                t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)[:, None]
                x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)[:, None]
                t, x = np.meshgrid(t, x)
                X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))

                # Predictions
                u_pred = self.predict_u(X_star)
                squared = u_pred ** 2
                self.squared_mean_height_solution.append(squared)

            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time

                loss_value = self.sess.run(self.loss, tf_dict)
                loss_bcs_value = self.sess.run(self.loss_bcs, tf_dict)
                # loss_ics_ut_value = self.sess.run(self.loss_ics_u_t, tf_dict)
                loss_ics_u_value = self.sess.run(self.loss_ics_u, tf_dict)
                loss_res_value = self.sess.run(self.loss_res, tf_dict)

                # Store losses
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)
                self.total_loss_log.append(loss_value)
                # self.loss_ut_ics_log.append(loss_ics_ut_value)
                self.loss_ics_log.append(loss_ics_u_value)

                # print('It: %d, Loss: %.3e, Loss_res: %.3e,  Loss_bcs: %.3e, Loss_ut_ics: %.3e,, Time: %.2f' %
                #       (it, loss_value, loss_res_value, loss_bcs_value, loss_ics_ut_value, elapsed))

                print('It: %d, Loss: %.3e, Loss_res: %.3e, Loss_bcs: %.3e, Loss_u_ics: %.3e,, Time: %.2f' %
                      (it, loss_value, loss_res_value, loss_bcs_value, loss_ics_u_value, elapsed))

                print('lambda_bound: {:.3e}'.format(self.lam_bound_val))
                print('lambda_ics: {:.3e}'.format(self.lam_ics_val))
                # print('lambda_ut: {:.3e}'.format(self.lam_ut_val))
                print('lambda_r: {:.3e}'.format(self.lam_r_val))
                # print('Learning rate: %f' % (self.sess.run(self.opt ._lr))) #ADAM
                # print('Learning rate: %f' % (self.sess.run(self.opt ._learning_rate))) #SGD


                start_time = timeit.default_timer()

            if log_NTK:
                if it % 100 == 0:
                    print("Compute NTK...")
                    X_bc_batch = np.vstack([X_ics_batch, X_bc1_batch, X_bc2_batch])
                    X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size)

                    tf_dict = {self.t_u_ntk_tf: X_bc_batch[:, 0:1], self.x_u_ntk_tf: X_bc_batch[:, 1:2],
                               self.t_ut_ntk_tf: X_ics_batch[:, 0:1], self.x_ut_ntk_tf: X_ics_batch[:, 1:2],
                               self.t_r_ntk_tf: X_res_batch[:, 0:1], self.x_r_ntk_tf: X_res_batch[:, 1:2]}

                    K_u_value, K_ut_value, K_r_value = self.sess.run([self.K_u, self.K_ut, self.K_r], tf_dict)

                    trace_K = np.trace(K_u_value) + np.trace(K_ut_value) + \
                              np.trace(K_r_value)

                    # Store NTK matrices
                    self.K_u_log.append(K_u_value)
                    self.K_ut_log.append(K_ut_value)
                    self.K_r_log.append(K_r_value)

                    # if update_lam:
                    #     self.lam_u_val = trace_K / np.trace(K_u_value)
                    #     self.lam_ut_val = trace_K / np.trace(K_ut_value)
                    #     self.lam_r_val = trace_K / np.trace(K_r_value)
                    #
                    #     # Store NTK weights
                    #     self.lam_u_log.append(self.lam_u_val)
                    #     self.lam_ut_log.append(self.lam_ut_val)
                    #     self.lam_r_log.append(self.lam_r_val)

    # Evaluates predictions at test points
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_u_tf: X_star[:, 0:1], self.x_u_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        return u_star

        # Evaluates predictions at test points

    def predict_r(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        tf_dict = {self.t_r_tf: X_star[:, 0:1], self.x_r_tf: X_star[:, 1:2]}
        r_star = self.sess.run(self.r_pred, tf_dict)
        return r_star