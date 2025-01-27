
from typing import Tuple, List, Union, Callable
import tensorflow as tf
import numpy as np
import sys



LOSS_TOTAL = "loss_total"
LOSS_BOUNDARY = "loss_boundary"
LOSS_INITIAL = "loss_initial"
LOSS_RESIDUAL = "loss_residual"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"

def create_history_dictionary() -> dict:
    """
    Creates a history dictionary.

    Returns:
        The history dictionary.
    """
    return {
        LOSS_TOTAL: [],
        LOSS_BOUNDARY: [],
        LOSS_INITIAL: [],
        LOSS_RESIDUAL: [],
        MEAN_ABSOLUTE_ERROR: []
    }

def create_dense_model(layers: List[Union[int, "tf.keras.layers.Layer"]], activation: "tf.keras.activations.Activation", \
    initializer: "tf.keras.initializers.Initializer", n_inputs: int, n_outputs: int, **kwargs) -> "tf.keras.Model":
    """
    Creates a dense model with the given layers, activation, and input and output sizes.

    Args:
        layers: The layers to use. Elements can be either an integer or a Layer instance. If an integer, a Dense layer with that many units will be used.
        activation: The activation function to use.
        initializer: The initializer to use.
        n_inputs: The number of inputs.
        n_outputs: The number of outputs.
        **kwargs: Additional arguments to pass to the Model constructor.

    Returns:
        The dense model.
    """
    inputs = tf.keras.Input(shape=(n_inputs,))
    x = inputs
    for layer in layers:
        if isinstance(layer, int):
            x = tf.keras.layers.Dense(layer, activation=activation, kernel_initializer=initializer)(x)
        else:
            x = layer(x)
    outputs = tf.keras.layers.Dense(n_outputs, kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, **kwargs)


class WavePinn(tf.keras.Model):
    """
    A model that solves the wave equation.
    """

    def __init__(self, backbone: "tf.keras.Model", c: float, loss_residual_weight=1.0, loss_initial_weight=1.0, \
                 loss_boundary_weight=1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            c: The wave speed.
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.c = c
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight",
                                                 dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight",
                                                 dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight",
                                                dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)

    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                and third input is the boundary data.
            training: Whether or not the model is training.

        Returns:
            The solution for the residual samples, the lhs residual, the solution for the initial samples, \
                and the solution for the boundary samples.
        """

        tx_samples = inputs[0]
        tx_init = inputs[1]
        tx_bnd = inputs[2]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(tx_samples)

            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(tx_samples)
                u_samples = self.backbone(tx_samples, training=training)

            first_order = tape1.batch_jacobian(u_samples, tx_samples)
        second_order = tape2.batch_jacobian(first_order, tx_samples)
        d2u_dt2 = second_order[..., 0, 0]
        d2u_dx2 = second_order[..., 1, 1]
        lhs_samples = d2u_dt2 - (self.c ** 2) * d2u_dx2

        u_bnd = self.backbone(tx_bnd, training=training)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(tx_init)
            u_initial = self.backbone(tx_init, training=training)
        du_dt_init = tape.batch_jacobian(u_initial, tx_init)[..., 0]

        return u_samples, lhs_samples, u_initial, du_dt_init, u_bnd

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. The outputs are the exact solutions for the samples, \
                the exact rhs for the samples, the exact solution for the initial, the exact derivative for the initial, \
                and the exact solution for the boundary.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, du_dt_init_exact, u_bnd_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_initial, du_dt_init, u_bnd = self(inputs, training=True)
            loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
            loss_initial_neumann = self.init_loss(du_dt_init_exact, du_dt_init)
            loss_initial_dirichlet = self.init_loss(u_initial_exact, u_initial)
            loss_initial = loss_initial_neumann + loss_initial_dirichlet
            loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                   self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        Performs a test step.
        """
        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, du_dt_init_exact, u_bnd_exact = outputs

        u_samples, lhs_samples, u_initial, du_dt_init, u_bnd = self(inputs, training=False)
        loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
        loss_initial_neumann = self.init_loss(du_dt_init_exact, du_dt_init)
        loss_initial_dirichlet = self.init_loss(u_initial_exact, u_initial)
        loss_initial = loss_initial_neumann + loss_initial_dirichlet
        loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
        loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
               self._loss_boundary_weight * loss_boundary

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}

    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 100):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary()

        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(
                    f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Initial: {metrs['loss_initial']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}, MAE: {metrs['mean_absolute_error']:0.4f}")

            # reset metrics
            for m in self.metrics:
                m.reset_state()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker,
                self.loss_boundary_tracker, self.mae_tracker]

