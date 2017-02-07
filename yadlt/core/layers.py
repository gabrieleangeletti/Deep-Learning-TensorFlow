"""Layer classes."""

from __future__ import absolute_import

import abc
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class BaseLayer(object):
    """Base layer interface."""

    @abc.abstractmethod
    def forward(self):
        """Layer forward propagation."""
        pass

    @abc.abstractmethod
    def backward(self):
        """Layer backward propagation."""
        pass

    @abc.abstractmethod
    def get_variables(self):
        """Get layer's tf variables."""
        pass

    @abc.abstractmethod
    def get_parameters(self):
        """Get the layer parameters."""
        pass


class Linear(BaseLayer):
    """Fully-Connected layer."""

    def __init__(self, shape, name="linear", vnames=["W", "b"]):
        """Create a new linear layer instance."""
        self.name = name
        self.vnames = vnames
        with tf.name_scope(self.name):
            self.W = tf.Variable(
                tf.truncated_normal(shape=shape, stddev=0.1), name=vnames[0])
            self.b = tf.Variable(
                tf.constant(0.1, shape=[shape[1]]), name=vnames[1])

    def forward(self, X):
        """Forward propagate X through the fc layer."""
        with tf.name_scope(self.name):
            return tf.add(tf.matmul(X, self.W), self.b)

    def backward(self, H):
        """Backward propagate H through the fc layer."""
        pass

    def get_variables(self):
        """Get layer's variables."""
        return [self.W, self.b]

    def get_parameters(self):
        """Return all the parameters of this layer."""
        with tf.Session() as sess:
            return {
                self.names[0]: sess.run(self.W),
                self.names[1]: sess.run(self.b)
            }


class Activation(BaseLayer):
    """Activation function layer."""

    def __init__(self, func, name="act_func"):
        """Create a new Activation layer instance."""
        self.name = name
        if func is not None:
            self.func = func
        else:
            self.func = tf.identity

    def forward(self, X):
        """Forward propagate X."""
        return self.func(X)

    def backward(self, H):
        """Backward propagate H through the fc layer."""
        pass

    def get_variables(self):
        """Return the layer's variables."""
        pass

    def get_parameters(self):
        """Return all the parameters of this layer."""
        pass


class SoftMax(BaseLayer):
    """SoftMax layer."""

    def __init__(self, prev_layer, n_classes, name="softmax"):
        """Create a new SoftMax layer instance."""
        self.prev_layer = prev_layer
        self.shape = (prev_layer.get_shape()[1].value, n_classes)
        self.n_classes = n_classes
        self.name = name
        self.vs = ['softmax_W', 'softmax_b']
        with tf.name_scope(self.name):
            self.W = tf.Variable(
                tf.truncated_normal(self.shape, stddev=0.1), name=self.vs[0])
            self.b = tf.Variable(
                tf.constant(0.1, shape=[n_classes]), name=self.vs[0])

    def forward(self, X):
        """Forward propagate X."""
        with tf.name_scope(self.name):
            return tf.add(tf.matmul(self.prev_layer, self.W), self.b)

    def backward(self, H):
        """Backward propagate H through the fc layer."""
        pass

    def get_variables(self):
        """Return the layer's variables."""
        return (self.W, self.b)

    def get_parameters(self):
        """Return all the parameters of this layer."""
        with tf.Session() as sess:
            return {
                self.vs[0]: sess.run(self.W),
                self.vs[1]: sess.run(self.b)
            }


class Regularization(BaseLayer):
    """Regularization function layer."""

    def __init__(self, variables, C, regtype="l2", name="act_func"):
        """Create a new Regularization layer instance."""
        assert regtype in ["l1", "l2"]

        self.variables = variables
        self.C = C
        self.regtype = regtype
        self.name = name

    def forward(self, X):
        """Forward propagate X."""
        regs = tf.constant(0.0)
        for v in self.variables:
            if self.regtype == "l1":
                regs = tf.add(regs, tf.reduce_sum(tf.abs(v)))

            elif self.regtype == "l2":
                regs = tf.add(regs, tf.nn.l2_loss(v))

        return tf.mul(self.C, regs)

    def backward(self, H):
        """Backward propagate H through the fc layer."""
        pass

    def get_variables(self):
        """Return the layer's variables."""
        pass

    def get_parameters(self):
        """Return all the parameters of this layer."""
        pass


class Loss(BaseLayer):
    """Loss function layer."""

    def __init__(self, mod_y, ref_y, loss_type, regterm=None,
                 summary=True, name="loss_func"):
        """Create a new Loss layer instance."""
        assert loss_type in ["cross_entropy", "softmax_cross_entropy",
                             "mean_squared"]

        self.mod_y = mod_y
        self.ref_y = ref_y
        self.loss_type = loss_type
        self.regterm = regterm
        self.name = name
        if loss_type == "cross_entropy":
            clip_inf = tf.clip_by_value(self.mod_y, 1e-10, float('inf'))
            clip_sup = tf.clip_by_value(1 - self.mod_y, 1e-10, float('inf'))
            loss = - tf.reduce_mean(tf.add(
                    tf.mul(self.ref_y, tf.log(clip_inf)),
                    tf.mul(tf.sub(1.0, self.ref_y), tf.log(clip_sup))))

        elif loss_type == "softmax_cross_entropy":
            loss = tf.contrib.losses.softmax_cross_entropy(
                self.mod_y, self.ref_y)

        elif loss_type == "mean_squared":
            loss = tf.sqrt(tf.reduce_mean(
                tf.square(tf.sub(self.ref_y, self.mod_y))))

        self.loss = loss + regterm if regterm is not None else loss
        if summary:
            tf.summary.scalar(self.name, self.loss)

    def forward(self, X):
        """Forward propagate X."""
        pass

    def backward(self, H):
        """Backward propagate H through the fc layer."""
        pass

    def get_variables(self):
        """Return the layer's variables."""
        pass

    def get_parameters(self):
        """Return all the parameters of this layer."""
        pass
