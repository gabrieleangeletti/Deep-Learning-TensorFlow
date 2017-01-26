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

    def __init__(self, shape=None, names=["W", "b"]):
        """Create a new linear layer instance."""
        self.names = names
        if shape:
            self.W = tf.Variable(
                tf.truncated_normal(shape=shape, stddev=0.1), name=names[0])
            self.b = tf.Variable(
                tf.constant(0.1, shape=shape[1]), name=names[1])

    def forward(self, X):
        """Forward propagate X through the fc layer."""
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
        """Create a new linear layer instance."""
        self.name = name
        self.activation = func

    def forward(self, X):
        """Forward propagate X."""
        return self.activation(X)

    def backward(self, H):
        """Backward propagate H through the fc layer."""
        pass

    def get_variables(self):
        """Return the layer's variables."""
        pass

    def get_parameters(self):
        """Return all the parameters of this layer."""
        pass
