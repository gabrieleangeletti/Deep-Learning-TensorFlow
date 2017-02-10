"""Collection of common layers."""

import tensorflow as tf


class Layers(object):
    """Collection of layers."""

    @staticmethod
    def softmax(last_layer, n_classes, name="softmax"):
        """Create a SoftMax layer.

        Parameters
        ----------

        last_layer : tf.Tensor
            Last layer's output node.

        n_classes : int
            Number of classes.

        Returns
        -------

        tuple (
            tf.Tensor : SoftMax output tensor
            tf.Tensor : SoftMax weights variable
            tf.Tensor : SoftMax biases variable
        )
        """
        with tf.name_scope(name):
            inp_dim = last_layer.get_shape()[1].value
            sm_W = tf.Variable(
                tf.truncated_normal([inp_dim, n_classes], stddev=0.1))
            sm_b = tf.Variable(tf.constant(0.1, shape=[n_classes]))
            sm_out = tf.add(tf.matmul(last_layer, sm_W), sm_b)
            return (sm_out, sm_W, sm_b)

    @staticmethod
    def regularization(variables, regtype, regcoef):
        """Compute the regularization tensor.

        Parameters
        ----------

            variables : list of tf.Variable
                List of model variables.

            regtype : str
                Type of regularization. Can be ["none", "l1", "l2"]

            regcoef : float,
                Regularization coefficient.

        Returns
        -------

            tf.Tensor : Regularization tensor.
        """
        if regtype != 'none':
            regs = tf.constant(0.0)
            for v in variables:
                if regtype == 'l2':
                    regs = tf.add(regs, tf.nn.l2_loss(v))
                elif regtype == 'l1':
                    regs = tf.add(regs, tf.reduce_sum(tf.abs(v)))

            return tf.mul(regcoef, regs)
        else:
            return None
