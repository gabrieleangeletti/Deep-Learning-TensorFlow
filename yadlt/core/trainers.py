"""Trainers module."""

import tensorflow as tf


class Trainer(object):
    """Wrapper of Tensorflow Optimizers."""

    def __init__(self, optimizer, **kw):
        """Constructor.

        Parameters
        ----------

        optimizer : string
            Which optimizer to use. Possible values are ["sgd", "adagrad",
            "adam", "momentum"]

        kw :
            the following arguments should be provided:
                * sgd: learning_rate (float)
                * adagrad: learning_rate (float), initial_accumulator_value
                (float, default=0.1)
                * adam: learning_rate (float, default=0.001), beta1 (float,
                default=0.9), beta2 (float, default=0.999), epsilon (float,
                default 1e-08)
                * momentum: learning_rate (float), use_nesterov (bool)

        """
        assert optimizer in ["sgd", "adagrad", "adam", "momentum"]

        def d(k, other=None):
            if other:
                return kw[k] if k in kw else other
            else:
                return kw[k]

        if optimizer == "sgd":
            self.opt_ = tf.train.GradientDescentOptimizer(d("learning_rate"))

        elif optimizer == "adagrad":
            self.opt_ = tf.train.AdagradOptimizer(
                d("learning_rate"), d("initial_accumulator_value", 0.1))

        elif optimizer == "adam":
            self.opt_ = tf.train.AdamOptimizer(d("learning_rate", 0.001),
                                               d("beta1", 0.9),
                                               d("beta2", 0.9),
                                               d("epsilon", 1e-08))

        elif optimizer == "momentum":
            self.opt_ = tf.train.MomentumOptimizer(
                d("learning_rate"), d("momentum"),
                use_nesterov=d("use_nesterov", False))

    def compile(self, cost, name_scope="train"):
        """Compile the optimizer with the given training parameters.

        Parameters
        ----------

        cost : Tensor
            A Tensor containing the value to minimize.

        name_scope : str , optional (default="train")
            Optional name scope for the optimizer graph ops.
        """
        with tf.name_scope(name_scope):
            return self.opt_.minimize(cost)
