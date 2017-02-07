"""Implementation of Denoising Autoencoder using TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from yadlt.core.layers import Linear, Activation, Regularization, Loss
from yadlt.core import Sequential
from yadlt.core import Trainer
from yadlt.utils import utilities


class DenoisingAutoencoder(Sequential):
    """Implementation of Denoising Autoencoders using TensorFlow.

    The interface of the class is sklearn-like.

    TODO: Add Regularization. Add tied weights.
    """

    def __init__(
        self, n_components, name='dae', enc_act_func=tf.nn.tanh,
        dec_act_func=None, loss_func='mean_squared', num_epochs=10,
        batch_size=10, opt='sgd', learning_rate=0.01, momentum=0.9,
        corr_type='none', corr_frac=0., verbose=1, regtype='none',
            l2reg=5e-4):
        """Constructor.

        Parameters
        ----------

        n_components : int
            Number of hidden units.

        name : str, optional (default = 'dae')
            Name of this model.

        enc_act_func : tf.nn.[activation function], optional (default = 'tanh')
            Activation function for the encoder.

        dec_act_func : tf.nn.[activation function], optional (default = None)
            Activation function for the decoder.

        loss_func : str, optional (default = 'mean_squared')
            Training loss function. Possible values: 'mean_squared' or
            'cross_entropy'.

        num_epochs : int, optional (default = 10)
            Training number of epochs.

        batch_size : int, optional (default = 10)
            Training batch size.

        opt : str, optional (default = 'sgd')
            Training optimizer. For a list of the optimizers available, cfr.
            `yadlt.core.trainers.py`.

        learning_rate : float, optional (default = 0.01)
            Training learning rate.

        momentum : float, optional (default = 0.9)
            Training momentum parameter (only for MomentumOptimizer).

        corr_type : str, optional (default = 'none')
            Type of input corruption. Can be one of ["none", "masking",
            "salt_and_pepper"].

        corr_frac : float, optional (default = 0.)
            Fraction of the input to corrupt. Considered only if corr_type is
            not "none".

        verbose : int, optional (default = 1)
            Level of verbosity.

        regtype : str, optional (default = "none")
            Regularization type. Can be one of ["none", "l1", "l2"].

        l2reg : float, optional (default = 5e-4)
            Regularization parameter. If 0, no regularization. Considered only
            if regtype is not "none".

        Attributes
        ----------
            name : str
                Name of this instance.
        """
        Sequential.__init__(self, name=name)

        self.trainer = Trainer(
            opt, learning_rate=learning_rate, momentum=momentum)

        self.train_params = {
            "learning_rate": learning_rate,
            "momentum": momentum,
            "opt": opt,
            "regtype": regtype,
            "l2reg": l2reg,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "loss_func": loss_func
        }

        self.n_components = n_components
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func
        self.corr_type = corr_type
        self.corr_frac = corr_frac
        self.verbose = verbose

    def _run_train_step(self, train_set):
        """Run a training step.

        A training step is made by randomly corrupting the training set,
        randomly shuffling it,  divide it into batches and run the optimizer
        for each batch.
        :param train_set: training set
        :return: self
        """
        x_corrupted = utilities._corrupt_input(
            train_set, self.corr_type, self.corr_frac, self.tf_session)

        shuff = list(zip(train_set, x_corrupted))
        np.random.shuffle(shuff)

        batches = [_ for _ in utilities.gen_batches(
            shuff, self.train_params["batch_size"])]

        for batch in batches:
            x_batch, x_corr_batch = zip(*batch)
            tr_feed = {self.placeholders["input_orig"]: x_batch,
                       self.placeholders["input_corr"]: x_corr_batch}
            self.tf_session.run(self.train_op, feed_dict=tr_feed)

    def build_model(self, n_feats):
        """Create the computational graph for a denoising autoencoder.

        Parameters
        ----------

        n_feats : int
            Number of features.

        Returns
        -------

        self
        """
        # Model Input Placeholders
        input_orig = tf.placeholder(tf.float32, [None, n_feats], name='x')
        input_corr = tf.placeholder(tf.float32, [None, n_feats], name='corr-x')
        self.add_placeholder("input_orig", input_orig)
        self.add_placeholder("input_corr", input_corr)

        # Model Architecture
        self.add(Linear((n_feats, self.n_components), name="encoder"))\
            .add(Activation(self.enc_act_func, name="encoder-act"))\
            .add(Linear((self.n_components, n_feats), name="decoder"))\
            .add(Activation(self.dec_act_func, name="decoder-act"))

        # Model Training loss and Backpropagation
        loss = Loss(self.forward("input_corr"), input_orig,
                    self.train_params["loss_func"])
        self.train_op = self.trainer.compile(loss.loss)

        return self

    def predict(self, X):
        """Reconstruct input."""
        pass

    def score(self, X, y):
        """Compute model score."""
        pass
