"""Unsupervised Model scheleton."""

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from yadlt.core.model import Model
from yadlt.utils import tf_utils


class UnsupervisedModel(Model):
    """Unsupervised Model scheleton class.

    The interface of the class is sklearn-like.

    Methods
    -------

    * fit(): model training procedure.
    * transform(): model inference procedure.
    * reconstruct(): model reconstruction procedure (autoencoders).
    * score(): model scoring procedure (mean error).
    """

    def __init__(self, name):
        """Constructor."""
        Model.__init__(self, name)

    def fit(self, train_X, train_Y=None, val_X=None, val_Y=None, graph=None):
        """Fit the model to the data.

        Parameters
        ----------

        train_X : array_like, shape (n_samples, n_features)
            Training data.

        train_Y : array_like, shape (n_samples, n_features)
            Training reference data.

        val_X : array_like, shape (N, n_features) optional, (default = None).
            Validation data.

        val_Y : array_like, shape (N, n_features) optional, (default = None).
            Validation reference data.

        graph : tf.Graph, optional (default = None)
            Tensorflow Graph object.

        Returns
        -------
        """
        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            # Build model
            self.build_model(train_X.shape[1])
            with tf.Session() as self.tf_session:
                # Initialize tf stuff
                summary_objs = tf_utils.init_tf_ops(self.tf_session)
                self.tf_merged_summaries = summary_objs[0]
                self.tf_summary_writer = summary_objs[1]
                self.tf_saver = summary_objs[2]
                # Train model
                self._train_model(train_X, train_Y, val_X, val_Y)
                # Save model
                self.tf_saver.save(self.tf_session, self.model_path)

    def transform(self, data, graph=None):
        """Transform data according to the model.

        Parameters
        ----------

        data : array_like, shape (n_samples, n_features)
            Data to transform.

        graph : tf.Graph, optional (default = None)
            Tensorflow Graph object

        Returns
        -------
        array_like, transformed data
        """
        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                feed = {self.input_data: data, self.keep_prob: 1}
                return self.encode.eval(feed)

    def reconstruct(self, data, graph=None):
        """Reconstruct data according to the model.

        Parameters
        ----------

        data : array_like, shape (n_samples, n_features)
            Data to transform.

        graph : tf.Graph, optional (default = None)
            Tensorflow Graph object

        Returns
        -------
        array_like, transformed data
        """
        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                feed = {self.input_data: data, self.keep_prob: 1}
                return self.reconstruction.eval(feed)

    def score(self, data, data_ref, graph=None):
        """Compute the reconstruction loss over the test set.

        Parameters
        ----------

        data : array_like
            Data to reconstruct.

        data_ref : array_like
            Reference data.

        Returns
        -------

        float: Mean error.
        """
        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                feed = {
                    self.input_data: data,
                    self.input_labels: data_ref,
                    self.keep_prob: 1
                }
                return self.cost.eval(feed)
