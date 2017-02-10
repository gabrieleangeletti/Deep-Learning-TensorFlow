"""Supervised Model scheleton."""

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from yadlt.core.model import Model
from yadlt.utils import tf_utils


class SupervisedModel(Model):
    """Supervised Model scheleton.

    The interface of the class is sklearn-like.

    Methods
    -------

    * fit(): model training procedure.
    * predict(): model inference procedure (predict labels).
    * score(): model scoring procedure (mean accuracy).
    """

    def __init__(self, name):
        """Constructor."""
        Model.__init__(self, name)

    def fit(self, train_X, train_Y, val_X=None, val_Y=None, graph=None):
        """Fit the model to the data.

        Parameters
        ----------

        train_X : array_like, shape (n_samples, n_features)
            Training data.

        train_Y : array_like, shape (n_samples, n_classes)
            Training labels.

        val_X : array_like, shape (N, n_features) optional, (default = None).
            Validation data.

        val_Y : array_like, shape (N, n_classes) optional, (default = None).
            Validation labels.

        graph : tf.Graph, optional (default = None)
            Tensorflow Graph object.

        Returns
        -------
        """
        if len(train_Y.shape) != 1:
            num_classes = train_Y.shape[1]
        else:
            raise Exception("Please convert the labels with one-hot encoding.")

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            # Build model
            self.build_model(train_X.shape[1], num_classes)
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

    def predict(self, test_X):
        """Predict the labels for the test set.

        Parameters
        ----------

        test_X : array_like, shape (n_samples, n_features)
            Test data.

        Returns
        -------

        array_like, shape (n_samples,) : predicted labels.
        """
        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                feed = {
                    self.input_data: test_X,
                    self.keep_prob: 1
                }
                return self.model_predictions.eval(feed)

    def score(self, test_X, test_Y):
        """Compute the mean accuracy over the test set.

        Parameters
        ----------

        test_X : array_like, shape (n_samples, n_features)
            Test data.

        test_Y : array_like, shape (n_samples, n_features)
            Test labels.

        Returns
        -------

        float : mean accuracy over the test set
        """
        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                feed = {
                    self.input_data: test_X,
                    self.input_labels: test_Y,
                    self.keep_prob: 1
                }
                return self.accuracy.eval(feed)
