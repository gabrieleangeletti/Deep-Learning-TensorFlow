"""Supervised Model scheleton."""

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from yadlt.core.model import Model


class SupervisedModel(Model):
    """Supervised Model scheleton."""

    def __init__(self, name):
        """Constructor."""
        Model.__init__(self, name)

    def fit(self, train_set, train_labels, validation_set=None,
            validation_labels=None, restore_previous_model=False, graph=None):
        """Fit the model to the data.

        :param train_set: Training data. shape(n_samples, n_features)
        :param train_labels: Training labels. shape(n_samples, n_classes)
        :param validation_set: optional, default None. Validation data.
            shape(nval_samples, n_features)
        :param validation_labels: optional, default None. Validation labels.
            shape(nval_samples, n_classes)
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk
                    to continue training.
        :param graph: tensorflow graph object
        :return: self
        """
        if len(train_labels.shape) != 1:
            num_classes = train_labels.shape[1]
        else:
            raise Exception("Please convert the labels with one-hot encoding.")

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            self.build_model(train_set.shape[1], num_classes)
            with tf.Session() as self.tf_session:
                self._initialize_tf_utilities_and_ops(restore_previous_model)
                self._train_model(
                    train_set, train_labels, validation_set, validation_labels)
                self.tf_saver.save(self.tf_session, self.model_path)

    def _run_validation_error_and_summaries(self, epoch, feed):
        """Run the summaries and error computation on the validation set.

        :param epoch: current epoch
        :param validation_set: validation data
        :return: self
        """
        try:
            result = self.tf_session.run(
                [self.tf_merged_summaries, self.accuracy], feed_dict=feed)
            summary_str = result[0]
            acc = result[1]
            self.tf_summary_writer.add_summary(summary_str, epoch)
        except tf.errors.InvalidArgumentError:
            if self.tf_summary_writer_available:
                print("Summary writer not available at the moment")
            self.tf_summary_writer_available = False
            acc = self.tf_session.run(self.accuracy, feed_dict=feed)

        return acc

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
