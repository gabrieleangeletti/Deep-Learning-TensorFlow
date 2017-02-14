"""Softmax classifier implementation using Tensorflow."""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from yadlt.core import Evaluation, Loss
from yadlt.core import SupervisedModel
from yadlt.utils import tf_utils, utilities


class LogisticRegression(SupervisedModel):
    """Simple Logistic Regression using TensorFlow.

    The interface of the class is sklearn-like.
    """

    def __init__(self, name='lr', loss_func='cross_entropy',
                 learning_rate=0.01, num_epochs=10, batch_size=10):
        """Constructor."""
        SupervisedModel.__init__(self, name)

        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.loss = Loss(self.loss_func)

        # Computational graph nodes
        self.input_data = None
        self.input_labels = None

        self.W_ = None
        self.b_ = None

        self.accuracy = None

    def build_model(self, n_features, n_classes):
        """Create the computational graph.

        :param n_features: number of features
        :param n_classes: number of classes
        :return: self
        """
        self._create_placeholders(n_features, n_classes)
        self._create_variables(n_features, n_classes)

        self.mod_y = tf.nn.softmax(
            tf.add(tf.matmul(self.input_data, self.W_), self.b_))

        self.cost = self.loss.compile(self.mod_y, self.input_labels)
        self.train_step = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.cost)
        self.accuracy = Evaluation.accuracy(self.mod_y, self.input_labels)

    def _create_placeholders(self, n_features, n_classes):
        """Create the TensorFlow placeholders for the model.

        :param n_features: number of features
        :param n_classes: number of classes
        :return: self
        """
        self.input_data = tf.placeholder(
            tf.float32, [None, n_features], name='x-input')
        self.input_labels = tf.placeholder(
            tf.float32, [None, n_classes], name='y-input')
        self.keep_prob = tf.placeholder(
            tf.float32, name='keep-probs')

    def _create_variables(self, n_features, n_classes):
        """Create the TensorFlow variables for the model.

        :param n_features: number of features
        :param n_classes: number of classes
        :return: self
        """
        self.W_ = tf.Variable(
            tf.zeros([n_features, n_classes]), name='weights')
        self.b_ = tf.Variable(
            tf.zeros([n_classes]), name='biases')

    def _train_model(self, train_set, train_labels,
                     validation_set, validation_labels):
        """Train the model.

        :param train_set: training set
        :param train_labels: training labels
        :param validation_set: validation set
        :param validation_labels: validation labels
        :return: self
        """
        pbar = tqdm(range(self.num_epochs))
        for i in pbar:

            shuff = list(zip(train_set, train_labels))
            np.random.shuffle(shuff)

            batches = [_ for _ in utilities.gen_batches(shuff, self.batch_size)]

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.tf_session.run(
                    self.train_step,
                    feed_dict={self.input_data: x_batch,
                               self.input_labels: y_batch})

            if validation_set is not None:
                feed = {self.input_data: validation_set,
                        self.input_labels: validation_labels}
                acc = tf_utils.run_summaries(
                    self.tf_session, self.tf_merged_summaries,
                    self.tf_summary_writer, i, feed, self.accuracy)
                pbar.set_description("Accuracy: %s" % (acc))
