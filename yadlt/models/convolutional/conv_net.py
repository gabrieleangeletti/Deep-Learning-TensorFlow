"""Implementation of Convolutional Neural Networks using TensorFlow."""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from yadlt.core import SupervisedModel
from yadlt.core import Evaluation, Loss, Trainer
from yadlt.utils import tf_utils, utilities


class ConvolutionalNetwork(SupervisedModel):
    """Implementation of Convolutional Neural Networks using TensorFlow.

    The interface of the class is sklearn-like.
    """

    def __init__(
        self, layers, original_shape, name='convnet',
        loss_func='softmax_cross_entropy', num_epochs=10, batch_size=10,
            opt='sgd', learning_rate=0.01, momentum=0.5, dropout=0.5):
        """Constructor.

        :param layers: string used to build the model.
            This string is a comma-separate specification of the layers.
            Supported values:
                conv2d-FX-FY-Z-S: 2d convolution with Z feature maps as output
                    and FX x FY filters. S is the strides size
                maxpool-X: max pooling on the previous layer. X is the size of
                    the max pooling
                full-X: fully connected layer with X units
                softmax: softmax layer
            For example:
                conv2d-5-5-32,maxpool-2,conv2d-5-5-64,maxpool-2,full-128,full-128,softmax

        :param original_shape: original shape of the images in the dataset
        :param dropout: Dropout parameter
        """
        SupervisedModel.__init__(self, name)

        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.opt = opt
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.momentum = momentum

        self.loss = Loss(self.loss_func)
        self.trainer = Trainer(
            opt, learning_rate=learning_rate,
            momentum=momentum)

        self.layers = layers
        self.original_shape = original_shape
        self.dropout = dropout

        self.W_vars = None
        self.B_vars = None

        self.accuracy = None

    def _train_model(self, train_set, train_labels,
                     validation_set, validation_labels):
        """Train the model.

        :param train_set: training set
        :param train_labels: training labels
        :param validation_set: validation set
        :param validation_labels: validation labels
        :return: self
        """
        shuff = list(zip(train_set, train_labels))

        pbar = tqdm(range(self.num_epochs))
        for i in pbar:

            np.random.shuffle(list(shuff))
            batches = [_ for _ in utilities.gen_batches(
                shuff, self.batch_size)]

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.tf_session.run(
                    self.train_step,
                    feed_dict={self.input_data: x_batch,
                               self.input_labels: y_batch,
                               self.keep_prob: self.dropout})

            if validation_set is not None:
                feed = {self.input_data: validation_set,
                        self.input_labels: validation_labels,
                        self.keep_prob: 1}
                acc = tf_utils.run_summaries(
                    self.tf_session, self.tf_merged_summaries,
                    self.tf_summary_writer, i, feed, self.accuracy)
                pbar.set_description("Accuracy: %s" % (acc))

    def build_model(self, n_features, n_classes):
        """Create the computational graph of the model.

        :param n_features: Number of features.
        :param n_classes: number of classes.
        :return: self
        """
        self._create_placeholders(n_features, n_classes)
        self._create_layers(n_classes)

        self.cost = self.loss.compile(self.mod_y, self.input_labels)
        self.train_step = self.trainer.compile(self.cost)
        self.accuracy = Evaluation.accuracy(self.mod_y, self.input_labels)

    def _create_placeholders(self, n_features, n_classes):
        """Create the TensorFlow placeholders for the model.

        :param n_features: number of features of the first layer
        :param n_classes: number of classes
        :return: self
        """
        self.input_data = tf.placeholder(
            tf.float32, [None, n_features], name='x-input')
        self.input_labels = tf.placeholder(
            tf.float32, [None, n_classes], name='y-input')
        self.keep_prob = tf.placeholder(
            tf.float32, name='keep-probs')

    def _create_layers(self, n_classes):
        """Create the layers of the model from self.layers.

        :param n_classes: number of classes
        :return: self
        """
        next_layer_feed = tf.reshape(self.input_data,
                                     [-1, self.original_shape[0],
                                      self.original_shape[1],
                                      self.original_shape[2]])
        prev_output_dim = self.original_shape[2]
        # this flags indicates whether we are building the first dense layer
        first_full = True

        self.W_vars = []
        self.B_vars = []

        for i, l in enumerate(self.layers.split(',')):

            node = l.split('-')
            node_type = node[0]

            if node_type == 'conv2d':

                # ################### #
                # Convolutional Layer #
                # ################### #

                # fx, fy = shape of the convolutional filter
                # feature_maps = number of output dimensions
                fx, fy, feature_maps, stride = int(node[1]),\
                     int(node[2]), int(node[3]), int(node[4])

                print('Building Convolutional layer with %d input channels\
                      and %d %dx%d filters with stride %d' %
                      (prev_output_dim, feature_maps, fx, fy, stride))

                # Create weights and biases
                W_conv = self.weight_variable(
                    [fx, fy, prev_output_dim, feature_maps])
                b_conv = self.bias_variable([feature_maps])
                self.W_vars.append(W_conv)
                self.B_vars.append(b_conv)

                # Convolution and Activation function
                h_conv = tf.nn.relu(
                    self.conv2d(next_layer_feed, W_conv, stride) + b_conv)

                # keep track of the number of output dims of the previous layer
                prev_output_dim = feature_maps
                # output node of the last layer
                next_layer_feed = h_conv

            elif node_type == 'maxpool':

                # ################# #
                # Max Pooling Layer #
                # ################# #

                ksize = int(node[1])

                print('Building Max Pooling layer with size %d' % ksize)

                next_layer_feed = self.max_pool(next_layer_feed, ksize)

            elif node_type == 'full':

                # ####################### #
                # Densely Connected Layer #
                # ####################### #

                if first_full:  # first fully connected layer

                    dim = int(node[1])
                    shp = next_layer_feed.get_shape()
                    tmpx = shp[1].value
                    tmpy = shp[2].value
                    fanin = tmpx * tmpy * prev_output_dim

                    print('Building fully connected layer with %d in units\
                          and %d out units' % (fanin, dim))

                    W_fc = self.weight_variable([fanin, dim])
                    b_fc = self.bias_variable([dim])
                    self.W_vars.append(W_fc)
                    self.B_vars.append(b_fc)

                    h_pool_flat = tf.reshape(next_layer_feed, [-1, fanin])
                    h_fc = tf.nn.relu(tf.add(
                        tf.matmul(h_pool_flat, W_fc),
                        b_fc))
                    h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

                    prev_output_dim = dim
                    next_layer_feed = h_fc_drop

                    first_full = False

                else:  # not first fully connected layer

                    dim = int(node[1])
                    W_fc = self.weight_variable([prev_output_dim, dim])
                    b_fc = self.bias_variable([dim])
                    self.W_vars.append(W_fc)
                    self.B_vars.append(b_fc)

                    h_fc = tf.nn.relu(tf.add(
                        tf.matmul(next_layer_feed, W_fc), b_fc))
                    h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

                    prev_output_dim = dim
                    next_layer_feed = h_fc_drop

            elif node_type == 'softmax':

                # ############# #
                # Softmax Layer #
                # ############# #

                print('Building softmax layer with %d in units and\
                      %d out units' % (prev_output_dim, n_classes))

                W_sm = self.weight_variable([prev_output_dim, n_classes])
                b_sm = self.bias_variable([n_classes])
                self.W_vars.append(W_sm)
                self.B_vars.append(b_sm)

                self.mod_y = tf.add(tf.matmul(next_layer_feed, W_sm), b_sm)

    @staticmethod
    def weight_variable(shape):
        """Create a weight variable."""
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """Create a bias variable."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W, stride):
        """2D Convolution operation."""
        return tf.nn.conv2d(
            x, W, strides=[1, stride, stride, 1], padding='SAME')

    @staticmethod
    def max_pool(x, dim):
        """Max pooling operation."""
        return tf.nn.max_pool(
            x, ksize=[1, dim, dim, 1], strides=[1, dim, dim, 1],
            padding='SAME')
