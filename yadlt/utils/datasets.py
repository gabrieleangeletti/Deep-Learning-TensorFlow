"""Datasets module. Provides utilities to load popular datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.models.rnn.ptb import reader

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import os


def load_mnist_dataset(mode='supervised', one_hot=True):
    """Load the MNIST handwritten digits dataset.

    :param mode: 'supervised' or 'unsupervised' mode
    :param one_hot: whether to get one hot encoded labels
    :return: train, validation, test data:
            for (X, y) if 'supervised',
            for (X) if 'unsupervised'
    """
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=one_hot)

    # Training set
    trX = mnist.train.images
    trY = mnist.train.labels

    # Validation set
    vlX = mnist.validation.images
    vlY = mnist.validation.labels

    # Test set
    teX = mnist.test.images
    teY = mnist.test.labels

    if mode == 'supervised':
        return trX, trY, vlX, vlY, teX, teY

    elif mode == 'unsupervised':
        return trX, vlX, teX


def load_cifar10_dataset(cifar_dir, mode='supervised'):
    """Load the cifar10 dataset.

    :param cifar_dir: path to the dataset directory
        (cPicle format from: https://www.cs.toronto.edu/~kriz/cifar.html)
    :param mode: 'supervised' or 'unsupervised' mode

    :return: train, test data:
            for (X, y) if 'supervised',
            for (X) if 'unsupervised'
    """
    # Training set
    trX = None
    trY = np.array([])

    # Test set
    teX = np.array([])
    teY = np.array([])

    for fn in os.listdir(cifar_dir):

        if not fn.startswith('batches') and not fn.startswith('readme'):
            fo = open(os.path.join(cifar_dir, fn), 'rb')
            data_batch = pickle.load(fo)
            fo.close()

            if fn.startswith('data'):

                if trX is None:
                    trX = data_batch['data']
                    trY = data_batch['labels']
                else:
                    trX = np.concatenate((trX, data_batch['data']), axis=0)
                    trY = np.concatenate((trY, data_batch['labels']), axis=0)

            if fn.startswith('test'):
                teX = data_batch['data']
                teY = data_batch['labels']

    trX = trX.astype(np.float32) / 255.
    teX = teX.astype(np.float32) / 255.

    if mode == 'supervised':
        return trX, trY, teX, teY

    elif mode == 'unsupervised':
        return trX, teX


def load_ptb_dataset(data_path):
    """Load the PTB dataset.

    You can download the PTB dataset from here:
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    :param data_path: path to the data/ dir of the PTB dataset.
    :return: train, validation, test data
    """
    raw_data = reader.ptb_raw_data(data_path)
    trX, vlX, teX, _ = raw_data
    return trX, vlX, teX
