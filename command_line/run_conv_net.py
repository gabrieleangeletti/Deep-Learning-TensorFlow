import tensorflow as tf
import numpy as np
import os

import config

from yadlt.models.convolutional_models import conv_net
from yadlt.utils import datasets, utilities

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10", "custom"]')
flags.DEFINE_string('original_shape', '28,28,1', 'Original shape of the images in the dataset.')
flags.DEFINE_string('train_dataset', '', 'Path to train set .npy file.')
flags.DEFINE_string('train_labels', '', 'Path to train labels .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('valid_labels', '', 'Path to valid labels .npy file.')
flags.DEFINE_string('test_dataset', '', 'Path to test set .npy file.')
flags.DEFINE_string('test_labels', '', 'Path to test labels .npy file.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_string('model_name', 'convnet', 'Model name.')
flags.DEFINE_string('main_dir', 'convnet/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_integer('verbose', 0, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')


# Convolutional Net parameters
flags.DEFINE_string('layers', '', 'String representing the architecture of the network.')
flags.DEFINE_string('loss_func',  'softmax_cross_entropy', 'Loss function. ["mean_squared" or "softmax_cross_entropy"]')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 10, 'Size of each mini-batch.')
flags.DEFINE_string('opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum", "adam"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')
flags.DEFINE_float('dropout', 1, 'Dropout parameter.')

assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
assert FLAGS.opt in ['gradient_descent', 'ada_grad', 'momentum', 'adam']
assert FLAGS.loss_func in ['mean_squared', 'softmax_cross_entropy']

if __name__ == '__main__':

    utilities.random_seed_np_tf(FLAGS.seed)

    if FLAGS.dataset == 'mnist':

        # ################# #
        #   MNIST Dataset   #
        # ################# #

        trX, trY, vlX, vlY, teX, teY = datasets.load_mnist_dataset(mode='supervised')

    elif FLAGS.dataset == 'cifar10':

        # ################### #
        #   Cifar10 Dataset   #
        # ################### #

        trX, trY, teX, teY = datasets.load_cifar10_dataset(FLAGS.cifar_dir, mode='supervised')
        vlX = teX[:5000]  # Validation set is the first half of the test set
        vlY = teY[:5000]

    elif FLAGS.dataset == 'custom':

        # ################## #
        #   Custom Dataset   #
        # ################## #

        def load_from_np(dataset_path):
            if dataset_path != '':
                return np.load(dataset_path)
            else:
                return None


        trX, trY = load_from_np(FLAGS.train_dataset), load_from_np(FLAGS.train_labels)
        vlX, vlY = load_from_np(FLAGS.valid_dataset), load_from_np(FLAGS.valid_labels)
        teX, teY = load_from_np(FLAGS.test_dataset), load_from_np(FLAGS.test_labels)

    else:
        trX, trY, vlX, vlY, teX, teY = None, None, None, None, None, None

    models_dir = os.path.join(config.models_dir, FLAGS.main_dir)
    data_dir = os.path.join(config.data_dir, FLAGS.main_dir)
    summary_dir = os.path.join(config.summary_dir, FLAGS.main_dir)

    # Create the model object
    convnet = conv_net.ConvolutionalNetwork(
        models_dir=models_dir, data_dir=data_dir, summary_dir=summary_dir, original_shape=[int(i) for i in FLAGS.original_shape.split(',')],
        layers=FLAGS.layers, model_name=FLAGS.model_name, main_dir=FLAGS.main_dir, loss_func=FLAGS.loss_func,
        num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size, dataset=FLAGS.dataset, opt=FLAGS.opt,
        learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum, dropout=FLAGS.dropout, verbose=FLAGS.verbose
    )

    # Model training
    print('Start Convolutional Network training...')
    convnet.fit(trX, trY, vlX, vlY, restore_previous_model=FLAGS.restore_previous_model)

    # Test the model
    print('Test set accuracy: {}'.format(convnet.compute_accuracy(teX, teY)))
