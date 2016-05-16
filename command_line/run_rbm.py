import tensorflow as tf
import numpy as np
import os

import config

from yadlt.models.rbm_models import rbm
from yadlt.utils import datasets, utilities

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_boolean('encode_train', False, 'Whether to encode and store the training set.')
flags.DEFINE_boolean('encode_valid', False, 'Whether to encode and store the validation set.')
flags.DEFINE_boolean('encode_test', False, 'Whether to encode and store the test set.')
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10", "custom"]')
flags.DEFINE_string('train_dataset', '', 'Path to train set .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('test_dataset', '', 'Path to test set .npy file.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_string('save_reconstructions', '', 'Path to a .npy file to save the reconstructions of the model.')
flags.DEFINE_string('save_parameters', '', 'Path to save the parameters of the model.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')

# RBM configuration
flags.DEFINE_integer('num_hidden', 250, 'Number of hidden units.')
flags.DEFINE_string('visible_unit_type', 'bin', 'Type of visible units. ["bin", "gauss"]')
flags.DEFINE_string('main_dir', 'rbm/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_string('model_name', 'rbm_model', 'Name for the model.')
flags.DEFINE_integer('verbose', 0, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_integer('gibbs_sampling_steps', 1, 'Number of gibbs sampling steps in Contrastive Divergence.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('stddev', 0.1, 'Standard deviation for the Gaussian visible units.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 10, 'Size of each mini-batch.')
flags.DEFINE_integer('transform_gibbs_sampling_steps', 10, 'Gibbs sampling steps for the transformation of data.')

assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
assert FLAGS.cifar_dir != '' if FLAGS.dataset == 'cifar10' else True
assert FLAGS.visible_unit_type in ['bin', 'gauss']

if __name__ == '__main__':

    utilities.random_seed_np_tf(FLAGS.seed)

    if FLAGS.dataset == 'mnist':

        # ################# #
        #   MNIST Dataset   #
        # ################# #

        trX, vlX, teX = datasets.load_mnist_dataset(mode='unsupervised')
        width, height = 28, 28

    elif FLAGS.dataset == 'cifar10':

        # ################### #
        #   Cifar10 Dataset   #
        # ################### #

        trX, teX = datasets.load_cifar10_dataset(FLAGS.cifar_dir, mode='unsupervised')
        vlX = teX[:5000]  # Validation set is the first half of the test set
        width, height = 32, 32

    elif FLAGS.dataset == 'custom':

        # ################## #
        #   Custom Dataset   #
        # ################## #

        def load_from_np(dataset_path):
            if dataset_path != '':
                return np.load(dataset_path)
            else:
                return None

        trX = load_from_np(FLAGS.train_dataset)
        vlX = load_from_np(FLAGS.valid_dataset)
        teX = load_from_np(FLAGS.test_dataset)

    else:
        trX, vlX, teX, width, height = None, None, None, None, None

    models_dir = os.path.join(config.models_dir, FLAGS.main_dir)
    data_dir = os.path.join(config.data_dir, FLAGS.main_dir)
    summary_dir = os.path.join(config.summary_dir, FLAGS.main_dir)

    # Create the object
    r = rbm.RBM(num_hidden=FLAGS.num_hidden, main_dir=FLAGS.main_dir,
                models_dir=models_dir, data_dir=data_dir, summary_dir=summary_dir,
                visible_unit_type=FLAGS.visible_unit_type, learning_rate=FLAGS.learning_rate,
                num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size, stddev=FLAGS.stddev, verbose=FLAGS.verbose,
                gibbs_sampling_steps=FLAGS.gibbs_sampling_steps, model_name=FLAGS.model_name)

    # Fit the model
    print('Start training...')
    r.fit(trX, teX, restore_previous_model=FLAGS.restore_previous_model)

    # Save the model paramenters
    if FLAGS.save_parameters:
        print('Saving the parameters of the model...')
        params = r.get_model_parameters()
        for p in params:
            np.save(FLAGS.save_parameters + '-' + p, params[p])

    # Save the reconstructions of the model
    if FLAGS.save_reconstructions:
        print('Saving the reconstructions for the test set...')
        np.save(FLAGS.save_reconstructions, r.reconstruct(teX))

    # Encode the training data and store it
    if FLAGS.encode_train:
        print('Transforming training data...')
        r.transform(trX, name='train', save=FLAGS.encode_train)

    if FLAGS.encode_valid:
        print('Transforming validation data...')
        r.transform(vlX, name='validation', save=FLAGS.encode_valid)

    if FLAGS.encode_test:
        print('Transforming test data...')
        r.transform(teX, name='test', save=FLAGS.encode_test)
