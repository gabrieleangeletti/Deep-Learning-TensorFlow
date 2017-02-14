import numpy as np
import tensorflow as tf

from yadlt.models.autoencoders import denoising_autoencoder
from yadlt.utils import datasets, utilities

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10", "custom"]')
flags.DEFINE_string('train_dataset', '', 'Path to train set .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('test_dataset', '', 'Path to test set .npy file.')
flags.DEFINE_string('name', 'dae', 'Model name.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_boolean('encode_train', False, 'Whether to encode and store the training set.')
flags.DEFINE_boolean('encode_valid', False, 'Whether to encode and store the validation set.')
flags.DEFINE_boolean('encode_test', False, 'Whether to encode and store the test set.')
flags.DEFINE_string('save_reconstructions', '', 'Path to a .npy file to save the reconstructions of the model.')
flags.DEFINE_string('save_parameters', '', 'Path to save the parameters of the model.')
flags.DEFINE_string('weights', None, 'Path to a numpy array containing the weights of the autoencoder.')
flags.DEFINE_string('h_bias', None, 'Path to a numpy array containing the encoder bias vector.')
flags.DEFINE_string('v_bias', None, 'Path to a numpy array containing the decoder bias vector.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')

# Stacked Denoising Autoencoder specific parameters
flags.DEFINE_integer('n_components', 256, 'Number of hidden units in the dae.')
flags.DEFINE_float('regcoef', 5e-4, 'Regularization parameter. If 0, no regularization.')
flags.DEFINE_string('corr_type', 'none', 'Type of input corruption. ["none", "masking", "salt_and_pepper"]')
flags.DEFINE_float('corr_frac', 0., 'Fraction of the input to corrupt.')
flags.DEFINE_string('enc_act_func', 'tanh', 'Activation function for the encoder. ["sigmoid", "tanh"]')
flags.DEFINE_string('dec_act_func', 'none', 'Activation function for the decoder. ["sigmoid", "tanh", "none"]')
flags.DEFINE_string('loss_func', 'mse', 'Loss function. ["mse" or "cross_entropy"]')
flags.DEFINE_string('opt', 'sgd', '["sgd", "adagrad", "momentum", "adam"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 10, 'Size of each mini-batch.')

assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
assert FLAGS.train_dataset != '' if FLAGS.dataset == 'custom' else True
assert FLAGS.enc_act_func in ['sigmoid', 'tanh']
assert FLAGS.dec_act_func in ['sigmoid', 'tanh', 'none']
assert FLAGS.corr_type in ['masking', 'salt_and_pepper', 'none']
assert 0. <= FLAGS.corr_frac <= 1.

if __name__ == '__main__':

    utilities.random_seed_np_tf(FLAGS.seed)

    if FLAGS.dataset == 'mnist':

        # ################# #
        #   MNIST Dataset   #
        # ################# #

        trX, vlX, teX = datasets.load_mnist_dataset(mode='unsupervised')

    elif FLAGS.dataset == 'cifar10':

        # ################### #
        #   Cifar10 Dataset   #
        # ################### #

        trX, teX = datasets.load_cifar10_dataset(FLAGS.cifar_dir, mode='unsupervised')
        vlX = teX[:5000]  # Validation set is the first half of the test set

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
        trX = None
        vlX = None
        teX = None

    # Create the object
    enc_act_func = utilities.str2actfunc(FLAGS.enc_act_func)
    dec_act_func = utilities.str2actfunc(FLAGS.dec_act_func)

    dae = denoising_autoencoder.DenoisingAutoencoder(
        name=FLAGS.name, n_components=FLAGS.n_components,
        enc_act_func=enc_act_func, dec_act_func=dec_act_func,
        corr_type=FLAGS.corr_type, corr_frac=FLAGS.corr_frac,
        loss_func=FLAGS.loss_func, opt=FLAGS.opt, regcoef=FLAGS.regcoef,
        learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum,
        num_epochs=FLAGS.num_epochs,
        batch_size=FLAGS.batch_size)

    # Fit the model
    W = None
    if FLAGS.weights:
        W = np.load(FLAGS.weights)

    bh = None
    if FLAGS.h_bias:
        bh = np.load(FLAGS.h_bias)

    bv = None
    if FLAGS.v_bias:
        bv = np.load(FLAGS.v_bias)

    dae.fit(trX, trX, vlX, vlX)

    # Save the model paramenters
    if FLAGS.save_parameters:
        print('Saving the parameters of the model...')
        params = dae.get_parameters()
        for p in params:
            np.save(FLAGS.save_parameters + '-' + p, params[p])

    # Save the reconstructions of the model
    if FLAGS.save_reconstructions:
        print('Saving the reconstructions for the test set...')
        np.save(FLAGS.save_reconstructions, dae.reconstruct(teX))
