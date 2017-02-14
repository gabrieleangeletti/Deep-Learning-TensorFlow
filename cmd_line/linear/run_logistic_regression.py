import numpy as np
import tensorflow as tf

from yadlt.models.linear import logistic_regression
from yadlt.utils import datasets, utilities

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10", "custom"]')
flags.DEFINE_string('train_dataset', '', 'Path to train set .npy file.')
flags.DEFINE_string('train_labels', '', 'Path to train labels .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('valid_labels', '', 'Path to valid labels .npy file.')
flags.DEFINE_string('test_dataset', '', 'Path to test set .npy file.')
flags.DEFINE_string('test_labels', '', 'Path to test labels .npy file.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_string('name', 'logreg', 'Name for the model.')
flags.DEFINE_string('loss_func', 'cross_entropy', 'Loss function. ["mse" or "cross_entropy"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 10, 'Size of each mini-batch.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')

assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']

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
        # Validation set is the first half of the test set
        vlX = teX[:5000]
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
        trX = None
        trY = None
        vlX = None
        vlY = None
        teX = None
        teY = None

    # Create the object
    l = logistic_regression.LogisticRegression(
        name=FLAGS.name, loss_func=FLAGS.loss_func,
        learning_rate=FLAGS.learning_rate, num_epochs=FLAGS.num_epochs,
        batch_size=FLAGS.batch_size)

    # Fit the model
    l.fit(trX, trY, vlX, vlY)

    # Test the model
    print('Test set accuracy: {}'.format(l.score(teX, teY)))
