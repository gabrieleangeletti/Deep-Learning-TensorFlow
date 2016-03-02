import tensorflow as tf

from tf_models import logistic_regression
from utils import datasets

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10"]')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_string('directory_name', 'lr/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_string('loss_func', 'cross_entropy', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_integer('verbose', 0, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_string('optimizer', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')
flags.DEFINE_integer('n_iter', 10, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 10, 'Size of each mini-batch.')

assert FLAGS.dataset in ['mnist', 'cifar10']
assert FLAGS.loss_func in ['cross_entropy', 'mean_squared']
assert FLAGS.optimizer in ['gradient_descent', 'ada_grad', 'momentum']

if __name__ == '__main__':

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

    else:  # cannot be reached, just for completeness
        trX = None
        trY = None
        vlX = None
        vlY = None
        teX = None
        teY = None

    # Create the object
    l = logistic_regression.LogisticRegression(
        dataset=FLAGS.dataset, loss_func=FLAGS.loss_func, directory_name=FLAGS.directory_name,
        optimizer=FLAGS.optimizer, learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum,
        verbose=FLAGS.verbose, n_iter=FLAGS.n_iter, batch_size=FLAGS.batch_size)

    # Fit the model
    l.fit(trX, trY, vlX, vlY)

    # Test the model
    print('Test set accuracy: {}'.format(l.predict(teX, teY)))
