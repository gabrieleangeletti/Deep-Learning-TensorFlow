"""Command line script to run LSTM model."""

import tensorflow as tf

from yadlt.models.recurrent.lstm import LSTM
from yadlt.utils import datasets
from yadlt.utils import utilities

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["ptb"]')
flags.DEFINE_string('ptb_dir', '', 'Path to the ptb dataset directory.')
flags.DEFINE_string('name', 'lstm', 'Model name.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0).\
    Useful for testing hyperparameters.')

# LSTM specific parameters
flags.DEFINE_integer('num_layers', 2, 'Number of layers.')
flags.DEFINE_integer('num_hidden', 200, 'Number of hidden units.')
flags.DEFINE_integer('vocab_size', 10000, 'Vocabulary size.')
flags.DEFINE_integer('batch_size', 20, 'Size of each mini-batch.')
flags.DEFINE_integer('num_steps', 35, 'Number of unrolled steps of LSTM.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
flags.DEFINE_float('learning_rate', 1.0, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.5, 'Dropout parameter.')
flags.DEFINE_float('init_scale', 0.05, 'initial scale of the weights.')
flags.DEFINE_integer('max_grad_norm', 5, 'Max norm of the gradient.')
flags.DEFINE_float('lr_decay', 0.8, 'lr decay after num_epochs/3.')

assert FLAGS.dataset in ['ptb']

if __name__ == '__main__':

    utilities.random_seed_np_tf(FLAGS.seed)

    if FLAGS.dataset == 'ptb':

        # ############### #
        #   PTB Dataset   #
        # ############### #

        trX, vlX, teX = datasets.load_ptb_dataset(FLAGS.ptb_dir)

    else:
        trX, vlX, teX = None, None, None

    model = LSTM(
        FLAGS.num_layers, FLAGS.num_hidden, FLAGS.vocab_size,
        FLAGS.batch_size, FLAGS.num_steps, FLAGS.num_epochs,
        FLAGS.learning_rate, FLAGS.dropout, FLAGS.init_scale,
        FLAGS.max_grad_norm, FLAGS.lr_decay
    )

    model.fit(trX, teX)
