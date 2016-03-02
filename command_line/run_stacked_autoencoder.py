import tensorflow as tf

from tf_models.autoencoder_models import stacked_denoising_autoencoder
from utils import datasets

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10"]')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')

# Supervised fine tuning parameters
flags.DEFINE_string('softmax_loss_func', 'cross_entropy', 'Last Layer Loss function.["cross_entropy", "mean_squared"]')
flags.DEFINE_integer('finetune_num_epochs', 30, 'Number of epochs for the fine-tuning phase.')
flags.DEFINE_float('finetune_learning_rate', 0.001, 'Learning rate for the fine-tuning phase.')
flags.DEFINE_string('finetune_opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum"]')
flags.DEFINE_integer('finetune_batch_size', 20, 'Size of each mini-batch for the fine-tuning phase.')
flags.DEFINE_integer('verbose', 0, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_string('main_dir', 'sdae/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_string('corr_type', 'none,', 'Type of input corruption. ["none", "masking", "salt_and_pepper"]')
flags.DEFINE_float('corr_frac', 0.0, 'Fraction of the input to corrupt.')
# Autoencoder layers specific parameters
flags.DEFINE_string('layers', '256,', 'Comma-separated values for the layers in the sdae.')
flags.DEFINE_string('xavier_init', '1,', 'Value for the constant in xavier weights initialization.')
flags.DEFINE_string('enc_act_func', 'tanh,', 'Activation function for the encoder. ["sigmoid", "tanh"]')
flags.DEFINE_string('dec_act_func', 'none,', 'Activation function for the decoder. ["sigmoid", "tanh", "none"]')
flags.DEFINE_string('loss_func', 'mean_squared,', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_string('opt', 'gradient_descent,', '["gradient_descent", "ada_grad", "momentum"]')
flags.DEFINE_string('learning_rate', '0.01,', 'Initial learning rate.')
flags.DEFINE_string('momentum', '0.5,', 'Momentum parameter.')
flags.DEFINE_string('num_epochs', '10,', 'Number of epochs.')
flags.DEFINE_string('batch_size', '10,', 'Size of each mini-batch.')

# Conversion of Autoencoder layers parameters from string to their specific type
layers = [int(_) for _ in FLAGS.layers.split(',') if _]
xavier_init = [int(_) for _ in FLAGS.xavier_init.split(',') if _]
enc_act_func = [_ for _ in FLAGS.enc_act_func.split(',') if _]
dec_act_func = [_ for _ in FLAGS.dec_act_func.split(',') if _]
loss_func = [_ for _ in FLAGS.loss_func.split(',') if _]
opt = [_ for _ in FLAGS.opt.split(',') if _]
learning_rate = [float(_) for _ in FLAGS.learning_rate.split(',') if _]
momentum = [float(_) for _ in FLAGS.momentum.split(',') if _]
num_epochs = [int(_) for _ in FLAGS.num_epochs.split(',') if _]
batch_size = [int(_) for _ in FLAGS.batch_size.split(',') if _]

# Parameters normalization: if a parameter is not specified, it must be made of the same length of the others
dae_params = {'layers': layers, 'xavier_init': xavier_init, 'enc_act_func': enc_act_func,
              'dec_act_func': dec_act_func, 'loss_func': loss_func, 'opt': opt,
              'learning_rate': learning_rate, 'momentum': momentum, 'num_epochs': num_epochs, 'batch_size': batch_size}

for p in dae_params:
    if len(dae_params[p]) != len(layers):
        # The current parameter is not specified by the user, should default it for all the layers
        dae_params[p] = [dae_params[p][0] for _ in layers]

# Parameters validation
assert 0. <= FLAGS.corr_frac <= 1.
assert FLAGS.corr_type in ['masking', 'salt_and_pepper', 'none']
assert FLAGS.dataset in ['mnist', 'cifar10']
assert len(layers) > 0
assert all([af in ['sigmoid', 'tanh'] for af in enc_act_func])
assert all([af in ['sigmoid', 'tanh', 'none'] for af in dec_act_func])
assert all([lf in ['cross_entropy', 'mean_squared'] for lf in loss_func])
assert all([o in ['gradient_descent', 'ada_grad', 'momentum'] for o in opt])

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
    sdae = stacked_denoising_autoencoder.StackedDenoisingAutoencoder(
        layers=dae_params['layers'], seed=FLAGS.seed, softmax_loss_func=FLAGS.softmax_loss_func,
        finetune_learning_rate=FLAGS.finetune_learning_rate, finetune_num_epochs=FLAGS.finetune_num_epochs,
        finetune_opt=FLAGS.finetune_opt, finetune_batch_size=FLAGS.finetune_batch_size,
        enc_act_func=dae_params['enc_act_func'], dec_act_func=dae_params['dec_act_func'],
        xavier_init=dae_params['xavier_init'], corr_type=FLAGS.corr_type, corr_frac=FLAGS.corr_frac,
        dataset=FLAGS.dataset, loss_func=dae_params['loss_func'], main_dir=FLAGS.main_dir, opt=dae_params['opt'],
        learning_rate=dae_params['learning_rate'], momentum=dae_params['momentum'], verbose=FLAGS.verbose,
        num_epochs=dae_params['num_epochs'], batch_size=dae_params['batch_size'])

    # Fit the model (unsupervised pretraining)
    encoded_X, encoded_vX = sdae.pretrain(trX, vlX)

    import numpy as np
    np.save('stored_models/sdae/testTRAINX', encoded_X)
    np.save('stored_models/sdae/testVALX', encoded_vX)

    # Supervised finetuning
    sdae.finetune(trX, trY, vlX, vlY)

    # Test the model
    print('Test set accuracy: {}'.format(sdae.predict(teX, teY)))
