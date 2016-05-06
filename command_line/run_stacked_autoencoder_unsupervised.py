import tensorflow as tf
import numpy as np

from models.autoencoder_models import stacked_deep_autoencoder
from utils import datasets, utilities

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10", "custom"]')
flags.DEFINE_string('train_dataset', '', 'Path to train set data .npy file.')
flags.DEFINE_string('train_ref', '', 'Path to train reference .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('valid_ref', '', 'Path to valid reference data .npy file.')
flags.DEFINE_string('test_dataset', '', 'Path to test set .npy file.')
flags.DEFINE_string('test_ref', '', 'Path to test reference data .npy file.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
flags.DEFINE_boolean('do_pretrain', True, 'Whether or not doing unsupervised pretraining.')
flags.DEFINE_string('save_reconstructions', '', 'Path to a .npy file to save the reconstructions of the model.')
flags.DEFINE_string('save_layers_output', '', 'Path to a .npy file to save output from all the layers of the model.')
flags.DEFINE_string('encweights', None, 'Path to a npz array containing the weights of the encoding layers.')
flags.DEFINE_string('encbiases', None, 'Path to a npz array containing the encoding layers biases.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_string('model_name', 'un_sdae', 'Name for the model.')

# Supervised fine tuning parameters
flags.DEFINE_string('finetune_loss_func', 'cross_entropy', 'Last Layer Loss function.["cross_entropy", "mean_squared"]')
flags.DEFINE_integer('finetune_num_epochs', 30, 'Number of epochs for the fine-tuning phase.')
flags.DEFINE_float('finetune_learning_rate', 0.001, 'Learning rate for the fine-tuning phase.')
flags.DEFINE_string('finetune_act_func', 'relu', 'Activation function for the fine-tuning phase.'
                                                 '["sigmoid, "tanh", "relu"]')
flags.DEFINE_float('dropout', 1, 'Dropout parameter.')
flags.DEFINE_string('finetune_opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum"]')
flags.DEFINE_integer('finetune_batch_size', 20, 'Size of each mini-batch for the fine-tuning phase.')
flags.DEFINE_integer('verbose', 0, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_string('main_dir', 'un_sdae/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_string('corr_type', 'none', 'Type of input corruption. ["none", "masking", "salt_and_pepper"]')
flags.DEFINE_float('corr_frac', 0.0, 'Fraction of the input to corrupt.')
# Autoencoder layers specific parameters
flags.DEFINE_string('layers', '256,', 'Comma-separated values for the layers in the sdae.')
flags.DEFINE_string('xavier_init', '1,', 'Value for the constant in xavier weights initialization.')
flags.DEFINE_string('enc_act_func', 'sigmoid,', 'Activation function for the encoder. ["sigmoid", "tanh"]')
flags.DEFINE_string('dec_act_func', 'none,', 'Activation function for the decoder. ["sigmoid", "tanh", "none"]')
flags.DEFINE_string('loss_func', 'mean_squared,', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_string('opt', 'gradient_descent,', '["gradient_descent", "ada_grad", "momentum", "adam"]')
flags.DEFINE_string('learning_rate', '0.01,', 'Initial learning rate.')
flags.DEFINE_string('momentum', '0.5,', 'Momentum parameter.')
flags.DEFINE_string('num_epochs', '10,', 'Number of epochs.')
flags.DEFINE_string('batch_size', '10,', 'Size of each mini-batch.')

# Conversion of Autoencoder layers parameters from string to their specific type
layers = [int(_) for _ in FLAGS.layers.split(',') if _]
xavier_init = [int(_) for _ in FLAGS.xavier_init.split(',') if _]
enc_act_func = [_ for _ in FLAGS.enc_act_func.split(',') if _]
dec_act_func = [_ for _ in FLAGS.dec_act_func.split(',') if _]
opt = [_ for _ in FLAGS.opt.split(',') if _]
loss_func = [_ for _ in FLAGS.loss_func.split(',') if _]
learning_rate = [float(_) for _ in FLAGS.learning_rate.split(',') if _]
momentum = [float(_) for _ in FLAGS.momentum.split(',') if _]
num_epochs = [int(_) for _ in FLAGS.num_epochs.split(',') if _]
batch_size = [int(_) for _ in FLAGS.batch_size.split(',') if _]

# Parameters normalization: if a parameter is not specified, it must be made of the same length of the others
dae_params = {'layers': layers, 'xavier_init': xavier_init, 'enc_act_func': enc_act_func,
              'dec_act_func': dec_act_func, 'loss_func': loss_func, 'learning_rate': learning_rate,
              'opt': opt,
              'momentum': momentum, 'num_epochs': num_epochs, 'batch_size': batch_size}

for p in dae_params:
    if len(dae_params[p]) != len(layers):
        # The current parameter is not specified by the user, should default it for all the layers
        dae_params[p] = [dae_params[p][0] for _ in layers]

# Parameters validation
assert 0. <= FLAGS.corr_frac <= 1.
assert FLAGS.corr_type in ['masking', 'salt_and_pepper', 'none']
assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
assert len(layers) > 0
assert all([af in ['sigmoid', 'tanh'] for af in enc_act_func])
assert all([af in ['sigmoid', 'tanh', 'none'] for af in dec_act_func])
assert all([lf in ['cross_entropy', 'mean_squared'] for lf in loss_func])
assert FLAGS.finetune_opt in ['gradient_descent', 'ada_grad', 'momentum', 'adam']

if __name__ == '__main__':

    utilities.random_seed_np_tf(FLAGS.seed)

    if FLAGS.dataset == 'mnist':

        # ################# #
        #   MNIST Dataset   #
        # ################# #

        trX, vlX, teX = datasets.load_mnist_dataset(mode='unsupervised')
        trRef = trX
        vlRef = vlX
        teRef = teX

    elif FLAGS.dataset == 'cifar10':

        # ################### #
        #   Cifar10 Dataset   #
        # ################### #

        trX, teX = datasets.load_cifar10_dataset(FLAGS.cifar_dir, mode='unsupervised')
        # Validation set is the first half of the test set
        vlX = teX[:5000]
        trRef = trX
        vlRef = vlX
        teRef = teX

    elif FLAGS.dataset == 'custom':

        # ################## #
        #   Custom Dataset   #
        # ################## #

        def load_from_np(dataset_path):
            if dataset_path != '':
                return np.load(dataset_path)
            else:
                return None

        trX, trRef = load_from_np(FLAGS.train_dataset), load_from_np(FLAGS.train_ref)
        vlX, vlRef = load_from_np(FLAGS.valid_dataset), load_from_np(FLAGS.valid_ref)
        teX, teRef = load_from_np(FLAGS.test_dataset), load_from_np(FLAGS.test_ref)

        if not trRef:
            trRef = trX
        if not vlRef:
            vlRef = vlX
        if not teRef:
            teRef = teX

    else:
        trX = None
        trRef = None
        vlX = None
        vlRef = None
        teX = None
        teRef = None

    # Create the object
    sdae = None

    sdae = stacked_deep_autoencoder.StackedDeepAutoencoder(
        do_pretrain=FLAGS.do_pretrain, model_name=FLAGS.model_name,
        layers=dae_params['layers'], finetune_loss_func=FLAGS.finetune_loss_func,
        finetune_learning_rate=FLAGS.finetune_learning_rate, finetune_num_epochs=FLAGS.finetune_num_epochs,
        finetune_opt=FLAGS.finetune_opt, finetune_batch_size=FLAGS.finetune_batch_size, dropout=FLAGS.dropout,
        enc_act_func=dae_params['enc_act_func'], dec_act_func=dae_params['dec_act_func'],
        xavier_init=dae_params['xavier_init'], corr_type=FLAGS.corr_type, corr_frac=FLAGS.corr_frac,
        dataset=FLAGS.dataset, loss_func=dae_params['loss_func'], main_dir=FLAGS.main_dir, opt=dae_params['opt'],
        learning_rate=dae_params['learning_rate'], momentum=dae_params['momentum'], verbose=FLAGS.verbose,
        num_epochs=dae_params['num_epochs'], batch_size=dae_params['batch_size'],
        finetune_act_func=FLAGS.finetune_act_func)

    def load_params_npz(npzfilepath):
        params = []
        npzfile = np.load(npzfilepath)
        for f in npzfile.files:
            params.append(npzfile[f])
        return params

    encodingw = None
    encodingb = None

    # Fit the model (unsupervised pretraining)
    if FLAGS.encweights and FLAGS.encbiases:
        encodingw = load_params_npz(FLAGS.encweights)
        encodingb = load_params_npz(FLAGS.encbiases)
    elif FLAGS.do_pretrain:
        encoded_X, encoded_vX = sdae.pretrain(trX, vlX)

    # Supervised finetuning
    sdae.build_model(trX.shape[1], encodingw, encodingb)
    sdae.fit(trX, trRef, vlX, vlRef, restore_previous_model=FLAGS.restore_previous_model)

    # Compute the reconstruction loss of the model
    print('Test set reconstruction loss: {}'.format(sdae.compute_reconstruction_loss(teX, teRef)))

    # Save the predictions of the model
    if FLAGS.save_reconstructions:
        print('Saving the reconstructions for the test set...')
        np.save(FLAGS.save_reconstructions, sdae.reconstruct(teX))

    # Save output from each layer of the model
    if FLAGS.save_layers_output:
        print('Saving the output of each layer for the test set')
        out = sdae.get_layers_output(teX)
        for i, o in enumerate(out):
            np.save(FLAGS.save_layers_output + '-layer-' + str(i + 1), o)



