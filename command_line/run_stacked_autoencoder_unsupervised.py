import tensorflow as tf
import numpy as np
import os

import config

from yadlt.models.autoencoder_models import stacked_deep_autoencoder
from yadlt.utils import datasets, utilities

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
flags.DEFINE_string('save_layers_output_test', '', 'Path to a .npy file to save test set output from all the layers of the model.')
flags.DEFINE_string('save_layers_output_train', '', 'Path to a .npy file to save train set output from all the layers of the model.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_string('model_name', 'un_sdae', 'Name for the model.')
flags.DEFINE_string('main_dir', 'un_sdae/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_integer('verbose', 0, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')
flags.DEFINE_boolean('tied_weights', True, 'Whether to use tied weights for the decoders.')

# Supervised fine tuning parameters
flags.DEFINE_string('finetune_loss_func', 'cross_entropy', 'Last Layer Loss function.["cross_entropy", "mean_squared"]')
flags.DEFINE_integer('finetune_num_epochs', 30, 'Number of epochs for the fine-tuning phase.')
flags.DEFINE_float('finetune_learning_rate', 0.001, 'Learning rate for the fine-tuning phase.')
flags.DEFINE_string('finetune_enc_act_func', 'relu', 'Activation function for the encoder fine-tuning phase. ["sigmoid, "tanh", "relu"]')
flags.DEFINE_string('finetune_dec_act_func', 'sigmoid', 'Activation function for the decoder fine-tuning phase. ["sigmoid, "tanh", "relu"]')
flags.DEFINE_float('finetune_dropout', 1, 'Dropout parameter.')
flags.DEFINE_string('finetune_opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum"]')
flags.DEFINE_integer('finetune_batch_size', 20, 'Size of each mini-batch for the fine-tuning phase.')

# Autoencoder layers specific parameters
flags.DEFINE_string('dae_layers', '256,', 'Comma-separated values for the layers in the sdae.')
flags.DEFINE_string('dae_l2reg', '5e-4,', 'Regularization parameter for the autoencoders. If 0, no regularization.')
flags.DEFINE_string('dae_enc_act_func', 'sigmoid,', 'Activation function for the encoder. ["sigmoid", "tanh"]')
flags.DEFINE_string('dae_dec_act_func', 'none,', 'Activation function for the decoder. ["sigmoid", "tanh", "none"]')
flags.DEFINE_string('dae_loss_func', 'mean_squared,', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_string('dae_opt', 'gradient_descent,', '["gradient_descent", "ada_grad", "momentum", "adam"]')
flags.DEFINE_string('dae_learning_rate', '0.01,', 'Initial learning rate.')
flags.DEFINE_string('dae_num_epochs', '10,', 'Number of epochs.')
flags.DEFINE_string('dae_batch_size', '10,', 'Size of each mini-batch.')
flags.DEFINE_string('dae_corr_type', 'none,', 'Type of input corruption. ["none", "masking", "salt_and_pepper"]')
flags.DEFINE_string('dae_corr_frac', '0.0,', 'Fraction of the input to corrupt.')

# Conversion of Autoencoder layers parameters from string to their specific type
dae_layers = utilities.flag_to_list(FLAGS.dae_layers, 'int')
dae_enc_act_func = utilities.flag_to_list(FLAGS.dae_enc_act_func, 'str')
dae_dec_act_func = utilities.flag_to_list(FLAGS.dae_dec_act_func, 'str')
dae_opt = utilities.flag_to_list(FLAGS.dae_opt, 'str')
dae_loss_func = utilities.flag_to_list(FLAGS.dae_loss_func, 'str')
dae_learning_rate = utilities.flag_to_list(FLAGS.dae_learning_rate, 'float')
dae_l2reg = utilities.flag_to_list(FLAGS.dae_l2reg, 'float')
dae_corr_type = utilities.flag_to_list(FLAGS.dae_corr_type, 'str')
dae_corr_frac = utilities.flag_to_list(FLAGS.dae_corr_frac, 'float')
dae_num_epochs = utilities.flag_to_list(FLAGS.dae_num_epochs, 'int')
dae_batch_size = utilities.flag_to_list(FLAGS.dae_batch_size, 'int')

# Parameters validation
assert all([0. <= cf <= 1. for cf in dae_corr_frac])
assert all([ct in ['masking', 'salt_and_pepper', 'none'] for ct in dae_corr_type])
assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
assert len(dae_layers) > 0
assert all([af in ['sigmoid', 'tanh'] for af in dae_enc_act_func])
assert all([af in ['sigmoid', 'tanh', 'none'] for af in dae_dec_act_func])
assert all([lf in ['cross_entropy', 'mean_squared'] for lf in dae_loss_func])
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

    models_dir = os.path.join(config.models_dir, FLAGS.main_dir)
    data_dir = os.path.join(config.data_dir, FLAGS.main_dir)
    summary_dir = os.path.join(config.summary_dir, FLAGS.main_dir)

    sdae = stacked_deep_autoencoder.StackedDeepAutoencoder(
        models_dir=models_dir, data_dir=data_dir, summary_dir=summary_dir,
        do_pretrain=FLAGS.do_pretrain, model_name=FLAGS.model_name,
        dae_layers=dae_layers, finetune_loss_func=FLAGS.finetune_loss_func,
        finetune_learning_rate=FLAGS.finetune_learning_rate, finetune_num_epochs=FLAGS.finetune_num_epochs,
        finetune_opt=FLAGS.finetune_opt, finetune_batch_size=FLAGS.finetune_batch_size,
        finetune_dropout=FLAGS.finetune_dropout,
        dae_enc_act_func=dae_enc_act_func, dae_dec_act_func=dae_dec_act_func,
        dae_corr_type=dae_corr_type, dae_corr_frac=dae_corr_frac, dae_l2reg=dae_l2reg,
        dataset=FLAGS.dataset, dae_loss_func=dae_loss_func, main_dir=FLAGS.main_dir,
        dae_opt=dae_opt, tied_weights=FLAGS.tied_weights,
        dae_learning_rate=dae_learning_rate, momentum=FLAGS.momentum, verbose=FLAGS.verbose,
        dae_num_epochs=dae_num_epochs, dae_batch_size=dae_batch_size,
        finetune_enc_act_func=FLAGS.finetune_enc_act_func, finetune_dec_act_func=FLAGS.finetune_dec_act_func)

    def load_params_npz(npzfilepath):
        params = []
        npzfile = np.load(npzfilepath)
        for f in npzfile.files:
            params.append(npzfile[f])
        return params

    encodingw = None
    encodingb = None

    # Fit the model (unsupervised pretraining)
    if FLAGS.do_pretrain:
        encoded_X, encoded_vX = sdae.pretrain(trX, vlX)

    # Supervised finetuning
    sdae.fit(trX, trRef, vlX, vlRef, restore_previous_model=FLAGS.restore_previous_model)

    # Compute the reconstruction loss of the model
    print('Test set reconstruction loss: {}'.format(sdae.compute_reconstruction_loss(teX, teRef)))

    # Save the predictions of the model
    if FLAGS.save_reconstructions:
        print('Saving the reconstructions for the test set...')
        np.save(FLAGS.save_reconstructions, sdae.reconstruct(teX))

    def save_layers_output(which_set):
        if which_set == 'train':
            trout = sdae.get_layers_output(trX)
            for i, o in enumerate(trout):
                np.save(FLAGS.save_layers_output_train + '-layer-' + str(i + 1) + '-train', o)

        elif which_set == 'test':
            teout = sdae.get_layers_output(teX)
            for i, o in enumerate(teout):
                np.save(FLAGS.save_layers_output_test + '-layer-' + str(i + 1) + '-test', o)

    # Save output from each layer of the model
    if FLAGS.save_layers_output_test:
        print('Saving the output of each layer for the test set')
        save_layers_output('test')

    # Save output from each layer of the model
    if FLAGS.save_layers_output_train:
        print('Saving the output of each layer for the train set')
        save_layers_output('train')

