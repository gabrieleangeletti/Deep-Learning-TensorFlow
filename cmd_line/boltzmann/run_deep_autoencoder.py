import numpy as np
import tensorflow as tf

from yadlt.models.boltzmann import deep_autoencoder
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
flags.DEFINE_string('name', 'srbm', 'Name of the model.')
flags.DEFINE_boolean('do_pretrain', True, 'Whether or not pretrain the network.')
flags.DEFINE_string('save_layers_output_test', '', 'Path to a .npy file to save test set output from all the layers of the model.')
flags.DEFINE_string('save_layers_output_train', '', 'Path to a .npy file to save train set output from all the layers of the model.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
flags.DEFINE_float('momentum', 0.7, 'Momentum parameter.')
flags.DEFINE_string('save_reconstructions', '', 'Path to a .npy file to save the reconstructions of the model.')

# RBMs layers specific parameters
flags.DEFINE_string('rbm_names', 'rbm', 'Name for the rbm stored_models.')
flags.DEFINE_string('rbm_layers', '256,', 'Comma-separated values for the layers in the srbm.')
flags.DEFINE_string('rbm_noise', 'gauss', 'Type of noise. Default: "gauss".')
flags.DEFINE_float('rbm_stddev', 0.1, 'Standard deviation for Gaussian visible units.')
flags.DEFINE_string('rbm_learning_rate', '0.01,', 'Initial learning rate.')
flags.DEFINE_string('rbm_num_epochs', '10,', 'Number of epochs.')
flags.DEFINE_string('rbm_batch_size', '10,', 'Size of each mini-batch.')
flags.DEFINE_string('rbm_gibbs_k', '1,', 'Gibbs sampling steps.')
# Supervised fine tuning parameters
flags.DEFINE_float('finetune_learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_string('finetune_enc_act_func', 'relu,', 'Activation function for the encoder fine-tuning phase. ["sigmoid, "tanh", "relu"]')
flags.DEFINE_string('finetune_dec_act_func', 'sigmoid,', 'Activation function for the decoder fine-tuning phase. ["sigmoid, "tanh", "relu"]')
flags.DEFINE_integer('finetune_num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('finetune_batch_size', 10, 'Size of each mini-batch.')
flags.DEFINE_string('finetune_opt', 'sgd', '["sgd", "ada_grad", "momentum", "adam"]')
flags.DEFINE_string('finetune_loss_func', 'mse', 'Loss function.')
flags.DEFINE_float('finetune_dropout', 1, 'Dropout parameter.')

# Conversion of Autoencoder layers parameters from string to their specific type
rbm_names = utilities.flag_to_list(FLAGS.rbm_names, 'str')
rbm_layers = utilities.flag_to_list(FLAGS.rbm_layers, 'int')
rbm_noise = utilities.flag_to_list(FLAGS.rbm_noise, 'str')
rbm_learning_rate = utilities.flag_to_list(FLAGS.rbm_learning_rate, 'float')
rbm_num_epochs = utilities.flag_to_list(FLAGS.rbm_num_epochs, 'int')
rbm_batch_size = utilities.flag_to_list(FLAGS.rbm_batch_size, 'int')
rbm_gibbs_k = utilities.flag_to_list(FLAGS.rbm_gibbs_k, 'int')

finetune_enc_act_func = utilities.flag_to_list(FLAGS.finetune_enc_act_func, 'str')
finetune_dec_act_func = utilities.flag_to_list(FLAGS.finetune_dec_act_func, 'str')

# Parameters validation
assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
assert len(rbm_layers) > 0

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

    finetune_enc_act_func = [utilities.str2actfunc(af) for af in finetune_enc_act_func]
    finetune_dec_act_func = [utilities.str2actfunc(af) for af in finetune_dec_act_func]

    # Create the object
    srbm = deep_autoencoder.DeepAutoencoder(
        name=FLAGS.name, do_pretrain=FLAGS.do_pretrain,
        layers=rbm_layers,
        learning_rate=rbm_learning_rate, gibbs_k=rbm_gibbs_k,
        num_epochs=rbm_num_epochs, momentum=FLAGS.momentum,
        batch_size=rbm_batch_size, finetune_learning_rate=FLAGS.finetune_learning_rate,
        finetune_enc_act_func=finetune_enc_act_func, finetune_dec_act_func=finetune_dec_act_func,
        finetune_num_epochs=FLAGS.finetune_num_epochs, finetune_batch_size=FLAGS.finetune_batch_size,
        finetune_opt=FLAGS.finetune_opt, finetune_loss_func=FLAGS.finetune_loss_func, finetune_dropout=FLAGS.finetune_dropout,
        noise=rbm_noise, stddev=FLAGS.rbm_stddev)


    def load_params_npz(npzfilepath):
        params = []
        npzfile = np.load(npzfilepath)
        for f in npzfile.files:
            params.append(npzfile[f])
        return params

    if FLAGS.do_pretrain:
        encoded_X, encoded_vX = srbm.pretrain(trX, vlX)

    # Supervised finetuning
    srbm.fit(trX, trRef, vlX, vlRef)

    # Compute the reconstruction loss of the model
    print('Test set reconstruction loss: {}'.format(srbm.compute_reconstruction_loss(teX, teRef)))

    # Save the predictions of the model
    if FLAGS.save_reconstructions:
        print('Saving the reconstructions for the test set...')
        np.save(FLAGS.save_reconstructions, srbm.reconstruct(teX))


    def save_layers_output(which_set):
        if which_set == 'train':
            trout = srbm.get_layers_output(trX)
            for i, o in enumerate(trout):
                np.save(FLAGS.save_layers_output_train + '-layer-' + str(i + 1) + '-train', o)

        elif which_set == 'test':
            teout = srbm.get_layers_output(teX)
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
