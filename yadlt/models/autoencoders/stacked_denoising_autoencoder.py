"""Implementation of Stacked Denoising Autoencoders for Supervised Learning."""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from yadlt.core import SupervisedModel
from yadlt.core import Evaluation, Layers, Loss, Trainer
from yadlt.models.autoencoders import denoising_autoencoder
from yadlt.utils import tf_utils, utilities


class StackedDenoisingAutoencoder(SupervisedModel):
    """Implementation of Stacked Denoising Autoencoders using TensorFlow.

    The interface of the class is sklearn-like.
    """

    def __init__(
        self, layers, name='sdae',
        enc_act_func=[tf.nn.tanh], dec_act_func=[None],
        loss_func=['cross_entropy'], num_epochs=[10], batch_size=[10],
        opt=['sgd'], regcoef=[5e-4], learning_rate=[0.01], momentum=0.5,
        finetune_dropout=1, corr_type=['none'], corr_frac=[0.],
        finetune_loss_func='softmax_cross_entropy',
        finetune_act_func=tf.nn.relu, finetune_opt='sgd',
        finetune_learning_rate=0.001, finetune_num_epochs=10,
            finetune_batch_size=20, do_pretrain=False):
        """Constructor.

        :param layers: list containing the hidden units for each layer
        :param enc_act_func: Activation function for the encoder.
            [tf.nn.tanh, tf.nn.sigmoid]
        :param dec_act_func: Activation function for the decoder.
            [tf.nn.tanh, tf.nn.sigmoid, None]
        :param finetune_loss_func: Loss function for the softmax layer.
            string, default ['softmax_cross_entropy', 'mse']
        :param finetune_dropout: dropout parameter
        :param finetune_learning_rate: learning rate for the finetuning.
            float, default 0.001
        :param finetune_act_func: activation function for the finetuning phase
        :param finetune_opt: optimizer for the finetuning phase
        :param finetune_num_epochs: Number of epochs for the finetuning.
            int, default 20
        :param finetune_batch_size: Size of each mini-batch for the finetuning.
            int, default 20
        :param corr_type: Type of input corruption. string, default 'none'.
            ["none", "masking", "salt_and_pepper"]
        :param corr_frac: Fraction of the input to corrupt. float, default 0.0
        :param do_pretrain: True: uses variables from pretraining,
            False: initialize new variables.
        """
        # WARNING! This must be the first expression in the function or else it
        # will send other variables to expanded_args()
        # This function takes all the passed parameters that are lists and
        # expands them to the number of layers, if the number
        # of layers is more than the list of the parameter.
        expanded_args = utilities.expand_args(**locals())

        SupervisedModel.__init__(self, name)

        self.loss_func = finetune_loss_func
        self.learning_rate = finetune_learning_rate
        self.opt = finetune_opt
        self.num_epochs = finetune_num_epochs
        self.batch_size = finetune_batch_size
        self.momentum = momentum
        self.dropout = finetune_dropout

        self.loss = Loss(self.loss_func)
        self.trainer = Trainer(
            finetune_opt, learning_rate=finetune_learning_rate,
            momentum=momentum)

        self.do_pretrain = do_pretrain
        self.layers = layers
        self.finetune_act_func = finetune_act_func

        # Model parameters
        self.encoding_w_ = []  # list of matrices of encoding weights per layer
        self.encoding_b_ = []  # list of arrays of encoding biases per layer

        self.last_W = None
        self.last_b = None

        self.autoencoders = []
        self.autoencoder_graphs = []

        for l, layer in enumerate(layers):
            dae_str = 'dae-' + str(l + 1)

            self.autoencoders.append(
                denoising_autoencoder.DenoisingAutoencoder(
                    n_components=layer,
                    name=self.name + '-' + dae_str,
                    enc_act_func=expanded_args['enc_act_func'][l],
                    dec_act_func=expanded_args['dec_act_func'][l],
                    loss_func=expanded_args['loss_func'][l],
                    opt=expanded_args['opt'][l], regcoef=expanded_args['regcoef'],
                    learning_rate=expanded_args['learning_rate'][l],
                    momentum=self.momentum,
                    corr_type=expanded_args['corr_type'][l],
                    corr_frac=expanded_args['corr_frac'][l],
                    num_epochs=expanded_args['num_epochs'][l],
                    batch_size=expanded_args['batch_size'][l]))

            self.autoencoder_graphs.append(tf.Graph())

    def pretrain(self, train_set, validation_set=None):
        """Perform Unsupervised pretraining of the autoencoder."""
        self.do_pretrain = True

        def set_params_func(autoenc, autoencgraph):
            params = autoenc.get_parameters(graph=autoencgraph)
            self.encoding_w_.append(params['enc_w'])
            self.encoding_b_.append(params['enc_b'])

        return SupervisedModel.pretrain_procedure(
            self, self.autoencoders, self.autoencoder_graphs,
            set_params_func=set_params_func, train_set=train_set,
            validation_set=validation_set)

    def _train_model(self, train_set, train_labels,
                     validation_set, validation_labels):
        """Train the model.

        :param train_set: training set
        :param train_labels: training labels
        :param validation_set: validation set
        :param validation_labels: validation labels
        :return: self
        """
        shuff = list(zip(train_set, train_labels))

        pbar = tqdm(range(self.num_epochs))
        for i in pbar:

            np.random.shuffle(shuff)

            batches = [_ for _ in utilities.gen_batches(
                shuff, self.batch_size)]

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.tf_session.run(
                    self.train_step,
                    feed_dict={self.input_data: x_batch,
                               self.input_labels: y_batch,
                               self.keep_prob: self.dropout})

            if validation_set is not None:
                feed = {self.input_data: validation_set,
                        self.input_labels: validation_labels,
                        self.keep_prob: 1}
                acc = tf_utils.run_summaries(
                    self.tf_session, self.tf_merged_summaries,
                    self.tf_summary_writer, i, feed, self.accuracy)
                pbar.set_description("Accuracy: %s" % (acc))

    def build_model(self, n_features, n_classes):
        """Create the computational graph.

        This graph is intented to be created for finetuning,
        i.e. after unsupervised pretraining.
        :param n_features: Number of features.
        :param n_classes: number of classes.
        :return: self
        """
        self._create_placeholders(n_features, n_classes)
        self._create_variables(n_features)

        next_train = self._create_encoding_layers()
        self.mod_y, _, _ = Layers.linear(next_train, n_classes)
        self.layer_nodes.append(self.mod_y)

        self.cost = self.loss.compile(self.mod_y, self.input_labels)
        self.train_step = self.trainer.compile(self.cost)
        self.accuracy = Evaluation.accuracy(self.mod_y, self.input_labels)

    def _create_placeholders(self, n_features, n_classes):
        """Create the TensorFlow placeholders for the model.

        :param n_features: number of features of the first layer
        :param n_classes: number of classes
        :return: self
        """
        self.input_data = tf.placeholder(
            tf.float32, [None, n_features], name='x-input')
        self.input_labels = tf.placeholder(
            tf.float32, [None, n_classes], name='y-input')
        self.keep_prob = tf.placeholder(
            tf.float32, name='keep-probs')

    def _create_variables(self, n_features):
        """Create the TensorFlow variables for the model.

        :param n_features: number of features
        :return: self
        """
        if self.do_pretrain:
            self._create_variables_pretrain()
        else:
            self._create_variables_no_pretrain(n_features)

    def _create_variables_no_pretrain(self, n_features):
        """Create model variables (no previous unsupervised pretraining).

        :param n_features: number of features
        :return: self
        """
        self.encoding_w_ = []
        self.encoding_b_ = []

        for l, layer in enumerate(self.layers):

            if l == 0:
                self.encoding_w_.append(tf.Variable(
                    tf.truncated_normal(
                        shape=[n_features, self.layers[l]], stddev=0.1)))
                self.encoding_b_.append(tf.Variable(
                    tf.constant(0.1, shape=[self.layers[l]])))
            else:
                self.encoding_w_.append(tf.Variable(
                    tf.truncated_normal(
                        shape=[self.layers[l-1], self.layers[l]], stddev=0.1)))
                self.encoding_b_.append(tf.Variable(
                    tf.constant(0.1, shape=[self.layers[l]])))

    def _create_variables_pretrain(self):
        """Create model variables (previous unsupervised pretraining).

        :return: self
        """
        for l, layer in enumerate(self.layers):
            self.encoding_w_[l] = tf.Variable(
                self.encoding_w_[l], name='enc-w-{}'.format(l))
            self.encoding_b_[l] = tf.Variable(
                self.encoding_b_[l], name='enc-b-{}'.format(l))

    def _create_encoding_layers(self):
        """Create the encoding layers for supervised finetuning.

        :return: output of the final encoding layer.
        """
        next_train = self.input_data
        self.layer_nodes = []

        for l, layer in enumerate(self.layers):

            with tf.name_scope("encode-{}".format(l)):

                y_act = tf.add(
                    tf.matmul(next_train, self.encoding_w_[l]),
                    self.encoding_b_[l]
                )

                if self.finetune_act_func:
                    layer_y = self.finetune_act_func(y_act)

                else:
                    layer_y = None

                # the input to the next layer is the output of this layer
                next_train = tf.nn.dropout(layer_y, self.keep_prob)

            self.layer_nodes.append(next_train)

        return next_train
