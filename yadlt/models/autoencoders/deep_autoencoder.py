"""Implementation of Stacked Denoising Autoencoders for Unsup. Learning."""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from yadlt.core import Layers, Loss, Trainer
from yadlt.core import UnsupervisedModel
from yadlt.models.autoencoders import denoising_autoencoder
from yadlt.utils import tf_utils, utilities


class DeepAutoencoder(UnsupervisedModel):
    """Implementation of Stacked Denoising Autoencoders using TensorFlow.

    The interface of the class is sklearn-like.
    """

    def __init__(
        self, layers, name='sdae',
        enc_act_func=[tf.nn.tanh], dec_act_func=[None],
        loss_func=['cross_entropy'], num_epochs=[10], batch_size=[10],
        opt=['sgd'], regtype=['none'], regcoef=[5e-4],
        learning_rate=[0.01], momentum=0.5, finetune_dropout=1,
        corr_type=['none'], finetune_regtype='none', corr_frac=[0.],
        finetune_loss_func='cross_entropy', finetune_enc_act_func=[tf.nn.relu],
        tied_weights=True, finetune_dec_act_func=[tf.nn.sigmoid],
        finetune_batch_size=20, do_pretrain=False,
        finetune_opt='sgd', finetune_learning_rate=0.001,
            finetune_num_epochs=10):
        """Constructor.

        :param layers: list containing the hidden units for each layer
        :param enc_act_func: Activation function for the encoder.
            [tf.nn.tanh, tf.nn.sigmoid]
        :param dec_act_func: Activation function for the decoder.
            [tf.nn.tanh, tf.nn.sigmoid, None]
        :param regtype: regularization type, can be 'l2','l1' and 'none'
        :param finetune_regtype: regularization type for finetuning
        :param finetune_loss_func: Loss function for the softmax layer.
            string, default ['cross_entropy', 'mse']
        :param finetune_dropout: dropout parameter
        :param finetune_learning_rate: learning rate for the finetuning.
            float, default 0.001
        :param finetune_enc_act_func: activation function for the encoder
            finetuning phase
        :param finetune_dec_act_func: activation function for the decoder
            finetuning phase
        :param finetune_opt: optimizer for the finetuning phase
        :param finetune_num_epochs: Number of epochs for the finetuning.
            int, default 20
        :param finetune_batch_size: Size of each mini-batch for the finetuning.
            int, default 20
        :param tied_weights: if True, the decoder layers weights are
            constrained to be the transpose of the encoder layers
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

        UnsupervisedModel.__init__(self, name)

        self.loss_func = finetune_loss_func
        self.learning_rate = finetune_learning_rate
        self.opt = finetune_opt
        self.num_epochs = finetune_num_epochs
        self.batch_size = finetune_batch_size
        self.momentum = momentum
        self.dropout = finetune_dropout
        self.regtype = finetune_regtype
        self.regcoef = regcoef

        self.loss = Loss(self.loss_func)
        self.trainer = Trainer(
            finetune_opt, learning_rate=finetune_learning_rate,
            momentum=momentum)

        self.do_pretrain = do_pretrain
        self.layers = layers
        self.tied_weights = tied_weights

        self.finetune_enc_act_func = expanded_args['finetune_enc_act_func']
        self.finetune_dec_act_func = expanded_args['finetune_dec_act_func']

        self.input_ref = None

        # Model parameters
        self.encoding_w_ = []  # list of matrices of encoding weights per layer
        self.encoding_b_ = []  # list of arrays of encoding biases per layer

        self.decoding_w = []  # list of matrices of decoding weights per layer
        self.decoding_b = []  # list of arrays of decoding biases per layer

        self.reconstruction = None
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
                    regtype=expanded_args['regtype'][l],
                    opt=expanded_args['opt'][l],
                    learning_rate=expanded_args['learning_rate'][l],
                    regcoef=expanded_args['regcoef'],
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

        return UnsupervisedModel.pretrain_procedure(
            self, self.autoencoders, self.autoencoder_graphs,
            set_params_func=set_params_func, train_set=train_set,
            validation_set=validation_set)

    def _train_model(self, train_set, train_ref,
                     validation_set, validation_ref):
        """Train the model.

        :param train_set: training set
        :param train_ref: training reference data
        :param validation_set: validation set
        :param validation_ref: validation reference data
        :return: self
        """
        shuff = list(zip(train_set, train_ref))

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
                        self.input_labels: validation_ref, self.keep_prob: 1}
                err = tf_utils.run_summaries(
                    self.tf_session, self.tf_merged_summaries,
                    self.tf_summary_writer, i, feed, self.cost)
                pbar.set_description("Reconstruction loss: %s" % (err))

    def build_model(self, n_features, encoding_w=None, encoding_b=None):
        """Create the computational graph for the reconstruction task.

        :param n_features: Number of features
        :param encoding_w: list of weights for the encoding layers.
        :param encoding_b: list of biases for the encoding layers.
        :return: self
        """
        self._create_placeholders(n_features, n_features)

        if encoding_w and encoding_b:
            self.encoding_w_ = encoding_w
            self.encoding_b_ = encoding_b
        else:
            self._create_variables(n_features)

        self._create_encoding_layers()
        self._create_decoding_layers()

        variables = []
        variables.extend(self.encoding_w_)
        variables.extend(self.encoding_b_)
        regterm = Layers.regularization(variables, self.regtype, self.regcoef)

        self.cost = self.loss.compile(
            self.reconstruction, self.input_labels, regterm=regterm)
        self.train_step = self.trainer.compile(self.cost)

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
                    tf.truncated_normal([self.layers[l]], stddev=0.1)))
            else:
                self.encoding_w_.append(tf.Variable(
                    tf.truncated_normal(
                        shape=[self.layers[l - 1], self.layers[l]],
                        stddev=0.1)))
                self.encoding_b_.append(tf.Variable(
                    tf.truncated_normal([self.layers[l]], stddev=0.1)))

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

                if self.finetune_enc_act_func[l] is not None:
                    layer_y = self.finetune_enc_act_func[l](y_act)
                else:
                    layer_y = None

                # the input to the next layer is the output of this layer
                next_train = tf.nn.dropout(layer_y, self.keep_prob)

            self.layer_nodes.append(next_train)

        self.encode = next_train

    def _create_decoding_layers(self):
        """Create the decoding layers for reconstruction finetuning.

        :return: output of the final encoding layer.
        """
        next_decode = self.encode

        for l, layer in reversed(list(enumerate(self.layers))):

            with tf.name_scope("decode-{}".format(l)):

                # Create decoding variables
                if self.tied_weights:
                    dec_w = tf.transpose(self.encoding_w_[l])
                else:
                    dec_w = tf.Variable(tf.transpose(
                        self.encoding_w_[l].initialized_value()))

                dec_b = tf.Variable(tf.constant(
                    0.1, shape=[dec_w.get_shape().dims[1].value]))
                self.decoding_w.append(dec_w)
                self.decoding_b.append(dec_b)

                y_act = tf.add(
                    tf.matmul(next_decode, dec_w),
                    dec_b
                )

                if self.finetune_dec_act_func[l] is not None:
                    layer_y = self.finetune_dec_act_func[l](y_act)
                else:
                    layer_y = None

                # the input to the next layer is the output of this layer
                next_decode = tf.nn.dropout(layer_y, self.keep_prob)

            self.layer_nodes.append(next_decode)

        self.reconstruction = next_decode
