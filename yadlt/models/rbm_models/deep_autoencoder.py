import tensorflow as tf
import numpy as np
import os

from yadlt.core.unsupervised_model import UnsupervisedModel
from yadlt.models.rbm_models import rbm
from yadlt.utils import utilities


class DeepAutoencoder(UnsupervisedModel):

    """ Implementation of a Deep Autoencoder as a stack of RBMs for Unsupervised Learning using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, rbm_layers, model_name='sdae', main_dir='sdae/', models_dir='models/', data_dir='data/', summary_dir='logs/',
                 rbm_num_epochs=[10], rbm_batch_size=[10], dataset='mnist', rbm_learning_rate=[0.01], rbm_gibbs_k=[1],
                 momentum=0.5, finetune_dropout=1, verbose=1, finetune_loss_func='cross_entropy', finetune_enc_act_func=[tf.nn.relu],
                 finetune_dec_act_func=[tf.nn.sigmoid], finetune_opt='gradient_descent', finetune_learning_rate=0.001,
                 finetune_num_epochs=10, rbm_gauss_visible=False, rbm_stddev=0.1, finetune_batch_size=20, do_pretrain=True):
        """
        :param rbm_layers: list containing the hidden units for each layer
        :param finetune_loss_func: Loss function for the softmax layer. string, default ['cross_entropy', 'mean_squared']
        :param finetune_dropout: dropout parameter
        :param finetune_learning_rate: learning rate for the finetuning. float, default 0.001
        :param finetune_enc_act_func: activation function for the encoder finetuning phase
        :param finetune_dec_act_func: activation function for the decoder finetuning phase
        :param finetune_opt: optimizer for the finetuning phase
        :param finetune_num_epochs: Number of epochs for the finetuning. int, default 20
        :param finetune_batch_size: Size of each mini-batch for the finetuning. int, default 20
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy. int, default 0
        :param do_pretrain: True: uses variables from pretraining, False: initialize new variables.
        """
        UnsupervisedModel.__init__(self, model_name, main_dir, models_dir, data_dir, summary_dir)

        self._initialize_training_parameters(loss_func=finetune_loss_func, learning_rate=finetune_learning_rate,
                                             num_epochs=finetune_num_epochs, batch_size=finetune_batch_size,
                                             dropout=finetune_dropout, dataset=dataset, opt=finetune_opt, momentum=momentum)

        self.do_pretrain = do_pretrain
        self.layers = rbm_layers
        self.verbose = verbose

        if len(finetune_enc_act_func) != len(rbm_layers):
            self.finetune_enc_act_func = [finetune_enc_act_func[0] for _ in rbm_layers]
        else:
            self.finetune_enc_act_func = finetune_enc_act_func

        if len(finetune_dec_act_func) != len(rbm_layers):
            self.finetune_dec_act_func = [finetune_dec_act_func[0] for _ in rbm_layers]
        else:
            self.finetune_dec_act_func = finetune_dec_act_func

        self.input_ref = None

        # Model parameters
        self.encoding_w_ = []  # list of matrices of encoding weights (one per layer)
        self.encoding_b_ = []  # list of arrays of encoding biases (one per layer)

        self.decoding_w = []  # list of matrices of decoding weights (one per layer)
        self.decoding_b = []  # list of arrays of decoding biases (one per layer)

        self.reconstruction = None

        rbm_params = {'num_epochs': rbm_num_epochs, 'gibbs_k': rbm_gibbs_k, 'batch_size': rbm_batch_size,
                      'learning_rate': rbm_learning_rate}

        for p in rbm_params:
            if len(rbm_params[p]) != len(rbm_layers):
                # The current parameter is not specified by the user, should default it for all the layers
                rbm_params[p] = [rbm_params[p][0] for _ in rbm_layers]

        self.rbms = []
        self.rbm_graphs = []

        for l, layer in enumerate(rbm_layers):
            rbm_str = 'rbm-' + str(l + 1)

            if l == 0 and rbm_gauss_visible:

                # Gaussian visible units

                self.rbms.append(rbm.RBM(model_name=self.model_name + '-' + rbm_str,
                    models_dir=os.path.join(self.models_dir, rbm_str), data_dir=os.path.join(self.data_dir, rbm_str),  summary_dir=os.path.join(self.tf_summary_dir, rbm_str),
                    visible_unit_type='gauss', stddev=rbm_stddev, num_hidden=rbm_params['layers'][l],
                    main_dir=self.main_dir, learning_rate=rbm_params['learning_rate'][l], gibbs_sampling_steps=rbm_gibbs_k[l],
                    verbose=self.verbose, num_epochs=rbm_params['num_epochs'][l], batch_size=rbm_params['batch_size'][l]))

            else:

                # Binary RBMs

                self.rbms.append(rbm.RBM(model_name=self.model_name + '-' + rbm_str,
                    models_dir=os.path.join(self.models_dir, rbm_str), data_dir=os.path.join(self.data_dir, rbm_str),  summary_dir=os.path.join(self.tf_summary_dir, rbm_str),
                    num_hidden=rbm_layers[l],
                    main_dir=self.main_dir, learning_rate=rbm_params['learning_rate'][l], gibbs_sampling_steps=rbm_params['gibbs_k'][l],
                    verbose=self.verbose, num_epochs=rbm_params['num_epochs'][l], batch_size=rbm_params['batch_size'][l]))

            self.rbm_graphs.append(tf.Graph())

    def pretrain(self, train_set, validation_set=None):
        self.do_pretrain = True

        def set_params_func(rbmmachine, rbmgraph):
            params = rbmmachine.get_model_parameters(graph=rbmgraph)
            self.encoding_w_.append(params['W'])
            self.encoding_b_.append(params['bh_'])

        return UnsupervisedModel.pretrain_procedure(self, self.rbms, self.rbm_graphs, set_params_func=set_params_func,
                                                    train_set=train_set, validation_set=validation_set)

    def _train_model(self, train_set, train_ref, validation_set, validation_ref):

        """ Train the model.
        :param train_set: training set
        :param train_ref: training reference data
        :param validation_set: validation set
        :param validation_ref: validation reference data
        :return: self
        """

        shuff = zip(train_set, train_ref)

        for i in range(self.num_epochs):

            np.random.shuffle(shuff)
            batches = [_ for _ in utilities.gen_batches(shuff, self.batch_size)]

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.tf_session.run(self.train_step, feed_dict={self.input_data: x_batch,
                                                                self.input_labels: y_batch,
                                                                self.keep_prob: self.dropout})

            if validation_set is not None:
                feed = {self.input_data: validation_set, self.input_labels: validation_ref, self.keep_prob: 1}
                self._run_validation_error_and_summaries(i, feed)

    def build_model(self, n_features, encoding_w=None, encoding_b=None):

        """ Creates the computational graph for the reconstruction task.
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

        self._create_cost_function_node(self.reconstruction, self.input_labels)
        self._create_train_step_node()

    def _create_placeholders(self, n_features, n_classes):

        """ Create the TensorFlow placeholders for the model.
        :param n_features: number of features of the first layer
        :param n_classes: number of classes
        :return: self
        """

        self.input_data = tf.placeholder('float', [None, n_features], name='x-input')
        self.input_labels = tf.placeholder('float', [None, n_classes], name='y-input')
        self.keep_prob = tf.placeholder('float', name='keep-probs')

    def _create_variables(self, n_features):

        """ Create the TensorFlow variables for the model.
        :param n_features: number of features
        :return: self
        """

        if self.do_pretrain:
            self._create_variables_pretrain()
        else:
            self._create_variables_no_pretrain(n_features)

    def _create_variables_no_pretrain(self, n_features):

        """ Create model variables (no previous unsupervised pretraining)
        :param n_features: number of features
        :return: self
        """

        self.encoding_w_ = []
        self.encoding_b_ = []

        for l, layer in enumerate(self.layers):

            if l == 0:
                self.encoding_w_.append(tf.Variable(tf.truncated_normal(shape=[n_features, self.layers[l]], stddev=0.1)))
                self.encoding_b_.append(tf.Variable(tf.truncated_normal([self.layers[l]], stddev=0.1)))
            else:
                self.encoding_w_.append(tf.Variable(tf.truncated_normal(shape=[self.layers[l-1], self.layers[l]], stddev=0.1)))
                self.encoding_b_.append(tf.Variable(tf.truncated_normal([self.layers[l]], stddev=0.1)))

    def _create_variables_pretrain(self):

        """ Create model variables (previous unsupervised pretraining)
        :return: self
        """

        for l, layer in enumerate(self.layers):
            self.encoding_w_[l] = tf.Variable(self.encoding_w_[l], name='enc-w-{}'.format(l))
            self.encoding_b_[l] = tf.Variable(self.encoding_b_[l], name='enc-b-{}'.format(l))

    def _create_encoding_layers(self):

        """ Create the encoding layers for supervised finetuning.
        :return: output of the final encoding layer.
        """

        next_train = self.input_data
        self.layer_nodes = []

        for l, layer in enumerate(self.layers):

            with tf.name_scope("encode-{}".format(l)):

                y_act = tf.matmul(next_train, self.encoding_w_[l]) + self.encoding_b_[l]

                if self.finetune_enc_act_func[l] is not None:
                    layer_y = self.finetune_enc_act_func[l](y_act)

                else:
                    layer_y = None

                # the input to the next layer is the output of this layer
                next_train = tf.nn.dropout(layer_y, self.keep_prob)

            self.layer_nodes.append(next_train)

        self.encode = next_train

    def _create_decoding_layers(self):

        """ Create the decoding layers for reconstruction finetuning.
        :return: output of the final encoding layer.
        """

        next_decode = self.encode

        for l, layer in reversed(list(enumerate(self.layers))):

            with tf.name_scope("decode-{}".format(l)):

                # Create decoding variables
                dec_w = tf.Variable(tf.transpose(self.encoding_w_[l].initialized_value()))
                dec_b = tf.Variable(tf.constant(0.1, shape=[dec_w.get_shape().dims[1].value]))
                self.decoding_w.append(dec_w)
                self.decoding_b.append(dec_b)

                y_act = tf.matmul(next_decode, dec_w) + dec_b

                if self.finetune_dec_act_func[l] is not None:
                    layer_y = self.finetune_dec_act_func[l](y_act)

                else:
                    layer_y = None

                # the input to the next layer is the output of this layer
                next_decode = tf.nn.dropout(layer_y, self.keep_prob)

            self.layer_nodes.append(next_decode)

        self.reconstruction = next_decode
