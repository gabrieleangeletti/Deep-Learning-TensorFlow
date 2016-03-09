import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import config
from tf_models.rbm_models import rbm
from utils import utilities


class DeepAutoencoder(object):

    """ Implementation of a Deep Autoencoder as a Stack of
    Restricted Boltzmann Machines using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, layers, do_pretrain=True, rbm_num_epochs=list([10]), rbm_batch_size=list([10]),
                 rbm_learning_rate=list([0.01]), rbm_names=list(['']), rbm_gibbs_k=list([1]), gauss_visible=False,
                 stddev=0.1, learning_rate=0.01, momentum=0.7, num_epochs=10, batch_size=10, dropout=1,
                 opt='gradient_descent', loss_func='mean_squared', model_name='deepae', main_dir='deepae/',
                 dataset='mnist', verbose=1):

        """
        :param layers: list containing the hidden units for each layer
        :param do_pretrain: whether to do unsupervised pretraining of the network
        :param rbm_num_epochs: number of epochs to train each rbm
        :param rbm_batch_size: batch size each rbm
        :param rbm_learning_rate: learning rate each rbm
        :param rbm_names: model name for each rbm
        :param rbm_gibbs_k: number of gibbs sampling steps for each rbm
        :param gauss_visible: whether the input layer should have gaussian units
        :param stddev: standard deviation for the gaussian layer
        :param learning_rate: Initial learning rate. float, default 0.01
        :param momentum: 'Momentum parameter. float, default 0.9
        :param num_epochs: Number of epochs. int, default 10
        :param batch_size: Size of each mini-batch. int, default 10
        :param dropout: dropout parameter
        :param opt: Optimizer to use. string, default 'gradient_descent'. ['gradient_descent', 'ada_grad', 'momentum']
        :param loss_func: Loss function. ['cross_entropy', 'mean_squared']. string, default 'mean_squared'
        :param model_name: name of the model, used as filename. string, default 'sdae'
        :param main_dir: main directory to put the stored_models, data and summary directories
        :param dataset: Optional name for the dataset. string, default 'mnist'
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy. int, default 0
        """

        self.layers = layers
        self.n_layers = len(layers)

        # RBM parameters
        self.do_pretrain = do_pretrain
        self.rbm_num_epochs = rbm_num_epochs
        self.rbm_batch_size = rbm_batch_size
        self.rbm_learning_rate = rbm_learning_rate
        self.rbm_names = rbm_names
        self.rbm_gibbs_k = rbm_gibbs_k

        # Stacked RBM parameters
        self.gauss_visible = gauss_visible
        self.stddev = stddev
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.opt = opt
        self.loss_func = loss_func
        self.model_name = model_name
        self.main_dir = main_dir
        self.dataset = dataset
        self.verbose = verbose

        self.models_dir, self.data_dir, self.tf_summary_dir = self._create_data_directories()
        self.model_path = self.models_dir + self.model_name

        self.W = None
        self.bh = None
        self.bv = None

        self.W_vars = None
        self.W_vars_t = None
        self.bh_vars = None
        self.bv_vars = None

        # Model traning and evaluation
        self.cost = None
        self.train_step = None

        # tensorflow objects
        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.tf_session = None
        self.tf_saver = None

        if self.do_pretrain:

            self.rbms = []

            for l in range(self.n_layers - 1):

                if l == 0 and self.gauss_visible:

                    # Gaussian visible units

                    self.rbms.append(rbm.RBM(
                        visible_unit_type='gauss', stddev=self.stddev,
                        model_name=self.rbm_names[l] + str(l), num_visible=self.layers[l], num_hidden=self.layers[l+1],
                        main_dir=self.main_dir, learning_rate=self.rbm_learning_rate[l],
                        verbose=self.verbose, num_epochs=self.rbm_num_epochs[l], batch_size=self.rbm_batch_size[l]))

                else:

                    # Binary RBMs

                    self.rbms.append(rbm.RBM(
                        model_name=self.rbm_names[l] + str(l), num_visible=self.layers[l], num_hidden=self.layers[l+1],
                        main_dir=self.main_dir, learning_rate=self.rbm_learning_rate[l],
                        verbose=self.verbose, num_epochs=self.rbm_num_epochs[l], batch_size=self.rbm_batch_size[l]))

    def pretrain(self, train_set, validation_set=None):

        """ Unsupervised pretraining procedure for the deep autoencoder.
        :param train_set: training set
        :param validation_set: validation set
        :return: self
        """

        self.W = []
        self.bh = []
        self.bv = []

        next_layer_feed = train_set
        next_layer_validation = validation_set

        for l, rboltz in enumerate(self.rbms):
            print('Training layer {}...'.format(l+1))

            # Training this layer
            rboltz.fit(next_layer_feed, next_layer_validation)

            # Model parameters
            params = rboltz.get_model_parameters()
            self.W.append(params['W'])
            self.bh.append(params['bh_'])
            self.bv.append(params['bv_'])

            # Encoding the features for the higher level
            next_layer_feed = rboltz.transform(next_layer_feed, models_dir=self.models_dir)
            next_layer_validation = rboltz.transform(next_layer_validation, models_dir=self.models_dir)

            # Reset tensorflow's default graph between different autoencoders
            ops.reset_default_graph()

    def _build_model(self, n_features):

        """ Creates the computational graph.
        This graph is intented to be created for finetuning,
        i.e. after unsupervised pretraining.
        :param n_features: number of features of the first layer.
        :return: self
        """

        self._create_placeholders(n_features)
        self._create_variables()

        encode_output = self._create_encoding_layers()
        self.encode = encode_output
        decode_output = self._create_decoding_layers(encode_output)

        self._create_cost_function_node(decode_output)
        self._create_train_step_node()

    def _create_placeholders(self, n_features):

        """ Create placeholders for the model
        :param n_features: number of features of the first layer
        :return: keep_probs, hrand, vrand
        """

        self.x = tf.placeholder('float', [None, n_features])
        self.keep_prob = tf.placeholder('float')
        self.hrand = [tf.placeholder('float', [None, self.layers[l+1]]) for l in range(self.n_layers-1)]
        self.vrand = [tf.placeholder('float', [None, self.layers[l]]) for l in range(self.n_layers-1)]

    def _create_variables(self):

        """ Create variables for the model
        :return: W_vars, W_vars_t, bh_vars, bv_vars
        """

        self.W_vars = [tf.Variable(self.W[l]) for l in range(self.n_layers-1)]
        self.W_vars_t = [tf.Variable(self.W[l].T) for l in range(self.n_layers-1)]
        self.bh_vars = [tf.Variable(self.bh[l]) for l in range(self.n_layers-1)]
        self.bv_vars = [tf.Variable(self.bv[l]) for l in range(self.n_layers-1)]

    def _create_encoding_layers(self):

        """ Create the encoding layers of the model.
        :return: output of the last encoding layer
        """

        next_layer_feed = self.x

        for l in range(self.n_layers-1):
            hprobs = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(next_layer_feed, self.W_vars[l]) + self.bh_vars[l]),
                                   self.keep_prob)

            hstates = utilities.sample_prob(hprobs, self.hrand[l])

            next_layer_feed = hstates

        return next_layer_feed

    def _create_decoding_layers(self, decode_input):

        """ Create the decoding layres of the model.
        :param decode_input: output of the last encoding layer
        :return: output of the last decoding layer
        """
        decode_output = decode_input

        for l in reversed(range(self.n_layers-1)):
            vprobs = tf.nn.sigmoid(tf.matmul(decode_output, self.W_vars_t[l]) + self.bv_vars[l])
            vstates = utilities.sample_prob(vprobs, self.vrand[l])

            decode_output = vstates

        return decode_output

    def _create_cost_function_node(self, decode_output):

        """ Create the cost function node.
        :param decode_output: output of the last decoding layer
        :return: self
        """

        with tf.name_scope("cost"):
            if self.loss_func == 'cross_entropy':
                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(decode_output, self.x))
                _ = tf.scalar_summary("cross_entropy", self.cost)

            elif self.loss_func == 'mean_squared':
                self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.x - decode_output)))
                _ = tf.scalar_summary("mean_squared", self.cost)

            else:
                self.cost = None

    def _create_train_step_node(self):

        """ Create the training step node of the network.
        :return: self
        """

        with tf.name_scope("train"):
            if self.opt == 'gradient_descent':
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt == 'ada_grad':
                self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt == 'momentum':
                self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cost)

            else:
                self.train_step = None

    def finetune(self, train_set, validation_set=None):

        """ Perform finetuning procedure of the deep autoencoder.
        :param train_set: trainin set
        :param validation_set: validation set
        :return: self
        """

        n_features = train_set.shape[1]

        self._build_model(n_features)

        with tf.Session() as self.tf_session:

            self._initialize_tf_utilities_and_ops()
            self._train_model(train_set, validation_set)
            self.tf_saver.save(self.tf_session, self.model_path)

    def _initialize_tf_utilities_and_ops(self):

        """ Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
        """

        self.tf_merged_summaries = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()
        self.tf_saver = tf.train.Saver()

        self.tf_session.run(init_op)

        self.tf_summary_writer = tf.train.SummaryWriter(self.tf_summary_dir, self.tf_session.graph_def)

    def _train_model(self, train_set, validation_set):

        """ Train the model.
        :param train_set: training set
        :param validation_set: validation set
        :return: self
        """

        batches = [_ for _ in utilities.gen_batches(train_set, self.batch_size)]

        for i in range(self.num_epochs):

            for batch in batches:
                feed = self._create_feed(batch, self.dropout)
                self.tf_session.run(self.train_step, feed_dict=feed)

            if i % 5 == 0:
                if validation_set is not None:
                    self._run_validation_error_and_summaries(i, validation_set)

    def _run_validation_error_and_summaries(self, epoch, validation_set):

        """ Run the summaries and error computation on the validation set.
        :param epoch: current epoch
        :param validation_set: validation data

        :return: self
        """

        feed = self._create_feed(validation_set, self.dropout)
        result = self.tf_session.run([self.tf_merged_summaries, self.cost], feed_dict=feed)
        summary_str = result[0]
        err = result[1]

        self.tf_summary_writer.add_summary(summary_str, epoch)

        if self.verbose == 1:
            print("Cost at step %s: %s" % (epoch, err))

    def get_model_parameters(self):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """

        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.model_path)

            def eval_tensors(tens, prefix):
                eval_tens = {}
                for i, t in enumerate(tens):
                    eval_tens[prefix + str(i)] = t.eval()

            return {
                'W': eval_tensors(self.W_vars, 'w'),
                'Wt': eval_tensors(self.W_vars_t, 'wt'),
                'bh': eval_tensors(self.bh_vars, 'bh'),
                'bv': eval_tensors(self.bv_vars, 'bv')
            }

    def transform(self, data, name='train', save=False):

        """ Return data encoded by the deep autoencoder.
        :param data: data to encode
        :param name: name to save the data
        :param save: if true, also save the data as numpy array
        :return: encoded data
        """

        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.model_path)
            encoded_data = self.encode.eval(self._create_feed(data, 1.))

            if save:
                np.save(self.data_dir + self.model_name + '-' + name, encoded_data)

            return encoded_data

    def _create_feed(self, data, keep_prob):

        """ Create dictionary to feed TensorFlow placeholders.
        :param data: input data
        :param keep_prob: dropout probabilities
        :return: feed dictionary
        """

        feed = {self.x: data, self.keep_prob: keep_prob}

        # Random uniform for encoding layers
        hrand = [np.random.rand(data.shape[0], l) for l in self.layers[1:]]
        for j, h in enumerate(hrand):
            feed[self.hrand[j]] = h

        # Random uniform for decoding layers
        vrand = [np.random.rand(data.shape[0], l) for l in self.layers[:-1]]
        for j, v in enumerate(vrand):
            feed[self.vrand[j]] = v

        return feed

    def _create_data_directories(self):

        """ Create the three directories for storing respectively the stored_models,
        the data generated by training and the TensorFlow's summaries.

        :return: tuple of strings(models_dir, data_dir, summary_dir)
        """

        self.main_dir = self.main_dir + '/' if self.main_dir[-1] != '/' else self.main_dir

        models_dir = config.models_dir + self.main_dir
        data_dir = config.data_dir + self.main_dir
        summary_dir = config.summary_dir + self.main_dir

        for d in [models_dir, data_dir, summary_dir]:
            if not os.path.isdir(d):
                os.mkdir(d)

        return models_dir, data_dir, summary_dir
