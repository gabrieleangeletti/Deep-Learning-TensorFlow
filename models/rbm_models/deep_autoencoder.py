import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from models.rbm_models import rbm
from utils import utilities
import model


class DeepAutoencoder(model.Model):

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
        :param dropout: dropout parameter
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy. int, default 0
        """
        model.Model.__init__(self, model_name, main_dir)

        self._initialize_training_parameters(loss_func, learning_rate, num_epochs, batch_size,
                                             dataset, opt, momentum)

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
        self.dropout = dropout
        self.verbose = verbose

        self.W_pretrain = None
        self.bh_pretrain = None
        self.bv_pretrain = None

        self.W_vars = None
        self.W_vars_t = None
        self.bh_vars = None
        self.bv_vars = None

        self.encode = None

        if self.do_pretrain:

            self.rbms = []

            for l in range(self.n_layers):

                if l == 0 and self.gauss_visible:

                    # Gaussian visible units

                    self.rbms.append(rbm.RBM(
                        visible_unit_type='gauss', stddev=self.stddev,
                        model_name=self.rbm_names[l] + str(l), num_hidden=self.layers[l],
                        main_dir=self.main_dir, learning_rate=self.rbm_learning_rate[l],
                        verbose=self.verbose, num_epochs=self.rbm_num_epochs[l], batch_size=self.rbm_batch_size[l]))

                else:

                    # Binary RBMs

                    self.rbms.append(rbm.RBM(
                        model_name=self.rbm_names[l] + str(l), num_hidden=self.layers[l],
                        main_dir=self.main_dir, learning_rate=self.rbm_learning_rate[l],
                        verbose=self.verbose, num_epochs=self.rbm_num_epochs[l], batch_size=self.rbm_batch_size[l]))

    def pretrain(self, train_set, validation_set=None):

        """ Unsupervised pretraining procedure for the deep autoencoder.
        :param train_set: training set
        :param validation_set: validation set
        :return: self
        """

        self.W_pretrain = []
        self.bh_pretrain = []
        self.bv_pretrain = []

        next_train = train_set
        next_valid = validation_set

        for l, rboltz in enumerate(self.rbms):
            print('Training layer {}...'.format(l+1))
            next_train, next_valid = self._pretrain_rbm_and_gen_feed(rboltz, next_train, next_valid)

            # Reset tensorflow's default graph between different models
            ops.reset_default_graph()

        return next_train, next_valid

    def _pretrain_rbm_and_gen_feed(self, rboltz, train_set, validation_set):

        """ Pretrain a single rbm and encode the data for the next layer.
        :param rboltz: rbm reference
        :param train_set: training set
        :param validation_set: validation set
        :return: encoded train data, encoded validation data
        """

        rboltz.build_model(train_set.shape[1])
        rboltz.fit(train_set, validation_set)
        params = rboltz.get_model_parameters()

        self.W_pretrain.append(params['W'])
        self.bh_pretrain.append(params['bh_'])
        self.bv_pretrain.append(params['bv_'])

        return rboltz.transform(train_set), rboltz.transform(validation_set)

    def fit(self, train_set, validation_set=None, restore_previous_model=False):

        """ Perform finetuning procedure of the deep autoencoder.
        :param train_set: trainin set
        :param validation_set: validation set
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :return: self
        """

        with tf.Session() as self.tf_session:
            self._initialize_tf_utilities_and_ops(restore_previous_model)
            self._train_model(train_set, validation_set)
            self.tf_saver.save(self.tf_session, self.model_path)

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

    def build_model(self, n_features):

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

        self._create_cost_function_node(self.loss_func, decode_output, self.x)
        self._create_train_step_node(self.opt, self.learning_rate, self.momentum)

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

        self.W_vars = [tf.Variable(self.W_pretrain[l]) for l in range(self.n_layers-1)]
        self.W_vars_t = [tf.Variable(self.W_pretrain[l].T) for l in range(self.n_layers-1)]
        self.bh_vars = [tf.Variable(self.bh_pretrain[l]) for l in range(self.n_layers-1)]
        self.bv_vars = [tf.Variable(self.bv_pretrain[l]) for l in range(self.n_layers-1)]

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
