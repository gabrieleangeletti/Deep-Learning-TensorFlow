import tensorflow as tf
import numpy as np

from yadlt.core.unsupervised_model import UnsupervisedModel
from yadlt.utils import utilities


class DenoisingAutoencoder(UnsupervisedModel):

    """ Implementation of Denoising Autoencoders using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, n_components, model_name='dae', main_dir='dae/', models_dir='models/', data_dir='data/', summary_dir='logs/',
                 enc_act_func=tf.nn.tanh, dec_act_func=None, loss_func='mean_squared', num_epochs=10, batch_size=10, dataset='mnist',
                 opt='gradient_descent', learning_rate=0.01, momentum=0.5, corr_type='none', corr_frac=0., verbose=1, l2reg=5e-4):
        """
        :param n_components: number of hidden units
        :param enc_act_func: Activation function for the encoder. [tf.nn.tanh, tf.nn.sigmoid]
        :param dec_act_func: Activation function for the decoder. [[tf.nn.tanh, tf.nn.sigmoid, None]
        :param corr_type: Type of input corruption. ["none", "masking", "salt_and_pepper"]
        :param corr_frac: Fraction of the input to corrupt.
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
        :param l2reg: Regularization parameter. If 0, no regularization.
        """
        UnsupervisedModel.__init__(self, model_name, main_dir, models_dir, data_dir, summary_dir)

        self._initialize_training_parameters(loss_func=loss_func, learning_rate=learning_rate, num_epochs=num_epochs,
                                             batch_size=batch_size, dataset=dataset, opt=opt, momentum=momentum,
                                             l2reg=l2reg)

        self.n_components = n_components
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func
        self.corr_type = corr_type
        self.corr_frac = corr_frac
        self.verbose = verbose

        self.input_data_orig = None
        self.input_data = None

        self.W_ = None
        self.bh_ = None
        self.bv_ = None

    def _train_model(self, train_set, validation_set, train_ref=None, Validation_ref=None):

        """Train the model.
        :param train_set: training set
        :param validation_set: validation set. optional, default None

        :return: self
        """

        for i in range(self.num_epochs):

            self._run_train_step(train_set)

            if validation_set is not None:
                feed = {self.input_data_orig: validation_set, self.input_data: validation_set}
                self._run_validation_error_and_summaries(i, feed)

    def _run_train_step(self, train_set):

        """ Run a training step. A training step is made by randomly corrupting the training set,
        randomly shuffling it,  divide it into batches and run the optimizer for each batch.
        :param train_set: training set
        :return: self
        """
        x_corrupted = self._corrupt_input(train_set)

        shuff = zip(train_set, x_corrupted)
        np.random.shuffle(shuff)

        batches = [_ for _ in utilities.gen_batches(shuff, self.batch_size)]

        for batch in batches:
            x_batch, x_corr_batch = zip(*batch)
            tr_feed = {self.input_data_orig: x_batch, self.input_data: x_corr_batch}
            self.tf_session.run(self.train_step, feed_dict=tr_feed)

    def _corrupt_input(self, data):

        """ Corrupt a fraction of 'data' according to the
        noise method of this autoencoder.
        :return: corrupted data
        """

        corruption_ratio = np.round(self.corr_frac * data.shape[1]).astype(np.int)

        if self.corr_type == 'none':
            return np.copy(data)

        if self.corr_frac > 0.0:
            if self.corr_type == 'masking':
                return utilities.masking_noise(data, self.tf_session, self.corr_frac)

            elif self.corr_type == 'salt_and_pepper':
                return utilities.salt_and_pepper_noise(data, corruption_ratio)
        else:
            return np.copy(data)

    def build_model(self, n_features, W_=None, bh_=None, bv_=None):

        """ Creates the computational graph.
        :param n_features: Number of features.
        :param W_: weight matrix np array
        :param bh_: hidden bias np array
        :param bv_: visible bias np array
        :return: self
        """

        self._create_placeholders(n_features)
        self._create_variables(n_features, W_, bh_, bv_)

        self._create_encode_layer()
        self._create_decode_layer()

        regularizers = tf.nn.l2_loss(self.W_) + tf.nn.l2_loss(self.bh_) + tf.nn.l2_loss(self.bv_)
        regterm = self.l2reg * regularizers

        self._create_cost_function_node(self.reconstruction, self.input_data_orig, regterm)
        self._create_train_step_node()

    def _create_placeholders(self, n_features):

        """ Create the TensorFlow placeholders for the model.
        :return: self
        """

        self.input_data_orig = tf.placeholder('float', [None, n_features], name='x-input')
        self.input_data = tf.placeholder('float', [None, n_features], name='x-corr-input')
        # not used in this model, created just to comply with unsupervised_model.py
        self.input_labels = tf.placeholder('float')
        self.keep_prob = tf.placeholder('float', name='keep-probs')

    def _create_variables(self, n_features, W_=None, bh_=None, bv_=None):

        """ Create the TensorFlow variables for the model.
        :return: self
        """

        if W_:
            self.W_ = tf.Variable(W_, name='enc-w')
        else:
            self.W_ = tf.Variable(tf.truncated_normal(shape=[n_features, self.n_components], stddev=0.1), name='enc-w')

        if bh_:
            self.bh_ = tf.Variable(bh_, name='hidden-bias')
        else:
            self.bh_ = tf.Variable(tf.constant(0.1, shape=[self.n_components]), name='hidden-bias')

        if bv_:
            self.bv_ = tf.Variable(bv_, name='visible-bias')
        else:
            self.bv_ = tf.Variable(tf.constant(0.1, shape=[n_features]), name='visible-bias')

    def _create_encode_layer(self):

        """ Create the encoding layer of the network.
        :return: self
        """

        with tf.name_scope("encoder"):

            activation = tf.matmul(self.input_data, self.W_) + self.bh_

            if self.enc_act_func:
                self.encode = self.enc_act_func(activation)
            else:
                self.encode = None

    def _create_decode_layer(self):

        """ Create the decoding layer of the network.
        :return: self
        """

        with tf.name_scope("decoder"):

            activation = tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_

            if self.dec_act_func:
                self.reconstruction = self.dec_act_func(activation)
            elif self.dec_act_func is None:
                self.reconstruction = activation
            else:
                self.reconstruction = None

    def get_model_parameters(self, graph=None):

        """ Return the model parameters in the form of numpy arrays.
        :param graph: tf graph object
        :return: model parameters
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)

                return {
                    'enc_w': self.W_.eval(),
                    'enc_b': self.bh_.eval(),
                    'dec_b': self.bv_.eval()
                }
