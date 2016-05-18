import tensorflow as tf
import numpy as np
import os

from yadlt.core import model
from yadlt.utils import utilities


class DenoisingAutoencoder(model.Model):

    """ Implementation of Denoising Autoencoders using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, model_name='dae', n_components=256, main_dir='dae/', models_dir='models/', data_dir='data/', summary_dir='logs/',
                 enc_act_func='tanh', dec_act_func='none', loss_func='mean_squared', num_epochs=10, batch_size=10, dataset='mnist',
                 opt='gradient_descent', learning_rate=0.01, momentum=0.5, corr_type='none', corr_frac=0., verbose=1, l2reg=5e-4):
        """
        :param n_components: number of hidden units
        :param enc_act_func: Activation function for the encoder. ['tanh', 'sigmoid']
        :param dec_act_func: Activation function for the decoder. ['tanh', 'sigmoid', 'none']
        :param corr_type: Type of input corruption. ["none", "masking", "salt_and_pepper"]
        :param corr_frac: Fraction of the input to corrupt.
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
        :param l2reg: Regularization parameter. If 0, no regularization.
        """
        model.Model.__init__(self, model_name, main_dir, models_dir, data_dir, summary_dir)

        self._initialize_training_parameters(loss_func=loss_func, learning_rate=learning_rate, num_epochs=num_epochs,
                                             batch_size=batch_size, dataset=dataset, opt=opt, momentum=momentum,
                                             l2reg=l2reg)

        self.n_components = n_components
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func
        self.corr_type = corr_type
        self.corr_frac = corr_frac
        self.verbose = verbose

        self.input_data = None
        self.input_data_corr = None

        self.W_ = None
        self.bh_ = None
        self.bv_ = None

        self.encode = None
        self.decode = None

    def fit(self, train_set, validation_set=None, restore_previous_model=False, graph=None):

        """ Fit the model to the data.
        :param train_set: Training data.
        :param validation_set: optional, default None. Validation data.
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :return: self
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            self.build_model(train_set.shape[1])
            with tf.Session() as self.tf_session:
                self._initialize_tf_utilities_and_ops(restore_previous_model)
                self._train_model(train_set, validation_set)
                self.tf_saver.save(self.tf_session, self.model_path)

    def _train_model(self, train_set, validation_set):

        """Train the model.
        :param train_set: training set
        :param validation_set: validation set. optional, default None

        :return: self
        """

        for i in range(self.num_epochs):

            self._run_train_step(train_set)

            if validation_set is not None:
                feed = {self.input_data: validation_set, self.input_data_corr: validation_set}
                self._run_unsupervised_validation_error_and_summaries(i, feed)

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
            tr_feed = {self.input_data: x_batch, self.input_data_corr: x_corr_batch}
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

    def reconstruct(self, data):

        """ Reconstruct the test set data using the learned model.
        :param data: Testing data. shape(n_test_samples, n_features)
        :return: labels
        """

        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                return self.decode.eval({self.input_data_corr: data})

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

        self._create_cost_function_node(self.decode, self.input_data, regterm)
        self._create_train_step_node()

    def _create_placeholders(self, n_features):

        """ Create the TensorFlow placeholders for the model.
        :return: self
        """

        self.input_data = tf.placeholder('float', [None, n_features], name='x-input')
        self.input_data_corr = tf.placeholder('float', [None, n_features], name='x-corr-input')

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

            activation = tf.matmul(self.input_data_corr, self.W_) + self.bh_

            if self.enc_act_func == 'sigmoid':
                self.encode = tf.nn.sigmoid(activation)

            elif self.enc_act_func == 'tanh':
                self.encode = tf.nn.tanh(activation)

            else:
                self.encode = None

    def _create_decode_layer(self):

        """ Create the decoding layer of the network.
        :return: self
        """

        with tf.name_scope("decoder"):

            activation = tf.matmul(self.encode, tf.transpose(self.W_)) + self.bv_

            if self.dec_act_func == 'sigmoid':
                self.decode = tf.nn.sigmoid(activation)

            elif self.dec_act_func == 'tanh':
                self.decode = tf.nn.tanh(activation)

            elif self.dec_act_func == 'none':
                self.decode = activation

            else:
                self.decode = None

    def transform(self, data, name='train', save=False, graph=None):

        """ Transform data according to the model.
        :param data: Data to transform
        :param name: Identifier for the data that is being encoded
        :param save: If true, save data to disk
        :param graph: tf graph object
        :return: transformed data
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                encoded_data = self.encode.eval({self.input_data_corr: data})

                if save:
                    np.save(self.data_dir + self.model_name + '-' + name, encoded_data)

                return encoded_data

    def load_model(self, shape, model_path):

        """ Restore a previously trained model from disk.
        :param shape: tuple(n_features, n_components)
        :param model_path: path to the trained model
        :return: self, the trained model
        """

        self.n_components = shape[1]
        self.build_model(shape[0])
        init_op = tf.initialize_all_variables()
        self.tf_saver = tf.train.Saver()

        with tf.Session() as self.tf_session:

            self.tf_session.run(init_op)
            self.tf_saver.restore(self.tf_session, model_path)

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

    def get_weights_as_images(self, width, height, outdir='img/', max_images=10, model_path=None):

        """ Save the weights of this autoencoder as images, one image per hidden unit.
        Useful to visualize what the autoencoder has learned.

        :param width: Width of the images. int
        :param height: Height of the images. int
        :param outdir: Output directory for the images. This path is appended to self.data_dir. string, default 'img/'
        :param max_images: Number of images to return. int, default 10
        :param model_path: if True, restore previous model with the same name of this autoencoder
        """

        assert max_images <= self.n_components

        outdir = self.data_dir + outdir

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        with tf.Session() as self.tf_session:

            if model_path is not None:
                self.tf_saver.restore(self.tf_session, model_path)
            else:
                self.tf_saver.restore(self.tf_session, self.model_path)

            enc_weights = self.W_.eval()

            perm = np.random.permutation(self.n_components)[:max_images]

            for p in perm:

                enc_w = np.array([i[p] for i in enc_weights])
                image_path = outdir + self.model_name + '-enc_weights_{}.png'.format(p)
                utilities.gen_image(enc_w, width, height, image_path)
