import tensorflow as tf
import numpy as np
import os

from utils import utilities
import config


class RBM(object):

    """ Restricted Boltzmann Machine implementation using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, num_visible, num_hidden, visible_unit_type='bin', main_dir='rbm', model_name='rbm_model',
                 gibbs_sampling_steps=1, learning_rate=0.01, batch_size=10, num_epochs=10, stddev=0.1, verbose=0):

        """
        :param num_visible: number of visible units
        :param num_hidden: number of hidden units
        :param visible_unit_type: type of the visible units (binary or gaussian)
        :param main_dir: main directory to put the stored_models, data and summary directories
        :param model_name: name of the model, used to save data
        :param gibbs_sampling_steps: optional, default 1
        :param learning_rate: optional, default 0.01
        :param batch_size: optional, default 10
        :param num_epochs: optional, default 10
        :param stddev: optional, default 0.1. Ignored if visible_unit_type is not 'gauss'
        :param verbose: level of verbosity. optional, default 0
        """

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.visible_unit_type = visible_unit_type
        self.main_dir = main_dir
        self.model_name = model_name
        self.gibbs_sampling_steps = gibbs_sampling_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.stddev = stddev
        self.verbose = verbose

        self.models_dir, self.data_dir, self.summary_dir = self._create_data_directories()
        self.model_path = self.models_dir + self.model_name

        self.W = None
        self.bh_ = None
        self.bv_ = None

        self.w_upd8 = None
        self.bh_upd8 = None
        self.bv_upd8 = None

        self.encode = None

        self.loss_function = None

        self.input_data = None
        self.hrand = None
        self.vrand = None
        self.validation_size = None

        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.tf_session = None
        self.tf_saver = None

    def fit(self, train_set, validation_set=None, restore_previous_model=False):

        """ Fit the model to the training data.

        :param train_set: training set
        :param validation_set: validation set. optional, default None
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.

        :return: self
        """

        if validation_set is not None:
            self.validation_size = validation_set.shape[0]

        self._build_model()

        with tf.Session() as self.tf_session:

            self._initialize_tf_utilities_and_ops(restore_previous_model)
            self._train_model(train_set, validation_set)
            self.tf_saver.save(self.tf_session, self.model_path)

    def _initialize_tf_utilities_and_ops(self, restore_previous_model):

        """ Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
        Restore a previously trained model if the flag restore_previous_model is true.
        """

        self.tf_merged_summaries = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()
        self.tf_saver = tf.train.Saver()

        self.tf_session.run(init_op)

        if restore_previous_model:
            self.tf_saver.restore(self.tf_session, self.model_path)

        self.tf_summary_writer = tf.train.SummaryWriter(self.summary_dir, self.tf_session.graph_def)

    def _train_model(self, train_set, validation_set):

        """ Train the model.
        :param train_set: training set
        :param validation_set: validation set. optional, default None
        :return: self
        """

        for i in range(self.num_epochs):
            self._run_train_step(train_set)

            if validation_set is not None:
                self._run_validation_error_and_summaries(i, validation_set)

    def _run_train_step(self, train_set):

        """ Run a training step. A training step is made by randomly shuffling the training set,
        divide into batches and run the variable update nodes for each batch.
        :param train_set: training set
        :return: self
        """

        np.random.shuffle(train_set)

        batches = [_ for _ in utilities.gen_batches(train_set, self.batch_size)]
        updates = [self.w_upd8, self.bh_upd8, self.bv_upd8]

        for batch in batches:
            self.tf_session.run(updates, feed_dict=self._create_feed_dict(batch))

    def _run_validation_error_and_summaries(self, epoch, validation_set):

        """ Run the summaries and error computation on the validation set.
        :param epoch: current epoch
        :param validation_set: validation data
        :return: self
        """

        result = self.tf_session.run([self.tf_merged_summaries, self.loss_function],
                                     feed_dict=self._create_feed_dict(validation_set))

        summary_str = result[0]
        err = result[1]

        self.tf_summary_writer.add_summary(summary_str, 1)

        if self.verbose == 1:
            print("Validation cost at step %s: %s" % (epoch, err))

    def _create_feed_dict(self, data):

        """ Create the dictionary of data to feed to TensorFlow's session during training.
        :param data: training/validation set batch
        :return: dictionary(self.input_data: data, self.hrand: random_uniform, self.vrand: random_uniform)
        """

        return {
            self.input_data: data,
            self.hrand: np.random.rand(data.shape[0], self.num_hidden),
            self.vrand: np.random.rand(data.shape[0], self.num_visible)
        }

    def _build_model(self):

        """ Build the Restricted Boltzmann Machine model in TensorFlow.
        :return: self
        """

        self._create_placeholders()
        self._create_variables()

        hprobs0, hstates0, vprobs, hprobs1, hstates1 = self.gibbs_sampling_step(self.input_data)
        positive = self.compute_positive_association(self.input_data, hprobs0, hstates0)

        nn_input = vprobs

        for step in range(self.gibbs_sampling_steps - 1):
            hprobs, hstates, vprobs, hprobs1, hstates1 = self.gibbs_sampling_step(nn_input)
            nn_input = vprobs

        negative = tf.matmul(tf.transpose(vprobs), hprobs1)

        self.encode = hprobs1  # encoded data, used by the transform method

        self.w_upd8 = self.W.assign_add(self.learning_rate * (positive - negative))
        self.bh_upd8 = self.bh_.assign_add(self.learning_rate * tf.reduce_mean(hprobs0 - hprobs1, 0))
        self.bv_upd8 = self.bv_.assign_add(self.learning_rate * tf.reduce_mean(self.input_data - vprobs, 0))

        self.loss_function = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - vprobs)))
        _ = tf.scalar_summary("cost", self.loss_function)

    def _create_placeholders(self):

        """ Create the TensorFlow placeholders for the model.
        :return: self
        """

        self.input_data = tf.placeholder('float', [None, self.num_visible], name='x-input')
        self.hrand = tf.placeholder('float', [None, self.num_hidden], name='hrand')
        self.vrand = tf.placeholder('float', [None, self.num_visible], name='vrand')

    def _create_variables(self):

        """ Create the TensorFlow variables for the model.
        :return: self
        """

        self.W = tf.Variable(tf.random_normal((self.num_visible, self.num_hidden), mean=0.0, stddev=0.01), name='weights')
        self.bh_ = tf.Variable(tf.zeros([self.num_hidden]), name='hidden-bias')
        self.bv_ = tf.Variable(tf.zeros([self.num_visible]), name='visible-bias')

    def gibbs_sampling_step(self, visible):

        """ Performs one step of gibbs sampling.
        :param visible: activations of the visible units
        :return: tuple(hidden probs, hidden states, visible probs,
                       new hidden probs, new hidden states)
        """

        hprobs, hstates = self.sample_hidden_from_visible(visible)
        vprobs = self.sample_visible_from_hidden(hprobs)
        hprobs1, hstates1 = self.sample_hidden_from_visible(vprobs)

        return hprobs, hstates, vprobs, hprobs1, hstates1

    def sample_hidden_from_visible(self, visible):

        """ Sample the hidden units from the visible units.
        This is the Positive phase of the Contrastive Divergence algorithm.

        :param visible: activations of the visible units
        :return: tuple(hidden probabilities, hidden binary states)
        """

        hprobs = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.bh_)
        hstates = utilities.sample_prob(hprobs, self.hrand)

        return hprobs, hstates

    def sample_visible_from_hidden(self, hidden):

        """ Sample the visible units from the hidden units.
        This is the Negative phase of the Contrastive Divergence algorithm.
        :param hidden: activations of the hidden units
        :return: visible probabilities
        """

        visible_activation = tf.matmul(hidden, tf.transpose(self.W)) + self.bv_

        if self.visible_unit_type == 'bin':
            vprobs = tf.nn.sigmoid(visible_activation)

        elif self.visible_unit_type == 'gauss':
            vprobs = tf.truncated_normal((1, self.num_visible), mean=visible_activation, stddev=self.stddev)

        else:
            vprobs = None

        return vprobs

    def compute_positive_association(self, visible, hidden_probs, hidden_states):

        """ Compute positive associations between visible and hidden units.
        :param visible: visible units
        :param hidden_probs: hidden units probabilities
        :param hidden_states: hidden units states
        :return: positive association = dot(visible.T, hidden)
        """

        if self.visible_unit_type == 'bin':
            positive = tf.matmul(tf.transpose(visible), hidden_states)

        elif self.visible_unit_type == 'gauss':
            positive = tf.matmul(tf.transpose(visible), hidden_probs)

        else:
            positive = None

        return positive

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

    def transform(self, data, name='train', save=False):

        """ Transform data according to the model.
        :param data: Data to transform
        :param name: Identifier for the data that is being encoded. string, default 'train'
        :param save: If true, save data to disk. boolean, default 'False'
        :return: transformed data
        """

        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.model_path)

            encoded_data = self.encode.eval(self._create_feed_dict(data))

            if save:
                np.save(self.data_dir + self.model_name + '-' + name, encoded_data)

            return encoded_data

    def load_model(self, shape, gibbs_sampling_steps, model_path):

        """ Load a trained model from disk. The shape of the model
        (num_visible, num_hidden) and the number of gibbs sampling steps
        must be known in order to restore the model.
        :param shape: tuple(num_visible, num_hidden)
        :param gibbs_sampling_steps:
        :param model_path:
        :return: self
        """

        self.num_visible, self.num_hidden = shape[0], shape[1]
        self.gibbs_sampling_steps = gibbs_sampling_steps

        self._build_model()

        init_op = tf.initialize_all_variables()
        self.tf_saver = tf.train.Saver()

        with tf.Session() as self.tf_session:

            self.tf_session.run(init_op)
            self.tf_saver.restore(self.tf_session, model_path)

    def get_model_parameters(self):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """

        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.model_path)

            return {
                'W': self.W.eval(),
                'bh_': self.bh_.eval(),
                'bv_': self.bv_.eval()
            }

    def get_weights_as_images(self, width, height, outdir='img/', n_images=10, img_type='grey'):

        """ Create and save the weights of the hidden units with respect to the
        visible units as images.
        :param width:
        :param height:
        :param outdir:
        :param n_images:
        :param img_type:
        :return: self
        """

        outdir = self.data_dir + outdir

        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.model_path)

            weights = self.W.eval()

            perm = np.random.permutation(self.num_hidden)[:n_images]

            for p in perm:
                w = np.array([i[p] for i in weights])
                image_path = outdir + self.model_name + '_{}.png'.format(p)
                utilities.gen_image(w, width, height, image_path, img_type)
