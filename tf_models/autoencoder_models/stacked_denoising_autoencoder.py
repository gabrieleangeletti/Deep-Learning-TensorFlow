import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import config
from tf_models.autoencoder_models import denoising_autoencoder
from utils import utilities


class StackedDenoisingAutoencoder(object):

    """ Implementation of Stacked Denoising Autoencoders using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, layers, model_name='sdae', main_dir='sdae/', enc_act_func=list(['tanh']),
                 dec_act_func=list(['none']), loss_func=list(['mean_squared']), num_epochs=list([10]),
                 batch_size=list([10]), dataset='mnist', xavier_init=list([1]), opt=list(['gradient_descent']),
                 learning_rate=list([0.01]), momentum=list([0.5]),  dropout=1, corr_type='none', corr_frac=0.,
                 verbose=1, seed=-1, softmax_loss_func='cross_entropy', finetune_act_func='relu',
                 finetune_opt='gradient_descent', finetune_learning_rate=0.001, finetune_num_epochs=10,
                 finetune_batch_size=20, do_pretrain=True):
        """
        :param model_name: name of the model, used as filename. string, default 'sdae'
        :param main_dir: main directory to put the stored_models, data and summary directories
        :param layers: list containing the hidden units for each layer
        :param enc_act_func: Activation function for the encoder. ['sigmoid', 'tanh']
        :param dec_act_func: Activation function for the decoder. ['sigmoid', 'tanh', 'none']
        :param softmax_loss_func: Loss function for the softmax layer. string, default ['cross_entropy', 'mean_squared']
        :param dropout: dropout parameter
        :param finetune_learning_rate: learning rate for the finetuning. float, default 0.001
        :param finetune_act_func: activation function for the finetuning phase
        :param finetune_opt: optimizer for the finetuning phase
        :param finetune_num_epochs: Number of epochs for the finetuning. int, default 20
        :param finetune_batch_size: Size of each mini-batch for the finetuning. int, default 20
        :param loss_func: Loss function. ['cross_entropy', 'mean_squared']. string, default 'mean_squared'
        :param xavier_init: Value of the constant for xavier weights initialization. int, default 1
        :param opt: Optimizer to use. string, default 'gradient_descent'. ['gradient_descent', 'ada_grad', 'momentum']
        :param learning_rate: Initial learning rate. float, default 0.01
        :param momentum: 'Momentum parameter. float, default 0.9
        :param corr_type: Type of input corruption. string, default 'none'. ["none", "masking", "salt_and_pepper"]
        :param corr_frac: Fraction of the input to corrupt. float, default 0.0
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy. int, default 0
        :param num_epochs: Number of epochs. int, default 10
        :param batch_size: Size of each mini-batch. int, default 10
        :param do_pretrain: True: uses variables from pretraining, False: initialize new variables.
        :param dataset: Optional name for the dataset. string, default 'mnist'
        :param seed: positive integer for seeding random generators. Ignored if < 0. int, default -1
        """

        self.layers = layers
        self.do_pretrain = do_pretrain

        # Autoencoder parameters
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func
        self.loss_func = loss_func
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.xavier_init = xavier_init
        self.opt = opt
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Stacked Autoencoder parameters
        self.model_name = model_name
        self.corr_type = corr_type
        self.corr_frac = corr_frac
        self.main_dir = main_dir
        self.dropout = dropout
        self.softmax_loss_func = softmax_loss_func
        self.finetune_opt = finetune_opt
        self.finetune_learning_rate = finetune_learning_rate
        self.finetune_act_func = finetune_act_func
        self.finetune_num_epochs = finetune_num_epochs
        self.finetune_batch_size = finetune_batch_size
        self.dataset = dataset
        self.verbose = verbose
        self.seed = seed

        if self.seed >= 0:
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)

        self.models_dir, self.data_dir, self.tf_summary_dir = self._create_data_directories()
        self.model_path = self.models_dir + self.model_name

        self.input_data = None
        self.input_labels = None
        self.keep_prob = None

        self.encode = None
        self.softmax_out = None

        # Model parameters
        self.encoding_w_ = None  # list of matrices of encoding weights (one per layer)
        self.encoding_b_ = None  # list of arrays of encoding biases (one per layer)

        self.softmax_W = None
        self.softmax_b = None

        # Model traning and evaluation
        self.train_step = None
        self.cost = None

        # tensorflow objects
        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.tf_session = None
        self.tf_saver = None

        self.autoencoders = []

        for l, layer in enumerate(layers):
            self.autoencoders.append(denoising_autoencoder.DenoisingAutoencoder(
                n_components=layer, main_dir=self.main_dir,
                enc_act_func=self.enc_act_func[l], dec_act_func=self.dec_act_func[l], loss_func=self.loss_func[l],
                xavier_init=self.xavier_init[l], opt=self.opt[l], learning_rate=self.learning_rate[l],
                momentum=self.momentum[l], corr_type=self.corr_type, corr_frac=self.corr_frac,
                verbose=self.verbose, num_epochs=self.num_epochs[l], batch_size=self.batch_size[l],
                dataset=self.dataset))

    def pretrain(self, train_set, validation_set=None):

        """ Perform unsupervised pretraining of the stack of denoising autoencoders.
        :param train_set: training set
        :param validation_set: validation set
        :return: return data encoded by the last layer
        """

        self.encoding_w_ = []
        self.encoding_b_ = []

        next_train = train_set
        next_valid = validation_set

        for l, autoenc in enumerate(self.autoencoders):
            print('Training layer {}...'.format(l+1))
            next_train, next_valid = self._pretrain_autoencoder_and_gen_feed(autoenc, next_train, next_valid)

            # Reset tensorflow's default graph between different autoencoders
            ops.reset_default_graph()

        return next_train, next_valid

    def _pretrain_autoencoder_and_gen_feed(self, autoenc, train_set, validation_set):

        """ Pretrain a single autoencoder and encode the data for the next layer.
        :param autoenc: autoencoder reference
        :param train_set: training set
        :param validation_set: validation set
        :return: encoded train data, encoded validation data
        """

        autoenc.fit(train_set, validation_set)

        params = autoenc.get_model_parameters()

        self.encoding_w_.append(params['enc_w'])
        self.encoding_b_.append(params['enc_b'])

        next_train = autoenc.transform(train_set)
        next_valid = autoenc.transform(validation_set)

        return next_train, next_valid

    def fit(self, train_set, train_labels, validation_set=None, validation_labels=None):

        """ Fit the model to the data.
        :param train_set: Training data. shape(n_samples, n_features)
        :param train_labels: Labels for the data. shape(n_samples, n_classes)
        :param validation_set: optional, default None. Validation data. shape(nval_samples, n_features)
        :param validation_labels: optional, default None. Labels for the validation data. shape(nval_samples, n_classes)
        :return: self
        """

        print('Starting supervised finetuning...')

        n_features = train_set.shape[1]
        n_classes = train_labels.shape[1]

        self._build_model(n_features, n_classes)

        with tf.Session() as self.tf_session:

            self._initialize_tf_utilities_and_ops()
            self._train_model(train_set, train_labels, validation_set, validation_labels)
            self.tf_saver.save(self.tf_session, self.model_path)

    def _initialize_tf_utilities_and_ops(self):

        """ Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
        """

        self.tf_merged_summaries = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()
        self.tf_saver = tf.train.Saver()

        self.tf_session.run(init_op)

        self.tf_summary_writer = tf.train.SummaryWriter(self.tf_summary_dir, self.tf_session.graph_def)

    def _train_model(self, train_set, train_labels, validation_set, validation_labels):

        """ Train the model.
        :param train_set: training set
        :param train_labels: training labels
        :param validation_set: validation set
        :param validation_labels: validation labels
        :return: self
        """

        shuff = zip(train_set, train_labels)

        for i in range(self.finetune_num_epochs):

            np.random.shuffle(shuff)
            batches = [_ for _ in utilities.gen_batches(shuff, self.finetune_batch_size)]

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.tf_session.run(self.train_step, feed_dict={self.input_data: x_batch,
                                                                self.input_labels: y_batch,
                                                                self.keep_prob: self.dropout})

            if validation_set is not None:
                self._run_validation_error_and_summaries(i, validation_set, validation_labels)

    def _run_validation_error_and_summaries(self, epoch, validation_set, validation_labels):

        """ Run the summaries and error computation on the validation set.
        :param epoch: current epoch
        :param validation_set: validation data
        :return: self
        """

        feed = {self.input_data: validation_set, self.input_labels: validation_labels, self.keep_prob: 1}
        result = self.tf_session.run([self.tf_merged_summaries, self.accuracy], feed_dict=feed)
        summary_str = result[0]
        acc = result[1]

        self.tf_summary_writer.add_summary(summary_str, epoch)

        if self.verbose == 1:
            print("Accuracy at step %s: %s" % (epoch, acc))

    def predict(self, test_set, test_labels):

        """ Compute the accuracy over the test set.
        :param test_set: Testing data. shape(n_test_samples, n_features)
        :param test_labels: Labels for the test data. shape(n_test_samples, n_classes)
        :return: accuracy
        """

        with tf.Session() as self.tf_session:
            self.tf_saver.restore(self.tf_session, self.model_path)
            return self.accuracy.eval({self.input_data: test_set,
                                       self.input_labels: test_labels,
                                       self.keep_prob: 1})

    def _build_model(self, n_features, n_classes):

        """ Creates the computational graph.
        This graph is intented to be created for finetuning,
        i.e. after unsupervised pretraining.
        :param n_features: Number of features.
        :param n_classes: number of classes.
        :return: self
        """

        self._create_placeholders(n_features, n_classes)
        self._create_variables(n_features)

        next_train = self._create_encoding_layers()
        self.encode = next_train

        self._create_softmax_layer(next_train, n_classes)

        self._create_cost_function_node()
        self._create_train_step_node()

        self._create_test_node()

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

        if not self.xavier_init[0]:
            xinit = 1
        else:
            xinit = self.xavier_init[0]

        self.encoding_w_ = []
        self.encoding_b_ = []

        for l, layer in enumerate(self.layers):

            if l == 0:
                self.encoding_w_.append(tf.Variable(utilities.xavier_init(n_features, self.layers[l], xinit)))
                self.encoding_b_.append(tf.Variable(tf.zeros([self.layers[l]])))
            else:
                self.encoding_w_.append(tf.Variable(utilities.xavier_init(self.layers[l-1], self.layers[l], xinit)))
                self.encoding_b_.append(tf.Variable(tf.zeros([self.layers[l]])))

    def _create_variables_pretrain(self):

        """ Create model variables (previous unsupervised pretraining)
        :return: self
        """

        for l, layer in enumerate(self.layers):
            self.encoding_w_[l] = tf.Variable(self.encoding_w_[l], name='enc-w-{}'.format(l))
            self.encoding_b_[l] = tf.Variable(self.encoding_b_[l], name='enc-b-{}'.format(l))

    def _create_encoding_layers(self):

        """ Create the encoding layers for finetuning.
        :return: output of the final encoding layer.
        """

        next_train = self.input_data

        for l, layer in enumerate(self.layers):

            with tf.name_scope("encode-{}".format(l)):

                y_act = tf.matmul(next_train, self.encoding_w_[l]) + self.encoding_b_[l]

                if self.finetune_act_func == 'sigmoid':
                    layer_y = tf.nn.sigmoid(y_act)

                elif self.finetune_act_func == 'tanh':
                    layer_y = tf.nn.tanh(y_act)

                elif self.finetune_act_func == 'relu':
                    layer_y = tf.nn.relu(y_act)

                else:
                    layer_y = None

            # the input to the next layer is the output of this layer
            next_train = tf.nn.dropout(layer_y, self.keep_prob)

        return next_train

    def _create_softmax_layer(self, last_layer, n_classes):

        """ Create the softmax layer for finetuning.
        :param last_layer: last layer output node
        :param n_classes: number of classes
        :return: self
        """

        self.softmax_W = tf.Variable(tf.truncated_normal([self.layers[-1], n_classes]),
                                     name='softmax-weigths')
        self.softmax_b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name='softmax-biases')

        with tf.name_scope("softmax_layer"):
            self.softmax_out = tf.matmul(last_layer, self.softmax_W) + self.softmax_b

    def _create_cost_function_node(self):

        """ Create the cost function node.
        :return: self
        """

        with tf.name_scope("cost"):
            if self.softmax_loss_func == 'cross_entropy':
                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.softmax_out, self.input_labels))
                _ = tf.scalar_summary("cross_entropy", self.cost)

            elif self.softmax_loss_func == 'mean_squared':
                self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.input_labels - self.softmax_out)))
                _ = tf.scalar_summary("mean_squared", self.cost)

            else:
                self.cost = None

    def _create_train_step_node(self):

        """ Create the training step node of the network.
        :return: self
        """

        with tf.name_scope("train"):
            if self.finetune_opt == 'gradient_descent':
                self.train_step = tf.train.GradientDescentOptimizer(self.finetune_learning_rate).minimize(self.cost)

            elif self.finetune_opt == 'ada_grad':
                self.train_step = tf.train.AdagradOptimizer(self.finetune_learning_rate).minimize(self.cost)

            elif self.finetune_opt == 'momentum':
                self.train_step = tf.train.MomentumOptimizer(self.finetune_learning_rate, self.momentum).minimize(self.cost)

            else:
                self.train_step = None

    def _create_test_node(self):

        """ Create the test node of the network.
        :return: self
        """

        with tf.name_scope("test"):
            correct_prediction = tf.equal(tf.argmax(self.softmax_out, 1), tf.argmax(self.input_labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            _ = tf.scalar_summary('accuracy', self.accuracy)

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
