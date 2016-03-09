import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import config
from tf_models.rbm_models import rbm
from utils import utilities


class DBN(object):

    """ Implementation of Deep Belief Network using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, layers, do_pretrain=True, rbm_num_epochs=list([10]), rbm_batch_size=list([10]), model_name='dbn',
                 rbm_learning_rate=list([0.01]), rbm_names=list(['']), rbm_gibbs_k=list([1]), gauss_visible=False,
                 stddev=0.1, learning_rate=0.01, momentum=0.7, num_epochs=10, batch_size=10, dropout=1,
                 opt='gradient_descent', verbose=1, act_func='relu', loss_func='mean_squared', main_dir='dbn/',
                 dataset='mnist'):

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

        # DBN parameters
        self.gauss_visible = gauss_visible
        self.stddev = stddev
        self.act_func = act_func
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

        self.W_pretrain = None
        self.bh_pretrain = None
        self.bv_pretrain = None

        self.W_vars = None
        self.bh_vars = None
        self.bv_vars = None

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

        """ Perform unsupervised pretraining of the stack of restricted boltzmann machines.
        :param train_set: training set
        :param validation_set: validation set
        :return: return data encoded by the last layer
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

        rboltz.fit(train_set, validation_set)
        params = rboltz.get_model_parameters()

        self.W_pretrain.append(params['W'])
        self.bh_pretrain.append(params['bh_'])
        self.bv_pretrain.append(params['bv_'])

        return rboltz.transform(train_set), rboltz.transform(validation_set)

    def finetune(self, train_set, train_labels, validation_set=None, validation_labels=None):

        """ Perform supervised finetuning of the model.
        :param train_set: training set
        :param train_labels: training labels
        :param validation_set: validation set
        :param validation_labels: validation labels
        :return: self
        """

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

    def _train_model(self, train_set, train_labels, validation_set=None, validation_labels=None):

        """ Train the model.
        :param train_set: training set
        :param train_labels: training labels
        :param validation_set: validation set
        :param validation_labels: validation labels
        :return:
        """

        shuff = zip(train_set, train_labels)

        for i in range(self.num_epochs):

            np.random.shuffle(shuff)
            batches = [_ for _ in utilities.gen_batches(shuff, self.batch_size)]

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                feed = self._create_feed(x_batch, y_batch, self.dropout)
                self.tf_session.run(self.train_step, feed_dict=feed)

            if validation_set is not None:
                self._run_validation_error_and_summaries(i, validation_set, validation_labels)

    def _run_validation_error_and_summaries(self, epoch, validation_set, validation_labels):

        """ Run the summaries and error computation on the validation set.
        :param epoch: current epoch number
        :param validation_set: validation set
        :param validation_labels: validation labels
        :return: self
        """

        feed = self._create_feed(validation_set, validation_labels, 1)
        result = self.tf_session.run([self.tf_merged_summaries, self.cost], feed_dict=feed)
        summary_str = result[0]
        err = result[1]

        self.tf_summary_writer.add_summary(summary_str, epoch)

        if self.verbose == 1:
            print("Cost at step %s: %s" % (epoch, err))

    def _build_model(self, n_features, n_classes):

        """ Assume self.W, self.bh and self.bv contains trained parameters.
        :param n_features: number of features
        :param n_classes: number of classes
        :return: self
        """

        self._create_placeholders(n_features, n_classes)
        self._create_variables()

        next_train = self._forward_pass()
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

        self.keep_prob = tf.placeholder('float')
        self.hrand = [tf.placeholder('float', [None, self.layers[l+1]]) for l in range(self.n_layers-1)]
        self.vrand = [tf.placeholder('float', [None, self.layers[l]]) for l in range(self.n_layers-1)]

        self.x = tf.placeholder('float', [None, n_features])
        self.y_ = tf.placeholder('float', [None, n_classes])

    def _create_variables(self):

        """ Create the TensorFlow variables for the model.
        :return: self
        """

        self.W_vars = [tf.Variable(self.W_pretrain[l]) for l in range(self.n_layers-1)]
        self.bh_vars = [tf.Variable(self.bh_pretrain[l]) for l in range(self.n_layers-1)]
        self.bv_vars = [tf.Variable(self.bv_pretrain[l]) for l in range(self.n_layers-1)]

    def _forward_pass(self):

        """ Perform a forward pass through the layers of the network.
        :return: sampled units at the last layer
        """

        next_train = self.x

        for l in range(self.n_layers-1):

            activation = tf.matmul(next_train, self.W_vars[l]) + self.bh_vars[l]

            if self.act_func == 'sigmoid':
                hprobs = tf.nn.sigmoid(activation)

            elif self.act_func == 'tanh':
                hprobs = tf.nn.tanh(activation)

            elif self.act_func == 'relu':
                hprobs = tf.nn.relu(activation)

            else:
                hprobs = None

            next_train = tf.nn.dropout(hprobs, self.keep_prob)

        return next_train

    def _create_softmax_layer(self, next_train, n_classes):

        """ Create nodes for the softmax layer build on top of the last layer.
        :param next_train: sampled units at the last layer
        :param n_classes: number of classes
        :return: self
        """
        self.softmax_W = tf.Variable(tf.truncated_normal([self.layers[-1], n_classes]),
                                     name='softmax-weights')
        self.softmax_b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name='softmax-biases')

        with tf.name_scope("softmax_layer"):
            self.y = tf.matmul(next_train, self.softmax_W) + self.softmax_b

    def _create_cost_function_node(self):

        """ Create the cost function node.
        :return: self
        """

        with tf.name_scope("cost"):
            if self.loss_func == 'cross_entropy':
                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
                _ = tf.scalar_summary("cross_entropy", self.cost)

            elif self.loss_func == 'mean_squared':
                self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.y_ - self.y)))
                _ = tf.scalar_summary("mean_squared", self.cost)

            else:
                self.cost = None

    def _create_train_step_node(self):

        """ Create TensorFlow optimizer for the model
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

    def _create_test_node(self):

        """ Create validation testing node.
        :return: self
        """

        with tf.name_scope("test"):
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            _ = tf.scalar_summary('accuracy', self.accuracy)

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
                'bh': eval_tensors(self.bh_vars, 'bh'),
                'bv': eval_tensors(self.bv_vars, 'bv'),
                'smaxW': self.softmax_W.eval(),
                'smaxb': self.softmax_b.eval()
            }

    def predict(self, test_set, test_labels):

        """ Predict the labels for the test set, and return the accuracy
        over the test set.
        :param test_set: test set
        :param test_labels: test labels
        :return: accuracy over the test set
        """

        with tf.Session() as self.tf_session:

            self.tf_saver.restore(self.tf_session, self.model_path)
            feed = self._create_feed(test_set, test_labels, 1)

            return self.accuracy.eval(feed)

    def _create_feed(self, datax, datay, keep_prob):

        """ Create dictionary to feed TensorFlow placeholders.
        :param datax: input data
        :param datay: input labels
        :param keep_prob: dropout probability
        :return: feed dictionary
        """

        feed = {self.x: datax, self.y_: datay, self.keep_prob: keep_prob}

        # Random uniform for encoding layers
        hrand = [np.random.rand(len(datax), l) for l in self.layers[1:]]
        for j, h in enumerate(hrand):
            feed[self.hrand[j]] = h

        # Random uniform for decoding layers
        vrand = [np.random.rand(len(datax), l) for l in self.layers[:-1]]
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
