import tensorflow as tf
import numpy as np
import os

from yadlt.core import model
from yadlt.models.rbm_models import rbm
from yadlt.utils import utilities


class DeepBeliefNetwork(model.Model):

    """ Implementation of Deep Belief Network for Supervised Learning using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, rbm_layers, model_name='srbm', do_pretrain=True, main_dir='srbm/', models_dir='models/', data_dir='data/',
                 summary_dir='logs/', rbm_num_epochs=list([10]), rbm_gibbs_k=list([1]), rbm_gauss_visible=False, rbm_stddev=0.1,
                 rbm_batch_size=list([10]), dataset='mnist', rbm_learning_rate=list([0.01]), momentum=0.5,
                 finetune_dropout=1, verbose=1, finetune_loss_func='softmax_cross_entropy', finetune_act_func='relu',
                 finetune_opt='gradient_descent', finetune_learning_rate=0.001, finetune_num_epochs=10,
                 finetune_batch_size=20):
        """
        :param rbm_layers: list containing the hidden units for each layer
        :param finetune_loss_func: Loss function for the softmax layer. string, default ['softmax_cross_entropy', 'mean_squared']
        :param finetune_dropout: dropout parameter
        :param finetune_learning_rate: learning rate for the finetuning. float, default 0.001
        :param finetune_act_func: activation function for the finetuning phase
        :param finetune_opt: optimizer for the finetuning phase
        :param finetune_num_epochs: Number of epochs for the finetuning. int, default 20
        :param finetune_batch_size: Size of each mini-batch for the finetuning. int, default 20
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy. int, default 0
        :param do_pretrain: True: uses variables from pretraining, False: initialize new variables.
        """

        model.Model.__init__(self, model_name, main_dir, models_dir, data_dir, summary_dir)

        self._initialize_training_parameters(loss_func=finetune_loss_func, learning_rate=finetune_learning_rate,
                                             dropout=finetune_dropout, num_epochs=finetune_num_epochs,
                                             batch_size=finetune_batch_size, dataset=dataset, opt=finetune_opt,
                                             momentum=momentum)

        self.do_pretrain = do_pretrain
        self.layers = rbm_layers
        self.finetune_act_func = finetune_act_func
        self.verbose = verbose

        # Model parameters
        self.encoding_w_ = []  # list of matrices of encoding weights (one per layer)
        self.encoding_b_ = []  # list of arrays of encoding biases (one per layer)

        self.softmax_W = None
        self.softmax_b = None

        rbm_params = {'num_epochs': rbm_num_epochs, 'gibbs_k': rbm_gibbs_k, 'batch_size': rbm_batch_size,
                      'learning_rate': rbm_learning_rate}
        for p in rbm_params:
            if len(rbm_params[p]) != len(rbm_layers):
                # The current parameter is not specified by the user, should default it for all the layers
                rbm_params[p] = [rbm_params[p][0] for _ in rbm_layers]

        self.rbms = []
        self.rbm_graphs = []

        for l, layer in enumerate(rbm_layers):
            rbm_str = 'rbm-' + str(l+1)

            if l == 0 and rbm_gauss_visible:
                self.rbms.append(rbm.RBM(model_name=self.model_name + '-' + rbm_str,
                    models_dir=os.path.join(self.models_dir, rbm_str), data_dir=os.path.join(self.data_dir, rbm_str),  summary_dir=os.path.join(self.tf_summary_dir, rbm_str),
                    num_hidden=layer, main_dir=self.main_dir, learning_rate=rbm_params['learning_rate'][l], dataset=self.dataset,
                    verbose=self.verbose, num_epochs=rbm_params['num_epochs'][l], batch_size=rbm_params['batch_size'][l],
                    gibbs_sampling_steps=rbm_params['gibbs_k'][l], visible_unit_type='gauss', stddev=rbm_stddev))

            else:
                self.rbms.append(rbm.RBM(model_name=self.model_name + '-' + rbm_str,
                    models_dir=os.path.join(self.models_dir, rbm_str), data_dir=os.path.join(self.data_dir, rbm_str),  summary_dir=os.path.join(self.tf_summary_dir, rbm_str),
                    num_hidden=layer, main_dir=self.main_dir, learning_rate=rbm_params['learning_rate'][l], dataset=self.dataset,
                    verbose=self.verbose, num_epochs=rbm_params['num_epochs'][l], batch_size=rbm_params['batch_size'][l],
                    gibbs_sampling_steps=rbm_params['gibbs_k'][l]))

            self.rbm_graphs.append(tf.Graph())

    def pretrain(self, train_set, validation_set=None):

        """ Perform unsupervised pretraining of the stack of denoising autoencoders.
        :param train_set: training set
        :param validation_set: validation set
        :return: return data encoded by the last layer
        """

        next_train = train_set
        next_valid = validation_set

        for l, rbmachine in enumerate(self.rbms):
            print('Training layer {}...'.format(l+1))
            next_train, next_valid = self._pretrain_rbm_and_gen_feed(rbmachine,
                                                                     next_train,
                                                                     next_valid,
                                                                     self.rbm_graphs[l])

        return next_train, next_valid

    def _pretrain_rbm_and_gen_feed(self, rbmachine, train_set, validation_set, graph=None):

        """ Pretrain a single autoencoder and encode the data for the next layer.
        :param rbmachine: autoencoder reference
        :param train_set: training set
        :param validation_set: validation set
        :param graph: tf object for the rbm
        :return: encoded train data, encoded validation data
        """

        rbmachine.fit(train_set, validation_set, graph=graph)

        with graph.as_default():
            params = rbmachine.get_model_parameters(graph=graph)
            self.encoding_w_.append(params['W'])
            self.encoding_b_.append(params['bh_'])

            next_train = rbmachine.transform(train_set, graph=graph)
            next_valid = rbmachine.transform(validation_set, graph=graph)

        return next_train, next_valid

    def fit(self, train_set, train_labels, validation_set=None, validation_labels=None, restore_previous_model=False):

        """ Fit the model to the data.
        :param train_set: Training data. shape(n_samples, n_features)
        :param train_labels: Labels for the data. shape(n_samples, n_classes)
        :param validation_set: optional, default None. Validation data. shape(nval_samples, n_features)
        :param validation_labels: optional, default None. Labels for the validation data. shape(nval_samples, n_classes)
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :return: self
        """

        print('Starting Supervised finetuning...')

        with self.tf_graph.as_default():
            self.build_model(train_set.shape[1], train_labels.shape[1])
            with tf.Session() as self.tf_session:
                self._initialize_tf_utilities_and_ops(restore_previous_model)
                self._train_model(train_set, train_labels, validation_set, validation_labels)
                self.tf_saver.save(self.tf_session, self.model_path)

    def _train_model(self, train_set, train_labels, validation_set, validation_labels):

        """ Train the model.
        :param train_set: training set
        :param train_labels: training labels
        :param validation_set: validation set
        :param validation_labels: validation labels
        :return: self
        """

        shuff = zip(train_set, train_labels)

        for i in range(self.num_epochs):

            np.random.shuffle(shuff)
            batches = [_ for _ in utilities.gen_batches(shuff, self.batch_size)]

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.tf_session.run(self.train_step, feed_dict={self.input_data: x_batch,
                                                                self.input_labels: y_batch,
                                                                self.keep_prob: self.dropout})

            if validation_set is not None:
                feed = {self.input_data: validation_set, self.input_labels: validation_labels, self.keep_prob: 1}
                self._run_supervised_validation_error_and_summaries(i, feed)

    def get_layers_output(self, dataset):

        """ Get output from each layer of the network.
        :param dataset: input data
        :return: list of np array, element i in the list is the output of layer i
        """

        layers_out = []

        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                for l in self.layer_nodes:
                    layers_out.append(l.eval({self.input_data: dataset,
                                              self.keep_prob: 1}))
        return layers_out

    def compute_accuracy(self, test_set, test_labels):

        """ Compute the accuracy over the test set.
        :param test_set: Testing data. shape(n_test_samples, n_features)
        :param test_labels: Labels for the test data. shape(n_test_samples, n_classes)
        :return: accuracy
        """

        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                return self.accuracy.eval({self.input_data: test_set,
                                           self.input_labels: test_labels,
                                           self.keep_prob: 1})

    def build_model(self, n_features, n_classes):

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
        last_out = self._create_last_layer(next_train, n_classes)

        self._create_cost_function_node(last_out, self.input_labels)
        self._create_train_step_node()

        self._create_supervised_test_node()

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
                self.encoding_b_.append(tf.Variable(tf.constant(0.1, shape=[self.layers[l]])))
            else:
                self.encoding_w_.append(tf.Variable(tf.truncated_normal(shape=[self.layers[l-1], self.layers[l]], stddev=0.1)))
                self.encoding_b_.append(tf.Variable(tf.constant(0.1, shape=[self.layers[l]])))

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

            self.layer_nodes.append(next_train)

        return next_train
