"""Implementation of Deep Belief Network Model using TensorFlow."""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from yadlt.core import SupervisedModel
from yadlt.core import Evaluation, Layers, Loss, Trainer
from yadlt.models.boltzmann import rbm
from yadlt.utils import tf_utils, utilities


class DeepBeliefNetwork(SupervisedModel):
    """Implementation of Deep Belief Network for Supervised Learning.

    The interface of the class is sklearn-like.
    """

    def __init__(
        self, rbm_layers, name='dbn', do_pretrain=False,
        rbm_num_epochs=[10], rbm_gibbs_k=[1],
        rbm_gauss_visible=False, rbm_stddev=0.1, rbm_batch_size=[10],
        rbm_learning_rate=[0.01], finetune_dropout=1,
        finetune_loss_func='softmax_cross_entropy',
        finetune_act_func=tf.nn.sigmoid, finetune_opt='sgd',
        finetune_learning_rate=0.001, finetune_num_epochs=10,
            finetune_batch_size=20, momentum=0.5):
        """Constructor.

        :param rbm_layers: list containing the hidden units for each layer
        :param finetune_loss_func: Loss function for the softmax layer.
            string, default ['softmax_cross_entropy', 'mse']
        :param finetune_dropout: dropout parameter
        :param finetune_learning_rate: learning rate for the finetuning.
            float, default 0.001
        :param finetune_act_func: activation function for the finetuning phase
        :param finetune_opt: optimizer for the finetuning phase
        :param finetune_num_epochs: Number of epochs for the finetuning.
            int, default 20
        :param finetune_batch_size: Size of each mini-batch for the finetuning.
            int, default 20
        :param do_pretrain: True: uses variables from pretraining,
            False: initialize new variables.
        """
        SupervisedModel.__init__(self, name)

        self.loss_func = finetune_loss_func
        self.learning_rate = finetune_learning_rate
        self.opt = finetune_opt
        self.num_epochs = finetune_num_epochs
        self.batch_size = finetune_batch_size
        self.momentum = momentum
        self.dropout = finetune_dropout

        self.loss = Loss(self.loss_func)
        self.trainer = Trainer(
            finetune_opt, learning_rate=finetune_learning_rate,
            momentum=momentum)

        self.do_pretrain = do_pretrain
        self.layers = rbm_layers
        self.finetune_act_func = finetune_act_func

        # Model parameters
        self.encoding_w_ = []  # list of matrices of encoding weights per layer
        self.encoding_b_ = []  # list of arrays of encoding biases per layer

        self.softmax_W = None
        self.softmax_b = None

        rbm_params = {
            'num_epochs': rbm_num_epochs, 'gibbs_k': rbm_gibbs_k,
            'batch_size': rbm_batch_size, 'learning_rate': rbm_learning_rate}

        for p in rbm_params:
            if len(rbm_params[p]) != len(rbm_layers):
                # The current parameter is not specified by the user,
                # should default it for all the layers
                rbm_params[p] = [rbm_params[p][0] for _ in rbm_layers]

        self.rbms = []
        self.rbm_graphs = []

        for l, layer in enumerate(rbm_layers):
            rbm_str = 'rbm-' + str(l+1)

            if l == 0 and rbm_gauss_visible:
                self.rbms.append(
                    rbm.RBM(
                        name=self.name + '-' + rbm_str,
                        num_hidden=layer,
                        learning_rate=rbm_params['learning_rate'][l],
                        num_epochs=rbm_params['num_epochs'][l],
                        batch_size=rbm_params['batch_size'][l],
                        gibbs_sampling_steps=rbm_params['gibbs_k'][l],
                        visible_unit_type='gauss', stddev=rbm_stddev))

            else:
                self.rbms.append(
                    rbm.RBM(
                        name=self.name + '-' + rbm_str,
                        num_hidden=layer,
                        learning_rate=rbm_params['learning_rate'][l],
                        num_epochs=rbm_params['num_epochs'][l],
                        batch_size=rbm_params['batch_size'][l],
                        gibbs_sampling_steps=rbm_params['gibbs_k'][l]))

            self.rbm_graphs.append(tf.Graph())

    def pretrain(self, train_set, validation_set=None):
        """Perform Unsupervised pretraining of the DBN."""
        self.do_pretrain = True

        def set_params_func(rbmmachine, rbmgraph):
            params = rbmmachine.get_parameters(graph=rbmgraph)
            self.encoding_w_.append(params['W'])
            self.encoding_b_.append(params['bh_'])

        return SupervisedModel.pretrain_procedure(
            self, self.rbms, self.rbm_graphs, set_params_func=set_params_func,
            train_set=train_set, validation_set=validation_set)

    def _train_model(self, train_set, train_labels,
                     validation_set, validation_labels):
        """Train the model.

        :param train_set: training set
        :param train_labels: training labels
        :param validation_set: validation set
        :param validation_labels: validation labels
        :return: self
        """
        shuff = list(zip(train_set, train_labels))

        pbar = tqdm(range(self.num_epochs))
        for i in pbar:

            np.random.shuffle(shuff)
            batches = [_ for _ in utilities.gen_batches(
                shuff, self.batch_size)]

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.tf_session.run(
                    self.train_step, feed_dict={
                        self.input_data: x_batch,
                        self.input_labels: y_batch,
                        self.keep_prob: self.dropout})

            if validation_set is not None:
                feed = {self.input_data: validation_set,
                        self.input_labels: validation_labels,
                        self.keep_prob: 1}
                acc = tf_utils.run_summaries(
                    self.tf_session, self.tf_merged_summaries,
                    self.tf_summary_writer, i, feed, self.accuracy)
                pbar.set_description("Accuracy: %s" % (acc))

    def build_model(self, n_features, n_classes):
        """Create the computational graph.

        This graph is intented to be created for finetuning,
        i.e. after unsupervised pretraining.
        :param n_features: Number of features.
        :param n_classes: number of classes.
        :return: self
        """
        self._create_placeholders(n_features, n_classes)
        self._create_variables(n_features)

        next_train = self._create_encoding_layers()
        self.mod_y, _, _ = Layers.linear(next_train, n_classes)
        self.layer_nodes.append(self.mod_y)

        self.cost = self.loss.compile(self.mod_y, self.input_labels)
        self.train_step = self.trainer.compile(self.cost)
        self.accuracy = Evaluation.accuracy(self.mod_y, self.input_labels)

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

        self.keep_prob = tf.placeholder(tf.float32, name='keep-probs')

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
                self.encoding_w_.append(tf.Variable(tf.truncated_normal(
                    shape=[n_features, self.layers[l]], stddev=0.1)))
                self.encoding_b_.append(tf.Variable(tf.constant(
                    0.1, shape=[self.layers[l]])))
            else:
                self.encoding_w_.append(tf.Variable(tf.truncated_normal(
                    shape=[self.layers[l-1], self.layers[l]], stddev=0.1)))
                self.encoding_b_.append(tf.Variable(tf.constant(
                    0.1, shape=[self.layers[l]])))

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

                if self.finetune_act_func:
                    layer_y = self.finetune_act_func(y_act)

                else:
                    layer_y = None

                # the input to the next layer is the output of this layer
                next_train = tf.nn.dropout(layer_y, self.keep_prob)

            self.layer_nodes.append(next_train)

        return next_train
