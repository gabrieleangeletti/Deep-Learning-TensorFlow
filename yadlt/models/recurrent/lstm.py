"""LSTM Tensorflow implementation."""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from yadlt.core import Model
from yadlt.utils import utilities


class LSTM(Model):
    """Long Short-Term Memory Network tensorflow implementation.

    The interface of the class is sklearn-like.
    """

    def __init__(self, num_layers=2, num_hidden=200, vocab_size=10000,
                 batch_size=20, num_steps=35, num_epochs=10, learning_rate=1.0,
                 dropout=0.5, init_scale=0.05, max_grad_norm=5,
                 lr_decay=0.8):
        """Constructor.

        :param num_layers: number of LSTM layers
        :param num_hidden: number of LSTM units
        :param vocab_size: size of the vocabulary
        :param batch_size: size of each mini batch
        :param num_steps: number of unrolled steps of LSTM
        :param num_epochs: number of training epochs
        :param learning_rate: learning rate parameter
        :param dropout: probability of the dropout layer
        :param init_scale: initial scale of the weights
        :param max_grad_norm: maximum permissible norm of the gradient
        :param lr_decay: learning rate decay for each epoch after num_epochs/3
        """
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.init_scale = init_scale
        self.max_grad_norm = max_grad_norm
        self.lr_decay = lr_decay

        self.initializer = tf.random_uniform_initializer(
            -self.init_scale, self.init_scale)

    def fit(self, train_set, test_set):
        """Fit the model to the given data.

        :param train_set: training data
        :param test_set: test data
        """
        with tf.Graph().as_default(), tf.Session() as self.tf_session:
            self.build_model()
            tf.initialize_all_variables().run()
            third = self.num_epochs // 3

            for i in range(self.num_epochs):
                lr_decay = self.lr_decay ** max(i - third, 0.0)
                self.tf_session.run(
                    tf.assign(self.lr_var, tf.mul(self.learning_rate, lr_decay)))

                train_perplexity = self._run_train_step(train_set, 'train')
                print("Epoch: %d Train Perplexity: %.3f"
                      % (i + 1, train_perplexity))

            test_perplexity = self._run_train_step(test_set, 'test')
            print("Test Perplexity: %.3f" % test_perplexity)

    def _run_train_step(self, data, mode='train'):
        """Run a single training step.

        :param data: input data
        :param mode: 'train' or 'test'.
        """
        epoch_size = ((len(data) // self.batch_size) - 1) // self.num_steps
        costs = 0.0
        iters = 0
        step = 0
        state = self._init_state.eval()
        op = self._train_op if mode == 'train' else tf.no_op()

        for step, (x, y) in enumerate(
            utilities.seq_data_iterator(
                data, self.batch_size, self.num_steps)):
            cost, state, _ = self.tf_session.run(
                [self.cost, self.final_state, op],
                {self.input_data: x,
                 self.input_labels: y,
                 self._init_state: state})

            costs += cost
            iters += self.num_steps

        if step % (epoch_size // 10) == 10:
            print("%.3f perplexity" % (step * 1.0 / epoch_size))

        return np.exp(costs / iters)

    def build_model(self):
        """Build the model's computational graph."""
        with tf.variable_scope(
                "model", reuse=None, initializer=self.initializer):
            self._create_placeholders()
            self._create_rnn_cells()
            self._create_initstate_and_embeddings()
            self._create_rnn_architecture()
            self._create_optimizer_node()

    def _create_placeholders(self):
        """Create the computational graph's placeholders."""
        self.input_data = tf.placeholder(
            tf.int32, [self.batch_size, self.num_steps])
        self.input_labels = tf.placeholder(
            tf.int32, [self.batch_size, self.num_steps])

    def _create_rnn_cells(self):
        """Create the LSTM cells."""
        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            self.num_hidden, forget_bias=0.0)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=self.dropout)
        self.cell = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell] * self.num_layers)

    def _create_initstate_and_embeddings(self):
        """Create the initial state for the cell and the data embeddings."""
        self._init_state = self.cell.zero_state(self.batch_size, tf.float32)
        embedding = tf.get_variable(
            "embedding", [self.vocab_size, self.num_hidden])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        self.inputs = tf.nn.dropout(inputs, self.dropout)

    def _create_rnn_architecture(self):
        """Create the training architecture and the last layer of the LSTM."""
        self.inputs = [tf.squeeze(i, [1]) for i in tf.split(
            1, self.num_steps, self.inputs)]
        outputs, state = tf.nn.rnn(
            self.cell, self.inputs, initial_state=self._init_state)

        output = tf.reshape(tf.concat(1, outputs), [-1, self.num_hidden])
        softmax_w = tf.get_variable(
            "softmax_w", [self.num_hidden, self.vocab_size])
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
        logits = tf.add(tf.matmul(output, softmax_w), softmax_b)
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.input_labels, [-1])],
            [tf.ones([self.batch_size * self.num_steps])])

        self.cost = tf.div(tf.reduce_sum(loss), self.batch_size)
        self.final_state = state

    def _create_optimizer_node(self):
        """Create the optimizer node of the graph."""
        self.lr_var = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          self.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr_var)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
