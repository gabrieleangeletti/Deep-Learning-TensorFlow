import tensorflow as tf
import numpy as np

from yadlt.core import model
from yadlt.utils import utilities


class LogisticRegression(model.Model):

    """Simple Logistic Regression using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, main_dir='lr/', model_name='lr', models_dir='models/', data_dir='data/', summary_dir='logs/',
                 loss_func='cross_entropy', dataset='mnist', learning_rate=0.01, verbose=0, num_epochs=10, batch_size=10):

        """
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
        """
        model.Model.__init__(self, model_name, main_dir, models_dir, data_dir, summary_dir)

        self._initialize_training_parameters(loss_func, learning_rate, num_epochs, batch_size,
                                             dataset, None, None)

        self.verbose = verbose

        # Computational graph nodes
        self.input_data = None
        self.input_labels = None

        self.W_ = None
        self.b_ = None

        self.accuracy = None

    def build_model(self, n_features, n_classes):

        """ Creates the computational graph.
        :param n_features: number of features
        :param n_classes: number of classes
        :return: self
        """

        self._create_placeholders(n_features, n_classes)
        self._create_variables(n_features, n_classes)

        self.last_out = tf.nn.softmax(tf.matmul(self.input_data, self.W_) + self.b_)

        self._create_cost_function_node(self.last_out, self.input_labels)
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        self._create_supervised_test_node()

    def _create_placeholders(self, n_features, n_classes):

        """ Create the TensorFlow placeholders for the model.
        :param n_features: number of features
        :param n_classes: number of classes
        :return: self
        """

        self.input_data = tf.placeholder("float", [None, n_features], name='x-input')
        self.input_labels = tf.placeholder("float", [None, n_classes], name='y-input')

    def _create_variables(self, n_features, n_classes):

        """ Create the TensorFlow variables for the model.
        :param n_features: number of features
        :param n_classes: number of classes
        :return: self
        """

        self.W_ = tf.Variable(tf.zeros([n_features, n_classes]), name='weights')
        self.b_ = tf.Variable(tf.zeros([n_classes]), name='biases')

    def fit(self, train_set, train_labels, validation_set=None, validation_labels=None, restore_previous_model=False):

        """ Fit the model to the data.
        :param train_set: Training data. shape(n_samples, n_features).
        :param train_labels: Labels for the data. shape(n_samples, n_classes).
        :param validation_set: optional, default None. Validation data. shape(n_validation_samples, n_features).
        :param validation_labels: optional, default None. Labels for the validation data. shape(n_validation_samples, n_classes).
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :return: self
        """

        with self.tf_graph.as_default():
            self.build_model(train_set.shape[1], train_labels.shape[1])
            with tf.Session() as self.tf_session:
                self._initialize_tf_utilities_and_ops(restore_previous_model)
                self._train_model(train_set, train_labels, validation_set, validation_labels)
                self.tf_saver.save(self.tf_session, self.models_dir + self.model_name)

    def _train_model(self, train_set, train_labels, validation_set, validation_labels):

        """ Train the model.
        :param train_set: training set
        :param train_labels: training labels
        :param validation_set: validation set
        :param validation_labels: validation labels
        :return: self
        """

        for i in range(self.num_epochs):

            shuff = zip(train_set, train_labels)
            np.random.shuffle(shuff)

            batches = [_ for _ in utilities.gen_batches(zip(train_set, train_labels), self.batch_size)]

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.tf_session.run(self.train_step, feed_dict={self.input_data: x_batch, self.input_labels: y_batch})

            if validation_set is not None:
                feed = {self.input_data: validation_set, self.input_labels: validation_labels}
                self._run_supervised_validation_error_and_summaries(i, feed)

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
