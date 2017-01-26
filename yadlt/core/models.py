"""Model classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import errno
import os
import six
import tensorflow as tf

import layers.BaseLayer as BaseLayer
from os.path import expanduser


@six.add_metaclass(abc.ABCMeta)
class BaseModel(object):
    """Base model interface."""

    @abc.abstractmethod
    def fit(self):
        """Model training procedure."""
        pass

    @abc.abstractmethod
    def predict(self):
        """Model inference procedure."""
        pass

    @abc.abstractmethod
    def score(self):
        """Compute the model score."""
        pass

    @abc.abstractmethod
    def save(self):
        """Save the model to disk."""
        pass

    @abc.abstractmethod
    def load(self):
        """Load the model from disk."""
        pass

    @abc.abstractmethod
    def get_parameters(self):
        """Get the model parameters."""
        pass


class Sequential(BaseModel):
    """Linear stack of layers model."""

    def __init__(self, layers=None):
        """Create a Sequential model."""
        self.layers = []

    def add(self, layer):
        """Add a layer to the stack."""
        if not isinstance(layer, BaseLayer):
            raise TypeError('Should be an instance of BaseLayer.')
        self.layers.append(layer)

    def forward(self, X):
        """Forward propagate X through the model."""
        out = X
        for l in self.layers:
            out = l.forward(out)
        return out


class Model(object):
    """Class representing an abstract Model."""

    def __init__(self, model_name, main_dir, models_dir,
                 data_dir, summary_dir):
        """Constructor.

        :param model_name: name of the model, used as filename.
            string, default 'dae'
        :param main_dir: main directory to put the stored_models,
            data and summary directories
        :param models_dir: directory to store trained models
        :param data_dir: directory to store generated data
        :param summary_dir: directory to store tensorflow logs
        """
        home = os.path.join(expanduser("~"), '.yadlt')
        main_dir = os.path.join(home, main_dir)
        models_dir = os.path.join(home, models_dir)
        data_dir = os.path.join(home, data_dir)
        summary_dir = os.path.join(home, summary_dir)

        self.model_name = model_name
        self.main_dir = main_dir
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.tf_summary_dir = summary_dir
        self.model_path = os.path.join(self.models_dir, self.model_name)

        print('Creating %s directory to save/restore models'
              % (self.models_dir))
        self._create_dir(self.models_dir)
        print('Creating %s directory to save model generated data'
              % (self.data_dir))
        self._create_dir(self.data_dir)
        print('Creating %s directory to save tensorboard logs'
              % (self.tf_summary_dir))
        self._create_dir(self.tf_summary_dir)

        self.input_data = None
        self.input_labels = None
        self.keep_prob = None
        self.layer_nodes = []  # list of layers of the final network
        self.last_out = None
        self.train_step = None
        self.cost = None
        self.verbose = 0

        # tensorflow objects
        self.tf_graph = tf.Graph()
        self.tf_session = None
        self.tf_saver = None
        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.tf_summary_writer_available = True

    def _create_dir(self, dirpath):
        """Create directory dirpath."""
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def _initialize_tf_utilities_and_ops(self, restore_previous_model):
        """Initialize TensorFlow operations.

        tf operations: summaries, init operations, saver, summary_writer.
        Restore a previously trained model if the flag restore_previous_model
            is true.
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk
                    to continue training.
        """
        self.tf_merged_summaries = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        self.tf_saver = tf.train.Saver()

        self.tf_session.run(init_op)

        if restore_previous_model:
            self.tf_saver.restore(self.tf_session, self.model_path)

        # Retrieve run identifier
        run_id = 0
        for e in os.listdir(self.tf_summary_dir):
            if e[:3] == 'run':
                r = int(e[3:])
                if r > run_id:
                    run_id = r
        run_id += 1
        run_dir = os.path.join(self.tf_summary_dir, 'run' + str(run_id))
        print('Tensorboard logs dir for this run is %s' % (run_dir))

        self.tf_summary_writer = tf.summary.FileWriter(
            run_dir, self.tf_session.graph)

    def compute_regularization(self, vars):
        """Compute the regularization tensor.

        :param vars: list of model variables
        :return:
        """
        if self.regtype != 'none':

            regularizers = tf.constant(0.0)

            for v in vars:
                if self.regtype == 'l2':
                    regularizers = tf.add(regularizers, tf.nn.l2_loss(v))
                elif self.regtype == 'l1':
                    regularizers = tf.add(
                        regularizers, tf.reduce_sum(tf.abs(v)))

            return tf.mul(self.l2reg, regularizers)
        else:
            return None

    def pretrain_procedure(self, layer_objs, layer_graphs, set_params_func,
                           train_set, validation_set=None):
        """Perform unsupervised pretraining of the model.

        :param layer_objs: list of model objects (autoencoders or rbms)
        :param layer_graphs: list of model tf.Graph objects
        :param set_params_func: function used to set the parameters after
            pretraining
        :param train_set: training set
        :param validation_set: validation set
        :return: return data encoded by the last layer
        """
        next_train = train_set
        next_valid = validation_set

        for l, layer_obj in enumerate(layer_objs):
            print('Training layer {}...'.format(l + 1))
            next_train, next_valid = self._pretrain_layer_and_gen_feed(
                layer_obj, set_params_func, next_train, next_valid,
                layer_graphs[l])

        return next_train, next_valid

    def _pretrain_layer_and_gen_feed(self, layer_obj, set_params_func,
                                     train_set, validation_set, graph):
        """Pretrain a single autoencoder and encode the data for the next layer.

        :param layer_obj: layer model
        :param set_params_func: function used to set the parameters after
            pretraining
        :param train_set: training set
        :param validation_set: validation set
        :param graph: tf object for the rbm
        :return: encoded train data, encoded validation data
        """
        layer_obj.fit(train_set, train_set,
                      validation_set, validation_set, graph=graph)

        with graph.as_default():
            set_params_func(layer_obj, graph)

            next_train = layer_obj.transform(train_set, graph=graph)
            if validation_set is not None:
                next_valid = layer_obj.transform(validation_set, graph=graph)
            else:
                next_valid = None

        return next_train, next_valid

    def get_layers_output(self, dataset):
        """Get output from each layer of the network.

        :param dataset: input data
        :return: list of np array, element i is the output of layer i
        """
        layers_out = []

        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                for l in self.layer_nodes:
                    layers_out.append(l.eval({self.input_data: dataset,
                                              self.keep_prob: 1}))

        if layers_out == []:
            raise Exception("This method is not implemented for this model")
        else:
            return layers_out

    def _create_last_layer(self, last_layer, n_classes):
        """Create the last layer for finetuning.

        :param last_layer: last layer output node
        :param n_classes: number of classes
        :return: self
        """
        with tf.name_scope("last_layer"):
            self.last_W = tf.Variable(
                tf.truncated_normal(
                    [last_layer.get_shape()[1].value, n_classes], stddev=0.1),
                name='sm-weigths')
            self.last_b = tf.Variable(tf.constant(
                0.1, shape=[n_classes]), name='sm-biases')
            last_out = tf.add(tf.matmul(last_layer, self.last_W), self.last_b)
            self.layer_nodes.append(last_out)
            self.last_out = last_out
            return last_out

    def _create_cost_function_node(self, model_output, ref_input,
                                   regterm=None):
        """Create the cost function node.

        :param model_output: model output node
        :param ref_input: reference input placeholder node
        :param regterm: regularization term
        :return: self
        """
        with tf.name_scope("cost"):
            if self.loss_func == 'cross_entropy':
                clip_inf = tf.clip_by_value(model_output, 1e-10, float('inf'))
                clip_sup = tf.clip_by_value(
                    1 - model_output, 1e-10, float('inf'))

                cost = - tf.reduce_mean(
                    tf.add(
                        tf.mul(ref_input, tf.log(clip_inf)),
                        tf.mul(tf.sub(1.0, ref_input), tf.log(clip_sup))
                    ))

            elif self.loss_func == 'softmax_cross_entropy':
                cost = tf.contrib.losses.softmax_cross_entropy(
                    model_output, ref_input)

            elif self.loss_func == 'mean_squared':
                cost = tf.sqrt(tf.reduce_mean(
                    tf.square(tf.sub(ref_input, model_output))))

            else:
                cost = None

        if cost is not None:
            self.cost = cost + regterm if regterm is not None else cost
            tf.summary.scalar(self.loss_func, self.cost)
        else:
            self.cost = None

    def _create_train_step_node(self):
        """Create the training step node of the network.

        :return: self
        """
        with tf.name_scope("train"):
            if self.opt == 'gradient_descent':
                self.train_step = tf.train.GradientDescentOptimizer(
                    self.learning_rate).minimize(self.cost)

            elif self.opt == 'ada_grad':
                self.train_step = tf.train.AdagradOptimizer(
                    self.learning_rate).minimize(self.cost)

            elif self.opt == 'momentum':
                self.train_step = tf.train.MomentumOptimizer(
                    self.learning_rate, self.momentum).minimize(self.cost)

            elif self.opt == 'adam':
                self.train_step = tf.train.AdamOptimizer(
                    self.learning_rate).minimize(self.cost)

            else:
                self.train_step = None

    def get_model_parameters(self, params, graph=None):
        """Get the parameters of the model.

        :param params: dictionary of keys (str names) and values (tensors).
        :return: evaluated tensors in params
        """
        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                out = {}
                for par in params:
                    if type(params[par]) == list:
                        for i, p in enumerate(params[par]):
                            out[par + '-' + str(i+1)] = p.eval()
                    else:
                        out[par] = params[par].eval()
                return out


class SupervisedModel(Model):
    """Supervised Model scheleton."""

    def __init__(self, model_name, main_dir, models_dir,
                 data_dir, summary_dir):
        """Constructor."""
        Model.__init__(
            self, model_name, main_dir, models_dir, data_dir, summary_dir)

    def fit(self, train_set, train_labels, validation_set=None,
            validation_labels=None, restore_previous_model=False, graph=None):
        """Fit the model to the data.

        :param train_set: Training data. shape(n_samples, n_features)
        :param train_labels: Training labels. shape(n_samples, n_classes)
        :param validation_set: optional, default None. Validation data.
            shape(nval_samples, n_features)
        :param validation_labels: optional, default None. Validation labels.
            shape(nval_samples, n_classes)
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk
                    to continue training.
        :param graph: tensorflow graph object
        :return: self
        """
        if len(train_labels.shape) != 1:
            num_classes = train_labels.shape[1]
        else:
            raise Exception("Please convert the labels with one-hot encoding.")

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            self.build_model(train_set.shape[1], num_classes)
            with tf.Session() as self.tf_session:
                self._initialize_tf_utilities_and_ops(restore_previous_model)
                self._train_model(
                    train_set, train_labels, validation_set, validation_labels)
                self.tf_saver.save(self.tf_session, self.model_path)

    def build_model(self, num_features, num_classes):
        """Build model method."""
        pass

    def _train_model(self, train_set, train_labels,
                     validation_set, validation_labels):
        pass

    def _run_validation_error_and_summaries(self, epoch, feed):
        """Run the summaries and error computation on the validation set.

        :param epoch: current epoch
        :param validation_set: validation data
        :return: self
        """
        try:
            result = self.tf_session.run(
                [self.tf_merged_summaries, self.accuracy], feed_dict=feed)
            summary_str = result[0]
            acc = result[1]
            self.tf_summary_writer.add_summary(summary_str, epoch)
        except tf.errors.InvalidArgumentError:
            if self.tf_summary_writer_available:
                print("Summary writer not available at the moment")
            self.tf_summary_writer_available = False
            acc = self.tf_session.run(self.accuracy, feed_dict=feed)

        if self.verbose == 1:
            print("Accuracy at step %s: %s" % (epoch, acc))

    def predict(self, test_set):
        """Predict the labels for the test set.

        :param test_set: Testing data. shape(n_test_samples, n_features)
        :return: labels
        """
        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                return self.model_predictions.eval({self.input_data: test_set,
                                                    self.keep_prob: 1})

    def compute_accuracy(self, test_set, test_labels):
        """Compute the accuracy over the test set.

        :param test_set: Testing data. shape(n_test_samples, n_features)
        :param test_labels: Labels for the test data.
            shape(n_test_samples, n_classes)
        :return: accuracy
        """
        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                return self.accuracy.eval({self.input_data: test_set,
                                           self.input_labels: test_labels,
                                           self.keep_prob: 1})

    def _create_accuracy_test_node(self):
        """Create the supervised test node of the network.

        :return: self
        """
        with tf.name_scope("test"):
            self.model_predictions = tf.argmax(self.last_out, 1)
            correct_prediction = tf.equal(
                self.model_predictions, tf.argmax(self.input_labels, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, "float"))
            tf.scalar_summary('accuracy', self.accuracy)


class UnsupervisedModel(Model):
    """Unsupervised Model scheleton class."""

    def __init__(self, model_name, main_dir, models_dir,
                 data_dir, summary_dir):
        """Constructor."""
        Model.__init__(
            self, model_name, main_dir, models_dir, data_dir, summary_dir)

        self.encode = None
        self.reconstruction = None

    def fit(self, train_set, train_ref, validation_set=None,
            validation_ref=None, restore_previous_model=False, graph=None):
        """Fit the model to the data.

        :param train_set: Training data. shape(n_samples, n_features)
        :param train_ref: Reference data. shape(n_samples, n_features)
        :param validation_set: optional, default None. Validation data.
            shape(nval_samples, n_features)
        :param validation_ref: optional, default None.
            Reference validation data. shape(nval_samples, n_features)
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk
                    to continue training.
        :param graph: tensorflow graph object
        :return: self
        """
        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            self.build_model(train_set.shape[1])
            with tf.Session() as self.tf_session:
                self._initialize_tf_utilities_and_ops(restore_previous_model)
                self._train_model(
                    train_set, train_ref, validation_set, validation_ref)
                self.tf_saver.save(self.tf_session, self.model_path)

    def build_model(self, num_features):
        """Build model method."""
        pass

    def _train_model(self, train_set, train_labels,
                     validation_set, validation_labels):
        pass

    def _run_validation_error_and_summaries(self, epoch, feed):
        """Run the summaries and error computation on the validation set.

        :param epoch: current epoch
        :param feed: feed dictionary
        :return: self
        """
        try:
            result = self.tf_session.run(
                [self.tf_merged_summaries, self.cost], feed_dict=feed)
            summary_str = result[0]
            err = result[1]
            self.tf_summary_writer.add_summary(summary_str, epoch)
        except tf.errors.InvalidArgumentError:
            if self.tf_summary_writer_available:
                print("Summary writer not available at the moment")
            self.tf_summary_writer_available = False
            err = self.tf_session.run(self.cost, feed_dict=feed)

        if self.verbose == 1:
            print("Reconstruction loss at step %s: %s" % (epoch, err))

    def transform(self, data, graph=None):
        """Transform data according to the model.

        :param data: Data to transform
        :param graph: tf graph object
        :return: transformed data
        """
        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                encoded_data = self.encode.eval(
                    {self.input_data: data, self.keep_prob: 1})
                return encoded_data

    def reconstruct(self, data, graph=None):
        """Reconstruct the test set data using the learned model.

        :param data: Data to reconstruct
        :graph: tf graph object
        :return: labels
        """
        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                return self.reconstruction.eval(
                    {self.input_data: data, self.keep_prob: 1})

    def compute_reconstruction_loss(self, data, data_ref, graph=None):
        """Compute the reconstruction loss over the test set.

        :param data: Data to reconstruct
        :param data_ref: Reference data.
        :return: reconstruction loss
        """
        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                return self.cost.eval(
                    {self.input_data: data,
                     self.input_labels: data_ref, self.keep_prob: 1})
