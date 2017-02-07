"""Model classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import six
import tensorflow as tf

from .config import Config
from .layers import BaseLayer
from yadlt.utils import tfutils


@six.add_metaclass(abc.ABCMeta)
class BaseModel(object):
    """Base model interface.

    This is the main interface the all models
    should implement, both dl models and non-dl models.
    """

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


@six.add_metaclass(abc.ABCMeta)
class NNContainer(BaseModel):
    """Base container interface.

    This is the interface all dl models should implement.
    """

    @abc.abstractmethod
    def add(self, layer):
        """Add the given `layer` to the container."""
        pass

    @abc.abstractmethod
    def get(self, index):
        """Return the layer at the given `index`."""
        pass

    @abc.abstractmethod
    def size(self):
        """Return the number of `layers`."""
        pass


class Sequential(NNContainer):
    """NN model composed by a linear stack of layers."""

    def __init__(self, layers=None, placeholders=None, name="model"):
        """Create a Sequential model."""
        if layers is not None:
            self.layers = layers
            self.placeholders = placeholders
            self.size = len(layers)
        else:
            self.layers = []
            self.placeholders = {}
            self.size = 0
        self.name = name

        # tensorflow objects
        self.tf_graph = tf.Graph()
        self.tf_session = None
        self.tf_saver = None
        self.tf_merged_summaries = None
        self.tf_summary_writer = None

    def add_placeholder(self, pkey, pobj):
        """Add an input placeholder."""
        self.placeholders[pkey] = pobj

    def add(self, layer):
        """Add a layer to the stack."""
        if not isinstance(layer, BaseLayer):
            raise TypeError('Should be an instance of BaseLayer.')
        self.layers.append(layer)
        self.size += 1
        return self

    def insert(self, layer, index=None):
        """Insert `layer` at the given index.

        If index is None this method is equivalent to add.
        """
        if not isinstance(layer, BaseLayer):
            raise TypeError('Should be an instance of BaseLayer.')
        if index is not None:
            self.layers.insert(layer, index)
        else:
            self.layers.append(layer)
        self.size += 1

    def get(self, index):
        """Return the layer at the given index."""
        return self.layers[index]

    def size(self):
        """Return the number of layers of this model."""
        return self.size

    def remove(self, index=None):
        """Remove the layer at the given `index`."""
        if index is not None:
            del self.layers[index]
        else:
            del self.layers[-1]
        self.size -= 1

    def forward(self, pkey):
        """Forward propagate through the model.

        Parameters
        ----------

        ph_index : str
            Input placeholder key in `self.placeholders`.
        """
        out = self.placeholders[pkey]
        for l in self.layers:
            out = l.forward(out)
        return out

    def fit(self, train_set, train_ref=None, val_set=None, val_ref=None):
        """Train the model.

        Parameters
        ----------
        train_set : array_like, shape (num_samples, num_features)
            Training data.

        train_ref : array_like, shape (num_samples, num_features), default None
            Reference training data.

        val_set : array_like, shape (val_samples, num_features) default None
            Validation set.

        val_ref : array_like, shape (val_samples, num_features) default None
            Reference validation data.

        Returns
        -------
        self : trained model instance
        """
        with self.tf_graph.as_default():
            # Build the model
            self.build_model(train_set.shape[1])
            with tf.Session() as self.tf_session:
                # Tensorflow initialization
                summary_objs = tfutils.init_tf_ops(self.tf_session)
                self.tf_merged_summaries, self.tf_summary_writer = summary_objs
                # Train the model
                for i in range(self.train_params["num_epochs"]):
                    self._run_train_step(train_set)
                    # if val_set is not None:
                    #     feed = {self.placeholders["input_orig"]: val_set,
                    #             self.placeholders["input_corr"]: val_set}
                    #     self._run_validation_error_and_summaries(i, feed)

    def get_parameters(self):
        """Get model parameters."""
        pass

    def save(self, path):
        """Save the model to disk."""
        pass

    def load(self, path):
        """Load model from disk."""
        pass


class Model(object):
    """Class representing an abstract Model."""

    def __init__(self, name):
        """Constructor.

        :param name: name of the model, used as filename.
            string, default 'dae'
        """
        self.name = name
        self.model_path = os.path.join(Config().models_dir, self.name)

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

    def get_parameters(self, params, graph=None):
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

    def __init__(self, name):
        """Constructor."""
        Model.__init__(self, name)

    def fit(self, train_set, train_labels, validation_set=None,
            validation_labels=None, graph=None):
        """Fit the model to the data.

        :param train_set: Training data. shape(n_samples, n_features)
        :param train_labels: Training labels. shape(n_samples, n_classes)
        :param validation_set: optional, default None. Validation data.
            shape(nval_samples, n_features)
        :param validation_labels: optional, default None. Validation labels.
            shape(nval_samples, n_classes)
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
                self.tf_merged_summaries, self.tf_summary_writer = tfutils.init_tf_ops(self.tf_session)
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

    def __init__(self, name):
        """Constructor."""
        Model.__init__(self, name)

        self.encode = None
        self.reconstruction = None

    def fit(self, train_set, train_ref, validation_set=None,
            validation_ref=None, graph=None):
        """Fit the model to the data.

        :param train_set: Training data. shape(n_samples, n_features)
        :param train_ref: Reference data. shape(n_samples, n_features)
        :param validation_set: optional, default None. Validation data.
            shape(nval_samples, n_features)
        :param validation_ref: optional, default None.
            Reference validation data. shape(nval_samples, n_features)
        :param graph: tensorflow graph object
        :return: self
        """
        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            self.build_model(train_set.shape[1])
            with tf.Session() as self.tf_session:
                self.tf_merged_summaries, self.tf_summary_writer = tfutils.init_tf_ops(self.tf_session)
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
