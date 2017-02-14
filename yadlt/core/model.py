"""Model scheleton."""

from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from .config import Config


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
        self.train_step = None
        self.cost = None

        # tensorflow objects
        self.tf_graph = tf.Graph()
        self.tf_session = None
        self.tf_saver = None
        self.tf_merged_summaries = None
        self.tf_summary_writer = None

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
