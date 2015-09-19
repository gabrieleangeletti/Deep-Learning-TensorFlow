from __future__ import print_function

from sklearn.linear_model import LogisticRegression
import numpy as np

from rbm import RBM
import utils

__author__ = 'blackecho'


class DBN(object):
    """Deep Belief Network implementation.
    """

    def __init__(self, num_layers, *args, **kwargs):
        """Initialization of the Deep Belief Network.
        :param num_layers: array whose elements are the number of
               units for each layer.
        """
        self.layers = [RBM(num_layers[i], num_layers[i+1]) for i in xrange(len(num_layers)-1)]

        # Logistic Regression classifier on top of the last rbm
        self.cls = LogisticRegression(*args, **kwargs)
        # Last layer rbm for supervised training (initialized in supervised training)
        self.last_rbm = None

        # Training performance metrics
        self.errors = np.array([])

    def unsupervised_pretrain(self,
                              data,
                              validation=None,
                              epochs=100,
                              batch_size=10,
                              alpha=0.01,
                              m=0.5,
                              gibbs_k=1,
                              alpha_update_rule='constant',
                              verbose=False,
                              display=None):
        """Unsupervised greedy layer-wise pre-training of the Deep Belief Net.
        :param data: the training set
        :param validation: the validation set
        :param epochs: number of training steps
        :param batch_size: size of each batch
        :param alpha: learning rate
        :param m: momentum parameter
        :param gibbs_k: number of gibbs sampling steps
        :param alpha_update_rule: type of update rule for the learning rate. Can be constant,
               linear or exponential
        :param verbose: if true display a progress bar through the loop
        :param display: function used to display reconstructed samples
                        after gibbs sampling for each epoch.
                        If batch_size is greater than one, one
                        random sample will be displayed.
        """
        middle_repr = None
        middle_val_repr = None
        for l, rbm in enumerate(self.layers):
            print('########## Training {}* RBM - ( {}, {} ) ##########'.format(l+1, rbm.num_visible, rbm.num_hidden))
            if l == 0:
                rbm.train(data,
                          validation=validation,
                          epochs=epochs,
                          batch_size=batch_size,
                          alpha=alpha,
                          m=m,
                          gibbs_k=gibbs_k,
                          alpha_update_rule=alpha_update_rule,
                          verbose=verbose,
                          display=display)
                # dataset's representation of the first rbm
                # TODO: for now, using the states, will try the probs
                _, middle_repr = rbm.sample_hidden_from_visible(data)
                # validation set representation of the first rbm
                _, middle_val_repr = rbm.sample_hidden_from_visible(validation)
            else:
                # train the next rbm using the representation of the previous rbm as visible layer
                rbm.train(middle_repr,
                          validation=middle_val_repr,
                          epochs=epochs,
                          batch_size=batch_size,
                          alpha=alpha,
                          m=m,
                          gibbs_k=gibbs_k,
                          alpha_update_rule=alpha_update_rule)
                # features representation of the current rbm
                # TODO: for now, using the states, will try the probs
                _, middle_repr = rbm.sample_hidden_from_visible(middle_repr)
                # validation set representation of the first rbm
                _, middle_val_repr = rbm.sample_hidden_from_visible(middle_val_repr)

    def supervised_pretrain(self,
                            num_last_layer,
                            data,
                            y,
                            epochs=100,
                            batch_size=1,
                            alpha=0.01,
                            m=0.5,
                            gibbs_k=1,
                            alpha_update_rule='constant'):
        """The last layer is a rbm trained on the joint distribution of the prev. layer and the labels.
        To be called after unsupervised pretrain of the first layers of the dbn.
        :param num_last_layer: number of hidden units for the last RBM
        :param data: input dataset
        :param y: dataset labels (labels must be integers)
        :param epochs: number of training epochs
        :param batch_size: size of each bach
        :param alpha: learning rate parameter
        :param m: momentum parameter
        :param gibbs_k: number of gibbs sampling steps
        :param alpha_update_rule: update rule for the learning rate
        """
        assert data.shape[0] == y.shape[0]

        # convert integer labels to binary
        bin_y = utils.int2binary_vect(y)
        # representation of the dataset by previous last rbm
        data_repr = self.forward(data)
        # initialize the last layer rbm
        self.last_rbm = RBM(data_repr[0].shape[0] + bin_y[0].shape[0], num_last_layer)
        # merge data_repr with y
        joint_data = []
        for i in range(data.shape[0]):
            joint_data.append(np.hstack([data_repr[i], bin_y[i]]))
        joint_data = np.array(joint_data)
        # now train the last rbm on the joint distribution between data_repr and y
        self.last_rbm.train(joint_data,
                            validation=None,
                            epochs=epochs,
                            batch_size=batch_size,
                            alpha=alpha,
                            m=m,
                            gibbs_k=gibbs_k,
                            alpha_update_rule=alpha_update_rule)

    def supervised_fine_tune(self,
                             data,
                             y,
                             batch_size=1,
                             epochs=100,
                             alpha=0.01,
                             alpha_update_rule='constant'):
        """Fine-tuning of the deep belief net using gradient descent.
        :param data: input dataset
        :param y: dataset labels
        :param batch_size: size of each bach
        :param epochs: number of training epochs
        :param alpha: learning rate parameter
        :param alpha_update_rule: update rule for the learning rate
        """
        assert data.shape[0] == y.shape[0]

        # convert integer labels to binary
        bin_y = utils.int2binary_vect(y)

        # divide data into batches
        batches = utils.generate_batches(data, batch_size)
        # divide labels into batches
        y_batches = utils.generate_batches(y, batch_size)

        # Learning rate update rule
        if alpha_update_rule == 'exp':
            alpha_rule = utils.ExpDecayParameter(alpha)
        elif alpha_update_rule == 'linear':
            alpha_rule = utils.LinearDecayParameter(alpha, epochs)
        elif alpha_update_rule == 'constant':
            alpha_rule = utils.ConstantParameter(alpha)
        else:
            raise Exception('alpha_update_rule must be in ["exp", "constant", "linear"]')
        assert alpha_rule is not None

        total_error = 0.

        for epoch in xrange(epochs):
            alpha = alpha_rule.update()  # learning rate update
            for i, batch in enumerate(batches):
                # do a forward pass to compute the last layer output
                out_layer = self.forward(data)

            print("Epoch {:d} : error is {:f}".format(epoch, total_error))
            self.errors = np.append(self.errors, total_error)
            total_error = 0.

    def generate_fantasy(self, data):
        """Return a representation of the data from the last undirected RBM.
        :return:
        """

    def forward(self, data):
        """Do a forward pass through the deep belief net and return the last
        representation layer.
        :param data: input data to the visible units of the first rbm
        """
        middle_repr = None
        for l, rbm in enumerate(self.layers):
            middle_repr, _ = rbm.sample_hidden_from_visible(data) if l == 0\
                    else rbm.sample_hidden_from_visible(middle_repr)
        return middle_repr

    def backward(self):
        """Do a backward pass through the deep belief net and generate a sample
        according to the model.
        """
        # TODO: for each rbm, sample visible and then pass to the prev layer
        pass

    def fit_cls(self, data, y):
        """Fit classifier for the given supervised dataset.
        :param data: supervised training set for the classification layer.
        """
        out_layer = self.forward(data)
        self.cls.fit(out_layer, y)

    def predict_cls(self, data):
        """Predict the labels for data using the classification layer top of the
        deep belief net.
        :param data: input data to the visible units of the first rbm
        """
        out_layer = self.forward(data)
        return self.cls.predict(out_layer)
