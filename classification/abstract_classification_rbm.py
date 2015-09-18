from abc import ABCMeta, abstractmethod

__author__ = 'blackecho'


class AbstractClsRBM(object):

    """Restricted Boltzmann Machine with a Logistic Regression layer built on top
    of the hidden units for classification.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def learn_unsupervised_features(self,
                                    data,
                                    validation=None,
                                    max_epochs=100,
                                    batch_size=1,
                                    alpha=0.1,
                                    m=0.5,
                                    gibbs_k=1,
                                    verbose=False,
                                    display=None):
        """Unsupervised learning of the features for the Restricted Boltzmann Machine
        with the given parameters.
        :param data: the training set
        :param validation: the validation set
        :param max_epochs: number of training steps
        :param batch_size: size of each batch
        :param alpha: learning rate
        :param m: momentum parameter
        :param gibbs_k: number of gibbs sampling steps
        :param verbose: if true display a progress bar through the loop
        :param display: function used to display reconstructed samples
                        after gibbs sampling for each epoch.
                        If batch_size is greater than one, one
                        random sample will be displayed.
        """
        pass

    @abstractmethod
    def fit_logistic_cls(self, data, labels):
        """Train the layer of Logistic Regression on top of the hidden units of the rbm.
        :param data: training set
        :param labels: labels for the training set
        """
        pass

    @abstractmethod
    def predict_logistic_cls(self, data):
        """Predict the labels for data using the Logistic Regression layer.
        :param data: test set
        :return: predictions made by the classifier
        """
        pass