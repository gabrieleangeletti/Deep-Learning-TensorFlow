from sklearn.linear_model import LogisticRegression
from abc import ABCMeta, abstractmethod
from enum import Enum

from multinomial_rbm import MultinomialRBM
from gaussian_rbm import GaussianRBM
from nrelu_rbm import NreluRBM
from rbm import RBM

__author__ = 'blackecho'


class AbstractClsRBM(object):

    """Restricted Boltzmann Machine with a Logistic Regression layer built on top
    of the hidden units for classification.
    """

    __metaclass__ = ABCMeta

    rbm_type = Enum('RBMType', 'rbm grbm mrbm nrelurbm')

    def __init__(self, num_visible, num_hidden, rbm_type, k_visible=None, k_hidden=None, *args, **kwargs):
        """Initialize a layer of Restricted Boltzmann Machine of the given type,
        with a logistic regression layer above it.
        :param num_visible: number of visible units
        :param num_hidden: number of hidden units
        :param k_visible: max values that the visible units can take (the length is n. visible) ONLY MRBM
        :param k_hidden: max values that the hidden units can take (the length is n. hidden) ONLY MRBM
        """
        assert rbm_type is not None
        assert lambda: True if rbm_type == AbstractClsRBM.rbm_type.mrbm and all([k_visible is None, k_hidden is None])\
            else False

        # Restricted Boltzmann Machine
        if rbm_type == AbstractClsRBM.rbm_type.rbm:
            self.rbm = RBM(num_visible, num_hidden)

        elif rbm_type == AbstractClsRBM.rbm_type.grbm:
            self.rbm = GaussianRBM(num_visible, num_hidden)

        elif rbm_type == AbstractClsRBM.rbm_type.mrbm:
            self.rbm = MultinomialRBM(num_visible, num_hidden, k_visible, k_hidden)

        elif rbm_type == AbstractClsRBM.rbm_type.nrelurbm:
            self.rbm = NreluRBM(num_visible, num_hidden)

        # Logistic Regression classifier on top of the rbm
        self.cls = LogisticRegression(*args, **kwargs)

    @abstractmethod
    def learn_unsupervised_features(self,
                                    data,
                                    validation=None,
                                    max_epochs=100,
                                    batch_size=1,
                                    alpha=0.1,
                                    m=0.5,
                                    gibbs_k=1,
                                    alpha_update_rule='constant',
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
        :param alpha_update_rule: type of update rule for the learning rate. Can be constant,
               linear or exponential
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