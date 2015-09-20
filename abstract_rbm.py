from abc import ABCMeta, abstractmethod
import utils

__author__ = 'blackecho'


class AbstractRBM(object):

    """Restricted Boltzmann Machine abstract representation.
    """

    __metaclass__ = ABCMeta

    def __init__(self, num_visible, num_hidden):
        """Initialize a RBM with the given number of visible and hidden units.
        :param num_visible: number of visible units
        :param num_hidden: number of hidden units
        """
        self.num_visible = num_visible
        self.num_hidden = num_hidden

    @abstractmethod
    def train(self, data, validation=None, epochs=100, batch_size=10,
              alpha=0.1, m=0.5, gibbs_k=1, alpha_update_rule='constant', verbose=False, display=None):
        """Train the restricted boltzmann machine with the given parameters.
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
        pass

    @abstractmethod
    def gibbs_sampling(self, v_in_0, k):
        """Performs Markov Chain Gibbs Sampling, starting from the visible units input.
        :param v_in_0: input of the visible units
        :param k: number of sampling steps
        :return difference between positive associations and negative
        associations after k steps of gibbs sampling
        """
        pass

    @abstractmethod
    def sample_visible_from_hidden(self, h_in, gibbs_k):
        """
        Assuming the RBM has been trained, run the network on a set of
        hidden units, to get a sample of the visible units.
        :param h_in: states of the hidden units.
        :param gibbs_k: number of gibbs sampling steps
        :return (visible units probabilities, visible units states)
        """
        pass

    @abstractmethod
    def sample_hidden_from_visible(self, v_in, gibbs_k):
        """
        Assuming the RBM has been trained, run the network on a set of
        visible units, to get a sample of the hidden units.
        :param v_in: states of the visible units.
        :param gibbs_k: number of gibbs sampling steps
        :return (hidden units probabilities, hidden units states)
        """
        pass

    @abstractmethod
    def visible_act_func(self, x):
        """Activation function for the visible units"""
        pass

    @abstractmethod
    def hidden_act_func(self, x):
        """Activation function for the hidden units"""
        pass

    @abstractmethod
    def avg_free_energy(self, data):
        """Compute the average free energy over a representative sample
        of the training set or the validation set.
        """
        pass

    @abstractmethod
    def save_configuration(self, outfile):
        """Save a json representation of the RBM object, for later use.
        :param outfile: path of the output file
        """
        pass

    @abstractmethod
    def load_configuration(self, infile):
        """Load a json representation of the RBM object previously trained.
        :param infile: path of the input file
        """
        pass
