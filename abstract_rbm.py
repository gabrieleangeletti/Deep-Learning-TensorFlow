from abc import ABCMeta, abstractmethod


class AbstractRBM(object):

    """Restricted Boltzmann Machine abstract representation.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self):
        """Train the restricted boltzmann machine with the given parameters.
        """
        pass

    @abstractmethod
    def gibbs_sampling(self):
        """Performs Markov Chain Gibbs Sampling, starting from the visible units input.
        """
        pass

    @abstractmethod
    def sample_visible_from_hidden(self):
        """
        Assuming the RBM has been trained, run the network on a set of
        hidden units, to get a sample of the visible units.
        """
        pass

    @abstractmethod
    def sample_hidden_from_visible(self):
        """
        Assuming the RBM has been trained, run the network on a set of
        visible units, to get a sample of the hidden units.
        """
        pass

    @abstractmethod
    def visible_act_func(self):
        """Activation function for the visible units"""
        pass

    @abstractmethod
    def hidden_act_func(self):
        """Activation function for the hidden units"""
        pass

    @abstractmethod
    def average_free_energy(self):
        """Compute the average free energy over a representative sample
        of the training set or the validation set.
        """
        pass

    @abstractmethod
    def save_configuration(self):
        """Save a json representation of the RBM object, for later use.
        """
        pass

    @abstractmethod
    def load_configuration(self):
        """Load a json representation of the RBM object previously trained.
        """
        pass
