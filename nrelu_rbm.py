__author__ = 'blackecho'

from pyprind import ProgPercent
import numpy as np
import json

from abstract_rbm import AbstractRBM
import utils


class NreluRBM(AbstractRBM):

    """Restricted Boltzmann Machine implementation with
    Noisy Rectified Linear units.
    """

    def __init__(self):
        pass

    def train(self, data, validation=None, max_epochs=100, batch_size=10,
              alpha=0.1, m=0.5, gibbs_k=1, verbose=False, display=None):
        """Train the rbm with the given parameters.
        """
        pass

    def gibbs_sampling(self, v_in_0, k):
        """Performs k steps of Gibbs Sampling, starting from the visible units input.
        """
        pass

    def sample_visible_from_hidden(self, h_in, gibbs_k):
        """Assuming the NreluRBM has been trained, run the network on a set of
        hidden units, to get a sample of the visible units.
        """
        pass

    def sample_hidden_from_visible(self, v_in, gibbs_k):
        """Assuming the NreluRBM has been trained, run the network on a set of
        visible units, to get a sample of the hidden units.
        """
        pass

    def visible_act_func(self, x):
        pass

    def hidden_act_func(self, x):
        pass

    def average_free_energy(self, data):
        """Compute the average free energy over a representative sample
        of the training set or the validation set.
        """
        pass

    def save_configuration(self, outfile):
        """Save a json representation of the NreluRBM object.
        """
        pass

    def load_configuration(self, infile):
        """Load a json representation of the NreluRBM object.
        """
        pass