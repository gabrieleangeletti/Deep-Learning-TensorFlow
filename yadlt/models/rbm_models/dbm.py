"""Deep Boltzmann Machine TensorFlow implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from yadlt.core import UnsupervisedModel
from yadlt.utils import utilities


class DBM(UnsupervisedModel): 
    """Restricted Boltzmann Machine implementation using TensorFlow.

    The interface of the class is sklearn-like.
    """
    def __init__(
        self, num_hidden, visible_unit_type='bin', main_dir='rbm/',
        models_dir='models/', data_dir='data/', summary_dir='logs/',
        model_name='rbm', dataset='mnist', loss_func='mean_squared',
        l2reg=5e-4, regtype='none', gibbs_sampling_steps=1, learning_rate=0.01,
        batch_size=10, num_epochs=10, stddev=0.1, verbose=0):
        """Constructor.

        :param num_hidden: number of hidden units
        :param loss_function: type of loss function
        :param visible_unit_type: type of the visible units (bin or gauss)
        :param gibbs_sampling_steps: optional, default 1
        :param stddev: default 0.1. Ignored if visible_unit_type is not 'gauss'
        :param verbose: level of verbosity. optional, default 0
        """

        