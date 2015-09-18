import numpy as np

from rbm import RBM


class DBN(object):
    """Deep Belief Network implementation.
    ========================= TODO ============================
    """

    def __init__(self, num_layers):
        """Initialization of the Deep Belief Network.
        : param num_layers: array whose elements are the number of
                units for each layer.
        """
        self.layers = [RBM(num_layers[i], num_layers[i+1]) for i in xrange(len(num_layers)-1)]

    def pretrain(self, data,
                 max_epochs=100, batch_size=10,
                 alpha=0.01, m=0.5, gibbs_k=1,
                 verbose=False, display=None):
        """Unsupervised, layer-wise training of the Deep Belief Net.
        :param data: the training set
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
        for l, rbm in enumerate(self.layers):
            if l == 0:
                l.train(data, max_epochs, batch_size, alpha, m, gibbs_k, verbose, display)
            else:
                prev_rbm = self.layers[l-1]
                
        pass

    def finetune(self):
        """
        """
        pass

    def forward(self):
        """
        """
        pass

    def backward(self):
        """
        """
        pass
