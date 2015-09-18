from sklearn.linear_model import LogisticRegression

from rbm import RBM

__author__ = 'blackecho'


class DBN(object):
    """Deep Belief Network implementation.
    ========================= TODO ============================
    """

    def __init__(self, num_layers, *args, **kwargs):
        """Initialization of the Deep Belief Network.
        :param num_layers: array whose elements are the number of
               units for each layer.
        """
        self.layers = [RBM(num_layers[i], num_layers[i+1]) for i in xrange(len(num_layers)-1)]

        # Logistic Regression classifier on top of the rbm
        self.cls = LogisticRegression(*args, **kwargs)

    def pretrain(self, data, validation=None,
                 max_epochs=100, batch_size=10,
                 alpha=0.01, m=0.5, gibbs_k=1, alpha_update_rule='constant',
                 verbose=False, display=None):
        """Unsupervised greedy layer-wise pre-training of the Deep Belief Net.
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
        middle_repr = None
        middle_val_repr = None
        for l, rbm in enumerate(self.layers):
            print('########## Training {}* RBM - ( {}, {} ) ##########'.format(l+1, rbm.num_visible, rbm.num_hidden))
            if l == 0:
                print data.shape
                rbm.train(data,
                          validation=validation,
                          max_epochs=max_epochs,
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
                print middle_repr.shape
                rbm.train(middle_repr,
                          validation=middle_val_repr,
                          max_epochs=max_epochs,
                          batch_size=batch_size,
                          alpha=alpha,
                          m=m,
                          gibbs_k=gibbs_k,
                          alpha_update_rule=alpha_update_rule,
                          verbose=verbose,
                          display=display)
                # features representation of the current rbm
                # TODO: for now, using the states, will try the probs
                _, middle_repr = rbm.sample_hidden_from_visible(middle_repr)
                # validation set representation of the first rbm
                _, middle_val_repr = rbm.sample_hidden_from_visible(middle_val_repr)

    def finetune(self):
        """
        """
        pass

    def fit_cls(self, data):
        """Fit classifier for the given supervised dataset.
        :param data: supervised training set for the classification layer.
        """
        # TODO: forward pass through the dbn, and fit logistic on the last layer
        pass

    def forward(self):
        """
        """
        # TODO: for each rbm, sample hidden and then pass to the next layer
        pass

    def backward(self):
        """
        """
        # TODO: for each rbm, sample visible and then pass to the prev layer
        pass
