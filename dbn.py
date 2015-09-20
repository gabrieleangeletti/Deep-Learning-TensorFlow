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

    def wake_sleep(self,
                   num_last_layer,
                   data,
                   y,
                   batch_size=1,
                   epochs=100,
                   alpha=0.01,
                   top_gibbs_k=1,
                   alpha_update_rule='constant'):
        """Fine-tuning of the deep belief net using the wake-sleep algorithm proposed by Hinton et al. 2006.
        :param num_last_layer: number of hidden units for the last RBM
        :param data: input dataset
        :param y: dataset labels
        :param batch_size: size of each bach
        :param epochs: number of training epochs
        :param alpha: learning rate parameter
        :param top_gibbs_k: number of gibbs sampling steps using the top level undirected associative
                            memory
        :param alpha_update_rule: update rule for the learning rate
        """
        assert data.shape[0] == y.shape[0]

        # convert integer labels to binary
        bin_y = utils.int2binary_vect(y)

        # divide data into batches
        batches = utils.generate_batches(data, batch_size)
        # divide labels into batches
        y_batches = utils.generate_batches(y, batch_size)

        # initialize the last layer rbm
        self.last_rbm = RBM(self.layers[-1] + bin_y[0].shape[0], num_last_layer)

        alpha_rule = utils.prepare_alpha_update(alpha_update_rule, alpha, epochs)

        total_error = 0.

        for epoch in xrange(epochs):
            alpha = alpha_rule.update()  # learning rate update
            for i, batch in enumerate(batches):
                # ========== WAKE/POSITIVE PHASE ==========
                middle_states = None  # states of middle layers
                wake_hid_states = None  # hidden states of the first layer, will be useful in the sleep phase
                for l, rbm in enumerate(self.layers):
                    if l == 0:
                        # Compute probs/states for all the layers
                        h_probs = rbm.hidden_act_func(np.dot(batch, rbm.W) + rbm.h_bias)
                        h_states = (h_probs > np.random.rand(h_probs.shape[0], h_probs.shape[1])).astype(np.int)
                        wake_hid_states = h_states
                    else:
                        # Compute probs/states for all the layers
                        h_probs = rbm.hidden_act_func(np.dot(middle_states, rbm.W) + rbm.h_bias)
                        h_states = (h_probs > np.random.rand(h_probs.shape[0], h_probs.shape[1])).astype(np.int)
                    middle_states = h_states

                # merge data_repr with y
                joint_data = []
                for j in range(middle_states.shape[0]):
                    joint_data.append(np.hstack([middle_states[j], bin_y[j]]))
                joint_data = np.array(joint_data)

                # Compute probs/states for the last rbm
                wake_probs = self.last_rbm.hidden_act_func(np.dot(joint_data, self.last_rbm.W) + self.last_rbm.h_bias)
                wake_states = (wake_probs > np.random.rand(wake_probs.shape[0], wake_probs.shape[1])).astype(np.int)
                # Positive statistics for the wake phase
                poslabtopstatistics = np.dot(bin_y.T, wake_states)
                pospentopstatistic = np.dot(middle_states.T, wake_states)
                # Perform gibbs sampling using the top level undirected associative memory
                neg_top_states = wake_states  # initialization
                for k in range(top_gibbs_k):

                    neg_pen_probs = self.last_rbm.visible_act_func(np.dot(neg_top_states, middle_states.T) +
                                                                   self.last_rbm.v_bias)
                    neg_pen_states = (neg_pen_probs >
                                      np.random.rand(neg_pen_probs.shape[0], neg_pen_probs.shape[1])).astype(np.int)

                    neg_lab_probs = utils.softmax(np.dot(neg_top_states, bin_y.T) + self.last_rbm.v_bias[self.last_layer[-1:]])

                    # merge data_repr with y
                    joint_data = []
                    for j in range(neg_pen_states.shape[0]):
                        joint_data.append(np.hstack([neg_pen_states[j], neg_lab_probs[j]]))
                    joint_data = np.array(joint_data)

                    neg_top_probs = self.last_rbm.visible_act_func(np.dot(joint_data, self.last_rbm.W) +
                                                                   self.last_rbm.h_bias)
                    neg_top_states = (neg_top_probs >
                                      np.random.rand(neg_top_probs.shape[0], neg_top_probs.shape[1])).astype(np.int)

                # ========== SLEEP/NEGATIVE PHASE ==========
                neg_pen_top_statistics = np.dot(neg_pen_states.T, neg_top_states)
                neg_lab_top_statistics = np.dot(neg_lab_probs.T, neg_top_states)

                # Starting from the end of the gibbs sampling run, perform a top-down generative
                # pass to get sleep/negative phase probabilities and sample states
                sleep_pen_states = neg_pen_states
                sleep_middle_states = None  # states of middle layers
                sleep_vis_probs = None  # probabilities of the first layer (raw input)
                for l, rbm in reversed(list(enumerate(self.layers))):
                    if l == len(self.layers)-1:
                        # Compute probs/states for all the layers
                        sleep_h_probs = rbm.hidden_act_func(np.dot(sleep_pen_states, rbm.W) + rbm.v_bias)
                        sleep_h_states = (sleep_h_probs > np.random.rand(sleep_h_probs.shape[0], sleep_h_probs.shape[1])).astype(np.int)
                    else:
                        # Compute probs/states for all the layers
                        sleep_h_probs = rbm.hidden_act_func(np.dot(sleep_middle_states, rbm.W) + rbm.v_bias)
                        sleep_h_states = (sleep_h_probs > np.random.rand(sleep_h_probs.shape[0], sleep_h_probs.shape[1])).astype(np.int)
                        if l == 0:
                            sleep_vis_probs = sleep_h_probs
                    sleep_middle_states = sleep_h_states

                # Predictions
                # psleeppenstates = logistic(sleephidstates*hidpen + penrecbiases);
                # psleephidstates = logistic(sleepvisprobs*vishid + hidrecbiases);
                # pvisprobs = self.layers[0].visible_act_func(np.dot(wake_hid_states, self.layers[0].W) + self.layers[0].v_bias)
                # phidprobs = logistic(wakepenstates*penhid + hidgenbiases);

                # ========== Updates to generative parameters ==========
                # ========== Updates to top level associative memory parameters ==========
                # ========== Updates to inference approximation parameters ==========

            print("Epoch {:d} : error is {:f}".format(epoch, total_error))
            self.errors = np.append(self.errors, total_error)
            total_error = 0.

    def fantasy(self, k=1):
        """Generate a sample from the DBN after n steps of gibbs sampling, starting with a
        random sample.
        :param k: number of gibbs sampling steps
        :return: what's in the mind of the DBN
        """
        pass

    def forward(self, data):
        """Do a forward pass through the deep belief net and return the last
        representation layer.
        :param data: input data to the visible units of the first rbm
        :return middle_repr: last representation layer
        """
        middle_repr = None
        for l, rbm in enumerate(self.layers):
            middle_repr, _ = rbm.sample_hidden_from_visible(data) if l == 0\
                    else rbm.sample_hidden_from_visible(middle_repr)
        return middle_repr

    def backward(self, data):
        """Do a backward pass through the deep belief net and generate a sample
        according to the model.
        :param data: input data to the hidden units of the last rbm
        :return middle_repr: first representation layer
        """
        middle_repr = None
        for l, rbm in reversed(list(enumerate(self.layers))):
            middle_repr, _ = rbm.sample_visible_from_hidden(data) if l == len(self.layers)-1\
                else rbm.sample_visible_from_hidden(middle_repr)
        return middle_repr

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
