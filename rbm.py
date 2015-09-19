from __future__ import print_function

from pyprind import ProgPercent
import numpy as np
import click
import json
import time

from abstract_rbm import AbstractRBM
import utils

__author__ = 'blackecho'


class RBM(AbstractRBM):

    """Restricted Boltzmann Machine implementation with
    visible and hidden Bernoulli units.
    """

    def __init__(self,
                 num_visible,
                 num_hidden,
                 w=None,
                 h_bias=None,
                 v_bias=None):
        """
        :param num_visible: number of visible units
        :param num_hidden: number of hidden units
        :param w: weights matrix
        :param h_bias: hidden units bias
        :param v_bias: visible units bias
        """

        super(RBM, self).__init__(num_visible, num_hidden)

        if w is None and any([h_bias, v_bias]) is True:
            raise Exception('If W is None, then also b and c must be None')

        if w is None:
            # Initialize the weight matrix, using
            # a Gaussian distribution with mean 0 and standard deviation 0.1
            self.W = 0.01 * np.random.randn(self.num_visible, self.num_hidden)
            self.h_bias = np.zeros(self.num_hidden)
            self.v_bias = np.ones(self.num_visible)
        else:
            self.W = w
            self.h_bias = h_bias
            self.v_bias = v_bias
        # debugging values
        self.costs = []
        self.train_free_energies = []
        self.validation_free_energies = []
        # last gradient, used for momentum
        self.last_velocity = 0.

    def train(self,
              data,
              validation=None,
              epochs=100,
              batch_size=10,
              alpha=0.1,
              m=0.5,
              gibbs_k=1,
              alpha_update_rule='constant',
              verbose=False,
              display=None):
        """Train the restricted boltzmann machine with the given parameters.
        """
        assert display is not None if verbose is True else True
        assert alpha_update_rule in ['constant', 'linear', 'exp']

        # Total error per epoch
        total_error = 0.

        # Set the visible bias weights to the approximate probability
        # of a neuron being activated in the training data.
        # This is the logarithm of the 1 / (1 - neuron_mean), where neuron_mean
        # is the proportion of training vectors in which unit i is
        neuron_mean = data.mean(axis=0)
        for i in xrange(neuron_mean.shape[0]):
            if neuron_mean[i] == 1:
                neuron_mean[i] = 0.99999
        neuron_mean = 1 / (1 - neuron_mean)
        self.v_bias = np.log(neuron_mean)

        # divide data into batches
        batches = utils.generate_batches(data, batch_size)
        n_batches = len(batches)

        alpha_rule = AbstractRBM._prepare_alpha_update(alpha_update_rule, alpha, epochs)

        # Momentum parameter update rule
        m_update = int(epochs / ((0.9 - m) / 0.01)) + 1

        start = time.clock()
        for epoch in xrange(epochs):
            alpha = alpha_rule.update()  # learning rate update
            prog_bar = ProgPercent(n_batches)
            for batch in batches:
                prog_bar.update()
                (associations_delta, h_bias_delta, v_probs, v_states, h_probs) = self.gibbs_sampling(batch, gibbs_k)

                # weights update
                dw = alpha * (associations_delta / float(batch_size)) + m*self.last_velocity
                self.W += dw
                self.last_velocity = dw
                # bias updates mean through the batch
                self.h_bias += alpha * h_bias_delta.mean(axis=0)
                self.v_bias += alpha * (batch - v_probs).mean(axis=0)

                error = np.sum((batch - v_probs) ** 2) / float(batch_size)
                total_error += error

            if verbose:
                print(display(v_states[np.random.randint(v_states.shape[0])]))

            print("Epoch {:d} : error is {:f}".format(epoch, total_error))
            if epoch % 10 == 0:
                self.train_free_energies.append(self.avg_free_energy(batches[0]))
                if validation is not None:
                    self.validation_free_energies.append(self.avg_free_energy(validation))
            if epoch % m_update == 0 and epoch > 0:
                m += 0.01
            self.costs.append(total_error)
            total_error = 0.
        end = time.clock()
        print("Training took {:f}s time".format(end - start))

    def gibbs_sampling(self, v_in_0, k):
        """Performs k steps of Gibbs Sampling, starting from the visible units input.
        """
        batch_size = v_in_0.shape[0]

        # Sample from the hidden units given the visible units - Positive
        # Constrastive Divergence phase
        h_activations_0 = np.dot(v_in_0, self.W) + self.h_bias
        h_probs_0 = self.hidden_act_func(h_activations_0)
        h_states = (h_probs_0 > np.random.rand(batch_size, self.num_hidden)).astype(np.int)
        pos_associations = np.dot(v_in_0.T, h_states)

        for gibbs_step in xrange(k):
            if gibbs_step > 0:
                # Not first step: sample hidden from new visible
                # Sample from the hidden units given the visible units -
                # Positive CD phase
                h_activations = np.dot(v_in_0, self.W) + self.h_bias
                h_probs = self.hidden_act_func(h_activations)
                h_states = (h_probs > np.random.rand(batch_size, self.num_hidden)).astype(np.int)

            # Reconstruct the visible units
            # Negative Contrastive Divergence phase
            v_activations = np.dot(h_states, self.W.T) + self.v_bias
            v_probs = self.visible_act_func(v_activations)
            # useful to compute the error
            v_states = (v_probs > np.random.rand(batch_size, self.num_visible)).astype(np.int)
            # Sampling again from the hidden units
            h_activations_new = np.dot(v_probs, self.W) + self.h_bias
            h_probs_new = self.hidden_act_func(h_activations_new)
            h_states_new = (h_probs_new > np.random.rand(batch_size, self.num_hidden)).astype(np.int)
            # Use the new sampled visible units in the next step
            v_in_0 = v_states
        # We are again using states but we could have used probabilities
        neg_associations = np.dot(v_probs.T, h_states_new)
        return pos_associations - neg_associations, h_probs_0 - h_probs_new, v_probs, v_states, h_probs_new

    def sample_visible_from_hidden(self, h_in, gibbs_k=1):
        """Assuming the RBM has been trained, run the network on a set of
        hidden units, to get a sample of the visible units.
        """
        (_, _, v_probs, _) = self.gibbs_sampling(h_in, gibbs_k)
        v_states = (v_probs > np.random.rand(v_probs.shape[0], v_probs.shape[1])).astype(np.int)
        return v_probs, v_states

    def sample_hidden_from_visible(self, v_in, gibbs_k=1):
        """Assuming the RBM has been trained, run the network on a set of
        visible units, to get a sample of the visible units.
        """
        (_, _, _, h_probs) = self.gibbs_sampling(v_in, gibbs_k)
        h_states = (h_probs > np.random.rand(h_probs.shape[0], h_probs.shape[1])).astype(np.int)
        return h_probs, h_states

    def visible_act_func(self, x):
        """Sigmoid function"""
        return 1. / (1. + np.exp(-x))

    def hidden_act_func(self, x):
        """Sigmoid function"""
        return 1. / (1. + np.exp(-x))

    def fantasy(self, k=1):
        """Generate a sample from the RBM after n steps of gibbs sampling, starting with a
        random sample.
        :param k: number of gibbs sampling steps
        :return: what's in the mind of the RBM
        """
        # random initial visible states
        v_in = (0.5 > np.random.random(self.num_visible)).astype(np.int)
        (_, _, v_probs, v_states, _) = self.gibbs_sampling(v_in, k)
        return v_probs, v_states

    def avg_free_energy(self, data):
        """Compute the average free energy over a representative sample
        of the training set or the validation set.
        """
        wx_b = np.dot(data, self.W) + self.h_bias
        vbias_term = np.dot(data, self.v_bias)
        hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis=1)
        return (- hidden_term - vbias_term).mean(axis=0)

    def save_weights_images(self, hs, width, height, outfile):
        """Save images of the learned weights by the RBM.
        :param hs: how much hidden unit weights to save
        :param width: width of the img
        :param height: height of the img
        :param outfile: output file, must be png
        """
        # select random hidden units
        randperm = np.random.permutation(self.num_hidden)
        randperm = randperm[:hs]
        for h in randperm:
            rand_h_w = np.array([i[h] for i in self.W])
            outfile_h = outfile[:-4] + '-' + str(h) + outfile[-4:]
            utils.gen_image(rand_h_w, width, height, outfile_h)

    def save_configuration(self, outfile):
        """Save a json representation of the RBM object.
        """
        with open(outfile, 'w') as f:
            f.write(json.dumps({'W': self.W.tolist(),
                                'h_bias': self.h_bias.tolist(),
                                'v_bias': self.v_bias.tolist(),
                                'num_hidden': self.num_hidden,
                                'num_visible': self.num_visible,
                                'costs': self.costs,
                                'train_free_energies':
                                    self.train_free_energies,
                                'validation_free_energies':
                                    self.validation_free_energies}))

    def load_configuration(self, infile):
        """Load a json representation of the RBM object.
        """
        with open(infile, 'r') as f:
            data = json.load(f)
            self.W = np.array(data['W'])
            self.h_bias = np.array(data['h_bias'])
            self.v_bias = np.array(data['v_bias'])
            self.num_hidden = data['num_hidden']
            self.num_visible = data['num_visible']
            self.costs = data['costs']
            self.train_free_energies = data['train_free_energies']
            self.validation_free_energies = data['validation_free_energies']


@click.command()
@click.option('--config', default='', help='json with the config of the rbm')
def main(config):
    with open(config, 'r') as f:
        data = json.load(f)
        num_visible = data['num_visible']
        num_hidden = data['num_hidden']
        act_func = data['act_func']
        dataset = np.array(data['dataset'])
        epochs = data['epochs']
        alpha = data['alpha']
        m = data['m']
        batch_size = data['batch_size']
        gibbs_k = data['gibbs_k']
        verbose = data['verbose']
        out = data['outfile']
        # create rbm object
        rbm = RBM(num_visible, num_hidden, act_func)
        rbm.train(dataset,
                  epochs=epochs,
                  alpha=alpha,
                  m=m,
                  batch_size=batch_size,
                  gibbs_k=gibbs_k,
                  verbose=verbose)
        rbm.save_configuration(out)


if __name__ == '__main__':
    main()
