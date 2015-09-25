from __future__ import print_function

from pyprind import ProgPercent
import numpy as np
import click
import json

from abstract_rbm import AbstractRBM
import utils

__author__ = 'blackecho'


class GaussianRBM(AbstractRBM):

    """Restricted Boltzmann Machine implementation with
    Gaussian visible units and Bernoulli hidden units.
    """

    def __init__(self, num_visible, num_hidden,
                 w=None, h_bias=None, v_bias=None, v_sigma=None):
        """
        :param num_visible: number of visible units
        :param num_hidden: number of hidden units
        :param w: weights matrix
        :param h_bias: hidden units bias
        :param v_bias: visible units bias
        :param v_sigma: standard deviation for the visible units
        """

        super(GaussianRBM, self).__init__(num_visible, num_hidden)

        if w is None and any([h_bias, v_bias, v_sigma]) is True:
            raise Exception('If W is None, then also b and c must be None')

        if w is None:
            # Initialize the weight matrix, using
            # a Gaussian distribution with mean 0 and standard deviation 0.1
            self.W = 0.01 * np.random.randn(self.num_visible, self.num_hidden)
            self.h_bias = np.zeros(self.num_hidden)
            self.v_bias = np.ones(self.num_visible)
            self.sigma = 0.2
            self.v_sigma = np.asmatrix(np.ones(num_visible))
            self.dv_sigma = np.ones_like(self.v_sigma)
        else:
            self.W = w
            self.h_bias = h_bias
            self.v_bias = v_bias
            self.v_sigma = v_sigma
        # debugging values
        self.costs = []
        self.train_free_energies = []
        self.validation_free_energies = []

    def train(self,
              data,
              validation=None,
              epochs=100,
              batch_size=1,
              alpha=[0.1],
              momentum=[0.5],
              gibbs_k=1,
              alpha_update_rule='constant',
              momentum_update_rule='constant',
              verbose=False,
              display=None):
        """Train the restricted boltzmann machine with the given parameters.
        """
        assert display is not None if verbose is True else True
        assert alpha_update_rule in ['constant', 'linear', 'exp']
        assert momentum_update_rule in ['constant', 'linear', 'exp']
        assert len(alpha) > 1 if alpha_update_rule == 'linear' else len(alpha) == 1
        assert len(momentum) > 1 if momentum_update_rule == 'linear' else len(momentum) == 1

        # Initialize total error
        batch_error = 0.

        # divide data into batches
        batches = utils.generate_batches(data, batch_size)

        # prepare parameters update rule
        alpha_rule = utils.prepare_parameter_update(alpha_update_rule, alpha, epochs)
        momentum_rule = utils.prepare_parameter_update(momentum_update_rule, momentum, epochs)

        # last gradient, used for momentum
        last_velocity = 0.

        for epoch in xrange(epochs):
            alpha = alpha_rule.update()  # learning rate update
            m = momentum_rule.update()  # momentum update
            prog_bar = ProgPercent(len(batches))
            for batch in batches:
                prog_bar.update()
                (associations_delta, h_bias_delta, v_values_new,
                 h_probs_new, v_sigma_delta_0, v_sigma_delta) = self.gibbs_sampling(batch, gibbs_k)

                # weights update
                dw = alpha * (associations_delta / float(batch_size)) + m * last_velocity
                self.W += dw
                last_velocity = dw
                # bias updates mean through the batch
                self.h_bias += alpha * np.array(h_bias_delta.mean(axis=0))[0, :]
                cst = 1 / np.square(self.sigma)
                self.v_bias += alpha * ((cst*batch) - (cst*v_values_new)).mean(axis=0)
                # standard deviation update
                # batch_square_norm = np.square(np.linalg.norm(batch))
                # reconstr_square_norm = np.square(np.linalg.norm(v_values_new))
                # self.v_sigma += alpha * \
                #    ((batch_square_norm - v_sigma_delta_0) / (self.v_sigma ** 3)).mean(axis=0) - \
                #    ((reconstr_square_norm - v_sigma_delta) / (self.v_sigma ** 3)).mean(axis=0)
                # self.v_sigma = self.v_sigma.mean(axis=0)

                batch_error += np.sum((batch - v_values_new) ** 2) / float(batch_size)

            if verbose:
                print(display(v_values_new[np.random.randint(v_values_new.shape[0])], threshold=0.5))

            print("Epoch %s : error is %s" % (epoch, batch_error))
            if epoch % 10 == 0:
                self.train_free_energies.append(self.avg_free_energy(batches[0]))
                if validation is not None:
                    self.validation_free_energies.append(self.avg_free_energy(validation))
            self.costs.append(batch_error)
            batch_error = 0

    def gibbs_sampling(self, v_in_0, k):
        """Performs k steps of Gibbs Sampling, starting from the visible units input.
        """

        # Sample from the hidden units given the visible units - Positive
        # Constrastive Divergence phase
        dot_prod = np.dot(v_in_0 / np.square(self.v_sigma), self.W)
        h_activations_0 = dot_prod + self.h_bias
        h_probs_0 = self.hidden_act_func(h_activations_0)
        h_states = utils.probs_to_binary(h_probs_0)
        pos_associations = np.dot(v_in_0.T, h_states)
        # positive delta for the standard deviation
        v_sigma_delta_0 = 0.  # 2 * dot_prod * h_probs_0

        for gibbs_step in xrange(k):
            if gibbs_step > 0:
                # Not first step: sample hidden from new visible
                # Sample from the hidden units given the visible units -
                # Positive CD phase
                h_activations = np.dot(v_in_0 / np.square(self.v_sigma), self.W) + self.h_bias
                h_probs = self.hidden_act_func(h_activations)
                h_states = utils.probs_to_binary(h_probs)

            # Reconstruct the visible units
            # units - Negative Contrastive Divergence phase
            v_activations = np.dot(h_states, self.W.T) + self.v_bias
            v_values = self.visible_act_func(v_activations)

            # Sampling again from the hidden units
            dot_prod_new = np.dot(v_values / np.square(self.v_sigma), self.W)
            h_activations_new = dot_prod_new + self.h_bias
            h_probs_new = self.hidden_act_func(h_activations_new)
            h_states_new = utils.probs_to_binary(h_probs_new)

            # We are again using states but we could have used probabilities
            neg_associations = np.dot(v_values.T, h_states_new)
            # negative delta for the standard deviation
            v_sigma_delta = 0.  # 2 * dot_prod_new * h_probs_new
            # Use the new sampled visible units in the next step
            v_in_0 = v_values
            pos_associations /= self.sigma ** 2
            neg_associations /= self.sigma ** 2
        return (pos_associations - neg_associations,
                h_probs_0 - h_probs_new,
                v_values,
                h_probs_new,
                v_sigma_delta_0,
                v_sigma_delta)

    def sample_visible_from_hidden(self, h_in, gibbs_k=1):
        """Assuming the RBM has been trained, run the network on a set of
        hidden units, to get a sample of the visible units.
        """
        (dummy, dummy, v_probs, dummy, _, _) = self.gibbs_sampling(h_in, gibbs_k)
        visible_states = utils.probs_to_binary(v_probs)
        return v_probs, visible_states

    def sample_hidden_from_visible(self, v_in, gibbs_k=1):
        """Assuming the RBM has been trained, run the network on a set of
        visible units, to get a sample of the visible units.
        """
        # This is exactly like the Positive Contrastive divergence phase
        h_activations = np.dot(v_in, self.W) + self.h_bias
        h_probs = self.hidden_act_func(h_activations)
        h_states = utils.probs_to_binary(h_probs)
        return h_probs, h_states

    def visible_act_func(self, x):
        """Sample from a Gaussian Density Function with mean x
        and standard deviation sigma."""
        return np.random.normal(x, np.square(self.sigma))

    def hidden_act_func(self, x):
        """Logistic function"""
        return utils.logistic(x)

    def avg_free_energy(self, data):
        """Compute the average free energy over a representative sample
        of the training set or the validation set.
        """
        wx_b = np.dot(data, self.W) + self.h_bias
        vbias_term = np.dot(data, self.v_bias)
        hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis=1)
        return (- hidden_term - vbias_term).mean(axis=0)

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
        grbm = GaussianRBM(num_visible, num_hidden, act_func)
        grbm.train(dataset,
                   epochs=epochs,
                   alpha=alpha,
                   m=m,
                   batch_size=batch_size,
                   gibbs_k=gibbs_k,
                   verbose=verbose)
        grbm.save_configuration(out)


if __name__ == '__main__':
    main()
