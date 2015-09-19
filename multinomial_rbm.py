from __future__ import print_function

import numpy as np
import json

import utils

from pyprind import ProgPercent

from abstract_rbm import AbstractRBM

__author__ = 'blackecho'


class MultinomialRBM(AbstractRBM):

    """Restricted Boltzmann Machine implementation with
    visible and hidden Multinomial units, that can assume K different discrete values.
    """

    def __init__(self, num_visible, num_hidden, k_visible, k_hidden,
                 w=None, h_bias=None, v_bias=None):
        """
        :param k_visible: max values that the visible units can take (the length is n. visible)
        :param k_hidden: max values that the hidden units can take (the length is n. hidden)
        :param w: weights matrix
        :param h_bias: hidden units bias
        :param v_bias: visible units bias
        """

        super(MultinomialRBM, self).__init__(num_visible, num_hidden)

        self.num_visible *= (k_visible + 1)
        self.num_hidden *= (k_hidden + 1)
        self.k_visible = k_visible
        self.k_hidden = k_hidden

        if w is None and any([h_bias, v_bias]) is None:
            raise Exception('If W is None, then also b and c must be None')

        if w is None:
            # Initialize the weight matrix, using
            # a Gaussian ddistribution with mean 0 and standard deviation 0.1
            self.W = 0.01 * np.random.randn(self.num_visible, self.num_hidden)
            self.h_bias = np.zeros(self.num_hidden)
            self.v_bias = np.zeros(self.num_visible)
        else:
            self.W = w
            self.h_bias = h_bias
            self.v_bias = v_bias
        # debugging values
        self.costs = []
        self.train_free_energies = []
        self.validation_free_energies = []
        # last gradient, used for momentum
        self.last_velocity = 0.0

    def train(self, data, validation=None, epochs=100, batch_size=10,
              alpha=0.1, m=0.5, gibbs_k=1, alpha_update_rule='constant', verbose=False, display=None):
        # Total error per epoch
        total_error = 0

        # divide data into batches
        batches = utils.generate_batches(data, batch_size)
        n_batches = len(batches)

        alpha_rule = AbstractRBM._prepare_alpha_update(alpha_update_rule, alpha, epochs)

        # Momentum parameter update rule
        m_update = int(epochs / ((0.9 - m) / 0.01)) + 1
        # Convert batches into binary visible states
        binary_batches = [self._convert_visible_state_to_binary(b) for b in batches]
        # Convert the validation into binary visible states
        binary_validation = None
        if validation is not None:
            binary_validation = self._convert_visible_state_to_binary(validation)

        for epoch in xrange(epochs):
            alpha = alpha_rule.update()  # learning rate update
            prog_bar = ProgPercent(n_batches)
            for batch in binary_batches:
                prog_bar.update()
                (associations_delta, h_bias_delta, v_probs,
                 h_probs) = self.gibbs_sampling(batch, gibbs_k)
                # Useful to compute the error
                v_states = (v_probs > np.random.rand(
                    batch_size, self.num_visible)).astype(np.int)

                # weights update
                deltaW = alpha * \
                    (associations_delta / float(batch_size)) + \
                    m*self.last_velocity
                self.W += deltaW
                self.last_velocity = deltaW
                # bias updates mean through the batch
                self.h_bias += alpha * (h_bias_delta).mean(axis=0)
                self.v_bias += alpha * \
                    (batch - v_probs).mean(axis=0)

                error = np.sum((batch - v_probs) ** 2) / float(batch_size)
                total_error += error

            if display and verbose:
                print("Reconstructed sample from the training set")
                rand_sample = v_states[np.random.randint(v_states.shape[0])]
                print(display(self._convert_visible_binary_to_multinomial([rand_sample])))

            print("Epoch %s : error is %s" % (epoch, total_error))
            if epoch % 25 == 0 and epoch > 0:
                self.train_free_energies.append(
                    self.average_free_energy(binary_batches[0]))
                if validation is not None:
                    self.validation_free_energies.append(
                        self.average_free_energy(binary_validation))
            if epoch % m_update == 0 and epoch > 0 and m < 0.9:
                m += 0.01
            self.costs.append(total_error)
            total_error = 0

    def gibbs_sampling(self, v_in, k):
        """Performs k steps of Gibbs Sampling, starting from the visible units input.
        :param v_in_0: input of the visible units
        :param k: number of sampling steps
        :return: difference between positive associations and negative
        associations after k steps of gibbs sampling
        """

        # Sample from the hidden units given the visible units - Positive
        # Contrastive Divergence phase
        h_activations_0 = np.dot(v_in, self.W) + self.h_bias
        h_probs_0 = self.hidden_act_func(h_activations_0)
        h_states_0 = []
        for ps in h_probs_0:
            tmp = [0] * len(ps)
            tmp[ps.tolist().index(max(ps))] = 1
            h_states_0.append(tmp)
        h_states_0 = np.array(h_states_0)
        pos_associations = np.dot(v_in.T, h_states_0)

        for gibbs_steps in xrange(k):
            if gibbs_steps == 0:
                # first step: we have already computed the hidden things
                h_states = h_states_0
            else:
                # Not first step: sample hidden from new visible
                # Sample from the hidden units given the visible units -
                # Positive CD phase
                h_activations = np.dot(v_in, self.W) + self.h_bias
                h_probs = self.hidden_act_func(h_activations)
                h_states = []
                for ps in h_probs:
                    tmp = [0] * len(ps)
                    tmp[ps.tolist().index(max(ps))] = 1
                    h_states.append(tmp)
                h_states = np.array(h_states)

            # Reconstruct the visible units
            # Negative Contrastive Divergence phase
            v_activations = np.dot(h_states, self.W.T) + self.v_bias
            v_probs = self.visible_act_func(v_activations)
            # Sampling again from the hidden units
            h_activations_new = np.dot(v_probs, self.W) + self.h_bias
            h_probs_new = self.hidden_act_func(h_activations_new)
            h_states_new = []
            for ps in h_probs_new:
                tmp = [0] * len(ps)
                tmp[ps.tolist().index(max(ps))] = 1
                h_states_new.append(tmp)
            h_states_new = np.array(h_states_new)
            neg_associations = np.dot(v_probs.T, h_states_new)
            # Use the new sampled visible units in the next step
            v_in = v_probs
        return pos_associations - neg_associations, h_probs_0 - h_probs_new, v_probs, h_probs_new

    def sample_visible_from_hidden(self, h_in, gibbs_k=1):
        """
        Assuming the RBM has been trained, run the network on a set of
        hidden units, to get a sample of the visible units.
        :param h_in: states of the hidden units.
        :param gibbs_k: number of gibbs sampling steps
        :return (visible units probabilities, visible units states)
        """
        # Convert data into binary visible states
        binary_in = self._convert_visible_state_to_binary(h_in)
        (_, _, v_probs, _) = self.gibbs_sampling(binary_in, gibbs_k)
        v_states = []
        for ps in v_probs:
            tmp = [0] * len(ps)
            tmp[ps.tolist().index(max(ps))] = 1
            v_states.append(tmp)
        v_states = np.array(v_states)
        return v_probs, v_states

    def sample_hidden_from_visible(self, v_in, gibbs_k=1):
        """
        Assuming the RBM has been trained, run the network on a set of
        visible units, to get a sample of the visible units.
        :param v_in: states of the visible units.
        :param gibbs_k: number of gibbs sampling steps
        :return (hidden units probabilities, hidden units states)
        """
        # Convert data into binary visible states
        binary_in = self._convert_visible_state_to_binary(v_in)
        (_, _, h_probs, _) = self.gibbs_sampling(binary_in, gibbs_k)
        h_states = []
        for ps in h_probs:
            tmp = [0] * len(ps)
            tmp[ps.tolist().index(max(ps))] = 1
            h_states.append(tmp)
        h_states = np.array(h_states)
        return h_probs, h_states

    def visible_act_func(self, x):
        """Softmax Activation Function (One of K multinomial states)
        :param x: input to the Multinomial units, converted in binary states
        :return: Softmax for the K states of the unit
        """
        out = []
        for sample in x:
            offset = 0 # offset through the units
            probs = []
            while offset + self.k_visible < len(sample):
                den = 0.0
                for i in sample[offset: offset + self.k_visible + 1]:
                    val = np.exp(i)
                    probs.append(val)
                    den += val
                offset += self.k_visible + 1
            out.append(probs / den)
        return np.array(out)

    def hidden_act_func(self, x):
        """Softmax Activation Function (One of K multinomial states)
        :param x: input to the Multinomial units, converted in binary states
        :return: Softmax for the K states of the unit
        """
        out = []
        for sample in x:
            offset = 0 # offset through the units
            probs = []
            while offset + self.k_hidden < len(sample):
                den = 0.0
                for i in sample[offset: offset + self.k_hidden + 1]:
                    val = np.exp(i)
                    probs.append(val)
                    den += val
                offset += self.k_hidden + 1
            out.append(probs / den)
        return np.array(out)

    def average_free_energy(self, data):
        """Compute the average free energy over a representative sample
        of the training set or the validation set, already converted to binary.
        """
        wx_b = np.dot(data, self.W) + self.h_bias
        vbias_term = np.dot(data, self.v_bias)
        hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis=1)
        return (- hidden_term - vbias_term).mean(axis=0)

    def save_configuration(self, outfile):
        """Save a json representation of the RBM object.
        :param outfile: path of the output file
        """
        with open(outfile, 'w') as f:
            f.write(json.dumps({'W': self.W.tolist(),
                                'h_bias': self.h_bias.tolist(),
                                'v_bias': self.v_bias.tolist(),
                                'num_hidden': self.num_hidden,
                                'num_visible': self.num_visible,
                                'k_visible': self.k_visible,
                                'k_hidden': self.k_hidden,
                                'costs': self.costs,
                                'train_free_energies':
                                    self.train_free_energies,
                                'validation_free_energies':
                                    self.validation_free_energies}))

    def load_configuration(self, infile):
        """Load a json representation of the RBM object.
        :param infile: path of the input file
        """
        with open(infile, 'r') as f:
            data = json.load(f)
            self.W = np.array(data['W'])
            self.h_bias = np.array(data['h_bias'])
            self.v_bias = np.array(data['v_bias'])
            self.num_hidden = data['num_hidden']
            self.num_visible = data['num_visible']
            self.k_hidden = data['k_hidden']
            self.k_visible = data['k_visible']
            self.costs = data['costs']
            self.train_free_energies = data['train_free_energies']
            self.validation_free_energies = data['validation_free_energies']

    def _convert_visible_state_to_binary(self, x):
        """Converts the Multinomial state x into binary units
        :param x: Multinomial batch of the visible units
        :return: binary states of the units
        """
        bin = []
        for sample in x:
            sample_multin = []
            for i,u in enumerate(sample):
                tmp = [0] * (self.k_visible + 1)
                tmp[int(u)] = 1
                sample_multin.append(tmp)
            bin.append(np.array([item for sublist in sample_multin for item in sublist]))
        return np.array(bin)

    def _convert_visible_binary_to_multinomial(self, x):
        """Converts the visible binary state x into multinomial states
        :param x: visible binary state
        :return: multinomial state vector
        """
        out = []
        offset = 0
        for sample in x:
            while offset + self.k_visible < len(sample):
                tmp = sample[offset: offset + self.k_visible + 1]
                out.append(tmp.tolist().index(max(tmp)))
                offset += self.k_visible + 1
        return np.array(out)

    def _convert_hidden_binary_to_multinomial(self, x):
        """Converts the hidden binary state x into multinomial states
        :param x: hidden binary state
        :return: multinomial state vector
        """
        out = []
        offset = 0
        for sample in x:
            while offset + self.k_hidden < len(sample):
                tmp = sample[offset: offset + self.k_hidden + 1]
                out.append(tmp.tolist().index(max(tmp)))
                offset += self.k_hidden + 1
        return np.array(out)

    def _convert_hidden_state_to_binary(self, x):
        """Converts the Multinomial state x into binary units
        :param x: Multinomial batch of the hidden units
        :return: binary states of the units
        """
        bin = []
        for sample in x:
            sample_multin = []
            for i,u in enumerate(sample):
                tmp = [0] * (self.k_hidden + 1)
                tmp[int(u)] = 1
                sample_multin.append(tmp)
            bin.append(np.array([item for sublist in sample_multin for item in sublist]))
        return np.array(bin)
