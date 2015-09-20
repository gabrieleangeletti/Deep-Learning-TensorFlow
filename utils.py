from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def logistic(x):
    """Logistic Function.
    """
    return 1. / (1. + np.exp(-x))


def softmax(x):
    """Return the softmax of x, that is an array of the same length as x, with all 0s and 1 in the index
    of the max element in x.
    :param x: numpy array whose elements are in the range [0, 1]
    :return: softmax of x
    """
    mx_i = np.where(x == max(x))[0][0]
    softmx = [0] * len(x)
    softmx[mx_i] = 1
    return softmx


def int2binary_vect(y):
    """Converts the integer vector y into a vector of binary vector, each representing one integer.
    Usually y are the labels of a supervised dataset, that must be converted to binary vector for use with
    stochastic binary neurons or RBM/DBN.
    :param y: integer vector
    :return: binary representation of y
    """
    n_bits = (lambda x: max(x) + 1 if 0 in x else max(x))(y)
    out = []
    for _, v in enumerate(y):
        tmp = [0] * n_bits
        tmp[v] = 1
        out.append(tmp)
    return np.array(out)


def normalize_dataset_to_binary(data):
    """This function normalize the values of data to be in {0,1}
    :param data: dataset matrix
    Normalize the values of data to be binary values - {0, 1}, so that they can be
    used in the Restricted Boltzmann Machine (RBM) layer of the DBN.
    The values are first normalized to be in [0,1], then the probability
    of the values is computed.
    """

    # min of the dataset
    mn = min([min(row) for row in data])
    # max of the dataset
    mx = max([max(row) for row in data])
    # normalize the dataset
    data = (data - mn) / float(mx - mn)
    # compute the probabilities
    rand = np.random.random(data.shape)
    return (data > rand).astype(np.int)


def normalize_dataset(data):
    """This function normalize the values of data to be in [0,1]
    :param data: dataset matrix
    Normalize the values of data to be real values in the range [0, 1], so that they can be
    used in the Gaussian Restricted Boltzmann Machine (RBM) layer of the DBN.
    """

    # min of the dataset
    mn = min([min(row) for row in data])
    # max of the dataset
    mx = max([max(row) for row in data])
    # normalize the dataset
    data = (data - mn) / float(mx - mn)
    return data


def generate_batches(data, batch_size):
    """Generate a list of batches from input data.
    :param batch_size: the size of each batch
    :return [batches]: a list of data batches
    """
    # divide the data into batches
    batches = []
    start_offset = 0
    offset = int(float(data.shape[0]) / batch_size)

    for i in xrange(offset):
        batches.append(data[start_offset:start_offset+batch_size])
        start_offset += batch_size

    return batches


def discretize_dataset(data, N):
    """Constraint dataset values to be integers between 0 and N.
    :param data: dataset to be converted
    :param N: max integer
    :return: the discretized dataset
    """
    # min of the dataset
    mn = min([min(sample) for sample in data])
    # max of the dataset
    mx = max([max(sample) for sample in data])
    threshold = int(round((mx - mn) / float(N+1)))
    threshold_vect = []
    for i in xrange(N+2):
        threshold_vect.append(threshold * i)
    converted_data = np.array([[0] * len(data[0])])
    for sample in data:
        converted_sample = np.array([])
        for elem in sample:
            for i, t in enumerate(threshold_vect):
                if elem <= t:
                    if i == 0:
                        converted_sample = np.append(converted_sample, 0)
                        break
                    else:
                        converted_sample = np.append(converted_sample, i-1)
                        break
        converted_data = np.append(converted_data, [converted_sample], axis=0)
    # remove the first row used only for np.append
    return np.delete(converted_data, 0, 0)


def gen_image(img, width, height, outfile):
    assert len(img) == width * height

    img = img.reshape(width, height)

    desired_width = 3  # in inches
    scale = desired_width / float(width)

    fig, ax = plt.subplots(1, 1, figsize=(desired_width, height*scale))
    ax.imshow(img, cmap=cm.Greys_r, interpolation='none')
    fig.savefig(outfile, dpi=300)


class AbstractUpdatingParameter(object):
    """Class representing an abstract update rule for a parameter during learning.
    """

    __metaclass__ = ABCMeta

    def __init__(self, param):
        """Initialize a parameter update object
        :param param: tuning parameter
        """
        self.param = param
        self.t = 0.0  # current number of time steps

    @abstractmethod
    def update(self):
        """update the parameter accoring to the update rule.
        """
        pass


class ConstantParameter(AbstractUpdatingParameter):
    """Class representing a constant parameter.
    """

    def __init__(self, param):
        """Initialization.
        :param param: constant parameter.
        """
        super(ConstantParameter, self).__init__(param)

    def update(self):
        return self.param


class LinearDecayParameter(AbstractUpdatingParameter):
    """Class representing a linearly decay parameter, following the formula:
    base_rate * (1 - progress) + final_rate * progress, where base rate is the initial value for the parameter
    and final_rate is the final value. Progress is the ratio between the current iteration and the total number
    of iterations.
    """

    def __init__(self, param, n):
        """Initialization.
        :param n: number of iterations
        """
        super(LinearDecayParameter, self).__init__(param)
        self.n = n
        self.final_rate = 0.001

    def update(self):
        self.t += 1.0
        progress = self.t / self.n
        return self.param * (1 - progress) + self.final_rate * progress


class ExpDecayParameter(AbstractUpdatingParameter):
    """Class representing an exponentially decay parameter, following the formula:
    1 / ( 1 + (t / param) ) where t is the number of time steps elapsed since the algorithm
    was initiated and param is a tuning parameter.
    """

    def __init__(self, param):
        """Initialization
        :param param: tuning parameter
        """
        super(ExpDecayParameter, self).__init__(param)

    def update(self):
        self.t += 1.0
        return 1 / (1 + (self.t / self.param))


def prepare_alpha_update(alpha_update_rule, alpha, epochs):
    # Learning rate update rule
    if alpha_update_rule == 'exp':
        alpha_rule = ExpDecayParameter(alpha)
    elif alpha_update_rule == 'linear':
        alpha_rule = LinearDecayParameter(alpha, epochs)
    elif alpha_update_rule == 'constant':
        alpha_rule = ConstantParameter(alpha)
    else:
        raise Exception('alpha_update_rule must be in ["exp", "constant", "linear"]')
    assert alpha_rule is not None

    return alpha_rule





