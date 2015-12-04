from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def logistic(x):
    """Logistic Function.
    """
    return 1. / (1. + np.exp(-x))


def logistic_dot(x):
    """Logistic Function derivative.
    """
    return (1. / (1. + np.exp(-x))) * (1 - (1. / (1. + np.exp(-x))))


def softmax(x):
    """Return the softmax of x, that is an array of the same length as x, with all 0s and 1 in the index
    of the max element in x.
    :param x: numpy array whose elements are in the range [0, 1]
    :return: softmax values of x, softmax binary array of x
    """
    exps = []
    out = []
    for sample in x:
        # Compute exponentials
        sample_exp = []
        den = 0.
        for s in sample:
            val = np.exp(s)
            den += val
            sample_exp.append(val)
        sample_exp /= den
        exps.append(sample_exp)
        # Compute max value
        mx_i = np.where(sample_exp == max(sample_exp))[0][0]
        softmx = [0] * len(sample_exp)
        softmx[mx_i] = 1
        out.append(softmx)
    return np.array(exps), np.array(out)


def probs_to_binary(x):
    """Convert an array of probabilities to an array of stochastic binary states.
    :param x: input probabilities
    :return: stochastic binary states defined by x
    """
    return (x > np.random.rand(x.shape[0], x.shape[1])).astype(np.int)


def filter_dropout(w, dropout):
    """Filter the column of the weights matrix corresponding to zeros in the
    dropout vector.
    :param w: weights matrix of dimension num_visible x num_hidden
    :param dropout: binary vector whose elements are 1 with probability 0.5
    """
    filtered_w = []
    for i in range(w.shape[0]):
        filtered_w.append([])
        for j in range(w.shape[1]):
            if dropout[j] == 1:
                filtered_w[i].append(w[i][j])
    return np.array(filtered_w)


def merge_data_labels(data, y):
    """Merge data array with labels array in a unique array data + labels.
    The two arrays can have an arbitrary number of samples, but the number of samples must be the same.
    :param data: data array
    :param y: labels array
    :return: joint array between data and labels
    """
    # merge data_repr with y
    joint_data = []
    for j in range(data.shape[0]):
        joint_data.append(np.hstack([data[j], y[j]]))
    return np.array(joint_data)


def compute_mean_square_error(preds, targets):
    """Compute the Mean Square Error (MSE) across a set of predictions and relative targets.
    :param preds: predictions made by the classifier
    :param targets: supervised targets
    :return: average mean square error over the batch
    """
    np.sum((targets - preds) ** 2) / float(preds.shape[0])


def compute_cross_entropy_error(preds, targets):
    """Compute the Average Cross Entropy error (ACE) across a set of predictions and relative targets.
    :param preds: predictions made by the classifier
    :param targets: supervised targets
    :return: average cross entropy error over the batch
    """
    batch_size = len(preds)
    num_labels = len(targets[0])
    err = 0.
    for i in range(batch_size):
        for k in range(num_labels):
            err += targets[i][k] * np.log(preds[i][k])
    return -err


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


def binary2int_vect(y):
    """Converts the binary softmax vector y into the integer label that y is representing.
    This is the decoding function with respect to int2binary_vect.
    :param y: binary softmax vector
    :return: integer representation of y
    """
    out = []
    for sample in y:
        out.append(np.where(np.array(sample) == 1)[0][0])
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
        self.t = 0.  # current number of time steps

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


class LinearParameter(AbstractUpdatingParameter):
    """Class representing a linearly ascending or decading parameter, following the formula:
    base_rate * (1 - progress) + final_rate * progress, where base rate is the initial value for the parameter
    and final_rate is the final value. Progress is the ratio between the current iteration and the total number
    of iterations.
    """

    def __init__(self, base, final, n):
        """Initialization.
        :param n: number of iterations
        """
        super(LinearParameter, self).__init__(base)
        self.n = n
        self.base = base
        self.final = final
        self.progress = 0.

    def update(self):
        self.progress += 1. / self.n
        return self.base * (1 - self.progress) + self.final * self.progress


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
        self.t += 1.
        return 1 / (1 + (self.t / self.param))


def prepare_parameter_update(param_update_rule, param, epochs):
    # Learning rate update rule
    if param_update_rule == 'exp':
        param_rule = ExpDecayParameter(param[0])
    elif param_update_rule == 'linear':
        param_rule = LinearParameter(param[0], param[1], epochs)
    elif param_update_rule == 'constant':
        param_rule = ConstantParameter(param[0])
    else:
        raise Exception('alpha_update_rule must be in ["exp", "constant", "linear"]')
    assert param_rule is not None

    return param_rule
