import numpy as np


def normalize_dataset(X):
    """This function normalize the values of X to be in {0,1}

    :param X: dataset matrix

    Normalize the values of X to be binary values - {0, 1}, so that they can be
    used in the Restricted Boltzmann Machine (RBM) layer of the DBM.
    The values are first normalized to be in [0,1], then the probability
    of the values is computed.

    """

    # min of the dataset
    mn = min([min(row) for row in X])
    # max of the dataset
    mx = max([max(row) for row in X])
    # normalize the dataset
    X = (X - mn) / float(mx - mn)
    # compute the probabilities
    rand = np.random.random(X.shape)
    return (X > rand).astype(np.int)


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
