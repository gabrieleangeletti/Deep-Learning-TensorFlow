import numpy as np


def normalize_dataset(data):
    """This function normalize the values of data to be in {0,1}
    :param data: dataset matrix
    Normalize the values of data to be binary values - {0, 1}, so that they can be
    used in the Restricted Boltzmann Machine (RBM) layer of the DBM.
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
            for i,t in enumerate(threshold_vect):
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





