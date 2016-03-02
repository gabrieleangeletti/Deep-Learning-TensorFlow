from scipy import misc
import tensorflow as tf
import numpy as np


# ################### #
#   Network helpers   #
# ################### #


def sample_prob(probs, rand):
    """ Takes a tensor of probabilities (as from a sigmoidal activation)
    and samples from all the distributions

    :param probs: tensor of probabilities
    :param rand: tensor (of the same shape as probs) of random values

    :return : binary sample of probabilities
    """
    return tf.nn.relu(tf.sign(probs - rand))


def xavier_init(fan_in, fan_out, const=1):
    """ Xavier initialization of network weights.
    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow

    :param fan_in: fan in of the network (n_features)
    :param fan_out: fan out of the network (n_components)
    :param const: multiplicative constant
    """
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)


# ################ #
#   Data helpers   #
# ################ #


def create_perfect_duplicates(data, n):
    """ Create perfect duplicates of n random data_points.

    :type data: array_like
    :param data: input data

    :type n: int
    :param n: number of duplicates to create

    :return: output data = input data + duplicates
    """
    assert n < data.shape[0]

    out_data = data.copy()

    for i in range(n):
        sample = data[np.random.randint(data.shape[0])]
        out_data = np.concatenate((out_data, [sample]), axis=0)

    return out_data


def create_masking_noise_duplicates(data, v, n):
    """ Create masking noise duplicates of n random data_points.

    :param data: input data
    :param v: fraction of elements to distort
    :param n: number of duplicates to create

    :return: output data = input data + masking noise duplicate
    """
    assert n < data.shape[0]

    out_data = data.copy()

    for i in range(n):
        sample = data[np.random.randint(data.shape[0])]

        # Apply masking noise
        mask = np.random.randint(0, data.shape[1], v)

        for m in mask:
            sample[m] = 0.

        # Add sample to outdata
        out_data = np.concatenate((out_data, [sample]), axis=0)

    return out_data


def create_blurred_duplicates(data, n, noise_type, v):
    """ Create blurred duplicates of n random data_points.

    :type data: array_like
    :param data: input data

    :type n: int
    :param n: number of duplicates to create

    :type noise_type: string
    :param noise_type: "masking" or "salt_and_pepper"

    :type v: float
    :param v: fraction of elements to distort

    :return: output data = input data + duplicates
    """
    assert n < data.shape[0]
    assert v < data.shape[1]
    assert noise_type in ['masking', 'salt_and_pepper']

    out_data = data.copy()

    corruption = int(v * data.shape[1])

    for i in range(n):
        sample = data[np.random.randint(len(data))]

        if noise_type == 'masking':
            sample = masking_noise(np.array([sample]), corruption)

        elif noise_type == 'salt_and_pepper':
            sample = salt_and_pepper_noise(np.array([sample]), corruption)

        out_data = np.concatenate((out_data, sample), axis=0)

    return out_data


def gen_batches(data, batch_size):
    """ Divide input data into batches.

    :param data: input data
    :param batch_size: size of each batch

    :return: data divided into batches
    """
    data = np.array(data)

    for i in range(0, data.shape[0], batch_size):
        yield data[i:i+batch_size]


def conv2bin(data):
    """ Convert a matrix of probabilities into
    binary values. If the matrix has values <= 0 or >= 1, the values are
    normalized to be in [0, 1].

    :type data: numpy array
    :param data: input matrix

    :return: converted binary matrix
    """
    if data.min() < 0 or data.max() > 1:
        data = normalize(data)

    out_data = data.copy()

    for i, sample in enumerate(out_data):

        for j, val in enumerate(sample):

            if np.random.random() <= val:
                out_data[i][j] = 1
            else:
                out_data[i][j] = 0

    return out_data


def normalize(data):
    """ Normalize the data to be in the [0, 1] range.

    :param data:

    :return: normalized data
    """
    out_data = data.copy()

    for i, sample in enumerate(out_data):
        out_data[i] /= sum(out_data[i])

    return out_data


def bins2sets(bin_data):
    """ Convert a binary matrix into a collection of sets.
    This function is used to convert binary matrix of feature activations into sets
    representing which feature detector was activated for which input sample.
    For example the matrix [ [1, 0, 1, 0], [0, 0, 1, 0] ] will be converted in:
    [ ['f0', 'f2'], ['f2'] ]

    :type bin_data: numpy array
    :param bin_data: binary matrix

    :return: list of sets representing the binary matrix
    """

    feat = 'f'  # feature id
    out_data = {}

    for i, sample in enumerate(bin_data):
        sample_set = set()

        for j, activation in enumerate(sample):
            if activation == 1.:
                sample_set.add(feat + str(j))

        out_data[i] = sample_set

    return out_data


def masking_noise(X, v):
    """ Apply masking noise to data in X, in other words a fraction v of elements of X
    (chosen at random) is forced to zero.

    :param X: array_like, Input data
    :param v: int, fraction of elements to distort

    :return: transformed data
    """
    X_noise = X.copy()

    n_samples = X.shape[0]
    n_features = X.shape[1]

    for i in range(n_samples):
        mask = np.random.randint(0, n_features, v)

        for m in mask:
            X_noise[i][m] = 0.

    return X_noise


def salt_and_pepper_noise(X, v):
    """ Apply salt and pepper noise to data in X, in other words a fraction v of elements of X
    (chosen at random) is set to its maximum or minimum value according to a fair coin flip.
    If minimum or maximum are not given, the min (max) value in X is taken.

    :param X: array_like, Input data
    :param v: int, fraction of elements to distort

    :return: transformed data
    """
    X_noise = X.copy()
    n_features = X.shape[1]

    mn = X.min()
    mx = X.max()

    for i, sample in enumerate(X):
        mask = np.random.randint(0, n_features, v)

        for m in mask:

            if np.random.random() < 0.5:
                X_noise[i][m] = mn
            else:
                X_noise[i][m] = mx

    return X_noise

# ############# #
#   Utilities   #
# ############# #


def gen_image(img, width, height, outfile, img_type='grey'):
    assert len(img) == width * height or len(img) == width * height * 3

    if img_type == 'grey':
        misc.imsave(outfile, img.reshape(width, height))

    elif img_type == 'color':
        misc.imsave(outfile, img.reshape(3, width, height))


def transform_from_params(data, gibbs_k, model_params):
    nvis = data.shape[1]
    W, bh, bv = model_params['W'], model_params['bh_'], model_params['bv_']

    nhid = W.shape[1]

    # Symbolic variables
    x = tf.placeholder('float', [None, nvis], name='x-input')

    hrand = tf.placeholder('float', [None, nhid], name='hrand')
    vrand = tf.placeholder('float', [None, nvis], name='vrand')

    # Biases
    bh_ = tf.Variable(bh, name='hidden-bias')
    bv_ = tf.Variable(bv, name='visible-bias')

    W_ = tf.Variable(W, name='weights')

    nn_input = x

    # Initialization
    hprobs1 = None

    for step in range(gibbs_k):

        if step % 10 == 0:
            print('Gibbs sampling step: {}...'.format(step))

        # Positive Contrastive Divergence phase
        hprobs = tf.nn.sigmoid(tf.matmul(nn_input, W_) + bh_)
        hstates = sample_prob(hprobs, hrand)

        # Negative Contrastive Divergence phase
        vprobs = tf.nn.sigmoid(tf.matmul(hstates, tf.transpose(W_)) + bv_)
        vstates = sample_prob(vprobs, vrand)

        # Sample again from the hidden units
        hprobs1 = tf.nn.sigmoid(tf.matmul(vprobs, W_) + bh_)

        # Use the reconstructed visible units as input for the next step
        nn_input = vstates

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:

        sess.run(init_op)

        return hprobs1.eval({x: data,
                             hrand: np.random.rand(data.shape[0], nhid),
                             vrand: np.random.rand(data.shape[0], nvis)})
