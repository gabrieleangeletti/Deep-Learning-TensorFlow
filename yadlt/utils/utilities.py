"""Utitilies module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import misc
import tensorflow as tf

# ################### #
#   Network helpers   #
# ################### #


def sample_prob(probs, rand):
    """Get samples from a tensor of probabilities.

    :param probs: tensor of probabilities
    :param rand: tensor (of the same shape as probs) of random values
    :return: binary sample of probabilities
    """
    return tf.nn.relu(tf.sign(probs - rand))


def corrupt_input(data, sess, corrtype, corrfrac):
    """Corrupt a fraction of data according to the chosen noise method.

    :return: corrupted data
    """
    corruption_ratio = np.round(corrfrac * data.shape[1]).astype(np.int)

    if corrtype == 'none':
        return np.copy(data)

    if corrfrac > 0.0:
        if corrtype == 'masking':
            return masking_noise(data, sess, corrfrac)

        elif corrtype == 'salt_and_pepper':
            return salt_and_pepper_noise(data, corruption_ratio)
    else:
        return np.copy(data)


def xavier_init(fan_in, fan_out, const=1):
    """Xavier initialization of network weights.

    https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow

    :param fan_in: fan in of the network (n_features)
    :param fan_out: fan out of the network (n_components)
    :param const: multiplicative constant
    """
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)


def seq_data_iterator(raw_data, batch_size, num_steps):
    """Sequence data iterator.

    Taken from tensorflow/models/rnn/ptb/reader.py
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps: (i+1) * num_steps]
        y = data[:, i * num_steps + 1: (i+1) * num_steps + 1]
    yield (x, y)


# ################ #
#   Data helpers   #
# ################ #


def gen_batches(data, batch_size):
    """Divide input data into batches.

    :param data: input data
    :param batch_size: size of each batch
    :return: data divided into batches
    """
    data = np.array(data)

    for i in range(0, data.shape[0], batch_size):
        yield data[i:i + batch_size]


def to_one_hot(dataY):
    """Convert the vector of labels dataY into one-hot encoding.

    :param dataY: vector of labels
    :return: one-hot encoded labels
    """
    nc = 1 + np.max(dataY)
    onehot = [np.zeros(nc, dtype=np.int8) for _ in dataY]
    for i, j in enumerate(dataY):
        onehot[i][j] = 1
    return onehot


def conv2bin(data):
    """Convert a matrix of probabilities into binary values.

    If the matrix has values <= 0 or >= 1, the values are
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
    """Normalize the data to be in the [0, 1] range.

    :param data:
    :return: normalized data
    """
    out_data = data.copy()

    for i, sample in enumerate(out_data):
        out_data[i] /= sum(out_data[i])

    return out_data


def masking_noise(data, sess, v):
    """Apply masking noise to data in X.

    In other words a fraction v of elements of X
    (chosen at random) is forced to zero.
    :param data: array_like, Input data
    :param sess: TensorFlow session
    :param v: fraction of elements to distort, float
    :return: transformed data
    """
    data_noise = data.copy()
    rand = tf.random_uniform(data.shape)
    data_noise[sess.run(tf.nn.relu(tf.sign(v - rand))).astype(np.bool)] = 0

    return data_noise


def salt_and_pepper_noise(X, v):
    """Apply salt and pepper noise to data in X.

    In other words a fraction v of elements of X
    (chosen at random) is set to its maximum or minimum value according to a
    fair coin flip.
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


def expand_args(**args_to_expand):
    """Expand the given lists into the length of the layers.

    This is used as a convenience so that the user does not need to specify the
    complete list of parameters for model initialization.
    IE the user can just specify one parameter and this function will expand it
    """
    layers = args_to_expand['layers']
    try:
        items = args_to_expand.iteritems()
    except AttributeError:
        items = args_to_expand.items()

    for key, val in items:
        if isinstance(val, list) and len(val) != len(layers):
            args_to_expand[key] = [val[0] for _ in layers]

    return args_to_expand


def flag_to_list(flagval, flagtype):
    """Convert a string of comma-separated tf flags to a list of values."""
    if flagtype == 'int':
        return [int(_) for _ in flagval.split(',') if _]

    elif flagtype == 'float':
        return [float(_) for _ in flagval.split(',') if _]

    elif flagtype == 'str':
        return [_ for _ in flagval.split(',') if _]

    else:
        raise Exception("incorrect type")


def str2actfunc(act_func):
    """Convert activation function name to tf function."""
    if act_func == 'sigmoid':
        return tf.nn.sigmoid

    elif act_func == 'tanh':
        return tf.nn.tanh

    elif act_func == 'relu':
        return tf.nn.relu


def random_seed_np_tf(seed):
    """Seed numpy and tensorflow random number generators.

    :param seed: seed parameter
    """
    if seed >= 0:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        return True
    else:
        return False


def gen_image(img, width, height, outfile, img_type='grey'):
    """Save an image with the given parameters."""
    assert len(img) == width * height or len(img) == width * height * 3

    if img_type == 'grey':
        misc.imsave(outfile, img.reshape(width, height))

    elif img_type == 'color':
        misc.imsave(outfile, img.reshape(3, width, height))


def get_weights_as_images(weights_npy, width, height, outdir='img/',
                          n_images=10, img_type='grey'):
    """Create and save the weights of the hidden units as images.
    :param weights_npy: path to the weights .npy file
    :param width: width of the images
    :param height: height of the images
    :param outdir: output directory
    :param n_images: number of images to generate
    :param img_type: 'grey' or 'color' (RGB)
    """
    weights = np.load(weights_npy)
    perm = np.random.permutation(weights.shape[1])[:n_images]

    for p in perm:
        w = np.array([i[p] for i in weights])
        image_path = outdir + 'w_{}.png'.format(p)
        gen_image(w, width, height, image_path, img_type)
