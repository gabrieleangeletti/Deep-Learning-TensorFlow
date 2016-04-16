from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """ 2D Convolution between input x and filter W.
    x and W must be 4D tensors.
    x: [batch, in_height, in_width, in_channels]
    W: [filter_height, filter_width, in_channels, out_channels]
    Must have strides[0] = strides[3] = 1. For most common case of the same
    horizontal and vertices strides, strides = [1, stride, stride, 1].
    For example stride of 1 is [1, 1, 1, 1].
    Strides represent the stride of the sliding window for each dimension of the
    input.
    -strides[0] is always equal to 1 because we want to process each bach.
    -strides[3] is always equal to 1 because we want to process each channel.
    padding can be 'SAME' or 'VALID', it's the type of padding algorithm we want to use.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_with_argmax_2x2(x):
    """ Performs max pooling on the input x.
    x: [batch, height, width, channels] and type tf.float32
    ksize: list of ints with length >= 4. The size of the window for each dimension
        of the input tensor.
    strides: list of ints with length >= 4. The stride of the sliding window for each dimension
        of the input tensor.
    padding: same as in conv2d
    """
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
    """ Performs max pooling on the input x.
    x: [batch, height, width, channels] and type tf.float32
    ksize: list of ints with length >= 4. The size of the window for each dimension
        of the input tensor.
    strides: list of ints with length >= 4. The stride of the sliding window for each dimension
        of the input tensor.
    padding: same as in conv2d
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def show_learning_images_numpy():

    # to visualize 1st conv layer Weights
    vv1 = sess.run(W_conv1)

    # to visualize 1st conv layer output
    vv2 = sess.run(h_conv1, feed_dict={x: batch[0], keep_prob: 1.0})
    # in case of bunch out - slice first img of the batch
    vv2 = vv2[0, :, :, :]

    def vis_conv(v, ix, iy, ch, cy, cx, p=0):
        # W_conv1 is [5, 5, 1, 32]
        # reshape to [5, 5, 32]
        # i.e. remove the image dimension (the outer []),
        # from: 5x5 matrix in which each element is a (1 x 32) array
        #      [ [ [[32]], [[32]], [[32]], [[32]], [[32]],
        #        [ [[32]], [[32]], [[32]], [[32]], [[32]],
        #        [ [[32]], [[32]], [[32]], [[32]], [[32]],
        #        [ [[32]], [[32]], [[32]], [[32]], [[32]],
        #        [ [[32]], [[32]], [[32]], [[32]], [[32]] ]
        # to:  5x5 matrix in which each element is a (32) array
        #      [ [ [32], [32], [32], [32], [32] ],
        #        [ [32], [32], [32], [32], [32] ],
        #        [ [32], [32], [32], [32], [32] ],
        #        [ [32], [32], [32], [32], [32] ],
        #        [ [32], [32], [32], [32], [32] ] ]
        v = np.reshape(v, (iy, ix, ch))

        # add a couple of pixels of zero pad around the image.
        # This padding is needed in order to put some space between
        # different images in the grid.
        # constant_values = p tells that the added pixels should be 0
        ix += 2
        iy += 2
        # for the first dimension, add one pixel to the left and
        # one pixel to the right
        # the same for the second dimension
        npad = ((1, 1), (1, 1), (0, 0))
        # From 5x5x32 v becomes 7x7x32
        v = np.pad(v, pad_width=npad, mode='constant', constant_values=p)

        # reshape from 7x7x32 to 7x7x4x8
        # from 32 channels to 4x8 channels
        # so that each feature map can be displayed as a
        # 4x8 image
        v = np.reshape(v, (iy, ix, cy, cx))

        # The current order (7x7x4x8), if flattened,
        # would list all the channels for the first pixel (iterating over cx and cy),
        # before listing the channels of the second pixel (incrementing ix).
        # Going across the rows of pixels (ix) before incrementing to the next row (iy).

        # We want the order that would lay out the images in a grid.
        # So you go across a row of an image (ix), before stepping along the row of channels (cx),
        # when you hit the end of the row of channels you step to the next row in the image (iy)
        # and when you run out or rows in the image you increment to the next row of channels (cy). so:
        # reshape from 7x7x4x8 to 4x7x8x7
        # from (iy, ix, cy, cx) to (cy,iy,cx,ix)
        v = np.transpose(v, (2, 0, 3, 1))

        # now that the pixels are in the right order, we can safely flatten it into a 2d tensor:
        # from 4x7x8x7 to 28x56
        v = np.reshape(v, (cy*iy, cx*ix))

        return v

    # W_conv1 - weights
    ix = 5  # data size
    iy = 5
    ch = 32
    cy = 4   # grid from channels:  32 = 4x8
    cx = 8
    v = vis_conv(vv1, ix, iy, ch, cy, cx)
    plt.figure(figsize=(8, 8))
    plt.imshow(v, cmap="Greys_r", interpolation='nearest')

    #  h_conv1 - processed image
    ix = 28  # data size
    iy = 28
    v = vis_conv(vv2, ix, iy, ch, cy, cx)
    plt.figure(figsize=(8, 8))
    plt.imshow(v, cmap="Greys_r", interpolation='nearest')
    plt.show()


def show_learning_images_tf():
    ix = 28
    iy = 28
    channels = 1

    # first slice off 1 image, and remove the image dimension
    Vs = tf.slice(x_image, (0, 0, 0, 0), (1, -1, -1, -1))  # V[0,...]
    Vs = tf.reshape(Vs, (iy, ix, channels))

    # Next add a couple of pixels of zero padding around the image
    ix += 4
    iy += 4
    Vs = tf.image.resize_image_with_crop_or_pad(Vs, iy, ix)

    # Then reshape so that instead of 32 channels you have 4x8 channels, lets call them cy=4 and cx=8.
    cy = 4
    cx = 8
    Vs = tf.reshape(Vs, (iy, ix, cy, cx))

    # Now the tricky part. tf seems to return results in C-order, numpy's default.
    # The current order, if flattened, would list all the channels for the first pixel
    # (iterating over cx and cy), before listing the channels of the second pixel (incrementing ix).
    # Going across the rows of pixels (ix) before incrementing to the next row (iy).
    # We want the order that would lay out the images in a grid. So you go across a row of an image (ix),
    # before stepping along the row of channels (cx),
    # when you hit the end of the row of channels you step to the next row in the image (iy)
    # and when you run out or rows in the image you increment to the next row of channels (cy). so:
    Vs = tf.transpose(Vs, (2, 0, 3, 1))  # cy,iy,cx,ix

    # now that the pixels are in the right order, we can safely flatten it into a 2d tensor:
    # image_summary needs 4d input
    Vs = tf.reshape(Vs, (1, cy*iy, cx*ix, 1))
    plt.figure(figsize=(8, 8))
    plt.imshow(Vs, cmap="Greys_r", interpolation='nearest')
    plt.show()


x = tf.placeholder('float', [None, 784])
y_ = tf.placeholder('float', [None, 10])

# Parameters for the first layer. The filter is [5, 5, 1, 32].
# This means that it's a 5x5 filter, which takes the only input channel
# (one because MNIST have grayscale images) and produces 32 features
# The bias vector have one component for each features
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# We must reshape the input to be a 4D tensor
# The first dimension is -1  because, from tf docs:
# If one component of shape is the special value -1,
# the size of that dimension is computed so that the total size remains constant.
# In particular, a shape of [-1] flattens into 1-D. At most one component of shape can be -1.
# The second and third dimensions are image height and width respectively.
# The fourth dimension is the number of color channels.
x_image = tf.reshape(x, [-1, 28, 28, 1])

# First layer, convolution + relu + max pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# indices1 = tf.reshape(indices1, [14, 14, 32])
# tf.constant(indices1)

# Parameters for the second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = weight_variable([64])

# Second layer, convolution + relu + max pooling
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer.
# Layer of 1024 neurons to allow processing of the entire image, then softmax layer
# We need to reshape the 4D tensor from the pooling layer into a batch of vectors
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Train the model
with tf.Session() as sess:
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 10 == 0:
            data_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0}
            train_accuracy = accuracy.eval(feed_dict=data_dict)
            print "step %d, training accuracy %g" % (i, train_accuracy)
        if i % 50 == 0:
            show_learning_images_numpy()

        # print(indices1.eval({x: batch[0], y_: batch[1], keep_prob: 0.5}))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print "test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
