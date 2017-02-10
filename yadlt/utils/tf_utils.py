"""Collection of Tensorflow specific utilities."""

import os
import tensorflow as tf

from ..core.config import Config


def init_tf_ops(sess):
    """Initialize TensorFlow operations.

    This function initialize the following tensorflow ops:
        * init variables ops
        * summary ops
        * create model saver

    Parameters
    ----------

    sess : object
        Tensorflow `Session` object

    Returns
    -------

    tuple : (summary_merged, summary_writer)
        * tf merged summaries object
        * tf summary writer object
        * tf saver object
    """
    summary_merged = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess.run(init_op)

    # Retrieve run identifier
    run_id = 0
    for e in os.listdir(Config().logs_dir):
        if e[:3] == 'run':
            r = int(e[3:])
            if r > run_id:
                run_id = r
    run_id += 1
    run_dir = os.path.join(Config().logs_dir, 'run' + str(run_id))
    print('Tensorboard logs dir for this run is %s' % (run_dir))

    summary_writer = tf.summary.FileWriter(run_dir, sess.graph)

    return (summary_merged, summary_writer, saver)


def run_summaries(
        sess, merged_summaries, summary_writer, epoch, feed, tens):
    """Run the summaries and error computation on the validation set.

    Parameters
    ----------

    sess : tf.Session
        Tensorflow session object.

    merged_summaries : tf obj
        Tensorflow merged summaries obj.

    summary_writer : tf.summary.FileWriter
        Tensorflow summary writer obj.

    epoch : int
        Current training epoch.

    feed : dict
        Validation feed dict.

    tens : tf.Tensor
        Tensor to display and evaluate during training.
        Can be self.accuracy for SupervisedModel or self.cost for
        UnsupervisedModel.

    Returns
    -------

    err : float, mean error over the validation set.
    """
    try:
        result = sess.run([merged_summaries, tens], feed_dict=feed)
        summary_str = result[0]
        out = result[1]
        summary_writer.add_summary(summary_str, epoch)
    except tf.errors.InvalidArgumentError:
        out = sess.run(tens, feed_dict=feed)

    return out
