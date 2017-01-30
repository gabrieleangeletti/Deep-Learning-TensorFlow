"""Collection of Tensorflow specific utilities."""

import os
import tensorflow as tf

from ..core.config import Config


def init_tf_ops(sess):
    """Initialize TensorFlow operations.

    This function initialize the following tensorflow ops:
        * init variables ops
        * summary ops

    Parameters
    ----------

    sess : object
        Tensorflow `Session` object

    Returns
    -------

    tuple : (summary_merged, summary_writer)
        * tf merged summaries object
        * tf summary writer object
    """
    summary_merged = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

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

    return (summary_merged, summary_writer)
