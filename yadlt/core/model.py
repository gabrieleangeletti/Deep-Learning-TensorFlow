import tensorflow as tf
import os, errno


class Model(object):

    """ Class representing an abstract Model.
    """

    def __init__(self, model_name, main_dir, models_dir, data_dir, summary_dir):

        """
        :param model_name: name of the model, used as filename. string, default 'dae'
        :param main_dir: main directory to put the stored_models, data and summary directories
        """

        self.model_name = model_name
        self.main_dir = main_dir
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.tf_summary_dir = summary_dir
        self.model_path = os.path.join(self.models_dir, self.model_name)
        
        print('Creating %s directory to save/restore models' % (self.models_dir))
        self._create_dir(self.models_dir)
        print('Creating %s directory to save model generated data' % (self.data_dir))
        self._create_dir(self.data_dir)
        print('Creating %s directory to save tensorboard logs' % (self.tf_summary_dir))
        self._create_dir(self.tf_summary_dir)

        self.input_data = None
        self.input_labels = None
        self.keep_prob = None
        self.layer_nodes = []  # list of layers of the final network
        self.last_out = None
        self.train_step = None
        self.cost = None
        self.verbose = 0

        # tensorflow objects
        self.tf_graph = tf.Graph()
        self.tf_session = None
        self.tf_saver = None
        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.tf_summary_writer_available = True

    def _create_dir(self, dirpath):

        """ 
        """
        
        try:
            os.makedirs(dirpath)
        except OSError, e:
            if e.errno != errno.EEXIST:
                raise

    def _initialize_tf_utilities_and_ops(self, restore_previous_model):

        """ Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
        Restore a previously trained model if the flag restore_previous_model is true.
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        """

        self.tf_merged_summaries = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()
        self.tf_saver = tf.train.Saver()

        self.tf_session.run(init_op)

        if restore_previous_model:
            self.tf_saver.restore(self.tf_session, self.model_path)

        # Retrieve run identifier
        run_id = 0
        for e in os.listdir(self.tf_summary_dir):
            if e[:3] == 'run':
                r = int(e[3:])
                if r > run_id:
                    run_id = r
        run_id += 1
        run_dir = os.path.join(self.tf_summary_dir, 'run' + str(run_id))
        print('Tensorboard logs dir for this run is %s' % (run_dir))

        self.tf_summary_writer = tf.train.SummaryWriter(run_dir, self.tf_session.graph)

    def _initialize_training_parameters(self, loss_func, learning_rate, num_epochs, batch_size,
                                        dataset, opt, dropout=1, momentum=None, l2reg=None):

        """ Initialize training parameters common to all models.
        :param loss_func: Loss function. ['mean_squared', 'cross_entropy']
        :param learning_rate: Initial learning rate
        :param num_epochs: Number of epochs
        :param batch_size: Size of each mini-batch
        :param dataset: Which dataset to use. ['mnist', 'cifar10', 'custom'].
        :param opt: Which tensorflow optimizer to use. ['gradient_descent', 'momentum', 'ada_grad']
        :param dropout: Dropout parameter
        :param momentum: Momentum parameter
        :param l2reg: regularization parameter
        :return: self
        """

        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dataset = dataset
        self.opt = opt
        self.momentum = momentum
        self.l2reg = l2reg

    def _run_unsupervised_validation_error_and_summaries(self, epoch, feed):

        """ Run the summaries and error computation on the validation set.
        :param epoch: current epoch
        :param validation_ref: validation reference data
        :return: self
        """

        try:
            result = self.tf_session.run([self.tf_merged_summaries, self.cost], feed_dict=feed)
            summary_str = result[0]
            err = result[1]
            self.tf_summary_writer.add_summary(summary_str, epoch)
        except tf.errors.InvalidArgumentError:
            if self.tf_summary_writer_available:
                print("Summary writer not available at the moment")
            self.tf_summary_writer_available = False
            err = self.tf_session.run(self.cost, feed_dict=feed)

        if self.verbose == 1:
            print("Reconstruction loss at step %s: %s" % (epoch, err))

    def _run_supervised_validation_error_and_summaries(self, epoch, feed):

        """ Run the summaries and error computation on the validation set.
        :param epoch: current epoch
        :param validation_set: validation data
        :return: self
        """

        try:
            result = self.tf_session.run([self.tf_merged_summaries, self.accuracy], feed_dict=feed)
            summary_str = result[0]
            acc = result[1]
            self.tf_summary_writer.add_summary(summary_str, epoch)
        except tf.errors.InvalidArgumentError:
            if self.tf_summary_writer_available:
                print("Summary writer not available at the moment")
            self.tf_summary_writer_available = False
            acc = self.tf_session.run(self.accuracy, feed_dict=feed)

        if self.verbose == 1:
            print("Accuracy at step %s: %s" % (epoch, acc))

    def predict(self, test_set):

        """ Predict the labels for the test set.
        :param test_set: Testing data. shape(n_test_samples, n_features)
        :return: labels
        """

        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                return self.model_predictions.eval({self.input_data: test_set,
                                                    self.keep_prob: 1})

    def _create_last_layer(self, last_layer, n_classes):

        """ Create the last layer for finetuning.
        :param last_layer: last layer output node
        :param n_classes: number of classes
        :return: self
        """

        with tf.name_scope("last_layer"):
            self.last_W = tf.Variable(tf.truncated_normal([last_layer.get_shape()[1].value, n_classes], stddev=0.1),
                                      name='sm-weigths')
            self.last_b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name='sm-biases')
            last_out = tf.matmul(last_layer, self.last_W) + self.last_b
            self.layer_nodes.append(last_out)
            self.last_out = last_out
            return last_out

    def _create_cost_function_node(self, model_output, ref_input, regterm=None):

        """ Create the cost function node.
        :param model_output: model output node
        :param ref_input: reference input placeholder node
        :param regterm: regularization term
        :return: self
        """

        with tf.name_scope("cost"):
            if self.loss_func == 'cross_entropy':
                cost = - tf.reduce_mean(ref_input * tf.log(tf.clip_by_value(model_output, 1e-10, float('inf'))) +
                                        (1 - ref_input) * tf.log(tf.clip_by_value(1 - model_output, 1e-10, float('inf'))))

            elif self.loss_func == 'softmax_cross_entropy':
                softmax = tf.nn.softmax(model_output)
                cost = - tf.reduce_mean(ref_input * tf.log(softmax) + (1 - ref_input) * tf.log(1 - softmax))

            elif self.loss_func == 'mean_squared':
                cost = tf.sqrt(tf.reduce_mean(tf.square(ref_input - model_output)))

            else:
                cost = None

        if cost is not None:
            self.cost = cost + regterm if regterm is not None else cost
            _ = tf.scalar_summary(self.loss_func, self.cost)
        else:
            self.cost = None

    def _create_train_step_node(self):

        """ Create the training step node of the network.
        :return: self
        """

        with tf.name_scope("train"):
            if self.opt == 'gradient_descent':
                self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt == 'ada_grad':
                self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt == 'momentum':
                self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cost)

            elif self.opt == 'adam':
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            else:
                self.train_step = None

    def _create_supervised_test_node(self):

        """ Create the supervised test node of the network.
        :return: self
        """

        with tf.name_scope("test"):
            self.model_predictions = tf.argmax(self.last_out, 1)
            correct_prediction = tf.equal(self.model_predictions, tf.argmax(self.input_labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            _ = tf.scalar_summary('accuracy', self.accuracy)
