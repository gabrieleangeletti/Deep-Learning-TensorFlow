import tensorflow as tf

import model


class SupervisedModel(model.Model):

    def __init__(self, model_name, main_dir, models_dir, data_dir, summary_dir):
        model.Model.__init__(self, model_name, main_dir, models_dir, data_dir, summary_dir)

    def fit(self, train_set, train_labels, validation_set=None, validation_labels=None, restore_previous_model=False, graph=None):

        """ Fit the model to the data.
        :param train_set: Training data. shape(n_samples, n_features)
        :param train_labels: Training labels. shape(n_samples, n_classes)
        :param validation_set: optional, default None. Validation data. shape(nval_samples, n_features)
        :param validation_labels: optional, default None. Validation labels. shape(nval_samples, n_classes)
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :param graph: tensorflow graph object
        :return: self
        """

        if len(train_labels.shape) != 1:
            num_classes = train_labels.shape[1]
        else:
            raise Exception("Please convert the labels with one-hot encoding.")

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            self.build_model(train_set.shape[1], num_classes)
            with tf.Session() as self.tf_session:
                self._initialize_tf_utilities_and_ops(restore_previous_model)
                self._train_model(train_set, train_labels, validation_set, validation_labels)
                self.tf_saver.save(self.tf_session, self.model_path)

    def build_model(self, num_features, num_classes):
        pass

    def _train_model(self, train_set, train_labels, validation_set, validation_labels):
        pass

    def _run_validation_error_and_summaries(self, epoch, feed):

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

    def compute_accuracy(self, test_set, test_labels):

        """ Compute the accuracy over the test set.
        :param test_set: Testing data. shape(n_test_samples, n_features)
        :param test_labels: Labels for the test data. shape(n_test_samples, n_classes)
        :return: accuracy
        """

        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                return self.accuracy.eval({self.input_data: test_set,
                                           self.input_labels: test_labels,
                                           self.keep_prob: 1})

    def _create_accuracy_test_node(self):

        """ Create the supervised test node of the network.
        :return: self
        """

        with tf.name_scope("test"):
            self.model_predictions = tf.argmax(self.last_out, 1)
            correct_prediction = tf.equal(self.model_predictions, tf.argmax(self.input_labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            _ = tf.scalar_summary('accuracy', self.accuracy)
