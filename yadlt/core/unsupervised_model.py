import tensorflow as tf

import model


class UnsupervisedModel(model.Model):

    def __init__(self, model_name, main_dir, models_dir, data_dir, summary_dir):
        model.Model.__init__(self, model_name, main_dir, models_dir, data_dir, summary_dir)

        self.encode = None
        self.reconstruction = None

    def fit(self, train_set, train_ref, validation_set=None, validation_ref=None, restore_previous_model=False, graph=None):

        """ Fit the model to the data.
        :param train_set: Training data. shape(n_samples, n_features)
        :param train_ref: Reference data. shape(n_samples, n_features)
        :param validation_set: optional, default None. Validation data. shape(nval_samples, n_features)
        :param validation_ref: optional, default None. Reference validation data. shape(nval_samples, n_features)
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :param graph: tensorflow graph object
        :return: self
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            self.build_model(train_set.shape[1])
            with tf.Session() as self.tf_session:
                self._initialize_tf_utilities_and_ops(restore_previous_model)
                self._train_model(train_set, train_ref, validation_set, validation_ref)
                self.tf_saver.save(self.tf_session, self.model_path)

    def build_model(self, num_features):
        pass

    def _train_model(self, train_set, train_labels, validation_set, validation_labels):
        pass

    def _run_validation_error_and_summaries(self, epoch, feed):

        """ Run the summaries and error computation on the validation set.
        :param epoch: current epoch
        :param feed: feed dictionary
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

    def transform(self, data, graph=None):

        """ Transform data according to the model.
        :param data: Data to transform
        :param graph: tf graph object
        :return: transformed data
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                encoded_data = self.encode.eval({self.input_data: data, self.keep_prob: 1})
                return encoded_data

    def reconstruct(self, data, graph=None):

        """ Reconstruct the test set data using the learned model.
        :param data: Data to reconstruct
        :graph: tf graph object
        :return: labels
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                return self.reconstruction.eval({self.input_data: data, self.keep_prob: 1})

    def compute_reconstruction_loss(self, data, data_ref, graph=None):

        """ Compute the reconstruction loss over the test set.
        :param data: Data to reconstruct
        :param data_ref: Reference data.
        :return: reconstruction loss
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                return self.cost.eval({self.input_data: data, self.input_labels: data_ref, self.keep_prob: 1})

