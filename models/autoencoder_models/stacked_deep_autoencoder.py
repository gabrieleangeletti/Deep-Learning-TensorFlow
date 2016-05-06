import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from models.autoencoder_models import denoising_autoencoder
import model
from utils import utilities


class StackedDeepAutoencoder(model.Model):

    """ Implementation of Stacked Denoising Autoencoders for Unsupervised Learning using TensorFlow.
    The interface of the class is sklearn-like.
    """

    def __init__(self, layers, model_name='un_sdae', main_dir='un_sdae/', enc_act_func=list(['tanh']),
                 dec_act_func=list(['none']), loss_func=list(['mean_squared']), num_epochs=list([10]),
                 batch_size=list([10]), dataset='mnist', xavier_init=list([1]), opt=list(['gradient_descent']),
                 learning_rate=list([0.01]), momentum=list([0.5]),  dropout=1, corr_type='none', corr_frac=0.,
                 verbose=1, finetune_loss_func='cross_entropy', finetune_act_func='relu',
                 finetune_opt='gradient_descent', finetune_learning_rate=0.001, finetune_num_epochs=10,
                 finetune_batch_size=20, do_pretrain=True):
        """
        :param layers: list containing the hidden units for each layer
        :param enc_act_func: Activation function for the encoder. ['sigmoid', 'tanh']
        :param dec_act_func: Activation function for the decoder. ['sigmoid', 'tanh', 'none']
        :param finetune_loss_func: Loss function for the softmax layer. string, default ['cross_entropy', 'mean_squared']
        :param dropout: dropout parameter
        :param finetune_learning_rate: learning rate for the finetuning. float, default 0.001
        :param finetune_act_func: activation function for the finetuning phase
        :param finetune_opt: optimizer for the finetuning phase
        :param finetune_num_epochs: Number of epochs for the finetuning. int, default 20
        :param finetune_batch_size: Size of each mini-batch for the finetuning. int, default 20
        :param xavier_init: Value of the constant for xavier weights initialization. int, default 1
        :param corr_type: Type of input corruption. string, default 'none'. ["none", "masking", "salt_and_pepper"]
        :param corr_frac: Fraction of the input to corrupt. float, default 0.0
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy. int, default 0
        :param do_pretrain: True: uses variables from pretraining, False: initialize new variables.
        """
        model.Model.__init__(self, model_name, main_dir)

        self._initialize_training_parameters(loss_func, learning_rate, num_epochs, batch_size,
                                             dataset, opt, momentum)

        self.layers = layers
        self.do_pretrain = do_pretrain

        # Autoencoder parameters
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func
        self.xavier_init = xavier_init

        # Stacked Autoencoder parameters
        self.corr_type = corr_type
        self.corr_frac = corr_frac
        self.dropout = dropout
        self.finetune_loss_func = finetune_loss_func
        self.finetune_opt = finetune_opt
        self.finetune_learning_rate = finetune_learning_rate
        self.finetune_act_func = finetune_act_func
        self.finetune_num_epochs = finetune_num_epochs
        self.finetune_batch_size = finetune_batch_size
        self.verbose = verbose

        self.input_data = None
        self.input_ref = None

        self.keep_prob = None

        self.layer_nodes = []  # list of layers of the final network

        # Model parameters
        self.encoding_w_ = []  # list of matrices of encoding weights (one per layer)
        self.encoding_b_ = []  # list of arrays of encoding biases (one per layer)

        self.decoding_w = []  # list of matrices of decoding weights (one per layer)
        self.decoding_b = []  # list of arrays of decoding biases (one per layer)

        self.reconstruction = None

        self.autoencoders = []

        for l, layer in enumerate(layers):
            autoencoder_name = self.model_name + '-dae-' + str(l+1)
            self.autoencoders.append(denoising_autoencoder.DenoisingAutoencoder(
                n_components=layer, main_dir=self.main_dir, model_name=autoencoder_name,
                enc_act_func=self.enc_act_func[l], dec_act_func=self.dec_act_func[l], loss_func=self.loss_func[l],
                xavier_init=self.xavier_init[l], opt=self.opt[l], learning_rate=self.learning_rate[l],
                momentum=self.momentum[l], corr_type=self.corr_type, corr_frac=self.corr_frac,
                verbose=self.verbose, num_epochs=self.num_epochs[l], batch_size=self.batch_size[l],
                dataset=self.dataset))

    def pretrain(self, train_set, validation_set=None):

        """ Perform unsupervised pretraining of the stack of denoising autoencoders.
        :param train_set: training set
        :param validation_set: validation set
        :return: return data encoded by the last layer
        """

        next_train = train_set
        next_valid = validation_set

        for l, autoenc in enumerate(self.autoencoders):
            print('Training layer {}...'.format(l+1))
            next_train, next_valid = self._pretrain_autoencoder_and_gen_feed(autoenc, next_train, next_valid)

            # Reset tensorflow's default graph between different autoencoders
            ops.reset_default_graph()

        return next_train, next_valid

    def _pretrain_autoencoder_and_gen_feed(self, autoenc, train_set, validation_set):

        """ Pretrain a single autoencoder and encode the data for the next layer.
        :param autoenc: autoencoder reference
        :param train_set: training set
        :param validation_set: validation set
        :return: encoded train data, encoded validation data
        """

        autoenc.build_model(train_set.shape[1])
        autoenc.fit(train_set, validation_set)

        params = autoenc.get_model_parameters()

        self.encoding_w_.append(params['enc_w'])
        self.encoding_b_.append(params['enc_b'])

        next_train = autoenc.transform(train_set)
        next_valid = autoenc.transform(validation_set)

        return next_train, next_valid

    def fit(self, train_set, train_ref, validation_set=None, validation_ref=None, restore_previous_model=False):

        """ Fit the model to the data.
        :param train_set: Training data. shape(n_samples, n_features)
        :param train_ref: Reference data. shape(n_samples, n_features)
        :param validation_set: optional, default None. Validation data. shape(nval_samples, n_features)
        :param validation_ref: optional, default None. Reference validation data. shape(nval_samples, n_features)
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :return: self
        """

        print('Starting Reconstruction finetuning...')

        with tf.Session() as self.tf_session:
            self._initialize_tf_utilities_and_ops(restore_previous_model)
            self._train_model(train_set, train_ref, validation_set, validation_ref)
            self.tf_saver.save(self.tf_session, self.model_path)

    def _train_model(self, train_set, train_ref, validation_set, validation_ref):

        """ Train the model.
        :param train_set: training set
        :param train_ref: training reference data
        :param validation_set: validation set
        :param validation_ref: validation reference data
        :return: self
        """

        shuff = zip(train_set, train_ref)

        for i in range(self.finetune_num_epochs):

            np.random.shuffle(shuff)
            batches = [_ for _ in utilities.gen_batches(shuff, self.finetune_batch_size)]

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.tf_session.run(self.train_step, feed_dict={self.input_data: x_batch,
                                                                self.input_labels: y_batch,
                                                                self.keep_prob: self.dropout})

            if validation_set is not None:
                self._run_validation_error_and_summaries(i, validation_set, validation_ref)

    def _run_validation_error_and_summaries(self, epoch, validation_set, validation_ref):

        """ Run the summaries and error computation on the validation set.
        :param epoch: current epoch
        :param validation_ref: validation reference data
        :return: self
        """

        feed = {self.input_data: validation_set, self.input_labels: validation_ref, self.keep_prob: 1}
        result = self.tf_session.run([self.tf_merged_summaries, self.cost], feed_dict=feed)

        summary_str = result[0]
        acc = result[1]

        self.tf_summary_writer.add_summary(summary_str, epoch)

        if self.verbose == 1:
            print("Reconstruction loss at step %s: %s" % (epoch, acc))

    def get_layers_output(self, dataset):

        """ Get output from each layer of the network.
        :param dataset: input data
        :return: list of np array, element i in the list is the output of layer i
        """

        layers_out = []

        with tf.Session() as self.tf_session:
            self.tf_saver.restore(self.tf_session, self.model_path)
            for l in self.layer_nodes:
                layers_out.append(l.eval({self.input_data: dataset,
                                          self.keep_prob: 1}))
        return layers_out

    def reconstruct(self, test_set):

        """ Reconstruct the test set data using the learned model.
        :param test_set: Testing data. shape(n_test_samples, n_features)
        :return: labels
        """

        with tf.Session() as self.tf_session:
            self.tf_saver.restore(self.tf_session, self.model_path)
            return self.reconstruction.eval({self.input_data: test_set,
                                             self.keep_prob: 1})

    def compute_reconstruction_loss(self, test_set, test_ref):

        """ Compute the reconstruction loss over the test set.
        :param test_set: Testing data. shape(n_test_samples, n_features)
        :param test_ref: Reference test data. shape(n_test_samples, n_features)
        :return: reconstruction loss
        """

        with tf.Session() as self.tf_session:
            self.tf_saver.restore(self.tf_session, self.model_path)
            return self.cost.eval({self.input_data: test_set,
                                   self.input_labels: test_ref,
                                   self.keep_prob: 1})

    def build_model(self, n_features, encoding_w=None, encoding_b=None):

        """ Creates the computational graph for the reconstruction task.
        :param n_features: Number of features
        :param encoding_w: list of weights for the encoding layers.
        :param encoding_b: list of biases for the encoding layers.
        :return: self
        """

        self._create_placeholders(n_features, n_features)

        if encoding_w and encoding_b:
            self.encoding_w_ = encoding_w
            self.encoding_b_ = encoding_b
        else:
            self._create_variables(n_features)

        next_train = self._create_encoding_layers()
        self.reconstruction = self._create_decoding_layers(next_train)

        self._create_cost_function_node(self.finetune_loss_func, self.reconstruction, self.input_labels)
        self._create_train_step_node(self.finetune_opt, self.finetune_learning_rate, self.cost,
                                                       self.momentum)

    def _create_placeholders(self, n_features, n_classes):

        """ Create the TensorFlow placeholders for the model.
        :param n_features: number of features of the first layer
        :param n_classes: number of classes
        :return: self
        """

        self.input_data = tf.placeholder('float', [None, n_features], name='x-input')
        self.input_labels = tf.placeholder('float', [None, n_classes], name='y-input')
        self.keep_prob = tf.placeholder('float', name='keep-probs')

    def _create_variables(self, n_features):

        """ Create the TensorFlow variables for the model.
        :param n_features: number of features
        :return: self
        """

        if self.do_pretrain:
            self._create_variables_pretrain()
        else:
            self._create_variables_no_pretrain(n_features)

    def _create_variables_no_pretrain(self, n_features):

        """ Create model variables (no previous unsupervised pretraining)
        :param n_features: number of features
        :return: self
        """

        if not self.xavier_init[0]:
            xinit = 1
        else:
            xinit = self.xavier_init[0]

        self.encoding_w_ = []
        self.encoding_b_ = []

        for l, layer in enumerate(self.layers):

            if l == 0:
                self.encoding_w_.append(tf.Variable(utilities.xavier_init(n_features, self.layers[l], xinit)))
                self.encoding_b_.append(tf.Variable(tf.truncated_normal([self.layers[l]], stddev=0.01)))
            else:
                self.encoding_w_.append(tf.Variable(utilities.xavier_init(self.layers[l-1], self.layers[l], xinit)))
                self.encoding_b_.append(tf.Variable(tf.truncated_normal([self.layers[l]], stddev=0.01)))

    def _create_variables_pretrain(self):

        """ Create model variables (previous unsupervised pretraining)
        :return: self
        """

        for l, layer in enumerate(self.layers):
            self.encoding_w_[l] = tf.Variable(self.encoding_w_[l], name='enc-w-{}'.format(l))
            self.encoding_b_[l] = tf.Variable(self.encoding_b_[l], name='enc-b-{}'.format(l))

    def _create_encoding_layers(self):

        """ Create the encoding layers for supervised finetuning.
        :return: output of the final encoding layer.
        """

        next_train = self.input_data
        self.layer_nodes = []

        for l, layer in enumerate(self.layers):

            with tf.name_scope("encode-{}".format(l)):

                y_act = tf.matmul(next_train, self.encoding_w_[l]) + self.encoding_b_[l]

                if self.finetune_act_func == 'sigmoid':
                    layer_y = tf.nn.sigmoid(y_act)

                elif self.finetune_act_func == 'tanh':
                    layer_y = tf.nn.tanh(y_act)

                elif self.finetune_act_func == 'relu':
                    layer_y = tf.nn.relu(y_act)

                else:
                    layer_y = None

                # the input to the next layer is the output of this layer
                next_train = tf.nn.dropout(layer_y, self.keep_prob)

            self.layer_nodes.append(next_train)

        return next_train

    def _create_decoding_layers(self, last_encode):

        """ Create the decoding layers for reconstruction finetuning.
        :param last_encode: output of the last encoding layer
        :return: output of the final encoding layer.
        """

        next_decode = last_encode

        for l, layer in reversed(list(enumerate(self.layers))):

            with tf.name_scope("decode-{}".format(l)):

                # Create decoding variables
                dec_w = tf.Variable(tf.transpose(self.encoding_w_[l].initialized_value()))
                dec_b = tf.Variable(tf.constant(0.01, shape=[dec_w.get_shape().dims[1].value]))
                self.decoding_w.append(dec_w)
                self.decoding_b.append(dec_b)

                y_act = tf.matmul(next_decode, dec_w) + dec_b

                if self.finetune_act_func == 'sigmoid':
                    layer_y = tf.nn.sigmoid(y_act)

                elif self.finetune_act_func == 'tanh':
                    layer_y = tf.nn.tanh(y_act)

                elif self.finetune_act_func == 'relu':
                    layer_y = tf.nn.relu(y_act)

                else:
                    layer_y = None

                # the input to the next layer is the output of this layer
                next_decode = tf.nn.dropout(layer_y, self.keep_prob)

            self.layer_nodes.append(next_decode)

        return next_decode
