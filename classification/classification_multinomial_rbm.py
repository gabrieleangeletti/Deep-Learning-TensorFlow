from abstract_classification_rbm import AbstractClsRBM

__author__ = 'blackecho'


class ClsMultiRBM(AbstractClsRBM):

    def __init__(self,
                 num_visible,
                 num_hidden,
                 k_visible,
                 k_hidden,
                 *args,
                 **kwargs):

        super(ClsMultiRBM, self).__init__(num_visible,
                                          num_hidden,
                                          AbstractClsRBM.rbm_type.mrbm,
                                          k_visible=k_visible,
                                          k_hidden=k_hidden,
                                          *args,
                                          **kwargs)

    def learn_unsupervised_features(self,
                                    data,
                                    validation=None,
                                    max_epochs=100,
                                    batch_size=1,
                                    alpha=0.1,
                                    m=0.5,
                                    gibbs_k=1,
                                    alpha_update_rule='constant',
                                    verbose=False,
                                    display=None):
        """Unsupervised learning of the Multinomial rbm
        """
        self.rbm.train(data,
                       validation=validation,
                       max_epochs=max_epochs,
                       batch_size=batch_size,
                       alpha=alpha,
                       m=m,
                       gibbs_k=gibbs_k,
                       verbose=verbose,
                       display=display)

    def fit_logistic_cls(self, data, labels):
        """Train a logistic regression classifier on top of the learned features
        by the Multinomial Restricted Boltzmann Machine.
        """
        (data_probs, data_states) = self.rbm.sample_hidden_from_visible(data, gibbs_k=1)
        self.cls.fit(data_probs, labels)

    def predict_logistic_cls(self, data):
        """Predict the labels for data using the Logistic Regression layer on top of the
        learned features by the Multinomial Restricted Boltzmann Machine.
        """
        (data_probs, data_states) = self.rbm.sample_hidden_from_visible(data, gibbs_k=1)
        return self.cls.predict(data_probs)

