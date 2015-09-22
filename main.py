from __future__ import print_function

import numpy as np
import sys

from sklearn.linear_model import LogisticRegression

from mnist import MNIST
import config
import utils
from classification import *
from dbn import DBN
import multinomial_rbm
import gaussian_rbm
import rbm

if __name__ == '__main__':
    assert config.BATCH_SIZE <= config.TRAIN_SET_SIZE
    # load MNIST dataset
    print('Initializing MNIST dataset...')
    mndata = MNIST('mnist')
    mndata.load_training()
    mndata.load_testing()
    display = mndata.display
    # load data into numpy
    randperm = np.random.permutation(config.TRAIN_SET_SIZE)
    randperm_test = np.random.permutation(config.TEST_SET_SIZE)
    mndata_images = np.array(mndata.train_images)
    mndata_labels = np.array(mndata.train_labels)
    test_images = np.array(mndata.test_images)
    test_labels = np.array(mndata.test_labels)
    X = mndata_images[randperm]
    X_test = test_images[randperm_test]
    y = mndata_labels[randperm]
    y_test = test_labels[randperm_test]
    # normalize dataset to be binary valued
    X_norm = utils.normalize_dataset_to_binary(X)
    X_norm_test = utils.normalize_dataset_to_binary(X_test)
    # normalize dataset to be real valued
    X_real = utils.normalize_dataset(X)
    X_real_test = utils.normalize_dataset(X_test)
    # Type of learning machines to train
    if len(sys.argv) > 1:
        run_type = sys.argv[1].split('=')[1]
    else:
        # default type is standard
        run_type = 'standard'

    # #####################################
    # Standard Restricted Boltzmann Machine
    # #####################################
    if run_type == 'standard':
        # create rbm
        r = rbm.RBM(config.NV, config.NH)
        print('Begin Training...')
        r.train(X_norm,
                validation=X_norm_test[0:config.BATCH_SIZE],
                epochs=config.EPOCHS,
                alpha=config.ALPHA,
                m=config.M,
                batch_size=config.BATCH_SIZE,
                gibbs_k=config.GIBBS_K,
                alpha_update_rule=config.ALPHA_UPDATE_RULE,
                verbose=config.VERBOSE,
                display=display)
        # save the rbm to a file
        print('Saving the RBM to outfile...')
        r.save_configuration(config.OUTFILE)
        # print('Reading the mind of the RBM...')
        # _, mind = r.fantasy(config.FANTASY_K)
        # print(display(mind[0]))
        print('Saving image of random hidden unit weights to outfile...')
        r.save_weights_images(config.HS,
                              config.WIDTH,
                              config.HEIGHT,
                              config.W_OUTFILE)

    # ########################################
    # Multinomial Restricted Boltzmann Machine
    # ########################################
    if run_type == 'multinomial':
        # discretization of data
        X_mu = utils.discretize_dataset(X, config.MULTI_KV)
        X_mu_test = utils.discretize_dataset(X_test, config.MULTI_KV)
        # create multinomial rbm
        mr = multinomial_rbm.MultinomialRBM(config.MULTI_NV, config.MULTI_NH, config.MULTI_KV, config.MULTI_KN)
        mr.train(X_mu,
                 validation=X_mu_test[0:config.M_BATCH_SIZE],
                 epochs=config.M_EPOCHS,
                 alpha=config.M_ALPHA,
                 m=config.M_M,
                 batch_size=config.M_BATCH_SIZE,
                 gibbs_k=config.M_GIBBS_K,
                 verbose=config.M_VERBOSE,
                 display=display)
        # save the rbm to a file
        print('Saving the Multinomial RBM to outfile...')
        mr.save_configuration(config.M_OUTFILE)

    # ###############################################
    # Gaussian-Bernoulli Restricted Boltzmann Machine
    # ###############################################
    if run_type == 'gaussian':
        # create gaussian rbm
        gr = gaussian_rbm.GaussianRBM(config.GAUSS_NV, config.GAUSS_NH)
        print('Begin Training...')
        gr.train(X_real,
                 validation=X_real_test[0:config.G_BATCH_SIZE],
                 epochs=config.G_EPOCHS,
                 alpha=config.G_ALPHA,
                 m=config.G_M,
                 batch_size=config.G_BATCH_SIZE,
                 gibbs_k=config.G_GIBBS_K,
                 alpha_update_rule=config.G_ALPHA_UPDATE_RULE,
                 verbose=config.G_VERBOSE,
                 display=display)
        # save the rbm to a file
        print('Saving the Gaussian RBM to outfile...')
        gr.save_configuration(config.G_OUTFILE)

    # #################################################
    # Classification rbm vs Logistic Regression
    # #################################################
    elif run_type == 'rbm-vs-logistic':
        cst = classification_rbm.ClsRBM(config.NV, config.NH)
        # unsupervised learning of features
        print('Starting unsupervised learning of the features...')
        cst.learn_unsupervised_features(X_norm,
                                        validation=X_norm_test[0:config.BATCH_SIZE],
                                        epochs=config.EPOCHS,
                                        batch_size=config.BATCH_SIZE,
                                        alpha=config.ALPHA,
                                        m=config.M,
                                        gibbs_k=config.GIBBS_K,
                                        alpha_update_rule=config.ALPHA_UPDATE_RULE,
                                        verbose=config.VERBOSE,
                                        display=display)

        # save the standard rbm to a file
        print('Saving the RBM to outfile...')
        cst.rbm.save_configuration(config.OUTFILE)
        print('Saving image of random hidden unit weights to outfile...')
        cst.rbm.save_weights_images(config.HS,
                                    config.WIDTH,
                                    config.HEIGHT,
                                    config.W_OUTFILE)
        # fit the Logistic Regression layer
        print('Fitting the Logistic Regression layer...')
        cst.fit_logistic_cls(X_norm, y)
        # sample the test set
        print('Testing the accuracy of the classifier...')
        # test the predictions of the LR layer
        preds_st = cst.predict_logistic_cls(X_norm_test)

        accuracy_st = sum(preds_st == y_test) / float(config.TEST_SET_SIZE)
        # Now train a normal logistic regression classifier and test it
        print('Training standard Logistic Regression Classifier...')
        lr_cls = LogisticRegression()
        lr_cls.fit(X, y)
        lr_cls_preds = lr_cls.predict(X_test)
        accuracy_lr = sum(lr_cls_preds == y_test) / float(config.TEST_SET_SIZE)
        print('Accuracy of the RBM classifier: %s' % accuracy_st)
        print('Accuracy of the Logistic classifier: %s' % accuracy_lr)

    # #####################################################
    # Classification Multinomial rbm vs Logistic Regression
    # #####################################################
    elif run_type == 'mrbm-vs-logistic':
        # discretization of data
        X_mu = utils.discretize_dataset(X, config.MULTI_KV)
        X_mu_test = utils.discretize_dataset(X_test, config.MULTI_KV)
        # create multinomial rbm
        csm = classification_multinomial_rbm.ClsMultiRBM(config.MULTI_NV, config.MULTI_NH, config.MULTI_KV, config.MULTI_KN)
        # unsupervised learning of features
        print('Starting unsupervised learning of the features...')
        csm.learn_unsupervised_features(X_mu,
                                        validation=X_mu_test[0:config.M_BATCH_SIZE],
                                        epochs=config.M_EPOCHS,
                                        alpha=config.M_ALPHA,
                                        m=config.M_M,
                                        batch_size=config.M_BATCH_SIZE,
                                        gibbs_k=config.M_GIBBS_K,
                                        verbose=config.M_VERBOSE,
                                        display=display)
        print('Saving the Multi RBM to outfile...')
        csm.rbm.save_configuration(config.M_OUTFILE)
        # fit the Logistic Regression layer
        print('Fitting the Logistic Regression layer...')
        csm.fit_logistic_cls(X_mu, y)
        # sample the test set
        print('Testing the accuracy of the classifier...')
        # test the predictions of the LR layer
        preds_m = csm.predict_logistic_cls(X_mu_test)

        accuracy_m = sum(preds_m == y_test) / float(config.TEST_SET_SIZE)
        # Now train a normal logistic regression classifier and test it
        print('Training standard Logistic Regression Classifier...')
        lr_cls = LogisticRegression()
        lr_cls.fit(X_mu, y)
        lr_cls_preds = lr_cls.predict(X_mu_test)
        accuracy_lr = sum(lr_cls_preds == y_test) / float(config.TEST_SET_SIZE)
        print('Accuracy of the RBM classifier: %s' % accuracy_m)
        print('Accuracy of the Logistic classifier: %s' % accuracy_lr)

    # ##################################################
    # Classification Gaussian rbm vs Logistic Regression
    # ##################################################
    elif run_type == 'grbm-vs-logistic':
        csg = classification_gaussian_rbm.ClsGaussianRBM(config.GAUSS_NV, config.GAUSS_NH)
        # unsupervised learning of features
        print('Starting unsupervised learning of the features...')
        csg.learn_unsupervised_features(X_real,
                                        validation=X_real_test[0:config.G_BATCH_SIZE],
                                        epochs=config.G_EPOCHS,
                                        batch_size=config.G_BATCH_SIZE,
                                        alpha=config.G_ALPHA,
                                        m=config.G_M,
                                        gibbs_k=config.G_GIBBS_K,
                                        alpha_update_rule=config.G_ALPHA_UPDATE_RULE,
                                        verbose=config.G_VERBOSE,
                                        display=display)

        # save the standard rbm to a file
        print('Saving the GRBM to outfile...')
        csg.rbm.save_configuration(config.G_OUTFILE)
        # fit the Logistic Regression layer
        print('Fitting the Logistic Regression layer...')
        csg.fit_logistic_cls(X_real, y)
        # sample the test set
        print('Testing the accuracy of the classifier...')
        # test the predictions of the LR layer
        preds_st = csg.predict_logistic_cls(X_real_test)

        accuracy_st = sum(preds_st == y_test) / float(config.TEST_SET_SIZE)
        # Now train a normal logistic regression classifier and test it
        print('Training standard Logistic Regression Classifier...')
        lr_cls = LogisticRegression()
        lr_cls.fit(X_real, y)
        lr_cls_preds = lr_cls.predict(X_real_test)
        accuracy_lr = sum(lr_cls_preds == y_test) / float(config.TEST_SET_SIZE)
        print('Accuracy of the Gaussian RBM classifier: %s' % accuracy_st)
        print('Accuracy of the Logistic classifier: %s' % accuracy_lr)

    # ##################################################
    # Deep Belief Network
    # ##################################################
    elif run_type == 'dbn-unsupervised':
        deep_net = DBN(config.DBN_LAYERS)
        # Unsupervised greedy layer-wise pre-training of the net
        print('Start unsupervised greedy layer-wise training...')
        deep_net.unsupervised_pretrain(X_norm,
                                       validation=X_norm_test[0:config.DBN_BATCH_SIZE],
                                       epochs=config.DBN_EPOCHS,
                                       alpha=config.DBN_ALPHA,
                                       m=config.DBN_M,
                                       batch_size=config.DBN_BATCH_SIZE,
                                       gibbs_k=config.DBN_GIBBS_K,
                                       alpha_update_rule=config.DBN_ALPHA_UPDATE_RULE,
                                       verbose=config.DBN_VERBOSE,
                                       display=display)
        print('Start supervised fine tuning using wake-sleep algorithm...')
        deep_net.wake_sleep(config.DBN_LAST_LAYER,
                            X_norm,
                            y,
                            config.DBN_FT_BATCH_SIZE,
                            config.DBN_FT_EPOCHS,
                            config.DBN_FT_ALPHA,
                            config.DBN_FT_TOP_GIBBS_K,
                            config.DBN_FT_ALPHA_UPDATE_RULE)

    # ##################################################
    # Deep Belief Network loaded from pretrained rbms
    # ##################################################
    elif run_type == 'dbn-from-trained-rbms':
        deep_net = DBN([])
        deep_net.load_rbms(config.DBN_INPUT_RBMS)
        print('Start supervised fine tuning using wake-sleep algorithm...')
        deep_net.wake_sleep(config.DBN_LAST_LAYER,
                            X_norm,
                            y,
                            config.DBN_FT_BATCH_SIZE,
                            config.DBN_FT_EPOCHS,
                            config.DBN_FT_ALPHA,
                            config.DBN_FT_TOP_GIBBS_K,
                            config.DBN_FT_ALPHA_UPDATE_RULE)
        print('Save configuration of rbm layers after training')
        deep_net.layers[0].save_configuration(config.DBN_OUTFILES[0])
        deep_net.layers[1].save_configuration(config.DBN_OUTFILES[1])
        deep_net.last_rbm.save_configuration(config.DBN_OUTFILES[2])
        print('Save performance metrics of the rbm during wake sleep...')
        deep_net.save_performance_metrics(config.DBN_PERFORMANCE_OUTFILE)
        print('Testing the accuracy of the dbn...')
        dbn_preds = deep_net.predict_ws(X_norm_test, config.DBN_TEST_TOP_GIBBS_K)

        accuracy_dbn = sum(dbn_preds == y_test) / float(config.TEST_SET_SIZE)
        # Now train a normal logistic regression classifier and test it
        print('Training standard Logistic Regression Classifier...')
        lr = LogisticRegression()
        lr.fit(X_norm, y)
        lr_preds = lr.predict(X_norm_test)
        accuracy_lr = sum(lr_preds == y_test) / float(config.TEST_SET_SIZE)
        print('Accuracy of the Deep Belief Network: %s' % accuracy_dbn)
        print('Accuracy of the Logistic classifier: %s' % accuracy_lr)

    # ##################################################
    # Deep Belief Network vs Logistic Regression
    # ##################################################
    elif run_type == 'dbn-vs-logistic':
        deep_net = DBN(config.DBN_LAYERS)
        # Unsupervised greedy layer-wise pre-training of the net
        deep_net.unsupervised_pretrain(X_norm,
                                       validation=X_norm_test[0:config.DBN_BATCH_SIZE],
                                       epochs=config.DBN_EPOCHS,
                                       alpha=config.DBN_ALPHA,
                                       m=config.DBN_M,
                                       batch_size=config.DBN_BATCH_SIZE,
                                       gibbs_k=config.DBN_GIBBS_K,
                                       alpha_update_rule=config.DBN_ALPHA_UPDATE_RULE,
                                       verbose=config.DBN_VERBOSE,
                                       display=display)

        # fit the Logistic Regression layer
        print('Fitting the Logistic Regression layer...')
        deep_net.fit_cls(X_norm, y)
        # sample the test set
        print('Testing the accuracy of the classifier...')
        # test the predictions of the LR layer
        preds_st = deep_net.predict_cls(X_norm_test)

        accuracy_st = sum(preds_st == y_test) / float(config.TEST_SET_SIZE)
        # Now train a normal logistic regression classifier and test it
        print('Training standard Logistic Regression Classifier...')
        lr_cls = LogisticRegression()
        lr_cls.fit(X_norm, y)
        lr_cls_preds = lr_cls.predict(X_norm_test)
        accuracy_lr = sum(lr_cls_preds == y_test) / float(config.TEST_SET_SIZE)
        print('Accuracy of the Deep Belief Network: %s' % accuracy_st)
        print('Accuracy of the Logistic classifier: %s' % accuracy_lr)

    # ##################################################
    # Temporary, development or debug executions
    # ##################################################
    elif run_type == 'tmp':
        print('Sampling from the first RBM')
        r = rbm.RBM(1, 1)
        r.load_configuration('models/rbm.json')
        _, data_repr = r.sample_hidden_from_visible(X_norm)
        _, data_val_repr = r.sample_hidden_from_visible(X_norm_test)

        print('Fitting the Logistic Regression layer of layer 1...')
        cls1 = LogisticRegression()
        cls1.fit(data_repr, y)
        preds_st1 = cls1.predict(data_val_repr)
        accuracy_st1 = sum(preds_st1 == y_test) / float(config.TEST_SET_SIZE)

        r2 = rbm.RBM(1, 1)
        r2.load_configuration('models/rbm_layer2.json')

        # fit the Logistic Regression layer
        print('Fitting the Logistic Regression layer of layer 2...')
        (data_probs, data_states) = r2.sample_hidden_from_visible(data_repr, gibbs_k=1)
        cls = LogisticRegression()
        cls.fit(data_states, y)
        # sample the test set
        print('Testing the accuracy of the classifier...')
        # test the predictions of the LR layer
        (test_probs, test_states) = r2.sample_hidden_from_visible(data_val_repr, gibbs_k=1)
        preds_st = cls.predict(test_states)

        accuracy_st = sum(preds_st == y_test) / float(config.TEST_SET_SIZE)
        # Now train a normal logistic regression classifier and test it
        print('Training standard Logistic Regression Classifier...')
        lr_cls = LogisticRegression()
        lr_cls.fit(X, y)
        lr_cls_preds = lr_cls.predict(X_test)
        accuracy_lr = sum(lr_cls_preds == y_test) / float(config.TEST_SET_SIZE)

        print('Accuracy of the RBM layer 1 classifier: %s' % accuracy_st1)
        print('Accuracy of the RBM layer 2 classifier: %s' % accuracy_st)
        print('Accuracy of the Logistic classifier: %s' % accuracy_lr)
