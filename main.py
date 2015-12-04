from __future__ import print_function

import numpy as np

from mnist import MNIST
import config
import utils
from classification import *
from dbn import DBN
import multinomial_rbm
import gaussian_rbm
import rbm

import argparse
import os


def main():

    # ###################### #
    # Command Line Arguments #
    # ###################### #
    parser = argparse.ArgumentParser(description='Execute RBMs and DBNs on the MNIST digits dataset.')

    parser.add_argument('-n', metavar='Neural Network type', type=str, required=True,
                        help='the kind of NN to run: rbm (binary), mrbm (multinomial), grbm (gaussian),'
                             'crbm (classification rbm), cmrbm (class. multinomial), cgrbm (class. gaussian)'
                             'dbn (deep net)...')

    parser.add_argument('-o', metavar='output file', type=str,
                        help='output file to store the configuration of trained NNs. If a file exists,'
                             'it will be overwritten')

    args = parser.parse_args()

    # Retrieve arguments from command line
    nn = args.n
    outfile = args.o

    # check if outfile already exists, and delete it
    if outfile:
        if os.path.isfile(outfile):
            print('Output file already exists, removing...')
            os.remove(outfile)

    assert nn in ['rbm', 'mrbm', 'grbm', 'crbm', 'cmrbm', 'cgrbm', 'dbn']

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

    # #####################################
    # Binary Restricted Boltzmann Machine
    # #####################################
    if nn == 'rbm':
        # create rbm
        r = rbm.RBM(config.NV, config.NH)
        print('Begin Training...')
        r.train(X_norm,
                validation=X_norm_test[0:config.BATCH_SIZE],
                epochs=config.EPOCHS,
                alpha=config.ALPHA,
                momentum=config.M,
                batch_size=config.BATCH_SIZE,
                gibbs_k=config.GIBBS_K,
                alpha_update_rule=config.ALPHA_UPDATE_RULE,
                momentum_update_rule=config.MOMENTUM_UPDATE_RULE,
                verbose=config.VERBOSE,
                display=display)

        if outfile:
            # save the rbm to a file
            print('Saving the RBM to outfile...')
            r.save_configuration(outfile)

    # ########################################
    # Multinomial Restricted Boltzmann Machine
    # ########################################
    elif nn == 'mrbm':
        # discretization of data
        X_mu = utils.discretize_dataset(X, config.MULTI_KV)
        X_mu_test = utils.discretize_dataset(X_test, config.MULTI_KV)
        # create multinomial rbm
        mr = multinomial_rbm.MultinomialRBM(config.MULTI_NV, config.MULTI_NH, config.MULTI_KV, config.MULTI_KN)
        mr.train(X_mu,
                 validation=X_mu_test[0:config.M_BATCH_SIZE],
                 epochs=config.M_EPOCHS,
                 alpha=config.M_ALPHA,
                 momentum=config.M_M,
                 batch_size=config.M_BATCH_SIZE,
                 alpha_update_rule=config.M_ALPHA_UPDATE_RULE,
                 momentum_update_rule=config.M_MOMENTUM_UPDATE_RULE,
                 gibbs_k=config.M_GIBBS_K,
                 verbose=config.M_VERBOSE,
                 display=display)

        if outfile:
            # save the rbm to a file
            print('Saving the Multinomial RBM to outfile...')
            mr.save_configuration(outfile)

    # ###############################################
    # Gaussian-Bernoulli Restricted Boltzmann Machine
    # ###############################################
    elif nn == 'grbm':
        # create gaussian rbm
        gr = gaussian_rbm.GaussianRBM(config.GAUSS_NV, config.GAUSS_NH)
        print('Begin Training...')
        gr.train(X_real,
                 validation=X_real_test[0:config.G_BATCH_SIZE],
                 epochs=config.G_EPOCHS,
                 alpha=config.G_ALPHA,
                 momentum=config.G_M,
                 batch_size=config.G_BATCH_SIZE,
                 gibbs_k=config.G_GIBBS_K,
                 alpha_update_rule=config.G_ALPHA_UPDATE_RULE,
                 momentum_update_rule=config.G_MOMENTUM_UPDATE_RULE,
                 verbose=config.G_VERBOSE,
                 display=display)
        # save the rbm to a file
        print('Saving the Gaussian RBM to outfile...')
        gr.save_configuration(outfile)

    # ###################
    # Classification rbm
    # ###################
    elif nn == 'crbm':
        cst = classification_rbm.ClsRBM(config.NV, config.NH)
        # unsupervised learning of features
        print('Starting unsupervised learning of the features...')
        cst.learn_unsupervised_features(X_norm,
                                        validation=X_norm_test[0:config.BATCH_SIZE],
                                        epochs=config.EPOCHS,
                                        batch_size=config.BATCH_SIZE,
                                        alpha=config.ALPHA,
                                        momentum=config.M,
                                        gibbs_k=config.GIBBS_K,
                                        alpha_update_rule=config.ALPHA_UPDATE_RULE,
                                        momentum_update_rule=config.MOMENTUM_UPDATE_RULE,
                                        verbose=config.VERBOSE,
                                        display=display)
        if outfile:
            # save the standard rbm to a file
            print('Saving the RBM to outfile...')
            cst.rbm.save_configuration(outfile)

        # fit the Logistic Regression layer
        print('Fitting the Logistic Regression layer...')
        cst.fit_logistic_cls(X_norm, y)
        # sample the test set
        print('Testing the accuracy of the classifier...')
        # test the predictions of the LR layer
        preds_st = cst.predict_logistic_cls(X_norm_test)

        accuracy_st = sum(preds_st == y_test) / float(config.TEST_SET_SIZE)
        print('Accuracy of the RBM classifier: %s' % accuracy_st)

    # ###############################
    # Classification Multinomial rbm
    # ###############################
    elif nn == 'cmrbm':
        # discretization of data
        X_mu = utils.discretize_dataset(X, config.MULTI_KV)
        X_mu_test = utils.discretize_dataset(X_test, config.MULTI_KV)
        # create multinomial rbm
        csm = classification_multinomial_rbm.ClsMultiRBM(config.MULTI_NV,
                                                         config.MULTI_NH,
                                                         config.MULTI_KV,
                                                         config.MULTI_KN)
        # unsupervised learning of features
        print('Starting unsupervised learning of the features...')
        csm.learn_unsupervised_features(X_mu,
                                        validation=X_mu_test[0:config.M_BATCH_SIZE],
                                        epochs=config.M_EPOCHS,
                                        alpha=config.M_ALPHA,
                                        momentum=config.M_M,
                                        batch_size=config.M_BATCH_SIZE,
                                        alpha_update_rule=config.M_ALPHA_UPDATE_RULE,
                                        momentum_update_rule=config.M_MOMENTUM_UPDATE_RULE,
                                        gibbs_k=config.M_GIBBS_K,
                                        verbose=config.M_VERBOSE,
                                        display=display)

        if outfile:
            print('Saving the Multi RBM to outfile...')
            csm.rbm.save_configuration(outfile)

        # fit the Logistic Regression layer
        print('Fitting the Logistic Regression layer...')
        csm.fit_logistic_cls(X_mu, y)
        # sample the test set
        print('Testing the accuracy of the classifier...')
        # test the predictions of the LR layer
        preds_m = csm.predict_logistic_cls(X_mu_test)

        accuracy_m = sum(preds_m == y_test) / float(config.TEST_SET_SIZE)
        print('Accuracy of the RBM classifier: %s' % accuracy_m)

    # ############################
    # Classification Gaussian rbm
    # ############################
    elif nn == 'cgrbm':
        csg = classification_gaussian_rbm.ClsGaussianRBM(config.GAUSS_NV, config.GAUSS_NH)
        # unsupervised learning of features
        print('Starting unsupervised learning of the features...')
        csg.learn_unsupervised_features(X_real,
                                        validation=X_real_test[0:config.G_BATCH_SIZE],
                                        epochs=config.G_EPOCHS,
                                        batch_size=config.G_BATCH_SIZE,
                                        alpha=config.G_ALPHA,
                                        momentum=config.G_M,
                                        gibbs_k=config.G_GIBBS_K,
                                        alpha_update_rule=config.G_ALPHA_UPDATE_RULE,
                                        momentum_update_rule=config.G_MOMENTUM_UPDATE_RULE,
                                        verbose=config.G_VERBOSE,
                                        display=display)

        if outfile:
            # save the standard rbm to a file
            print('Saving the GRBM to outfile...')
            csg.rbm.save_configuration(outfile)

        # fit the Logistic Regression layer
        print('Fitting the Logistic Regression layer...')
        csg.fit_logistic_cls(X_real, y)
        # sample the test set
        print('Testing the accuracy of the classifier...')
        # test the predictions of the LR layer
        preds_st = csg.predict_logistic_cls(X_real_test)

        accuracy_st = sum(preds_st == y_test) / float(config.TEST_SET_SIZE)
        print('Accuracy of the Gaussian RBM classifier: %s' % accuracy_st)

    # #####################
    # Deep Belief Network
    # #####################
    elif nn == 'dbn':
        deep_net = DBN(config.DBN_LAYERS)
        # Unsupervised greedy layer-wise pre-training of the net
        print('Start unsupervised greedy layer-wise training...')
        deep_net.unsupervised_pretrain(X_norm,
                                       validation=X_norm_test[0:config.DBN_BATCH_SIZE],
                                       epochs=config.DBN_EPOCHS,
                                       alpha=config.DBN_ALPHA,
                                       momentum=config.DBN_M,
                                       batch_size=config.DBN_BATCH_SIZE,
                                       gibbs_k=config.DBN_GIBBS_K,
                                       alpha_update_rule=config.DBN_ALPHA_UPDATE_RULE,
                                       momentum_update_rule=config.DBN_MOMENTUM_UPDATE_RULE,
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

        print('Save configuration of rbm layers after training')
        deep_net.layers[0].save_configuration(config.DBN_OUTFILES[0])
        deep_net.layers[1].save_configuration(config.DBN_OUTFILES[1])
        deep_net.last_rbm.save_configuration(config.DBN_OUTFILES[2])

        print('Save performance metrics of the rbm during wake sleep...')
        deep_net.save_performance_metrics(config.DBN_WS_PERFORMANCE_OUTFILE)

        print('Testing the accuracy of the dbn...')
        dbn_preds = deep_net.predict_ws(X_norm_test, config.DBN_TEST_TOP_GIBBS_K)

        accuracy_dbn = sum(dbn_preds == y_test) / float(config.TEST_SET_SIZE)
        print('Accuracy of the Deep Belief Network: %s' % accuracy_dbn)


if __name__ == '__main__':
    main()
