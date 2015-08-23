import numpy as np
import sys

from sklearn.linear_model import LogisticRegression

from mnist import MNIST
import config
import util
from classification import *
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
    # normalize dataset to be between 0 and 1
    X_norm = util.normalize_dataset(X)
    X_norm_test = util.normalize_dataset(X_test)
    # Type of learning machines to train
    if len(sys.argv) > 1:
        type = sys.argv[1].split('=')[1]
    else:
        # default type is standard
        type = 'standard'

    # #####################################
    # Standard Restricted Boltzmann Machine
    # #####################################
    if type == 'standard':
        # create rbm
        r = rbm.RBM(config.NV, config.NH)
        print('Begin Training...')
        r.train(X_norm,
                validation=X_norm_test[0:config.BATCH_SIZE],
                max_epochs=config.MAX_EPOCHS,
                alpha=config.ALPHA,
                m=config.M,
                batch_size=config.BATCH_SIZE,
                gibbs_k=config.GIBBS_K,
                verbose=config.VERBOSE,
                display=display)
        # save the rbm to a file
        print('Saving the RBM to outfile...')
        r.save_configuration(config.OUTFILE)

    # ########################################
    # Multinomial Restricted Boltzmann Machine
    # ########################################
    if type == 'multinomial':
        # discretization of data
        X_mu = util.discretize_dataset(X, config.MULTI_KV)
        X_mu_test = util.discretize_dataset(X_test, config.MULTI_KV)
        # create multinomial rbm
        mr = multinomial_rbm.MultinomialRBM(config.MULTI_NV, config.MULTI_NH, config.MULTI_KV, config.MULTI_KN)
        mr.train(X_mu,
                validation=X_mu_test[0:config.M_BATCH_SIZE],
                max_epochs=config.M_MAX_EPOCHS,
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
    # Bernoulli Gaussian Restricted Boltzmann Machine
    # ###############################################
    if type == 'gaussian':
        # create gaussian rbm
        gr = gaussian_rbm.GaussianRBM(config.GAUSS_NV, config.GAUSS_NH)
        print('Begin Training...')
        gr.train(X,
                 validation=X_test[0:config.G_BATCH_SIZE],
                 max_epochs=config.G_MAX_EPOCHS,
                 alpha=config.G_ALPHA,
                 m=config.G_M,
                 batch_size=config.G_BATCH_SIZE,
                 gibbs_k=config.G_GIBBS_K,
                 verbose=config.G_VERBOSE,
                 display=display)
        # save the rbm to a file
        print('Saving the Gaussian RBM to outfile...')
        gr.save_configuration(config.G_OUTFILE)

    # #################################################
    # Classification rbm vs Logistic Regression
    # #################################################
    elif type == 'rbm-vs-logistic':
        cst = classification_rbm.ClsRBM(
            config.NV, config.NH)
        # unsupervised learning of features
        print('Starting unsupervised learning of the features...')
        cst.learn_unsupervised_features(X_norm,
                                        validation=X_norm_test[0:config.BATCH_SIZE],
                                        max_epochs=config.MAX_EPOCHS,
                                        batch_size=config.BATCH_SIZE,
                                        alpha=config.ALPHA,
                                        m=config.M,
                                        gibbs_k=config.GIBBS_K,
                                        verbose=config.VERBOSE,
                                        display=display)

        # save the standard rbm to a file
        print('Saving the RBM to outfile...')
        cst.rbm.save_configuration(config.OUTFILE)
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
        print('Accuracy of the RBM classifier: %s' %
              (accuracy_st))
        print('Accuracy of the Logistic classifier: %s' %
              (accuracy_lr))

    # #####################################################
    # Classification Multinomial rbm vs Logistic Regression
    # #####################################################
    elif type == 'mrbm-vs-logistic':
        # discretization of data
        X_mu = util.discretize_dataset(X, config.MULTI_KV)
        X_mu_test = util.discretize_dataset(X_test, config.MULTI_KV)
        # create multinomial rbm
        csm = classification_multinomial_rbm.ClsMultiRBM(config.MULTI_NV, config.MULTI_NH, config.MULTI_KV, config.MULTI_KN)
        # unsupervised learning of features
        print('Starting unsupervised learning of the features...')
        csm.learn_unsupervised_features(X_mu,
                                        validation=X_mu_test[0:config.M_BATCH_SIZE],
                                        max_epochs=config.M_MAX_EPOCHS,
                                        alpha=config.M_ALPHA,
                                        m=config.M_M,
                                        batch_size=config.M_BATCH_SIZE,
                                        gibbs_k=config.M_GIBBS_K,
                                        verbose=config.M_VERBOSE,
                                        display=display)
        print('Saving the Multi RBM to outfile...')
        csm.mrbm.save_configuration(config.M_OUTFILE)
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
        print('Accuracy of the RBM classifier: %s' %
              (accuracy_m))
        print('Accuracy of the Logistic classifier: %s' %
              (accuracy_lr))


    # ##################################################
    # Classification Gaussian rbm vs Logistic Regression
    # ##################################################
    elif type == 'grbm-vs-logistic':
        csg = classification_gaussian_rbm.ClsGaussianRBM(
            config.GAUSS_NV, config.GAUSS_NH)
        # unsupervised learning of features
        print('Starting unsupervised learning of the features...')
        csg.learn_unsupervised_features(X,
                                        validation=X_test[0:config.G_BATCH_SIZE],
                                        max_epochs=config.G_MAX_EPOCHS,
                                        batch_size=config.G_BATCH_SIZE,
                                        alpha=config.G_ALPHA,
                                        m=config.G_M,
                                        gibbs_k=config.G_GIBBS_K,
                                        verbose=config.G_VERBOSE,
                                        display=display)

        # save the standard rbm to a file
        print('Saving the GRBM to outfile...')
        csg.grbm.save_configuration(config.G_OUTFILE)
        # fit the Logistic Regression layer
        print('Fitting the Logistic Regression layer...')
        csg.fit_logistic_cls(X, y)
        # sample the test set
        print('Testing the accuracy of the classifier...')
        # test the predictions of the LR layer
        preds_st = csg.predict_logistic_cls(X_test)

        accuracy_st = sum(preds_st == y_test) / float(config.TEST_SET_SIZE)
        # Now train a normal logistic regression classifier and test it
        print('Training standard Logistic Regression Classifier...')
        lr_cls = LogisticRegression()
        lr_cls.fit(X, y)
        lr_cls_preds = lr_cls.predict(X_test)
        accuracy_lr = sum(lr_cls_preds == y_test) / float(config.TEST_SET_SIZE)
        print('Accuracy of the RBM classifier: %s' %
              (accuracy_st))
        print('Accuracy of the Logistic classifier: %s' %
              (accuracy_lr))
