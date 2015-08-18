Deep Belief Network and Restricted Boltzmann Machine Python Implementation, with MNIST dataset ready for training.

The MNIST dataset and MNIST dataset loader script are taken from https://github.com/sorki/python-mnist

Requirements:
- numpy
- scikitlearn

Usage:

- config.py:
Configuration file, used to set the training parameters.

- main.py:
Main file used for training.

- Set all the parameters in config.py.
- run the command:
    python main.py type=...
    where type can be equal to the following (without quotes):

    "standard" : unsupervised training of a standard rbm and saving to outfile for later retrieving

    "gaussian" : unsupervised training of a gaussian rbm and saving to outfile for later retrieving

    "rbm-vs-logistic" : learn two classifiers and print their accuracies in the test set:
                         - standard rbm with a layer of LogisticRegression (this will be also saved to outfile)
                         - standard LogisticRegression

    "grbm-vs-logistic" : learn two classifiers and print their accuracies in the test set:
                     - gaussian rbm with a layer of LogisticRegression (this will be also saved to outfile)
                     - standard LogisticRegression
