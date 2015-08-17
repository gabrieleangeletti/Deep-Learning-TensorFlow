Spline Restricted Boltzmann Machine Implementation.

files in the package:

classification_rbm.py:
A Restricted Boltzmann Machine with a Logistic Regression layer build on top of the hidden units. Can use both Standard RBM and
Spline RBM, setting the parameter 'type' in the constructor to 'standard' or 'spline'.

config.py:
Configuration file, used to set all the training parameters.

main.py:
Main file used for training.

rbm.py:
Standard Restricted Boltzmann Machine implementation.

spline_rbm.py:
Spline Restricted Boltzmann Machine implementation.

util.py:
File with utility functions:
normalize_dataset(X) - normalize the value in X to be in {0, 1}
generate_batches(data, batch_size) - divide the data in batches of the given size

Requirements:
- numpy
- scikitlearn, for the LogisticRegression classifier

Usage:
- Set all the parameters in config.py.
- run the command:
    python main.py type=...
    where type can be equal to the following (without quotes):
    "standard" : unsupervised training of a standard rbm and saving to outfile for later retrieving
    "spline" : unsupervised training of a spline rbm and saving to outfile for later retrieving
    "srbm-vs-logistic" : learn two classifiers and print their accuracies in the test set:
                         - spline rbm with a layer of LogisticRegression (this will be also saved to outfile)
                         - standard LogisticRegression
    "rbm-vs-srbm-vs-logistic" : learn three classifiers and print their accuracies in the test set:
                         - standard rbm with a layer of LogisticRegression (this will be also saved to outfile)   
                         - spline rbm with a layer of LogisticRegression (this will be also saved to outfile)
                         - standard LogisticRegression
