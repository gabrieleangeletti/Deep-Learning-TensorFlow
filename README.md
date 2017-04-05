# Deep Learning algorithms with TensorFlow

This repository is a collection of various Deep Learning algorithms implemented using the
[TensorFlow](http://www.tensorflow.org) library. This package is intended as a command line utility you can use to quickly train and
evaluate popular Deep Learning models and maybe use them as benchmark/baseline in comparison to your custom models/datasets.
If you want to use the package from ipython or maybe integrate it in your code, I published a pip package named `yadlt`: Yet Another Deep Learning Tool.

### Requirements:

* tensorflow >= 1.0

### List of available models:

* Convolutional Network
* Restricted Boltzmann Machine
* Deep Belief Network
* Deep Autoencoder as stack of RBMs
* Denoising Autoencoder
* Stacked Denoising Autoencoder
* Deep Autoencoder as stack of Denoising Autoencoders
* MultiLayer Perceptron
* Logistic Regression

### Installation

#### Through pip:

    pip install yadlt

You can learn the basic usage of the models by looking at the ``command_line/`` directory. Or you can take a look at the [documentation](http://deep-learning-tensorflow.readthedocs.io/en/latest/).

**Note**: the documentation is still a work in progress for the pip package, but the package usage is very simple. The classes have a sklearn-like interface, so basically you just have to create the object
(e.g. `sdae = StackedDenoisingAutoencoder()`) and call the fit/predict methods, and the pretrain() method if the model supports it
(e.g. `sdae.pretrain(X_train, y_train)`, `sdae.fit(X_train, y_train)` and `predictions = sdae.predict(X_test)`)

#### Through github:

* cd in a directory where you want to store the project, e.g. ``/home/me``
* clone the repository: ``git clone https://github.com/blackecho/Deep-Learning-TensorFlow.git``
* ``cd Deep-Learning-TensorFlow``
* now you can configure the software and run the models (see the [documentation](http://deep-learning-tensorflow.readthedocs.io/en/latest/))!

### Documentation:

You can find the documentation for this project at this [link](http://deep-learning-tensorflow.readthedocs.io/en/latest/).

### Models TODO list

* Recurrent Networks (LSTMs)
* Variational Autoencoders
* Deep Q Reinforcement Learning
